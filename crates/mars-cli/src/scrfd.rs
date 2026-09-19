//! SCRFD face detection on ONNX Runtime, without OpenCV.
//!
//! `rusty_scrfd` could not be reused: it takes an `opencv::core::Mat` and depends on
//! `opencv` unconditionally, so building it needs a system OpenCV install. Everything
//! OpenCV did for this path -- letterbox resize, per-channel normalisation into an NCHW
//! tensor, and the `Mat` container -- is a few dozen lines over the codec's own
//! [`Image`], and the rest (anchor decoding and NMS) is arithmetic. The only dependency is
//! `ort`, which downloads a prebuilt ONNX Runtime at build time and needs no system
//! package.
//!
//! The preprocessing and decode helpers here are pure and unit-tested without a model
//! (they are always compiled); only [`detect_faces`] needs the `face-detect` build feature
//! and an SCRFD ONNX model.

use anyhow::{bail, Result};
use mars_core::image::{ColorSpace, Image};

use crate::human::FaceDetection;

/// Side length of SCRFD's square input canvas. The insightface SCRFD model-zoo exports
/// (for example `det_500m.onnx`, `det_10g.onnx`) all take a 640x640 RGB input.
pub const INPUT_SIZE: usize = 640;
/// Per-channel normalisation SCRFD was trained with: `(pixel - 127.5) / 128`.
const PIXEL_MEAN: f32 = 127.5;
const PIXEL_STD: f32 = 128.0;
/// SCRFD's three feature-pyramid strides.
const STRIDES: [usize; 3] = [8, 16, 32];
/// Anchors per feature-map position.
const ANCHORS_PER_POSITION: usize = 2;
/// SCRFD's five landmarks per face: left eye, right eye, nose, left mouth corner, right
/// mouth corner.
pub const KEYPOINTS_PER_FACE: usize = 5;
/// NMS IoU threshold, matching `rusty_scrfd`'s default.
pub const DEFAULT_IOU_THRESHOLD: f32 = 0.4;

/// Aspect-preserving letterbox geometry: the source is scaled by `scale` into the top-left
/// of the canvas, and the uncovered remainder is left as zero padding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub width: usize,
    pub height: usize,
    pub scale: f32,
}

impl Letterbox {
    /// Fit `source_width x source_height` inside [`INPUT_SIZE`] without changing the aspect
    /// ratio, matching insightface's `SCRFD.detect` (`det_scale = new_height / height`).
    pub fn fit(source_width: usize, source_height: usize) -> Self {
        let ratio = source_height as f32 / source_width as f32;
        let (width, height) = if ratio > 1.0 {
            ((INPUT_SIZE as f32 / ratio).round() as usize, INPUT_SIZE)
        } else {
            (INPUT_SIZE, (INPUT_SIZE as f32 * ratio).round() as usize)
        };
        let width = width.clamp(1, INPUT_SIZE);
        let height = height.clamp(1, INPUT_SIZE);
        Self {
            width,
            height,
            scale: height as f32 / source_height as f32,
        }
    }

    /// Map a coordinate in canvas space back to source pixels.
    pub fn to_source(&self, value: f32) -> f32 {
        value / self.scale
    }
}

/// One raw ONNX output tensor: its shape and its flattened `f32` data.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// One detected face in canvas coordinates, before mapping back to source pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedFace {
    pub score: f32,
    pub bbox: [f32; 4],
    pub keypoints: [[f32; 2]; KEYPOINTS_PER_FACE],
}

/// One FPN level's raw model outputs, flattened.
struct FpnLevel<'a> {
    stride: usize,
    rows: usize,
    cols: usize,
    scores: &'a [f32],
    box_deltas: &'a [f32],
    keypoint_deltas: &'a [f32],
}

/// One source pixel as RGB, replicating luma for a grayscale input (SCRFD is a 3-channel
/// model, so a gray image is fed as three identical channels).
fn source_pixel(image: &Image, x: usize, y: usize) -> [f32; 3] {
    let index = y * image.width() + x;
    let planes = image.planes();
    if image.color() == ColorSpace::Gray {
        let value = f32::from(planes[0].as_slice()[index]);
        [value, value, value]
    } else {
        [
            f32::from(planes[0].as_slice()[index]),
            f32::from(planes[1].as_slice()[index]),
            f32::from(planes[2].as_slice()[index]),
        ]
    }
}

/// Bilinear sample of the source at fractional pixel coordinates, with edge clamping.
/// Half-pixel centres match OpenCV's `INTER_LINEAR` `resize`, which is what the SCRFD
/// reference implementation uses.
fn sample_bilinear(image: &Image, x: f32, y: f32) -> [f32; 3] {
    let max_x = image.width().saturating_sub(1);
    let max_y = image.height().saturating_sub(1);
    let x0 = (x.floor().max(0.0) as usize).min(max_x);
    let y0 = (y.floor().max(0.0) as usize).min(max_y);
    let x1 = (x0 + 1).min(max_x);
    let y1 = (y0 + 1).min(max_y);
    let fx = (x - x0 as f32).clamp(0.0, 1.0);
    let fy = (y - y0 as f32).clamp(0.0, 1.0);

    let top_left = source_pixel(image, x0, y0);
    let top_right = source_pixel(image, x1, y0);
    let bottom_left = source_pixel(image, x0, y1);
    let bottom_right = source_pixel(image, x1, y1);
    let mut out = [0.0f32; 3];
    for channel in 0..3 {
        let top = top_left[channel] * (1.0 - fx) + top_right[channel] * fx;
        let bottom = bottom_left[channel] * (1.0 - fx) + bottom_right[channel] * fx;
        out[channel] = top * (1.0 - fy) + bottom * fy;
    }
    out
}

/// SCRFD's input tensor: a `1x3x640x640` NCHW `f32` buffer, RGB, normalised, with the
/// letterbox padding left at exactly `0.0`. Returns the buffer and the geometry needed to
/// map detections back to source pixels.
pub fn prepare_input(image: &Image) -> (Vec<f32>, Letterbox) {
    let letterbox = Letterbox::fit(image.width(), image.height());
    let mut tensor = vec![0.0f32; 3 * INPUT_SIZE * INPUT_SIZE];
    let plane = INPUT_SIZE * INPUT_SIZE;
    for y in 0..letterbox.height {
        for x in 0..letterbox.width {
            let sx = (x as f32 + 0.5) / letterbox.scale - 0.5;
            let sy = (y as f32 + 0.5) / letterbox.scale - 0.5;
            let rgb = sample_bilinear(image, sx, sy);
            let offset = y * INPUT_SIZE + x;
            for channel in 0..3 {
                tensor[channel * plane + offset] = (rgb[channel] - PIXEL_MEAN) / PIXEL_STD;
            }
        }
    }
    (tensor, letterbox)
}

/// Which of SCRFD's three per-stride tensors an output is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputKind {
    Scores,
    BoxDeltas,
    KeypointDeltas,
}

/// Resolve one output tensor to `(stride slot, kind)` from its element count and last
/// dimension. The three strides give distinguishable `(length, last_dim)` pairs, so this
/// does not depend on the order the model happens to emit its nine tensors in.
fn classify(output: &OutputTensor) -> Option<(usize, OutputKind)> {
    let last = *output.shape.last()?;
    for (slot, &stride) in STRIDES.iter().enumerate() {
        let side = INPUT_SIZE / stride;
        let positions = side * side * ANCHORS_PER_POSITION;
        let kind = if last == 1 && output.data.len() == positions {
            OutputKind::Scores
        } else if last == 4 && output.data.len() == positions * 4 {
            OutputKind::BoxDeltas
        } else if last == 10 && output.data.len() == positions * 10 {
            OutputKind::KeypointDeltas
        } else {
            continue;
        };
        return Some((slot, kind));
    }
    None
}

/// SCRFD's anchor centres for one level, in the model's output order: row-major over the
/// feature map, with `ANCHORS_PER_POSITION` copies of each position, each scaled by the
/// stride.
fn anchor_centers(rows: usize, cols: usize, stride: usize) -> Vec<[f32; 2]> {
    let mut anchors = Vec::with_capacity(rows * cols * ANCHORS_PER_POSITION);
    for row in 0..rows {
        for col in 0..cols {
            let center = [(col * stride) as f32, (row * stride) as f32];
            anchors.extend(std::iter::repeat_n(center, ANCHORS_PER_POSITION));
        }
    }
    anchors
}

/// Decode one FPN level: `distance2bbox`/`distance2kps` around each above-threshold anchor.
fn decode_level(level: &FpnLevel, confidence: f32) -> Vec<DetectedFace> {
    let anchors = anchor_centers(level.rows, level.cols, level.stride);
    let stride = level.stride as f32;
    let mut faces = Vec::new();
    for (index, &anchor) in anchors.iter().enumerate() {
        let score = level.scores[index];
        // Non-finite scores are not detections (`NaN <= x` is false, so test it directly).
        if score.is_nan() || score <= confidence {
            continue;
        }
        let box_delta = &level.box_deltas[index * 4..index * 4 + 4];
        let keypoint_base = index * 2 * KEYPOINTS_PER_FACE;
        let mut keypoints = [[0.0f32; 2]; KEYPOINTS_PER_FACE];
        for (slot, point) in keypoints.iter_mut().enumerate() {
            point[0] = anchor[0] + level.keypoint_deltas[keypoint_base + slot * 2] * stride;
            point[1] = anchor[1] + level.keypoint_deltas[keypoint_base + slot * 2 + 1] * stride;
        }
        faces.push(DetectedFace {
            score,
            bbox: [
                anchor[0] - box_delta[0] * stride,
                anchor[1] - box_delta[1] * stride,
                anchor[0] + box_delta[2] * stride,
                anchor[1] + box_delta[3] * stride,
            ],
            keypoints,
        });
    }
    faces
}

/// Intersection over union of two `[x1, y1, x2, y2]` boxes.
fn intersection_over_union(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Greedy NMS: sort by descending score and drop any later box overlapping a kept one by
/// more than `iou_threshold`.
fn non_maximum_suppression(mut faces: Vec<DetectedFace>, iou_threshold: f32) -> Vec<DetectedFace> {
    faces.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut suppressed = vec![false; faces.len()];
    let mut kept = Vec::with_capacity(faces.len());
    for index in 0..faces.len() {
        if suppressed[index] {
            continue;
        }
        kept.push(faces[index]);
        for other in (index + 1)..faces.len() {
            if !suppressed[other]
                && intersection_over_union(&faces[index].bbox, &faces[other].bbox) > iou_threshold
            {
                suppressed[other] = true;
            }
        }
    }
    kept
}

/// Decode SCRFD's nine raw output tensors into canvas-space faces, NMS-suppressed.
///
/// Tensors are classified by shape, not position (see [`classify`]), so a model emitting
/// them in a different order still decodes; a model missing one, duplicating one, or with
/// an unexpected count is reported as an error rather than silently mis-decoded.
pub fn decode_outputs(
    outputs: &[OutputTensor],
    confidence: f32,
    iou_threshold: f32,
) -> Result<Vec<DetectedFace>> {
    if outputs.len() != STRIDES.len() * 3 {
        bail!(
            "SCRFD model produced {} output tensors, expected {}",
            outputs.len(),
            STRIDES.len() * 3
        );
    }
    let mut scores: [Option<&[f32]>; 3] = [None; 3];
    let mut box_deltas: [Option<&[f32]>; 3] = [None; 3];
    let mut keypoint_deltas: [Option<&[f32]>; 3] = [None; 3];

    for (index, output) in outputs.iter().enumerate() {
        let Some((slot, kind)) = classify(output) else {
            bail!(
                "SCRFD output {index} ({} elements, last dimension {}) matches no \
                 stride/kind combination",
                output.data.len(),
                output.shape.last().copied().unwrap_or(0)
            );
        };
        let target = match kind {
            OutputKind::Scores => &mut scores,
            OutputKind::BoxDeltas => &mut box_deltas,
            OutputKind::KeypointDeltas => &mut keypoint_deltas,
        };
        if target[slot].replace(output.data.as_slice()).is_some() {
            bail!(
                "SCRFD output {index} duplicates the stride-{} {kind:?} tensor",
                STRIDES[slot]
            );
        }
    }

    let mut faces = Vec::new();
    for slot in 0..STRIDES.len() {
        let (Some(scores), Some(box_deltas), Some(keypoint_deltas)) =
            (scores[slot], box_deltas[slot], keypoint_deltas[slot])
        else {
            bail!(
                "SCRFD model is missing one or more stride-{} output tensors",
                STRIDES[slot]
            );
        };
        let side = INPUT_SIZE / STRIDES[slot];
        faces.extend(decode_level(
            &FpnLevel {
                stride: STRIDES[slot],
                rows: side,
                cols: side,
                scores,
                box_deltas,
                keypoint_deltas,
            },
            confidence,
        ));
    }
    Ok(non_maximum_suppression(faces, iou_threshold))
}

/// Map canvas-space faces back to source pixels, keeping at most `max_faces` (the list is
/// already sorted by descending score, so this keeps the most confident ones).
pub fn faces_to_source(
    faces: &[DetectedFace],
    letterbox: &Letterbox,
    max_faces: usize,
) -> Vec<FaceDetection> {
    faces
        .iter()
        .take(max_faces)
        .map(|face| FaceDetection {
            bbox: face.bbox.map(|value| letterbox.to_source(value)),
            keypoints: face
                .keypoints
                .map(|[x, y]| [letterbox.to_source(x), letterbox.to_source(y)]),
        })
        .collect()
}

/// Detect faces in an already-decoded image with the SCRFD ONNX model at `model`,
/// returning bounding boxes and landmarks in source pixel coordinates.
///
/// Requires the `face-detect` build feature. An image with no detectable faces returns an
/// empty vector.
#[cfg(feature = "face-detect")]
pub fn detect_faces(
    model: &std::path::Path,
    image: &Image,
    confidence: f32,
    max_faces: usize,
) -> Result<Vec<FaceDetection>> {
    let (tensor, letterbox) = prepare_input(image);
    let outputs = run_model(model, tensor)?;
    let faces = decode_outputs(&outputs, confidence, DEFAULT_IOU_THRESHOLD)?;
    Ok(faces_to_source(&faces, &letterbox, max_faces))
}

/// Load the model, run one inference, and return its raw output tensors.
#[cfg(feature = "face-detect")]
fn run_model(model: &std::path::Path, tensor: Vec<f32>) -> Result<Vec<OutputTensor>> {
    use ort::value::Tensor;

    let mut session = ort::session::Session::builder()
        .map_err(|err| anyhow::anyhow!("creating an ONNX Runtime session: {err}"))?
        .commit_from_file(model)
        .map_err(|err| anyhow::anyhow!("loading SCRFD model {}: {err}", model.display()))?;
    let input = Tensor::from_array((
        [1usize, 3, INPUT_SIZE, INPUT_SIZE],
        tensor.into_boxed_slice(),
    ))
    .map_err(|err| anyhow::anyhow!("building the SCRFD input tensor: {err}"))?;
    let outputs = session
        .run(ort::inputs![input])
        .map_err(|err| anyhow::anyhow!("running the SCRFD model: {err}"))?;

    let mut raw = Vec::with_capacity(outputs.len());
    for (_, value) in outputs.iter() {
        let array: ndarray::ArrayViewD<f32> = value
            .try_extract_array()
            .map_err(|err| anyhow::anyhow!("extracting an SCRFD output tensor: {err}"))?;
        raw.push(OutputTensor {
            shape: array.shape().to_vec(),
            data: array.iter().copied().collect(),
        });
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mars_core::image::Plane;

    fn gray(width: usize, height: usize, value: u8) -> Image {
        Image::gray(Plane::filled(width, height, value))
    }

    /// Nine zero tensors with SCRFD's real shapes, with one high-scoring anchor at the
    /// (0, 0) position of the stride-8 level so a decode produces exactly one face.
    fn synthetic_outputs() -> Vec<OutputTensor> {
        let mut outputs = Vec::new();
        for stride in STRIDES {
            let side = INPUT_SIZE / stride;
            let positions = side * side * ANCHORS_PER_POSITION;
            let mut scores = vec![0.0f32; positions];
            let mut box_deltas = vec![0.0f32; positions * 4];
            let mut keypoint_deltas = vec![0.0f32; positions * 10];
            if stride == STRIDES[0] {
                scores[0] = 0.9;
                box_deltas[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
                for (index, value) in keypoint_deltas[0..10].iter_mut().enumerate() {
                    *value = (index + 1) as f32;
                }
            }
            outputs.push(OutputTensor {
                shape: vec![1, positions, 1],
                data: scores,
            });
            outputs.push(OutputTensor {
                shape: vec![1, positions, 4],
                data: box_deltas,
            });
            outputs.push(OutputTensor {
                shape: vec![1, positions, 10],
                data: keypoint_deltas,
            });
        }
        outputs
    }

    #[test]
    fn letterbox_preserves_aspect_ratio_within_the_canvas() {
        let wide = Letterbox::fit(1280, 720);
        assert_eq!((wide.width, wide.height), (INPUT_SIZE, 360));
        assert!((wide.scale - 0.5).abs() < 1e-6);

        let tall = Letterbox::fit(720, 1280);
        assert_eq!((tall.width, tall.height), (360, INPUT_SIZE));
        assert!((tall.scale - 0.5).abs() < 1e-6);

        let square = Letterbox::fit(100, 100);
        assert_eq!((square.width, square.height), (INPUT_SIZE, INPUT_SIZE));
        assert!((square.scale - 6.4).abs() < 1e-6);
    }

    #[test]
    fn to_source_inverts_the_letterbox_scale() {
        let letterbox = Letterbox::fit(1280, 720);
        assert!((letterbox.to_source(320.0) - 640.0).abs() < 1e-3);
    }

    #[test]
    fn prepare_input_normalises_rgb_channels_into_nchw() {
        let image = Image::rgb(
            Plane::filled(2, 2, 255),
            Plane::filled(2, 2, 127),
            Plane::filled(2, 2, 0),
        );
        let (tensor, letterbox) = prepare_input(&image);
        assert_eq!(tensor.len(), 3 * INPUT_SIZE * INPUT_SIZE);
        // A 2x2 source upscales to cover the whole canvas, so all pixels are source colour.
        assert_eq!(
            (letterbox.width, letterbox.height),
            (INPUT_SIZE, INPUT_SIZE)
        );
        let plane = INPUT_SIZE * INPUT_SIZE;
        assert!((tensor[0] - (255.0 - PIXEL_MEAN) / PIXEL_STD).abs() < 1e-5);
        assert!((tensor[plane] - (127.0 - PIXEL_MEAN) / PIXEL_STD).abs() < 1e-5);
        assert!((tensor[2 * plane] - (0.0 - PIXEL_MEAN) / PIXEL_STD).abs() < 1e-5);
        // ...including the far corner, which the letterbox scale still covers.
        assert!((tensor[plane - 1] - (255.0 - PIXEL_MEAN) / PIXEL_STD).abs() < 1e-5);
    }

    #[test]
    fn prepare_input_leaves_uncovered_canvas_at_exactly_zero() {
        // A 640x80 source fills only the top eighth of each channel's square canvas. The
        // tail of every channel plane must be untouched padding -- exactly 0.0, not the
        // normalised value a black pixel would produce.
        let (tensor, letterbox) = prepare_input(&gray(640, 80, 200));
        assert_eq!(letterbox.height, 80);
        assert!(tensor[0] > 0.0, "the covered strip is populated");
        let plane = INPUT_SIZE * INPUT_SIZE;
        let covered = 80 * INPUT_SIZE;
        for (channel, plane_data) in tensor.chunks_exact(plane).enumerate() {
            for (index, value) in plane_data.iter().enumerate().skip(covered) {
                assert_eq!(
                    *value, 0.0,
                    "channel {channel} padding at index {index} must stay zero"
                );
            }
        }
    }

    #[test]
    fn anchor_centers_follow_scrfd_output_order() {
        assert_eq!(
            anchor_centers(2, 2, 8),
            vec![
                [0.0, 0.0],
                [0.0, 0.0],
                [8.0, 0.0],
                [8.0, 0.0],
                [0.0, 8.0],
                [0.0, 8.0],
                [8.0, 8.0],
                [8.0, 8.0],
            ]
        );
    }

    #[test]
    fn decode_level_turns_deltas_into_a_box_and_landmarks() {
        let scores = vec![0.9f32, 0.0];
        let mut box_deltas = vec![0.0f32; 8];
        box_deltas[0..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mut keypoint_deltas = vec![0.0f32; 20];
        for keypoint in 0..KEYPOINTS_PER_FACE {
            let value = (keypoint + 1) as f32;
            keypoint_deltas[keypoint * 2] = value;
            keypoint_deltas[keypoint * 2 + 1] = value;
        }
        let faces = decode_level(
            &FpnLevel {
                stride: 8,
                rows: 1,
                cols: 1,
                scores: &scores,
                box_deltas: &box_deltas,
                keypoint_deltas: &keypoint_deltas,
            },
            0.5,
        );
        assert_eq!(faces.len(), 1, "only the first anchor clears the threshold");
        assert_eq!(faces[0].bbox, [-8.0, -16.0, 24.0, 32.0]);
        assert_eq!(faces[0].keypoints[0], [8.0, 8.0]);
        assert_eq!(faces[0].keypoints[4], [40.0, 40.0]);
    }

    #[test]
    fn nms_keeps_the_highest_scoring_overlapping_box() {
        let face = |score: f32, bbox: [f32; 4]| DetectedFace {
            score,
            bbox,
            keypoints: [[0.0; 2]; KEYPOINTS_PER_FACE],
        };
        let kept = non_maximum_suppression(
            vec![
                face(0.8, [0.0, 0.0, 10.0, 10.0]),
                face(0.9, [1.0, 1.0, 11.0, 11.0]),
                face(0.7, [100.0, 100.0, 110.0, 110.0]),
            ],
            0.4,
        );
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.9, "kept in descending score order");
        assert_eq!(kept[1].bbox, [100.0, 100.0, 110.0, 110.0]);
    }

    #[test]
    fn decode_outputs_classifies_by_shape_not_output_order() {
        let outputs = synthetic_outputs();
        let decoded = decode_outputs(&outputs, 0.5, DEFAULT_IOU_THRESHOLD).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].bbox, [-8.0, -16.0, 24.0, 32.0]);

        // The same nine tensors in reverse order must decode identically.
        let mut reversed = outputs;
        reversed.reverse();
        assert_eq!(
            decode_outputs(&reversed, 0.5, DEFAULT_IOU_THRESHOLD).unwrap(),
            decoded
        );
    }

    #[test]
    fn decode_outputs_rejects_a_wrong_output_count() {
        let error = decode_outputs(&[], 0.5, DEFAULT_IOU_THRESHOLD).unwrap_err();
        assert!(error.to_string().contains("expected 9"), "{error}");
    }

    #[test]
    fn faces_map_back_to_source_pixels_and_respect_max_faces() {
        let letterbox = Letterbox {
            width: 320,
            height: 640,
            scale: 0.5,
        };
        let face = |score: f32, bbox: [f32; 4], keypoint: [f32; 2]| DetectedFace {
            score,
            bbox,
            keypoints: [keypoint; KEYPOINTS_PER_FACE],
        };
        let faces = vec![
            face(0.9, [10.0, 20.0, 30.0, 40.0], [2.0, 4.0]),
            face(0.8, [100.0, 100.0, 120.0, 140.0], [50.0, 60.0]),
        ];
        let mapped = faces_to_source(&faces, &letterbox, 1);
        assert_eq!(
            mapped.len(),
            1,
            "max_faces keeps only the best-scoring face"
        );
        assert_eq!(mapped[0].bbox, [20.0, 40.0, 60.0, 80.0]);
        assert_eq!(mapped[0].keypoints[0], [4.0, 8.0]);
    }

    /// End-to-end check against a real SCRFD ONNX model. Skipped unless `MARS_SCRFD_MODEL`
    /// names one, so the default test run needs no model or network:
    /// `MARS_SCRFD_MODEL=det_500m.onnx cargo test -p mars-cli --features face-detect`.
    #[cfg(feature = "face-detect")]
    #[test]
    fn detect_faces_runs_a_real_model_when_one_is_configured() {
        let Ok(model) = std::env::var("MARS_SCRFD_MODEL") else {
            return;
        };
        // A flat image has no faces; the point is that load -> preprocess -> infer ->
        // decode -> map-back runs end to end.
        let faces = detect_faces(std::path::Path::new(&model), &gray(64, 64, 128), 0.5, 8)
            .expect("a real SCRFD model must run");
        assert!(faces.is_empty());
    }
}
