//! Human-adaptive encoding: turn face detections into [`LambdaRegion`]s that lower the RD
//! lambda around faces, and more so around the eyes, nose, and mouth, so the encoder spends
//! more bits where human viewers notice compression errors most.
//!
//! The region *geometry* here has no external dependencies and is unit-tested directly.
//! Only [`detect_faces`] needs the opt-in `face-detect` build feature (OpenCV + ONNX
//! Runtime via [`rusty_scrfd`]), so the ordinary `cargo test`/`cargo build` never pulls
//! those native libraries in.

use mars_codec::encode::LambdaRegion;

/// One detected face: a pixel-space bounding box plus SCRFD's five landmarks in SCRFD's
/// fixed order -- left eye, right eye, nose, left mouth corner, right mouth corner.
///
/// Coordinates are in plane pixels (the top-left origin this codec uses everywhere), not
/// SCRFD's relative 0..1 output; [`detect_faces`] converts before returning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceDetection {
    /// `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    pub keypoints: [[f32; 2]; 5],
}

impl FaceDetection {
    /// Distance between the two eye landmarks, in pixels. Used as the scale unit for the
    /// feature boxes so they track face size rather than a fixed pixel radius.
    fn eye_distance(&self) -> f32 {
        let dx = self.keypoints[0][0] - self.keypoints[1][0];
        let dy = self.keypoints[0][1] - self.keypoints[1][1];
        (dx * dx + dy * dy).sqrt()
    }
}

/// The axis-aligned bounds of a set of points, as `(min_x, min_y, max_x, max_y)`.
fn points_bounds(points: &[[f32; 2]]) -> (f32, f32, f32, f32) {
    let mut bounds = (
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );
    for p in points {
        bounds.0 = bounds.0.min(p[0]);
        bounds.1 = bounds.1.min(p[1]);
        bounds.2 = bounds.2.max(p[0]);
        bounds.3 = bounds.3.max(p[1]);
    }
    bounds
}

/// A region covering the `(min_x, min_y, max_x, max_y)` box expanded by `pad` pixels on
/// every side and clamped to the image, or `None` if nothing of it lands on-screen.
fn padded_region(
    (min_x, min_y, max_x, max_y): (f32, f32, f32, f32),
    pad: f32,
    width: u32,
    height: u32,
    scale: f64,
) -> Option<LambdaRegion> {
    let min_x = (min_x - pad).floor().max(0.0);
    let min_y = (min_y - pad).floor().max(0.0);
    let max_x = (max_x + pad).ceil().min(width as f32);
    let max_y = (max_y + pad).ceil().min(height as f32);
    if max_x <= min_x || max_y <= min_y {
        return None;
    }
    Some(LambdaRegion {
        row: min_y as u32,
        col: min_x as u32,
        height: (max_y - min_y) as u32,
        width: (max_x - min_x) as u32,
        scale,
    })
}

/// Build the region set for a list of faces: one box per face at `face_scale`, plus an
/// eyes, a nose, and a mouth box per face at `feature_scale`.
///
/// Feature boxes are derived from the landmarks relative to the interocular distance, so
/// the mapping is scale-invariant and needs no assumptions about image resolution or the
/// detector's input size. Degenerate landmarks fall back to a fraction of the face box.
pub fn regions_from_faces(
    faces: &[FaceDetection],
    width: u32,
    height: u32,
    face_scale: f64,
    feature_scale: f64,
) -> Vec<LambdaRegion> {
    let mut regions = Vec::with_capacity(faces.len() * 4);
    for face in faces {
        if let Some(region) = padded_region(
            (face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3]),
            0.0,
            width,
            height,
            face_scale,
        ) {
            regions.push(region);
        }

        let mut eye_distance = face.eye_distance();
        if !eye_distance.is_finite() || eye_distance < 1.0 {
            let box_width = (face.bbox[2] - face.bbox[0]).abs();
            let box_height = (face.bbox[3] - face.bbox[1]).abs();
            eye_distance = 0.25 * box_width.max(box_height).max(1.0);
        }

        // Eyes: both keypoints, padded by half the interocular distance so the brows and
        // lashes -- where errors are most visible -- are covered too.
        if let Some(region) = padded_region(
            points_bounds(&face.keypoints[0..2]),
            0.5 * eye_distance,
            width,
            height,
            feature_scale,
        ) {
            regions.push(region);
        }
        // Nose: a zero-area box around the single landmark, padded outward.
        if let Some(region) = padded_region(
            points_bounds(&face.keypoints[2..3]),
            0.4 * eye_distance,
            width,
            height,
            feature_scale,
        ) {
            regions.push(region);
        }
        // Mouth: both corners, padded by well under the interocular distance so it does
        // not swallow the nose above it.
        if let Some(region) = padded_region(
            points_bounds(&face.keypoints[3..5]),
            0.4 * eye_distance,
            width,
            height,
            feature_scale,
        ) {
            regions.push(region);
        }
    }
    regions
}

/// Detect faces in `image_path` with the SCRFD ONNX model at `model`, returning pixel-space
/// [`FaceDetection`]s (at most `max_faces`).
///
/// Requires the `face-detect` build feature, OpenCV at build/link time, and a compatible
/// SCRFD ONNX model. An image with no detectable faces returns an empty vector, not an
/// error -- `rusty_scrfd` reports that case as an error, which is translated here.
#[cfg(feature = "face-detect")]
pub fn detect_faces(
    model: &std::path::Path,
    image_path: &std::path::Path,
    conf_threshold: f32,
    max_faces: usize,
) -> anyhow::Result<Vec<FaceDetection>> {
    use std::collections::HashMap;

    use anyhow::Context;
    use opencv::prelude::MatTraitConst;

    let session = ort::session::Session::builder()
        .map_err(|err| anyhow::anyhow!("creating an ONNX Runtime session builder: {err}"))?
        .commit_from_file(model)
        .map_err(|err| anyhow::anyhow!("loading SCRFD model {}: {err}", model.display()))?;
    let mut detector = rusty_scrfd::SCRFDBuilder::new(session)
        .set_input_size((640, 640))
        .set_conf_thres(conf_threshold)
        .set_iou_thres(0.4)
        // Relative 0..1 output; converted to pixels below so `FaceDetection` is always in
        // the codec's own coordinate convention.
        .set_relative_output(true)
        .build()
        .map_err(|err| anyhow::anyhow!("initialising the SCRFD detector: {err}"))?;

    let path = image_path
        .to_str()
        .with_context(|| format!("{} is not valid UTF-8", image_path.display()))?;
    let mat = opencv::imgcodecs::imread(path, opencv::imgcodecs::IMREAD_COLOR)
        .map_err(|err| anyhow::anyhow!("OpenCV could not read {}: {err}", image_path.display()))?;
    let (image_width, image_height) = (mat.cols() as f32, mat.rows() as f32);

    let mut center_cache = HashMap::new();
    let (boxes, keypoints) = match detector.detect(&mat, max_faces, "max", &mut center_cache) {
        Ok(found) => found,
        // No faces is a normal outcome for this feature, not a failure.
        Err(err) if err.to_string().contains("No faces detected") => return Ok(Vec::new()),
        Err(err) => return Err(anyhow::anyhow!("SCRFD detection failed: {err}")),
    };

    let mut detections = Vec::with_capacity(boxes.nrows());
    for (index, row) in boxes.rows().into_iter().enumerate() {
        let bbox = [
            row[0] * image_width,
            row[1] * image_height,
            row[2] * image_width,
            row[3] * image_height,
        ];
        // Keypoints are optional in SCRFD and land in the same NMS order as the boxes, so
        // row `index` of each refers to the same face.
        let keypoints = match &keypoints {
            Some(keypoints) if index < keypoints.shape()[0] => {
                let face = keypoints.index_axis(ndarray::Axis(0), index);
                let mut points = [[0.0f32; 2]; 5];
                for (slot, point) in points.iter_mut().zip(face.rows()) {
                    *slot = [point[0] * image_width, point[1] * image_height];
                }
                points
            }
            _ => [[0.0f32; 2]; 5],
        };
        detections.push(FaceDetection { bbox, keypoints });
    }
    Ok(detections)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mars_codec::encode::lambda_scale_for_block;

    /// A 100x100 face with eyes 40px apart, centred in a 200x200 image.
    fn centred_face() -> FaceDetection {
        FaceDetection {
            bbox: [50.0, 40.0, 150.0, 160.0],
            keypoints: [
                [80.0, 80.0],   // left eye
                [120.0, 80.0],  // right eye (40px apart)
                [100.0, 100.0], // nose
                [85.0, 130.0],  // left mouth corner
                [115.0, 130.0], // right mouth corner
            ],
        }
    }

    #[test]
    fn a_face_yields_face_plus_three_feature_regions() {
        let regions = regions_from_faces(&[centred_face()], 200, 200, 0.5, 0.25);
        assert_eq!(regions.len(), 4, "one face box + eyes + nose + mouth");
        assert_eq!(regions[0].scale, 0.5, "the face box uses the face scale");
        for feature in &regions[1..] {
            assert_eq!(
                feature.scale, 0.25,
                "eye/nose/mouth boxes use the feature scale"
            );
        }
    }

    #[test]
    fn the_face_box_covers_the_detected_bounding_box() {
        let regions = regions_from_faces(&[centred_face()], 200, 200, 0.5, 0.25);
        let face = regions[0];
        assert!(face.col <= 50 && face.row <= 40);
        assert!(face.col + face.width >= 150);
        assert!(face.row + face.height >= 160);
    }

    #[test]
    fn feature_boxes_are_inside_the_face_box() {
        let regions = regions_from_faces(&[centred_face()], 200, 200, 0.5, 0.25);
        let face = regions[0];
        for feature in &regions[1..] {
            assert!(
                feature.col >= face.col
                    && feature.row >= face.row
                    && feature.col + feature.width <= face.col + face.width
                    && feature.row + feature.height <= face.row + face.height,
                "feature region {feature:?} must sit inside the face region {face:?}"
            );
        }
    }

    #[test]
    fn a_block_inside_the_eyes_sees_the_feature_scale_not_the_face_scale() {
        let regions = regions_from_faces(&[centred_face()], 200, 200, 0.5, 0.25);
        // A block centred between the eyes is fully inside the eyes region and the face
        // region; the smaller (feature) scale must win.
        let scale = lambda_scale_for_block(&regions, 72, 88, 16);
        assert!(
            (scale - 0.25).abs() < 1e-9,
            "expected the feature scale, got {scale}"
        );
        // A block on the cheek is inside the face only.
        let cheek = lambda_scale_for_block(&regions, 140, 60, 8);
        assert!(
            (cheek - 0.5).abs() < 1e-9,
            "expected the face scale, got {cheek}"
        );
        // A block on the background is untouched.
        assert_eq!(lambda_scale_for_block(&regions, 0, 0, 8), 1.0);
    }

    #[test]
    fn regions_are_clamped_to_the_image() {
        let edge_face = FaceDetection {
            bbox: [-20.0, -30.0, 30.0, 10.0],
            keypoints: [
                [-10.0, -15.0],
                [10.0, -15.0],
                [0.0, -5.0],
                [-8.0, 5.0],
                [8.0, 5.0],
            ],
        };
        let regions = regions_from_faces(&[edge_face], 64, 64, 0.5, 0.25);
        assert!(!regions.is_empty());
        for region in &regions {
            assert!(region.col + region.width <= 64, "{region:?} exceeds width");
            assert!(
                region.row + region.height <= 64,
                "{region:?} exceeds height"
            );
        }
    }

    #[test]
    fn a_face_fully_off_screen_contributes_no_region() {
        let off_screen = FaceDetection {
            bbox: [500.0, 500.0, 600.0, 600.0],
            keypoints: [
                [520.0, 520.0],
                [560.0, 520.0],
                [540.0, 540.0],
                [520.0, 560.0],
                [560.0, 560.0],
            ],
        };
        assert!(regions_from_faces(&[off_screen], 64, 64, 0.5, 0.25).is_empty());
    }

    #[test]
    fn degenerate_landmarks_fall_back_to_a_fraction_of_the_face_box() {
        let mut face = centred_face();
        face.keypoints = [[0.0; 2]; 5];
        let regions = regions_from_faces(&[face], 200, 200, 0.5, 0.25);
        // Face box plus three feature boxes, all on-screen -- the fallback scale unit must
        // still produce usable feature regions rather than panicking or collapsing.
        assert_eq!(regions.len(), 4);
        for region in &regions[1..] {
            assert!(region.width > 0 && region.height > 0);
        }
    }

    #[test]
    fn no_faces_yields_no_regions() {
        assert!(regions_from_faces(&[], 200, 200, 0.5, 0.25).is_empty());
    }
}
