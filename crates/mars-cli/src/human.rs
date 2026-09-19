//! Human-adaptive encoding: turn face detections into [`LambdaRegion`]s that lower the RD
//! lambda around faces, and more so around the eyes, nose, and mouth, so the encoder spends
//! more bits where human viewers notice compression errors most. [`plan_regions`] tags each
//! rectangle with the feature it came from, which the `--debug-regions` overlay uses to
//! tell face outlines from feature outlines.
//!
//! The region *geometry* here has no external dependencies and is unit-tested directly;
//! face detection itself lives in [`crate::scrfd`].

use mars_codec::encode::LambdaRegion;

/// One detected face: a pixel-space bounding box plus SCRFD's five landmarks in SCRFD's
/// fixed order -- left eye, right eye, nose, left mouth corner, right mouth corner.
///
/// Coordinates are in plane pixels (the top-left origin this codec uses everywhere), not
/// SCRFD's normalised model output; [`crate::scrfd`] converts before returning.
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
        // Face regions override lambda only; the subdivision floor is applied separately by
        // the caller (`outside_min_size_regions`).
        min_size: None,
    })
}

/// Which part of a face a region was derived from. The `--debug-regions` overlay colours
/// `Face` outlines differently from the eyes/nose/mouth outlines, and `encmars --color`
/// selects which kinds keep colour; the encoder otherwise treats every region identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// The whole detected face box.
    Face,
    /// The eyes box: both eye landmarks, padded.
    Eyes,
    /// The nose box: the single nose landmark, padded outward.
    Nose,
    /// The mouth box: both mouth corners, padded.
    Mouth,
}

/// A planned region: the encoder-facing rectangle plus the feature class it came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HumanRegion {
    pub kind: RegionKind,
    pub region: LambdaRegion,
}

/// Build the region plan for a list of faces: one box per face at `face_scale`, an eyes box
/// at `eye_scale`, and a nose and mouth box per face at `feature_scale` -- so the eyes can be
/// refined harder (or more cheaply) than the rest of the face without touching anything else.
///
/// Feature boxes are derived from the landmarks relative to the interocular distance, so
/// the mapping is scale-invariant and needs no assumptions about image resolution or the
/// detector's input size. Degenerate landmarks fall back to a fraction of the face box.
pub fn plan_regions(
    faces: &[FaceDetection],
    width: u32,
    height: u32,
    face_scale: f64,
    feature_scale: f64,
    eye_scale: f64,
) -> Vec<HumanRegion> {
    let mut regions = Vec::with_capacity(faces.len() * 4);
    for face in faces {
        if let Some(region) = padded_region(
            (face.bbox[0], face.bbox[1], face.bbox[2], face.bbox[3]),
            0.0,
            width,
            height,
            face_scale,
        ) {
            regions.push(HumanRegion {
                kind: RegionKind::Face,
                region,
            });
        }

        let mut eye_distance = face.eye_distance();
        if !eye_distance.is_finite() || eye_distance < 1.0 {
            let box_width = (face.bbox[2] - face.bbox[0]).abs();
            let box_height = (face.bbox[3] - face.bbox[1]).abs();
            eye_distance = 0.25 * box_width.max(box_height).max(1.0);
        }

        // Eyes: both keypoints, padded by half the interocular distance so the brows and
        // lashes -- where errors are most visible -- are covered too. This is the one box
        // with a scale of its own (`eye_scale`), so it can out-refine the nose and mouth.
        if let Some(region) = padded_region(
            points_bounds(&face.keypoints[0..2]),
            0.5 * eye_distance,
            width,
            height,
            eye_scale,
        ) {
            regions.push(HumanRegion {
                kind: RegionKind::Eyes,
                region,
            });
        }
        // Nose: a zero-area box around the single landmark, padded outward.
        if let Some(region) = padded_region(
            points_bounds(&face.keypoints[2..3]),
            0.4 * eye_distance,
            width,
            height,
            feature_scale,
        ) {
            regions.push(HumanRegion {
                kind: RegionKind::Nose,
                region,
            });
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
            regions.push(HumanRegion {
                kind: RegionKind::Mouth,
                region,
            });
        }
    }
    regions
}

/// The encoder-facing projection of [`plan_regions`]: just the rectangles, in the same
/// order.
pub fn regions_from_faces(
    faces: &[FaceDetection],
    width: u32,
    height: u32,
    face_scale: f64,
    feature_scale: f64,
    eye_scale: f64,
) -> Vec<LambdaRegion> {
    plan_regions(faces, width, height, face_scale, feature_scale, eye_scale)
        .into_iter()
        .map(|planned| planned.region)
        .collect()
}

/// The full-image region that carries a coarse subdivision floor: `scale` 1.0 so it affects
/// only the floor, never lambda.
fn full_image_region(width: u32, height: u32, min_size: u32) -> LambdaRegion {
    LambdaRegion {
        row: 0,
        col: 0,
        height,
        width,
        scale: 1.0,
        min_size: Some(min_size),
    }
}

/// `base` expanded by `pad` pixels on every side and clamped to the image, carrying
/// `min_size` as its subdivision floor.
fn expand_region(
    base: &LambdaRegion,
    pad: u32,
    width: u32,
    height: u32,
    min_size: u32,
) -> LambdaRegion {
    let row = base.row.saturating_sub(pad);
    let col = base.col.saturating_sub(pad);
    let right = (base.col + base.width).saturating_add(pad).min(width);
    let bottom = (base.row + base.height).saturating_add(pad).min(height);
    LambdaRegion {
        row,
        col,
        height: bottom.saturating_sub(row),
        width: right.saturating_sub(col),
        scale: 1.0,
        min_size: Some(min_size),
    }
}

/// The graded subdivision-floor regions for `--outside-min-size-ramp`.
///
/// The floor starts at `--min-size` (`S`) just outside a face box and doubles with distance,
/// each band being as wide as the floor it applies:
///
/// ```text
/// [0, S)         -> S
/// [S, 3S)        -> 2S
/// [3S, 7S)       -> 4S
/// [(2^k - 1)S, (2^(k+1) - 1)S) -> 2^k S
/// ```
///
/// so with `--min-size 8`: 8 for the first 8 px, 16 for the next 16 px out to 24, 32 for the
/// next 32 px out to 56, and so on, capped at `--max-size` (a full-image region at that floor
/// covers everything past the last band). `min_size_for_block` keeps the *smallest* cap among
/// the regions a block overlaps, so concentric rings express the staircase with no codec
/// change at all. Distances are the same rectangular (L-infinity) metric the region system
/// uses everywhere else, so a face near an image edge ramps over the same distance as one in
/// the centre.
///
/// A plan with no face box is a no-op -- nothing to be outside of.
pub fn outside_min_size_regions(
    plan: &[HumanRegion],
    width: u32,
    height: u32,
    min_size: u32,
    max_size: u32,
) -> Vec<LambdaRegion> {
    let faces: Vec<LambdaRegion> = plan
        .iter()
        .filter(|planned| planned.kind == RegionKind::Face)
        .map(|planned| planned.region)
        .collect();
    if faces.is_empty() {
        return Vec::new();
    }
    let mut regions = Vec::new();
    let mut cap = min_size;
    // The ring for `cap` covers every block within `pad` pixels of a face box; the next band
    // is twice as wide, so `pad` becomes `2 * pad + min_size`.
    let mut pad = min_size;
    while cap < max_size {
        for face in &faces {
            let region = expand_region(face, pad, width, height, cap);
            if region.width > 0 && region.height > 0 {
                regions.push(region);
            }
        }
        cap *= 2;
        pad = pad.saturating_mul(2).saturating_add(min_size);
    }
    regions.push(full_image_region(width, height, max_size));
    regions
}

#[cfg(test)]
mod tests {
    use super::*;
    use mars_codec::encode::{lambda_scale_for_block, min_size_for_block};

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

    /// The eyes can be refined harder than the nose and mouth: `eye_scale` applies to the
    /// eyes box alone, and the deeper region still nests inside the face.
    #[test]
    fn the_eye_scale_applies_to_the_eyes_box_alone() {
        let regions = regions_from_faces(&[centred_face()], 200, 200, 0.5, 0.25, 0.125);
        assert_eq!(regions[1].scale, 0.125, "the eyes box uses the eye scale");
        assert_eq!(regions[2].scale, 0.25, "the nose keeps the feature scale");
        assert_eq!(regions[3].scale, 0.25, "the mouth keeps the feature scale");
        // A block between the eyes now sees the eye scale, not the feature scale.
        let between_eyes = lambda_scale_for_block(&regions, 72, 88, 16);
        assert!(
            (between_eyes - 0.125).abs() < 1e-9,
            "expected the eye scale, got {between_eyes}"
        );
    }

    #[test]
    fn a_face_yields_face_plus_three_feature_regions() {
        let regions = regions_from_faces(&[centred_face()], 200, 200, 0.5, 0.25, 0.25);
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
        let regions = regions_from_faces(&[centred_face()], 200, 200, 0.5, 0.25, 0.25);
        let face = regions[0];
        assert!(face.col <= 50 && face.row <= 40);
        assert!(face.col + face.width >= 150);
        assert!(face.row + face.height >= 160);
    }

    /// For this centred fixture the landmark padding happens to stay within the detected
    /// box. That is a property of the fixture, not a guarantee: feature boxes are clamped to
    /// the *image*, not to the face box, so on a real detection the eye box can extend a few
    /// pixels past the face box (`--debug-regions` shows exactly where).
    #[test]
    fn feature_boxes_of_a_centred_face_sit_inside_its_face_box() {
        let regions = regions_from_faces(&[centred_face()], 200, 200, 0.5, 0.25, 0.25);
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
        let regions = regions_from_faces(&[centred_face()], 200, 200, 0.5, 0.25, 0.25);
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
        let regions = regions_from_faces(&[edge_face], 64, 64, 0.5, 0.25, 0.25);
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
        assert!(regions_from_faces(&[off_screen], 64, 64, 0.5, 0.25, 0.25).is_empty());
    }

    #[test]
    fn degenerate_landmarks_fall_back_to_a_fraction_of_the_face_box() {
        let mut face = centred_face();
        face.keypoints = [[0.0; 2]; 5];
        let regions = regions_from_faces(&[face], 200, 200, 0.5, 0.25, 0.25);
        // Face box plus three feature boxes, all on-screen -- the fallback scale unit must
        // still produce usable feature regions rather than panicking or collapsing.
        assert_eq!(regions.len(), 4);
        for region in &regions[1..] {
            assert!(region.width > 0 && region.height > 0);
        }
    }

    #[test]
    fn no_faces_yields_no_regions() {
        assert!(regions_from_faces(&[], 200, 200, 0.5, 0.25, 0.25).is_empty());
    }

    #[test]
    fn the_plan_tags_each_region_with_its_feature_kind() {
        let plan = plan_regions(&[centred_face()], 200, 200, 0.5, 0.25, 0.25);
        let kinds: Vec<RegionKind> = plan.iter().map(|planned| planned.kind).collect();
        assert_eq!(
            kinds,
            vec![
                RegionKind::Face,
                RegionKind::Eyes,
                RegionKind::Nose,
                RegionKind::Mouth,
            ],
            "one face box, then the eyes, nose and mouth boxes"
        );
    }

    /// The floor starts at `--min-size` itself and doubles per band, each band as wide as the
    /// floor it applies. `centred_face`'s box covers cols 50..150, rows 40..160 in 200x200.
    #[test]
    fn the_outside_floor_doubles_with_distance_from_the_face() {
        let plan = plan_regions(&[centred_face()], 200, 200, 0.5, 0.25, 0.25);
        // --min-size 4, --max-size 16: bands [0,4)->4, [4,12)->8, [12,..)->16.
        let regions = outside_min_size_regions(&plan, 200, 200, 4, 16);
        assert_eq!(regions.len(), 3, "two rings plus the full-image max floor");
        assert!(
            regions
                .iter()
                .any(|r| r.min_size == Some(16) && (r.width, r.height) == (200, 200)),
            "the coarsest floor is the full-image max region: {regions:?}"
        );
        assert_eq!(
            regions[0].scale, 1.0,
            "a floor region must not touch lambda"
        );

        let floor = |row, col, size| min_size_for_block(&regions, row, col, size, 4);
        // Just left of the face box, inside the first band (which reaches col 46) -> min-size.
        assert_eq!(floor(100, 45, 4), 4);
        // Past the first band but inside the second (col 38) -> doubled.
        assert_eq!(floor(100, 40, 4), 8);
        // Near the left edge, outside every ring -> the saturated max floor.
        assert_eq!(floor(100, 20, 4), 16);
    }

    /// The user's `--min-size 8` case: 8 for the first 8 px, 16 for the next 16 px out to 24,
    /// then 32 -- each band as wide as its own floor.
    #[test]
    fn each_outside_band_is_as_wide_as_its_floor() {
        let plan = plan_regions(&[centred_face()], 200, 200, 0.5, 0.25, 0.25);
        let regions = outside_min_size_regions(&plan, 200, 200, 8, 64);
        let floor = |row, col, size| min_size_for_block(&regions, row, col, size, 8);
        // Face box cols 50..150: ring 8 reaches col 42, ring 16 reaches col 26, ring 32 covers
        // the rest of the width in this 200 px image.
        assert_eq!(floor(100, 45, 4), 8, "within 8 px of the box: min-size");
        assert_eq!(floor(100, 35, 4), 16, "8..24 px out: 16");
        assert_eq!(floor(100, 10, 4), 32, "24..56 px out: 32");
    }

    /// With no room to grow (`--min-size == --max-size`) the only floor is the full image.
    #[test]
    fn no_room_to_grow_gives_a_single_full_image_floor() {
        let plan = plan_regions(&[centred_face()], 200, 200, 0.5, 0.25, 0.25);
        let regions = outside_min_size_regions(&plan, 200, 200, 16, 16);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].min_size, Some(16));
    }

    /// No face box means nothing to be outside of, so no floor is generated -- the caller must
    /// not coarsen the whole image because detection failed. Feature boxes alone do not anchor
    /// the ramp either.
    #[test]
    fn a_plan_with_no_faces_yields_no_outside_regions() {
        assert!(outside_min_size_regions(&[], 200, 200, 4, 32).is_empty());
        let features: Vec<HumanRegion> = plan_regions(&[centred_face()], 200, 200, 0.5, 0.25, 0.25)
            .into_iter()
            .filter(|planned| planned.kind != RegionKind::Face)
            .collect();
        assert!(outside_min_size_regions(&features, 200, 200, 4, 32).is_empty());
    }

    /// A ring can extend past the image; it must be clamped rather than underflow or spill.
    #[test]
    fn outside_rings_stay_inside_the_image() {
        let plan = plan_regions(&[centred_face()], 200, 200, 0.5, 0.25, 0.25);
        let regions = outside_min_size_regions(&plan, 200, 200, 4, 64);
        assert!(!regions.is_empty());
        for region in &regions {
            assert!(
                region.col + region.width <= 200 && region.row + region.height <= 200,
                "{region:?} exceeds the 200x200 image"
            );
        }
    }
}
