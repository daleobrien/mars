use mars_codec::postprocess::smooth_boundaries;
use mars_core::Plane;

fn grid(width: usize, height: usize, size: usize) -> Vec<(usize, usize, usize)> {
    (0..height)
        .step_by(size)
        .flat_map(|row| (0..width).step_by(size).map(move |col| (row, col, size)))
        .collect()
}

fn psnr(reference: &Plane, decoded: &Plane) -> f64 {
    let mse = reference
        .as_slice()
        .iter()
        .zip(decoded.as_slice())
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / reference.as_slice().len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

#[test]
fn constants_are_byte_identical_including_extremes() {
    for value in [0, 1, 100, 128, 254, 255] {
        let plane = Plane::filled(9, 7, value);
        assert_eq!(smooth_boundaries(&plane, &grid(9, 7, 4)), plane);
    }
}

#[test]
fn strong_edges_and_thin_lines_are_preserved_exactly() {
    for vertical in [false, true] {
        for thin in [false, true] {
            let pixels = (0..64)
                .map(|i| {
                    let coordinate = if vertical { i % 8 } else { i / 8 };
                    if if thin {
                        coordinate == 4
                    } else {
                        coordinate >= 4
                    } {
                        240
                    } else {
                        16
                    }
                })
                .collect();
            let plane = Plane::from_vec(8, 8, pixels);
            assert_eq!(smooth_boundaries(&plane, &grid(8, 8, 1)), plane);
        }
    }
}

#[test]
fn block_artifact_psnr_improves_and_natural_ramp_does_not_degrade() {
    // A flat 120 reference corrupted by alternating 8x8 DC offsets (100/140),
    // plus fixed +/-3 sample noise. The checker here is a blocking artifact, not
    // true scene texture: this test does not claim all checkerboards benefit.
    let reference = Plane::filled(32, 32, 120);
    let pixels = (0..1024)
        .map(|i| {
            let base = if ((i / 32 / 8) + (i % 32 / 8)) % 2 == 0 {
                100
            } else {
                140
            };
            (base + (i * 37 % 7) as i32 - 3) as u8
        })
        .collect();
    let decoded = Plane::from_vec(32, 32, pixels);
    let leaves = grid(32, 32, 8);
    let filtered = smooth_boundaries(&decoded, &leaves);
    let before = psnr(&reference, &decoded);
    let after = psnr(&reference, &filtered);
    println!("block artifact PSNR: {before:.6} -> {after:.6} dB");
    assert!(after > before, "{before} -> {after}");

    // A gently varying affine scene, with a two-level reconstruction bias so PSNR
    // is finite. A unit neighbor difference should not become a rounded blur step.
    let ramp = Plane::from_vec(
        32,
        32,
        (0..1024).map(|i| (60 + i / 32 + i % 32) as u8).collect(),
    );
    let decoded_ramp = Plane::from_vec(32, 32, ramp.as_slice().iter().map(|p| p + 2).collect());
    let filtered_ramp = smooth_boundaries(&decoded_ramp, &leaves);
    let before_ramp = psnr(&ramp, &decoded_ramp);
    let after_ramp = psnr(&ramp, &filtered_ramp);
    println!("natural ramp PSNR: {before_ramp:.6} -> {after_ramp:.6} dB");
    assert!(after_ramp + 0.01 >= before_ramp);
    assert_eq!(filtered_ramp, decoded_ramp);
    assert_eq!(smooth_boundaries(&ramp, &leaves), ramp);
}

#[test]
fn mixed_boundaries_corners_are_deduplicated_and_input_is_immutable() {
    // A size-4 leaf meets size-2 leaves at a T junction in an odd-height image.
    let leaves = [(0, 0, 4), (0, 4, 2), (2, 4, 2), (4, 0, 4), (4, 4, 2)];
    let plane = Plane::from_vec(
        6,
        5,
        (0..30)
            .map(|i| {
                if i % 6 < 4 {
                    100
                } else if i / 6 < 2 {
                    120
                } else {
                    140
                }
            })
            .collect(),
    );
    let original = plane.clone();
    let filtered = smooth_boundaries(&plane, &leaves);
    assert_eq!(plane, original);
    assert_ne!(filtered, plane);
    assert!(filtered.get(3, 0) > 100);
    assert!(filtered.get(3, 3) > 100);
    assert_eq!(filtered.get(1, 1), 100); // Not adjacent to any boundary.
    let mut duplicates = leaves.to_vec();
    duplicates.extend(leaves);
    duplicates.reverse();
    assert_eq!(smooth_boundaries(&plane, &duplicates), filtered);
    assert_eq!(smooth_boundaries(&plane, &leaves), filtered);

    let corner = Plane::from_vec(
        4,
        4,
        (0..16)
            .map(|i| 100 + (i % 4 / 2) as u8 * 20 + (i / 4 / 2) as u8 * 40)
            .collect(),
    );
    // At (1,1), g=40, cross-neighbor mean=130, w=0.0703125:
    // round(100 + 30*w)=102, not the result of sequential directional updates.
    assert_eq!(smooth_boundaries(&corner, &grid(4, 4, 2)).get(1, 1), 102);
}

#[test]
fn borders_empty_geometry_and_odd_sizes_are_safe() {
    for (width, height) in [
        (0, 0),
        (0, 5),
        (5, 0),
        (1, 1),
        (1, 7),
        (7, 1),
        (3, 5),
        (9, 7),
    ] {
        let plane = Plane::from_vec(
            width,
            height,
            (0..width * height).map(|i| (i * 31 % 256) as u8).collect(),
        );
        assert_eq!(smooth_boundaries(&plane, &[]), plane);
        assert_eq!(smooth_boundaries(&plane, &[(0, 0, usize::MAX)]), plane);
        let mut leaves = grid(width, height, 2);
        leaves.extend([(usize::MAX, 0, 4), (0, usize::MAX, 4), (0, 0, 0)]);
        let filtered = smooth_boundaries(&plane, &leaves);
        assert_eq!((filtered.width(), filtered.height()), (width, height));
        assert_eq!(
            smooth_boundaries(&filtered, &[(0, 0, usize::MAX)]),
            filtered
        );
    }
}

#[test]
fn weight_decreases_to_zero_with_increasing_contrast() {
    // A single cross-boundary neighbor makes the observed blend ratio explicit.
    let mut last_ratio = 1.0;
    for contrast in [16u8, 32, 48, 64, 128] {
        let plane = Plane::from_vec(2, 1, vec![64, 64 + contrast]);
        let filtered = smooth_boundaries(&plane, &[(0, 0, 1), (0, 1, 1)]);
        let ratio = f64::from(filtered.get(0, 0) - 64) / f64::from(contrast);
        assert!(ratio <= last_ratio);
        last_ratio = ratio;
        if contrast >= 64 {
            assert_eq!(filtered, plane);
        }
    }
}
