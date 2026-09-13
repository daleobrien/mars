//! Step 1's known-answer vectors.
//!
//! These are the tests the step "lives or dies on". Every number downstream inherits
//! whatever is wrong here, and §2.1's failure mode — silent plausible wrongness — means
//! nothing downstream will surface it. So the assertions are against hand-computed
//! values and exact identities, not against whatever the implementation happens to
//! produce today.

use mars_core::image::{Image, Plane};
use mars_core::metrics::{
    bpp, ms_ssim, mse, psnr, psnr_from_mse, quality, ssim, Downsample, MsSsimConfig, SsimConfig,
    Unavailable, MSSSIM_MIN_DIM,
};
use mars_core::tolerance::{MSE_KNOWN_ANSWER_ABS, SSIM_SELF_ABS};

fn plane(w: usize, h: usize, v: &[u8]) -> Plane {
    Plane::from_vec(w, h, v.to_vec())
}

/// Deterministic pseudo-random plane. A fixed LCG rather than a crate, so the fixture is
/// reproducible with no dependency and no seed-from-clock hazard (§2.3).
fn lcg_plane(w: usize, h: usize, seed: u64) -> Plane {
    let mut s = seed;
    let data = (0..w * h)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 33) as u8
        })
        .collect();
    Plane::from_vec(w, h, data)
}

// ---------------------------------------------------------------------------
// MSE / PSNR
// ---------------------------------------------------------------------------

#[test]
fn mse_of_identical_planes_is_zero_and_psnr_is_null() {
    let p = lcg_plane(32, 32, 1);
    assert_eq!(mse(&p, &p), 0.0);
    // §M2: infinity is emitted as null, never as a large sentinel float. A 99.0 or 1e9
    // sentinel would survive an average and poison it.
    assert_eq!(psnr(&p, &p), None);
    assert_eq!(psnr_from_mse(0.0), None);
}

#[test]
fn mse_matches_the_hand_computed_value() {
    // Differences: 1, -2, 3, -4 over 4 pixels.
    // MSE = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    let a = plane(2, 2, &[10, 20, 30, 40]);
    let b = plane(2, 2, &[11, 18, 33, 36]);
    assert!((mse(&a, &b) - 7.5).abs() < MSE_KNOWN_ANSWER_ABS);

    // A second vector whose PSNR is also hand-checkable:
    // every pixel differs by exactly 1 => MSE = 1 => PSNR = 10*log10(65025) dB.
    let a = plane(4, 4, &[100; 16]);
    let b = plane(4, 4, &[101; 16]);
    assert!((mse(&a, &b) - 1.0).abs() < MSE_KNOWN_ANSWER_ABS);
    let expected = 10.0 * 65_025f64.log10();
    assert!((psnr(&a, &b).unwrap() - expected).abs() < MSE_KNOWN_ANSWER_ABS);
    // ...which is 48.13 dB, the number every codec paper's "+-1 LSB" case reports.
    assert!((psnr(&a, &b).unwrap() - 48.130_803_608_679_02).abs() < 1e-11);
}

#[test]
fn maximum_error_gives_zero_db() {
    // 0 vs 255 everywhere: MSE = 255^2, so PSNR = 10*log10(1) = 0 dB exactly.
    let a = Plane::filled(8, 8, 0);
    let b = Plane::filled(8, 8, 255);
    assert!((mse(&a, &b) - 65_025.0).abs() < MSE_KNOWN_ANSWER_ABS);
    assert!(psnr(&a, &b).unwrap().abs() < 1e-12);
}

#[test]
fn mse_is_exact_for_large_accumulations() {
    // The sum of squared differences is accumulated in u64, so it is exact rather than
    // subject to f32/f64 accumulation drift. 1024x1024 of maximal error is the worst
    // case: 255^2 * 2^20, which must come back exactly.
    let a = Plane::filled(1024, 1024, 0);
    let b = Plane::filled(1024, 1024, 255);
    assert_eq!(mse(&a, &b), 65_025.0);
}

#[test]
fn padding_is_never_measured() {
    // Mars 1 pads to virtual_size (next power of two). §M2 measures the original W x H
    // region only. Here the original is 3x3 and the "decode" is 4x4 with garbage in the
    // pad, which must not move the number at all.
    let original = Image::gray(plane(3, 3, &[10, 20, 30, 40, 50, 60, 70, 80, 90]));
    let padded = Image::gray(plane(
        4,
        4,
        &[
            10, 20, 30, 255, //
            40, 50, 60, 0, //
            70, 80, 90, 255, //
            0, 0, 0, 0,
        ],
    ));
    let q = quality(
        &original,
        &padded,
        Some((3, 3)),
        &SsimConfig::default(),
        &MsSsimConfig::default(),
    );
    assert_eq!(q.mse_per_plane, vec![0.0]);
    assert_eq!(q.psnr_per_plane, vec![None]);
    assert_eq!((q.width, q.height), (3, 3));
}

#[test]
fn bpp_is_over_the_whole_file() {
    // 512x512 at 8192 bytes: 8 * 8192 / 262144 = 0.25 bpp.
    assert!((bpp(8192, 512, 512) - 0.25).abs() < 1e-15);
    // An uncompressed 8-bit image is exactly 8.0 bpp, which is the sanity anchor.
    assert!((bpp(512 * 512, 512, 512) - 8.0).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// SSIM
// ---------------------------------------------------------------------------

#[test]
fn ssim_of_an_image_with_itself_is_one() {
    for seed in [1u64, 2, 3] {
        let p = lcg_plane(64, 48, seed);
        let r = ssim(&p, &p, &SsimConfig::default()).unwrap();
        assert!(
            (r.ssim - 1.0).abs() < SSIM_SELF_ABS,
            "SSIM(x,x) = {} for seed {seed}",
            r.ssim
        );
        assert!((r.cs - 1.0).abs() < SSIM_SELF_ABS, "CS(x,x) = {}", r.cs);
    }
    // Also for a flat plane, where every variance is zero and only the stabilising
    // constants keep the ratio defined.
    let flat = Plane::filled(32, 32, 128);
    assert!((ssim(&flat, &flat, &SsimConfig::default()).unwrap().ssim - 1.0).abs() < SSIM_SELF_ABS);
}

#[test]
fn ssim_is_symmetric() {
    let a = lcg_plane(64, 64, 7);
    let b = lcg_plane(64, 64, 8);
    let cfg = SsimConfig::default();
    let ab = ssim(&a, &b, &cfg).unwrap().ssim;
    let ba = ssim(&b, &a, &cfg).unwrap().ssim;
    assert!((ab - ba).abs() < 1e-15, "{ab} vs {ba}");
}

#[test]
fn ssim_has_a_closed_form_for_two_flat_planes() {
    // Every window has mu1 = a, mu2 = b, all variances and the covariance zero. So
    //   luminance = (2ab + C1) / (a^2 + b^2 + C1),  contrast = C2 / C2 = 1.
    let cfg = SsimConfig::default();
    let c1 = (cfg.k1 * cfg.peak).powi(2);
    for (a, b) in [(100u8, 120u8), (0, 255), (200, 201)] {
        let (fa, fb) = (f64::from(a), f64::from(b));
        let expected = (2.0 * fa * fb + c1) / (fa * fa + fb * fb + c1);
        let got = ssim(&Plane::filled(32, 32, a), &Plane::filled(32, 32, b), &cfg).unwrap();
        assert!(
            (got.ssim - expected).abs() < 1e-12,
            "({a},{b}): got {}, expected {expected}",
            got.ssim
        );
        assert!((got.cs - 1.0).abs() < 1e-12, "contrast should be exactly 1");
    }
}

#[test]
fn ssim_decreases_as_distortion_grows() {
    // A property that holds regardless of implementation (§A3), so it survives any
    // later rewrite of the filter.
    let base = lcg_plane(96, 96, 11);
    let cfg = SsimConfig::default();
    let mut last = 1.000_000_1;
    for amp in [0i32, 2, 5, 10, 25, 60] {
        let noisy: Vec<u8> = base
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let d = if i % 2 == 0 { amp } else { -amp };
                (i32::from(v) + d).clamp(0, 255) as u8
            })
            .collect();
        let s = ssim(&base, &Plane::from_vec(96, 96, noisy), &cfg)
            .unwrap()
            .ssim;
        assert!(
            s < last,
            "SSIM did not decrease at amplitude {amp}: {s} >= {last}"
        );
        last = s;
    }
}

#[test]
fn ssim_is_unavailable_rather_than_wrong_on_a_tiny_image() {
    let p = lcg_plane(10, 10, 1);
    assert_eq!(
        ssim(&p, &p, &SsimConfig::default()),
        Err(Unavailable::TooSmall)
    );
}

// ---------------------------------------------------------------------------
// MS-SSIM
// ---------------------------------------------------------------------------

#[test]
fn ms_ssim_of_an_image_with_itself_is_one() {
    let p = lcg_plane(256, 192, 21);
    let v = ms_ssim(&p, &p, &MsSsimConfig::default()).unwrap();
    assert!((v - 1.0).abs() < 1e-12, "MS-SSIM(x,x) = {v}");
}

#[test]
fn ms_ssim_with_one_scale_reduces_to_ssim() {
    // A structural identity: with weights [1,0,0,0,0] the product collapses to the
    // first scale's SSIM. This is what ties MS-SSIM to the cross-validated SSIM —
    // if the scale chain or the weighting is wrong, this breaks.
    let a = lcg_plane(256, 256, 31);
    let b = {
        let mut v = a.as_slice().to_vec();
        for (i, p) in v.iter_mut().enumerate() {
            *p = p.wrapping_add((i % 7) as u8);
        }
        Plane::from_vec(256, 256, v)
    };
    // One scale: the product has a single factor, and at the final scale that factor is
    // the full SSIM (luminance included) rather than contrast-structure alone.
    let cfg = MsSsimConfig {
        weights: vec![1.0],
        ..MsSsimConfig::default()
    };
    let got = ms_ssim(&a, &b, &cfg).unwrap();
    let want = ssim(&a, &b, &SsimConfig::default()).unwrap().ssim;
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
}

#[test]
fn only_the_final_scale_contributes_luminance() {
    // Two scales with weights [1, 0] must give exactly cs at scale 1, while [0, 1] gives
    // the SSIM of the decimated image. If luminance leaked into the earlier scales, the
    // first of these would return SSIM instead of CS.
    let a = lcg_plane(256, 256, 33);
    let b = Plane::from_vec(
        256,
        256,
        a.as_slice().iter().map(|&v| v.wrapping_add(9)).collect(),
    );
    let cfg = |w: Vec<f64>| MsSsimConfig {
        weights: w,
        ..MsSsimConfig::default()
    };
    let scale1 = ssim(&a, &b, &SsimConfig::default()).unwrap();
    let got_cs = ms_ssim(&a, &b, &cfg(vec![1.0, 0.0])).unwrap();
    assert!(
        (got_cs - scale1.cs).abs() < 1e-12,
        "{got_cs} vs cs {}",
        scale1.cs
    );
    assert!(
        (got_cs - scale1.ssim).abs() > 1e-9,
        "a uniform +9 offset must make luminance differ from 1, or this test proves nothing"
    );
}

#[test]
fn ms_ssim_below_the_minimum_dimension_is_flagged_not_rescaled() {
    // §M2: images below 176 in the smaller dimension are flagged and excluded. Silently
    // rescaling them would put an incomparable number in the same column as the rest.
    let p = lcg_plane(MSSSIM_MIN_DIM - 1, 256, 41);
    assert_eq!(
        ms_ssim(&p, &p, &MsSsimConfig::default()),
        Err(Unavailable::TooSmall)
    );
    let ok = lcg_plane(MSSSIM_MIN_DIM, MSSSIM_MIN_DIM, 42);
    assert!(ms_ssim(&ok, &ok, &MsSsimConfig::default()).is_ok());
}

#[test]
fn ms_ssim_decreases_as_distortion_grows() {
    let base = lcg_plane(256, 256, 51);
    let mut last = 1.000_000_1;
    for amp in [0i32, 3, 8, 20, 50] {
        let noisy: Vec<u8> = base
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let d = if (i / 3) % 2 == 0 { amp } else { -amp };
                (i32::from(v) + d).clamp(0, 255) as u8
            })
            .collect();
        let v = ms_ssim(
            &base,
            &Plane::from_vec(256, 256, noisy),
            &MsSsimConfig::default(),
        )
        .unwrap();
        assert!(
            v < last,
            "MS-SSIM did not decrease at amplitude {amp}: {v} >= {last}"
        );
        last = v;
    }
}

#[test]
fn the_two_downsample_phases_really_do_differ() {
    // The `ScipyUniform2` variant exists only so the scale chain can be cross-validated
    // against `sewar`. If it ever became identical to the pinned `Box2x2`, the
    // cross-validation would be vacuous, so assert that it is a genuine difference.
    let a = lcg_plane(256, 256, 61);
    let b = {
        let mut v = a.as_slice().to_vec();
        for (i, p) in v.iter_mut().enumerate() {
            *p = p.wrapping_add((i % 11) as u8);
        }
        Plane::from_vec(256, 256, v)
    };
    let boxed = ms_ssim(&a, &b, &MsSsimConfig::default()).unwrap();
    let scipy = ms_ssim(
        &a,
        &b,
        &MsSsimConfig {
            downsample: Downsample::ScipyUniform2,
            ..MsSsimConfig::default()
        },
    )
    .unwrap();
    assert_ne!(boxed, scipy);
    // ...but they must still be close, or one of them is wrong rather than merely
    // phase-shifted.
    assert!((boxed - scipy).abs() < 0.05, "{boxed} vs {scipy}");
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

#[test]
fn psnr_yuv_uses_the_pinned_weighting() {
    let r = lcg_plane(64, 64, 71);
    let g = lcg_plane(64, 64, 72);
    let b = lcg_plane(64, 64, 73);
    let original = Image::rgb(r.clone(), g.clone(), b.clone());

    let bump = |p: &Plane| {
        Plane::from_vec(
            64,
            64,
            p.as_slice().iter().map(|&v| v.saturating_add(3)).collect(),
        )
    };
    let decoded = Image::rgb(bump(&r), bump(&g), bump(&b));

    let q = quality(
        &original,
        &decoded,
        None,
        &SsimConfig::default(),
        &MsSsimConfig::default(),
    );
    assert_eq!(q.mse_per_plane.len(), 3, "per-channel MSE for colour input");
    let (y, cb, cr) = (q.psnr_y.unwrap(), q.psnr_cb.unwrap(), q.psnr_cr.unwrap());
    let expected = (6.0 * y + cb + cr) / 8.0;
    assert!((q.psnr_yuv.unwrap() - expected).abs() < 1e-12);
}

#[test]
fn identical_colour_images_give_null_everywhere_rather_than_a_finite_psnr_yuv() {
    // A weighted mean that silently drops infinite components would read as a finite,
    // wrong number. It must be null instead.
    let img = Image::rgb(
        lcg_plane(64, 64, 81),
        lcg_plane(64, 64, 82),
        lcg_plane(64, 64, 83),
    );
    let q = quality(
        &img,
        &img,
        None,
        &SsimConfig::default(),
        &MsSsimConfig::default(),
    );
    assert_eq!(q.psnr_per_plane, vec![None, None, None]);
    assert_eq!(q.psnr_y, None);
    assert_eq!(q.psnr_yuv, None);
}

#[test]
fn grayscale_reports_no_chroma_columns() {
    let a = lcg_plane(64, 64, 91);
    let b = lcg_plane(64, 64, 92);
    let q = quality(
        &Image::gray(a),
        &Image::gray(b),
        None,
        &SsimConfig::default(),
        &MsSsimConfig::default(),
    );
    assert_eq!(q.mse_per_plane.len(), 1);
    assert_eq!(q.psnr_cb, None);
    assert_eq!(q.psnr_cr, None);
    assert_eq!(q.psnr_yuv, None, "PSNR-YUV is meaningless for grayscale");
    // For grayscale, PSNR-Y must be exactly the plane PSNR: no colour conversion is
    // applied, so there is no +-1 LSB drift between the two columns.
    assert_eq!(q.psnr_y, q.psnr_per_plane[0]);
}
