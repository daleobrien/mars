//! Claims about the 1998 reference that the Step 2 sweep depends on but does not test.
//!
//! These need the built binaries (`just mars1`) and so are `#[ignore]`d; `just gate-2`
//! runs them. Each one is a *binary* check of something the sweep would otherwise assume,
//! and the sweep's 4680 encodes would not reveal a violation of any of them.

#![allow(clippy::items_after_test_module)]

use std::path::{Path, PathBuf};

use mars_bench::mars1::{
    decode, encode, DecodeMode, DecodeParams, EncodeParams, Mars1Binaries, Mars1Error, Method,
    DEFAULT_ITERATIONS,
};
use mars_bench::measure::{measure, MeasureRequest};

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/mars-bench.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}

/// The binaries plus a short scratch directory with `lena.raw` in it (D3: the reference
/// stores filenames in `char[50]`, so everything is driven with relative names).
fn setup(tag: &str) -> Option<(Mars1Binaries, PathBuf, u32, u32)> {
    let root = repo_root();
    let bins = Mars1Binaries::from_dir(root.join("target/mars1")).ok()?;
    let work = PathBuf::from(format!("/tmp/mars1-t-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&work).expect("scratch dir");
    std::fs::copy(root.join("reference/mars1/lena.raw"), work.join("i.raw")).expect("lena");
    Some((bins, work, 512, 512))
}

macro_rules! require_binaries {
    ($tag:expr) => {
        match setup($tag) {
            Some(v) => v,
            None => {
                eprintln!("skipping: target/mars1 not built; run `just mars1`");
                return;
            }
        }
    };
}

fn base() -> EncodeParams {
    EncodeParams {
        method: Method::Fisher,
        t_rms: 8.0,
        ..Default::default()
    }
}

/// **P2.1.** With `T_ENT = 8.0` an 8-bit block's entropy cannot exceed 8.0, and with
/// `T_VAR = 1e6` an 8-bit variance (bounded by ~16256) cannot exceed it either. So at the
/// 1998 defaults *neither pre-split can ever fire*, and the partition is purely RMS-driven.
///
/// The test is in two halves, and both matter:
///
/// - raising either threshold far above its default must produce a **byte-identical**
///   bitstream, which is what "inert" means;
/// - lowering `-e` to a value real blocks *do* exceed must change the bitstream, which is
///   what stops the first half from passing vacuously because the flag is ignored.
#[test]
#[ignore = "needs the 1998 binaries; run via `just gate-2`"]
fn the_presplit_thresholds_are_inert_at_the_1998_defaults() {
    let (bins, work, w, h) = require_binaries!("inert");

    let encode_to = |name: &str, params: &EncodeParams| -> Vec<u8> {
        encode(&bins, &work, "i.raw", name, w, h, params).expect("encode");
        std::fs::read(work.join(name)).expect("read bitstream")
    };

    let default = encode_to("d.ifs", &base());

    // Inert: the defaults are already unreachable, so raising them changes nothing.
    let raised_ent = encode_to(
        "e.ifs",
        &EncodeParams {
            t_ent: Some(1.0e9),
            ..base()
        },
    );
    assert_eq!(
        default, raised_ent,
        "raising T_ENT above its default changed the bitstream, so the entropy pre-split \
         does fire at 8.0 and the partition is not purely RMS-driven"
    );

    let raised_var = encode_to(
        "v.ifs",
        &EncodeParams {
            t_var: Some(1.0e9),
            ..base()
        },
    );
    assert_eq!(
        default, raised_var,
        "raising T_VAR above its default changed the bitstream, so the variance pre-split \
         does fire at 1e6"
    );

    // Not vacuous: the flags are read, and a reachable threshold does change the output.
    let lowered_ent = encode_to(
        "el.ifs",
        &EncodeParams {
            t_ent: Some(1.0),
            ..base()
        },
    );
    assert_ne!(
        default, lowered_ent,
        "-e is being ignored entirely; the inertness assertions above would then prove \
         nothing about the thresholds"
    );

    let lowered_var = encode_to(
        "vl.ifs",
        &EncodeParams {
            t_var: Some(1.0),
            ..base()
        },
    );
    assert_ne!(
        default, lowered_var,
        "-v is being ignored entirely; see above"
    );

    std::fs::remove_dir_all(&work).ok();
}

/// The check that makes every `method` label in the result store trustworthy.
///
/// `globals.h:201` is `EXTERN int method INIT(= MassCenter)`: **there is no exhaustive
/// mode**, and a dropped flag silently selects MassCenter rather than failing. So each
/// flag must both be accepted and produce the label the driver expects, and the six must
/// produce six genuinely different searches rather than six names for one.
#[test]
#[ignore = "needs the 1998 binaries; run via `just gate-2`"]
fn each_method_flag_selects_a_distinct_search() {
    let (bins, work, w, h) = require_binaries!("methods");

    let mut comparisons = Vec::new();
    for m in Method::ALL {
        let params = EncodeParams {
            method: m,
            ..base()
        };
        // `encode` itself asserts that the reported method equals the requested one.
        let out = encode(&bins, &work, "i.raw", "o.ifs", w, h, &params).expect("encode");
        assert_eq!(out.stats.method_reported, m.reported_label());
        assert!(out.stats.transforms > 0);
        comparisons.push((m, out.stats.comparisons));
    }

    // Six methods, six different search costs. If two matched exactly, the more likely
    // explanation is that one flag is not taking effect than that two published
    // classification schemes evaluate an identical number of domains.
    let mut counts: Vec<u64> = comparisons.iter().map(|(_, c)| *c).collect();
    let before = counts.len();
    counts.sort_unstable();
    counts.dedup();
    assert_eq!(
        counts.len(),
        before,
        "two methods reported identical comparison counts: {comparisons:?}"
    );

    std::fs::remove_dir_all(&work).ok();
}

/// The driver's own oracle, exercised against a real run: the encoder's printed byte
/// count equals the file on disk, and its printed ratio equals its printed integers.
/// `encode` returns an error rather than a row if either fails.
#[test]
#[ignore = "needs the 1998 binaries; run via `just gate-2`"]
fn the_encoder_summary_is_self_consistent() {
    let (bins, work, w, h) = require_binaries!("consistent");
    let out = encode(&bins, &work, "i.raw", "o.ifs", w, h, &base()).expect("encode");
    assert_eq!(
        out.coded_bytes,
        std::fs::metadata(work.join("o.ifs")).unwrap().len()
    );
    assert_eq!(out.stats.output_name, "o.ifs");
    assert_eq!(
        format!("{:.6}", out.stats.evals_per_transform()),
        format!("{:.6}", out.stats.comparisons_per_transform_reported)
    );
    std::fs::remove_dir_all(&work).ok();
}

/// A wrong `-W`/`-H` must not quietly produce a plausible result. This is the failure
/// that would corrupt the whole sweep if the image-set index ever disagreed with the raw
/// files, so the decoder's echoed size is checked against what we asked for.
#[test]
#[ignore = "needs the 1998 binaries; run via `just gate-2`"]
fn a_dimension_mismatch_is_caught_not_absorbed() {
    let (bins, work, w, h) = require_binaries!("dims");
    encode(&bins, &work, "i.raw", "o.ifs", w, h, &base()).expect("encode");
    let err = decode(
        &bins,
        &work,
        "o.ifs",
        "d.pgm",
        (256, 256), // deliberately wrong
        &DecodeParams::default(),
    )
    .expect_err("a size mismatch must be an error");
    assert!(matches!(err, Mars1Error::DecodeSizeMismatch { .. }));
    std::fs::remove_dir_all(&work).ok();
}

/// **P2.2, measured.** The plan and the `mars1-reference` skill both warn that comparing
/// a pyramidal decode against an iterative one is "a silent 0.2–1 dB error". Measured on
/// the same bitstream, that is true at **5** iterations and false at the 1998 default of
/// **10**, where the two agree to ~0.004 dB.
///
/// The mechanism is convergence, not approximation: an IFS has a unique attracting fixed
/// point, so both decoders approach the same image. Pyramidal is an *accelerator* — it
/// iterates at reduced resolution first and arrives sooner — rather than a cheaper,
/// worse reconstruction.
///
/// This test pins both halves of that, because each protects a different mistake:
/// the modes must be **distinguishable at 1 iteration** (or `-i` is not taking effect and
/// recording the mode measures nothing), and **converged by the default count** (or the
/// baseline's decode mode really would be a confound and every row's PSNR would need a
/// mode-matched comparison).
#[test]
#[ignore = "needs the 1998 binaries; run via `just gate-2`"]
fn the_decode_modes_diverge_when_unconverged_and_agree_once_converged() {
    let (bins, work, w, h) = require_binaries!("decode");
    encode(&bins, &work, "i.raw", "o.ifs", w, h, &base()).expect("encode");

    let psnr_at = |iterations: u32, mode: DecodeMode, name: &str| -> f64 {
        let params = DecodeParams {
            mode,
            iterations: Some(iterations),
            postprocess: false,
        };
        decode(&bins, &work, "o.ifs", name, (w, h), &params).expect("decode");
        let req = MeasureRequest::new(work.join("i.raw"), work.join(name))
            .with_raw_dims(w as usize, h as usize);
        measure(&req)
            .expect("measure")
            .psnr_y
            .expect("a lossy decode has finite PSNR")
    };

    // Unconverged: the two modes are far apart, so `-i` demonstrably takes effect.
    let gap_1 =
        psnr_at(1, DecodeMode::Pyramidal, "p1.pgm") - psnr_at(1, DecodeMode::Iterative, "i1.pgm");
    assert!(
        gap_1 > 1.0,
        "at 1 iteration the modes differed by only {gap_1:.4} dB; -i may not be taking \
         effect, which would make the convergence assertion below vacuous"
    );

    // Converged: at the 1998 default the mode is worth ~nothing. If this ever fails, the
    // baseline's decode mode is a confound and every PSNR comparison must be mode-matched.
    let gap_10 = (psnr_at(DEFAULT_ITERATIONS, DecodeMode::Pyramidal, "p10.pgm")
        - psnr_at(DEFAULT_ITERATIONS, DecodeMode::Iterative, "i10.pgm"))
    .abs();
    assert!(
        gap_10 < 0.05,
        "at the default {DEFAULT_ITERATIONS} iterations the modes differed by {gap_10:.4} dB, \
         far more than the ~0.004 dB measured when this was written; the two decoders are \
         no longer converging to the same fixed point"
    );

    // And the ordering that makes "accelerator, not approximation" the right description.
    assert!(
        gap_1 > gap_10,
        "the modes did not converge towards each other as iterations increased"
    );

    std::fs::remove_dir_all(&work).ok();
}

/// **D11.** `decmars` decodes at `1/2^levels` scale before raising resolution, so a range
/// block of `min_size` occupies `min_size / 2^levels` pixels there. At or above one pixel
/// the two decode modes agree; at half a pixel pyramidal loses 5+ dB and never recovers it.
///
/// Both directions are asserted. The whole-pixel case alone would not distinguish "the
/// rule holds" from "`-i` does nothing"; the sub-pixel case alone would not distinguish
/// "sub-pixel blocks break" from "small blocks break", which is the wrong lesson and the
/// one that would send someone hunting in the encoder.
#[test]
#[ignore = "needs the 1998 binaries; run via `just gate-2`"]
fn pyramidal_decode_breaks_only_on_sub_pixel_range_blocks() {
    let (bins, work, w, h) = require_binaries!("subpixel");

    // lena is 512x512, so decmars builds a 2-level pyramid: min_size 2 lands at half a
    // pixel there, min_size 4 at a whole one.
    let gap_at = |min_size: u32| -> f64 {
        let params = EncodeParams {
            min_size,
            t_rms: 4.0,
            ..base()
        };
        encode(&bins, &work, "i.raw", "o.ifs", w, h, &params).expect("encode");
        let psnr = |mode: DecodeMode, name: &str| {
            decode(
                &bins,
                &work,
                "o.ifs",
                name,
                (w, h),
                &DecodeParams {
                    mode,
                    iterations: None,
                    postprocess: false,
                },
            )
            .expect("decode");
            let req = MeasureRequest::new(work.join("i.raw"), work.join(name))
                .with_raw_dims(w as usize, h as usize);
            measure(&req).expect("measure").psnr_y.expect("lossy")
        };
        psnr(DecodeMode::Iterative, "it.pgm") - psnr(DecodeMode::Pyramidal, "py.pgm")
    };

    let sub_pixel = gap_at(2);
    let whole_pixel = gap_at(4);

    assert!(
        sub_pixel > 1.0,
        "min_size=2 on a 512x512 image puts range blocks at half a pixel in a 2-level \
         pyramid, which cost 5+ dB when measured; the gap here was only {sub_pixel:.3} dB"
    );
    assert!(
        whole_pixel.abs() < 0.05,
        "min_size=4 keeps range blocks at a whole pixel and the modes agreed to ~0.05 dB \
         when measured; the gap here was {whole_pixel:.3} dB, so the defect is not the \
         sub-pixel one D11 describes"
    );

    std::fs::remove_dir_all(&work).ok();
}
