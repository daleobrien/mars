//! Claims about the 1998 reference that the Step 2 sweep depends on but does not test.
//!
//! These need the built binaries (`just mars1`) and so are `#[ignore]`d; `just gate-2`
//! runs them. Each one is a *binary* check of something the sweep would otherwise assume,
//! and the sweep's 4680 encodes would not reveal a violation of any of them.

#![allow(clippy::items_after_test_module)]

use std::path::{Path, PathBuf};

use mars_bench::mars1::{
    decode, encode, DecodeMode, DecodeParams, EncodeParams, Mars1Binaries, Mars1Error, Method,
};

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

/// **P2.2's mechanism.** Pyramidal and iterative decode the *same* bitstream, so any
/// difference between them is a decoder property, not a coding one — which is exactly why
/// mixing the two silently corrupts a comparison. This asserts only that the two modes are
/// distinguishable; the size of the difference is measured by the sweep, not decided here.
#[test]
#[ignore = "needs the 1998 binaries; run via `just gate-2`"]
fn the_two_decode_modes_really_do_differ() {
    let (bins, work, w, h) = require_binaries!("decode");
    encode(&bins, &work, "i.raw", "o.ifs", w, h, &base()).expect("encode");

    for (mode, name) in [
        (DecodeMode::Pyramidal, "p.pgm"),
        (DecodeMode::Iterative, "i.pgm"),
    ] {
        let params = DecodeParams {
            mode,
            iterations: None,
            postprocess: false,
        };
        decode(&bins, &work, "o.ifs", name, (w, h), &params).expect("decode");
    }
    let p = std::fs::read(work.join("p.pgm")).expect("pyramidal output");
    let i = std::fs::read(work.join("i.pgm")).expect("iterative output");
    assert_ne!(
        p, i,
        "the two decode modes produced identical output; either -i is not taking effect \
         or the modes have converged, and in both cases recording the mode per row would \
         be measuring nothing"
    );
    std::fs::remove_dir_all(&work).ok();
}
