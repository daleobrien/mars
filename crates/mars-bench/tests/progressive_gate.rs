//! Step 19's exit-criterion measurement: the RD curve *of the prefixes* against
//! separately-encoded single-layer streams at the same rates, and the number the brief
//! calls the honest one -- the **progressive penalty** (BD-rate lost to truncatability).
//!
//! **Not part of `just gate-19`.** Per the brief's own instruction ("keep it fast --
//! expensive corpus sweeps belong in `docs/predictions.md`'s measured outcome, run once by
//! hand") and this project's gate-18/gate-14/gate-15/gate-16 precedent for exactly this
//! split: `just gate-19` runs only `mars-codec`'s fast synthetic-image unit tests. This
//! file is gated behind `MARS_RUN_PROGRESSIVE_GATE=1` and is run once by hand; its output
//! is recorded verbatim in `docs/predictions.md`'s Step 19 outcome.
//!
//! **Design of the comparison.** One progressive stream (one λ, `LAMBDA_GRID[1] = 200`,
//! the RD sweep's own mid-range point) gives the "curve of the prefixes": 4 points, one
//! per layer boundary, strictly increasing in both bpp and PSNR by construction whenever
//! quality does not regress layer to layer (checked, not assumed). The reference is 4
//! *separately-encoded* single-layer `.mars` v0 streams, one per λ in the same
//! `LAMBDA_GRID` every other RD gate in this project uses (D28/D36/D39/D40/D43's
//! precedent) -- literally the same encoder, same leaves at each λ, just serialised
//! without layering. `bd_metrics`'s PCHIP-on-log-bpp interpolation (§M3) is exactly the
//! tool that operationalises "at the same rate": it restricts the comparison to the PSNR
//! range the two curves actually overlap, and refuses (rather than fabricates) a number if
//! they do not overlap at all.
//!
//! Scoped to `kodim01` only (not the full 24-image `standard/` corpus) -- the same
//! session-time scope cut every prior RD gate in this project has made, stated plainly per
//! the D28/Step-13/Step-18 precedent.

use mars_codec::encode::{encode_image_rd, EncodeParams};
use mars_codec::ifs::Header;
use mars_codec::mars_format;
use mars_codec::progressive;
use mars_core::io::read_raw;
use mars_core::metrics::psnr;
use mars_core::Plane;

use mars_bench::bdrate::{bd_metrics, RdCurve, RdPoint};
use mars_bench::rd_opt::check_convex_and_monotonic;

/// Identical to every other RD gate's `BASE` in this project.
const BASE: EncodeParams = EncodeParams {
    min_size: 4,
    max_size: 16,
    shift: 4,
    bits_alfa: 4,
    bits_beta: 7,
    max_alfa: 1.0,
    t_rms: 0.0,
    zero_threshold: 0,
    lambda: None,
};

/// Identical to `rd_gate.rs`'s own `LAMBDA_GRID` -- the same, already-validated operating
/// range on `kodim01`, so this gate's single-layer reference curve is directly comparable
/// to every other step's own lambda sweep on this image.
const LAMBDA_GRID: [f64; 4] = [50.0, 200.0, 800.0, 3200.0];

/// The one λ whose progressive stream supplies the "curve of the prefixes" -- the sweep's
/// own mid-range point, chosen so its layer-4 (terminal) rate sits inside the reference
/// curve's own bpp range rather than at an extreme.
const PROGRESSIVE_LAMBDA: f64 = LAMBDA_GRID[1];

fn kodim01() -> Plane {
    let path = std::path::Path::new("../../corpus/images/kodak-gray/kodim01.raw");
    read_raw(path, 768, 512).expect(
        "corpus/images/kodak-gray/kodim01.raw must exist -- see docs/predictions.md's Step \
         19 entry for the fetch+ppm2raw provenance this gate assumes was run by hand first",
    )
}

/// The single-layer reference curve: `LAMBDA_GRID` swept through the ordinary, non-
/// progressive `.mars` v0 encoder -- exactly `mars_bench::rd_opt::lambda_curve`'s own
/// shape, reimplemented here (not imported) only so this file states its own BASE/GRID
/// choice explicitly rather than depending on another gate's constants silently staying
/// in sync.
fn single_layer_reference_curve(image: &Plane) -> RdCurve {
    let mut points = Vec::with_capacity(LAMBDA_GRID.len());
    for &lambda in &LAMBDA_GRID {
        let params = EncodeParams {
            lambda: Some(lambda),
            ..BASE
        };
        let (hdr, leaves, _evals, _stats) = encode_image_rd(image, &params);
        let bytes = mars_format::write(&hdr, &leaves)
            .expect("a partition encode_image_rd produced must always be a writable .mars v0 tree");
        let decoded =
            mars_codec::ifs::decode_iterative(&hdr, &leaves, progressive::DECODE_ITERATIONS);
        let psnr_db = psnr(image, &decoded).unwrap_or(f64::INFINITY);
        points.push(RdPoint::from_size(
            bytes.len() as u64,
            image.width(),
            image.height(),
            psnr_db,
        ));
    }
    RdCurve::new("single-layer (.mars v0)", points)
}

/// The progressive "curve of the prefixes": one λ's own stream, truncated at each of its 4
/// layer boundaries, each prefix's byte length giving bpp and its decode giving PSNR.
fn progressive_prefix_curve(
    image: &Plane,
    hdr: &Header,
    leaves: &[mars_codec::ifs::Leaf],
) -> RdCurve {
    let stream = progressive::encode(image, hdr, leaves).expect("progressive::encode");
    let offsets = progressive::layer_end_offsets(&stream).expect("layer_end_offsets");

    let mut points = Vec::with_capacity(4);
    for &end in &offsets {
        let decoded = progressive::decode(&stream[..end])
            .expect("progressive::decode of a layer-boundary prefix");
        let psnr_db = psnr(image, &decoded.image).unwrap_or(f64::INFINITY);
        points.push(RdPoint::from_size(
            end as u64,
            image.width(),
            image.height(),
            psnr_db,
        ));
    }
    RdCurve::new(
        format!("progressive prefixes (lambda={PROGRESSIVE_LAMBDA})"),
        points,
    )
}

#[test]
fn progressive_penalty_on_kodim01() {
    if std::env::var("MARS_RUN_PROGRESSIVE_GATE").is_err() {
        eprintln!("skipping: set MARS_RUN_PROGRESSIVE_GATE=1 to run this gate");
        return;
    }

    let image = kodim01();

    let reference = single_layer_reference_curve(&image);
    println!("single-layer reference curve ({}):", reference.label);
    for p in &reference.points {
        println!("  bpp={:.4}  psnr={:.3}", p.bpp, p.psnr);
    }
    check_convex_and_monotonic(&reference.points, 1.0)
        .expect("single-layer reference curve must be convex/monotonic -- a non-convex curve is a rate-estimation bug, not a result");

    let params = EncodeParams {
        lambda: Some(PROGRESSIVE_LAMBDA),
        ..BASE
    };
    let (hdr, leaves, _evals, stats) = encode_image_rd(&image, &params);
    println!(
        "progressive source encode (lambda={PROGRESSIVE_LAMBDA}): {} leaves, mode histogram {:?}",
        leaves.len(),
        stats.leaf_modes
    );

    let test = progressive_prefix_curve(&image, &hdr, &leaves);
    println!("progressive prefix curve ({}):", test.label);
    for p in &test.points {
        println!("  bpp={:.4}  psnr={:.3}", p.bpp, p.psnr);
    }

    let result = bd_metrics(&reference, &test).expect(
        "BD metrics must be defined -- if this fails with NoOverlap or NonMonotonic, that \
         is itself the Step 19 result and must be reported, not routed around (§A7)",
    );
    println!(
        "\nprogressive penalty (BD-rate of progressive prefixes vs. single-layer streams, \
         reference={}, test={}): {:.2}% over PSNR interval {:.2}-{:.2} dB, bpp interval \
         {:.4}-{:.4}",
        result.reference,
        result.test,
        result.bd_rate_pct,
        result.psnr_interval_db.0,
        result.psnr_interval_db.1,
        result.bpp_interval.0,
        result.bpp_interval.1,
    );
}
