//! `gate-16`'s programmatic checks (Step 16 -- adaptive partitioning), mirroring
//! `residual_gate.rs`'s precedent exactly: same corpus scope (`kodim01`/`kodim02`), same λ
//! grid, same convexity/monotonicity discipline, but comparing a same-codebase
//! `adaptive_density: true` curve against `adaptive_density: false` (Step 15's own
//! partition/density behaviour, unchanged) via
//! `mars_codec::encode::encode_image_rd_with_modes_and_density`'s new flag -- not a diff
//! against a separate git revision (`mars_bench::density_gate`'s own doc, and
//! `docs/decisions.md`'s D40, explain why this same-codebase A/B is the fairer comparison).
//!
//! **Scope cut, stated up front (recorded again in `docs/decisions.md`).** `kodim01`/
//! `kodim02` only, the same 4-point λ grid every prior RD gate in this project uses
//! (D28/D36/D39/D40's precedent), not the full 24-image `standard/` corpus, for the same
//! session-time reasons.
//!
//! **What this checks, and what was measured (`docs/decisions.md`'s D43 has the full
//! writeup):**
//! 1. **BD-rate of the adaptive-density curve vs. the fixed-density curve**, both measured
//!    at `.mars` v0 bpp, both using the identical rate-estimation snapshot machinery and
//!    identical mode mask (`[true; 4]`) -- the only variable is whether `walk_rd` computes
//!    a per-block domain-search stride from local pixel-domain RMS instead of always using
//!    the run's fixed `params.shift`. **Measured: mean -6.82%** (kodim01 -6.75%, kodim02
//!    -6.88%) -- a real, clean improvement, not a shortfall or a regression like the
//!    previous three gates (`gate-13`/D36, `gate-14`/D39, `gate-15`/D40) needed their bars
//!    calibrated around.
//! 2. **Convexity/monotonicity of the adaptive-density λ sweep** -- passed cleanly on both
//!    images.
//! 3. **Encode-time cost accounting** -- summed wall-clock encode time (this process, this
//!    run, matched thread count) for the adaptive arm vs. the fixed arm, reported alongside
//!    the BD-rate number per the brief's own explicit instruction not to quote the gain
//!    alone. **Measured: kodim01 1.63x, kodim02 1.09x, overall 1.42x** -- inside P16.3's
//!    predicted 1.3-2.5x band, well under the brief's own "4x time for 3% gain" cautionary
//!    framing. Not a `benchmark-protocol`-grade throughput claim (no A/B interleaving, no
//!    median-of-N, no anchor codec) -- see `mars_bench::density_gate`'s own doc for why that
//!    rigour is not attempted here.
//! 4. **Partition statistics** -- leaves per size (a proxy for quadtree depth) and, within
//!    each size bucket, the mean local pixel-domain RMS of the leaves at that size, printed
//!    for both arms on both images. At the sweep's highest lambda (coarsest partition),
//!    both arms land at essentially the same size distribution (~1540-1550 leaves, mean
//!    ~16px) -- the density knob changes *which* domain gets matched, not the block-size
//!    geometry itself, which is exactly the intended separation of concerns (`walk_rd`'s
//!    own leaf-vs-split decision is unchanged by this step).

use mars_bench::density_gate::{density_curve, format_partition_stats, partition_stats};
use mars_bench::rd_opt::check_convex_and_monotonic;
use mars_codec::encode::EncodeParams;
use mars_core::io::read_raw;
use mars_core::Plane;

/// §8's 1998 defaults, matching every other RD gate's `BASE` in this project.
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

/// Identical to `rd_gate.rs`/`residual_gate.rs`'s own `LAMBDA_GRID`.
const LAMBDA_GRID: [f64; 4] = [50.0, 200.0, 800.0, 3200.0];

const CONVEXITY_SLACK_DB_PER_BPP: f64 = 1.0;

/// **Unlike `gate-14`/`gate-15`'s own bars, this one is calibrated to a genuine, clean
/// improvement, not a shortfall or a regression** -- see `docs/decisions.md`'s D43 for the
/// full writeup and why this breaks rather than extends the "three consecutive gates
/// calibrated to a shortfall/regression" pattern D40 flagged as a kill-criteria audit
/// trigger. Measured mean BD-rate (adaptive vs. fixed density, kodim01/kodim02, this
/// session): **-6.82%** (kodim01 -6.75%, kodim02 -6.88%). This ceiling is set at **-3.0%**
/// -- an improvement *floor*, not merely an upper bound on regression -- with real margin
/// (~3.75 points) below the measured worst case, mirroring `gate-14`'s own `D39` precedent
/// of calibrating a brand-new gate's bar with margin below what was actually measured.
const BD_RATE_CEILING_PCT: f64 = -3.0;

/// A same-run, same-thread-count wall-clock ceiling on how much more expensive the
/// adaptive arm may be than the fixed arm. Measured this session: kodim01 1.63x, kodim02
/// 1.09x, overall (summed across the sweep) **1.42x** -- inside P16.3's predicted 1.3-2.5x
/// band. Set at 3.0x, comfortably above the measured 1.63x worst case, as a real
/// regression check (catches a future change that makes the density search
/// pathologically expensive) rather than a formal `benchmark-protocol` throughput claim.
const MAX_ENCODE_TIME_RATIO: f64 = 3.0;

fn kodim(n: u32) -> Plane {
    read_raw(
        std::path::Path::new(&format!("../../corpus/images/kodak-gray/kodim{n:02}.raw")),
        768,
        512,
    )
    .unwrap_or_else(|e| panic!("kodim{n:02}.raw (run `just corpus-gray` first): {e}"))
}

/// This gate's fixed-density arm alone is already as expensive as `gate-15`'s own sweep;
/// the adaptive arm adds a second full sweep whose denser blocks cost more evals still.
/// Opt in with `MARS_RUN_DENSITY_GATE=1` (`just gate-16` does this automatically),
/// matching `rd_gate.rs`/`residual_gate.rs`'s own convention.
#[test]
fn adaptive_density_stays_within_the_regression_ceiling_and_reports_cost_and_partition_stats() {
    if std::env::var("MARS_RUN_DENSITY_GATE").as_deref() != Ok("1") {
        eprintln!(
            "skipping: set MARS_RUN_DENSITY_GATE=1 to run gate-16's multi-minute RD sweep \
             (`just gate-16` does this automatically)"
        );
        return;
    }

    let mut bd_rates = Vec::new();
    let mut total_fixed_secs = 0.0;
    let mut total_adaptive_secs = 0.0;

    for &n in &[1u32, 2] {
        let image = kodim(n);
        let label = format!("kodim{n:02}");

        let (fixed_curve, fixed_samples, fixed_stats, fixed_secs) = density_curve(
            format!("{label} fixed-density"),
            &image,
            &BASE,
            &LAMBDA_GRID,
            false,
        );
        let (adaptive_curve, adaptive_samples, adaptive_stats, adaptive_secs) = density_curve(
            format!("{label} adaptive-density"),
            &image,
            &BASE,
            &LAMBDA_GRID,
            true,
        );

        eprintln!("-- {label} --");
        for s in &fixed_samples {
            eprintln!(
                "  fixed:    bpp={:.4} psnr={:.3} evals={} leaves={} encode_secs={:.2}",
                s.sample.point.bpp,
                s.sample.point.psnr,
                s.sample.evals,
                s.sample.leaves,
                s.encode_secs
            );
        }
        for s in &adaptive_samples {
            eprintln!(
                "  adaptive: bpp={:.4} psnr={:.3} evals={} leaves={} encode_secs={:.2}",
                s.sample.point.bpp,
                s.sample.point.psnr,
                s.sample.evals,
                s.sample.leaves,
                s.encode_secs
            );
        }

        // Partition statistics (this step's own exit criterion): leaves per size, mean
        // local RMS per size bucket, for both arms -- at the last (highest) lambda point,
        // where the partition is coarsest and the density-routing effect should be most
        // visible in the size distribution.
        let last_fixed = &fixed_samples.last().unwrap().leaves;
        let last_adaptive = &adaptive_samples.last().unwrap().leaves;
        eprintln!(
            "  partition (fixed,    lambda={}): {}",
            LAMBDA_GRID.last().unwrap(),
            format_partition_stats(&partition_stats(&image, last_fixed))
        );
        eprintln!(
            "  partition (adaptive, lambda={}): {}",
            LAMBDA_GRID.last().unwrap(),
            format_partition_stats(&partition_stats(&image, last_adaptive))
        );

        eprintln!(
            "  encode time: fixed={fixed_secs:.2}s adaptive={adaptive_secs:.2}s ratio={:.2}x",
            adaptive_secs / fixed_secs.max(1e-9)
        );
        eprintln!(
            "  mode histogram (fixed):    {}",
            mars_bench::mode_gate::format_mode_histogram(&fixed_stats)
        );
        eprintln!(
            "  mode histogram (adaptive): {}",
            mars_bench::mode_gate::format_mode_histogram(&adaptive_stats)
        );

        total_fixed_secs += fixed_secs;
        total_adaptive_secs += adaptive_secs;

        check_convex_and_monotonic(&adaptive_curve.points, CONVEXITY_SLACK_DB_PER_BPP).unwrap_or_else(|e| {
            panic!("{label}: adaptive-density lambda sweep failed the convexity/monotonicity check: {e}")
        });

        let bd = mars_bench::bdrate::bd_metrics(&fixed_curve, &adaptive_curve)
            .unwrap_or_else(|e| panic!("{label}: BD-rate computation failed: {e}"));
        eprintln!(
            "  BD-rate (adaptive vs fixed density): {:.2}%  BD-PSNR: {:.3} dB  (bpp overlap {:?})",
            bd.bd_rate_pct, bd.bd_psnr_db, bd.bpp_interval
        );
        bd_rates.push((label, bd.bd_rate_pct));
    }

    let mean_bd_rate: f64 = bd_rates.iter().map(|(_, r)| *r).sum::<f64>() / bd_rates.len() as f64;
    let overall_time_ratio = total_adaptive_secs / total_fixed_secs.max(1e-9);
    eprintln!(
        "mean BD-rate across {} image(s): {mean_bd_rate:.2}% (negative = adaptive density \
         needs fewer bits than fixed density; ceiling <= {BD_RATE_CEILING_PCT:.1}%); overall \
         encode-time ratio (adaptive/fixed, summed across the sweep): {overall_time_ratio:.2}x \
         (ceiling <= {MAX_ENCODE_TIME_RATIO:.1}x) -- see docs/decisions.md's D43 for the full \
         cost-accounting writeup this exit criterion requires",
        bd_rates.len()
    );

    assert!(
        mean_bd_rate <= BD_RATE_CEILING_PCT,
        "mean BD-rate {mean_bd_rate:.2}% does not clear this gate's improvement floor of \
         {BD_RATE_CEILING_PCT:.1}% (a more negative number is a bigger improvement; \
         per-image: {bd_rates:?})"
    );
    assert!(
        overall_time_ratio <= MAX_ENCODE_TIME_RATIO,
        "adaptive-density encode time is {overall_time_ratio:.2}x fixed-density, exceeding \
         this gate's {MAX_ENCODE_TIME_RATIO:.1}x ceiling"
    );
}
