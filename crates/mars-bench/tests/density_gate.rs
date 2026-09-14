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
//! **CONTRACT-CHANGE, `docs/decisions.md` D48.** Step 16 originally had a *denser*
//! stride branch (high-RMS blocks searched at half `params.shift`) alongside the sparser
//! one below, and this gate originally measured mean BD-rate **-6.82%** from that denser
//! branch. D48 found and root-caused a real correctness bug: a denser stride can produce
//! domain positions that are not multiples of `hdr.shift`, and `mars_format`'s
//! domain-position encoding (`row_units = leaf.dom_row / hdr.shift`, per Mars 1's pinned
//! §4.3 formula) silently truncates such positions on write -- corrupting the decode.
//! This was invisible to this gate because it decodes the search's in-memory `Leaf`s
//! directly (see `sample_with_density`'s own doc), never round-tripping through the real
//! `.mars` bitstream, so the corruption never showed up in *this* file's own numbers; it
//! was only caught once `encmars-decmars-cli-plan.md`'s CLI-A work exercised the real
//! bitstream and measured a multi-dB PSNR collapse. D48's fix removed the denser branch
//! entirely (format-unsafe by construction, not merely disabled) and kept only the
//! sparser (doubling) branch, which is format-safe. **What this checks now:**
//! 1. **BD-rate of the adaptive-density curve vs. the fixed-density curve**, both measured
//!    at `.mars` v0 bpp, both using the identical rate-estimation snapshot machinery and
//!    identical mode mask (`[true; 4]`) -- the only variable is whether low-RMS blocks
//!    search at double `params.shift` instead of the run's fixed stride. **Re-measured
//!    after D48's fix: mean -0.14%** (kodim01 -0.03%, kodim02 -0.25%) -- essentially
//!    neutral, not the -6.82% improvement the removed branch produced. The surviving
//!    mechanism's real benefit is evals/wall-clock (see 3 below), not BD-rate: a near-flat
//!    block's winning mode is almost always mode 0 (flat) regardless of which domain
//!    candidates were considered, so halving the candidate count rarely changes the final
//!    leaf, only how many evals it took to get there.
//! 2. **Convexity/monotonicity of the adaptive-density λ sweep** -- passed cleanly on both
//!    images.
//! 3. **Encode-time cost accounting** -- summed wall-clock encode time (this process, this
//!    run, matched thread count) for the adaptive arm vs. the fixed arm, reported alongside
//!    the BD-rate number per the brief's own explicit instruction not to quote the gain
//!    alone. **Re-measured after D48's fix: kodim01 0.97x, kodim02 0.74x, overall 0.87x**
//!    -- genuinely *faster*, not the 1.42x slower the removed denser branch cost (fewer
//!    domain candidates to try on low-RMS blocks, no denser-branch blocks paying extra).
//!    Not a `benchmark-protocol`-grade throughput claim (no A/B interleaving, no
//!    median-of-N, no anchor codec) -- see `mars_bench::density_gate`'s own doc for why that
//!    rigour is not attempted here.
//! 4. **Partition statistics** -- leaves per size (a proxy for quadtree depth) and, within
//!    each size bucket, the mean local pixel-domain RMS of the leaves at that size, printed
//!    for both arms on both images. Post-D48, the highest-lambda partition is now
//!    identical leaf-for-leaf between arms on kodim01 (1548 vs 1548) and nearly so on
//!    kodim02 (1542 vs 1542) -- consistent with "mode 0 wins regardless of stride" above.

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

/// **D48 CONTRACT-CHANGE.** Before D48, this bar was an improvement *floor* (-3.0%),
/// calibrated under the since-removed, format-unsafe denser branch's measured -6.82%. The
/// surviving (sparsify-only) mechanism is essentially BD-rate-neutral by construction (see
/// this file's module doc) -- re-measured mean **-0.14%** (kodim01 -0.03%, kodim02
/// -0.25%). This bar is now a **regression ceiling**, not an improvement floor: it exists
/// to catch a future change that makes the sparsify branch actively worse, not to claim a
/// quality win. Set at +2.0%, comfortably above measurement noise around the ~0% true
/// value.
const BD_RATE_CEILING_PCT: f64 = 2.0;

/// A same-run, same-thread-count wall-clock ceiling on how much more expensive the
/// adaptive arm may be than the fixed arm. Pre-D48 (denser branch included) this measured
/// 1.42x *slower*; post-D48 (sparsify-only) it measures **0.87x -- genuinely faster**
/// (kodim01 0.97x, kodim02 0.74x), since there is no denser-search cost left to pay. Set
/// at 1.2x: real margin above the measured 0.97x worst case, while still catching a future
/// regression that makes the surviving branch slower than the fixed path.
const MAX_ENCODE_TIME_RATIO: f64 = 1.2;

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
        "mean BD-rate across {} image(s): {mean_bd_rate:.2}% (post-D48, this is a regression \
         ceiling, not an improvement floor -- the surviving sparsify-only mechanism is \
         expected to be roughly neutral on BD-rate; ceiling <= {BD_RATE_CEILING_PCT:.1}%); \
         overall encode-time ratio (adaptive/fixed, summed across the sweep): \
         {overall_time_ratio:.2}x (ceiling <= {MAX_ENCODE_TIME_RATIO:.1}x) -- see \
         docs/decisions.md's D48 for the full corrected writeup",
        bd_rates.len()
    );

    assert!(
        mean_bd_rate <= BD_RATE_CEILING_PCT,
        "mean BD-rate {mean_bd_rate:.2}% exceeds this gate's regression ceiling of \
         {BD_RATE_CEILING_PCT:.1}% (per-image: {bd_rates:?})"
    );
    assert!(
        overall_time_ratio <= MAX_ENCODE_TIME_RATIO,
        "adaptive-density encode time is {overall_time_ratio:.2}x fixed-density, exceeding \
         this gate's {MAX_ENCODE_TIME_RATIO:.1}x ceiling"
    );
}
