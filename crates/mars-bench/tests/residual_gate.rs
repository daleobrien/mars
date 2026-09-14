//! `gate-15`'s programmatic checks (Step 15 -- residual mode), mirroring `rd_gate.rs`'s
//! precedent exactly: same corpus scope (`kodim01`/`kodim02`), same λ grid, same
//! convexity/monotonicity discipline, but comparing Step 15's full four-mode competition
//! against a same-codebase "Step 14 equivalent" curve (`mars_bench::mode_gate::STEP14_MODES`
//! -- modes 0/2 only, via `mars_codec::encode::encode_image_rd_with_modes`'s mode mask)
//! rather than against a different git revision (`mars_bench::mode_gate`'s own doc explains
//! why this is the fairer A/B).
//!
//! **Scope cut, stated up front (recorded again in `docs/decisions.md`).** Step 15's
//! bottom-up walk evaluates four leaf-mode candidates (flat, affine, fractal, fractal +
//! residual) at every quadtree node instead of Step 14's two (flat, fractal), so it is
//! strictly more expensive per encode than the already-slow Step 14 walk `rd_gate.rs`
//! documents taking ~1 minute per `kodim01` encode. This gate therefore keeps `rd_gate.rs`'s
//! exact scope: `kodim01`/`kodim02` only, 4-point λ grid.
//!
//! **What this checks:**
//! 1. **BD-rate of the Step 15 (4-mode) curve vs. the Step 14-equivalent (2-mode) curve**,
//!    both measured at `.mars` v0 (entropy-coded) bpp, both using the identical rate-
//!    estimation snapshot machinery -- the only variable is which modes `J` could pick.
//! 2. **Convexity/monotonicity of the Step 15 λ sweep** -- the same diagnostic `rd_gate.rs`
//!    uses, for the same reason (a non-convex curve means a rate-estimation bug, not a
//!    result to report).
//! 3. **The mode-usage histogram** -- printed for both images (this is the brief's own
//!    "more scientifically interesting than the BD-rate number" header finding), with a
//!    sanity assertion that every one of the four leaf modes is actually reachable
//!    somewhere in the sweep (not a hard requirement of the format, but a harness-sanity
//!    check: if a mode is *never* picked across two images and four λ points, the far more
//!    likely explanation is a wiring bug in its `J` pricing than a genuine, total absence
//!    of any block that benefits from it).

use mars_bench::mode_gate::{format_mode_histogram, mode_curve, STEP14_MODES, STEP15_MODES};
use mars_bench::rd_opt::check_convex_and_monotonic;
use mars_codec::encode::{EncodeParams, ModeStats};
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

/// Identical to `rd_gate.rs`'s own `LAMBDA_GRID` -- same quality-knob range, so the two
/// steps' BD-rate numbers are read on the same axis.
const LAMBDA_GRID: [f64; 4] = [50.0, 200.0, 800.0, 3200.0];

const CONVEXITY_SLACK_DB_PER_BPP: f64 = 1.0;

/// Calibrated the same way `gate-14`'s own `BD_RATE_TARGET_PCT` was (`docs/decisions.md`'s
/// D36 precedent: set once, to what this session actually measured, not to an aspirational
/// number -- the brief states no explicit target for Step 15, unlike Step 14's 10%). Set
/// after this session's own first real measurement; see `docs/decisions.md`'s Step 15 entry
/// for the exact number and margin.
const BD_RATE_TARGET_PCT: f64 = -1.0;

fn kodim(n: u32) -> Plane {
    read_raw(
        std::path::Path::new(&format!(
            "../../corpus/images/kodak-gray/kodim{n:02}.raw"
        )),
        768,
        512,
    )
    .unwrap_or_else(|e| panic!("kodim{n:02}.raw (run `just corpus-gray` first): {e}"))
}

/// This gate's encodes are even more expensive than `gate-14`'s (four mode candidates per
/// node instead of two): unguarded, it would add real minutes to every plain
/// `cargo test -p mars-bench --release`. Opt in with `MARS_RUN_RESIDUAL_GATE=1`
/// (`just gate-15` sets this), matching `rd_gate.rs`'s own convention.
#[test]
fn four_mode_competition_beats_the_step14_equivalent_and_covers_every_mode() {
    if std::env::var("MARS_RUN_RESIDUAL_GATE").as_deref() != Ok("1") {
        eprintln!(
            "skipping: set MARS_RUN_RESIDUAL_GATE=1 to run gate-15's multi-minute RD sweep \
             (`just gate-15` does this automatically)"
        );
        return;
    }

    let mut bd_rates = Vec::new();
    let mut corpus_stats = ModeStats::default();

    for &n in &[1u32, 2] {
        let image = kodim(n);
        let label = format!("kodim{n:02}");

        let (reference, ref_samples, ref_stats) =
            mode_curve(format!("{label} step14-equivalent (modes 0/2)"), &image, &BASE, &LAMBDA_GRID, STEP14_MODES);
        let (test, test_samples, test_stats) =
            mode_curve(format!("{label} step15 (modes 0-3)"), &image, &BASE, &LAMBDA_GRID, STEP15_MODES);

        eprintln!("-- {label} --");
        for s in &ref_samples {
            eprintln!(
                "  step14-equiv: bpp={:.4} psnr={:.3} evals={} leaves={}",
                s.point.bpp, s.point.psnr, s.evals, s.leaves
            );
        }
        for s in &test_samples {
            eprintln!(
                "  step15:       bpp={:.4} psnr={:.3} evals={} leaves={}",
                s.point.bpp, s.point.psnr, s.evals, s.leaves
            );
        }
        eprintln!("  step14-equiv mode histogram: {}", format_mode_histogram(&ref_stats));
        eprintln!("  step15       mode histogram: {}", format_mode_histogram(&test_stats));

        for i in 0..4 {
            corpus_stats.leaf_modes[i] += test_stats.leaf_modes[i];
        }
        corpus_stats.split_decisions += test_stats.split_decisions;
        corpus_stats.leaf_decisions += test_stats.leaf_decisions;

        check_convex_and_monotonic(&test.points, CONVEXITY_SLACK_DB_PER_BPP).unwrap_or_else(|e| {
            panic!("{label}: Step 15 lambda sweep failed the convexity/monotonicity check: {e}")
        });

        let bd = mars_bench::bdrate::bd_metrics(&reference, &test)
            .unwrap_or_else(|e| panic!("{label}: BD-rate computation failed: {e}"));
        eprintln!(
            "  BD-rate (step15 vs step14-equivalent): {:.2}%  BD-PSNR: {:.3} dB  (bpp overlap {:?})",
            bd.bd_rate_pct, bd.bd_psnr_db, bd.bpp_interval
        );
        bd_rates.push((label, bd.bd_rate_pct));
    }

    let mean_bd_rate: f64 = bd_rates.iter().map(|(_, r)| *r).sum::<f64>() / bd_rates.len() as f64;
    eprintln!(
        "mean BD-rate across {} image(s): {mean_bd_rate:.2}% (gate floor <= {BD_RATE_TARGET_PCT:.1}%)",
        bd_rates.len()
    );
    eprintln!("corpus-wide (kodim01+kodim02) mode histogram: {}", format_mode_histogram(&corpus_stats));

    assert!(
        mean_bd_rate <= BD_RATE_TARGET_PCT,
        "mean BD-rate {mean_bd_rate:.2}% does not clear this gate's calibrated floor of \
         {BD_RATE_TARGET_PCT:.1}% (per-image: {bd_rates:?}) -- adding modes under fair `J` \
         competition must not make the codec worse; a regression here is a bug"
    );

    for (mode, count) in corpus_stats.leaf_modes.iter().enumerate() {
        assert!(
            *count > 0,
            "mode {mode} was never picked anywhere in the sweep -- the far more likely \
             explanation is a wiring bug in that mode's `J` pricing than a genuine total \
             absence of any block that benefits from it (histogram: {corpus_stats:?})"
        );
    }
}
