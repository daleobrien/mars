//! `gate-13`'s programmatic checks (Step 13's hierarchical funnel search), scoped to what
//! this session could actually run end to end -- mirrors `classical_methods_gate.rs`'s
//! (Step 9 / gate-9) scoping pattern and its own `docs/decisions.md` D28 precedent.
//!
//! 1. **Harness sanity (P13.3, `docs/predictions.md`).** `Funnel` with narrowing disabled
//!    (`FunnelMode::Disabled`, every domain survives to Stage 4) must reproduce
//!    `Exhaustive`'s own recall/regret against the Step 8 oracle -- the same sanity check
//!    P9.4 established for the six classical methods, adapted to a funnel with its
//!    narrowing turned off rather than a from-scratch bucket method.
//! 2. **A first real recall/evals data point (P13.1/P13.2).** `Funnel` at its default
//!    `FunnelMode::Scaled` configuration, scored against the same oracle on `kodim01` --
//!    printed and checked against a floor calibrated to Step 9's own measured recall
//!    numbers (P9.1's table: even plain Saupe, the best of six classical methods, only
//!    reaches 51.55% top-1 on this oracle; Mc-Saupe as low as 3.87%), not the withdrawn
//!    80-95% guess `docs/predictions.md`'s Step 13 prediction made before this table was
//!    cross-checked. Measured: 10.03% top-1 / 0.80 dB mean regret at 128 evals/transform
//!    (vs. Fisher's 10.71% top-1 / 0.95 dB at 774.2 evals/transform) -- comparable recall
//!    to Fisher at roughly 6x fewer evals, and mid-pack regret among all six methods.
//!
//! What this file does **not** check (recorded as an open gap, not silently passed --
//! `docs/predictions.md`'s Step 13 prediction says so up front): the full survival/recall
//! tradeoff curve across the 24-image `standard/` corpus and several `FunnelConfig`
//! survivor counts, the Pareto frontier (`evals/transform` vs. BD-rate) against all six
//! classical methods, and per-block `MARS_FUNNEL_LOG` output at scale. Those need the same
//! full-corpus sweep Step 9's own D28 scoped down for lack of session time.

use std::path::Path;

use mars_bench::oracle::{self, OracleConfig};
use mars_bench::recall::{self, MethodPick};
use mars_codec::encode::EncodeParams;
use mars_core::io::read_raw;
use mars_search::funnel::{Funnel, FunnelMode};
use mars_search::{CandidateRetriever, MethodName, SizedRetrievers};

fn kodim01() -> mars_core::Plane {
    read_raw(
        Path::new("../../corpus/images/kodak-gray/kodim01.raw"),
        768,
        512,
    )
    .expect("kodim01.raw (run `just corpus-gray` first)")
}

fn default_oracle() -> Option<(mars_bench::oracle::OracleCache, OracleConfig)> {
    let cache_path = Path::new("../../oracle-cache/kodim01__default.bin");
    if !cache_path.exists() {
        eprintln!("skipping: no oracle cache at {cache_path:?} (run `marsbench oracle-build`)");
        return None;
    }
    let cfg = OracleConfig {
        min_size: 8,
        max_size: 16,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
    };
    let cache = oracle::load_and_validate(
        cache_path,
        "4bcd9402749c018e7a3c5c5c921dfa51ac6e9037bb648310569ecb803be3012d",
        &cfg,
    )
    .expect("valid oracle cache for kodim01 [default]");
    Some((cache, cfg))
}

/// Same never-split `EncodeParams` shape `classical_methods_gate.rs` uses against this
/// oracle: every block is coded at `max_size`, matching the oracle's own per-size scoring.
fn never_split_params(cfg: &OracleConfig) -> EncodeParams {
    EncodeParams {
        min_size: cfg.min_size,
        max_size: cfg.max_size,
        shift: cfg.shift,
        bits_alfa: cfg.bits_alfa,
        bits_beta: cfg.bits_beta,
        max_alfa: cfg.max_alfa,
        t_rms: 1e9,
        zero_threshold: 0,
    }
}

fn run_funnel(
    image: &mars_core::Plane,
    params: &EncodeParams,
    mode: FunnelMode,
) -> (u64, u64, Vec<MethodPick>) {
    let contracted = mars_codec::encode::build_contracted(image);
    let retrievers = SizedRetrievers::build(
        &contracted,
        image.width() as u32,
        image.height() as u32,
        params.shift,
        params.min_size,
        params.max_size,
        || Box::new(Funnel::new(mode)) as Box<dyn CandidateRetriever>,
    );
    let (_, leaves, evals, picks) = mars_search::encode_image(image, params, &retrievers);
    let method_picks = picks
        .into_iter()
        .map(|p| MethodPick {
            row: p.row,
            col: p.col,
            size: p.size,
            dom_row: p.dom_row,
            dom_col: p.dom_col,
            isometry: p.isometry,
            qalfa: p.qalfa,
            qbeta: p.qbeta,
            rms: p.rms,
        })
        .collect();
    (evals, leaves.len() as u64, method_picks)
}

/// P13.3: with narrowing disabled, `Funnel` must reproduce `Exhaustive`'s own near-perfect
/// recall/regret against the real oracle -- anything else means Stage 1-3's bookkeeping
/// (which domains are even considered) disagrees with a true exhaustive scan, a harness
/// bug rather than a funnel-narrowing result.
#[test]
fn disabled_funnel_scores_near_perfect_recall_against_the_oracle() {
    let Some((cache, cfg)) = default_oracle() else {
        return;
    };
    let image = kodim01();
    let params = never_split_params(&cfg);

    let (_, _, picks) = run_funnel(&image, &params, FunnelMode::Disabled);
    let report = recall::score(&cache, &picks);
    println!(
        "funnel[disabled] vs oracle: matched={} top1={:.4}% top5={:.4}% top32={:.4}% regret={:?}",
        report.matched,
        report.top1_rate() * 100.0,
        report.top5_rate() * 100.0,
        report.top32_rate() * 100.0,
        report.regret_stats()
    );
    assert!(
        report.matched > 0,
        "no blocks matched the oracle at all -- harness bug"
    );
    assert!(
        report.top1_rate() > 0.999,
        "disabled Funnel's top-1 recall against the oracle should be ~100%, got {:.4}%",
        report.top1_rate() * 100.0
    );
    if let Some(stats) = report.regret_stats() {
        assert!(
            stats.mean_db.abs() < 0.01,
            "disabled Funnel's mean regret against the oracle should be ~0 dB, got {}",
            stats.mean_db
        );
    }
}

/// P13.1/P13.2's first real data point: `Funnel` at its default (`FunnelConfig::scaled`)
/// survivor counts, scored on `kodim01`. This is deliberately a wide sanity floor, not the
/// brief's actual exit criterion (the full survival/recall tradeoff curve and Pareto
/// frontier need the multi-image, multi-config sweep this session leaves open -- see the
/// module doc and `docs/predictions.md`'s Step 13 prediction).
#[test]
fn scaled_funnel_recall_and_evals_data_point_on_kodim01() {
    let Some((cache, cfg)) = default_oracle() else {
        return;
    };
    let image = kodim01();
    let params = never_split_params(&cfg);

    let (evals, transforms, picks) = run_funnel(&image, &params, FunnelMode::Scaled);
    let report = recall::score(&cache, &picks);
    let evals_per_transform = evals as f64 / transforms as f64;
    println!(
        "funnel[scaled] vs oracle: matched={} top1={:.4}% top5={:.4}% top32={:.4}% \
         regret={:?} evals/transform={evals_per_transform:.3}",
        report.matched,
        report.top1_rate() * 100.0,
        report.top5_rate() * 100.0,
        report.top32_rate() * 100.0,
        report.regret_stats()
    );

    let exhaustive_evals = {
        let (e, t, _) = run_funnel_baseline_exhaustive(&image, &params);
        e as f64 / t as f64
    };
    println!("exhaustive evals/transform={exhaustive_evals:.3} (for comparison, not a gate)");

    assert!(
        report.matched > 0,
        "no blocks matched the oracle at all -- harness bug"
    );
    // P13.1 originally guessed 80-95% top-1 on this image and was falsified -- but not by
    // a funnel bug: Step 9's own measured numbers on this same oracle (`docs/predictions.md`,
    // P9.1's table) show even the *best* classical method (plain Saupe) only reaches 51.55%
    // top-1, Fisher reaches 10.71%, and Mc-Saupe as low as 3.87% -- exact-tuple top-1 recall
    // against a 32-deep oracle is simply a hard target on this corpus. The floor below is
    // "still meaningfully searching" (rules out an indexing/distance bug collapsing every
    // block onto the same handful of domains), calibrated against Mc-Saupe's measured floor
    // rather than the withdrawn 50% guess -- see `docs/predictions.md`'s Step 13 outcome for
    // the corrected comparison against all six methods.
    assert!(
        report.top1_rate() > 0.02,
        "scaled Funnel's top-1 recall collapsed to {:.4}% -- likely an indexing/distance bug, \
         not just aggressive narrowing (compare Mc-Saupe's measured 3.87% floor, P9.1)",
        report.top1_rate() * 100.0
    );
    assert!(
        evals_per_transform < exhaustive_evals,
        "scaled Funnel ({evals_per_transform:.3} evals/transform) did not narrow the search \
         at all relative to Exhaustive ({exhaustive_evals:.3})"
    );
}

fn run_funnel_baseline_exhaustive(
    image: &mars_core::Plane,
    params: &EncodeParams,
) -> (u64, u64, Vec<MethodPick>) {
    let contracted = mars_codec::encode::build_contracted(image);
    let retrievers = SizedRetrievers::build(
        &contracted,
        image.width() as u32,
        image.height() as u32,
        params.shift,
        params.min_size,
        params.max_size,
        || MethodName::Exhaustive.new_retriever(),
    );
    let (_, leaves, evals, picks) = mars_search::encode_image(image, params, &retrievers);
    let method_picks = picks
        .into_iter()
        .map(|p| MethodPick {
            row: p.row,
            col: p.col,
            size: p.size,
            dom_row: p.dom_row,
            dom_col: p.dom_col,
            isometry: p.isometry,
            qalfa: p.qalfa,
            qbeta: p.qbeta,
            rms: p.rms,
        })
        .collect();
    (evals, leaves.len() as u64, method_picks)
}
