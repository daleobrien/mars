//! `gate-17`'s programmatic checks (Step 17 -- learned candidate pruning), mirroring
//! `funnel_gate.rs`'s precedent exactly: same corpus/config scope, same harness-sanity +
//! first-real-data-point structure.
//!
//! 1. **Harness sanity.** `Learned` with `survivors = usize::MAX` (no narrowing -- every
//!    domain position is scored but none is dropped) must reproduce `Exhaustive`'s own
//!    near-perfect recall/regret against the oracle, exactly like `funnel_gate.rs`'s own
//!    `FunnelMode::Disabled` check. Anything else means the candidate-emission plumbing
//!    (not the model) is broken.
//! 2. **A real recall/evals/wall-clock data point, in-sample (`kodim01`, the training
//!    image) and held-out (`kodim02`).** `Learned` at `DEFAULT_SURVIVORS = 16` (matching
//!    the R&D plan's own funnel endpoint) scored against the oracle, alongside a
//!    same-run `Funnel` recomputation for a direct, same-machine comparison, and the
//!    already-recorded `saupe-fisher` numbers (`results/classical-methods-sample.jsonl`,
//!    D28) as the cheapest classical baseline. Wall-clock includes feature computation
//!    and MLP inference for `Learned`, not just the surviving Stage-4 affine fits -- the
//!    brief's own explicit instruction that inference cost must be counted, not
//!    hand-waved by the `evals` counter alone.
//!
//! **Scope cut (mirrors D28/D36/D39/D40/D43's precedent, stated here and in
//! `docs/decisions.md`'s Step 17 entry).** `kodim01`/`kodim02` only, `default` oracle
//! config, size 16 only (not size 8, not the full 24-image `standard/` corpus). The full
//! 8-way method comparison, the CLIC/USC-SIPI generalisation test, and a wired-in BD-rate
//! measurement are all out of scope this session -- see `docs/decisions.md`.

use std::path::Path;
use std::time::Instant;

use mars_bench::oracle::{self, OracleCache, OracleConfig};
use mars_bench::recall::{self, MethodPick};
use mars_codec::encode::EncodeParams;
use mars_core::io::read_raw;
use mars_core::Plane;
use mars_search::funnel::{Funnel, FunnelMode};
use mars_search::learned::{self, Learned};
use mars_search::{CandidateRetriever, MethodName, SizedRetrievers};

fn kodim(n: u32) -> Plane {
    read_raw(
        Path::new(&format!("../../corpus/images/kodak-gray/kodim{n:02}.raw")),
        768,
        512,
    )
    .unwrap_or_else(|e| panic!("kodim{n:02}.raw (run `just corpus-gray` first): {e}"))
}

const KODIM01_SHA: &str = "4bcd9402749c018e7a3c5c5c921dfa51ac6e9037bb648310569ecb803be3012d";
const KODIM02_SHA: &str = "4f5bc5300234a605ffdcd2170cb80a439f3409d9502e077e41985e3f74546d5b";

fn default_oracle(n: u32, sha: &str) -> Option<(OracleCache, OracleConfig)> {
    let cache_path_owned = format!("../../oracle-cache/kodim{n:02}__default.bin");
    let cache_path = Path::new(&cache_path_owned);
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
    let cache = oracle::load_and_validate(cache_path, sha, &cfg)
        .unwrap_or_else(|e| panic!("valid oracle cache for kodim{n:02} [default]: {e}"));
    Some((cache, cfg))
}

/// Same never-split `EncodeParams` shape `classical_methods_gate.rs`/`funnel_gate.rs` use
/// against this oracle: every block is coded at `max_size`, matching the oracle's own
/// per-size scoring.
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
        lambda: None,
    }
}

struct RunResult {
    evals: u64,
    transforms: u64,
    picks: Vec<MethodPick>,
    wall_clock: std::time::Duration,
}

fn run_method(
    image: &Plane,
    params: &EncodeParams,
    mut make: impl FnMut() -> Box<dyn CandidateRetriever>,
) -> RunResult {
    let contracted = mars_codec::encode::build_contracted(image);
    let start = Instant::now();
    let retrievers = SizedRetrievers::build(
        &contracted,
        image.width() as u32,
        image.height() as u32,
        params.shift,
        params.min_size,
        params.max_size,
        &mut make,
    );
    let (_, leaves, evals, picks) = mars_search::encode_image(image, params, &retrievers);
    let wall_clock = start.elapsed();
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
    RunResult {
        evals,
        transforms: leaves.len() as u64,
        picks: method_picks,
        wall_clock,
    }
}

fn print_report(label: &str, r: &RunResult, cache: &OracleCache) {
    let report = recall::score(cache, &r.picks);
    let evals_per_transform = r.evals as f64 / r.transforms as f64;
    println!(
        "{label}: matched={} top1={:.4}% top5={:.4}% top32={:.4}% regret={:?} \
         evals/transform={evals_per_transform:.3} wall_clock={:.3?}",
        report.matched,
        report.top1_rate() * 100.0,
        report.top5_rate() * 100.0,
        report.top32_rate() * 100.0,
        report.regret_stats(),
        r.wall_clock,
    );
}

/// P17 harness sanity: `Learned` with unlimited survivors (every domain scored, none
/// dropped) must reproduce `Exhaustive`'s own near-100% top-1 / ~0dB-regret against the
/// real oracle -- confirms the model's candidate-emission plumbing (not its ranking
/// quality) is correct.
#[test]
fn unlimited_survivors_learned_scores_near_perfect_recall_against_the_oracle() {
    let Some((cache, cfg)) = default_oracle(1, KODIM01_SHA) else {
        return;
    };
    let image = kodim(1);
    let params = never_split_params(&cfg);

    let r = run_method(&image, &params, || {
        Box::new(Learned::new(&learned::trained::WEIGHTS, usize::MAX))
    });
    print_report("learned[unlimited] vs oracle (kodim01)", &r, &cache);
    let report = recall::score(&cache, &r.picks);
    assert!(
        report.matched > 0,
        "no blocks matched the oracle at all -- harness bug"
    );
    assert!(
        report.top1_rate() > 0.999,
        "unlimited-survivor Learned's top-1 recall against the oracle should be ~100%, got \
         {:.4}%",
        report.top1_rate() * 100.0
    );
    if let Some(stats) = report.regret_stats() {
        assert!(
            stats.mean_db.abs() < 0.01,
            "unlimited-survivor Learned's mean regret against the oracle should be ~0 dB, got {}",
            stats.mean_db
        );
    }
}

/// P17's real data point: `Learned` at `DEFAULT_SURVIVORS` vs. a same-run `Funnel`
/// recomputation, in-sample on `kodim01` (the training image) and held-out on `kodim02`.
/// See `docs/decisions.md`'s Step 17 entry for the full numbers and the abort-rule
/// decision made from them.
#[test]
fn learned_vs_funnel_recall_evals_and_wall_clock_on_both_images() {
    for (n, sha) in [(1u32, KODIM01_SHA), (2, KODIM02_SHA)] {
        let Some((cache, cfg)) = default_oracle(n, sha) else {
            continue;
        };
        let image = kodim(n);
        let params = never_split_params(&cfg);
        let tag = if n == 1 { "in-sample" } else { "held-out" };

        let learned = run_method(&image, &params, || {
            Box::new(Learned::new(
                &learned::trained::WEIGHTS,
                learned::DEFAULT_SURVIVORS,
            ))
        });
        print_report(
            &format!("learned[k=16] kodim{n:02} ({tag})"),
            &learned,
            &cache,
        );

        let funnel = run_method(&image, &params, || {
            Box::new(Funnel::new(FunnelMode::Scaled))
        });
        print_report(
            &format!("funnel[scaled] kodim{n:02} ({tag})"),
            &funnel,
            &cache,
        );

        let exhaustive = run_method(&image, &params, || MethodName::Exhaustive.new_retriever());
        let exhaustive_evals_per_t = exhaustive.evals as f64 / exhaustive.transforms as f64;
        println!(
            "exhaustive kodim{n:02}: evals/transform={exhaustive_evals_per_t:.3} (ceiling, not a gate)"
        );

        let report = recall::score(&cache, &learned.picks);
        assert!(
            report.matched > 0,
            "no blocks matched the oracle at all on kodim{n:02} -- harness bug"
        );
        let learned_evals_per_t = learned.evals as f64 / learned.transforms as f64;
        assert!(
            learned_evals_per_t < exhaustive_evals_per_t,
            "Learned ({learned_evals_per_t:.3} evals/transform) did not narrow the search at \
             all relative to Exhaustive ({exhaustive_evals_per_t:.3}) on kodim{n:02}"
        );
    }
}
