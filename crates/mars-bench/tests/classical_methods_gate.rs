//! `gate-9`'s programmatic checks (Step 9), scoped to what this session could actually
//! run end to end -- see `docs/decisions.md` for the full list of what this narrows
//! relative to the Step 9 brief's "full `standard/` corpus, all six methods, all rates"
//! ask, and why.
//!
//! 1. **Harness sanity (P9.4).** `Exhaustive` run through the new `mars_search` driver,
//!    scored against the Step 8 oracle cache at the oracle's own `default` config, must
//!    reproduce ~100% top-1 recall / ~0 dB regret -- the same sanity check `gate-8`'s own
//!    self-test performs, applied to the new driver instead of `oracle_top1_as_picks`.
//! 2. **`evals/transform` vs. the C reference (5%, per the Step 9 brief's tolerance),**
//!    for `kodim01` at the exact `EncodeParams` a `results/baseline-mars1.jsonl` row
//!    already recorded (`min_size=4, max_size=16, shift=4, bits_alfa=4, bits_beta=7,
//!    max_alfa=1.0, t_rms=8.0`) -- reusing Step 2's already-captured C-binary numbers
//!    rather than rebuilding `reference/mars1` in this session (this sandbox's `xcrun`
//!    cannot compile it -- see `docs/decisions.md`).
//!
//! What this file does **not** check (recorded as open gaps, not silently passed):
//! RD-curve-within-0.2dB against the Mars 1 baseline (would need the decode + PSNR
//! pipeline wired to `mars_search`'s output, not done this session) and the full
//! 24-image sweep for evals/transform (only `kodim01` is checked here).

use std::path::Path;

use mars_bench::oracle::{self, OracleConfig};
use mars_bench::recall::{self, MethodPick};
use mars_bench::store::read_rows;
use mars_codec::encode::EncodeParams;
use mars_core::io::read_raw;
use mars_search::{MethodName, SizedRetrievers};

fn kodim01() -> mars_core::Plane {
    read_raw(
        Path::new("../../corpus/images/kodak-gray/kodim01.raw"),
        768,
        512,
    )
    .expect("kodim01.raw (run `just corpus-gray` first)")
}

fn run(
    image: &mars_core::Plane,
    params: &EncodeParams,
    method: MethodName,
) -> (u64, u64, Vec<MethodPick>) {
    let contracted = mars_codec::encode::build_contracted(image);
    let retrievers = SizedRetrievers::build(
        &contracted,
        image.width() as u32,
        image.height() as u32,
        params.shift,
        params.min_size,
        params.max_size,
        || method.new_retriever(),
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

/// P9.4: the new driver's `Exhaustive`, scored against the real oracle, must land at
/// (effectively) 100% top-1 recall and 0 dB regret -- anything else means the driver's
/// partition or moments disagree with the oracle's own config, a harness bug per the
/// Step 9 brief, not a search-method result.
#[test]
fn exhaustive_scores_near_perfect_recall_against_the_oracle() {
    let image = kodim01();
    let cache_path = Path::new("../../oracle-cache/kodim01__default.bin");
    if !cache_path.exists() {
        eprintln!("skipping: no oracle cache at {cache_path:?} (run `marsbench oracle-build`)");
        return;
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

    let params = EncodeParams {
        min_size: cfg.min_size,
        max_size: cfg.max_size,
        shift: cfg.shift,
        bits_alfa: cfg.bits_alfa,
        bits_beta: cfg.bits_beta,
        max_alfa: cfg.max_alfa,
        t_rms: 1e9, // never split -- code every block at max_size, matching the oracle's own per-size-independent scoring
        zero_threshold: 0,
        lambda: None,
    };

    let (_, _, picks) = run(&image, &params, MethodName::Exhaustive);
    let report = recall::score(&cache, &picks);
    println!(
        "exhaustive vs oracle: matched={} top1={:.4}% top5={:.4}% top32={:.4}% regret={:?}",
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
        "Exhaustive's own top-1 recall against the oracle should be ~100%, got {:.4}%",
        report.top1_rate() * 100.0
    );
    if let Some(stats) = report.regret_stats() {
        assert!(
            stats.mean_db.abs() < 0.01,
            "Exhaustive's own mean regret against the oracle should be ~0 dB, got {}",
            stats.mean_db
        );
    }
}

/// Step 9 exit criterion: each method's `evals/transform` within 5% of the C
/// implementation's own counter -- checked here against `results/baseline-mars1.jsonl`'s
/// already-captured `kodim01` rows at `min_size=4, max_size=16, shift=4, t_rms=8.0`
/// (Step 2's own harness ran the real `reference/mars1` binaries; this sandbox cannot
/// compile them, so those numbers are reused rather than re-derived -- `docs/decisions.md`).
#[test]
fn evals_per_transform_is_within_5pct_of_the_c_reference_on_kodim01() {
    let baseline_path = Path::new("../../results/baseline-mars1.jsonl");
    if !baseline_path.exists() {
        eprintln!("skipping: no {baseline_path:?}");
        return;
    }
    let rows = read_rows(baseline_path).expect("read baseline-mars1.jsonl");

    let mut baseline: std::collections::HashMap<String, Vec<f64>> =
        std::collections::HashMap::new();
    for row in &rows {
        if row.kind != "baseline-mars1" {
            continue;
        }
        let d = &row.data;
        if d.get("image").and_then(|v| v.as_str()) != Some("kodim01") {
            continue;
        }
        let p = &d["params"];
        let is_target_config = p.get("min_size").and_then(|v| v.as_u64()) == Some(4)
            && p.get("max_size").and_then(|v| v.as_u64()) == Some(16)
            && p.get("shift").and_then(|v| v.as_u64()) == Some(4)
            && p.get("bits_alfa").and_then(|v| v.as_u64()) == Some(4)
            && p.get("bits_beta").and_then(|v| v.as_u64()) == Some(7)
            && p.get("t_rms").and_then(|v| v.as_f64()) == Some(8.0);
        if !is_target_config {
            continue;
        }
        let Some(method) = p.get("method").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(ept) = d
            .get("encode")
            .and_then(|e| e.get("evals_per_transform"))
            .and_then(|v| v.as_f64())
        else {
            continue;
        };
        baseline.entry(method.to_string()).or_default().push(ept);
    }
    assert!(
        !baseline.is_empty(),
        "no matching baseline-mars1 rows found for kodim01 at the target config -- \
         baseline-mars1.jsonl schema may have changed"
    );

    let image = kodim01();
    let params = EncodeParams {
        min_size: 4,
        max_size: 16,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda: None,
    };

    let methods = [
        (MethodName::Fisher, "fisher"),
        (MethodName::Hurtgen, "hurtgen"),
        (MethodName::MassCenter, "masscenter"),
        (MethodName::Saupe, "saupe"),
        (MethodName::SaupeFisher, "saupe-fisher"),
        (MethodName::McSaupe, "mc-saupe"),
    ];

    let mut failures = Vec::new();
    for (method, key) in methods {
        let Some(baseline_values) = baseline.get(key) else {
            eprintln!("skipping {key}: no baseline rows for it");
            continue;
        };
        let baseline_mean = baseline_values.iter().sum::<f64>() / baseline_values.len() as f64;

        let (evals, transforms, _) = run(&image, &params, method);
        let got = evals as f64 / transforms as f64;
        let pct_diff = (got - baseline_mean).abs() / baseline_mean * 100.0;
        println!(
            "{key:<14} rust={got:>10.3}  c_baseline={baseline_mean:>10.3} (n={})  diff={pct_diff:.2}%",
            baseline_values.len()
        );
        if pct_diff > 5.0 {
            failures.push(format!(
                "{key}: {pct_diff:.2}% (rust={got:.3}, c={baseline_mean:.3})"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "evals/transform outside the 5% tolerance of the C reference: {failures:#?}"
    );
}
