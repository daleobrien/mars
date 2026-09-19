//! Step 22/O7 acceptance: same-run modes0/2, fixed8, and adaptive RD curves.
//! Explicit invocation requires the corpus; routine tests never run the Kodak sweep.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use mars_bench::bdrate::{bd_metrics, RdCurve, RdPoint};
use mars_bench::mode_gate::{STEP14_MODES, STEP15_MODES};
use mars_bench::provenance::{sha256_hex, Provenance};
use mars_bench::rd_opt::check_convex_and_monotonic;
use mars_codec::encode::{
    encode_image_with_options, EncodeOptions, EncodeParams, ResidualQuantisation,
};
use mars_codec::ifs::decode_iterative;
use mars_codec::mars_format;
use mars_codec::quant::ResidualQstep;
use mars_core::{io::read_raw, metrics::psnr, Plane};
use serde_json::{json, Value};

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
const LAMBDAS: [f64; 4] = [50.0, 200.0, 800.0, 3200.0];
const HISTORICAL_REGRESSION_PCT: f64 = 2.05;
const CEILING_PCT: f64 = 1.025;
const ITERATIONS: u32 = 10;
const PREDICTION: &str = "Before measuring: adaptive sqrt(6*lambda/ln(2)) should reduce the fixed8 regression, but the high-rate approximation may not clear the +1.025% mean ceiling. No fitted constants or guaranteed improvement.";
const ARMS: [&str; 3] = [
    "baseline-modes02-fixed8",
    "all-modes-fixed8",
    "all-modes-adaptive",
];

fn options(arm: usize) -> EncodeOptions {
    EncodeOptions {
        allowed_modes: if arm == 0 { STEP14_MODES } else { STEP15_MODES },
        residual_quantisation: if arm == 2 {
            ResidualQuantisation::LambdaAdaptive
        } else {
            ResidualQuantisation::Fixed(ResidualQstep::LEGACY)
        },
        ..EncodeOptions::default()
    }
}

fn accepts(mean: f64) -> bool {
    mean.is_finite() && mean <= CEILING_PCT
}

fn command(root: &Path, program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()?;
    ensure!(
        output.status.success(),
        "{program} {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn row(file: &mut File, value: Value) -> Result<()> {
    serde_json::to_writer(&mut *file, &value)?;
    writeln!(file)?;
    // Preserve completed points and the pre-measurement prediction on timeout/failure.
    file.flush()?;
    file.sync_data()?;
    Ok(())
}

fn corpus(root: &Path) -> Result<Vec<(String, Plane, String)>> {
    [1, 2]
        .into_iter()
        .map(|n| {
            let label = format!("kodim{n:02}");
            let path = root.join(format!("corpus/images/kodak-gray/{label}.raw"));
            let bytes = fs::read(&path).with_context(|| {
                format!(
                    "required corpus missing: {}; run `just corpus-gray` first",
                    path.display()
                )
            })?;
            ensure!(
                bytes.len() == 768 * 512,
                "{}: wrong raw image length",
                path.display()
            );
            let image = read_raw(&path, 768, 512)?;
            Ok((label, image, sha256_hex(&bytes)))
        })
        .collect()
}

fn measure(image: &Plane, lambda: f64, arm: usize) -> Result<(RdPoint, Value)> {
    let started = Instant::now();
    let params = EncodeParams {
        lambda: Some(lambda),
        ..BASE
    };
    let (header, leaves, evals, stats) = encode_image_with_options(image, &params, &options(arm));
    let encode_elapsed_s = started.elapsed().as_secs_f64();
    let bytes = mars_format::write(&header, &leaves)?;
    let (wire_header, wire_leaves) = mars_format::read(&bytes)?;
    ensure!(
        wire_header == header && wire_leaves == leaves,
        "stream round-trip changed header/leaves"
    );
    let expected_step = if arm == 2 {
        ResidualQstep::from_lambda(lambda)
    } else {
        ResidualQstep::LEGACY
    };
    ensure!(
        wire_header.residual_qstep == expected_step,
        "wrong wire qstep"
    );
    let mut counts = [0_u64; 4];
    for leaf in &wire_leaves {
        ensure!(
            usize::from(leaf.mode) < 4 && options(arm).allowed_modes[usize::from(leaf.mode)],
            "disallowed leaf mode"
        );
        counts[usize::from(leaf.mode)] += 1;
    }
    ensure!(
        counts == stats.leaf_modes,
        "mode statistics disagree with stream"
    );
    ensure!(!wire_leaves.is_empty(), "empty partition");
    let shares = counts.map(|n| 100.0 * n as f64 / wire_leaves.len() as f64);
    // PSNR must be computed from the parsed stream, never the encoder's in-memory leaves.
    let decoded = decode_iterative(&wire_header, &wire_leaves, ITERATIONS);
    let psnr_db = psnr(image, &decoded).context("lossless point has no finite BD-rate PSNR")?;
    ensure!(psnr_db.is_finite(), "nonfinite PSNR");
    let point = RdPoint::from_size(bytes.len() as u64, image.width(), image.height(), psnr_db);
    let elapsed_s = started.elapsed().as_secs_f64();
    let details = json!({
        "lambda": lambda, "arm": ARMS[arm], "allowed_modes": options(arm).allowed_modes,
        "qstep": wire_header.residual_qstep.get(), "bpp": point.bpp, "psnr_db": point.psnr,
        "bytes": bytes.len(), "stream_sha256": sha256_hex(&bytes), "evals": evals,
        "leaves": wire_leaves.len(), "mode_counts": counts, "mode_shares_pct": shares,
        "split_decisions": stats.split_decisions, "leaf_decisions": stats.leaf_decisions,
        "encode_elapsed_s": encode_elapsed_s, "elapsed_s": elapsed_s
    });
    Ok((point, details))
}

fn sweep(root: &Path, file: &mut File, provenance: &Value) -> Result<()> {
    // Validate both inputs before spending minutes on the first encode.
    let images = corpus(root)?;
    let mut rates = [Vec::new(), Vec::new()];
    for (label, image, image_sha256) in images {
        let mut curves: [RdCurve; 3] =
            std::array::from_fn(|i| RdCurve::new(format!("{label} {}", ARMS[i]), Vec::new()));
        for lambda in LAMBDAS {
            // Interleave all three arms at each lambda; never run competing benchmarks.
            for arm in 0..3 {
                eprintln!("starting {label} lambda={lambda} {}", ARMS[arm]);
                let (point, details) = measure(&image, lambda, arm)?;
                eprintln!("{label}: {details}");
                row(
                    file,
                    json!({"kind": "sample", "image": label, "image_sha256": image_sha256,
                    "measurement": details, "provenance": provenance}),
                )?;
                curves[arm].points.push(point);
            }
        }
        for curve in &curves {
            check_convex_and_monotonic(&curve.points, 1.0).with_context(|| curve.label.clone())?;
        }
        for arm in 1..3 {
            let bd = bd_metrics(&curves[0], &curves[arm])?;
            ensure!(bd.bd_rate_pct.is_finite(), "nonfinite BD-rate");
            eprintln!(
                "{label} {} vs baseline: BD-rate {:+.6}%; PSNR overlap {:?} dB; traversed bpp {:?}; BD-PSNR bpp overlap {:?}; historical {:+.2}%",
                ARMS[arm],
                bd.bd_rate_pct,
                bd.psnr_interval_db,
                bd.bpp_interval,
                bd.bd_psnr_bpp_interval,
                HISTORICAL_REGRESSION_PCT
            );
            row(
                file,
                json!({"kind": "comparison", "image": label, "arm": ARMS[arm], "bd": bd,
                "historical_regression_pct": HISTORICAL_REGRESSION_PCT, "provenance": provenance}),
            )?;
            rates[arm - 1].push(bd.bd_rate_pct);
        }
    }
    let means = rates.map(|r| r.iter().sum::<f64>() / r.len() as f64);
    eprintln!(
        "Mean BD-rate vs baseline: fixed8 {:+.6}%; adaptive {:+.6}%; historical {:+.2}%; adaptive ceiling <= {:+.3}% (positive = regression)",
        means[0], means[1], HISTORICAL_REGRESSION_PCT, CEILING_PCT
    );
    row(
        file,
        json!({"kind": "summary", "fixed8_mean_bd_rate_pct": means[0],
        "adaptive_mean_bd_rate_pct": means[1], "historical_regression_pct": HISTORICAL_REGRESSION_PCT,
        "improvement_from_historical_percentage_points": HISTORICAL_REGRESSION_PCT - means[1],
        "adaptive_ceiling_pct": CEILING_PCT, "passed": accepts(means[1]), "provenance": provenance}),
    )?;
    ensure!(
        accepts(means[1]),
        "O7 ABORT: adaptive mean {:+.6}% exceeds +1.025%; improvement is less than half the historical +2.05% regression",
        means[1]
    );
    Ok(())
}

#[test]
#[ignore = "multi-minute Kodak measurement; explicitly run with just gate-22"]
fn step22_o7_acceptance() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let path = root.join(format!(
        "results/step22-o7-{stamp}-{}.jsonl",
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    eprintln!("gate-22 result rows: {}", path.display());
    eprintln!("PREDICTION: {PREDICTION}");
    let params = json!({"min_size": BASE.min_size, "max_size": BASE.max_size, "shift": BASE.shift,
        "bits_alfa": BASE.bits_alfa, "bits_beta": BASE.bits_beta, "max_alfa": BASE.max_alfa,
        "t_rms": BASE.t_rms, "zero_threshold": BASE.zero_threshold, "lambdas": LAMBDAS,
        "arms": ARMS, "adaptive_density": false, "decode_iterations": ITERATIONS,
        "adaptive_mapping": "sqrt(6*lambda/ln(2)), bounded [1,65535], rounded to binary32",
        "width": 768, "height": 512, "convexity_slack_db_per_bpp": 1.0});
    let mut detected = Provenance::detect(0).with_parameter_set(&params);
    detected.machine.rustc_version = command(&root, "rustc", &["--version"])?;
    let provenance = json!({"detected": detected,
        "git_status": command(&root, "git", &["--no-pager", "--no-optional-locks", "status", "--porcelain"] )?,
        "tracked_diff": command(&root, "git", &["--no-pager", "diff", "--binary", "HEAD"] )?,
        "harness_source": include_str!("residual_qstep_gate.rs"),
        "cargo_lock_sha256": sha256_hex(&fs::read(root.join("Cargo.lock"))?),
        "rayon_threads": rayon::current_num_threads(),
        "rustflags": std::env::var("RUSTFLAGS").ok(),
        "command": "cargo test -p mars-bench --release --test residual_qstep_gate step22_o7_acceptance -- --ignored --exact --nocapture --test-threads=1"});
    row(
        &mut file,
        json!({"kind": "metadata", "schema": "step22-o7-v1", "prediction": PREDICTION,
        "parameters": params, "provenance": provenance, "historical_regression_pct": HISTORICAL_REGRESSION_PCT,
        "adaptive_ceiling_pct": CEILING_PCT, "timing": "single foreground observation per point; elapsed includes encode/write/read/decode/PSNR, excludes row IO; not a speedup claim"}),
    )?;
    let started = Instant::now();
    let result = sweep(&root, &mut file, &provenance);
    row(
        &mut file,
        json!({"kind": "completion", "passed": result.is_ok(),
        "error": result.as_ref().err().map(|e| format!("{e:#}")), "elapsed_s": started.elapsed().as_secs_f64()}),
    )?;
    result
}

#[test]
fn acceptance_is_half_the_historical_regression_without_slack() {
    assert_eq!(CEILING_PCT, HISTORICAL_REGRESSION_PCT / 2.0);
    assert!(accepts(1.025));
    assert!(accepts(-2.0));
    assert!(!accepts(1.025000001));
    assert!(!accepts(f64::NAN));
    assert!(!accepts(f64::INFINITY));
    assert!(!accepts(f64::NEG_INFINITY));
}

#[test]
fn arms_and_real_stream_measurement_are_wired() -> Result<()> {
    assert_eq!(options(0).allowed_modes, STEP14_MODES);
    for arm in [0, 1] {
        assert_eq!(
            options(arm).residual_quantisation,
            ResidualQuantisation::Fixed(ResidualQstep::LEGACY)
        );
    }
    assert_eq!(
        options(2).residual_quantisation,
        ResidualQuantisation::LambdaAdaptive
    );
    let image = Plane::from_vec(
        32,
        32,
        (0..1024)
            .map(|i| ((i * 13 + i / 32 * 7) % 256) as u8)
            .collect(),
    );
    for arm in 0..3 {
        assert!(!options(arm).adaptive_density);
        let (point, details) = measure(&image, 200.0, arm)?;
        assert!(point.bpp > 0.0 && point.psnr.is_finite());
        let shares = details["mode_shares_pct"].as_array().unwrap();
        assert!((shares.iter().map(|v| v.as_f64().unwrap()).sum::<f64>() - 100.0).abs() < 1e-10);
    }
    Ok(())
}

#[test]
fn missing_corpus_is_an_error() {
    // A source file cannot contain the required corpus subdirectory, on any host.
    let impossible_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/residual_qstep_gate.rs");
    assert!(corpus(&impossible_root).is_err());
}
