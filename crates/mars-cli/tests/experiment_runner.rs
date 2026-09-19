//! Real end-to-end P0.3: `marsbench experiment` drives the actual encmars/decmars over
//! the shipped `configs/research-smoke.json`, then resumes without re-encoding and
//! renders a report. No stubs, no library shortcuts.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mars-cli-experiment-{label}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run(command: &mut Command) -> Output {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if start.elapsed() > Duration::from_secs(120) {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "marsbench exceeded 120s: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn experiment_command(out: &Path, extra: &[&str]) -> Command {
    let root = workspace_root();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_marsbench"));
    cmd.arg("experiment")
        .arg("--config")
        .arg(root.join("configs/research-smoke.json"))
        .arg("--root")
        .arg(&root)
        .arg("--out-dir")
        .arg(out)
        .arg("--encmars")
        .arg(env!("CARGO_BIN_EXE_encmars"))
        .arg("--decmars")
        .arg(env!("CARGO_BIN_EXE_decmars"))
        .args(extra);
    cmd
}

#[test]
fn the_smoke_stage_runs_resumes_and_reports() {
    let tmp = Scratch::new("smoke");
    let out = tmp.path("artifacts");

    let result = run(&mut experiment_command(&out, &["--stage", "smoke"]));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let summary: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(summary["runner"], "experiment");
    assert_eq!(summary["stage"], "smoke");
    assert_eq!(summary["planned"], 1);
    assert_eq!(summary["executed"], 1);
    assert_eq!(summary["resumed"], 0);
    assert_eq!(summary["statuses"]["succeeded"], 1);
    assert_eq!(summary["statuses"]["failed"], 0);

    let store = out.join("results.jsonl");
    let lines = fs::read_to_string(&store).unwrap();
    assert_eq!(lines.lines().count(), 1);
    let row: Value = serde_json::from_str(lines.lines().next().unwrap()).unwrap();
    assert_eq!(row["kind"], "experiment");
    assert_eq!(row["data"]["status"], "succeeded");
    assert_eq!(row["data"]["container"], "MARC");
    assert_eq!(row["data"]["arm"], "rd-lambda200-modes0-2");
    assert_eq!(row["data"]["metrics"]["width"], 64);
    assert!(row["data"]["metrics"]["psnr_y"]
        .as_f64()
        .unwrap()
        .is_finite());
    assert_eq!(row["data"]["repetitions"].as_array().unwrap().len(), 1);
    // The encoder's own line and its parsed counters are both retained.
    assert!(
        row["data"]["encode_counters"]["search_evals"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(row["data"]["encode_summary"]
        .as_str()
        .unwrap()
        .contains("bpp"));
    assert!(row["data"]["build"]["git_sha"].is_string());
    assert!(row["provenance"]["parameter_set_sha256"].is_string());
    assert!(out.join("summary.json").is_file());
    assert!(out.join("cases").is_dir());

    // A second run resumes the completed case: no new row, nothing re-encoded.
    let result = run(&mut experiment_command(&out, &["--stage", "smoke"]));
    assert!(result.status.success());
    let summary: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(summary["executed"], 0);
    assert_eq!(summary["resumed"], 1);
    assert_eq!(fs::read_to_string(&store).unwrap().lines().count(), 1);

    // The report renders from the store alone.
    let report = run(Command::new(env!("CARGO_BIN_EXE_marsbench"))
        .arg("experiment-report")
        .arg("--store")
        .arg(&store)
        .arg("--reference-group")
        .arg("rd-lambda200-modes0-2"));
    assert!(
        report.status.success(),
        "{}",
        String::from_utf8_lossy(&report.stderr)
    );
    let markdown = String::from_utf8_lossy(&report.stdout);
    assert!(
        markdown.contains("# Experiment `research-smoke-"),
        "{markdown}"
    );
    assert!(markdown.contains("| succeeded | 1 |"), "{markdown}");
    assert!(markdown.contains("## Per-arm summary"), "{markdown}");
    assert!(
        markdown.contains("## BD-rate vs group `rd-lambda200-modes0-2`"),
        "{markdown}"
    );
}

#[test]
fn an_undefined_stage_is_rejected_explicitly() {
    let tmp = Scratch::new("stage");
    let out = tmp.path("artifacts");
    let result = run(&mut experiment_command(&out, &["--stage", "not-a-stage"]));
    assert!(!result.status.success());
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("not defined"), "{stderr}");
    assert!(stderr.contains("smoke"), "{stderr}");
}
