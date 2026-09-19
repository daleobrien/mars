//! Real end-to-end P5a: `marsbench rate-audit` drives the actual codec over the pinned
//! tiny64 fixture and appends auditable rows. The 64x64 fixture is smoke evidence, not a
//! corpus measurement -- corpus numbers come from running the same command on Kodak.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use mars_bench::rate_audit::{PartitionKind, RateAuditRow};
use mars_bench::store::read_rows;

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mars-cli-rate-audit-{label}-{}",
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
                "marsbench rate-audit exceeded 120s: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn command(scratch: &Scratch, store: &Path, report: &Path) -> Command {
    let root = workspace_root();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_marsbench"));
    cmd.arg("rate-audit")
        .arg("--input")
        .arg(root.join("fixtures/mars1/tiny64.raw"))
        .args(["--dims", "64x64"])
        .args(["--lambdas", "200"])
        .args(["--thresholds", "8"])
        .args(["--modes", "0,2"])
        .args(["--iterations", "10"])
        .arg("--store")
        .arg(store)
        .arg("--scratch")
        .arg(scratch.path("scratch"))
        .arg("--markdown-out")
        .arg(report)
        .arg("--root")
        .arg(&root);
    cmd
}

#[test]
fn the_audit_runs_the_codec_appends_rows_and_appends_again_on_a_second_run() {
    let scratch = Scratch::new("run");
    let store = scratch.path("results.jsonl");
    let report = scratch.path("report.md");

    let first = run(&mut command(&scratch, &store, &report));
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );

    let rows: Vec<RateAuditRow> = read_rows(&store)
        .unwrap()
        .iter()
        .map(|row| {
            assert_eq!(row.kind, "rate-audit");
            serde_json::from_value(row.data.clone()).unwrap()
        })
        .collect();
    assert_eq!(rows.len(), 2, "one row per requested partition policy");

    let rd = rows
        .iter()
        .find(|row| row.partition == PartitionKind::Rd)
        .expect("an RD row");
    let threshold = rows
        .iter()
        .find(|row| row.partition == PartitionKind::Threshold)
        .expect("a threshold row");

    assert_eq!(rd.lambda, Some(200.0));
    assert_eq!(rd.t_rms, None);
    assert_eq!(rd.estimated_provenance, "rd_warmup_snapshot");
    assert_eq!(threshold.lambda, None);
    assert_eq!(threshold.t_rms, Some(8.0));
    assert_eq!(threshold.estimated_provenance, "reference_only");

    for row in &rows {
        assert_eq!(row.image, "tiny64");
        assert_eq!((row.width, row.height), (64, 64));
        // Nothing may go unattributed: a reader must be able to add the categories up.
        assert_eq!(row.unattributed_events, 0);
        let events: u64 = row.categories.iter().map(|c| c.events).sum();
        assert_eq!(events, row.events);
        assert_eq!(
            row.categories
                .iter()
                .map(|c| c.category.as_str())
                .collect::<Vec<_>>(),
            vec![
                "partition",
                "modes",
                "coordinates",
                "coefficients",
                "residuals"
            ]
        );
        assert!(row.estimated_bits > 0.0);
        assert!(row.actual_event_bits >= 0.0);
        assert!(row.inner_stream_bits >= row.actual_event_bits);
        assert!(row.overhead_bits >= 0.0);
        assert!(row.leaves > 0);
        // P5a starts at modes 0/2, so no residual fields exist to code.
        let residuals = row
            .categories
            .iter()
            .find(|c| c.category == "residuals")
            .unwrap();
        assert_eq!(residuals.events, 0);
        assert!(row.quality.psnr_y.unwrap().is_finite());
        assert!(row.quality.bpp.unwrap() > 0.0);
        assert!(row.container_bytes() > row.inner_stream_bytes);
    }

    let markdown = fs::read_to_string(&report).unwrap();
    assert!(markdown.contains("| tiny64 | rd | λ=200 |"));
    assert!(markdown.contains("| tiny64 | threshold | t_rms=8 |"));
    assert!(markdown.contains("## Category breakdown"));

    // The store is append-only: a second run adds rows rather than replacing them.
    let second = run(&mut command(&scratch, &store, &report));
    assert!(second.status.success());
    assert_eq!(read_rows(&store).unwrap().len(), 4);

    // A malformed request is refused before anything is appended.
    let bad = run(Command::new(env!("CARGO_BIN_EXE_marsbench"))
        .arg("rate-audit")
        .arg("--input")
        .arg(workspace_root().join("fixtures/mars1/tiny64.raw"))
        .args(["--dims", "64x64", "--modes", "7"])
        .arg("--store")
        .arg(scratch.path("bad.jsonl"))
        .arg("--root")
        .arg(workspace_root()));
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("--modes"));
    assert!(!scratch.path("bad.jsonl").exists());

    // A `--dims` without `--input` is likewise refused.
    let bad = run(Command::new(env!("CARGO_BIN_EXE_marsbench"))
        .arg("rate-audit")
        .args(["--dims", "64x64"])
        .arg("--store")
        .arg(scratch.path("bad2.jsonl")));
    assert!(!bad.status.success());
    assert!(!scratch.path("bad2.jsonl").exists());
}

#[test]
fn the_markdown_report_is_well_formed_json_free_table_output() {
    let scratch = Scratch::new("report");
    let store = scratch.path("results.jsonl");
    let report = scratch.path("report.md");
    let result = run(&mut command(&scratch, &store, &report));
    assert!(result.status.success());

    let markdown = fs::read_to_string(&report).unwrap();
    // Deterministic, self-describing headings; no stray JSON or unrendered placeholders.
    assert!(markdown.starts_with("# P5a rate audit"));
    assert!(markdown.contains("## Notes"));
    assert!(!markdown.contains('{'));
    assert!(!markdown.contains("NaN"));
}
