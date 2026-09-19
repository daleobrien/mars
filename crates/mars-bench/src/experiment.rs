//! One file-to-file smoke case, not the full P0 experiment/corpus runner.
//! Timings include process startup and file I/O; the timeout applies per codec process.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::experiment_config::{DecoderSpec, ExperimentConfig, Plan, PlannedCase, METHOD_KEYS};
/// Kept at its original path; the type now lives in `experiment_report` so a stored row
/// can be read back.
pub use crate::experiment_report::FileIdentity;
use crate::experiment_report::{
    BuildIdentity, CaseStatus, EncodeCounters, ExperimentCaseRow, Repetition, StatusTally,
};
use crate::measure::{measure, MeasureRequest, Measurement};
use crate::provenance::{sha256_hex, Provenance};
use crate::store::ResultStore;

/// The initial smoke profile is deliberately restricted to lambda 200 and modes 0,2.
#[derive(Debug, Clone, Serialize)]
pub struct SmokeOptions {
    pub input: PathBuf,
    /// Must not exist; its parent must already exist.
    pub out_dir: PathBuf,
    pub encmars: PathBuf,
    pub decmars: PathBuf,
    pub raw_dims: Option<(usize, usize)>,
    pub lambda: f64,
    pub modes: Vec<u8>,
    /// Fixed decoder iterations, not benchmark repetitions.
    pub iterations: u32,
    /// Encoder flag and RAYON_NUM_THREADS for both processes.
    pub threads: usize,
    pub timeout_secs: u64,
    /// Optional `encmars --method` key; `None` omits the flag entirely, so the default
    /// invocation encodes exactly the same bytes as before this field existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// `--budget`: required positive with `method: Some("random")`, rejected otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget: Option<usize>,
    /// `--seed`: only valid with `method: Some("random")` (encmars defaults it to 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Exact invocation and outcome. Logs live in the new artifact directory.
#[derive(Debug, Serialize)]
pub struct Phase {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub stdout: PathBuf,
    pub stderr: PathBuf,
    pub status: String,
    pub wall_secs: Option<f64>,
    pub pid: Option<u32>,
    pub exit_code: Option<i32>,
    pub exit_status: Option<String>,
    pub reaped: bool,
    pub error: Option<String>,
}

/// Persisted at checkpoints, including ordinary validation, spawn and timeout failures.
#[derive(Debug, Serialize)]
pub struct SmokeReport {
    pub schema_version: u32,
    pub runner: &'static str,
    pub scope: String,
    pub status: String,
    pub error: Option<String>,
    pub options: SmokeOptions,
    pub working_directory: PathBuf,
    /// Describes the harness checkout/build, not the external executable builds.
    pub provenance: Option<Provenance>,
    pub encoder_binary: Option<FileIdentity>,
    pub decoder_binary: Option<FileIdentity>,
    pub input: Option<FileIdentity>,
    pub stream: Option<FileIdentity>,
    pub decoded: Option<FileIdentity>,
    pub encode: Phase,
    pub decode: Phase,
    pub metrics: Option<Measurement>,
}

fn new_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {} without overwriting", path.display()))
}

fn identity(path: &Path) -> Result<FileIdentity> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    ensure!(
        file.metadata()?.is_file(),
        "{} is not a regular file",
        path.display()
    );
    let mut hash = Sha256::new();
    let mut bytes = 0;
    let mut buffer = [0_u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        bytes += n as u64;
    }
    Ok(FileIdentity {
        path: path.to_owned(),
        bytes,
        sha256: format!("{:x}", hash.finalize()),
    })
}

fn text(path: &Path) -> Result<String> {
    path.to_str()
        .map(str::to_owned)
        .context("experiment-smoke requires UTF-8 paths for exact JSON command arguments")
}

fn phase(program: &Path, args: Vec<String>, dir: &Path, name: &str, threads: usize) -> Phase {
    Phase {
        program: program.to_owned(),
        args,
        env: BTreeMap::from([("RAYON_NUM_THREADS".into(), threads.to_string())]),
        stdout: dir.join(format!("{name}.stdout.log")),
        stderr: dir.join(format!("{name}.stderr.log")),
        status: "not_started".into(),
        wall_secs: None,
        pid: None,
        exit_code: None,
        exit_status: None,
        reaped: false,
        error: None,
    }
}

fn persist(report: &SmokeReport) -> Result<()> {
    let dir = &report.options.out_dir;
    let temporary = dir.join("report.json.tmp");
    let mut file = new_file(&temporary)?;
    serde_json::to_writer_pretty(&mut file, report)?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()?;
    fs::rename(&temporary, dir.join("report.json"))?;
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn execute(phase: &mut Phase, cwd: &Path, timeout: Duration) -> Result<()> {
    let start = Instant::now();
    phase.status = "running".into();
    let result = (|| -> Result<()> {
        // Files rather than pipes avoid a noisy child blocking before it can exit.
        let stdout = new_file(&phase.stdout)?;
        let stderr = new_file(&phase.stderr)?;
        let mut child = Command::new(&phase.program)
            .args(&phase.args)
            .envs(&phase.env)
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(stdout.try_clone()?)
            .stderr(stderr.try_clone()?)
            .spawn()
            .with_context(|| format!("spawning {}", phase.program.display()))?;
        phase.pid = Some(child.id());
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if start.elapsed() < timeout => {
                    std::thread::sleep(
                        Duration::from_millis(10).min(timeout.saturating_sub(start.elapsed())),
                    );
                }
                outcome => {
                    let reason = match outcome {
                        Err(error) => format!("polling child: {error}"),
                        _ => {
                            phase.status = "timed_out".into();
                            format!("process timeout after {} seconds", timeout.as_secs())
                        }
                    };
                    // Always wait even if kill races with an exit or reports an error.
                    let killed = child.kill();
                    let waited = child.wait();
                    if let Ok(status) = &waited {
                        phase.reaped = true;
                        phase.exit_code = status.code();
                        phase.exit_status = Some(status.to_string());
                    }
                    stdout.sync_all()?;
                    stderr.sync_all()?;
                    bail!("{reason}; kill={killed:?}; wait={waited:?}");
                }
            }
        };
        phase.reaped = true;
        phase.exit_code = status.code();
        phase.exit_status = Some(status.to_string());
        stdout.sync_all()?;
        stderr.sync_all()?;
        ensure!(
            status.success(),
            "{} exited with {status}",
            phase.program.display()
        );
        Ok(())
    })();
    phase.wall_secs = Some(start.elapsed().as_secs_f64());
    match &result {
        Ok(()) => phase.status = "succeeded".into(),
        Err(error) => {
            if phase.status != "timed_out" {
                phase.status = "failed".into();
            }
            phase.error = Some(format!("{error:#}"));
        }
    }
    result
}

fn validate(options: &SmokeOptions) -> Result<()> {
    ensure!(
        options.lambda == 200.0,
        "initial smoke profile supports only --lambda 200"
    );
    ensure!(
        options.modes == [0, 2] || options.modes == [2, 0],
        "initial smoke profile supports only --modes 0,2 (no duplicates)"
    );
    ensure!(options.iterations > 0, "iterations must be positive");
    ensure!(options.threads > 0, "threads must be positive");
    ensure!(options.timeout_secs > 0, "timeout-secs must be positive");
    match options.method.as_deref() {
        // `random` needs a positive budget and may be seeded.
        Some("random") => {
            ensure!(
                options.budget.is_some_and(|budget| budget > 0),
                "--method random requires an explicit positive --budget"
            );
        }
        // `apcc` needs a positive budget and has no seed.
        Some("apcc") => {
            ensure!(
                options.budget.is_some_and(|budget| budget > 0),
                "--method apcc requires an explicit positive --budget"
            );
            ensure!(
                options.seed.is_none(),
                "--seed is only valid with --method random"
            );
        }
        Some(method) => {
            ensure!(
                METHOD_KEYS.contains(&method),
                "unsupported --method {method:?}; encmars accepts one of {METHOD_KEYS:?}"
            );
            ensure!(
                options.budget.is_none(),
                "--budget is only valid with --method random or apcc"
            );
            ensure!(
                options.seed.is_none(),
                "--seed is only valid with --method random"
            );
        }
        None => ensure!(
            options.budget.is_none() && options.seed.is_none(),
            "--budget/--seed are only valid with --method random or apcc"
        ),
    }
    let ext = options
        .input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let raw = matches!(ext.as_str(), "raw" | "y" | "gray");
    ensure!(
        raw == options.raw_dims.is_some(),
        "raw input requires --raw-dims; dimensions are not accepted for non-raw input"
    );
    if let Some((w, h)) = options.raw_dims {
        ensure!(
            w > 0 && h > 0 && w.checked_mul(h).is_some(),
            "raw dimensions must be positive and not overflow"
        );
        ensure!(
            fs::metadata(&options.input)?.len() == (w * h) as u64,
            "raw dimensions do not match input file size"
        );
    }
    Ok(())
}

fn run_case(report: &mut SmokeReport, stream: &Path, decoded: &Path) -> Result<()> {
    report.input = Some(identity(&report.options.input)?);
    validate(&report.options)?;
    report.encoder_binary =
        Some(identity(&report.options.encmars).context("encmars prerequisite")?);
    report.decoder_binary =
        Some(identity(&report.options.decmars).context("decmars prerequisite")?);
    report.provenance = Some(Provenance::detect(0).with_parameter_set(&report.options));
    persist(report)?;
    let timeout = Duration::from_secs(report.options.timeout_secs);
    execute(&mut report.encode, &report.working_directory, timeout)?;
    report.stream = Some(identity(stream).context("encoder output")?);
    let mut magic = [0_u8; 4];
    File::open(stream)?.read_exact(&mut magic)?;
    ensure!(
        &magic == b"MARC",
        "encoder output is not a whole MARC container"
    );
    persist(report)?;
    execute(&mut report.decode, &report.working_directory, timeout)?;
    report.decoded = Some(identity(decoded).context("decoder output")?);
    persist(report)?;
    let mut request = MeasureRequest::new(&report.options.input, decoded).with_coded_bytes(
        report
            .stream
            .as_ref()
            .context("missing stream identity")?
            .bytes,
    );
    request.raw_dims = report.options.raw_dims;
    report.metrics = Some(measure(&request).context("file-based metrics")?);
    // Do not silently measure an input that changed while the codec was running.
    ensure!(
        identity(&report.options.input)?.sha256
            == report
                .input
                .as_ref()
                .context("missing input identity")?
                .sha256,
        "input changed during experiment-smoke"
    );
    Ok(())
}

/// Run exactly one real encode -> MARC file -> real decode -> PNG -> file metrics case.
///
/// Returns an error on failure, with `report.json` retained in the new directory.
/// Existing destinations are never modified. Failure to create the directory, unsupported
/// non-UTF-8 paths, or storage errors cannot guarantee a report. No resume or corpus sweep.
/// Timeout cleanup kills/reaps the direct codec child, not arbitrary descendant processes.
pub fn run_smoke(options: &SmokeOptions) -> Result<SmokeReport> {
    let cwd = std::env::current_dir()?;
    let mut options = options.clone();
    for path in [
        &mut options.input,
        &mut options.out_dir,
        &mut options.encmars,
        &mut options.decmars,
    ] {
        if path.is_relative() {
            *path = cwd.join(&*path);
        }
        text(path)?;
    }
    let stream = options.out_dir.join("coded.mars");
    let decoded = options.out_dir.join("decoded.png");
    let mut encode_args = vec![text(&options.input)?, text(&stream)?];
    for (key, value) in [
        ("--lambda", options.lambda.to_string()),
        (
            "--modes",
            options
                .modes
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ),
        ("--threads", options.threads.to_string()),
        ("--min-size", "4".into()),
        ("--max-size", "16".into()),
        ("--shift", "4".into()),
        ("--bits-alfa", "4".into()),
        ("--bits-beta", "7".into()),
        ("--max-alfa", "1".into()),
        ("--zero-threshold", "0".into()),
        ("--subsampling", "444".into()),
        ("--t-rms", "8".into()),
        ("--chroma-t-rms", "8".into()),
    ] {
        encode_args.extend([key.into(), value]);
    }
    // Passthrough flags are appended only when explicitly requested; omission must keep
    // the encode argv exactly what it was before this passthrough existed.
    if let Some(method) = &options.method {
        encode_args.extend(["--method".into(), method.clone()]);
    }
    if let Some(budget) = options.budget {
        encode_args.extend(["--budget".into(), budget.to_string()]);
    }
    if let Some(seed) = options.seed {
        encode_args.extend(["--seed".into(), seed.to_string()]);
    }
    if let Some((w, h)) = options.raw_dims {
        encode_args.extend([
            "--raw-width".into(),
            w.to_string(),
            "--raw-height".into(),
            h.to_string(),
        ]);
    }
    let decode_args = vec![
        text(&stream)?,
        text(&decoded)?,
        "--iterations".into(),
        options.iterations.to_string(),
        "--zoom".into(),
        "1".into(),
    ];
    let mut report = SmokeReport {
        schema_version: 1, runner: "experiment-smoke",
        scope: format!("One case only; partial P0 smoke, not a corpus benchmark or full experiment runner. Single-layer MARC, RD lambda 200, modes 0,2; adaptive density/residual and progressive off. Method passthrough: method={:?}, budget={:?}, seed={:?}; None omits the flag. Methods are grayscale-only; random's omitted seed defaults to 0 in encmars and its budget excludes RD warm-up. Process wall times include startup and file I/O. Provenance describes the harness; binary SHA256 identifies the external codecs. Other environment variables are inherited.", options.method, options.budget, options.seed),
        status: "running".into(), error: None,
        encode: phase(&options.encmars, encode_args, &options.out_dir, "encode", options.threads),
        decode: phase(&options.decmars, decode_args, &options.out_dir, "decode", options.threads),
        options, working_directory: cwd, provenance: None,
        encoder_binary: None, decoder_binary: None, input: None, stream: None, decoded: None, metrics: None,
    };
    fs::create_dir(&report.options.out_dir).with_context(|| {
        format!(
            "artifact directory must be new (parent must exist): {}",
            report.options.out_dir.display()
        )
    })?;
    persist(&report)?;
    let outcome = run_case(&mut report, &stream, &decoded);
    if let Err(error) = &outcome {
        report.status = "failed".into();
        report.error = Some(format!("{error:#}"));
        for phase in [&mut report.encode, &mut report.decode] {
            if phase.status == "not_started" {
                phase.status = "blocked".into();
                phase.error = Some(format!("not run because case failed: {error:#}"));
            }
        }
        // Preserve identities of partial output files even when a process failed.
        for (path, slot) in [
            (&stream, &mut report.stream),
            (&decoded, &mut report.decoded),
        ] {
            if path.exists() && slot.is_none() {
                match identity(path) {
                    Ok(value) => *slot = Some(value),
                    Err(error) => report
                        .error
                        .as_mut()
                        .unwrap()
                        .push_str(&format!("; artifact hash: {error:#}")),
                }
            }
        }
    } else {
        report.status = "succeeded".into();
    }
    persist(&report).context("persisting final experiment-smoke report")?;
    outcome.with_context(|| {
        format!(
            "experiment-smoke failed; report retained at {}",
            report.options.out_dir.join("report.json").display()
        )
    })?;
    Ok(report)
}

// ================================================================ P0.3 experiment runner
//
// The strict, resumable, config-driven runner (§3 P0.3, §14). It plans `(image × arm)`
// into content-addressed cases, fails fast on missing prerequisites, appends each case's
// row the moment it completes, and resumes only where an already-completed case's config
// and build identity match exactly. Every planned case is represented in the store —
// including `unsupported` ones — so a short result file can never masquerade as a run
// that merely had fewer cases.

/// Options for one `marsbench experiment` invocation.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub config_path: PathBuf,
    pub stage: Option<String>,
    /// Artifact root; defaults to `<root>/target/experiments/<experiment_id>`.
    pub out_dir: Option<PathBuf>,
    /// Repo root for resolving config-relative index/input paths.
    pub root: PathBuf,
    pub encmars: PathBuf,
    pub decmars: PathBuf,
    /// Cap the number of planned cases, for cheap smoke runs and tests.
    pub limit: Option<usize>,
}

/// The runner's result. `is_clean` is the exit condition: any non-succeeded case makes
/// the summary unclean and the command exits non-zero.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentSummary {
    pub runner: &'static str,
    pub schema_version: u32,
    pub experiment: String,
    pub experiment_id: String,
    pub stage: String,
    pub config_sha256: String,
    pub planned: usize,
    pub resumed: usize,
    pub executed: usize,
    pub statuses: StatusTally,
    pub store: PathBuf,
    pub summary_path: PathBuf,
    pub artifact_root: PathBuf,
    pub build: BuildIdentity,
}

impl ExperimentSummary {
    pub fn is_clean(&self) -> bool {
        self.statuses.is_clean()
    }
}

/// Why the subprocess runner cannot report per-block search diagnostics.
const DIAGNOSTICS_NOTE: &str = "candidate coverage/regret and leaf/mode histograms are not exposed by the subprocess codec; use `marsbench recall` against the oracle cache for oracle-based coverage";

fn failed(error: anyhow::Error) -> (CaseStatus, String) {
    (CaseStatus::Failed, format!("{error:#}"))
}

fn git_capture(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).to_string())
}

/// The source/build identity resume compares. Recorded per row so two rows are only
/// ever treated as the same measurement when their inputs actually agree.
fn build_identity(root: &Path) -> BuildIdentity {
    let git_sha = git_capture(root, &["rev-parse", "HEAD"])
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let status =
        git_capture(root, &["status", "--porcelain", "--untracked-files=no"]).unwrap_or_default();
    let git_dirty = !status.trim().is_empty();
    let git_patch_sha256 = if git_dirty {
        git_capture(root, &["--no-pager", "diff", "HEAD"]).map(|diff| sha256_hex(diff.as_bytes()))
    } else {
        None
    };
    let lockfile_sha256 = std::fs::read(root.join("Cargo.lock"))
        .ok()
        .map(|bytes| sha256_hex(&bytes));
    let rustc_version = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    BuildIdentity {
        git_sha,
        git_dirty,
        git_patch_sha256,
        lockfile_sha256,
        rustc_version,
    }
}

fn container_of(path: &Path) -> Result<String> {
    let mut magic = [0_u8; 4];
    File::open(path)?.read_exact(&mut magic)?;
    Ok(String::from_utf8_lossy(&magic).to_string())
}

fn encode_args(
    plan: &Plan,
    case: &PlannedCase,
    input: &Path,
    stream: &Path,
) -> Result<Vec<String>> {
    let mut args = vec![text(input)?, text(stream)?];
    if let Some(lambda) = case.arm.lambda {
        args.extend(["--lambda".into(), lambda.to_string()]);
    }
    if let Some(modes) = &case.arm.modes {
        args.extend([
            "--modes".into(),
            modes
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ]);
    }
    if let Some(method) = &case.arm.method {
        args.extend(["--method".into(), method.clone()]);
    }
    if let Some(budget) = case.arm.budget {
        args.extend(["--budget".into(), budget.to_string()]);
    }
    if let Some(seed) = case.arm.seed {
        args.extend(["--seed".into(), seed.to_string()]);
    }
    if case.arm.adaptive_density {
        args.push("--adaptive-density".into());
    }
    if case.arm.rd_candidates != 1 {
        args.extend(["--rd-candidates".into(), case.arm.rd_candidates.to_string()]);
    }
    if case.arm.adaptive_residual {
        args.push("--adaptive-residual".into());
    }
    if case.arm.progressive {
        args.push("--progressive".into());
    }
    let codec = &plan.codec;
    for (key, value) in [
        ("--min-size", codec.min_size.to_string()),
        ("--max-size", codec.max_size.to_string()),
        ("--shift", codec.shift.to_string()),
        ("--bits-alfa", codec.bits_alfa.to_string()),
        ("--bits-beta", codec.bits_beta.to_string()),
        ("--max-alfa", codec.max_alfa.to_string()),
        ("--zero-threshold", codec.zero_threshold.to_string()),
        ("--subsampling", codec.subsampling.clone()),
        ("--t-rms", codec.t_rms.to_string()),
        ("--chroma-t-rms", codec.chroma_t_rms.to_string()),
        ("--threads", codec.threads.to_string()),
    ] {
        args.extend([key.into(), value]);
    }
    if let Some((width, height)) = case.image.raw_dims {
        args.extend([
            "--raw-width".into(),
            width.to_string(),
            "--raw-height".into(),
            height.to_string(),
        ]);
    }
    Ok(args)
}

fn decode_args(stream: &Path, decoded: &Path, decoder: &DecoderSpec) -> Result<Vec<String>> {
    let mut args = vec![text(stream)?, text(decoded)?];
    args.extend(["--iterations".into(), decoder.iterations.to_string()]);
    args.extend(["--zoom".into(), decoder.zoom.to_string()]);
    if decoder.smooth {
        args.push("--smooth".into());
    }
    if decoder.auto {
        args.push("--auto".into());
        args.extend(["--threshold".into(), decoder.threshold.to_string()]);
    }
    if let Some(layer) = decoder.layer {
        args.extend(["--layer".into(), layer.to_string()]);
    }
    Ok(args)
}

/// The provenance for one case: harness/machine identity plus this case's own parameter
/// hash and its source corpus index hash.
fn case_provenance(plan: &Plan, case: &PlannedCase, root: &Path) -> Provenance {
    let params = serde_json::json!({
        "experiment_id": plan.experiment_id,
        "config_sha256": plan.config_sha256,
        "stage": plan.stage,
        "case_id": case.case_id,
        "image": case.image.name,
        "arm": case.arm,
        "codec": plan.codec,
        "decoder": case.decoder,
    });
    let mut provenance = Provenance::detect(case.index as u32).with_parameter_set(&params);
    if let Some(index) = &case.image.corpus_index {
        if let Ok(bytes) = std::fs::read(root.join(index)) {
            provenance = provenance.with_corpus_manifest(&bytes);
        }
    }
    provenance
}

struct CaseContext<'a> {
    plan: &'a Plan,
    root: &'a Path,
    artifact_root: &'a Path,
    build: &'a BuildIdentity,
    encmars: &'a Path,
    decmars: &'a Path,
    encoder_binary: Option<FileIdentity>,
    decoder_binary: Option<FileIdentity>,
}

fn base_row(ctx: &CaseContext, case: &PlannedCase) -> ExperimentCaseRow {
    ExperimentCaseRow {
        schema: crate::experiment_report::EXPERIMENT_SCHEMA,
        experiment: ctx.plan.experiment.clone(),
        experiment_id: ctx.plan.experiment_id.clone(),
        config_sha256: ctx.plan.config_sha256.clone(),
        stage: ctx.plan.stage.clone(),
        case_id: case.case_id.clone(),
        case_index: case.index,
        total_cases: case.total,
        arm: case.arm.label.clone(),
        arm_description: case.arm.description.clone(),
        image: case.image.name.clone(),
        image_file: case.image.file.display().to_string(),
        corpus_index: case
            .image
            .corpus_index
            .as_ref()
            .map(|path| path.display().to_string()),
        raw_dims: case.image.raw_dims,
        planes: case.image.planes,
        artifact_dir: None,
        container: None,
        codec: ctx.plan.codec.clone(),
        arm_params: case.arm.clone(),
        decoder: case.decoder.clone(),
        encode_args: Vec::new(),
        decode_args: Vec::new(),
        input: None,
        encoder_binary: ctx.encoder_binary.clone(),
        decoder_binary: ctx.decoder_binary.clone(),
        build: ctx.build.clone(),
        status: CaseStatus::Failed,
        error: None,
        unsupported_reason: None,
        repetitions: Vec::new(),
        encode_wall_secs: None,
        decode_wall_secs: None,
        stream: None,
        decoded: None,
        metrics: None,
        metrics_unavailable: None,
        encode_summary: None,
        encode_counters: None,
        diagnostics_unavailable: None,
        timeout_secs: ctx.plan.timeout_secs,
        notes: String::new(),
    }
}

/// One explicit `unsupported` record for an image/arm pair the codec refuses.
fn unsupported_row(ctx: &CaseContext, case: &PlannedCase, reason: &str) -> ExperimentCaseRow {
    let mut row = base_row(ctx, case);
    row.status = CaseStatus::Unsupported;
    row.unsupported_reason = Some(reason.to_string());
    row.error = Some(format!("not run: {reason}"));
    row
}

/// Run every repetition for one case, mutating `row`. Errors become statuses, never
/// panics: a single bad case must not take the run's accounting down with it.
fn run_attempt(
    ctx: &CaseContext,
    case: &PlannedCase,
    row: &mut ExperimentCaseRow,
    attempt: usize,
) -> std::result::Result<(), (CaseStatus, String)> {
    let input = ctx.root.join(&case.image.file);
    let artifact_dir = ctx
        .artifact_root
        .join("cases")
        .join(&case.case_id)
        .join(format!("attempt-{attempt}"));
    row.artifact_dir = Some(artifact_dir.clone());
    if !input.is_file() {
        return Err((
            CaseStatus::MissingPrerequisite,
            format!("input {} is missing", input.display()),
        ));
    }
    if !ctx.encmars.is_file() {
        return Err((
            CaseStatus::MissingPrerequisite,
            format!("encmars {} is missing", ctx.encmars.display()),
        ));
    }
    if !ctx.decmars.is_file() {
        return Err((
            CaseStatus::MissingPrerequisite,
            format!("decmars {} is missing", ctx.decmars.display()),
        ));
    }
    row.input = Some(identity(&input).map_err(failed)?);
    fs::create_dir_all(&artifact_dir).map_err(|error| {
        (
            CaseStatus::Failed,
            format!("creating {}: {error}", artifact_dir.display()),
        )
    })?;

    let timeout = Duration::from_secs(ctx.plan.timeout_secs);
    let mut repetitions: Vec<Repetition> = Vec::new();
    let mut container: Option<String> = None;
    for rep in 0..ctx.plan.repetitions {
        let rep_dir = artifact_dir.join(format!("rep-{rep:03}"));
        fs::create_dir_all(&rep_dir).map_err(|error| {
            (
                CaseStatus::Failed,
                format!("creating {}: {error}", rep_dir.display()),
            )
        })?;
        let stream = rep_dir.join("coded.mars");
        let decoded = rep_dir.join(format!("decoded.{}", case.decoder.output));

        let args = encode_args(ctx.plan, case, &input, &stream).map_err(failed)?;
        if row.encode_args.is_empty() {
            row.encode_args = args.clone();
        }
        let mut encode = phase(
            ctx.encmars,
            args,
            &rep_dir,
            "encode",
            ctx.plan.codec.threads,
        );
        if let Err(error) = execute(&mut encode, ctx.root, timeout) {
            let status = if encode.status == "timed_out" {
                CaseStatus::TimedOut
            } else {
                CaseStatus::Failed
            };
            return Err((status, format!("encode: {error:#}")));
        }
        let stream_identity = identity(&stream).map_err(failed)?;
        let magic = container_of(&stream).map_err(failed)?;
        if !matches!(magic.as_str(), "MARC" | "MPRG") {
            return Err((
                CaseStatus::Failed,
                format!("encoder output is not a whole container (magic {magic:?})"),
            ));
        }
        container = Some(magic);

        let args = decode_args(&stream, &decoded, &case.decoder).map_err(failed)?;
        if row.decode_args.is_empty() {
            row.decode_args = args.clone();
        }
        let mut decode = phase(
            ctx.decmars,
            args,
            &rep_dir,
            "decode",
            ctx.plan.codec.threads,
        );
        if let Err(error) = execute(&mut decode, ctx.root, timeout) {
            let status = if decode.status == "timed_out" {
                CaseStatus::TimedOut
            } else {
                CaseStatus::Failed
            };
            return Err((status, format!("decode: {error:#}")));
        }
        let decoded_identity = identity(&decoded).map_err(failed)?;
        repetitions.push(Repetition {
            index: rep,
            encode_wall_secs: encode.wall_secs.unwrap_or(0.0),
            decode_wall_secs: decode.wall_secs.unwrap_or(0.0),
            stream: stream_identity,
            decoded: decoded_identity,
            decode_iterations: case.decoder.iterations,
            encode_status: encode.status.clone(),
            decode_status: decode.status.clone(),
        });
    }
    if repetitions.len() != ctx.plan.repetitions as usize {
        return Err((CaseStatus::Failed, "not every repetition completed".into()));
    }
    // §2.3: a deterministic codec must produce identical bytes and pixels across repeats.
    if repetitions
        .iter()
        .any(|r| r.stream.sha256 != repetitions[0].stream.sha256)
    {
        return Err((
            CaseStatus::Failed,
            "encoder produced different bytes across repetitions".into(),
        ));
    }
    if repetitions
        .iter()
        .any(|r| r.decoded.sha256 != repetitions[0].decoded.sha256)
    {
        return Err((
            CaseStatus::Failed,
            "decoder produced different pixels across repetitions".into(),
        ));
    }
    let (first_encode, first_decode) = (
        repetitions[0].encode_wall_secs,
        repetitions[0].decode_wall_secs,
    );
    let (first_stream, first_decoded) = (
        repetitions[0].stream.clone(),
        repetitions[0].decoded.clone(),
    );
    row.repetitions = repetitions;
    row.container = container;
    row.encode_wall_secs = Some(first_encode);
    row.decode_wall_secs = Some(first_decode);
    row.stream = Some(first_stream.clone());
    row.decoded = Some(first_decoded);

    if let Ok(stdout) = fs::read_to_string(artifact_dir.join("rep-000").join("encode.stdout.log")) {
        row.encode_summary = stdout
            .lines()
            .rev()
            .find(|line| line.contains(" bpp"))
            .map(str::to_string);
        row.encode_counters = EncodeCounters::parse(&stdout);
    }

    let mut request = MeasureRequest::new(
        &input,
        artifact_dir
            .join("rep-000")
            .join(format!("decoded.{}", case.decoder.output)),
    )
    .with_coded_bytes(first_stream.bytes);
    request.raw_dims = case.image.raw_dims;
    match measure(&request) {
        Ok(measurement) => row.metrics = Some(measurement),
        Err(error) => return Err((CaseStatus::Failed, format!("metrics: {error}"))),
    }

    // The input must not have changed under the run (the smoke runner's own check).
    let after = identity(&input).map_err(failed)?;
    if row.input.as_ref().map(|identity| &identity.sha256) != Some(&after.sha256) {
        return Err((CaseStatus::Failed, "input changed during the run".into()));
    }

    row.diagnostics_unavailable = Some(DIAGNOSTICS_NOTE.to_string());
    row.status = CaseStatus::Succeeded;
    Ok(())
}

fn execute_case(ctx: &CaseContext, case: &PlannedCase, attempt: usize) -> ExperimentCaseRow {
    let mut row = base_row(ctx, case);
    if let Err((status, error)) = run_attempt(ctx, case, &mut row, attempt) {
        row.status = status;
        row.error = Some(error);
    }
    row
}

fn write_summary(summary: &ExperimentSummary) -> Result<()> {
    let temporary = summary.summary_path.with_extension("json.tmp");
    let _ = fs::remove_file(&temporary);
    let mut file = new_file(&temporary)?;
    serde_json::to_writer_pretty(&mut file, summary)?;
    file.write_all(b"\n")?;
    file.flush()?;
    file.sync_all()?;
    fs::rename(&temporary, &summary.summary_path)?;
    let parent = summary
        .summary_path
        .parent()
        .context("summary path has no parent")?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

/// Run one stage of one experiment, resuming where a completed case's build matches.
///
/// Returns `Ok` even when some cases fail — the summary carries the tally, and the CLI
/// exits non-zero on an unclean summary. `Err` is reserved for setup failures that
/// prevent the run from starting honestly at all (bad config, missing input, unreadable
/// store), so a missing prerequisite never looks like a completed run.
pub fn run_experiment(options: &RunOptions) -> Result<ExperimentSummary> {
    let config = ExperimentConfig::read(&options.config_path)?;
    let stage = config.stage(options.stage.as_deref())?.clone();
    let images = config.resolve(&options.root)?;
    let mut plan = crate::experiment_config::plan(&config, &stage, &images)?;
    if let Some(limit) = options.limit {
        plan.cases.truncate(limit);
        let total = plan.cases.len();
        for (index, case) in plan.cases.iter_mut().enumerate() {
            case.index = index;
            case.total = total;
        }
    }

    let artifact_root = options.out_dir.clone().unwrap_or_else(|| {
        options
            .root
            .join("target/experiments")
            .join(&plan.experiment_id)
    });
    fs::create_dir_all(&artifact_root)
        .with_context(|| format!("creating {}", artifact_root.display()))?;
    let store_path = artifact_root.join("results.jsonl");
    let summary_path = artifact_root.join("summary.json");

    let build = build_identity(&options.root);
    // Preflight every prerequisite before any expensive work.
    let encoder_binary = identity(&options.encmars).context("encmars prerequisite")?;
    let decoder_binary = identity(&options.decmars).context("decmars prerequisite")?;

    // Resume scan: only an already-completed case with an identical config and build is
    // resumable; anything else is re-attempted, and a *completed* case under a different
    // build is a hard error rather than a silent re-run.
    let prior = if store_path.exists() {
        crate::experiment_report::read_cases(&store_path, Some(&plan.experiment_id))?
    } else {
        Vec::new()
    };
    let mut attempts: BTreeMap<String, usize> = BTreeMap::new();
    for row in &prior {
        *attempts.entry(row.case_id.clone()).or_insert(0) += 1;
    }
    let latest = crate::experiment_report::latest_by_case(&prior);
    let mut conflicts: Vec<&str> = Vec::new();
    for case in &plan.cases {
        let Some(row) = latest.iter().find(|row| row.case_id == case.case_id) else {
            continue;
        };
        let completed = if case.unsupported.is_some() {
            matches!(row.status, CaseStatus::Unsupported)
        } else {
            row.status.is_succeeded()
        };
        if completed && (row.config_sha256 != plan.config_sha256 || row.build != build) {
            conflicts.push(case.case_id.as_str());
        }
    }
    if !conflicts.is_empty() {
        bail!(
            "resume requires a matching config/build identity: {} completed case(s) were recorded under a different build ({}); use a fresh --out-dir or remove {}",
            conflicts.len(),
            conflicts.iter().take(3).copied().collect::<Vec<_>>().join(", "),
            store_path.display()
        );
    }

    let ctx = CaseContext {
        plan: &plan,
        root: &options.root,
        artifact_root: &artifact_root,
        build: &build,
        encmars: &options.encmars,
        decmars: &options.decmars,
        encoder_binary: Some(encoder_binary),
        decoder_binary: Some(decoder_binary),
    };

    let mut summary = ExperimentSummary {
        runner: "experiment",
        schema_version: crate::experiment_report::EXPERIMENT_SCHEMA,
        experiment: plan.experiment.clone(),
        experiment_id: plan.experiment_id.clone(),
        stage: plan.stage.clone(),
        config_sha256: plan.config_sha256.clone(),
        planned: plan.cases.len(),
        resumed: 0,
        executed: 0,
        statuses: StatusTally::default(),
        store: store_path.clone(),
        summary_path: summary_path.clone(),
        artifact_root: artifact_root.clone(),
        build: build.clone(),
    };
    write_summary(&summary)?;

    let mut store = ResultStore::open(&store_path)?;
    for case in &plan.cases {
        if let Some(row) = latest.iter().find(|row| row.case_id == case.case_id) {
            let completed = if case.unsupported.is_some() {
                matches!(row.status, CaseStatus::Unsupported)
            } else {
                row.status.is_succeeded()
            };
            if completed && row.config_sha256 == plan.config_sha256 && row.build == build {
                summary.resumed += 1;
                summary.statuses.add(row.status);
                write_summary(&summary)?;
                continue;
            }
        }
        let attempt = attempts.get(&case.case_id).copied().unwrap_or(0);
        let row = match &case.unsupported {
            Some(reason) => unsupported_row(&ctx, case, reason),
            None => execute_case(&ctx, case, attempt),
        };
        let provenance = case_provenance(&plan, case, &options.root);
        crate::experiment_report::append_case(&mut store, &provenance, &row)?;
        summary.statuses.add(row.status);
        summary.executed += 1;
        // Persist after every case: an interrupted run's store is complete up to the
        // case in flight, and resuming re-runs exactly that case.
        write_summary(&summary)?;
    }

    write_summary(&summary)?;
    Ok(summary)
}
