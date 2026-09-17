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

use crate::measure::{measure, MeasureRequest, Measurement};
use crate::provenance::Provenance;

/// Every `encmars --method` key: the nine indexed providers plus opt-in `random`.
const METHOD_KEYS: [&str; 10] = [
    "exhaustive",
    "fisher",
    "hurtgen",
    "masscenter",
    "saupe",
    "saupe-fisher",
    "mc-saupe",
    "funnel",
    "learned",
    "random",
];

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

/// A hash of the file bytes, not of decoded pixels or an inner stream payload.
#[derive(Debug, Serialize)]
pub struct FileIdentity {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
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
    match &options.method {
        Some(method) => {
            ensure!(
                METHOD_KEYS.contains(&method.as_str()),
                "unsupported --method {method:?}; encmars accepts one of {METHOD_KEYS:?}"
            );
            if method == "random" {
                ensure!(
                    options.budget.is_some_and(|budget| budget > 0),
                    "--method random requires an explicit positive --budget"
                );
            } else {
                ensure!(
                    options.budget.is_none(),
                    "--budget is only valid with --method random"
                );
                ensure!(
                    options.seed.is_none(),
                    "--seed is only valid with --method random"
                );
            }
        }
        None => ensure!(
            options.budget.is_none() && options.seed.is_none(),
            "--budget/--seed are only valid with --method random"
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
