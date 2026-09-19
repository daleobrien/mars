//! Library-level checks for the P0.3 runner: preflight, per-case accounting, explicit
//! unsupported/timeout/failure rows, and build-matched resume. Codec stubs are native
//! Rust executables, never shell scripts.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use mars_bench::experiment::{run_experiment, RunOptions};
use mars_bench::experiment_config::{
    Arm, CodecSpec, DecoderSpec, ExperimentConfig, InputSpec, StageSpec, CONFIG_SCHEMA,
};
use mars_bench::experiment_report::{read_cases, CaseStatus};
use mars_core::image::{Image, Plane};

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mars-bench-runner-{label}-{}", std::process::id()));
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

fn compile_stub(tmp: &Scratch) -> PathBuf {
    let source = tmp.path("stub.rs");
    let binary = tmp.path(&format!("stub{}", std::env::consts::EXE_SUFFIX));
    // A stub codec: encode writes a whole MARC container holding the original input
    // path; decode writes a valid 8x8 PGM so file metrics can succeed. Behaviour is
    // keyed off the input file's name, so one binary serves every case.
    fs::write(
        &source,
        r#"
use std::{env, fs, thread, time::Duration};
fn main() {
    let args: Vec<_> = env::args().collect();
    let input = &args[1];
    let output = &args[2];
    if output.ends_with("coded.mars") {
        if input.contains("encode-timeout") { thread::sleep(Duration::from_secs(60)); }
        if input.contains("encode-fail") { std::process::exit(7); }
        fs::write(output, format!("MARC{input}")).unwrap();
        println!("8x8 gray -> {} (10 bytes, 1.250 bpp, 7 evals)", output);
    } else {
        let label = String::from_utf8(fs::read(input).unwrap()).unwrap();
        if label.contains("decode-timeout") { thread::sleep(Duration::from_secs(60)); }
        if label.contains("decode-fail") { std::process::exit(9); }
        let mut bytes = b"P5\n8 8\n255\n".to_vec();
        bytes.extend(0..64u8);
        fs::write(output, bytes).unwrap();
    }
}
"#,
    )
    .unwrap();
    let log = fs::File::create(tmp.path("rustc.log")).unwrap();
    let mut child = Command::new("rustc")
        .arg("--edition=2021")
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .stdin(Stdio::null())
        .stdout(log.try_clone().unwrap())
        .stderr(log)
        .spawn()
        .unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(
                status.success(),
                "{}",
                fs::read_to_string(tmp.path("rustc.log")).unwrap()
            );
            break;
        }
        if start.elapsed() > Duration::from_secs(30) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("stub compilation exceeded 30s");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    binary
}

fn arm(label: &str) -> Arm {
    Arm {
        label: label.into(),
        description: "test arm".into(),
        group: None,
        lambda: Some(200.0),
        modes: Some(vec![0, 2]),
        method: None,
        budget: None,
        seed: None,
        adaptive_density: false,
        rd_candidates: 1,
        adaptive_residual: false,
        progressive: false,
        iterations: None,
        smooth: None,
    }
}

fn raw_input(name: &str, file: PathBuf) -> InputSpec {
    InputSpec {
        name: name.into(),
        file,
        width: Some(8),
        height: Some(8),
        sha256: None,
    }
}

fn config(
    inputs: Vec<InputSpec>,
    arms: Vec<Arm>,
    repetitions: u32,
    timeout_secs: u64,
) -> ExperimentConfig {
    ExperimentConfig {
        schema_version: CONFIG_SCHEMA,
        name: "runner-test".into(),
        description: "library-level runner test".into(),
        indexes: Vec::new(),
        images: Vec::new(),
        inputs,
        codec: CodecSpec {
            min_size: 4,
            max_size: 8,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            zero_threshold: 0,
            subsampling: "444".into(),
            t_rms: 8.0,
            chroma_t_rms: 8.0,
            threads: 1,
        },
        decoder: DecoderSpec {
            // The stub writes a hand-rolled PGM; `measure` picks its reader by extension.
            output: "pgm".into(),
            ..DecoderSpec::default()
        },
        stages: vec![StageSpec {
            name: "stage".into(),
            description: "one stage".into(),
            repetitions,
            timeout_secs,
            arms,
        }],
    }
}

fn options(tmp: &Scratch, config: &ExperimentConfig, stub: &Path, out: &Path) -> RunOptions {
    let config_path = tmp.path("config.json");
    fs::write(&config_path, serde_json::to_string_pretty(config).unwrap()).unwrap();
    RunOptions {
        config_path,
        stage: None,
        out_dir: Some(out.to_path_buf()),
        root: tmp.0.clone(),
        encmars: stub.to_path_buf(),
        decmars: stub.to_path_buf(),
        limit: None,
    }
}

fn write_raw(path: &Path) {
    fs::write(path, (0..64u8).collect::<Vec<u8>>()).unwrap();
}

#[test]
fn a_successful_case_is_recorded_and_resumed_without_re_encoding() {
    let tmp = Scratch::new("resume");
    let stub = compile_stub(&tmp);
    let input = tmp.path("a.raw");
    write_raw(&input);
    let cfg = config(
        vec![raw_input("a", input.clone())],
        vec![arm("baseline")],
        1,
        10,
    );
    let out = tmp.path("out");
    let options = options(&tmp, &cfg, &stub, &out);

    let first = run_experiment(&options).unwrap();
    let store = out.join("results.jsonl");
    let rows = read_cases(&store, None).unwrap();
    assert!(first.is_clean(), "{first:?} rows={rows:#?}");
    assert_eq!((first.planned, first.executed, first.resumed), (1, 1, 0));
    assert_eq!(first.statuses.succeeded, 1);

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].status, CaseStatus::Succeeded);
    assert!(rows[0].metrics.is_some());
    assert!(rows[0].input.is_some());
    assert_eq!(rows[0].container.as_deref(), Some("MARC"));
    assert!(rows[0].encode_counters.is_some());
    assert!(rows[0]
        .encode_summary
        .as_deref()
        .unwrap()
        .contains("1.250 bpp"));
    assert!(rows[0].diagnostics_unavailable.is_some());
    assert!(rows[0].artifact_dir.as_ref().unwrap().is_dir());
    assert!(rows[0].build.lockfile_sha256.is_none());

    // A clean case resumes: nothing re-encoded, no new row.
    let second = run_experiment(&options).unwrap();
    assert!(second.is_clean(), "{second:?}");
    assert_eq!((second.executed, second.resumed), (0, 1));
    assert_eq!(read_cases(&store, None).unwrap().len(), 1);
}

#[test]
fn a_failed_case_is_retried_on_resume_rather_than_skipped() {
    let tmp = Scratch::new("retry");
    let stub = compile_stub(&tmp);
    let input = tmp.path("encode-fail.raw");
    write_raw(&input);
    let cfg = config(vec![raw_input("bad", input)], vec![arm("baseline")], 1, 10);
    let out = tmp.path("out");
    let options = options(&tmp, &cfg, &stub, &out);

    let first = run_experiment(&options).unwrap();
    assert!(!first.is_clean());
    assert_eq!(first.statuses.failed, 1);
    // A non-success is not a completed case: the second run re-attempts it.
    let second = run_experiment(&options).unwrap();
    assert_eq!((second.executed, second.resumed), (1, 0));
    let rows = read_cases(&out.join("results.jsonl"), None).unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|row| row.status == CaseStatus::Failed));
    // The two attempts get distinct artifact directories: nothing is overwritten.
    let dirs: Vec<&PathBuf> = rows
        .iter()
        .filter_map(|row| row.artifact_dir.as_ref())
        .collect();
    assert_eq!(dirs.len(), 2);
    assert_ne!(dirs[0], dirs[1]);
}

#[test]
fn an_unsupported_arm_is_recorded_as_a_case_and_fails_the_run() {
    let tmp = Scratch::new("unsupported");
    let stub = compile_stub(&tmp);
    let input = tmp.path("color.png");
    let plane = Plane::from_vec(8, 8, (0..64u8).collect());
    mars_core::io::write_png(&input, &Image::rgb(plane.clone(), plane.clone(), plane)).unwrap();
    let mut method_arm = arm("fisher");
    method_arm.method = Some("fisher".into());
    let cfg = config(
        vec![InputSpec {
            name: "color".into(),
            file: input,
            width: None,
            height: None,
            sha256: None,
        }],
        vec![method_arm],
        1,
        10,
    );
    let out = tmp.path("out");
    let summary = run_experiment(&options(&tmp, &cfg, &stub, &out)).unwrap();
    assert!(!summary.is_clean());
    assert_eq!(summary.statuses.unsupported, 1);
    assert_eq!(summary.planned, 1);
    let rows = read_cases(&out.join("results.jsonl"), None).unwrap();
    assert_eq!(rows[0].status, CaseStatus::Unsupported);
    assert!(rows[0]
        .unsupported_reason
        .as_deref()
        .unwrap()
        .contains("grayscale"));
    assert!(rows[0].artifact_dir.is_none());
    assert!(!out.join("cases").exists());
}

#[test]
fn a_timed_out_codec_is_recorded_and_fails_the_run() {
    let tmp = Scratch::new("timeout");
    let stub = compile_stub(&tmp);
    let input = tmp.path("encode-timeout.raw");
    write_raw(&input);
    let cfg = config(vec![raw_input("slow", input)], vec![arm("baseline")], 1, 1);
    let out = tmp.path("out");
    let start = Instant::now();
    let summary = run_experiment(&options(&tmp, &cfg, &stub, &out)).unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(20),
        "timeout was not enforced"
    );
    assert!(!summary.is_clean());
    assert_eq!(summary.statuses.timed_out, 1);
    let rows = read_cases(&out.join("results.jsonl"), None).unwrap();
    assert_eq!(rows[0].status, CaseStatus::TimedOut);
    assert!(rows[0].error.as_deref().unwrap().contains("timeout"));
}

#[test]
fn missing_inputs_and_binaries_are_preflight_errors_not_partial_runs() {
    let tmp = Scratch::new("preflight");
    let stub = compile_stub(&tmp);

    // A referenced input that does not exist fails before anything is written.
    let cfg = config(
        vec![raw_input("missing", tmp.path("nope.raw"))],
        vec![arm("baseline")],
        1,
        10,
    );
    let out = tmp.path("out-missing");
    let error = run_experiment(&options(&tmp, &cfg, &stub, &out))
        .unwrap_err()
        .to_string();
    assert!(error.contains("nope.raw"), "{error}");
    assert!(!out.join("results.jsonl").exists());

    // A missing codec binary is likewise a preflight error.
    let input = tmp.path("good.raw");
    write_raw(&input);
    let cfg = config(vec![raw_input("good", input)], vec![arm("baseline")], 1, 10);
    let out = tmp.path("out-binary");
    let error = run_experiment(&options(&tmp, &cfg, &tmp.path("no-such-bin"), &out))
        .unwrap_err()
        .to_string();
    assert!(error.contains("encmars prerequisite"), "{error}");
    assert!(!out.join("results.jsonl").exists());
}

#[test]
fn many_repetitions_are_recorded_raw_and_must_be_byte_identical() {
    let tmp = Scratch::new("reps");
    let stub = compile_stub(&tmp);
    let input = tmp.path("a.raw");
    write_raw(&input);
    let cfg = config(vec![raw_input("a", input)], vec![arm("baseline")], 3, 10);
    let out = tmp.path("out");
    let summary = run_experiment(&options(&tmp, &cfg, &stub, &out)).unwrap();
    let rows = read_cases(&out.join("results.jsonl"), None).unwrap();
    assert!(summary.is_clean(), "{summary:?} rows={rows:#?}");
    assert_eq!(rows[0].repetitions.len(), 3);
    let first = &rows[0].repetitions[0].stream.sha256;
    assert!(rows[0]
        .repetitions
        .iter()
        .all(|r| &r.stream.sha256 == first));
    assert!(rows[0]
        .repetitions
        .iter()
        .all(|r| r.decode_iterations == 10));
}
