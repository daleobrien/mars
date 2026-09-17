//! Corpus-free failure preservation checks. Codec stubs are native Rust executables,
//! never shell scripts; the real codec round trip lives in mars-cli's integration test.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use mars_bench::experiment::{run_smoke, SmokeOptions};
use mars_bench::provenance::sha256_hex;
use serde_json::Value;

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mars-bench-smoke-{label}-{}", std::process::id()));
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

fn options(tmp: &Scratch, name: &str) -> SmokeOptions {
    let input = tmp.path(&format!("{name}.pgm"));
    let mut bytes = b"P5\n8 8\n255\n".to_vec();
    bytes.extend(0..64);
    fs::write(&input, bytes).unwrap();
    SmokeOptions {
        input,
        out_dir: tmp.path(name),
        encmars: tmp.path("missing-encoder"),
        decmars: tmp.path("missing-decoder"),
        raw_dims: None,
        lambda: 200.0,
        modes: vec![0, 2],
        iterations: 3,
        threads: 1,
        timeout_secs: 1,
    }
}

fn report(options: &SmokeOptions) -> Value {
    serde_json::from_slice(&fs::read(options.out_dir.join("report.json")).unwrap()).unwrap()
}

#[test]
fn missing_prerequisites_and_invalid_parameters_are_reported_without_skips() {
    let tmp = Scratch::new("prerequisites");
    for case in [
        "input",
        "encoder",
        "decoder",
        "lambda",
        "modes",
        "threads",
        "iterations",
        "timeout",
        "raw-dims",
        "ignored-dims",
    ] {
        let mut o = options(&tmp, case);
        match case {
            "input" => fs::remove_file(&o.input).unwrap(),
            "decoder" => o.encmars = std::env::current_exe().unwrap(),
            "lambda" => o.lambda = 201.0,
            "modes" => o.modes = vec![0, 1, 2],
            "threads" => o.threads = 0,
            "iterations" => o.iterations = 0,
            "timeout" => o.timeout_secs = 0,
            "raw-dims" => {
                let raw = o.input.with_extension("raw");
                fs::rename(&o.input, &raw).unwrap();
                o.input = raw;
            }
            "ignored-dims" => o.raw_dims = Some((8, 8)),
            _ => {}
        }
        assert!(run_smoke(&o).is_err(), "{case}");
        let r = report(&o);
        assert_eq!(r["status"], "failed", "{case}");
        assert!(r["error"].as_str().unwrap().len() > 5);
        assert_eq!(r["encode"]["status"], "blocked");
        assert_eq!(r["decode"]["status"], "blocked");
        assert!(!o.out_dir.join("coded.mars").exists());
        if case != "input" {
            assert!(r["input"]["sha256"].is_string());
        }
        if case == "decoder" {
            assert!(r["error"]
                .as_str()
                .unwrap()
                .contains("decmars prerequisite"));
        }
    }
}

#[test]
fn existing_directory_file_and_symlink_are_never_overwritten() {
    let tmp = Scratch::new("existing");
    let o = options(&tmp, "directory");
    fs::create_dir(&o.out_dir).unwrap();
    fs::write(o.out_dir.join("sentinel"), b"keep me").unwrap();
    assert!(run_smoke(&o).is_err());
    assert_eq!(fs::read(o.out_dir.join("sentinel")).unwrap(), b"keep me");
    assert!(!o.out_dir.join("report.json").exists());
    let o = options(&tmp, "file");
    fs::write(&o.out_dir, b"keep file").unwrap();
    assert!(run_smoke(&o).is_err());
    assert_eq!(fs::read(&o.out_dir).unwrap(), b"keep file");
    #[cfg(unix)]
    {
        let o = options(&tmp, "symlink");
        let target = tmp.path("nonexistent");
        std::os::unix::fs::symlink(&target, &o.out_dir).unwrap();
        assert!(run_smoke(&o).is_err());
        assert!(!target.exists());
    }
}

fn compile_stub(tmp: &Scratch) -> PathBuf {
    let source = tmp.path("stub.rs");
    let binary = tmp.path(&format!("stub{}", std::env::consts::EXE_SUFFIX));
    fs::write(&source, r#"
use std::{env, fs, thread, time::Duration};
fn main() {
    let args: Vec<_> = env::args().collect();
    let input = &args[1];
    let output = &args[2];
    let encode = output.ends_with("coded.mars");
    let label = if encode { input.clone() } else { String::from_utf8(fs::read(input).unwrap()).unwrap() };
    fs::write(output, if encode { format!("MARC{label}") } else { "partial PNG".into() }).unwrap();
    println!("stub output retained");
    eprintln!("stub diagnostic retained");
    if (encode && label.contains("encode-timeout")) || (!encode && label.contains("decode-timeout")) {
        thread::sleep(Duration::from_secs(60));
    }
    if (encode && label.contains("encode-fail")) || (!encode && label.contains("decode-fail")) {
        std::process::exit(7);
    }
}
"#).unwrap();
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
        if start.elapsed() > Duration::from_secs(20) {
            let _ = child.kill();
            let _ = child.wait();
            panic!("stub compilation exceeded 20s");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    binary
}

fn assert_identity(value: &Value, path: &Path) {
    let bytes = fs::read(path).unwrap();
    assert_eq!(value["bytes"], bytes.len() as u64);
    assert_eq!(value["sha256"], sha256_hex(&bytes));
}

#[test]
fn process_failure_timeout_and_metric_failure_preserve_report_logs_and_partial_files() {
    let tmp = Scratch::new("processes");
    let stub = compile_stub(&tmp);
    for case in [
        "encode-fail",
        "encode-timeout",
        "decode-fail",
        "decode-timeout",
        "metrics-fail",
    ] {
        let mut o = options(&tmp, case);
        o.encmars = stub.clone();
        o.decmars = stub.clone();
        let start = Instant::now();
        assert!(run_smoke(&o).is_err());
        assert!(start.elapsed() < Duration::from_secs(10), "{case}");
        let r = report(&o);
        assert_eq!(r["status"], "failed");
        let phase = if case.starts_with("encode") {
            "encode"
        } else {
            "decode"
        };
        let status = if case.contains("timeout") {
            "timed_out"
        } else if case == "metrics-fail" {
            "succeeded"
        } else {
            "failed"
        };
        assert_eq!(r[phase]["status"], status, "{r}");
        assert_eq!(r[phase]["reaped"], true);
        assert!(r[phase]["wall_secs"].as_f64().unwrap() > 0.0);
        if case.contains("timeout") {
            assert!(r[phase]["wall_secs"].as_f64().unwrap() >= 1.0);
        } else if case != "metrics-fail" {
            assert_eq!(r[phase]["exit_code"], 7);
        }
        assert!(
            fs::read_to_string(o.out_dir.join(format!("{phase}.stderr.log")))
                .unwrap()
                .contains("stub diagnostic retained")
        );
        assert_identity(&r["stream"], &o.out_dir.join("coded.mars"));
        assert_identity(&r["input"], &o.input);
        if phase == "decode" {
            assert_identity(&r["decoded"], &o.out_dir.join("decoded.png"));
        } else {
            assert_eq!(r["decode"]["status"], "blocked");
        }
        assert!(r["metrics"].is_null());
        assert!(!o.out_dir.join("report.json.tmp").exists());
        let saved = fs::read(o.out_dir.join("report.json")).unwrap();
        assert!(run_smoke(&o).is_err());
        assert_eq!(fs::read(o.out_dir.join("report.json")).unwrap(), saved);
    }
}

#[test]
fn non_executable_prerequisite_is_a_persisted_spawn_failure() {
    let tmp = Scratch::new("spawn");
    let mut o = options(&tmp, "spawn-fail");
    o.encmars = o.input.clone();
    o.decmars = std::env::current_exe().unwrap();
    assert!(run_smoke(&o).is_err());
    let r = report(&o);
    assert_eq!(r["encode"]["status"], "failed");
    assert_eq!(r["encode"]["reaped"], false);
    assert!(r["encode"]["error"].as_str().unwrap().contains("spawning"));
    assert_eq!(r["decode"]["status"], "blocked");
}
