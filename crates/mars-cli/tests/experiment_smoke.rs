//! Actual marsbench -> encmars -> MARC file -> decmars -> PNG -> file metrics.
//! No corpus, opt-in skips, shell commands, or library encode/decode shortcuts.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use mars_bench::measure::{measure, MeasureRequest};
use mars_bench::provenance::sha256_hex;
use mars_core::image::{Image, Plane};
use serde_json::Value;

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mars-cli-smoke-{label}-{}", std::process::id()));
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
        if start.elapsed() > Duration::from_secs(20) {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "CLI exceeded 20s: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn actual_file_round_trips_raw_pgm_and_rgb_png_with_exact_artifact_accounting() {
    let tmp = Scratch::new("roundtrip");
    let pixels: Vec<u8> = (0..256)
        .map(|n| ((n * 23 + n / 16 * 11) % 256) as u8)
        .collect();
    for format in ["raw", "pgm", "png"] {
        // Spaces and shell metacharacters remain literal path arguments.
        let input = tmp.path(&format!("input ; literal.{format}"));
        match format {
            "raw" => fs::write(&input, &pixels).unwrap(),
            "pgm" => {
                let mut bytes = b"P5\n16 16\n255\n".to_vec();
                bytes.extend(&pixels);
                fs::write(&input, bytes).unwrap();
            }
            _ => {
                let plane = Plane::from_vec(16, 16, pixels.clone());
                let image = Image::rgb(plane.clone(), plane.clone(), plane);
                mars_core::io::write_png(&input, &image).unwrap();
            }
        }
        let out_dir = tmp.path(&format!("artifacts {format}"));
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_marsbench"));
        cmd.arg("experiment-smoke")
            .arg(&input)
            .arg("--out-dir")
            .arg(&out_dir)
            .args([
                "--lambda",
                "200",
                "--modes",
                "0,2",
                "--iterations",
                "3",
                "--threads",
                "1",
                "--timeout-secs",
                "10",
            ]);
        if format == "raw" {
            cmd.args(["--raw-dims", "16x16"]);
        }
        // Also exercise default sibling discovery, not PATH lookup.
        if format != "pgm" {
            cmd.arg("--encmars")
                .arg(env!("CARGO_BIN_EXE_encmars"))
                .arg("--decmars")
                .arg(env!("CARGO_BIN_EXE_decmars"));
        }
        let result = run(&mut cmd);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("partial P0 only"));
        let report_path = out_dir.join("report.json");
        let saved = fs::read(&report_path).unwrap();
        let r: Value = serde_json::from_slice(&saved).unwrap();
        assert_eq!(r["runner"], "experiment-smoke");
        assert_eq!(r["status"], "succeeded");
        assert_eq!(r["options"]["iterations"], 3);
        assert_eq!(r["options"]["modes"], serde_json::json!([0, 2]));
        assert_eq!(r["options"]["lambda"], 200.0);
        assert_eq!(r["options"]["threads"], 1);
        assert_eq!(r["options"]["timeout_secs"], 10);
        for phase in ["encode", "decode"] {
            assert_eq!(r[phase]["status"], "succeeded");
            assert_eq!(r[phase]["exit_code"], 0);
            assert_eq!(r[phase]["reaped"], true);
            assert!(r[phase]["wall_secs"].as_f64().unwrap() > 0.0);
            assert_eq!(r[phase]["env"]["RAYON_NUM_THREADS"], "1");
            assert!(out_dir.join(format!("{phase}.stdout.log")).is_file());
            assert!(out_dir.join(format!("{phase}.stderr.log")).is_file());
        }
        let stream = out_dir.join("coded.mars");
        let decoded = out_dir.join("decoded.png");
        assert!(fs::read(&stream).unwrap().starts_with(b"MARC"));
        assert!(fs::read(&decoded)
            .unwrap()
            .starts_with(b"\x89PNG\r\n\x1a\n"));
        for (key, path) in [
            ("input", &input),
            ("stream", &stream),
            ("decoded", &decoded),
        ] {
            let bytes = fs::read(path).unwrap();
            assert_eq!(r[key]["bytes"], bytes.len() as u64);
            assert_eq!(r[key]["sha256"], sha256_hex(&bytes));
        }
        assert!(r["encoder_binary"]["sha256"].is_string());
        assert!(r["decoder_binary"]["sha256"].is_string());
        assert!(r["provenance"]["parameter_set_sha256"].is_string());
        let mut request = MeasureRequest::new(&input, &decoded)
            .with_coded_bytes(fs::metadata(&stream).unwrap().len());
        if format == "raw" {
            request.raw_dims = Some((16, 16));
        }
        // Compare through the same JSON round trip: serde_json's default float
        // parser can differ by one ULP from the original in-memory f64.
        let expected: Value =
            serde_json::from_slice(&serde_json::to_vec(&measure(&request).unwrap()).unwrap())
                .unwrap();
        assert_eq!(r["metrics"], expected);
        assert_eq!(r["metrics"]["planes"], if format == "png" { 3 } else { 1 });
        assert!(r["metrics"]["ms_ssim_unavailable"].is_string());
        let args = r["encode"]["args"].as_array().unwrap();
        assert_eq!(args[0], input.to_str().unwrap());
        assert_eq!(args[1], stream.to_str().unwrap());
        assert!(args
            .windows(2)
            .any(|w| w == [serde_json::json!("--lambda"), serde_json::json!("200")]));
        assert_eq!(
            r["decode"]["args"],
            serde_json::json!([stream, decoded, "--iterations", "3", "--zoom", "1"])
        );
        assert!(!run(&mut cmd).status.success());
        assert_eq!(
            fs::read(&report_path).unwrap(),
            saved,
            "rerun must not overwrite"
        );
    }
}

#[test]
fn cli_missing_binary_fails_and_points_to_persisted_report() {
    let tmp = Scratch::new("missing");
    let input = tmp.path("input.raw");
    fs::write(&input, [128; 64]).unwrap();
    let out = tmp.path("artifacts");
    let result = run(Command::new(env!("CARGO_BIN_EXE_marsbench"))
        .arg("experiment-smoke")
        .arg(&input)
        .arg("--out-dir")
        .arg(&out)
        .args(["--raw-dims", "8x8"])
        .arg("--encmars")
        .arg(tmp.path("missing"))
        .arg("--decmars")
        .arg(env!("CARGO_BIN_EXE_decmars")));
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("report retained at"));
    let r: Value = serde_json::from_slice(&fs::read(out.join("report.json")).unwrap()).unwrap();
    assert_eq!(r["status"], "failed");
    assert_eq!(r["encode"]["status"], "blocked");
    assert_eq!(r["decode"]["status"], "blocked");
}
