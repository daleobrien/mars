//! Bounded, corpus-free parameter and capability checks through the real CLI.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mars-cli-research-options-{label}-{}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create scratch directory");
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(path: &Path, rgb: bool) {
    // Keep 4:2:0 chroma at 8x8 too; the codec's 4x4 no-domain fallback currently
    // produces leaves that the decoder cannot reconstruct (outside CLI ownership).
    let size = if rgb { 16 } else { 8 };
    let mut bytes = format!("{}\n{size} {size}\n255\n", if rgb { "P6" } else { "P5" }).into_bytes();
    for y in 0..size {
        for x in 0..size {
            bytes.push((x * 23 + y * 11) as u8);
            if rgb {
                bytes.push((x * 7 + y * 19) as u8);
                bytes.push((x * 13 + y * 17) as u8);
            }
        }
    }
    std::fs::write(path, bytes).expect("write tiny PNM");
}

fn run(command: &mut Command) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn CLI");
    let start = Instant::now();
    loop {
        if child.try_wait().expect("poll CLI").is_some() {
            return child.wait_with_output().expect("collect CLI output");
        }
        if start.elapsed() > Duration::from_secs(10) {
            let _ = child.kill();
            let output = child.wait_with_output().expect("reap timed-out CLI");
            panic!(
                "{command:?} exceeded 10s: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn encode(input: &Path, output: &Path, args: &[&str]) -> Output {
    run(Command::new(env!("CARGO_BIN_EXE_encmars"))
        .arg(input)
        .arg(output)
        .args(["--threads", "1"])
        .args(args))
}

fn rejected(input: &Path, output: &Path, args: &[&str], message: &str) {
    let result = encode(input, output, args);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "accepted {args:?}");
    assert!(
        stderr.contains(message),
        "{args:?}: expected {message:?}: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "{args:?}: {stderr}");
    assert!(
        result.stdout.is_empty(),
        "invalid options must not report an encode"
    );
    assert!(!output.exists(), "invalid options must not create output");
}

fn accepted(input: &Path, output: &Path, args: &[&str]) -> Vec<u8> {
    let result = encode(input, output, args);
    assert!(
        result.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    std::fs::read(output).expect("read encoded stream")
}

#[test]
fn dangerous_numeric_options_fail_before_image_io() {
    let tmp = Scratch::new("numeric");
    let input = tmp.path("in.pgm");
    fixture(&input, false);
    let missing = tmp.path("missing.pgm");
    let output = tmp.path("out.mars");
    let mut cases: Vec<(Vec<String>, &str)> = Vec::new();
    for flag in ["--lambda", "--t-rms", "--chroma-t-rms"] {
        for value in ["NaN", "inf", "-inf", "-1"] {
            cases.push((vec![format!("{flag}={value}")], flag));
        }
    }
    for (flag, values) in [
        ("--min-size", &["0", "3", "256", "4294967295"][..]),
        ("--max-size", &["0", "3", "256", "4294967295"][..]),
        ("--shift", &["0", "1", "3", "255", "256", "4294967295"][..]),
        ("--bits-alfa", &["0", "1", "25", "32", "4294967295"][..]),
        ("--bits-beta", &["0", "25", "32", "4294967295"][..]),
        (
            "--max-alfa",
            &[
                "NaN", "inf", "-inf", "-1", "0", "0.001", "1.01", "8", "1e300",
            ][..],
        ),
    ] {
        for value in values {
            cases.push((vec![format!("{flag}={value}")], flag));
        }
    }
    cases.push((
        vec!["--min-size=16".into(), "--max-size=8".into()],
        "--min-size",
    ));
    for (args, message) in cases {
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        // A valid tiny image catches panics/hangs; a missing image proves validation
        // wins over I/O rather than merely failing later while reading the fixture.
        for path in [&input, &missing] {
            rejected(path, &output, &args, message);
        }
    }
}

#[test]
fn unsupported_research_options_are_errors_not_noops() {
    let tmp = Scratch::new("capabilities");
    let input = tmp.path("in.pgm");
    fixture(&input, false);
    let output = tmp.path("out.mars");
    for legacy in [
        vec!["--t-rms", "8"],
        vec!["--chroma-t-rms", "8"],
        vec!["--method", "exhaustive"],
        vec!["--progressive", "--t-rms", "8"],
    ] {
        for option in [vec!["--modes", "0,2"], vec!["--adaptive-density"]] {
            let mut args = legacy.clone();
            args.extend_from_slice(&option);
            rejected(&input, &output, &args, option[0]);
        }
    }
    for args in [
        vec!["--lambda", "50", "--adaptive-residual"],
        vec!["--lambda", "50", "--adaptive-residual", "--modes", "0,1,2"],
        vec!["--lambda", "50", "--adaptive-residual", "--progressive"],
    ] {
        rejected(&input, &output, &args, "requires mode 3");
    }
    for args in [
        vec!["--modes", "4"],
        vec!["--method", "fisher", "--modes", "4"],
        vec!["--t-rms", "8", "--modes", "255"],
        vec!["--modes="],
    ] {
        rejected(&input, &output, &args, "--modes");
    }
    rejected(
        &input,
        &output,
        &["--method", "fisher", "--lambda", "50"],
        "mutually exclusive",
    );
    rejected(
        &input,
        &output,
        &["--method", "fisher", "--progressive"],
        "mutually exclusive",
    );
}

#[test]
fn supported_profiles_still_encode_and_decode() {
    let tmp = Scratch::new("profiles");
    let gray = tmp.path("in.pgm");
    let rgb = tmp.path("in.ppm");
    fixture(&gray, false);
    fixture(&rgb, true);
    let profiles: &[(&Path, &[&str])] = &[
        (&gray, &[]),
        (&gray, &["--t-rms", "0"]),
        // Keep every indexed domain size within this tiny fixture. DomainPool currently
        // emits (0, 0) even when a domain cannot fit; that mars-search bug is separate.
        (&gray, &["--method", "exhaustive", "--max-size", "4"]),
        (&gray, &["--progressive"]),
        (&gray, &["--progressive", "--t-rms", "8"]),
        (&gray, &["--adaptive-density"]),
        (
            &gray,
            &["--lambda", "50", "--modes", "0,2,3", "--adaptive-residual"],
        ),
        (
            &gray,
            &[
                "--lambda",
                "50",
                "--modes",
                "3",
                "--adaptive-residual",
                "--progressive",
            ],
        ),
        (
            &rgb,
            &[
                "--subsampling",
                "420",
                "--lambda",
                "50",
                "--modes",
                "0,2,3",
                "--adaptive-residual",
            ],
        ),
        (&rgb, &["--t-rms", "0", "--chroma-t-rms", "16"]),
        (
            &gray,
            &[
                "--min-size",
                "1",
                "--max-size",
                "1",
                "--shift",
                "2",
                "--bits-alfa",
                "2",
                "--bits-beta",
                "1",
                "--max-alfa",
                "0.03125",
            ],
        ),
        (
            &gray,
            &[
                "--min-size",
                "2",
                "--max-size",
                "128",
                "--shift",
                "254",
                "--max-alfa",
                "7.96875",
            ],
        ),
    ];
    for (i, (input, args)) in profiles.iter().enumerate() {
        let output = tmp.path(&format!("{i}.mars"));
        assert!(!accepted(input, &output, args).is_empty());
        let decoded = tmp.path(&format!("{i}.png"));
        let result = run(Command::new(env!("CARGO_BIN_EXE_decmars"))
            .arg(&output)
            .arg(&decoded)
            .args(["--iterations", "1"]));
        assert!(
            result.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(decoded.exists());
    }
}

#[test]
fn explicit_lambda_ignores_thresholds_including_warmup() {
    let tmp = Scratch::new("threshold-precedence");
    for (label, rgb, progressive) in [
        ("gray", false, false),
        ("rgb", true, false),
        ("progressive", false, true),
    ] {
        let input = tmp.path(if rgb { "in.ppm" } else { "in.pgm" });
        fixture(&input, rgb);
        let mut args = vec!["--lambda", "50", "--modes", "0,2,3", "--adaptive-density"];
        if progressive {
            args.push("--progressive");
        }
        let baseline = accepted(&input, &tmp.path(&format!("{label}-base.mars")), &args);
        args.extend_from_slice(&["--t-rms", "0", "--chroma-t-rms", "1000000"]);
        let thresholds = accepted(
            &input,
            &tmp.path(&format!("{label}-thresholds.mars")),
            &args,
        );
        assert_eq!(
            baseline, thresholds,
            "explicit thresholds must not alter RD or its warm-up"
        );
    }
}

#[test]
fn help_describes_fixed_warmup_and_capabilities() {
    let output = run(Command::new(env!("CARGO_BIN_EXE_encmars")).arg("--help"));
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("fixed threshold of 8"));
    assert!(help.contains("Requires mode 3"));
    assert!(help.contains("explicit mask is rejected"));
    assert!(!help.contains("seeds the internal"));
}
