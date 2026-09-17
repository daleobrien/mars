//! Bounded CLI coverage for the opt-in APCC production search provider; no corpus.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use mars_codec::quant::ResidualQstep;
use mars_codec::{ifs, mars_format};
use mars_core::image::Image;
use mars_core::io::read_image;

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mars-cli-apcc-{label}-{}", std::process::id()));
        std::fs::create_dir(&path).expect("create scratch directory");
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// Upscaled overlapping frequencies retain texture while giving tiny-image
    /// ranges matching APCC buckets (the unscaled texture has none at this size).
    fn textured(&self) -> PathBuf {
        let path = self.path("textured.pgm");
        let mut bytes = b"P5\n16 16\n255\n".to_vec();
        for y in 0..16 {
            for x in 0..16 {
                let (x, y) = (x / 2, y / 2);
                let v = ((x * 7 + y * 11) % 256) + ((x * x + y) % 37) * 5;
                bytes.push((v % 256) as u8);
            }
        }
        std::fs::write(&path, bytes).expect("write textured fixture");
        path
    }

    /// An asymmetric linear ramp: contraction preserves quadrant ordering and equal
    /// quadrant variances, so the single domain and all ranges share a bucket.
    /// Different x/y slopes avoid isometry ties for the best positive-contrast fit.
    fn single_domain(&self) -> PathBuf {
        let path = self.path("single-domain.pgm");
        let mut bytes = b"P5\n8 8\n255\n".to_vec();
        for y in 0..8 {
            for x in 0..8 {
                bytes.push((x * 13 + y * 17) as u8);
            }
        }
        std::fs::write(&path, bytes).expect("write single-domain fixture");
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
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
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn encode(input: &Path, output: &Path, threads: &str, args: &[&str]) -> Output {
    run(Command::new(env!("CARGO_BIN_EXE_encmars"))
        .arg(input)
        .arg(output)
        .args(["--threads", threads])
        .args(args))
}

fn accepted(input: &Path, output: &Path, threads: &str, args: &[&str]) -> (Vec<u8>, String) {
    let result = encode(input, output, threads, args);
    assert!(
        result.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    (
        std::fs::read(output).expect("read encoded stream"),
        String::from_utf8(result.stdout).expect("stdout"),
    )
}

fn rejected(input: &Path, output: &Path, args: &[&str], message: &str) {
    let result = encode(input, output, "1", args);
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

fn counter(stdout: &str, name: &str) -> u64 {
    stdout
        .split_once(&format!("{name}="))
        .expect("counter label")
        .1
        .split(',')
        .next()
        .unwrap()
        .parse()
        .expect("integer counter")
}

/// Parse the gray MARC container this CLI writes and decode it two ways: the real
/// decmars binary, and the library's own iterative decoder on the parsed leaves.
fn assert_roundtrip(output: &Path, bytes: &[u8], iterations: u32) {
    assert_eq!(&bytes[..7], b"MARC\0\0\x01", "gray container framing");
    let length = u32::from_le_bytes(bytes[7..11].try_into().unwrap()) as usize;
    assert_eq!(length + 11, bytes.len(), "trailing MARC data");
    let (header, leaves) = mars_format::read(&bytes[11..]).expect("parse plane stream");
    assert!(!leaves.is_empty());
    let decoded = output.with_extension("png");
    let result = run(Command::new(env!("CARGO_BIN_EXE_decmars"))
        .arg(output)
        .arg(&decoded)
        .args(["--iterations", &iterations.to_string()]));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        read_image(&decoded, None).expect("read decoded PNG"),
        Image::gray(ifs::decode_iterative(&header, &leaves, iterations)),
        "decmars must reproduce the library decode of the serialized leaves"
    );
}

#[test]
fn apcc_budget_and_seed_options_are_validated_before_image_io() {
    let tmp = Scratch::new("options");
    let input = tmp.textured();
    let output = tmp.path("invalid.mars");
    let missing = tmp.path("missing.pgm");
    for args in [
        vec!["--method", "apcc"],
        vec!["--method", "apcc", "--seed", "1"],
        vec!["--method", "apcc", "--budget", "0"],
    ] {
        // The missing input proves validation wins over I/O rather than failing later.
        for path in [&input, &missing] {
            rejected(
                path,
                &output,
                &args,
                "--method apcc requires an explicit positive --budget",
            );
        }
    }
    for args in [
        vec!["--budget", "8"],
        vec!["--method", "exhaustive", "--budget", "8"],
        vec!["--method", "fisher", "--budget", "8"],
    ] {
        rejected(
            &input,
            &output,
            &args,
            "--budget is only valid with --method random or apcc",
        );
    }
    rejected(
        &input,
        &output,
        &["--method", "apcc", "--budget", "8", "--seed", "5"],
        "--seed is only valid with --method random",
    );
    rejected(
        &input,
        &output,
        &["--method", "apcc", "--budget", "8", "--progressive"],
        "mutually exclusive",
    );
    let rgb = tmp.path("rgb.ppm");
    let mut bytes = b"P6\n4 4\n255\n".to_vec();
    bytes.extend(std::iter::repeat_n(128u8, 48));
    std::fs::write(&rgb, bytes).expect("write rgb fixture");
    rejected(
        &rgb,
        &output,
        &["--method", "apcc", "--budget", "8"],
        "only supports grayscale",
    );
}

#[test]
fn apcc_encodes_decodes_is_thread_deterministic_and_reports_budget() {
    let tmp = Scratch::new("roundtrip");
    let input = tmp.textured();
    let args = [
        "--lambda", "200", "--modes", "0,2", "--budget", "4", "--method", "apcc",
    ];
    let output = tmp.path("one.mars");
    let (one, stdout) = accepted(&input, &output, "1", &args);
    let (two, stdout_two) = accepted(&input, &tmp.path("two.mars"), "2", &args);
    assert_eq!(one, two, "thread count changed bytes");
    assert!(stdout.contains("method apcc,"), "{stdout}");
    assert!(stdout.contains("budget=4,"), "{stdout}");
    let search = counter(&stdout, "search_evals");
    let warmup = counter(&stdout, "warmup_evals");
    assert!(search > 0);
    assert!(warmup > 0, "explicit lambda runs the RD warm-up");
    assert_eq!(counter(&stdout, "total_evals"), search + warmup);
    assert!(
        stdout.contains(&format!(" {search} evals,")),
        "historical evals is search-only"
    );
    // 16x16 with max-size 8 visits at most 4 size-8 + 16 size-4 queries; each budgeted
    // position costs eight isometry fits, and the exhaustive warm-up is counted apart.
    assert!(search <= 20 * 8 * 4, "production budget exceeded: {search}");
    for name in ["search_evals", "warmup_evals", "total_evals"] {
        assert_eq!(counter(&stdout, name), counter(&stdout_two, name), "{name}");
    }
    assert_roundtrip(&output, &one, 2);
}

#[test]
fn different_apcc_budgets_produce_different_serialized_output() {
    let tmp = Scratch::new("budgets");
    let input = tmp.textured();
    let mut streams = Vec::new();
    for budget in ["1", "9"] {
        let output = tmp.path(&format!("budget-{budget}.mars"));
        let (bytes, _) = accepted(
            &input,
            &output,
            "1",
            &[
                "--lambda", "200", "--modes", "0,2", "--budget", budget, "--method", "apcc",
            ],
        );
        assert_roundtrip(&output, &bytes, 2);
        streams.push(bytes);
    }
    assert_ne!(
        streams[0], streams[1],
        "a one-position bucket sample must not equal the full nine-position pool scan"
    );
}

#[test]
fn full_budget_matches_exhaustive_when_the_bucket_covers_the_whole_pool() {
    let tmp = Scratch::new("equivalence");
    let input = tmp.single_domain();
    // Pool: one 4x4 domain at shift 4 on an 8x8 image, so budget 1 >= pool size and
    // the matching bucket is the whole pool for every range.
    let geometry = ["--min-size", "4", "--max-size", "4", "--shift", "4"];
    for (label, profile) in [
        ("legacy", vec![]),
        ("rd", vec!["--lambda", "200", "--modes", "0,2"]),
    ] {
        let mut apcc_args = vec!["--method", "apcc", "--budget", "1"];
        apcc_args.extend_from_slice(&geometry);
        apcc_args.extend_from_slice(&profile);
        let mut exhaustive_args = vec!["--method", "exhaustive"];
        exhaustive_args.extend_from_slice(&geometry);
        exhaustive_args.extend_from_slice(&profile);
        let output = tmp.path(&format!("apcc-{label}.mars"));
        let (apcc, stats) = accepted(&input, &output, "1", &apcc_args);
        assert_eq!(
            counter(&stats, "search_evals"),
            4 * 8,
            "every range must fit the single matching domain"
        );
        let (exhaustive, _) = accepted(
            &input,
            &tmp.path(&format!("exhaustive-{label}.mars")),
            "1",
            &exhaustive_args,
        );
        assert_eq!(
            apcc, exhaustive,
            "{label}: full-budget APCC must equal exhaustive"
        );
        assert_roundtrip(&output, &apcc, 2);
    }
    let (_, stdout) = accepted(
        &input,
        &tmp.path("legacy-check.mars"),
        "1",
        &[
            "--method",
            "apcc",
            "--budget",
            "1",
            "--min-size",
            "4",
            "--max-size",
            "4",
            "--shift",
            "4",
        ],
    );
    assert_eq!(
        counter(&stdout, "warmup_evals"),
        0,
        "method alone has no RD warm-up"
    );
}

#[test]
fn apcc_roundtrips_with_density_mode3_and_adaptive_residual() {
    let tmp = Scratch::new("density-residual");
    let input = tmp.textured();
    let args = [
        "--lambda",
        "50",
        "--modes",
        "0,2,3",
        "--adaptive-density",
        "--adaptive-residual",
        "--budget",
        "4",
        "--method",
        "apcc",
    ];
    let output = tmp.path("apcc.mars");
    let (one, stdout) = accepted(&input, &output, "1", &args);
    let (two, stdout_two) = accepted(&input, &tmp.path("two.mars"), "2", &args);
    assert_eq!(one, two, "thread count changed bytes");
    assert_eq!(
        counter(&stdout, "search_evals"),
        counter(&stdout_two, "search_evals")
    );
    assert_eq!(&one[..7], b"MARC\0\0\x01");
    let length = u32::from_le_bytes(one[7..11].try_into().unwrap()) as usize;
    let (header, leaves) = mars_format::read(&one[11..11 + length]).expect("parse plane stream");
    assert_eq!(header.residual_qstep, ResidualQstep::from_lambda(50.0));
    assert!(
        leaves
            .iter()
            .any(|leaf| leaf.mode == 3 && leaf.residual.iter().any(|&v| v != 0)),
        "fixture must exercise nonzero mode-3 residual reconstruction"
    );
    assert_roundtrip(&output, &one, 2);
}
