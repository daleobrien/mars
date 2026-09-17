//! Bounded CLI coverage for opt-in random production search; no corpus or benchmarks.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use mars_codec::{ifs, mars_format};
use mars_core::{image::Image, io::read_image};

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mars-cli-random-{label}-{}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
    fn texture(&self) -> PathBuf {
        let path = self.path("texture.pgm");
        let mut bytes = b"P5\n16 16\n255\n".to_vec();
        for y in 0..16 {
            for x in 0..16 {
                bytes.push((((x * 7 + y * 11) % 256 + ((x * x + y) % 37) * 5) % 256) as u8);
            }
        }
        std::fs::write(&path, bytes).unwrap();
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
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if start.elapsed() > Duration::from_secs(10) {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
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
        .args(["--min-size", "4", "--max-size", "8", "--threads", threads])
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
        std::fs::read(output).unwrap(),
        String::from_utf8(result.stdout).unwrap(),
    )
}

fn rejected(input: &Path, output: &Path, args: &[&str], message: &str) {
    let result = encode(input, output, "1", args);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "accepted {args:?}");
    assert!(stderr.contains(message), "{args:?}: {stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(result.stdout.is_empty());
    assert!(!output.exists(), "failed request must not create output");
}

fn counter(stdout: &str, key: &str) -> u64 {
    stdout
        .split_once(&format!("{key}="))
        .unwrap()
        .1
        .split(',')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn decoded_matches_stream(input: &Path, output: &Path, bytes: &[u8]) {
    assert_eq!(&bytes[..7], b"MARC\0\0\x01");
    let length = u32::from_le_bytes(bytes[7..11].try_into().unwrap()) as usize;
    assert_eq!(length + 11, bytes.len());
    let (header, leaves) = mars_format::read(&bytes[11..]).unwrap();
    let result = run(Command::new(env!("CARGO_BIN_EXE_decmars"))
        .arg(input)
        .arg(output)
        .args(["--iterations", "2"]));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        read_image(output, None).unwrap(),
        Image::gray(ifs::decode_iterative(&header, &leaves, 2))
    );
}

#[test]
fn random_options_require_explicit_positive_budget_and_random_method() {
    let tmp = Scratch::new("validation");
    let input = tmp.texture();
    let output = tmp.path("invalid.mars");
    for args in [
        vec!["--method", "random"],
        vec!["--method", "random", "--seed", "0"],
        vec!["--method", "random", "--budget", "0"],
    ] {
        for path in [&input, &tmp.path("missing.pgm")] {
            rejected(
                path,
                &output,
                &args,
                "requires an explicit positive --budget",
            );
        }
    }
    for args in [
        vec!["--budget", "8"],
        vec!["--method", "exhaustive", "--budget", "8"],
    ] {
        rejected(
            &input,
            &output,
            &args,
            "--budget is only valid with --method random",
        );
    }
    for args in [
        vec!["--seed", "0"],
        vec!["--method", "fisher", "--seed", "1"],
    ] {
        rejected(
            &input,
            &output,
            &args,
            "--seed is only valid with --method random",
        );
    }
    rejected(
        &input,
        &output,
        &["--method", "random", "--budget", "8", "--progressive"],
        "mutually exclusive",
    );
    rejected(
        &input,
        &output,
        &["--method", "random", "--budget", "8", "--modes", "0,2"],
        "requires the RD path",
    );
    let rgb = tmp.path("rgb.ppm");
    let mut bytes = b"P6\n4 4\n255\n".to_vec();
    bytes.extend([128; 48]);
    std::fs::write(&rgb, bytes).unwrap();
    rejected(
        &rgb,
        &output,
        &["--method", "random", "--budget", "8"],
        "only supports grayscale",
    );
}

#[test]
fn seeded_search_is_reproducible_across_threads_and_reports_warmup_separately() {
    let tmp = Scratch::new("reproducibility");
    let input = tmp.texture();
    for (label, profile) in [
        ("threshold", &[][..]),
        (
            "rd",
            &[
                "--lambda",
                "50",
                "--modes",
                "0,2,3",
                "--adaptive-density",
                "--adaptive-residual",
            ][..],
        ),
    ] {
        let mut args = vec![
            "--method",
            "random",
            "--budget",
            "8",
            "--seed",
            "18446744073709551615",
        ];
        args.extend_from_slice(profile);
        let out = tmp.path(&format!("{label}-one.mars"));
        let (one, stdout) = accepted(&input, &out, "1", &args);
        let (repeat, _) = accepted(
            &input,
            &tmp.path(&format!("{label}-repeat.mars")),
            "1",
            &args,
        );
        let (two, stdout_two) =
            accepted(&input, &tmp.path(&format!("{label}-two.mars")), "2", &args);
        assert_eq!(one, repeat);
        assert_eq!(one, two);
        assert!(stdout.contains("method random,"));
        assert!(stdout.contains("budget=8,"));
        assert!(stdout.contains("seed=18446744073709551615,"));
        assert!(stdout.contains(&format!(
            "rng_version={},",
            mars_search::random::RNG_VERSION
        )));
        let search = counter(&stdout, "search_evals");
        let warmup = counter(&stdout, "warmup_evals");
        assert!(search > 0);
        // At most 4 size-8 + 16 size-4 queries in this geometry, with eight
        // isometry fits per sampled position. Warm-up is counted separately.
        assert!(
            search <= 20 * 8 * 8,
            "production fit budget exceeded: {search}"
        );
        assert_eq!(counter(&stdout, "total_evals"), search + warmup);
        assert!(stdout.contains(&format!(" {search} evals,")));
        if label == "rd" {
            assert!(warmup > search);
        } else {
            assert_eq!(warmup, 0);
        }
        for key in ["search_evals", "warmup_evals", "total_evals"] {
            assert_eq!(counter(&stdout, key), counter(&stdout_two, key));
        }
        decoded_matches_stream(&out, &tmp.path(&format!("{label}.png")), &one);
    }
    let base = ["--method", "random", "--budget", "8"];
    let (implicit, _) = accepted(&input, &tmp.path("implicit-seed.mars"), "1", &base);
    let mut explicit = base.to_vec();
    explicit.extend_from_slice(&["--seed", "0"]);
    let (zero, _) = accepted(&input, &tmp.path("zero-seed.mars"), "1", &explicit);
    assert_eq!(implicit, zero, "omitted seed must be zero");
    explicit.extend_from_slice(&["--t-rms", "8"]);
    let (threshold8, _) = accepted(&input, &tmp.path("threshold8.mars"), "1", &explicit);
    assert_eq!(implicit, threshold8, "method alone must retain threshold 8");
}

#[test]
fn different_seeds_change_serialized_results_on_texture() {
    let tmp = Scratch::new("different-seeds");
    let input = tmp.texture();
    let mut streams = Vec::new();
    for seed in ["0", "1", "2", "3"] {
        let out = tmp.path(&format!("seed-{seed}.mars"));
        let (bytes, _) = accepted(
            &input,
            &out,
            "1",
            &[
                "--method", "random", "--budget", "8", "--seed", seed, "--lambda", "0", "--modes",
                "0,2,3",
            ],
        );
        decoded_matches_stream(&out, &tmp.path(&format!("seed-{seed}.png")), &bytes);
        streams.push(bytes);
    }
    assert!(
        streams[1..].iter().any(|bytes| bytes != &streams[0]),
        "seed must affect serialized search results, not just reporting"
    );
}

#[test]
fn full_budget_is_byte_identical_to_exhaustive_for_legacy_rd_density_and_residual() {
    let tmp = Scratch::new("full-budget");
    let input = tmp.texture();
    // Largest pool: 3x3 positions at size 4 / stride 4. Budget 9 covers all
    // positions, each evaluated at all eight isometries.
    let profiles: &[&[&str]] = &[
        &[],
        &["--t-rms", "0"],
        &["--lambda", "200"],
        &["--lambda", "50", "--adaptive-density"],
        &[
            "--lambda",
            "50",
            "--modes",
            "0,1,2,3",
            "--adaptive-residual",
        ],
        &[
            "--lambda",
            "0",
            "--t-rms",
            "0",
            "--modes",
            "0,2,3",
            "--adaptive-density",
            "--adaptive-residual",
        ],
    ];
    for (i, profile) in profiles.iter().enumerate() {
        let mut args = vec!["--method", "exhaustive"];
        args.extend_from_slice(profile);
        let (exhaustive, stats) = accepted(
            &input,
            &tmp.path(&format!("exhaustive-{i}.mars")),
            "1",
            &args,
        );
        let mut args = vec!["--method", "random", "--budget", "9", "--seed", "17"];
        args.extend_from_slice(profile);
        let output = tmp.path(&format!("random-{i}.mars"));
        let (random, random_stats) = accepted(&input, &output, "2", &args);
        assert_eq!(random, exhaustive, "full budget: {profile:?}");
        for key in ["search_evals", "warmup_evals", "total_evals"] {
            assert_eq!(
                counter(&stats, key),
                counter(&random_stats, key),
                "{profile:?}: {key}"
            );
        }
        decoded_matches_stream(&output, &tmp.path(&format!("full-{i}.png")), &random);
    }
}
