//! Production search-provider CLI regressions: tiny fixtures, exact comparisons, no corpus.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use mars_codec::quant::ResidualQstep;
use mars_codec::{ifs, mars_format};
use mars_core::image::Image;
use mars_core::io::read_image;

const METHODS: [&str; 9] = [
    "exhaustive",
    "fisher",
    "hurtgen",
    "masscenter",
    "saupe",
    "saupe-fisher",
    "mc-saupe",
    "funnel",
    "learned",
];

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mars-production-search-{label}-{}",
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

fn fixture(path: &Path) {
    let mut bytes = b"P5\n16 16\n255\n".to_vec();
    for y in 0..16 {
        for x in 0..16 {
            let v = ((x * 7 + y * 11) % 256) + ((x * x + y) % 37) * 5;
            bytes.push((v % 256) as u8);
        }
    }
    std::fs::write(path, bytes).unwrap();
}

fn run_output(command: &mut Command) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn CLI");
    let start = Instant::now();
    loop {
        if child.try_wait().expect("poll CLI").is_some() {
            return child.wait_with_output().expect("collect output");
        }
        if start.elapsed() > Duration::from_secs(10) {
            let _ = child.kill();
            let output = child.wait_with_output().expect("reap CLI");
            panic!(
                "{command:?} exceeded 10s: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn run(command: &mut Command) -> Output {
    let output = run_output(command);
    assert!(
        output.status.success(),
        "{command:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn encode_output(input: &Path, output: &Path, threads: &str, args: &[&str]) -> Output {
    run_output(
        Command::new(env!("CARGO_BIN_EXE_encmars"))
            .arg(input)
            .arg(output)
            .args(["--min-size", "4", "--max-size", "8", "--threads", threads])
            .args(args),
    )
}

fn assert_mask_rejected(result: &Output, output: &Path, modes: &str) {
    assert_eq!(result.status.code(), Some(1), "must be a normal CLI error");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains(&format!("--modes {modes} cannot represent this image with selected search; enable mode 0 for DC fallback")), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(result.stdout.is_empty(), "must not report an encode");
    assert!(!output.exists(), "rejected mask must not create output");
}

fn encode(input: &Path, output: &Path, threads: &str, args: &[&str]) -> (Vec<u8>, String) {
    let result = encode_output(input, output, threads, args);
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

fn parse(bytes: &[u8]) -> (mars_format::Header, Vec<ifs::Leaf>) {
    assert_eq!(&bytes[..7], b"MARC\0\0\x01");
    let len = u32::from_le_bytes(bytes[7..11].try_into().unwrap()) as usize;
    assert_eq!(len + 11, bytes.len());
    mars_format::read(&bytes[11..]).expect("parse production stream")
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

#[test]
fn every_method_supports_rd_modes_and_is_thread_deterministic() {
    let tmp = Scratch::new("rd-modes");
    let input = tmp.path("in.pgm");
    fixture(&input);

    for method in METHODS {
        for modes in ["0", "1", "2", "3", "0,2", "0,1,2,3"] {
            let mut args = vec![
                "--method",
                method,
                "--lambda",
                "50",
                "--modes",
                modes,
                "--adaptive-density",
            ];
            if modes.contains('3') {
                args.push("--adaptive-residual");
            }
            let label = format!("{method}-{modes}");
            let output = tmp.path(&format!("{label}-1.mars"));
            let output_two = tmp.path(&format!("{label}-2.mars"));
            let first = encode_output(&input, &output, "1", &args);
            let second = encode_output(&input, &output_two, "2", &args);
            assert_eq!(
                first.status.success(),
                second.status.success(),
                "{label}: thread-dependent acceptance"
            );
            if !first.status.success() {
                assert!(
                    matches!(modes, "2" | "3"),
                    "{label}: required successful profile: {}",
                    String::from_utf8_lossy(&first.stderr)
                );
                assert_mask_rejected(&first, &output, modes);
                assert_mask_rejected(&second, &output_two, modes);
                assert_eq!(first.stderr, second.stderr);
                continue;
            }
            let one = std::fs::read(&output).unwrap();
            let two = std::fs::read(&output_two).unwrap();
            let stdout = String::from_utf8(first.stdout).unwrap();
            let stdout_two = String::from_utf8(second.stdout).unwrap();
            assert_eq!(one, two, "{label}: thread counts changed bytes");
            let search = counter(&stdout, "search_evals");
            let warmup = counter(&stdout, "warmup_evals");
            let total = counter(&stdout, "total_evals");
            assert!(warmup > 0, "{label}: must account for RD warm-up");
            assert_eq!(total, search + warmup);
            assert!(
                stdout.contains(&format!(" {search} evals,")),
                "historical evals is search-only"
            );
            for name in ["search_evals", "warmup_evals", "total_evals"] {
                assert_eq!(
                    counter(&stdout, name),
                    counter(&stdout_two, name),
                    "{label}: {name}"
                );
            }
            let (hdr, leaves) = parse(&one);
            assert!(!leaves.is_empty());
            for leaf in &leaves {
                assert!(
                    modes
                        .split(',')
                        .any(|m| m.parse::<u8>().unwrap() == leaf.mode),
                    "{label}: disallowed mode {}",
                    leaf.mode
                );
            }
            assert_eq!(
                hdr.residual_qstep,
                if modes.contains('3') {
                    ResidualQstep::from_lambda(50.0)
                } else {
                    ResidualQstep::LEGACY
                }
            );
            if modes == "3" {
                assert!(
                    leaves
                        .iter()
                        .any(|leaf| leaf.residual.iter().any(|&v| v != 0)),
                    "{label}: exercise residual reconstruction"
                );
            }
            let decoded = tmp.path(&format!("{label}.png"));
            run(Command::new(env!("CARGO_BIN_EXE_decmars"))
                .arg(&output)
                .arg(&decoded)
                .args(["--iterations", "2"]));
            assert_eq!(
                read_image(&decoded, None).unwrap(),
                Image::gray(ifs::decode_iterative(&hdr, &leaves, 2)),
                "{label}: serialized decode"
            );
        }
    }
}

#[test]
fn mandatory_dc_requires_explicit_permission_across_output_paths() {
    let tmp = Scratch::new("mandatory-dc");
    // 4x4 has no fitting 2:1 domain at min-size 4; 16x16 has domains but
    // constant fits have zero contrast, so neither can supply mode 2 or 3.
    for size in [4, 16] {
        let gray = tmp.path(&format!("constant-{size}.pgm"));
        let rgb = tmp.path(&format!("constant-{size}.ppm"));
        for (path, magic, channels) in [(&gray, "P5", 1), (&rgb, "P6", 3)] {
            let mut bytes = format!("{magic}\n{size} {size}\n255\n").into_bytes();
            bytes.extend(vec![128; size * size * channels]);
            std::fs::write(path, bytes).unwrap();
        }
        let profiles: &[(&str, &Path, &[&str])] = &[
            ("gray", &gray, &[]),
            ("exhaustive", &gray, &["--method", "exhaustive"]),
            ("fisher", &gray, &["--method", "fisher"]),
            ("progressive", &gray, &["--progressive"]),
            ("rgb444", &rgb, &["--subsampling", "444"]),
            ("rgb420", &rgb, &["--subsampling", "420"]),
        ];
        for (label, input, profile) in profiles {
            let mut baseline = None;
            for threads in ["1", "2"] {
                let mut args = vec!["--lambda", "50"];
                args.extend_from_slice(profile);
                let (default, _) = encode(
                    input,
                    &tmp.path(&format!("{size}-{label}-{threads}-default.mars")),
                    threads,
                    &args,
                );
                for modes in ["2", "3"] {
                    let mut strict = args.clone();
                    strict.extend_from_slice(&["--modes", modes]);
                    let output = tmp.path(&format!("{size}-{label}-{threads}-reject-{modes}.mars"));
                    let result = encode_output(input, &output, threads, &strict);
                    assert_mask_rejected(&result, &output, modes);
                }
                args.extend_from_slice(&["--modes", "0,2"]);
                let output = tmp.path(&format!("{size}-{label}-{threads}-allowed.mars"));
                let (allowed, _) = encode(input, &output, threads, &args);
                assert_eq!(
                    allowed, default,
                    "{label}: explicit permission must preserve default bytes"
                );
                if let Some(previous) = &baseline {
                    assert_eq!(previous, &allowed, "{label}: thread-dependent bytes");
                }
                baseline = Some(allowed.clone());
                if *label == "progressive" {
                    // No public progressive leaf reader: compare exact serialization of
                    // the production leaves after independently checking their modes.
                    let image = read_image(input, None).unwrap();
                    let params = mars_codec::encode::EncodeParams {
                        min_size: 4,
                        max_size: 8,
                        shift: 4,
                        bits_alfa: 4,
                        bits_beta: 7,
                        max_alfa: 1.0,
                        t_rms: 8.0,
                        zero_threshold: 0,
                        lambda: Some(50.0),
                    };
                    let options = mars_codec::encode::EncodeOptions {
                        allowed_modes: [true, false, true, false],
                        ..mars_codec::encode::EncodeOptions::default()
                    };
                    let (hdr, leaves, _, _) = mars_codec::encode::encode_image_with_options(
                        &image.planes()[0],
                        &params,
                        &options,
                    );
                    assert!(!leaves.is_empty());
                    assert!(leaves.iter().all(|leaf| leaf.mode == 0));
                    assert_eq!(
                        allowed,
                        mars_codec::progressive::encode(&image.planes()[0], &hdr, &leaves).unwrap()
                    );
                } else {
                    assert_eq!(&allowed[..4], b"MARC");
                    let mut remaining = &allowed[7..];
                    for _ in 0..allowed[6] {
                        let len = u32::from_le_bytes(remaining[..4].try_into().unwrap()) as usize;
                        let (_, leaves) = mars_format::read(&remaining[4..4 + len]).unwrap();
                        assert!(!leaves.is_empty());
                        assert!(
                            leaves.iter().all(|leaf| leaf.mode == 0),
                            "constant planes must use allowed DC fallback"
                        );
                        remaining = &remaining[4 + len..];
                    }
                    assert!(remaining.is_empty());
                }
                let decoded = tmp.path(&format!("{size}-{label}-{threads}.png"));
                run(Command::new(env!("CARGO_BIN_EXE_decmars"))
                    .arg(&output)
                    .arg(&decoded)
                    .args(["--iterations", "2"]));
                let decoded = read_image(&decoded, None).unwrap();
                assert_eq!((decoded.width(), decoded.height()), (size, size));
            }
        }
    }
}

#[test]
fn exhaustive_matches_no_method_bytes_for_identical_threshold_and_rd_profiles() {
    let tmp = Scratch::new("exhaustive-identity");
    let input = tmp.path("in.pgm");
    fixture(&input);
    let profiles: &[&[&str]] = &[
        &["--t-rms", "8"],
        &["--t-rms", "0"],
        &["--lambda", "200"],
        &["--lambda", "0", "--modes", "0,1,2,3"],
        &[
            "--lambda",
            "50",
            "--t-rms",
            "0",
            "--modes",
            "3",
            "--adaptive-density",
            "--adaptive-residual",
        ],
    ];
    for (i, profile) in profiles.iter().enumerate() {
        let (plain, _) = encode(&input, &tmp.path(&format!("plain-{i}.mars")), "1", profile);
        let mut args = profile.to_vec();
        args.extend_from_slice(&["--method", "exhaustive"]);
        let (method, _) = encode(&input, &tmp.path(&format!("method-{i}.mars")), "1", &args);
        assert_eq!(
            plain, method,
            "{profile:?}: exhaustive provider must be byte-identical"
        );
    }
    let (method_alone, stdout) = encode(
        &input,
        &tmp.path("method-alone.mars"),
        "1",
        &["--method", "exhaustive"],
    );
    let (threshold, _) = encode(&input, &tmp.path("threshold8.mars"), "1", &["--t-rms", "8"]);
    assert_eq!(
        method_alone, threshold,
        "method alone retains threshold 8, not default RD"
    );
    assert_eq!(counter(&stdout, "warmup_evals"), 0);
}
