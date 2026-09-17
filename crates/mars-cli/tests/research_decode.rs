//! Tiny deterministic decode fixtures; no encoder search or corpus required.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use mars_codec::color;
use mars_codec::ifs::{self, Leaf};
use mars_codec::mars_format::{self, Header};
use mars_codec::progressive;
use mars_codec::quant::ResidualQstep;
use mars_core::io::{read_image, read_pgm};
use mars_core::{Plane, image::Image};

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mars-research-decode-{label}-{}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
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

fn fixture() -> (Vec<u8>, Vec<u8>, Header, Vec<Leaf>) {
    let hdr = Header {
        geometry: ifs::Header {
            bits_alfa: 4,
            bits_beta: 7,
            min_size: 4,
            max_size: 4,
            shift: 4,
            width: 8,
            height: 8,
            int_max_alfa: 32,
        },
        residual_qstep: ResidualQstep::LEGACY,
    };
    let leaves = [(0, 0), (4, 0), (0, 4), (4, 4)]
        .into_iter()
        .enumerate()
        .map(|(i, (row, col))| {
            let mut residual = vec![0; 16];
            residual[0] = 4;
            residual[1] = -2;
            Leaf {
                row,
                col,
                size: 4,
                mode: 3,
                qalfa: 8,
                qbeta: 45 + i as u32 * 8,
                isometry: 0,
                dom_row: 0,
                dom_col: 0,
                qgx: 0,
                qgy: 0,
                residual,
            }
        })
        .collect::<Vec<_>>();
    let source = Plane::from_vec(8, 8, (0..64).map(|i| (i * 37 % 256) as u8).collect());
    let progressive = progressive::encode(&source, &hdr, &leaves).unwrap();
    let regular = color::wrap_gray_stream(mars_format::write(&hdr, &leaves).unwrap());
    (progressive, regular, hdr, leaves)
}

fn decode(input: &Path, output: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_decmars"))
        .arg(input)
        .arg(output)
        .args(args)
        .output()
        .expect("run decmars")
}
fn read_output(path: &Path) -> Image {
    // write_pnm selects P5/P6 by plane count, not by filename extension.
    if path.extension().is_some_and(|ext| ext == "png") {
        read_image(path, None).unwrap()
    } else {
        Image::gray(read_pgm(path).unwrap())
    }
}

fn accepted(input: &Path, output: &Path, args: &[&str]) -> Image {
    let result = decode(input, output, args);
    assert!(
        result.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    read_output(output)
}
fn rejected(input: &Path, output: &Path, args: &[&str], message: &str) {
    let result = decode(input, output, args);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "accepted {args:?}");
    assert!(stderr.contains(message), "expected {message:?}: {stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(
        result.stdout.is_empty(),
        "must not report successful decode"
    );
}

#[test]
fn progressive_iterations_match_api_for_all_layers_and_ifs_for_full_stream() {
    let tmp = Scratch::new("iterations");
    let (bytes, regular, hdr, leaves) = fixture();
    let input = tmp.path("progressive.mars");
    let output = tmp.path("out.pgm");
    std::fs::write(&input, &bytes).unwrap();
    let offsets = progressive::layer_end_offsets(&bytes).unwrap();
    for (i, end) in offsets.into_iter().enumerate() {
        let layer = (i + 1).to_string();
        let mut images = Vec::new();
        for iterations in [1, 10] {
            let count = iterations.to_string();
            let image = accepted(
                &input,
                &output,
                &["--layer", &layer, "--iterations", &count],
            );
            assert_eq!(
                image,
                Image::gray(
                    progressive::decode_with_iterations(&bytes[..end], iterations)
                        .unwrap()
                        .image
                )
            );
            if i == 3 {
                assert_eq!(
                    image,
                    Image::gray(ifs::decode_iterative(&hdr, &leaves, iterations))
                );
                assert_eq!(image, accepted(&input, &output, &["--iterations", &count]));
            }
            images.push(image);
        }
        if i >= 2 {
            assert_ne!(images[0], images[1]);
        }
    }
    let expected = Image::gray(ifs::decode_iterative(&hdr, &leaves, 10));
    assert_eq!(accepted(&input, &output, &[]), expected);
    std::fs::write(&input, regular).unwrap();
    assert_eq!(accepted(&input, &output, &[]), expected);
}

#[test]
fn shortened_progressive_files_report_unavailable_layers() {
    let tmp = Scratch::new("prefixes");
    let (bytes, _, _, _) = fixture();
    let input = tmp.path("prefix.mars");
    let output = tmp.path("out.pgm");
    let offsets = progressive::layer_end_offsets(&bytes).unwrap();
    for i in 0..3 {
        for end in [offsets[i], offsets[i + 1] - 1] {
            std::fs::write(&input, &bytes[..end]).unwrap();
            let layer = (i + 1).to_string();
            assert_eq!(
                accepted(&input, &output, &[]),
                accepted(&input, &output, &["--layer", &layer])
            );
            std::fs::remove_file(&output).unwrap();
            for unavailable in i + 2..=4 {
                rejected(
                    &input,
                    &output,
                    &["--layer", &unavailable.to_string()],
                    "is unavailable",
                );
                assert!(!output.exists());
            }
        }
    }
}

#[test]
fn invalid_options_fail_before_input_io() {
    let tmp = Scratch::new("validation");
    let input = tmp.path("missing.mars");
    let output = tmp.path("out.pgm");
    rejected(&input, &output, &["--iterations=0"], "--iterations");
    for value in ["0", "-1", "NaN", "inf", "-inf"] {
        rejected(
            &input,
            &output,
            &[&format!("--zoom={value}")],
            "--zoom must be finite and positive",
        );
    }
    for threshold in ["0", "1"] {
        rejected(&input, &output, &["--threshold", threshold], "--auto");
    }
    assert!(!output.exists());
}

#[test]
fn progression_write_failures_are_fatal_and_successful_frames_match_final() {
    let tmp = Scratch::new("frames");
    let (_, regular, _, _) = fixture();
    let input = tmp.path("regular.mars");
    std::fs::write(&input, regular).unwrap();
    for ext in ["png", "pgm", "ppm"] {
        for auto in [false, true] {
            let dir = tmp.path(&format!("frames-{ext}-{auto}"));
            std::fs::create_dir(&dir).unwrap();
            let output = tmp.path(&format!("out-{ext}-{auto}.{ext}"));
            let mut args = vec!["--iterations", "2", "--progression", dir.to_str().unwrap()];
            if auto {
                args.extend(["--auto", "--threshold", "0"]);
            }
            // A directory at the second frame path fails portably, even when run as root.
            let blocked = dir.join(format!("iter-2.{ext}"));
            std::fs::create_dir(&blocked).unwrap();
            rejected(&input, &output, &args, "writing progression frame");
            assert!(dir.join(format!("iter-1.{ext}")).exists());
            assert!(!output.exists());
            std::fs::remove_dir(&blocked).unwrap();
            let final_image = accepted(&input, &output, &args);
            assert_eq!(read_output(&blocked), final_image);
        }
    }
}

#[test]
fn regular_debug_rects_use_helper_at_native_resolution_and_propagate_errors() {
    let tmp = Scratch::new("debug");
    let (_, regular, _, _) = fixture();
    let input = tmp.path("regular.mars");
    let output = tmp.path("out.pgm");
    std::fs::write(&input, &regular).unwrap();
    let expected = color::quadtree_image(&regular).unwrap();
    for ext in ["png", "pgm", "ppm"] {
        let debug = tmp.path(&format!("rects.{ext}"));
        let image = accepted(
            &input,
            &output,
            &["--zoom", "2", "--debug-rects", debug.to_str().unwrap()],
        );
        assert_eq!(image.width(), 16);
        assert_eq!(read_output(&debug), expected);
    }
    std::fs::remove_file(&output).unwrap();
    let blocked = tmp.path("blocked.png");
    std::fs::create_dir(&blocked).unwrap();
    rejected(
        &input,
        &output,
        &["--debug-rects", blocked.to_str().unwrap()],
        "writing",
    );
    assert!(!output.exists());
}

#[test]
fn final_output_failures_and_unsupported_progressive_options_are_errors() {
    let tmp = Scratch::new("outputs");
    let (progressive, regular, _, _) = fixture();
    let input = tmp.path("in.mars");
    let output = tmp.path("blocked.png");
    std::fs::create_dir(&output).unwrap();
    for bytes in [&regular, &progressive] {
        std::fs::write(&input, bytes).unwrap();
        rejected(&input, &output, &[], "writing");
    }
    for args in [
        vec!["--auto"],
        vec!["--zoom", "2"],
        vec!["--progression", "unused"],
        vec!["--debug-rects", "unused.png"],
    ] {
        rejected(&input, &output, &args, "not supported");
    }
}
