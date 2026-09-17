//! Fast, corpus-free residual-qstep regressions through the real CLI binaries.

use std::path::{Path, PathBuf};
use std::process::Command;

use mars_codec::color::upsample_nearest;
use mars_codec::ifs::{self, Leaf};
use mars_codec::mars_format::{self, Header};
use mars_codec::progressive;
use mars_codec::quant::ResidualQstep;
use mars_core::image::Image;
use mars_core::io::read_image;
use mars_core::metrics::rgb_from_ycbcr;

const SIZE: usize = 32;
const ITERATIONS: u32 = 10;

struct Scratch(PathBuf);

impl Scratch {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "mars-cli-residual-qstep-{label}-{}",
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

fn write_texture(path: &Path, rgb: bool) {
    let mut bytes = format!("{}\n{SIZE} {SIZE}\n255\n", if rgb { "P6" } else { "P5" }).into_bytes();
    for y in 0..SIZE {
        for x in 0..SIZE {
            // Match the overlapping-frequency texture used by cli_b_gate, not a flat
            // fixture that could pass without ever reconstructing a residual.
            let v = ((x * 7 + y * 11) % 256) + ((x * x + y) % 37) * 5;
            bytes.push((v % 256) as u8);
            if rgb {
                bytes.push(((v + x * 13 + y * 3) % 256) as u8);
                bytes.push(((v + x * 3 + y * 17) % 256) as u8);
            }
        }
    }
    std::fs::write(path, bytes).expect("write textured PNM");
}

fn run(cmd: &mut Command) {
    let output = cmd.output().expect("run CLI binary");
    assert!(
        output.status.success(),
        "{cmd:?} failed ({}):\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn encode(input: &Path, output: &Path, extra: &[&str]) -> Vec<u8> {
    run(Command::new(env!("CARGO_BIN_EXE_encmars"))
        .arg(input)
        .arg(output)
        .args([
            "--lambda",
            "50",
            "--modes",
            "3",
            "--min-size",
            "4",
            "--max-size",
            "4",
            "--threads",
            "1",
        ])
        .args(extra));
    std::fs::read(output).expect("read encoded stream")
}

fn decode(input: &Path, output: &Path, extra: &[&str]) -> Image {
    run(Command::new(env!("CARGO_BIN_EXE_decmars"))
        .arg(input)
        .arg(output)
        .args(["--iterations", &ITERATIONS.to_string()])
        .args(extra));
    read_image(output, None).expect("read decoded PNG pixels")
}

fn parse_planes(bytes: &[u8], mode: u8, count: u8) -> Vec<(Header, Vec<Leaf>)> {
    assert_eq!(&bytes[..7], &[b'M', b'A', b'R', b'C', 0, mode, count]);
    let mut offset = 7;
    let mut planes = Vec::new();
    for _ in 0..count {
        let len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let stream = &bytes[offset..offset + len];
        assert_eq!(&stream[..4], b"MARS");
        planes.push(mars_format::read(stream).expect("parse nested MARS plane"));
        offset += len;
    }
    assert_eq!(offset, bytes.len(), "trailing MARC data");
    planes
}

fn assert_nonzero_residual(leaves: &[Leaf]) {
    assert!(
        leaves
            .iter()
            .any(|leaf| leaf.mode == 3 && leaf.residual.iter().any(|&v| v != 0)),
        "fixture must exercise nonzero mode-3 residual reconstruction"
    );
}

#[test]
fn gray_default_is_fixed8_and_adaptive_decode_uses_serialized_step() {
    let tmp = Scratch::new("gray");
    let input = tmp.path("in.pgm");
    write_texture(&input, false);

    for (label, flags, step) in [
        ("fixed", &[][..], ResidualQstep::LEGACY),
        (
            "adaptive",
            &["--adaptive-residual"][..],
            ResidualQstep::from_lambda(50.0),
        ),
    ] {
        let encoded = tmp.path(&format!("{label}.mars"));
        let bytes = encode(&input, &encoded, flags);
        let planes = parse_planes(&bytes, 0, 1);
        let (hdr, leaves) = &planes[0];
        assert_eq!(hdr.residual_qstep, step);
        assert_nonzero_residual(leaves);
        let expected = ifs::decode_iterative(hdr, leaves, ITERATIONS);
        let actual = decode(&encoded, &tmp.path(&format!("{label}.png")), &[]);
        assert_eq!(actual, Image::gray(expected.clone()));
        if label == "adaptive" {
            assert_ne!(step.get(), 8.0);
            assert_ne!(
                expected,
                ifs::decode_iterative(&hdr.geometry, leaves, ITERATIONS),
                "discarding the serialized qstep must change these pixels"
            );
        } else {
            assert_eq!(step.get(), 8.0);
        }
    }
}

#[test]
fn adaptive_progressive_prefixes_match_layer_flags_and_full_ifs_decode() {
    let tmp = Scratch::new("progressive");
    let input = tmp.path("in.pgm");
    write_texture(&input, false);
    let single = encode(&input, &tmp.path("single.mars"), &["--adaptive-residual"]);
    let planes = parse_planes(&single, 0, 1);
    let (hdr, leaves) = &planes[0];
    assert_eq!(hdr.residual_qstep, ResidualQstep::from_lambda(50.0));
    assert_nonzero_residual(leaves);
    let expected = Image::gray(ifs::decode_iterative(hdr, leaves, ITERATIONS));

    let encoded = tmp.path("progressive.mars");
    let bytes = encode(&input, &encoded, &["--adaptive-residual", "--progressive"]);
    assert!(progressive::is_progressive(&bytes));
    // The progressive header stores the same wire-rounded f32 after its layer lengths.
    assert_eq!(&bytes[31..35], &(hdr.residual_qstep.get() as f32).to_le_bytes());
    let offsets = progressive::layer_end_offsets(&bytes).expect("parse layer offsets");
    assert_eq!(offsets[3], bytes.len());
    for (i, end) in offsets.into_iter().enumerate() {
        let layer = (i + 1).to_string();
        let prefix = &bytes[..end];
        let parsed = progressive::decode(prefix).expect("decode progressive prefix");
        assert_eq!(usize::from(parsed.layers), i + 1);
        let truncated = tmp.path(&format!("prefix-{layer}.mars"));
        std::fs::write(&truncated, prefix).unwrap();
        let via_prefix = decode(&truncated, &tmp.path(&format!("prefix-{layer}.png")), &[]);
        let via_flag = decode(
            &encoded,
            &tmp.path(&format!("layer-{layer}.png")),
            &["--layer", &layer],
        );
        assert_eq!(via_prefix, Image::gray(parsed.image));
        assert_eq!(via_flag, via_prefix, "layer {layer}");
        if i == 3 {
            assert_eq!(via_flag, expected);
        }
    }
    assert_eq!(decode(&encoded, &tmp.path("full.png"), &[]), expected);
}

#[test]
fn adaptive_rgb420_preserves_each_plane_step_end_to_end() {
    let tmp = Scratch::new("rgb420");
    let input = tmp.path("in.ppm");
    write_texture(&input, true);
    let encoded = tmp.path("rgb.mars");
    let bytes = encode(
        &input,
        &encoded,
        &["--adaptive-residual", "--subsampling", "420"],
    );
    let planes = parse_planes(&bytes, 2, 3);
    let mut decoded = Vec::new();
    for (i, (hdr, leaves)) in planes.iter().enumerate() {
        assert_eq!(hdr.residual_qstep, ResidualQstep::from_lambda(50.0));
        let size = if i == 0 { SIZE } else { SIZE / 2 } as u32;
        assert_eq!((hdr.width, hdr.height), (size, size));
        assert_nonzero_residual(leaves);
        decoded.push(ifs::decode_iterative(hdr, leaves, ITERATIONS));
    }
    let expected = rgb_from_ycbcr(
        &decoded[0],
        &upsample_nearest(&decoded[1], SIZE, SIZE),
        &upsample_nearest(&decoded[2], SIZE, SIZE),
    );
    assert_eq!(decode(&encoded, &tmp.path("rgb.png"), &[]), expected);
}

fn assert_rejected(input: &Path, output: &Path, args: &[&str], messages: &[&str]) {
    let result = Command::new(env!("CARGO_BIN_EXE_encmars"))
        .arg(input)
        .arg(output)
        .args(args)
        .output()
        .expect("encmars runs");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "{args:?} unexpectedly accepted");
    for message in messages {
        assert!(
            stderr.contains(message),
            "{args:?}: expected {message:?} in {stderr:?}"
        );
    }
    assert!(
        !output.exists(),
        "invalid arguments must not produce a stream"
    );
}

#[test]
fn nonfinite_lambda_is_rejected_with_and_without_adaptive_residual() {
    let tmp = Scratch::new("invalid-lambda");
    let input = tmp.path("in.pgm");
    write_texture(&input, false);
    for lambda in ["NaN", "inf", "-inf"] {
        for adaptive in [false, true] {
            let lambda_arg = format!("--lambda={lambda}");
            let mut args = vec![lambda_arg.as_str()];
            if adaptive {
                args.push("--adaptive-residual");
            }
            assert_rejected(
                &input,
                &tmp.path("invalid.mars"),
                &args,
                &["--lambda", "finite"],
            );
        }
    }
}

#[test]
fn adaptive_residual_requires_lambda_and_conflicts_with_method() {
    let tmp = Scratch::new("invalid-combinations");
    let input = tmp.path("in.pgm");
    write_texture(&input, false);
    assert_rejected(
        &input,
        &tmp.path("missing-lambda.mars"),
        &["--adaptive-residual"],
        &["required", "--lambda"],
    );
    assert_rejected(
        &input,
        &tmp.path("method.mars"),
        &[
            "--adaptive-residual",
            "--lambda",
            "50",
            "--method",
            "exhaustive",
        ],
        &["cannot be used with", "--adaptive-residual", "--method"],
    );
}
