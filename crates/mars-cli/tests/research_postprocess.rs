//! Real decmars subprocesses, tiny constructed streams, no encoder search or corpus.
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use mars_codec::color;
use mars_codec::ifs::{self, Leaf};
use mars_codec::mars_format::{self, Header};
use mars_codec::postprocess::smooth_boundaries;
use mars_codec::progressive;
use mars_core::io::read_pgm;
use mars_core::Plane;

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mars-postprocess-{label}-{}", std::process::id()));
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

fn fixture() -> (Header, Vec<Leaf>, Vec<u8>) {
    let header = Header::from(ifs::Header {
        bits_alfa: 4,
        bits_beta: 7,
        min_size: 4,
        max_size: 4,
        shift: 4,
        width: 8,
        height: 8,
        int_max_alfa: 32,
    });
    let leaves = [(0, 0), (4, 0), (0, 4), (4, 4)]
        .into_iter()
        .map(|(row, col)| Leaf {
            row,
            col,
            size: 4,
            mode: 0,
            qalfa: 0,
            qbeta: if (row + col) % 8 == 0 { 50 } else { 70 },
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
            qgx: 0,
            qgy: 0,
            residual: vec![],
        })
        .collect::<Vec<_>>();
    let stream = mars_format::write(&header, &leaves).unwrap();
    (header, leaves, stream)
}

fn run(input: &Path, output: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_decmars"))
        .arg(input)
        .arg(output)
        .args(args)
        .output()
        .unwrap()
}

fn accepted(input: &Path, output: &Path, args: &[&str]) -> Plane {
    let result = run(input, output, args);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&result.stdout).lines().count(), 1);
    assert!(result.stderr.is_empty());
    read_pgm(output).unwrap()
}

fn hash(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn smooth_gray_matches_api_changes_pixels_and_never_changes_stream() {
    let tmp = Scratch::new("gray");
    let (header, leaves, stream) = fixture();
    let bytes = color::wrap_gray_stream(stream);
    let input = tmp.path("gray.mars");
    let output = tmp.path("out.pgm");
    std::fs::write(&input, &bytes).unwrap();
    let before = hash(&std::fs::read(&input).unwrap());
    let plain = accepted(&input, &output, &[]);
    assert_eq!(plain, ifs::decode_iterative(&header, &leaves, 10));
    let geometry = leaves
        .iter()
        .map(|l| (l.row as usize, l.col as usize, l.size as usize))
        .collect::<Vec<_>>();
    let expected = smooth_boundaries(&plain, &geometry);
    let smoothed = accepted(&input, &output, &["--smooth"]);
    assert_ne!(plain, smoothed);
    assert_eq!(smoothed, expected);
    assert_eq!(accepted(&input, &output, &["--smooth"]), smoothed);
    assert_eq!(accepted(&input, &output, &["--smooth", "--auto"]), expected);

    // Reuse the decoder's zoom geometry, rather than rescaling a boundary bitmap.
    let (zoom_header, zoom_leaves) = ifs::zoom_leaves(&header, &leaves, 2.0);
    let zoom_geometry = zoom_leaves
        .iter()
        .map(|l| (l.row as usize, l.col as usize, l.size as usize))
        .collect::<Vec<_>>();
    assert_eq!(
        accepted(&input, &output, &["--smooth", "--zoom", "2"]),
        smooth_boundaries(
            &ifs::decode_iterative(&zoom_header, &zoom_leaves, 10),
            &zoom_geometry
        )
    );

    let frames = tmp.path("frames");
    assert_eq!(
        accepted(
            &input,
            &output,
            &[
                "--smooth",
                "--iterations",
                "2",
                "--progression",
                frames.to_str().unwrap()
            ]
        ),
        expected
    );
    assert_eq!(read_pgm(&frames.join("iter-2.pgm")).unwrap(), plain);
    let debug = tmp.path("rects.pgm");
    accepted(
        &input,
        &output,
        &["--smooth", "--debug-rects", debug.to_str().unwrap()],
    );
    assert_eq!(
        &read_pgm(&debug).unwrap(),
        &color::quadtree_image(&bytes).unwrap().planes()[0]
    );
    let after = std::fs::read(&input).unwrap();
    assert_eq!(hash(&after), before);
    assert_eq!(after, bytes); // Stronger than the hash alone.
}

fn rejected(input: &Path, output: &Path, bytes: &[u8], message: &str) {
    std::fs::write(input, bytes).unwrap();
    let before = hash(bytes);
    let result = run(input, output, &["--smooth"]);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success());
    assert!(
        stderr.contains("--smooth") && stderr.contains(message),
        "{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(result.stdout.is_empty());
    assert!(!output.exists());
    let after = std::fs::read(input).unwrap();
    assert_eq!(hash(&after), before);
    assert_eq!(after, bytes);
}

#[test]
fn progressive_and_both_color_modes_are_clearly_refused() {
    let tmp = Scratch::new("refusal");
    let (header, leaves, stream) = fixture();
    let source = ifs::decode_iterative(&header, &leaves, 10);
    let bytes = progressive::encode(&source, &header, &leaves).unwrap();
    let input = tmp.path("input.mars");
    let output = tmp.path("out.pgm");
    rejected(&input, &output, &bytes, "progressive");
    for mode in [1, 2] {
        let mut color_bytes = b"MARC\0".to_vec();
        color_bytes.extend([mode, 3]);
        for _ in 0..3 {
            color_bytes.extend((stream.len() as u32).to_le_bytes());
            color_bytes.extend(&stream);
        }
        // Demonstrate this is otherwise a decodable color container.
        assert_eq!(
            color::decode_color_image(&color_bytes, 1)
                .unwrap()
                .planes()
                .len(),
            3
        );
        rejected(&input, &output, &color_bytes, "color containers");
    }
}

#[test]
fn unknown_and_truncated_envelopes_refuse_without_guessing_boundaries() {
    let tmp = Scratch::new("envelope");
    let (_, _, stream) = fixture();
    let input = tmp.path("input.mars");
    let output = tmp.path("out.pgm");
    rejected(&input, &output, &stream, "cannot expose leaves");
    let mut bytes = color::wrap_gray_stream(stream);
    for end in [0, 5, 7, 10, bytes.len() - 1] {
        rejected(&input, &output, &bytes[..end], "--smooth");
    }
    bytes.push(0);
    rejected(&input, &output, &bytes, "trailing");
}
