//! `gate-cli-e` -- CLI-E of `encmars-decmars-cli-plan.md`: `encmars --progressive` /
//! `decmars --layer`. Grayscale-only this first cut, per the plan's own scope-cut clause
//! (colour composition with `mars_codec::color`'s YCbCr/subsampling wrapping is
//! unresolved and explicitly deferred -- see `encmars --help`'s `--progressive` text).
//!
//! Exit criterion, per the plan: `decmars --layer N` on a full progressive file, for each
//! `N`, must be byte-identical to `decmars` run on a file truncated to that layer's own
//! end offset -- P19.1's "every prefix decodes" property, exercised through the real
//! binaries rather than only through `progressive_gate.rs`'s internal test.

use std::path::Path;
use std::process::Command;

fn encmars_bin() -> &'static str {
    env!("CARGO_BIN_EXE_encmars")
}
fn decmars_bin() -> &'static str {
    env!("CARGO_BIN_EXE_decmars")
}

fn write_test_pgm(path: &Path) {
    let (w, h) = (48usize, 48usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = ((x * 7 + y * 11) % 256) + (((x * x + y) % 37) * 5);
            data[y * w + x] = (v % 256) as u8;
        }
    }
    let mut pgm = format!("P5\n{w} {h}\n255\n").into_bytes();
    pgm.extend_from_slice(&data);
    std::fs::write(path, pgm).expect("write test pgm");
}

/// Byte offset of " layer end-offsets [a, b, c, d]" in `encmars --progressive`'s stdout,
/// parsed rather than recomputed independently -- this test cross-checks the CLI against
/// itself (prefix decode vs. `--layer`), not against a third implementation of the offset
/// math.
fn parse_offsets(stdout: &str) -> [usize; 4] {
    let marker = "layer end-offsets [";
    let start = stdout.find(marker).unwrap_or_else(|| {
        panic!("encmars --progressive stdout has no {marker:?} to parse: {stdout:?}")
    }) + marker.len();
    let end = stdout[start..].find(']').unwrap() + start;
    let nums: Vec<usize> = stdout[start..end]
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();
    [nums[0], nums[1], nums[2], nums[3]]
}

#[test]
fn every_layer_prefix_matches_decoding_a_truncated_file() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-e-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.pgm");
    write_test_pgm(&input);

    let progressive = tmp.join("prog.mars");
    let output = Command::new(encmars_bin())
        .arg(&input)
        .arg(&progressive)
        .args(["--progressive", "--lambda", "200"])
        .output()
        .expect("encmars runs");
    assert!(
        output.status.success(),
        "encmars --progressive failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let offsets = parse_offsets(&String::from_utf8_lossy(&output.stdout));

    let full_bytes = std::fs::read(&progressive).unwrap();

    for (i, &off) in offsets.iter().enumerate() {
        let layer = i + 1;

        let via_layer_flag = tmp.join(format!("via-layer-{layer}.png"));
        let status = Command::new(decmars_bin())
            .arg(&progressive)
            .arg(&via_layer_flag)
            .args(["--layer", &layer.to_string()])
            .status()
            .expect("decmars runs");
        assert!(status.success(), "decmars --layer {layer} failed");

        let truncated_path = tmp.join(format!("truncated-{layer}.mars"));
        std::fs::write(&truncated_path, &full_bytes[..off]).unwrap();
        let via_truncation = tmp.join(format!("via-truncation-{layer}.png"));
        let status = Command::new(decmars_bin())
            .arg(&truncated_path)
            .arg(&via_truncation)
            .status()
            .expect("decmars runs");
        assert!(
            status.success(),
            "decmars on a file truncated to layer {layer}'s own end offset failed"
        );

        assert_eq!(
            std::fs::read(&via_layer_flag).unwrap(),
            std::fs::read(&via_truncation).unwrap(),
            "layer {layer}: --layer {layer} must be byte-identical to decoding a file \
             truncated to that layer's own end offset (P19.1's own property)"
        );
    }

    // Omitting --layer decodes every layer present -- byte-identical to --layer 4 (all
    // layers) on this full, untruncated file.
    let via_omitted = tmp.join("via-omitted.png");
    let status = Command::new(decmars_bin())
        .arg(&progressive)
        .arg(&via_omitted)
        .status()
        .expect("decmars runs");
    assert!(status.success());
    let via_layer_4 = tmp.join("via-layer-4.png");
    assert_eq!(
        std::fs::read(&via_omitted).unwrap(),
        std::fs::read(&via_layer_4).unwrap(),
        "omitting --layer should be byte-identical to --layer 4 on a full stream"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn layer_on_a_non_progressive_file_is_refused() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-e-nonprog-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.pgm");
    write_test_pgm(&input);
    let out = tmp.join("out.mars");
    let status = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out)
        .status()
        .expect("encmars runs");
    assert!(status.success());

    let decoded = tmp.join("out.png");
    let output = Command::new(decmars_bin())
        .arg(&out)
        .arg(&decoded)
        .args(["--layer", "2"])
        .output()
        .expect("decmars runs");
    assert!(
        !output.status.success(),
        "--layer on a single-layer container should be refused"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn progressive_on_colour_input_is_refused() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-e-rgb-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.ppm");
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h * 3];
    for i in 0..w * h {
        data[i * 3] = (i % 256) as u8;
        data[i * 3 + 1] = ((i * 2) % 256) as u8;
        data[i * 3 + 2] = ((i * 3) % 256) as u8;
    }
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    ppm.extend_from_slice(&data);
    std::fs::write(&input, ppm).expect("write test ppm");
    let out = tmp.join("out.mars");

    let output = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out)
        .arg("--progressive")
        .output()
        .expect("encmars runs");
    assert!(
        !output.status.success(),
        "--progressive on colour input should be refused (grayscale-only for now)"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
