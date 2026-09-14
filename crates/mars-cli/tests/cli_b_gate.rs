//! `gate-cli-b` -- CLI-B of `encmars-decmars-cli-plan.md`: exposing Step 15's per-leaf
//! mode mask as `encmars --modes`. Both checks run the real `encmars` binary
//! (`CARGO_BIN_EXE_encmars`), fast (small synthetic image, one lambda point each).
//!
//! 1. `--modes 0,1,2,3` (every mode explicitly listed) must be byte-identical to omitting
//!    the flag entirely -- the "the whole point is additive, not a second code path"
//!    guarantee CLI-A's own default-off flag was held to.
//! 2. `--modes 2` (fractal-only) on a known image must match a hand-computed
//!    "fractal-only" leaf count. The `.mars` colour container's grayscale framing is
//!    fixed and small (`COLOR_HEADER_LEN` = magic(4) + version(1) + mode(1) +
//!    section_count(1), then one `u32` LE length prefix before the single plane stream --
//!    `color.rs`'s own `write_container`), so this test strips that framing itself and
//!    parses the plane stream with the public `mars_codec::mars_format::read` to inspect
//!    real `Leaf::mode` values, rather than trusting the CLI's own stdout summary.

use std::path::Path;
use std::process::Command;

fn encmars_bin() -> &'static str {
    env!("CARGO_BIN_EXE_encmars")
}

/// Strip `color.rs`'s grayscale `MARC` container framing (fixed 11-byte header for a
/// single-stream gray container: 4-byte magic + version + mode + section_count(=1) + a
/// 4-byte LE length prefix) and return the inner `mars_format` plane stream.
fn strip_gray_container(bytes: &[u8]) -> &[u8] {
    const COLOR_MAGIC: &[u8; 4] = b"MARC";
    assert_eq!(&bytes[0..4], COLOR_MAGIC, "not a MARC colour container");
    assert_eq!(bytes[4], 0, "unexpected colour container version");
    assert_eq!(
        bytes[5], 0,
        "expected MODE_GRAY (0) for a single-plane test image"
    );
    assert_eq!(
        bytes[6], 1,
        "expected exactly one section for a gray container"
    );
    let len = u32::from_le_bytes(bytes[7..11].try_into().unwrap()) as usize;
    let stream = &bytes[11..11 + len];
    assert_eq!(
        11 + len,
        bytes.len(),
        "trailing bytes after the one section"
    );
    stream
}

fn write_test_pgm(path: &Path) {
    // A uniformly textured synthetic image, no flat/near-flat regions anywhere --
    // deliberately avoiding any block whose best fractal fit collapses to `qalfa == 0`
    // (a genuinely domain-independent block, for which `best_mode_leaf` never generates a
    // mode-2/3 candidate at all regardless of `allowed_modes`, per its own doc; a flat
    // corner here would make this test's "every leaf is mode 2" expectation wrong for a
    // reason that has nothing to do with `--modes` wiring). Two overlapping frequencies
    // keep every block texture-rich so a domain match is always available.
    let (w, h) = (64usize, 64usize);
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

#[test]
fn all_modes_explicit_is_byte_identical_to_omitting_the_flag() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-b-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.pgm");
    write_test_pgm(&input);

    let out_omitted = tmp.join("omitted.mars");
    let status = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out_omitted)
        .args(["--lambda", "200"])
        .status()
        .expect("encmars runs");
    assert!(status.success());

    let out_explicit = tmp.join("explicit.mars");
    let status = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out_explicit)
        .args(["--lambda", "200"])
        .args(["--modes", "0,1,2,3"])
        .status()
        .expect("encmars runs");
    assert!(status.success());

    let bytes_omitted = std::fs::read(&out_omitted).unwrap();
    let bytes_explicit = std::fs::read(&out_explicit).unwrap();
    assert_eq!(
        bytes_omitted, bytes_explicit,
        "--modes 0,1,2,3 must be byte-identical to omitting --modes"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn modes_2_forces_fractal_only_matching_a_hand_computed_leaf_count() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-b-fractal-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.pgm");
    write_test_pgm(&input);

    let out_all = tmp.join("all.mars");
    let status = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out_all)
        .args(["--lambda", "200"])
        .status()
        .expect("encmars runs");
    assert!(status.success());

    let out_fractal_only = tmp.join("fractal-only.mars");
    let status = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out_fractal_only)
        .args(["--lambda", "200"])
        .args(["--modes", "2"])
        .status()
        .expect("encmars runs");
    assert!(status.success());

    let bytes_all = std::fs::read(&out_all).unwrap();
    let bytes_fractal_only = std::fs::read(&out_fractal_only).unwrap();

    let (_hdr_all, leaves_all) =
        mars_codec::mars_format::read(strip_gray_container(&bytes_all)).expect("valid stream");
    let (_hdr_fo, leaves_fractal_only) =
        mars_codec::mars_format::read(strip_gray_container(&bytes_fractal_only))
            .expect("valid stream");

    // Hand-computed expectation: with only mode 2 allowed, *every* leaf in the
    // fractal-only encode must report mode 2 -- there is no other mode `walk_rd` could
    // have picked.
    assert!(
        leaves_fractal_only.iter().all(|l| l.mode == 2),
        "with --modes 2, every leaf must be mode 2 (fractal); got modes: {:?}",
        leaves_fractal_only
            .iter()
            .map(|l| l.mode)
            .collect::<Vec<_>>()
    );
    assert!(
        !leaves_fractal_only.is_empty(),
        "fractal-only encode produced no leaves at all -- suspicious for a 64x64 image"
    );

    // Sanity: the unrestricted (all-modes) encode on the same image, at the same lambda,
    // must use at least one non-fractal mode -- otherwise this test's synthetic image
    // does not actually exercise the mask (it would pass trivially even with a wiring bug
    // that ignores --modes entirely).
    assert!(
        leaves_all.iter().any(|l| l.mode != 2),
        "expected the unrestricted encode to use at least one non-fractal mode on this \
         mixed-complexity image, or this test cannot distinguish a real mask from a no-op"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn out_of_range_mode_is_rejected() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-b-bad-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.pgm");
    write_test_pgm(&input);
    let out = tmp.join("out.mars");

    let output = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out)
        .args(["--modes", "7"])
        .output()
        .expect("encmars runs");
    assert!(!output.status.success(), "mode 7 should be rejected");

    let _ = std::fs::remove_dir_all(&tmp);
}
