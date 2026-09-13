//! Known-answer tests for the `.ifs` reader (Step 5), against the two worked examples in
//! `docs/mars1-format.md` (§11 `flat128`, §12 `tiny64`). The expected numbers here —
//! transform counts, leaf fields, decode hashes — are copied from `fixtures/mars1/
//! manifest.toml`, which the C encoder printed and `just gate-3` has already validated
//! independently in Python. This test exercises the *Rust port* of the same spec.

use mars_codec::ifs::{decode_iterative, parse};
use sha2::{Digest, Sha256};

const FLAT128: &str = "../../fixtures/mars1/flat128__default__fisher__r8.ifs";
const TINY64: &str = "../../fixtures/mars1/tiny64__walkthrough__fisher__r8.ifs";

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Byte-for-byte what `decmars` writes: `P5\n{w} {h}\n255\n` then the raster, no comments.
fn pgm_bytes(width: usize, height: usize, raster: &[u8]) -> Vec<u8> {
    let mut out = format!("P5\n{width} {height}\n255\n").into_bytes();
    out.extend_from_slice(raster);
    out
}

#[test]
fn flat128_matches_the_section_11_worked_example() {
    let data = std::fs::read(FLAT128).expect("golden fixture is committed");
    let (hdr, leaves) = parse(&data).expect("a golden fixture must parse");

    assert_eq!(hdr.bits_alfa, 4);
    assert_eq!(hdr.bits_beta, 7);
    assert_eq!(hdr.min_size, 4);
    assert_eq!(hdr.max_size, 16);
    assert_eq!(hdr.shift, 4);
    assert_eq!(hdr.width, 256);
    assert_eq!(hdr.height, 256);
    assert_eq!(hdr.max_alfa(), 1.0);

    // §11: every leaf is DC-only, on the exact 16x16 grid, qbeta = 64 everywhere.
    assert_eq!(leaves.len(), 256, "256 leaves, the plain 16x16 grid");
    assert!(leaves.iter().all(|l| l.dc_only()));
    assert!(leaves.iter().all(|l| l.size == 16));
    assert!(leaves.iter().all(|l| l.qbeta == 64));

    let decoded = decode_iterative(&hdr, &leaves, 10);
    // §11: every pixel reconstructs to 129, one grey level above the constant 128 input.
    assert!(decoded.as_slice().iter().all(|&p| p == 129));
    assert_eq!(
        sha256_hex(&pgm_bytes(256, 256, decoded.as_slice())),
        "672b2432384fed7fc1a6618e3a0df4ba2b1bb80bb824ebc18a5578f7f0988594",
        "must match decmars -i's own output exactly (fixtures/mars1/manifest.toml)"
    );
}

#[test]
fn tiny64_matches_the_section_12_walkthrough() {
    let data = std::fs::read(TINY64).expect("golden fixture is committed");
    let (hdr, leaves) = parse(&data).expect("a golden fixture must parse");

    assert_eq!(hdr.min_size, 16);
    assert_eq!(hdr.max_size, 32);
    assert_eq!(hdr.width, 64);
    assert_eq!(hdr.height, 64);
    assert_eq!(hdr.bits_coord_row(), 4);
    assert_eq!(hdr.bits_coord_col(), 4);

    assert_eq!(leaves.len(), 7, "7 transforms, per the walkthrough header");
    assert_eq!(
        leaves.iter().filter(|l| l.dc_only()).count(),
        3,
        "3 of them DC-only"
    );

    // §12's table, in parse order: (row, col, size, qalfa, qbeta, isometry, dom_row, dom_col).
    let expected = [
        (0, 0, 32, 0, 50, 0, 0, 0),
        (32, 0, 16, 8, 49, 0, 32, 0),
        (48, 0, 16, 8, 70, 0, 32, 0),
        (32, 16, 16, 8, 49, 0, 32, 0),
        (48, 16, 16, 8, 70, 0, 32, 0),
        (0, 32, 32, 0, 85, 0, 0, 0),
        (32, 32, 32, 0, 30, 0, 0, 0),
    ];
    assert_eq!(leaves.len(), expected.len());
    for (leaf, &(row, col, size, qalfa, qbeta, isometry, dom_row, dom_col)) in
        leaves.iter().zip(expected.iter())
    {
        assert_eq!(leaf.row, row);
        assert_eq!(leaf.col, col);
        assert_eq!(leaf.size, size);
        assert_eq!(leaf.qalfa, qalfa);
        assert_eq!(leaf.qbeta, qbeta);
        if leaf.qalfa != 0 {
            assert_eq!(leaf.isometry, isometry);
            assert_eq!(leaf.dom_row, dom_row);
            assert_eq!(leaf.dom_col, dom_col);
        }
    }

    let decoded = decode_iterative(&hdr, &leaves, 10);
    assert_eq!(
        sha256_hex(&pgm_bytes(64, 64, decoded.as_slice())),
        "341a36ae1b0938650ecee68024c750773534ce9f1db50bf6bc3b4c777ba23f44",
        "must match decmars -i's own output exactly (fixtures/mars1/manifest.toml)"
    );
}
