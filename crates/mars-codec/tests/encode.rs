//! Integration test for the exhaustive encoder (Step 6): encode a fixture image, write it,
//! parse the result back, and check it against the same closed-form worked example
//! `docs/mars1-format.md` §11 already gives for `flat128` under the 1998 defaults.

use mars_codec::encode::{encode_image, EncodeParams};
use mars_codec::ifs::{decode_iterative, parse, write};
use mars_core::io::read_raw;

const FLAT128: &str = "../../fixtures/images/flat128.raw";
const MIXED_129X127: &str = "../../fixtures/images/mixed_129x127.raw";

fn params() -> EncodeParams {
    EncodeParams {
        min_size: 4,
        max_size: 16,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda: None,
    }
}

#[test]
fn flat128_encodes_to_the_section_11_partition() {
    let image = read_raw(std::path::Path::new(FLAT128), 256, 256).expect("fixture is committed");
    let (hdr, leaves, evals) = encode_image(&image, &params());

    // §11: a constant image never needs to split — every leaf is the plain 16x16 grid,
    // DC-only, qbeta = 64.
    assert_eq!(leaves.len(), 256);
    assert!(leaves.iter().all(|l| l.dc_only()));
    assert!(leaves.iter().all(|l| l.size == 16));
    assert!(leaves.iter().all(|l| l.qbeta == 64));
    assert!(
        evals > 0,
        "the search ran, even though every leaf ended up DC-only"
    );

    // The writer round-trips: what we wrote parses back to the same tree.
    let bytes = write(&hdr, &leaves).expect("a tiled leaf set must serialise");
    let (hdr2, leaves2) = parse(&bytes).expect("the encoder's own output must be parseable");
    assert_eq!(hdr2, hdr);
    assert_eq!(leaves2, leaves);

    // §11: every pixel reconstructs to 129, one grey level above the constant 128 input.
    let decoded = decode_iterative(&hdr, &leaves, 10);
    assert!(decoded.as_slice().iter().all(|&p| p == 129));
}

/// `docs/mars1-format.md` §5.3: on a non-multiple-of-`min_size` image, the forced
/// subdivision branch drives leaves below `min_size`, down to size 1, as **pure
/// geometry** — independent of the search method or the RMS threshold. The exhaustive
/// encoder must reproduce exactly the same size-1/size-2 counts the 1998 encoder does
/// (`fixtures/mars1/manifest.toml`'s `mixed_129x127__default__*__r8` fixtures, all three
/// methods), which is also the one case that exercises the `tip == 0` raw-pixel special
/// case (§5.3) instead of the ordinary search.
#[test]
fn mixed_129x127_reproduces_the_forced_subdivision_geometry() {
    let image =
        read_raw(std::path::Path::new(MIXED_129X127), 129, 127).expect("fixture is committed");
    let (hdr, leaves, _evals) = encode_image(&image, &params());

    assert_eq!(leaves.iter().filter(|l| l.size == 1).count(), 255);
    assert_eq!(leaves.iter().filter(|l| l.size == 2).count(), 64);

    // A size-1 leaf's qbeta is the raw pixel, truncated to bits_beta (7) bits — never the
    // §8.1 mean-refit value, which would require a domain search that size 1 never runs.
    for leaf in leaves.iter().filter(|l| l.size == 1) {
        assert_eq!(leaf.qalfa, 0);
        assert!(leaf.qbeta < 128, "7-bit field");
    }

    let bytes = write(&hdr, &leaves).expect("a tiled leaf set must serialise");
    let (hdr2, leaves2) = parse(&bytes).expect("the encoder's own output must be parseable");
    assert_eq!(hdr2, hdr);
    assert_eq!(leaves2, leaves);
}
