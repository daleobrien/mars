//! Reader tests.
//!
//! A reader bug is indistinguishable from a codec bug in the final numbers, and a
//! transposed or mis-offset read produces a plausible image and a wrong PSNR — exactly
//! §2.1's failure mode. So the fixtures are hand-written and the expected bytes are
//! spelled out.

use std::path::PathBuf;

use mars_core::io::{read_image, read_pgm, read_raw, ImageError};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mars-io-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

#[test]
fn p2_ascii_pgm_reads_in_row_major_order() {
    let path = scratch("a.pgm");
    std::fs::write(
        &path,
        "P2\n# a comment, which must be skipped\n3 2\n255\n10 20 30\n40 50 60\n",
    )
    .unwrap();
    let p = read_pgm(&path).unwrap();
    assert_eq!((p.width(), p.height()), (3, 2));
    assert_eq!(p.as_slice(), &[10, 20, 30, 40, 50, 60]);
    assert_eq!(p.get(2, 1), 60, "get(x, y), not get(row, col)");
}

#[test]
fn p5_binary_pgm_reads_the_same_image() {
    let path = scratch("b.pgm");
    let mut bytes = b"P5\n3 2\n255\n".to_vec();
    bytes.extend_from_slice(&[10, 20, 30, 40, 50, 60]);
    std::fs::write(&path, &bytes).unwrap();
    let p = read_pgm(&path).unwrap();
    assert_eq!((p.width(), p.height()), (3, 2));
    assert_eq!(p.as_slice(), &[10, 20, 30, 40, 50, 60]);
}

#[test]
fn p5_header_whitespace_is_consumed_exactly_once() {
    // The classic PNM bug: eating all trailing whitespace after maxval, which silently
    // drops a leading 0x20, 0x0a or 0x09 pixel and shifts the whole raster by one.
    let path = scratch("ws.pgm");
    let mut bytes = b"P5\n2 2\n255\n".to_vec();
    bytes.extend_from_slice(&[b' ', b'\n', 9, 0]); // pixels 32, 10, 9, 0
    std::fs::write(&path, &bytes).unwrap();
    let p = read_pgm(&path).unwrap();
    assert_eq!(p.as_slice(), &[32, 10, 9, 0]);
}

#[test]
fn a_non_255_maxval_is_refused_rather_than_rescaled() {
    let path = scratch("maxval.pgm");
    std::fs::write(&path, "P2\n2 1\n15\n7 8\n").unwrap();
    let err = read_pgm(&path).unwrap_err();
    assert!(
        matches!(&err, ImageError::Malformed { reason, .. } if reason.contains("maxval")),
        "{err}"
    );
}

#[test]
fn a_truncated_raster_is_an_error() {
    let path = scratch("short.pgm");
    let mut bytes = b"P5\n4 4\n255\n".to_vec();
    bytes.extend_from_slice(&[1, 2, 3]);
    std::fs::write(&path, &bytes).unwrap();
    assert!(read_pgm(&path).is_err());

    let path = scratch("short2.pgm");
    std::fs::write(&path, "P2\n4 4\n255\n1 2 3\n").unwrap();
    let err = read_pgm(&path).unwrap_err();
    assert!(
        matches!(&err, ImageError::Malformed { reason, .. } if reason.contains("3 of 16")),
        "{err}"
    );
}

#[test]
fn p6_and_other_magics_are_rejected() {
    let path = scratch("c.ppm");
    std::fs::write(&path, "P6\n1 1\n255\n\0\0\0").unwrap();
    // Dispatch is by extension, and .ppm is not a format the harness reads.
    assert!(matches!(
        read_image(&path, None),
        Err(ImageError::UnknownFormat { .. })
    ));
    let path = scratch("c.pgm");
    std::fs::write(&path, "P6\n1 1\n255\n\0").unwrap();
    assert!(matches!(read_pgm(&path), Err(ImageError::Malformed { .. })));
}

#[test]
fn raw_requires_exact_dimensions() {
    let path = scratch("d.raw");
    std::fs::write(&path, [1u8, 2, 3, 4, 5, 6]).unwrap();
    assert_eq!(
        read_raw(&path, 3, 2).unwrap().as_slice(),
        &[1, 2, 3, 4, 5, 6]
    );
    // A size that does not divide exactly is a silent transpose waiting to happen.
    assert!(read_raw(&path, 2, 2).is_err());
    assert!(read_raw(&path, 4, 2).is_err());
    // And dispatch refuses to guess.
    assert!(matches!(
        read_image(&path, None),
        Err(ImageError::RawDimensionsRequired { .. })
    ));
}

#[test]
fn lena_raw_from_the_1998_reference_is_512x512() {
    // The one real fixture available at Step 1 (§M9: lena.raw as a fixture and
    // smoke-test image). 512*512 = 262144 bytes, headerless.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/mars1/lena.raw")
        .canonicalize()
        .expect("reference/mars1/lena.raw must exist");
    let p = read_raw(&path, 512, 512).unwrap();
    assert_eq!(p.as_slice().len(), 512 * 512);
    // A transposed read would still be 512x512 and still look like Lena, so check a
    // known asymmetric statistic instead: the top row is much brighter than the left
    // column in this image.
    let top: u64 = p.row(0).iter().map(|&v| u64::from(v)).sum();
    let left: u64 = (0..512).map(|y| u64::from(p.get(0, y))).sum();
    assert_ne!(top, left);
}
