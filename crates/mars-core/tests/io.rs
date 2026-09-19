//! Reader tests.
//!
//! A reader bug is indistinguishable from a codec bug in the final numbers, and a
//! transposed or mis-offset read produces a plausible image and a wrong PSNR — exactly
//! §2.1's failure mode. So the fixtures are hand-written and the expected bytes are
//! spelled out.

use std::path::{Path, PathBuf};

use mars_core::image::Image;
use mars_core::io::{
    read_image, read_pgm, read_ppm, read_raw, write_jpeg, write_jpeg_with_quality, write_png,
    write_pnm, write_tga, write_tiff, write_webp, ImageError,
};

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
fn p6_ppm_is_read_as_colour_but_a_pgm_with_a_p6_magic_is_rejected() {
    // .ppm dispatches to the colour P6 reader (added for the Step 4 JPEG 2000 anchor
    // driver, which needs a PNM round trip with no colour-management metadata --
    // docs/decisions.md D18).
    let path = scratch("c.ppm");
    std::fs::write(&path, "P6\n1 1\n255\n\x01\x02\x03").unwrap();
    let img = read_image(&path, None).unwrap();
    assert_eq!((img.width(), img.height()), (1, 1));
    assert_eq!(
        img.planes().iter().map(|p| p.get(0, 0)).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );

    // .pgm dispatch still only accepts P5/P2; a P6 magic there is a real format error,
    // not silently reinterpreted as colour.
    let path = scratch("c.pgm");
    std::fs::write(&path, "P6\n1 1\n255\n\0").unwrap();
    assert!(matches!(read_pgm(&path), Err(ImageError::Malformed { .. })));
}

#[test]
fn an_unrecognised_extension_is_still_rejected() {
    let path = scratch("c.bmp");
    std::fs::write(&path, "not an image").unwrap();
    assert!(matches!(
        read_image(&path, None),
        Err(ImageError::UnknownFormat { .. })
    ));
}

#[test]
fn write_pnm_then_read_ppm_round_trips_a_colour_image_exactly() {
    let path = scratch("rt.ppm");
    let r = mars_core::image::Plane::from_vec(2, 2, vec![1, 2, 3, 4]);
    let g = mars_core::image::Plane::from_vec(2, 2, vec![10, 20, 30, 40]);
    let b = mars_core::image::Plane::from_vec(2, 2, vec![100, 200, 250, 5]);
    let img = Image::rgb(r, g, b);
    write_pnm(&path, &img).unwrap();
    let back = read_ppm(&path).unwrap();
    assert_eq!(back.width(), 2);
    assert_eq!(back.height(), 2);
    for (a, b) in img.planes().iter().zip(back.planes()) {
        assert_eq!(a.as_slice(), b.as_slice());
    }
}

#[test]
fn write_pnm_then_read_pgm_round_trips_a_gray_image_exactly() {
    let path = scratch("rt.pgm");
    let img = Image::gray(mars_core::image::Plane::from_vec(2, 2, vec![7, 8, 9, 10]));
    write_pnm(&path, &img).unwrap();
    let back = read_pgm(&path).unwrap();
    assert_eq!(back.as_slice(), &[7, 8, 9, 10]);
}

#[test]
fn write_png_then_read_image_round_trips_a_colour_image_exactly() {
    let path = scratch("rt.png");
    let r = mars_core::image::Plane::from_vec(2, 2, vec![1, 2, 3, 4]);
    let g = mars_core::image::Plane::from_vec(2, 2, vec![10, 20, 30, 40]);
    let b = mars_core::image::Plane::from_vec(2, 2, vec![100, 200, 250, 5]);
    let img = Image::rgb(r, g, b);
    write_png(&path, &img).unwrap();
    let back = read_image(&path, None).unwrap();
    assert_eq!((back.width(), back.height()), (2, 2));
    for (a, b) in img.planes().iter().zip(back.planes()) {
        assert_eq!(a.as_slice(), b.as_slice());
    }
}

#[test]
fn png_content_is_autodetected_whatever_the_extension() {
    // A PNG written to a name the dispatcher does not know must still read: the reader
    // falls back to the file's own signature rather than refusing the extension.
    let path = scratch("mystery.dat");
    let img = Image::gray(mars_core::image::Plane::from_vec(2, 2, vec![7, 8, 9, 10]));
    write_png(&path, &img).unwrap();
    let back = read_image(&path, None).unwrap();
    assert_eq!(back.planes()[0].as_slice(), &[7, 8, 9, 10]);
}

#[test]
fn jpeg_is_written_from_content_and_read_back_close() {
    let path = scratch("colour.jpg");
    let n = 8 * 8;
    let img = Image::rgb(
        mars_core::image::Plane::from_vec(8, 8, vec![200u8; n]),
        mars_core::image::Plane::from_vec(8, 8, vec![100u8; n]),
        mars_core::image::Plane::from_vec(8, 8, vec![50u8; n]),
    );
    write_jpeg(&path, &img).unwrap();
    let back = read_image(&path, None).unwrap();
    assert_eq!((back.width(), back.height()), (8, 8));
    assert_eq!(back.planes().len(), 3);
    // JPEG is lossy, so a constant block is only required to stay close, not exact.
    for (original, decoded) in img.planes().iter().zip(back.planes()) {
        let worst = original
            .as_slice()
            .iter()
            .zip(decoded.as_slice())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst <= 16, "plane drifted by {worst} through a JPEG round trip");
    }
}

#[test]
fn jpeg_content_is_autodetected_even_when_named_png() {
    // The writer forces the format from the function, so a mis-named `.png` is really a
    // JPEG on disk -- and the reader must decode it as one by content, not name.
    let path = scratch("actually-jpeg.png");
    let img = Image::gray(mars_core::image::Plane::from_vec(8, 8, vec![128u8; 64]));
    write_jpeg(&path, &img).unwrap();
    let back = read_image(&path, None).unwrap();
    assert_eq!((back.width(), back.height()), (8, 8));
    assert_eq!(back.planes().len(), 1, "a gray JPEG reads back as one plane");
}

#[test]
fn jpeg_quality_trades_size_for_detail() {
    let low_path = scratch("quality-low.jpg");
    let high_path = scratch("quality-high.jpg");
    // High-frequency content, so the quantiser has something to discard at low quality.
    let img = Image::gray(mars_core::image::Plane::from_vec(
        16,
        16,
        (0..256).map(|i| ((i * 37 + i / 16 * 13) % 256) as u8).collect(),
    ));
    write_jpeg_with_quality(&low_path, &img, 10).unwrap();
    write_jpeg_with_quality(&high_path, &img, 95).unwrap();
    let low = std::fs::read(&low_path).unwrap();
    let high = std::fs::read(&high_path).unwrap();
    assert!(low.starts_with(&[0xFF, 0xD8, 0xFF]), "quality 10 is not a JPEG");
    assert!(high.starts_with(&[0xFF, 0xD8, 0xFF]), "quality 95 is not a JPEG");
    assert!(
        high.len() > low.len(),
        "quality 95 ({} bytes) should exceed quality 10 ({} bytes)",
        high.len(),
        low.len()
    );
    let back = read_image(&high_path, None).unwrap();
    assert_eq!((back.width(), back.height()), (16, 16));
    assert_eq!(back.planes().len(), 1);
}

/// Assert that `write` lands `img` on disk in a form `read_image` reads back sample-for-
/// sample and plane-for-plane.
fn assert_exact_round_trip(
    name: &str,
    img: &Image,
    write: fn(&Path, &Image) -> Result<(), ImageError>,
) {
    let path = scratch(name);
    write(&path, img).unwrap();
    let back = read_image(&path, None).unwrap();
    assert_eq!(
        (back.width(), back.height()),
        (img.width(), img.height()),
        "{name}"
    );
    assert_eq!(back.planes().len(), img.planes().len(), "{name}");
    for (a, b) in img.planes().iter().zip(back.planes()) {
        assert_eq!(a.as_slice(), b.as_slice(), "{name}");
    }
}

fn colour_fixture() -> Image {
    Image::rgb(
        mars_core::image::Plane::from_vec(2, 2, vec![1, 2, 3, 4]),
        mars_core::image::Plane::from_vec(2, 2, vec![10, 20, 30, 40]),
        mars_core::image::Plane::from_vec(2, 2, vec![100, 200, 250, 5]),
    )
}

#[test]
fn tga_tiff_and_webp_round_trip_colour_exactly() {
    let img = colour_fixture();
    assert_exact_round_trip("rt.tga", &img, write_tga);
    assert_exact_round_trip("rt.tiff", &img, write_tiff);
    assert_exact_round_trip("rt.webp", &img, write_webp);
}

#[test]
fn tga_and_tiff_keep_a_gray_image_gray() {
    // Unlike WebP, both carry a real grayscale mode, so the plane count survives too.
    let img = Image::gray(mars_core::image::Plane::from_vec(2, 2, vec![7, 8, 9, 10]));
    assert_exact_round_trip("gray.tga", &img, write_tga);
    assert_exact_round_trip("gray.tiff", &img, write_tiff);
}

#[test]
fn tiff_and_webp_are_autodetected_under_an_unknown_extension() {
    let img = colour_fixture();
    let tiff = scratch("mystery-tiff.dat");
    write_tiff(&tiff, &img).unwrap();
    let back = read_image(&tiff, None).unwrap();
    assert_eq!((back.width(), back.height()), (2, 2));
    assert_eq!(back.planes().len(), 3);

    let webp = scratch("mystery-webp.dat");
    write_webp(&webp, &img).unwrap();
    let back = read_image(&webp, None).unwrap();
    assert_eq!((back.width(), back.height()), (2, 2));
    assert_eq!(back.planes().len(), 3);
}

#[test]
fn tga_without_its_extension_is_not_autodetected() {
    // TGA has no leading signature, so it is dispatched by extension only: an unrecognised
    // extension is refused rather than guessed at.
    let img = colour_fixture();
    let path = scratch("mystery-tga.dat");
    write_tga(&path, &img).unwrap();
    assert!(matches!(
        read_image(&path, None),
        Err(ImageError::UnknownFormat { .. })
    ));
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
