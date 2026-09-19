//! Ordinary valid tiny/odd-geometry images must encode, serialise, parse and decode
//! without panicking, with independent DC and border pixel oracles.
use mars_codec::color::{self, ColorEncodeParams, Subsampling};
use mars_codec::encode::{
    EncodeOptions, EncodeParams, ResidualQuantisation, encode_image, encode_image_with_options,
};
use mars_codec::ifs::{self, Leaf};
use mars_codec::mars_format::{self, Header};
use mars_core::Plane;
use mars_core::image::Image;

fn params(lambda: Option<f64>) -> EncodeParams {
    EncodeParams {
        min_size: 4,
        max_size: 4,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda,
    }
}

fn image(width: usize, height: usize) -> Plane {
    let mut state = 0x5eed_1979_u32;
    let pixels = (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect();
    Plane::from_vec(width, height, pixels)
}

fn assert_roundtrip(image: &Plane, p: &EncodeParams) {
    let (hdr, leaves, _) = encode_image(image, p);
    let bytes = mars_format::write(&hdr, &leaves).unwrap();
    let (parsed_hdr, parsed_leaves) = mars_format::read(&bytes).unwrap();
    assert_eq!(parsed_hdr, hdr);
    assert_eq!(parsed_leaves, leaves);
    let decoded = ifs::decode_iterative(&hdr, &leaves, 10);
    assert_eq!(decoded.width(), image.width());
    assert_eq!(decoded.height(), image.height());
    let reparsed = ifs::decode_iterative(&parsed_hdr, &parsed_leaves, 10);
    assert_eq!(decoded, reparsed);
}

#[test]
fn tiny_and_odd_images_roundtrip_without_panicking() {
    for (w, h) in [(1usize, 1usize), (1, 5), (5, 1), (3, 5), (4, 4), (8, 8)] {
        let img = image(w, h);
        assert_roundtrip(&img, &params(None));
        assert_roundtrip(&img, &params(Some(2.0)));
    }
}

#[test]
fn four_by_four_decodes_and_dc_leaves_never_read_a_domain() {
    let img = image(4, 4);
    let (hdr, leaves, _) = encode_image(&img, &params(None));
    assert_eq!(leaves.len(), 1);
    let leaf = &leaves[0];
    assert_eq!(
        leaf.mode, 0,
        "4x4 has no legal domain; the leaf must be DC-only"
    );
    assert_eq!(leaf.qalfa, 0);
    assert_eq!(leaf.dom_row, 0);
    assert_eq!(leaf.dom_col, 0);
    let sum: u32 = img.as_slice().iter().map(|&p| u32::from(p)).sum();
    let qbeta = (f64::from(sum) / 16.0 / 255.0 * 127.0 + 0.5) as u32;
    assert_eq!(leaf.qbeta, qbeta);
    let expected = (0.5 + f64::from(qbeta) / 127.0 * 255.0) as u8;
    let decoded = ifs::decode_iterative(&hdr, &leaves, 10);
    assert_eq!(decoded.as_slice(), &[expected; 16]);
    assert_eq!(decoded.width(), 4);
    assert_eq!(decoded.height(), 4);
}

#[test]
fn rd_modes_0_and_2_only_still_roundtrip_on_tiny_images() {
    for (w, h) in [(1usize, 1usize), (1, 5), (5, 1), (3, 5), (4, 4), (8, 8)] {
        let img = image(w, h);
        let p = params(Some(2.0));
        let (hdr, leaves, _, _) = encode_image_with_options(
            &img,
            &p,
            &EncodeOptions {
                allowed_modes: [true, false, true, false],
                ..EncodeOptions::default()
            },
        );
        assert!(leaves.iter().all(|l| matches!(l.mode, 0 | 2)));
        let bytes = mars_format::write(&hdr, &leaves).unwrap();
        let (parsed_hdr, parsed_leaves) = mars_format::read(&bytes).unwrap();
        assert_eq!((parsed_hdr, &parsed_leaves), (hdr, &leaves));
        assert_eq!(
            ifs::decode_iterative(&hdr, &leaves, 10),
            ifs::decode_iterative(&parsed_hdr, &parsed_leaves, 10)
        );
    }
}

#[test]
fn color_8x8_420_roundtrips_through_the_existing_wrapper() {
    let img = Image::rgb(image(8, 8), image(8, 8), image(8, 8));
    let cfg = ColorEncodeParams {
        y: params(Some(2.0)),
        chroma: params(Some(50.0)),
        subsampling: Subsampling::Yuv420,
        adaptive_density: false,
        allowed_modes: [true; 4],
        rd_candidates: 1,
    };
    let (bytes, _) = color::encode_color_image_with_residual_quantisation(
        &img,
        &cfg,
        ResidualQuantisation::default(),
    );
    let decoded = color::decode_color_image(&bytes, 10).unwrap();
    assert_eq!(decoded.width(), 8);
    assert_eq!(decoded.height(), 8);
}

#[test]
fn odd_border_pixels_use_dc_quantisation_not_low_bit_masking() {
    for lambda in [None, Some(2.0)] {
        for bits_beta in [4, 7, 8] {
            let mut p = params(lambda);
            p.bits_beta = bits_beta;
            let img = Plane::from_vec(3, 5, vec![200; 15]);
            let (hdr, leaves, _) = encode_image(&img, &p);
            let max_beta = (1u32 << bits_beta) - 1;
            let expected_qbeta = (200.0 / 255.0 * f64::from(max_beta) + 0.5) as u32;
            assert!(leaves.iter().any(|l| l.size == 1));
            for leaf in &leaves {
                assert_eq!(
                    leaf.qbeta, expected_qbeta,
                    "size={}, lambda={lambda:?}, bits={bits_beta}",
                    leaf.size
                );
            }
            let bytes = mars_format::write(&hdr, &leaves).unwrap();
            let (hdr, leaves) = mars_format::read(&bytes).unwrap();
            let expected = (0.5 + f64::from(expected_qbeta) / f64::from(max_beta) * 255.0) as u8;
            assert_eq!(
                ifs::decode_iterative(&hdr, &leaves, 10).as_slice(),
                &[expected; 15]
            );
        }
    }
}

#[test]
fn zero_alpha_residual_keeps_domain_local_orientation_without_domain_reads() {
    let hdr = Header::from(ifs::Header {
        bits_alfa: 4,
        bits_beta: 7,
        min_size: 2,
        max_size: 2,
        shift: 4,
        width: 2,
        height: 2,
        int_max_alfa: 32,
    });
    // In-memory mode 3 supports zero alpha, although the wire only codes positive alpha.
    let residual = vec![1, 2, -3, 4];
    let coefficients: Vec<f64> = residual.iter().map(|&l| f64::from(l) * 8.0).collect();
    let spatial = mars_codec::dct::inverse_dct2d(&coefficients, 2);
    let pixels: Vec<u8> = spatial
        .iter()
        .map(|&r| (0.5 + 64.0 / 127.0 * 255.0 + r) as u8)
        .collect();
    // Range-raster indices into the domain-raster residual, independent of isometry::map.
    for (iso, order) in [
        [0, 1, 2, 3],
        [1, 3, 0, 2],
        [2, 0, 3, 1],
        [3, 2, 1, 0],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [0, 2, 1, 3],
        [3, 1, 2, 0],
    ]
    .into_iter()
    .enumerate()
    {
        let leaf = Leaf {
            row: 0,
            col: 0,
            size: 2,
            mode: 3,
            qalfa: 0,
            qbeta: 64,
            isometry: iso as u8,
            dom_row: 0,
            dom_col: 0,
            qgx: 0,
            qgy: 0,
            residual: residual.clone(),
        };
        let expected: Vec<u8> = order.into_iter().map(|i| pixels[i]).collect();
        for seed in [0, 128, 255] {
            assert_eq!(
                ifs::decode_step(&hdr, std::slice::from_ref(&leaf), &[seed; 4]),
                expected
            );
        }
    }
}

#[test]
fn one_by_one_contracted_plane_is_well_formed() {
    let img = image(1, 1);
    let contracted = mars_codec::encode::build_contracted(&img);
    assert_eq!(contracted.width(), 0);
    assert_eq!(contracted.height(), 0);
    assert!(contracted.raw().0.is_empty());
}
