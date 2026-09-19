//! O7/Step22 exact wire/reconstruction oracles, not an RD acceptance measurement.
use mars_codec::encode::{
    EncodeOptions, EncodeParams, ResidualQuantisation, encode_image_rd,
    encode_image_with_options,
};
use mars_codec::ifs::{self, Leaf};
use mars_codec::mars_format::{self, Header, MarsFormatError};
use mars_codec::progressive::{self, ProgressiveError};
use mars_codec::quant::ResidualQstep;
use mars_core::Plane;

fn params(lambda: Option<f64>) -> EncodeParams {
    EncodeParams {
        min_size: 4, max_size: 4, shift: 4, bits_alfa: 4, bits_beta: 7,
        max_alfa: 1.0, t_rms: 8.0, zero_threshold: 0, lambda,
    }
}

fn image() -> Plane {
    Plane::from_vec(16, 16, (0..256).map(|i| ((i * 37 + i / 16 * 13) % 256) as u8).collect())
}

fn residual_fixture(step: ResidualQstep) -> (Header, Vec<Leaf>) {
    let geometry = ifs::Header {
        bits_alfa: 4, bits_beta: 7, min_size: 4, max_size: 4,
        shift: 4, width: 8, height: 8, int_max_alfa: 32,
    };
    let leaves = [(0, 0), (4, 0), (0, 4), (4, 4)].into_iter().map(|(row, col)| {
        let mut residual = vec![0; 16];
        residual[0] = 4;
        residual[1] = -2;
        residual[5] = 1;
        Leaf {
            row, col, size: 4, mode: 3, qalfa: 4, qbeta: 64,
            isometry: 0, dom_row: 0, dom_col: 0, qgx: 0, qgy: 0, residual,
        }
    }).collect();
    (Header { geometry, residual_qstep: step }, leaves)
}

#[test]
fn qstep_closed_form_bounds_and_wire_rounding() {
    assert_eq!(ResidualQstep::LEGACY.get(), 8.0);
    let mut previous = 0.0;
    for lambda in [0.0, 0.01, 1.0, 50.0, 200.0, 800.0, 3200.0, f64::MAX] {
        let step = ResidualQstep::from_lambda(lambda);
        let expected = (lambda.sqrt() * (6.0 / std::f64::consts::LN_2).sqrt())
            .clamp(1.0, 65535.0) as f32;
        assert_eq!(step.get(), f64::from(expected));
        assert!(step.get() >= previous);
        assert!(step.get().is_finite());
        previous = step.get();
    }
    for lambda in [f64::NAN, f64::NEG_INFINITY, -1.0, f64::MIN_POSITIVE] {
        assert_eq!(ResidualQstep::from_lambda(lambda).get(), 1.0);
    }
    assert_eq!(ResidualQstep::from_lambda(f64::INFINITY).get(), 65535.0);
    for invalid in [0.0, -1.0, 0.99, 65536.0, f64::NAN, f64::INFINITY] {
        assert!(ResidualQstep::new(invalid).is_none());
    }
    let step = ResidualQstep::new(17.123456789).unwrap();
    assert_eq!(step.get(), f64::from(17.123456789f64 as f32));
}

#[test]
fn production_fixed_default_and_explicit_adaptive_roundtrip() {
    assert_eq!(ResidualQuantisation::default(), ResidualQuantisation::Fixed(ResidualQstep::LEGACY));
    assert_eq!(EncodeOptions::default().residual_quantisation, ResidualQuantisation::default());
    let source = image();
    for lambda in [None, Some(0.0), Some(50.0), Some(3200.0)] {
        let p = params(lambda);
        let (hdr, leaves, _, _) = encode_image_rd(&source, &p);
        assert_eq!(hdr.residual_qstep, ResidualQstep::LEGACY);
        let (fixed, fixed_leaves, _, _) = encode_image_with_options(&source, &p, &EncodeOptions {
            residual_quantisation: ResidualQuantisation::Fixed(ResidualQstep::LEGACY),
            ..EncodeOptions::default()
        });
        assert_eq!((hdr, &leaves), (fixed, &fixed_leaves));
        let (adaptive, adaptive_leaves, _, _) = encode_image_with_options(&source, &p, &EncodeOptions {
            residual_quantisation: ResidualQuantisation::LambdaAdaptive,
            ..EncodeOptions::default()
        });
        assert_eq!(adaptive.residual_qstep, lambda.map_or(ResidualQstep::LEGACY, ResidualQstep::from_lambda));
        for (h, ls) in [(hdr, leaves), (fixed, fixed_leaves), (adaptive, adaptive_leaves)] {
            let bytes = mars_format::write(&h, &ls).unwrap();
            assert_eq!(bytes[4], 0);
            assert_eq!(&bytes[20..24], &(h.residual_qstep.get() as f32).to_le_bytes());
            let (read_h, read_ls) = mars_format::read(&bytes).unwrap();
            assert_eq!(read_h, h);
            assert_eq!(read_ls, ls);
            assert_eq!(ifs::decode_iterative(&h, &ls, 10), ifs::decode_iterative(&read_h, &read_ls, 10));
        }
    }
}

#[test]
fn nonzero_residual_roundtrips_and_decoder_uses_step() {
    for step in [1.0, 8.0, 17.123456789, 65535.0] {
        let (hdr, leaves) = residual_fixture(ResidualQstep::new(step).unwrap());
        let bytes = mars_format::write(&hdr, &leaves).unwrap();
        let (decoded_hdr, decoded_leaves) = mars_format::read(&bytes).unwrap();
        assert_eq!((decoded_hdr, &decoded_leaves), (hdr, &leaves));
        assert_eq!(mars_format::write(&decoded_hdr, &decoded_leaves).unwrap(), bytes);
        let expected = ifs::decode_iterative(&hdr, &leaves, 10);
        assert_eq!(ifs::decode_iterative(&decoded_hdr, &decoded_leaves, 10), expected);
        let (zoomed, _) = ifs::zoom_leaves(&hdr, &leaves, 1.0);
        assert_eq!(zoomed.residual_qstep, hdr.residual_qstep);
        if step != 8.0 {
            assert_ne!(expected, ifs::decode_iterative(&hdr.geometry, &leaves, 10));
        }
        let source = Plane::from_vec(8, 8, vec![128; 64]);
        let progressive = progressive::encode(&source, &hdr, &leaves).unwrap();
        assert_eq!(progressive[4], 0);
        assert_eq!(&progressive[31..35], &(hdr.residual_qstep.get() as f32).to_le_bytes());
        assert_eq!(progressive::decode(&progressive).unwrap().image, expected);
        let ends = progressive::layer_end_offsets(&progressive).unwrap();
        for (i, end) in ends.into_iter().enumerate() {
            assert_eq!(progressive::decode(&progressive[..end]).unwrap().layers, i as u8 + 1);
            if i > 0 {
                assert_eq!(progressive::decode(&progressive[..end - 1]).unwrap().layers, i as u8);
            }
        }
    }
}


#[test]
fn color_planes_serialize_fixed_default_and_explicit_per_plane_lambda_steps() {
    use mars_codec::color::{self, ColorEncodeParams, Subsampling};
    use mars_core::image::Image;
    use mars_core::metrics::{rgb_from_ycbcr, ycbcr};
    let img = Image::rgb(image(), image(), image());
    for (subsampling, policy) in [Subsampling::Yuv444, Subsampling::Yuv420].into_iter().flat_map(|s| {
        [ResidualQuantisation::Fixed(ResidualQstep::LEGACY), ResidualQuantisation::LambdaAdaptive]
            .map(|p| (s, p))
    }) {
        let cfg = ColorEncodeParams {
            y: params(Some(2.0)), chroma: params(Some(200.0)), subsampling,
            adaptive_density: false, allowed_modes: [true; 4],
            rd_candidates: 1,
            lambda_regions: Vec::new(),
            color_regions: Vec::new(),
        };
        let (bytes, _) = color::encode_color_image_with_residual_quantisation(&img, &cfg, policy);
        if policy == ResidualQuantisation::default() {
            assert_eq!(bytes, color::encode_color_image(&img, &cfg).0);
        }
        assert_eq!(&bytes[..7], &[b'M', b'A', b'R', b'C', 0, if subsampling == Subsampling::Yuv444 { 1 } else { 2 }, 3]);
        let [y, cb, cr] = ycbcr(&img);
        let sources = if subsampling == Subsampling::Yuv420 {
            [y, color::downsample_box(&cb), color::downsample_box(&cr)]
        } else { [y, cb, cr] };
        let mut offset = 7;
        let mut planes = Vec::new();
        for (i, source) in sources.iter().enumerate() {
            let len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let stream = &bytes[offset..offset + len];
            offset += len;
            assert_eq!(stream[4], 0);
            let (hdr, leaves) = mars_format::read(stream).unwrap();
            let p = if i == 0 { &cfg.y } else { &cfg.chroma };
            let (expected_hdr, expected_leaves, _, _) = encode_image_with_options(source, p, &EncodeOptions {
                allowed_modes: cfg.allowed_modes,
                adaptive_density: cfg.adaptive_density,
                residual_quantisation: policy,
                rd_candidates: 1,
                lambda_regions: Vec::new(),
            });
            let expected_step = match policy {
                ResidualQuantisation::Fixed(step) => step,
                ResidualQuantisation::LambdaAdaptive => ResidualQstep::from_lambda(p.lambda.unwrap()),
            };
            assert_eq!(hdr.residual_qstep, expected_step);
            assert_eq!((hdr, &leaves), (expected_hdr, &expected_leaves));
            planes.push(ifs::decode_iterative(&expected_hdr, &expected_leaves, 10));
        }
        assert_eq!(offset, bytes.len());
        if subsampling == Subsampling::Yuv420 {
            planes[1] = color::upsample_nearest(&planes[1], 16, 16);
            planes[2] = color::upsample_nearest(&planes[2], 16, 16);
        }
        let expected = rgb_from_ycbcr(&planes[0], &planes[1], &planes[2]);
        assert_eq!(color::decode_color_image(&bytes, 10).unwrap(), expected);
        assert_eq!(color::decode_color_image_zoomed(&bytes, 10, 1.0).unwrap(), expected);
        let (progression, _) = color::decode_color_image_progression(&bytes, 10, None, 1.0, |_, _| {}).unwrap();
        assert_eq!(progression, expected);
    }
}

#[test]
fn gray_color_wrapper_preserves_explicit_policy_and_fixed_default() {
    use mars_codec::color::{self, ColorEncodeParams, Subsampling};
    use mars_core::image::Image;
    let img = Image::gray(image());
    let cfg = ColorEncodeParams {
        y: params(Some(50.0)), chroma: params(Some(200.0)),
        subsampling: Subsampling::Yuv420, adaptive_density: true,
        allowed_modes: [true, false, true, true],
        rd_candidates: 1,
        lambda_regions: Vec::new(),
        color_regions: Vec::new(),
    };
    for policy in [ResidualQuantisation::default(), ResidualQuantisation::LambdaAdaptive] {
        let (bytes, _) = color::encode_color_image_with_residual_quantisation(&img, &cfg, policy);
        if policy == ResidualQuantisation::default() {
            assert_eq!(bytes, color::encode_color_image(&img, &cfg).0);
        }
        let (hdr, leaves, _, _) = encode_image_with_options(&img.planes()[0], &cfg.y, &EncodeOptions {
            allowed_modes: cfg.allowed_modes, adaptive_density: cfg.adaptive_density,
            residual_quantisation: policy,
            rd_candidates: 1,
            lambda_regions: Vec::new(),
        });
        assert_eq!(bytes, color::wrap_gray_stream(mars_format::write(&hdr, &leaves).unwrap()));
        assert_eq!(color::decode_color_image(&bytes, 10).unwrap().planes()[0],
            ifs::decode_iterative(&hdr, &leaves, 10));
    }
}

#[test]
fn malformed_qstep_headers_are_rejected() {
    let (hdr, leaves) = residual_fixture(ResidualQstep::LEGACY);
    let source = Plane::from_vec(8, 8, vec![128; 64]);
    let plane = mars_format::write(&hdr, &leaves).unwrap();
    let prog = progressive::encode(&source, &hdr, &leaves).unwrap();
    for value in [0.0f32, -0.0, -1.0, 0.5, 65536.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bytes = plane.clone();
        bytes[20..24].copy_from_slice(&value.to_le_bytes());
        assert_eq!(mars_format::read(&bytes), Err(MarsFormatError::InvalidResidualQstep));
        let mut bytes = prog.clone();
        bytes[31..35].copy_from_slice(&value.to_le_bytes());
        assert_eq!(progressive::decode(&bytes).unwrap_err(), ProgressiveError::InvalidResidualQstep);
    }
    for end in 20..24 {
        assert_eq!(mars_format::read(&plane[..end]), Err(MarsFormatError::Truncated));
    }
    for end in 31..35 {
        assert_eq!(progressive::decode(&prog[..end]).unwrap_err(), ProgressiveError::Truncated);
    }
    let mut plane = plane;
    plane[4] = 2;
    assert_eq!(mars_format::read(&plane), Err(MarsFormatError::UnsupportedVersion(2)));
    let mut prog = prog;
    prog[4] = 2;
    assert_eq!(progressive::decode(&prog).unwrap_err(), ProgressiveError::UnsupportedVersion(2));
}
