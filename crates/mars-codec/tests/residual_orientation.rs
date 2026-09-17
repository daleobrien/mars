//! Independent range-to-domain oracle: no production isometry map or decoder is used
//! to compute the expected coefficients and pixels.
use mars_codec::encode::{
    EncodeOptions, EncodeParams, ResidualQuantisation, encode_image_with_options,
};
use mars_codec::{dct, ifs, mars_format, quant};
use mars_core::Plane;

// Pull a range-local output (i, j) back to its domain-local source (u, v).
// In particular, the two quarter turns are inverses, not self-inverse.
fn source(k: u8, i: usize, j: usize, n: usize) -> (usize, usize) {
    match k {
        0 => (i, j),
        1 => (j, n - 1 - i),
        2 => (n - 1 - j, i),
        3 => (n - 1 - i, n - 1 - j),
        4 => (i, n - 1 - j),
        5 => (n - 1 - i, j),
        6 => (j, i),
        7 => (n - 1 - j, n - 1 - i),
        _ => panic!("invalid isometry"),
    }
}

fn asymmetric_image() -> Plane {
    let mut state = 0x5eed_1979_u32;
    let pixels = (0..32 * 32)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        })
        .collect();
    Plane::from_vec(32, 32, pixels)
}

fn prediction(h: &mars_format::Header, leaf: &ifs::Leaf, px: &[u8], u: usize, v: usize) -> f64 {
    let stride = h.width as usize;
    let r = leaf.dom_row as usize + 2 * u;
    let c = leaf.dom_col as usize + 2 * v;
    let domain = [
        px[r * stride + c],
        px[r * stride + c + 1],
        px[(r + 1) * stride + c],
        px[(r + 1) * stride + c + 1],
    ]
    .into_iter()
    .map(f64::from)
    .sum::<f64>()
        / 4.0;
    let alpha = f64::from(leaf.qalfa) / f64::from(1u32 << h.bits_alfa) * h.max_alfa();
    let beta = f64::from(leaf.qbeta) / f64::from((1u32 << h.bits_beta) - 1)
        * ((1.0 + alpha.abs()) * 255.0)
        - alpha * 255.0;
    (0.5 + domain * alpha) + beta
}

#[test]
fn encoded_residuals_reconstruct_in_all_eight_orientations_after_wire_roundtrip() {
    let image = asymmetric_image();
    for (policy, lambda, expected_step) in [
        (
            ResidualQuantisation::Fixed(quant::ResidualQstep::LEGACY),
            2.0,
            quant::ResidualQstep::LEGACY,
        ),
        (
            ResidualQuantisation::Fixed(quant::ResidualQstep::new(17.123456789).unwrap()),
            2.0,
            quant::ResidualQstep::new(17.123456789).unwrap(),
        ),
        (
            ResidualQuantisation::LambdaAdaptive,
            2.0,
            quant::ResidualQstep::from_lambda(2.0),
        ),
        (
            ResidualQuantisation::LambdaAdaptive,
            50.0,
            quant::ResidualQstep::from_lambda(50.0),
        ),
    ] {
        let params = EncodeParams {
            min_size: 4,
            max_size: 4,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            t_rms: 8.0,
            zero_threshold: 0,
            lambda: Some(lambda),
        };
        let (header, leaves, _, _) = encode_image_with_options(
            &image,
            &params,
            &EncodeOptions {
                allowed_modes: [false, false, false, true],
                adaptive_density: false,
                residual_quantisation: policy,
            },
        );
        assert_eq!(header.residual_qstep, expected_step);
        let bytes = mars_format::write(&header, &leaves).unwrap();
        let (header, parsed) = mars_format::read(&bytes).unwrap();
        assert_eq!(parsed, leaves);
        assert_eq!(header.residual_qstep, expected_step);
        assert_eq!(mars_format::write(&header, &parsed).unwrap(), bytes);

        let px = image.as_slice();
        let mut expected = vec![0; px.len()];
        let mut seen = [false; 8];
        for leaf in &parsed {
            assert_eq!(leaf.mode, 3, "fixture must exercise real residual encoding");
            assert!(leaf.qalfa > 0);
            let n = leaf.size as usize;
            let mut residual = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    let (u, v) = source(leaf.isometry, i, j, n);
                    let pos = (leaf.row as usize + i) * image.width() + leaf.col as usize + j;
                    residual[u * n + v] = f64::from(px[pos]) - prediction(&header, leaf, px, u, v);
                }
            }
            let expected_levels: Vec<i32> = dct::forward_dct2d(&residual, n)
                .into_iter()
                .map(|x| quant::dead_zone_quantize(x, expected_step.get(), 0.5).clamp(-63, 63))
                .collect();
            assert_eq!(
                leaf.residual, expected_levels,
                "policy={policy:?}, isometry={}",
                leaf.isometry
            );
            assert!(
                expected_levels.iter().skip(1).any(|&x| x != 0),
                "asymmetric AC residual required"
            );
            seen[leaf.isometry as usize] = true;
            let dequantized: Vec<f64> = expected_levels
                .into_iter()
                .map(|x| quant::dead_zone_dequantize(x, expected_step.get(), 0.5))
                .collect();
            let reconstructed = dct::inverse_dct2d(&dequantized, n);
            for i in 0..n {
                for j in 0..n {
                    let (u, v) = source(leaf.isometry, i, j, n);
                    let value = prediction(&header, leaf, px, u, v) + reconstructed[u * n + v];
                    expected[(leaf.row as usize + i) * image.width() + leaf.col as usize + j] =
                        value.clamp(0.0, 255.0) as u8;
                }
            }
        }
        assert_eq!(seen, [true; 8], "each policy must cover every isometry");
        // Use the source as the previous iteration: this is the prediction state priced
        // by the encoder, not a claim that the eventual IFS fixed point equals the source.
        assert_eq!(
            ifs::decode_step(&header, &parsed, px),
            expected,
            "policy={policy:?}"
        );
    }
}
