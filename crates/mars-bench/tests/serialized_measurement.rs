use mars_bench::bdrate::RdPoint;
use mars_bench::density_gate::sample_with_density;
use mars_bench::mode_gate::{STEP14_MODES, STEP15_MODES, sample_with_modes};
use mars_bench::rd_opt::{RdSample, sample};
use mars_codec::encode::{
    EncodeOptions, EncodeParams, ModeStats, ResidualQuantisation, encode_image,
    encode_image_rd_with_modes_and_density, encode_image_with_options,
};
use mars_codec::ifs::{Leaf, decode_iterative};
use mars_codec::mars_format::{self, Header};
use mars_codec::quant::ResidualQstep;
use mars_core::Plane;
use mars_core::metrics::psnr;

fn image() -> Plane {
    Plane::from_vec(
        16,
        16,
        (0..256)
            .map(|i| ((i * 37 + i / 16 * 13) % 256) as u8)
            .collect(),
    )
}

fn params(lambda: Option<f64>) -> EncodeParams {
    EncodeParams {
        min_size: 4,
        max_size: 8,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda,
    }
}

fn assert_serialized_sample(
    image: &Plane,
    reported: &RdSample,
    header: &Header,
    leaves: &[Leaf],
    evals: u64,
) -> Vec<Leaf> {
    let bytes = mars_format::write(header, leaves).unwrap();
    let (wire_header, wire_leaves) = mars_format::read(&bytes).unwrap();
    assert_eq!(wire_header, *header);
    assert_eq!(wire_leaves, leaves);
    let decoded = decode_iterative(&wire_header, &wire_leaves, 10);
    assert_eq!(decoded, decode_iterative(header, leaves, 10));
    let expected = RdPoint::from_size(
        bytes.len() as u64,
        image.width(),
        image.height(),
        psnr(image, &decoded).unwrap_or(f64::INFINITY),
    );
    assert_eq!(reported.point, expected);
    assert!(reported.point.psnr.is_finite());
    assert_eq!(reported.evals, evals);
    assert_eq!(reported.leaves, wire_leaves.len());
    wire_leaves
}

fn assert_stats(reported: &ModeStats, expected: &ModeStats, leaves: &[Leaf]) {
    let mut counts = [0; 4];
    for leaf in leaves {
        counts[usize::from(leaf.mode)] += 1;
    }
    assert_eq!(reported.leaf_modes, counts);
    assert_eq!(reported.leaf_modes, expected.leaf_modes);
    assert_eq!(reported.split_decisions, expected.split_decisions);
    assert_eq!(reported.leaf_decisions, expected.leaf_decisions);
}

#[test]
fn rd_sample_reports_serialized_rate_and_quality_for_threshold_and_lambda() {
    let image = image();
    for lambda in [None, Some(0.0), Some(100.0)] {
        let params = params(lambda);
        let (header, leaves, evals) = encode_image(&image, &params);
        let reported = sample(&image, &params);
        assert_serialized_sample(&image, &reported, &header, &leaves, evals);
    }
}

#[test]
fn mode_samples_report_serialized_rate_quality_and_histograms() {
    let image = image();
    let params = params(Some(100.0));
    for allowed_modes in [STEP14_MODES, STEP15_MODES] {
        let options = EncodeOptions {
            allowed_modes,
            residual_quantisation: ResidualQuantisation::Fixed(ResidualQstep::LEGACY),
            ..EncodeOptions::default()
        };
        let (header, leaves, evals, stats) = encode_image_with_options(&image, &params, &options);
        let (reported, reported_stats) = sample_with_modes(&image, &params, allowed_modes);
        assert_eq!(header.residual_qstep, ResidualQstep::LEGACY);
        let wire_leaves = assert_serialized_sample(&image, &reported, &header, &leaves, evals);
        assert!(
            wire_leaves
                .iter()
                .all(|leaf| allowed_modes[usize::from(leaf.mode)])
        );
        assert_stats(&reported_stats, &stats, &wire_leaves);
    }
}

#[test]
fn density_samples_report_serialized_rate_quality_and_partition() {
    let image = image();
    let params = params(Some(100.0));
    for adaptive_density in [false, true] {
        let (header, leaves, evals, stats) =
            encode_image_rd_with_modes_and_density(&image, &params, [true; 4], adaptive_density);
        let reported = sample_with_density(&image, &params, adaptive_density);
        let wire_leaves =
            assert_serialized_sample(&image, &reported.sample, &header, &leaves, evals);
        assert_eq!(reported.leaves, wire_leaves);
        assert_stats(&reported.stats, &stats, &wire_leaves);
        assert!(reported.encode_secs.is_finite() && reported.encode_secs >= 0.0);
    }
}
