//! Step 15's own comparison tools: a "Step 14 equivalent" curve (modes 0/2 only, via
//! `mars_codec::encode::EncodeOptions`' mode mask) and a "Step 15" curve
//! (all four modes), both run through the exact same rate-estimation and search machinery
//! so the only difference between the two curves is which modes `J` was allowed to pick
//! from -- plus the mode-usage histogram the brief calls the step's real header finding.
//! Both arms explicitly retain fixed8 residual quantisation after Step 22/O7.
//!
//! See `mars_codec::encode::Ctx::allowed_modes`'s doc for why this same-codebase A/B was
//! chosen over diffing against a separate git revision.

use mars_codec::encode::{
    EncodeOptions, EncodeParams, ModeStats, ResidualQuantisation, encode_image_with_options,
};
use mars_codec::mars_format;
use mars_codec::quant::ResidualQstep;
use mars_core::Plane;

use crate::bdrate::RdCurve;
use crate::rd_opt::{RdSample, sample_from_bytes};

/// Modes 0 (flat) and 2 (fractal) only -- Step 14's own decision rule, reproduced exactly
/// on the current codebase (§ this module's doc).
pub const STEP14_MODES: [bool; 4] = [true, false, true, false];
/// All four of Step 15's modes competing.
pub const STEP15_MODES: [bool; 4] = [true, true, true, true];

/// One sample under an explicit mode mask, plus the [`ModeStats`] histogram that encode
/// produced -- the mode-mask counterpart of `rd_opt::sample`.
pub fn sample_with_modes(
    image: &Plane,
    params: &EncodeParams,
    allowed_modes: [bool; 4],
) -> (RdSample, ModeStats) {
    // Gate 15 measures the historical fixed8 competition, not the new adaptive default.
    let options = EncodeOptions {
        allowed_modes,
        residual_quantisation: ResidualQuantisation::Fixed(ResidualQstep::LEGACY),
        ..EncodeOptions::default()
    };
    let (hdr, leaves, evals, stats) = encode_image_with_options(image, params, &options);
    let bytes = mars_format::write(&hdr, &leaves)
        .expect("a partition `encode_image_with_options` produced must always be writable");
    let (sample, _) = sample_from_bytes(image, &bytes, evals)
        .expect("the serialized `encode_image_with_options` partition must be readable");
    (sample, stats)
}

/// A λ-swept curve under a fixed mode mask, plus the mode-usage histogram summed across
/// every λ point on the sweep (the corpus/rate-wide picture the brief's exit criterion
/// asks for, not just one operating point's mix).
pub fn mode_curve(
    label: impl Into<String>,
    image: &Plane,
    base: &EncodeParams,
    lambda_grid: &[f64],
    allowed_modes: [bool; 4],
) -> (RdCurve, Vec<RdSample>, ModeStats) {
    let mut samples = Vec::with_capacity(lambda_grid.len());
    let mut total = ModeStats::default();
    for &lambda in lambda_grid {
        let params = EncodeParams {
            lambda: Some(lambda),
            ..*base
        };
        let (sample, stats) = sample_with_modes(image, &params, allowed_modes);
        for i in 0..4 {
            total.leaf_modes[i] += stats.leaf_modes[i];
        }
        total.split_decisions += stats.split_decisions;
        total.leaf_decisions += stats.leaf_decisions;
        samples.push(sample);
    }
    let curve = RdCurve::new(label, samples.iter().map(|s| s.point).collect());
    (curve, samples, total)
}

/// A human-readable one-line mode-usage summary: each mode's share of leaves, plus the
/// mode-4 (subdivide) share of leaf-vs-split decision points.
pub fn format_mode_histogram(stats: &ModeStats) -> String {
    let total_leaves: u64 = stats.leaf_modes.iter().sum();
    let pct = |n: u64| {
        if total_leaves == 0 {
            0.0
        } else {
            100.0 * n as f64 / total_leaves as f64
        }
    };
    let total_decisions = stats.split_decisions + stats.leaf_decisions;
    let split_pct = if total_decisions == 0 {
        0.0
    } else {
        100.0 * stats.split_decisions as f64 / total_decisions as f64
    };
    format!(
        "mode0(flat)={:.1}% mode1(affine)={:.1}% mode2(fractal)={:.1}% mode3(fractal+residual)={:.1}% \
         | mode4(subdivide) chosen at {:.1}% of {} leaf-vs-split decision points | {} leaves total",
        pct(stats.leaf_modes[0]),
        pct(stats.leaf_modes[1]),
        pct(stats.leaf_modes[2]),
        pct(stats.leaf_modes[3]),
        split_pct,
        total_decisions,
        total_leaves,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step14_mask_never_produces_modes_1_or_3() {
        let mut data = vec![0u8; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                data[r * 64 + c] = (((r * 13 + c * 7) % 256) as i32).clamp(0, 255) as u8;
            }
        }
        let image = Plane::from_vec(64, 64, data);
        let base = EncodeParams {
            min_size: 4,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            t_rms: 8.0,
            zero_threshold: 0,
            lambda: None,
        };
        let (_curve, _samples, stats) = mode_curve("test", &image, &base, &[100.0], STEP14_MODES);
        assert_eq!(
            stats.leaf_modes[1], 0,
            "mode 1 must never appear under the Step-14 mask"
        );
        assert_eq!(
            stats.leaf_modes[3], 0,
            "mode 3 must never appear under the Step-14 mask"
        );
    }
}
