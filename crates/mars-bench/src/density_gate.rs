//! Step 16's own comparison tools: content-adaptive domain-pool density
//! (`mars_codec::encode::encode_image_rd_with_modes_and_density`'s `adaptive_density`
//! flag), mirroring `mode_gate.rs`'s precedent exactly -- same corpus scope, same λ grid,
//! same same-codebase A/B discipline (`false` vs `true`, not a diff against a separate git
//! revision -- `docs/decisions.md`'s D40 has the reasoning for why that comparison is
//! fairer).
//!
//! Also carries this step's own two extra exit-criterion obligations the brief names
//! explicitly and prior gates did not need: **encode-time cost accounting** (wall-clock,
//! same machine/run/thread-count, alongside every BD-rate number -- not a
//! `benchmark-protocol`-grade throughput claim, just the brief's own "report the cost
//! alongside the gain" instruction) and **partition statistics** (leaves per depth, mean
//! block size vs. local variance).

use std::time::Instant;

use mars_codec::encode::{EncodeParams, ModeStats, encode_image_rd_with_modes_and_density};
use mars_codec::ifs::Leaf;
use mars_codec::mars_format;

use mars_core::Plane;

use crate::bdrate::RdCurve;
use crate::rd_opt::{RdSample, sample_from_bytes};

/// One operating point under an explicit `adaptive_density` flag: the usual RD sample plus
/// the mode histogram, the leaves themselves (for partition-statistics reporting), and the
/// wall-clock encode time this specific call took.
pub struct DensitySample {
    pub sample: RdSample,
    pub stats: ModeStats,
    pub leaves: Vec<Leaf>,
    pub encode_secs: f64,
}

/// Encode `image` once under `adaptive_density`, timing only the encode call itself (not
/// the subsequent bitstream write / decode / PSNR measurement, which are identical work on
/// both arms and would only dilute the comparison).
pub fn sample_with_density(
    image: &Plane,
    params: &EncodeParams,
    adaptive_density: bool,
) -> DensitySample {
    let t0 = Instant::now();
    let (hdr, leaves, evals, stats) =
        encode_image_rd_with_modes_and_density(image, params, [true; 4], adaptive_density);
    let encode_secs = t0.elapsed().as_secs_f64();

    let bytes = mars_format::write(&hdr, &leaves).expect(
        "a partition `encode_image_rd_with_modes_and_density` produced must always be writable",
    );
    let (sample, leaves) = sample_from_bytes(image, &bytes, evals)
        .expect("the serialized density partition must be readable");
    DensitySample {
        sample,
        stats,
        leaves,
        encode_secs,
    }
}

/// A λ-swept curve under a fixed `adaptive_density` flag: the RD curve, every sample (for
/// per-λ reporting), the mode histogram summed across the sweep, and the total wall-clock
/// encode time summed across every λ point -- the quantity Step 16's cost-accounting exit
/// criterion is read from.
pub fn density_curve(
    label: impl Into<String>,
    image: &Plane,
    base: &EncodeParams,
    lambda_grid: &[f64],
    adaptive_density: bool,
) -> (RdCurve, Vec<DensitySample>, ModeStats, f64) {
    let mut samples = Vec::with_capacity(lambda_grid.len());
    let mut total_stats = ModeStats::default();
    let mut total_secs = 0.0;
    for &lambda in lambda_grid {
        let params = EncodeParams {
            lambda: Some(lambda),
            ..*base
        };
        let s = sample_with_density(image, &params, adaptive_density);
        for i in 0..4 {
            total_stats.leaf_modes[i] += s.stats.leaf_modes[i];
        }
        total_stats.split_decisions += s.stats.split_decisions;
        total_stats.leaf_decisions += s.stats.leaf_decisions;
        total_secs += s.encode_secs;
        samples.push(s);
    }
    let curve = RdCurve::new(label, samples.iter().map(|s| s.sample.point).collect());
    (curve, samples, total_stats, total_secs)
}

/// Step 16's partition-statistics exit criterion: leaves bucketed by size (a proxy for
/// quadtree depth -- `depth = log2(max_size / size)`), and, within each size bucket, the
/// mean of each leaf's own local pixel-domain RMS (the same quantity
/// `mars_codec::encode::block_rms` computes internally, recomputed here from `(image,
/// leaf)` since that function is private to `mars-codec` and this is read-only reporting,
/// not a codec decision).
#[derive(Debug, Clone, Default)]
pub struct PartitionSummary {
    /// `(leaf_size, count, mean_local_rms)`, one row per distinct leaf size present,
    /// sorted smallest-size (deepest) first.
    pub by_size: Vec<(u32, u64, f64)>,
    pub total_leaves: u64,
    pub mean_leaf_size: f64,
}

fn leaf_rms(image: &Plane, leaf: &Leaf) -> f64 {
    let px = image.as_slice();
    let stride = image.width();
    let size = leaf.size as usize;
    if leaf.row as usize + size > image.height() || leaf.col as usize + size > image.width() {
        return 0.0; // a forced, image-edge-truncated leaf -- not a real content sample
    }
    let (mut t0, mut t2) = (0i64, 0i64);
    for i in 0..size {
        let src = (leaf.row as usize + i) * stride + leaf.col as usize;
        for j in 0..size {
            let v = i64::from(px[src + j]);
            t0 += v;
            t2 += v * v;
        }
    }
    let s0 = (size * size) as i64;
    let mean = t0 as f64 / s0 as f64;
    (t2 as f64 / s0 as f64 - mean * mean).max(0.0).sqrt()
}

pub fn partition_stats(image: &Plane, leaves: &[Leaf]) -> PartitionSummary {
    use std::collections::BTreeMap;
    let mut by_size: BTreeMap<u32, (u64, f64)> = BTreeMap::new();
    let mut total_size_sum = 0u64;
    for leaf in leaves {
        let rms = leaf_rms(image, leaf);
        let entry = by_size.entry(leaf.size).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += rms;
        total_size_sum += u64::from(leaf.size);
    }
    let mut by_size_vec: Vec<(u32, u64, f64)> = by_size
        .into_iter()
        .map(|(size, (count, rms_sum))| (size, count, rms_sum / count as f64))
        .collect();
    by_size_vec.sort_by_key(|&(size, ..)| size);
    let total_leaves = leaves.len() as u64;
    let mean_leaf_size = if total_leaves == 0 {
        0.0
    } else {
        total_size_sum as f64 / total_leaves as f64
    };
    PartitionSummary {
        by_size: by_size_vec,
        total_leaves,
        mean_leaf_size,
    }
}

pub fn format_partition_stats(s: &PartitionSummary) -> String {
    let mut out = format!(
        "{} leaves, mean size {:.2}px | by size (size: count, mean local RMS): ",
        s.total_leaves, s.mean_leaf_size
    );
    for (i, (size, count, mean_rms)) in s.by_size.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{size}px: {count} (rms={mean_rms:.2})"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_stats_totals_match_leaf_count_and_mean_size() {
        let image = Plane::from_vec(16, 16, vec![0u8; 16 * 16]);
        let leaves = vec![
            Leaf {
                row: 0,
                col: 0,
                size: 8,
                mode: 0,
                qalfa: 0,
                qbeta: 0,
                isometry: 0,
                dom_row: 0,
                dom_col: 0,
                qgx: 0,
                qgy: 0,
                residual: Vec::new(),
            },
            Leaf {
                row: 8,
                col: 0,
                size: 4,
                mode: 0,
                qalfa: 0,
                qbeta: 0,
                isometry: 0,
                dom_row: 0,
                dom_col: 0,
                qgx: 0,
                qgy: 0,
                residual: Vec::new(),
            },
        ];
        let stats = partition_stats(&image, &leaves);
        assert_eq!(stats.total_leaves, 2);
        assert!((stats.mean_leaf_size - 6.0).abs() < 1e-9);
        assert_eq!(stats.by_size.len(), 2);
    }
}
