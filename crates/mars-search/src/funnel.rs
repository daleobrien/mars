//! Step 13 — hierarchical funnel search (R&D plan §6): four narrowing stages per range
//! block, instead of Step 9's single bucket/tree lookup, so the exact affine fit (Stage 4,
//! shared [`crate::search_block`]) only ever runs on a small survivor set drawn from the
//! whole domain pool.
//!
//! **Isometry simplification (`docs/predictions.md` P13, written before this file).**
//! Stages 1-3 filter *domain positions* only, not `(domain, isometry)` pairs: `mean`,
//! `variance`, `min`, `max`, `range` are invariant to any isometry of a square block, and
//! this port treats the R&D plan's gradient/edge-energy and orientation features as a
//! rotation-agnostic magnitude/angle rather than porting eight per-orientation variants —
//! a documented simplification, not an oversight. Stage 4 still evaluates all 8 isometries
//! of each surviving domain, exactly like every other method in this crate.
//!
//! **Affine-invariance choice.** A domain reaches the range only after Stage 4's own
//! `alfa`/`beta` remap, so matching *raw* brightness/contrast between a domain and a range
//! block would filter out good candidates for the wrong reason. Every feature below is
//! computed on the block's own zero-mean, variance- (or norm-) normalised statistics, the
//! same invariance `classify::compute_saupe_vector` already relies on for Saupe/Mc-Saupe.
//!
//! **Stages**
//! - Stage 1 — six cheap normalised scalars (§6: mean/variance/min/max/range collapse to
//!   `norm_min/norm_max/norm_range` once contrast-normalised; plus horizontal/vertical/
//!   combined gradient energy).
//! - Stage 2 — quadrant means/variances (normalised) + three low-frequency 2D-DCT
//!   coefficients + one gradient-orientation angle.
//! - Stage 3 — [`classify::compute_saupe_vector`]-based thumbnails at 4×4 and (when the
//!   block is large enough) 8×8, L2-compared — the closest cheap proxy to the true SSD
//!   Stage 4 computes.
//! - Stage 4 — the shared exact affine fit, via [`crate::search_block`], on whichever
//!   domains survive Stage 3.
//!
//! Each stage's survivor count is [`FunnelConfig`], resolved once per `index()` call
//! against the pool's actual size (see [`FunnelConfig::scaled`]) and logged per block when
//! `MARS_FUNNEL_LOG` is set (`diag::log_block`) — the raw material for the survival/recall
//! tradeoff curve the Step 13 brief's exit criteria ask for.

use std::f64::consts::PI;

use crate::classify::{compute_saupe_vector, Block};
use crate::{Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock};

/// Per-stage survivor counts. Each stage keeps at most this many domains (by ascending
/// distance to the range block's own features); a count `>=` the number of domains still
/// in play is a no-op for that stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FunnelConfig {
    pub stage1_survivors: usize,
    pub stage2_survivors: usize,
    pub stage3_survivors: usize,
}

impl FunnelConfig {
    /// Disables every stage's narrowing (every domain survives to Stage 4) — the
    /// configuration the harness-sanity check (P13.3) uses to compare against
    /// [`crate::exhaustive::Exhaustive`] directly.
    pub fn disabled() -> Self {
        Self {
            stage1_survivors: usize::MAX,
            stage2_survivors: usize::MAX,
            stage3_survivors: usize::MAX,
        }
    }

    /// R&D plan §6's illustrative ratios (10,000 → 1,000 → 100 → 16, i.e. 10% → 10% of
    /// that → a fixed final count) rescaled to `pool_len` domain positions, since this
    /// project's pools (at `shift >= 4`) are far smaller than the plan's illustrative
    /// 10,000. Each stage is clamped so it never *widens* what the previous stage already
    /// narrowed, and every stage keeps at least one survivor.
    pub fn scaled(pool_len: usize) -> Self {
        let stage1 = (pool_len / 10).clamp(1, pool_len.max(1));
        let stage2 = (stage1 / 10).clamp(1, stage1);
        let stage3 = 16usize.clamp(1, stage2);
        Self {
            stage1_survivors: stage1,
            stage2_survivors: stage2,
            stage3_survivors: stage3,
        }
    }
}

/// How a [`Funnel`] resolves its [`FunnelConfig`] at `index()` time — deferred rather than
/// fixed at construction, since [`FunnelConfig::scaled`] needs the pool's actual size,
/// which isn't known until `index()` runs.
#[derive(Debug, Clone, Copy, Default)]
pub enum FunnelMode {
    /// [`FunnelConfig::scaled`], resolved against each size's own domain pool.
    #[default]
    Scaled,
    /// [`FunnelConfig::disabled`] — the harness-sanity configuration.
    Disabled,
    /// A caller-supplied, fixed configuration (e.g. for the survival/recall sweep the
    /// exit criteria ask for, run at several explicit survivor counts).
    Fixed(FunnelConfig),
}

/// One domain position's precomputed features, built once at `index()` time and reused
/// across every range block's `candidates()` call at this size.
struct DomainEntry {
    dom_row: u32,
    dom_col: u32,
    stage1: Stage1Features,
    stage2: Stage2Features,
    thumbs: Vec<Vec<f32>>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Stage1Features {
    norm_min: f64,
    norm_max: f64,
    norm_range: f64,
    grad_h: f64,
    grad_v: f64,
    edge: f64,
}

fn stage1_features(b: &Block) -> Stage1Features {
    let n = (b.size * b.size) as f64;
    let mean = b.data.iter().sum::<f64>() / n;
    let var = b.data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = var.sqrt();

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in &b.data {
        min = min.min(v);
        max = max.max(v);
    }

    let mut grad_h = 0.0;
    let mut grad_v = 0.0;
    for i in 0..b.size {
        for j in 0..b.size {
            if j + 1 < b.size {
                let d = b.at(i, j + 1) - b.at(i, j);
                grad_h += d * d;
            }
            if i + 1 < b.size {
                let d = b.at(i + 1, j) - b.at(i, j);
                grad_v += d * d;
            }
        }
    }

    // A flat block (std ~ 0) has no meaningful contrast-normalised shape; every
    // normalised feature is defined as 0 rather than dividing by ~0 (mirrors
    // `classify::compute_mc`'s `mass != 0.0` guard style).
    if std < 1e-9 {
        return Stage1Features::default();
    }
    let s2 = std * std;
    Stage1Features {
        norm_min: (min - mean) / std,
        norm_max: (max - mean) / std,
        norm_range: (max - min) / std,
        grad_h: grad_h / (n * s2),
        grad_v: grad_v / (n * s2),
        edge: (grad_h + grad_v) / (n * s2),
    }
}

fn stage1_distance(a: &Stage1Features, b: &Stage1Features) -> f64 {
    let d = [
        a.norm_min - b.norm_min,
        a.norm_max - b.norm_max,
        a.norm_range - b.norm_range,
        a.grad_h - b.grad_h,
        a.grad_v - b.grad_v,
        a.edge - b.edge,
    ];
    d.iter().map(|x| x * x).sum()
}

#[derive(Debug, Clone, Copy, Default)]
struct Stage2Features {
    quad_mean: [f64; 4],
    quad_var: [f64; 4],
    dct01: f64,
    dct10: f64,
    dct11: f64,
    orientation: f64,
}

/// `[TL, TR, BL, BR]` mean and variance over the top-level block's four quadrants,
/// contrast-normalised by the whole block's own mean/std (so a domain and a range block
/// with matching *shape* but different brightness/contrast compare as close, per the
/// module doc's affine-invariance choice).
fn quadrant_stats(b: &Block, mean: f64, std: f64) -> ([f64; 4], [f64; 4]) {
    let h = b.size / 2;
    let mut sum = [0.0f64; 4];
    let mut sum2 = [0.0f64; 4];
    for i in 0..h {
        for j in 0..h {
            let quads = [(i, j), (i, j + h), (i + h, j), (i + h, j + h)];
            for (k, &(qi, qj)) in quads.iter().enumerate() {
                let v = b.at(qi, qj);
                sum[k] += v;
                sum2[k] += v * v;
            }
        }
    }
    let n = (h * h) as f64;
    let mut means = [0.0f64; 4];
    let mut vars = [0.0f64; 4];
    for k in 0..4 {
        let m = sum[k] / n;
        let v = (sum2[k] / n - m * m).max(0.0);
        if std < 1e-9 {
            means[k] = 0.0;
            vars[k] = 0.0;
        } else {
            means[k] = (m - mean) / std;
            vars[k] = v / (std * std);
        }
    }
    (means, vars)
}

/// A low-frequency 2D DCT-II coefficient `(u, v)` of the block's own contrast-normalised
/// samples — a cheap proxy for "structural signature" (R&D plan §6's "DCT low-frequency
/// coefficients"). Only ever called with small `u, v` (the low-frequency AC terms), so the
/// direct `O(size^2)` sum is cheap relative to Stage 4's exact fit.
fn dct_coeff(b: &Block, mean: f64, std: f64, u: usize, v: usize) -> f64 {
    if std < 1e-9 {
        return 0.0;
    }
    let n = b.size as f64;
    let mut acc = 0.0;
    for i in 0..b.size {
        for j in 0..b.size {
            let x = (b.at(i, j) - mean) / std;
            let cu = (PI / n * (i as f64 + 0.5) * u as f64).cos();
            let cv = (PI / n * (j as f64 + 0.5) * v as f64).cos();
            acc += x * cu * cv;
        }
    }
    acc / n
}

fn stage2_features(b: &Block) -> Stage2Features {
    let n = (b.size * b.size) as f64;
    let mean = b.data.iter().sum::<f64>() / n;
    let var = b.data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let std = var.sqrt();

    let (quad_mean, quad_var) = quadrant_stats(b, mean, std);
    let dct01 = dct_coeff(b, mean, std, 0, 1);
    let dct10 = dct_coeff(b, mean, std, 1, 0);
    let dct11 = dct_coeff(b, mean, std, 1, 1);

    let mut gx = 0.0;
    let mut gy = 0.0;
    for i in 0..b.size {
        for j in 0..b.size {
            if j + 1 < b.size {
                gx += b.at(i, j + 1) - b.at(i, j);
            }
            if i + 1 < b.size {
                gy += b.at(i + 1, j) - b.at(i, j);
            }
        }
    }
    let orientation = gy.atan2(gx);

    Stage2Features {
        quad_mean,
        quad_var,
        dct01,
        dct10,
        dct11,
        orientation,
    }
}

/// Smallest angular difference between two angles, in `[0, PI]` — orientation is a
/// direction, not a scalar, so a plain subtraction would treat angles near `0`/`2*PI` as
/// maximally distant when they are in fact adjacent.
fn angle_distance(a: f64, b: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let d = (a - b).rem_euclid(two_pi);
    d.min(two_pi - d)
}

fn stage2_distance(a: &Stage2Features, b: &Stage2Features) -> f64 {
    let mut d = 0.0;
    for k in 0..4 {
        let dm = a.quad_mean[k] - b.quad_mean[k];
        let dv = a.quad_var[k] - b.quad_var[k];
        d += dm * dm + dv * dv;
    }
    d += (a.dct01 - b.dct01).powi(2);
    d += (a.dct10 - b.dct10).powi(2);
    d += (a.dct11 - b.dct11).powi(2);
    let ad = angle_distance(a.orientation, b.orientation);
    d += ad * ad;
    d
}

/// Stage 3's thumbnails: [`compute_saupe_vector`] (already zero-mean, unit-L2-norm, so
/// contrast-invariant by construction) at 4×4 and, when the block is large enough, 8×8 —
/// reusing Saupe's own feature vector rather than a bespoke downsampler, per the module
/// doc's affine-invariance choice.
fn thumbnails(b: &Block) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(2);
    if b.size >= 4 {
        out.push(compute_saupe_vector(b, (b.size / 4).max(1)));
    }
    if b.size >= 8 {
        out.push(compute_saupe_vector(b, (b.size / 8).max(1)));
    }
    out
}

fn thumbnail_distance(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    let mut total = 0.0f64;
    for (ta, tb) in a.iter().zip(b.iter()) {
        for (&x, &y) in ta.iter().zip(tb.iter()) {
            let d = f64::from(x - y);
            total += d * d;
        }
    }
    total
}

fn sort_and_truncate(idx: &mut Vec<usize>, keep: usize, mut dist: impl FnMut(usize) -> f64) {
    if keep >= idx.len() {
        return;
    }
    idx.sort_by(|&a, &b| {
        dist(a)
            .partial_cmp(&dist(b))
            .expect("features are never NaN")
    });
    idx.truncate(keep);
}

/// The Step 13 funnel: `index()` builds every domain's Stage 1-3 features once per size;
/// `candidates()` narrows the whole pool down to a small survivor set per range block,
/// stage by stage, and hands the survivors' 8 isometries each to Stage 4
/// ([`crate::search_block`]) — see the module doc for the full stage-by-stage design.
pub struct Funnel {
    mode: FunnelMode,
    resolved: FunnelConfig,
    domains: Vec<DomainEntry>,
}

impl Funnel {
    pub fn new(mode: FunnelMode) -> Self {
        Self {
            mode,
            resolved: FunnelConfig::disabled(),
            domains: Vec::new(),
        }
    }
}

impl Default for Funnel {
    fn default() -> Self {
        Self::new(FunnelMode::default())
    }
}

impl CandidateRetriever for Funnel {
    fn index(&mut self, pool: &DomainPool) {
        let positions: Vec<(u32, u32)> = pool.domain_positions().collect();
        self.resolved = match self.mode {
            FunnelMode::Scaled => FunnelConfig::scaled(positions.len()),
            FunnelMode::Disabled => FunnelConfig::disabled(),
            FunnelMode::Fixed(c) => c,
        };
        self.domains = positions
            .into_iter()
            .map(|(dom_row, dom_col)| {
                let block = pool.domain_block(dom_row, dom_col);
                DomainEntry {
                    dom_row,
                    dom_col,
                    stage1: stage1_features(&block),
                    stage2: stage2_features(&block),
                    thumbs: thumbnails(&block),
                }
            })
            .collect();
    }

    fn candidates(&self, range: &RangeBlock) -> CandidateIter<'_> {
        let block = range.as_block();
        let r1 = stage1_features(&block);
        let r2 = stage2_features(&block);
        let r3 = thumbnails(&block);

        let pool_len = self.domains.len();
        let mut idx: Vec<usize> = (0..pool_len).collect();

        sort_and_truncate(&mut idx, self.resolved.stage1_survivors, |i| {
            stage1_distance(&r1, &self.domains[i].stage1)
        });
        let after1 = idx.len();

        sort_and_truncate(&mut idx, self.resolved.stage2_survivors, |i| {
            stage2_distance(&r2, &self.domains[i].stage2)
        });
        let after2 = idx.len();

        sort_and_truncate(&mut idx, self.resolved.stage3_survivors, |i| {
            thumbnail_distance(&r3, &self.domains[i].thumbs)
        });
        let after3 = idx.len();

        if diag::enabled() {
            diag::log_block(
                range.row, range.col, range.size, pool_len, after1, after2, after3,
            );
        }

        let mut out = Vec::with_capacity(idx.len() * 8);
        for i in idx {
            let d = &self.domains[i];
            for isometry in 0u8..8 {
                out.push(Candidate {
                    dom_row: d.dom_row,
                    dom_col: d.dom_col,
                    isometry,
                });
            }
        }
        CandidateIter::new(out)
    }
}

/// Per-block survivor logging (Step 13 brief: "each stage's survivor count is ...
/// logged per block"), opt-in via `MARS_FUNNEL_LOG=<path>` — the raw material for the
/// survival/recall tradeoff curve the exit criteria ask for, reconstructed offline by
/// joining this log against the Step 8 oracle's recall score for the same blocks, rather
/// than computing the curve inline (which would need this crate to depend on
/// `mars-bench`, the wrong direction — see [`crate::Pick`]'s own doc).
pub mod diag {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    static LOG: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

    fn log_file() -> &'static Option<Mutex<std::fs::File>> {
        LOG.get_or_init(|| {
            std::env::var("MARS_FUNNEL_LOG").ok().map(|path| {
                let f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .unwrap_or_else(|e| panic!("MARS_FUNNEL_LOG={path}: {e}"));
                Mutex::new(f)
            })
        })
    }

    pub fn enabled() -> bool {
        log_file().is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn log_block(row: u32, col: u32, size: u32, pool: usize, s1: usize, s2: usize, s3: usize) {
        if let Some(m) = log_file() {
            let line = format!(
                "{{\"row\":{row},\"col\":{col},\"size\":{size},\"pool\":{pool},\
                 \"stage1\":{s1},\"stage2\":{s2},\"stage3\":{s3}}}\n"
            );
            let mut f = m.lock().expect("MARS_FUNNEL_LOG mutex poisoned");
            let _ = f.write_all(line.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search_block;
    use mars_codec::encode::build_contracted;
    use mars_codec::encode::EncodeParams;
    use mars_core::Plane;

    fn synthetic_plane(w: usize, h: usize, seed: u64) -> Plane {
        let mut s = seed;
        let mut data = vec![0u8; w * h];
        for b in &mut data {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s & 0xff) as u8;
        }
        Plane::from_vec(w, h, data)
    }

    fn range_pixels(image: &Plane, row: u32, col: u32, size: u32) -> Vec<u8> {
        let size_u = size as usize;
        let px = image.as_slice();
        let stride = image.width();
        let mut pixels = vec![0u8; size_u * size_u];
        for i in 0..size_u {
            let src = (row as usize + i) * stride + col as usize;
            pixels[i * size_u..(i + 1) * size_u].copy_from_slice(&px[src..src + size_u]);
        }
        pixels
    }

    /// P13.3 — the harness sanity check: with narrowing disabled, the funnel must
    /// reproduce `Exhaustive`'s exact candidate coverage (same evals, same winner) per
    /// block, since every domain reaches Stage 4 unfiltered.
    #[test]
    fn disabled_funnel_matches_exhaustive_search_exactly() {
        let image = synthetic_plane(32, 32, 0xC0FFEE);
        let contracted = build_contracted(&image);
        let params = EncodeParams {
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

        for size in [4u32, 8, 16] {
            let pool = DomainPool {
                contracted: &contracted,
                size,
                shift: params.shift,
                image_width: image.width() as u32,
                image_height: image.height() as u32,
            };
            let mut funnel = Funnel::new(FunnelMode::Disabled);
            funnel.index(&pool);
            let mut exhaustive = crate::exhaustive::Exhaustive::default();
            exhaustive.index(&pool);

            for row in (0..image.height() as u32 - size + 1).step_by(size as usize) {
                for col in (0..image.width() as u32 - size + 1).step_by(size as usize) {
                    let pixels = range_pixels(&image, row, col, size);
                    let range = RangeBlock {
                        row,
                        col,
                        size,
                        pixels: &pixels,
                    };

                    let (got, got_evals) = search_block(
                        &funnel,
                        &contracted,
                        &range,
                        params.max_alfa,
                        params.bits_alfa,
                        params.bits_beta,
                    );
                    let (want, want_evals) = search_block(
                        &exhaustive,
                        &contracted,
                        &range,
                        params.max_alfa,
                        params.bits_alfa,
                        params.bits_beta,
                    );

                    assert_eq!(got_evals, want_evals, "evals at ({row},{col},{size})");
                    match (got, want) {
                        (None, None) => {}
                        (Some((c, qa, qb, rms)), Some(w)) => {
                            assert_eq!(
                                (c.dom_row, c.dom_col, c.isometry, qa, qb),
                                (w.0.dom_row, w.0.dom_col, w.0.isometry, w.1, w.2)
                            );
                            assert!((rms - w.3).abs() < 1e-9);
                        }
                        other => panic!("mismatched Option at ({row},{col},{size}): {other:?}"),
                    }
                }
            }
        }
    }

    /// Every stage must never *widen* the survivor set, and the final survivor count must
    /// respect the configured Stage 3 cap — the funnel's basic narrowing contract, checked
    /// directly rather than only inferred from a recall number.
    #[test]
    fn stages_narrow_monotonically_and_respect_the_configured_caps() {
        let image = synthetic_plane(64, 64, 0xBADA55);
        let contracted = build_contracted(&image);
        let size = 8u32;
        let pool = DomainPool {
            contracted: &contracted,
            size,
            shift: 4,
            image_width: image.width() as u32,
            image_height: image.height() as u32,
        };
        let config = FunnelConfig {
            stage1_survivors: 20,
            stage2_survivors: 8,
            stage3_survivors: 3,
        };
        let mut funnel = Funnel::new(FunnelMode::Fixed(config));
        funnel.index(&pool);
        assert!(
            funnel.domains.len() > 20,
            "test needs a pool bigger than stage1's cap"
        );

        let pixels = range_pixels(&image, 0, 0, size);
        let range = RangeBlock {
            row: 0,
            col: 0,
            size,
            pixels: &pixels,
        };
        let candidates: Vec<_> = funnel.candidates(&range).collect();
        // 3 surviving domains x 8 isometries each.
        assert_eq!(candidates.len(), 3 * 8);
        let distinct_domains: std::collections::HashSet<(u32, u32)> =
            candidates.iter().map(|c| (c.dom_row, c.dom_col)).collect();
        assert_eq!(distinct_domains.len(), 3);
    }

    /// `FunnelConfig::scaled` must never widen a later stage past an earlier one, and must
    /// always leave at least one survivor even for a tiny pool (the R&D plan's ratios,
    /// naively applied, could otherwise round a small pool down to zero survivors).
    #[test]
    fn scaled_config_is_monotonic_and_never_empty() {
        for pool_len in [1usize, 2, 7, 16, 100, 5000] {
            let c = FunnelConfig::scaled(pool_len);
            assert!(c.stage1_survivors >= 1);
            assert!(c.stage2_survivors >= 1);
            assert!(c.stage3_survivors >= 1);
            assert!(c.stage1_survivors <= pool_len.max(1));
            assert!(c.stage2_survivors <= c.stage1_survivors);
            assert!(c.stage3_survivors <= c.stage2_survivors);
        }
    }
}
