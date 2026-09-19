//! Step 17 — learned candidate pruning (R&D plan's headline experiment,
//! `implementation-plan.md` ~line 993).
//!
//! Hypothesis: a lightweight model scoring `P(domain in top-k | range features, domain
//! features, relative position)` can narrow the domain pool before Stage 4's exact affine
//! fit more cheaply and more accurately than Step 13's hand-engineered [`crate::funnel`].
//!
//! **Model.** A from-scratch, dependency-free 2-layer MLP (`FEATURE_DIM` inputs -> 8
//! hidden units (tanh) -> 1 output (sigmoid)), trained offline by
//! `crates/mars-bench/examples/train_learned.rs` against the Step 8 oracle cache and baked
//! into this file as `TRAINED_WEIGHTS` (see that example's own doc for the seed and
//! training procedure). Baking the weights into the binary — rather than reading a file at
//! `index()` time — keeps this retriever free of runtime I/O and trivially deterministic
//! (§2.3: no wall-clock seeds, no path that could silently pick up a stale file), at the
//! cost of needing a rebuild to pick up a retrained model; `docs/decisions.md`'s Step 17
//! entry records this choice and the training seed explicitly.
//!
//! **Feature vector (`FEATURE_DIM = 10`), per (range block, candidate domain) pair.**
//! Built from [`crate::funnel`]'s own `Stage1Features` (`pub`, shared rather than
//! reimplemented — including by the offline training example in a different crate — so
//! every consumer scores domains from the identical cheap-feature definition) plus this
//! block's own mean/std ratio and the domain's position relative to the range block,
//! normalised by image size:
//!
//! ```text
//! [0..6): (range.stage1 - domain.stage1), component-wise (norm_min/max/range/grad_h/grad_v/edge)
//! [6]:    ln((domain_std + eps) / (range_std + eps))   -- contrast-ratio signal Stage1Features drops
//! [7]:    (domain_row - range_row) / image_height
//! [8]:    (domain_col - range_col) / image_width
//! [9]:    sqrt(feature[7]^2 + feature[8]^2)              -- relative-position magnitude
//! ```
//!
//! **Inference cost is real and counted, not hand-waved.** `candidates()` scores every
//! domain position in the pool with one MLP forward pass (`FEATURE_DIM * HIDDEN +
//! HIDDEN + HIDDEN + 1` multiply-adds each, ~97 per domain at this architecture) before
//! keeping the top `k` survivors — the same big-O shape as [`crate::funnel`]'s Stage 1,
//! and `crates/mars-bench/tests/learned_gate.rs` reports wall-clock time for this method
//! (feature computation + inference included) alongside `evals/transform`, exactly so a
//! model that costs more than it saves cannot hide behind the `evals` counter alone (the
//! brief's own explicit warning).

use crate::classify::Block;
use crate::funnel::{stage1_features, Stage1Features};
use crate::{Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock};

pub const FEATURE_DIM: usize = 10;
pub const HIDDEN_DIM: usize = 8;

/// A 2-layer MLP's weights: `FEATURE_DIM -> HIDDEN_DIM` (tanh) `-> 1` (sigmoid). Plain
/// flat arrays (row-major `w1[h][f]`), not a matrix crate — this project has no linear
/// algebra dependency and the model is small enough that hand-rolled loops are simpler
/// than adding one.
#[derive(Debug, Clone, Copy)]
pub struct MlpWeights {
    pub w1: [[f32; FEATURE_DIM]; HIDDEN_DIM],
    pub b1: [f32; HIDDEN_DIM],
    pub w2: [f32; HIDDEN_DIM],
    pub b2: f32,
}

impl MlpWeights {
    /// Forward pass -> a raw sigmoid score in `(0, 1)`, higher = more likely to be a true
    /// top-32 domain for this range block. `#[inline]` since this runs once per (range
    /// block, domain) pair — the hot loop this whole method exists to keep cheap.
    #[inline]
    pub fn score(&self, features: &[f32; FEATURE_DIM]) -> f32 {
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for (h, row) in self.w1.iter().enumerate() {
            let acc: f32 = self.b1[h] + row.iter().zip(features).map(|(w, x)| w * x).sum::<f32>();
            hidden[h] = acc.tanh();
        }
        let out: f32 = self.b2 + self.w2.iter().zip(&hidden).map(|(w, h)| w * h).sum::<f32>();
        1.0 / (1.0 + (-out).exp())
    }
}

/// Trained by `crates/mars-bench/examples/train_learned.rs` on `kodim01`'s oracle cache
/// (`default` config, size 16), seed `RNG_SEED = 0xC0FFEE_17` (see that example). Kept as
/// a `include!`'d file (plain Rust source, generated once and committed) rather than
/// typed out by hand here, so regenerating the model is "run the example, `git diff`
/// this file" and the diff shows exactly what changed.
pub mod trained {
    include!("learned_weights.rs");
}

/// Build the feature vector for one (range, domain) pair. `range_std`/`domain_std` are
/// each block's own raw pixel standard deviation (not part of `Stage1Features`, which is
/// already contrast-normalised and so has discarded absolute scale) — passed in
/// separately since both caller sites (`index()` for domains, `candidates()` for the
/// range block) already compute mean/std once per block for `stage1_features` itself.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn build_features(
    range: &Stage1Features,
    range_std: f64,
    domain: &Stage1Features,
    domain_std: f64,
    range_row: u32,
    range_col: u32,
    domain_row: u32,
    domain_col: u32,
    image_width: u32,
    image_height: u32,
) -> [f32; FEATURE_DIM] {
    const EPS: f64 = 1e-6;
    // Clamped: a near-flat block on either side sends this ratio towards +-14 (ln of a
    // ~1e6 ratio against EPS), which dominated every other feature and destabilised
    // training (`docs/decisions.md`'s Step 17 entry) until this was found and fixed --
    // the clamp keeps the *sign and rough magnitude* of the contrast-ratio signal while
    // removing the outlier's ability to blow up gradient descent.
    let log_std_ratio = ((domain_std + EPS) / (range_std + EPS))
        .ln()
        .clamp(-5.0, 5.0);
    let rel_dr = (f64::from(domain_row) - f64::from(range_row)) / f64::from(image_height.max(1));
    let rel_dc = (f64::from(domain_col) - f64::from(range_col)) / f64::from(image_width.max(1));
    let rel_dist = (rel_dr * rel_dr + rel_dc * rel_dc).sqrt();
    [
        (range.norm_min - domain.norm_min) as f32,
        (range.norm_max - domain.norm_max) as f32,
        (range.norm_range - domain.norm_range) as f32,
        (range.grad_h - domain.grad_h) as f32,
        (range.grad_v - domain.grad_v) as f32,
        (range.edge - domain.edge) as f32,
        log_std_ratio as f32,
        rel_dr as f32,
        rel_dc as f32,
        rel_dist as f32,
    ]
}

/// A block's own raw (non-normalised) mean and standard deviation — `Stage1Features`
/// deliberately drops this (affine-invariance, `crate::funnel`'s module doc), but the
/// learned model wants it back as `log_std_ratio`, one real scale signal Stage1-3 never
/// uses anywhere in this crate.
pub fn mean_std(b: &Block) -> (f64, f64) {
    let n = (b.size * b.size) as f64;
    let mean = b.data.iter().sum::<f64>() / n;
    let var = b.data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

struct DomainEntry {
    dom_row: u32,
    dom_col: u32,
    stage1: Stage1Features,
    std: f64,
}

/// How many domain positions survive `candidates()`'s scoring to reach Stage 4 (the exact
/// affine fit) — the one knob this method exposes, mirroring [`crate::funnel::FunnelConfig`]'s
/// role. `16` matches the R&D plan's own funnel endpoint (`10,000 -> 1,000 -> 100 -> 16`)
/// so the comparison against `Funnel` is apples-to-apples on final survivor count.
pub const DEFAULT_SURVIVORS: usize = 16;

pub struct Learned {
    weights: &'static MlpWeights,
    survivors: usize,
    domains: Vec<DomainEntry>,
    image_width: u32,
    image_height: u32,
}

impl Learned {
    pub fn new(weights: &'static MlpWeights, survivors: usize) -> Self {
        Self {
            weights,
            survivors,
            domains: Vec::new(),
            image_width: 1,
            image_height: 1,
        }
    }
}

impl Default for Learned {
    fn default() -> Self {
        Self::new(&trained::WEIGHTS, DEFAULT_SURVIVORS)
    }
}

impl CandidateRetriever for Learned {
    fn index(&mut self, pool: &DomainPool) {
        self.image_width = pool.image_width;
        self.image_height = pool.image_height;
        self.domains = pool
            .domain_positions()
            .map(|(dom_row, dom_col)| {
                let block = pool.domain_block(dom_row, dom_col);
                let stage1 = stage1_features(&block);
                let (_, std) = mean_std(&block);
                DomainEntry {
                    dom_row,
                    dom_col,
                    stage1,
                    std,
                }
            })
            .collect();
    }

    fn candidates(&self, range: &RangeBlock) -> CandidateIter<'_> {
        let block = range.as_block();
        let r1 = stage1_features(&block);
        let (_, r_std) = mean_std(&block);

        let mut scored: Vec<(f32, usize)> = self
            .domains
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let feats = build_features(
                    &r1,
                    r_std,
                    &d.stage1,
                    d.std,
                    range.row,
                    range.col,
                    d.dom_row,
                    d.dom_col,
                    self.image_width,
                    self.image_height,
                );
                (self.weights.score(&feats), i)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("scores are never NaN"));
        scored.truncate(self.survivors);

        let mut out = Vec::with_capacity(scored.len() * 8);
        for (_, i) in scored {
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
