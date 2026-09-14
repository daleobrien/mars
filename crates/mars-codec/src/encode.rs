//! The exhaustive Rust encoder — Step 6.
//!
//! Partition (§5's split rule, driven by RMS rather than a speed-up method's restricted
//! candidate set), domain pool (§6's legal range), all 8 isometries (§9), integer-exact
//! moments and the affine fit + quantisation (§8, including the §8.1 zero-alfa override).
//! No speed-up method: every valid `(domain, isometry)` pair is evaluated for every range
//! block, which is exactly what `evals` (§M5) counts.
//!
//! The numerics decision (`implementation-plan.md` Step 6) is the point of this module:
//! range pixels are `u8`, but domain samples are accumulated as the *sum* of a 2x2 pixel
//! block (`D = 4d`, an integer 0..=1020) rather than their floating-point mean, which makes
//! every moment below an exact integer. Only the final fit — `alfa`, `beta`, `rms` — needs
//! floating point, and [`fit_f64`]/[`fit_f32`] exist side by side so that choice can be
//! measured rather than assumed (`just gate-6`'s f32-vs-f64 divergence check).

use mars_core::Plane;

use crate::ifs::{Header, Leaf};
use crate::isometry;
use crate::mars_format::{leaf_events, FIELD_SPLIT};
use crate::rate::RateModels;

/// What the encoder needs beyond the header fields `docs/mars1-format.md` already names.
#[derive(Debug, Clone, Copy)]
pub struct EncodeParams {
    pub min_size: u32,
    pub max_size: u32,
    pub shift: u32,
    pub bits_alfa: u32,
    pub bits_beta: u32,
    pub max_alfa: f64,
    /// `-r`, the split threshold: a block splits when its best RMS exceeds this and it is
    /// larger than `min_size`. Step 14: this remains the split rule for `mars-search`'s
    /// candidate-restriction methods (Step 9/13), which are out of this step's scope, but
    /// `mars-codec`'s own [`walk`] switches to [`Self::lambda`]-driven bottom-up RD
    /// pruning whenever `lambda` is `Some` -- see `docs/decisions.md`'s Step 14 entry for
    /// why `t_rms` was kept rather than removed.
    pub t_rms: f64,
    /// `-z`. The 1998 default is 0, meaning the override (§8.1) fires exactly when
    /// `qalfa == 0`.
    pub zero_threshold: u32,
    /// Step 14's `J = D + λR` quality knob. `None` preserves the legacy `t_rms`
    /// top-down threshold split exactly (used by every pre-Step-14 caller and test, and
    /// by `mars-search`'s own methods); `Some(lambda)` switches [`walk`] to bottom-up RD
    /// pruning: search/code the four children first, then keep whichever of "this block
    /// as one leaf" or "the four children as they stand" has the smaller `D + lambda*R`.
    pub lambda: Option<f64>,
}

/// The six integer moments of one candidate fit, kept alongside the winning candidate so
/// the f32-vs-f64 divergence measurement can redo the fit at either precision without
/// re-running the search. `s1`/`s2`/`t1` are stored as `4x` and `16x` their mathematical
/// value (§ the module doc) — exact integers where the true quantities are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawMoments {
    pub s0: i64,
    pub s1_x4: i64,
    pub s2_x16: i64,
    pub t0: i64,
    pub t1_x4: i64,
    pub t2: i64,
}

/// A candidate (or winning) fit for one range block.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub dom_row: u32,
    pub dom_col: u32,
    pub isometry: u8,
    pub qalfa: u32,
    pub qbeta: u32,
    pub rms: f64,
    pub moments: RawMoments,
}

/// §8's fit and quantisation, computed in `f64`. Returns `(qalfa, qbeta, rms)`.
pub fn fit_f64(m: RawMoments, max_alfa: f64, bits_alfa: u32, bits_beta: u32) -> (u32, u32, f64) {
    let s0 = m.s0 as f64;
    let s1 = m.s1_x4 as f64 / 4.0;
    let s2 = m.s2_x16 as f64 / 16.0;
    let t0 = m.t0 as f64;
    let t1 = m.t1_x4 as f64 / 4.0;
    let t2 = m.t2 as f64;

    let det = s0 * s2 - s1 * s1;
    let mut alfa = if det == 0.0 {
        0.0
    } else {
        (s0 * t1 - s1 * t0) / det
    };
    if alfa < 0.0 {
        alfa = 0.0;
    }

    let max_qalfa = (1u32 << bits_alfa) - 1;
    let qalfa = quantise_f64(alfa / max_alfa * f64::from(1u32 << bits_alfa), max_qalfa);
    let alfa2 = f64::from(qalfa) / f64::from(1u32 << bits_alfa) * max_alfa;

    let mut beta = (t0 - alfa2 * s1) / s0;
    if alfa2 > 0.0 {
        beta += alfa2 * 255.0;
    }
    let max_qbeta = (1u32 << bits_beta) - 1;
    let qbeta = quantise_f64(
        beta / ((1.0 + alfa2.abs()) * 255.0) * f64::from(max_qbeta),
        max_qbeta,
    );
    let mut beta2 = f64::from(qbeta) / f64::from(max_qbeta) * ((1.0 + alfa2.abs()) * 255.0);
    if alfa2 > 0.0 {
        beta2 -= alfa2 * 255.0;
    }

    let sum = t2 - 2.0 * alfa2 * t1 - 2.0 * beta2 * t0
        + alfa2 * alfa2 * s2
        + 2.0 * alfa2 * beta2 * s1
        + s0 * beta2 * beta2;
    let rms = (sum / s0).max(0.0).sqrt();
    (qalfa, qbeta, rms)
}

/// The same fit, in `f32` — for measuring how often it picks a different `qalfa`/`qbeta`
/// than [`fit_f64`] (`just gate-6`'s exit criterion). The moments themselves stay the
/// integer-exact `i64` values computed once by the search; only the fit arithmetic changes
/// precision.
pub fn fit_f32(m: RawMoments, max_alfa: f32, bits_alfa: u32, bits_beta: u32) -> (u32, u32, f32) {
    let s0 = m.s0 as f32;
    let s1 = m.s1_x4 as f32 / 4.0;
    let s2 = m.s2_x16 as f32 / 16.0;
    let t0 = m.t0 as f32;
    let t1 = m.t1_x4 as f32 / 4.0;
    let t2 = m.t2 as f32;

    let det = s0 * s2 - s1 * s1;
    let mut alfa = if det == 0.0 {
        0.0
    } else {
        (s0 * t1 - s1 * t0) / det
    };
    if alfa < 0.0 {
        alfa = 0.0;
    }

    let max_qalfa = (1u32 << bits_alfa) - 1;
    let qalfa = quantise_f32(alfa / max_alfa * (1u32 << bits_alfa) as f32, max_qalfa);
    let alfa2 = qalfa as f32 / (1u32 << bits_alfa) as f32 * max_alfa;

    let mut beta = (t0 - alfa2 * s1) / s0;
    if alfa2 > 0.0 {
        beta += alfa2 * 255.0;
    }
    let max_qbeta = (1u32 << bits_beta) - 1;
    let qbeta = quantise_f32(
        beta / ((1.0 + alfa2.abs()) * 255.0) * max_qbeta as f32,
        max_qbeta,
    );
    let mut beta2 = qbeta as f32 / max_qbeta as f32 * ((1.0 + alfa2.abs()) * 255.0);
    if alfa2 > 0.0 {
        beta2 -= alfa2 * 255.0;
    }

    let sum = t2 - 2.0 * alfa2 * t1 - 2.0 * beta2 * t0
        + alfa2 * alfa2 * s2
        + 2.0 * alfa2 * beta2 * s1
        + s0 * beta2 * beta2;
    let rms = (sum / s0).max(0.0).sqrt();
    (qalfa, qbeta, rms)
}

// ------------------------------------------------------------------ Step 15: mode constants
//
// Modes 1 (affine) and 3 (fractal + residual) need a couple of quantisation parameters
// that have no header field of their own (`docs/decisions.md`'s Step 15 entry records the
// scope decision): rather than growing `.mars` v0's fixed header to carry a per-image
// residual quantisation step (which would also mean threading it through every
// `mars_format`/`ifs` call site that touches a `Header`), Step 15 fixes these as compile-
// time constants shared by encoder and decoder alike, the same way `crate::residual`'s own
// `LEVEL_CLAMP` is fixed. This means residual quantisation is not λ-adaptive this step —
// a real, documented simplification, not a silent one — but the RD search still decides,
// per block and per λ, whether paying for a mode-3 residual actually beats every other
// mode's `J` at this fixed step, which is exactly the brief's "let J report honestly
// whether residuals earn their bits".

/// Mode 1's gradient fixed-point scale: `qgx`/`qgy` store `round(gradient * SCALE)`, so a
/// gradient of e.g. 2.0 (steep contrast across even a 16-wide leaf) round-trips to a
/// non-degenerate integer instead of being crushed by too coarse a scale.
pub(crate) const AFFINE_GRAD_SCALE: f64 = 64.0;
/// Generous enough that no plane fit this project's fixtures produce is ever clamped
/// (`+-2048/64 = +-32` intensity units per pixel step, far beyond any real image's local
/// gradient) — the clamp exists only to bound the bitstream alphabet, not to bite in
/// practice.
pub(crate) const AFFINE_GRAD_CLAMP: i32 = 2048;

// ------------------------------------------------------------------ Step 16: adaptive
// domain-pool density
//
// `search`'s domain-position stride is, before this step, a single global constant
// (`params.shift`) applied everywhere regardless of content. Step 16 varies it per block:
// a denser stride (more candidate domain positions) where the block's own pixel-domain
// complexity is high and a better match is more likely to be worth the extra evals, a
// sparser stride where the block is nearly flat and every nearby domain position fits
// about equally well. This reuses the *existing* `FIELD_DOM_ROW`/`FIELD_DOM_COL` bitstream
// fields (only their coded *values* change, not the field vocabulary), which is why this
// is a materially lower-risk change than Step 15's new leaf modes were against the frozen
// `RateModels` snapshot (`docs/decisions.md`'s D40): there is no new field type for the
// snapshot to have zero observations of.

/// Pixel-domain RMS at or above this threshold routes a block to the denser
/// (half-`params.shift`) search stride. `16.0` sits above `RD_WARMUP_T_RMS` (`8.0`) --
/// deliberately higher, since this threshold gates *domain-search density*, a strictly
/// more expensive knob than the warm-up's own leaf-vs-split threshold, and should only
/// fire for blocks whose own pixel content (not just their eventual fractal-fit residual)
/// is genuinely busy.
const DENSITY_HIGH_RMS: f64 = 16.0;
/// Pixel-domain RMS at or below this threshold routes a block to the sparser
/// (double-`params.shift`) search stride -- a near-flat block, where doubling the domain
/// stride costs almost nothing in fit quality (every nearby domain position is itself
/// nearly flat) but roughly quarters the eval count spent searching it.
const DENSITY_LOW_RMS: f64 = 4.0;

/// This block's own pixel-domain RMS (population standard deviation of its `size x size`
/// pixels), computed once, before the domain search runs, as the pre-search proxy for
/// "local complexity" [`adaptive_shift`] routes on. Deliberately the *range* block's raw
/// pixel statistics, not a fractal-fit residual (which is not known until after a domain
/// search has already been spent) -- the same `t0`/`t2` accumulation [`search_with_shift`]
/// performs internally for its own moments, duplicated here (an O(size²) pass, size <= 16
/// in every config this project runs, so negligible next to the domain search itself) so
/// the density decision can be made *before* that search runs at all.
fn block_rms(image: &Plane, row: u32, col: u32, size: u32) -> f64 {
    let px = image.as_slice();
    let stride = image.width();
    let size_u = size as usize;
    let (mut t0, mut t2) = (0i64, 0i64);
    for i in 0..size_u {
        let src = (row as usize + i) * stride + col as usize;
        for j in 0..size_u {
            let v = i64::from(px[src + j]);
            t0 += v;
            t2 += v * v;
        }
    }
    let s0 = i64::from(size) * i64::from(size);
    let mean = t0 as f64 / s0 as f64;
    let variance = (t2 as f64 / s0 as f64 - mean * mean).max(0.0);
    variance.sqrt()
}

/// Map a block's own [`block_rms`] to a domain-search stride, relative to the run's base
/// `params.shift`: denser (half, floor 1) above [`DENSITY_HIGH_RMS`], sparser (double)
/// below [`DENSITY_LOW_RMS`], unchanged in between. Pure function of already-computed
/// scalars -- no shared state, so calling it from parallel `rayon::join` subtrees
/// introduces no thread-count dependence (the same determinism argument `crate::rate`'s
/// module doc makes for the frozen `RateModels` snapshot).
fn adaptive_shift(base_shift: u32, rms: f64) -> u32 {
    if rms >= DENSITY_HIGH_RMS {
        (base_shift / 2).max(1)
    } else if rms <= DENSITY_LOW_RMS {
        base_shift.saturating_mul(2)
    } else {
        base_shift
    }
}

/// Mode 3's fixed dead-zone quantisation step for DCT coefficients (`crate::quant`). `8.0`
/// is a mid-range choice on an 8-bit pixel scale: large enough that most AC coefficients of
/// a well-predicted fractal residual quantise to zero (residual coding should cost little
/// when the fractal prediction is already good), small enough that a genuinely bad fractal
/// match still has room to correct itself.
pub(crate) const RESIDUAL_QSTEP_DEFAULT: f64 = 8.0;
/// Standard JPEG-style dead-zone width (half the step) — mode 3's coefficient quantiser
/// deliberately biases small values to exactly zero rather than +-1 (see `crate::quant`'s
/// module doc).
pub(crate) const RESIDUAL_DEAD_ZONE: f64 = 0.5;

/// `int(0.5 + x)` (C truncation toward zero, `x >= 0` here since `alfa`/`beta` are
/// pre-clamped to be non-negative before this point) then clamped to `[0, max]`.
fn quantise_f64(x: f64, max: u32) -> u32 {
    (0.5 + x).trunc().clamp(0.0, f64::from(max)) as u32
}

fn quantise_f32(x: f32, max: u32) -> u32 {
    (0.5 + x).trunc().clamp(0.0, max as f32) as u32
}

/// §8.1: the block mean, quantised with `alfa` fixed at 0 — what a DC-only leaf's `qbeta`
/// is refit to, discarding whatever the search found.
fn best_beta(range_sum: i64, pixel_count: i64, bits_beta: u32) -> u32 {
    let mean = range_sum as f64 / pixel_count as f64;
    let max_qbeta = (1u32 << bits_beta) - 1;
    quantise_f64(mean / 255.0 * f64::from(max_qbeta), max_qbeta)
}

/// The 2:1 box-sum contraction of the whole image, computed once: `at(row, col)` is
/// `D(row, col)` of §9 (the *sum*, not the mean, of the four pixels at
/// `(2row,2col),(2row+1,2col),(2row,2col+1),(2row+1,2col+1)`), so that every domain
/// position's `size x size` sample block is an O(size²) slice of this rather than a
/// redundant box-sum recomputation per candidate.
pub struct Contracted {
    data: Vec<i32>,
    stride: usize,
}

impl Contracted {
    /// Step 12: each output row is an independent reduction over its own two input rows
    /// (no cross-row state), so this is exact data parallelism -- one Rayon task per
    /// output row, each writing only its own `stride`-wide slice, with no possibility of
    /// a different result at any thread count (every row computes the same sum either
    /// way; only which thread does it changes).
    fn build(image: &Plane) -> Self {
        use rayon::prelude::*;

        let (w, h) = (image.width(), image.height());
        let (stride, rows) = (w / 2, h / 2);
        let px = image.as_slice();
        let mut data = vec![0i32; stride * rows];
        data.par_chunks_mut(stride)
            .enumerate()
            .for_each(|(i, row_out)| {
                let r0 = 2 * i;
                for (j, out) in row_out.iter_mut().enumerate() {
                    let c0 = 2 * j;
                    *out = i32::from(px[r0 * w + c0])
                        + i32::from(px[r0 * w + c0 + 1])
                        + i32::from(px[(r0 + 1) * w + c0])
                        + i32::from(px[(r0 + 1) * w + c0 + 1]);
                }
            });
        Self { data, stride }
    }

    /// The 2:1 box-sum `D(row, col)` at one contracted-image position. `pub` (Step 9):
    /// `mars-search`'s candidate-restriction methods need direct access to the same
    /// contracted plane the exhaustive search uses, rather than recomputing it.
    pub fn at(&self, row: usize, col: usize) -> i32 {
        self.data[row * self.stride + col]
    }

    /// Contracted-plane dimensions (`image width/height / 2`), for callers that need to
    /// bound-check domain positions without re-deriving them from the original `Plane`.
    pub fn width(&self) -> usize {
        self.stride
    }

    pub fn height(&self) -> usize {
        self.data.len().checked_div(self.stride).unwrap_or(0)
    }

    /// The raw `(plane, stride)` pair backing `.at` -- `pub` (Step 11): `mars-bench`'s
    /// SIMD speedup report needs to hand the same strided plane [`domain_sums`]/
    /// [`cross_term`] already use to `mars_simd` kernels directly, rather than rebuilding
    /// one element-by-element through `.at` just to time it.
    pub fn raw(&self) -> (&[i32], usize) {
        (&self.data, self.stride)
    }
}

/// Public entry point for [`search`], for callers outside this crate that need the
/// ground-truth per-block result without running the full quadtree `walk` — Step 7's GPU
/// differential test (`mars-gpu`/`marsbench`) is the reason this exists: it needs to zip
/// the CPU's per-`(row, col, size)` winner against the GPU's, at exactly the positions the
/// GPU enumerated, not at whatever positions the RMS-driven partition happened to visit.
/// A thin visibility wrapper only — [`search`]'s behaviour is untouched.
pub fn search_block(
    image: &Plane,
    contracted: &Contracted,
    row: u32,
    col: u32,
    size: u32,
    params: &EncodeParams,
) -> (Option<Candidate>, u64) {
    search(image, contracted, row, col, size, params)
}

/// Public constructor for [`Contracted`], needed by [`search_block`]'s callers to build
/// the 2:1 box-sum plane once per image rather than per block.
pub fn build_contracted(image: &Plane) -> Contracted {
    Contracted::build(image)
}

/// Exhaustively search every valid domain position and isometry for the range block at
/// `(row, col, size)`. Returns the winning candidate (`None` if no domain position is
/// legal at this size — never happens for the configs this project runs, but a block that
/// small on a tiny image is not ruled out by the format) and the number of evals spent.
fn search(
    image: &Plane,
    contracted: &Contracted,
    row: u32,
    col: u32,
    size: u32,
    params: &EncodeParams,
) -> (Option<Candidate>, u64) {
    search_with_shift(image, contracted, row, col, size, params, params.shift)
}

/// [`search`]'s actual body, taking the domain-search stride explicitly instead of always
/// reading `params.shift` -- Step 16's hook for content-adaptive domain-pool density.
/// [`search`] itself is `search_with_shift(.., params.shift)`, so every pre-Step-16 caller
/// is byte-for-byte unaffected; only [`walk_rd`]'s new `Ctx::adaptive_density` path calls
/// this directly with a per-block stride computed from [`block_rms`].
fn search_with_shift(
    image: &Plane,
    contracted: &Contracted,
    row: u32,
    col: u32,
    size: u32,
    params: &EncodeParams,
    shift: u32,
) -> (Option<Candidate>, u64) {
    let (width, height) = (image.width() as u32, image.height() as u32);
    let px = image.as_slice();
    let stride = image.width();
    let size_u = size as usize;

    // The range block's own moments (§8: t0, t2) — independent of the domain, computed
    // once. `range` is a size x size copy so isometry correlation below stays a simple
    // linear scan.
    let mut range = vec![0u8; size_u * size_u];
    let (mut t0, mut t2) = (0i64, 0i64);
    for i in 0..size_u {
        let src = (row as usize + i) * stride + col as usize;
        for j in 0..size_u {
            let r = px[src + j];
            range[i * size_u + j] = r;
            t0 += i64::from(r);
            t2 += i64::from(r) * i64::from(r);
        }
    }
    let s0 = i64::from(size) * i64::from(size);

    // Permuted once per isometry here, not once per `(domain position, isometry)` pair
    // inside the loop below — the NEON dot product `cross_term_permuted` calls only pays
    // off once this cost is amortised across every domain position, not repeated for
    // each one (docs/decisions.md's Step 11 entry has the measured before/after).
    let range_by_iso: Vec<Vec<u8>> = isometry::ALL
        .iter()
        .map(|&k| permute_range(&range, k, size_u))
        .collect();

    let Some(max_dom_row) = height.checked_sub(2 * size) else {
        return (None, 0);
    };
    let Some(max_dom_col) = width.checked_sub(2 * size) else {
        return (None, 0);
    };

    let mut best: Option<Candidate> = None;
    let mut evals = 0u64;
    let mut dom_row = 0u32;
    while dom_row <= max_dom_row {
        let mut dom_col = 0u32;
        while dom_col <= max_dom_col {
            let (dr_half, dc_half) = ((dom_row / 2) as usize, (dom_col / 2) as usize);
            let (s1_x4, s2_x16) = domain_sums(contracted, dr_half, dc_half, size_u);

            for &k in &isometry::ALL {
                let t1_x4 = cross_term_permuted(
                    contracted,
                    dr_half,
                    dc_half,
                    size_u,
                    &range_by_iso[k as usize],
                );
                let moments = RawMoments {
                    s0,
                    s1_x4,
                    s2_x16,
                    t0,
                    t1_x4,
                    t2,
                };
                let (qalfa, qbeta, rms) =
                    fit_f64(moments, params.max_alfa, params.bits_alfa, params.bits_beta);
                evals += 1;
                if best.is_none_or(|b| rms < b.rms) {
                    best = Some(Candidate {
                        dom_row,
                        dom_col,
                        isometry: k,
                        qalfa,
                        qbeta,
                        rms,
                        moments,
                    });
                }
            }
            dom_col += shift;
        }
        dom_row += shift;
    }
    (best, evals)
}

/// `(ΣD, ΣD²)` over one domain position's `size x size` samples — independent of isometry,
/// since summing is invariant under any permutation of the terms.
///
/// `pub` (Step 9): every `mars-search` candidate-restriction method needs exactly this
/// quantity for whatever domain positions its own indexing restricts the search to; it is
/// not specific to the exhaustive walk in this module.
/// Step 11: delegates to `mars_simd::moments::domain_sums`, exact-equal by construction
/// (that kernel's own differential test) to the strided scalar double-loop this used to
/// be — a NEON win with no risk to `search`'s output, since `Contracted::data`/`stride`
/// are exactly the `(plane, stride)` that kernel expects.
pub fn domain_sums(contracted: &Contracted, dr: usize, dc: usize, size: usize) -> (i64, i64) {
    mars_simd::moments::domain_sums(&contracted.data, contracted.stride, dr, dc, size)
}

/// `Σ r·D` for one domain position under isometry `k` — the one quantity that genuinely
/// depends on the isometry, since it pairs each domain sample with the range pixel it
/// would land on.
///
/// `pub` (Step 9): shared with `mars-search`, same reasoning as [`domain_sums`].
///
/// Step 11: permutes `range` into the domain's raster order once (`range_k`), then hands
/// both `range_k` and `Contracted`'s own strided plane to
/// `mars_simd::moments::dot_u8_i32_window`, so the accumulation itself is NEON rather than
/// a scalar loop indexing through `isometry::map` on every element — the permutation cost
/// is unchanged (the scalar version paid it too, just inline), only the dot product is new.
pub fn cross_term(
    contracted: &Contracted,
    dr: usize,
    dc: usize,
    size: usize,
    k: u8,
    range: &[u8],
) -> i64 {
    cross_term_permuted(contracted, dr, dc, size, &permute_range(range, k, size))
}

/// `range` permuted into `(u, v)` raster order under isometry `k` -- what [`cross_term`]
/// used to compute inline, element by element, on every call. Split out because
/// `search`'s hot loop calls all 8 isometries against the *same* domain position's
/// `range`: permuting once per isometry per range block, outside the domain-position
/// loop, instead of once per `(domain position, isometry)` pair, is what actually made
/// the NEON dot product in [`cross_term_permuted`] a net win rather than a regression --
/// see `docs/decisions.md`'s Step 11 entry for the measured before/after.
pub fn permute_range(range: &[u8], k: u8, size: usize) -> Vec<u8> {
    let mut range_k = vec![0u8; size * size];
    for u in 0..size {
        for v in 0..size {
            let (i, j) = isometry::map(k, u, v, size);
            range_k[u * size + v] = range[i * size + j];
        }
    }
    range_k
}

/// [`cross_term`] with the permutation already done -- the actual hot-path entry point
/// `search` uses, since it can amortise `range_k` across every domain position for a
/// fixed isometry (see [`permute_range`]'s doc). `pub` (Gate C, D35): `mars-search`'s
/// `search_block` amortises the same way, across a range block's candidate list instead
/// of `search`'s domain-position loop.
pub fn cross_term_permuted(
    contracted: &Contracted,
    dr: usize,
    dc: usize,
    size: usize,
    range_k: &[u8],
) -> i64 {
    mars_simd::moments::dot_u8_i32_window(
        range_k,
        &contracted.data,
        contracted.stride,
        dr,
        dc,
        size,
    )
}

/// Recompute one leaf's raw moments from its stored domain reference. Used only by
/// [`f32_f64_divergence`], which needs the moments a search assigns but a leaf itself does
/// not carry — meaningless for a DC-only leaf (`qalfa == 0`), whose domain fields are not
/// the one the search actually found (§8.1 zeroes them out).
fn moments_for(contracted: &Contracted, image: &Plane, leaf: &Leaf) -> RawMoments {
    let px = image.as_slice();
    let stride = image.width();
    let size_u = leaf.size as usize;
    let mut range = vec![0u8; size_u * size_u];
    let (mut t0, mut t2) = (0i64, 0i64);
    for i in 0..size_u {
        let src = (leaf.row as usize + i) * stride + leaf.col as usize;
        for j in 0..size_u {
            let r = px[src + j];
            range[i * size_u + j] = r;
            t0 += i64::from(r);
            t2 += i64::from(r) * i64::from(r);
        }
    }
    let s0 = i64::from(leaf.size) * i64::from(leaf.size);
    let (dr_half, dc_half) = ((leaf.dom_row / 2) as usize, (leaf.dom_col / 2) as usize);
    let (s1_x4, s2_x16) = domain_sums(contracted, dr_half, dc_half, size_u);
    let t1_x4 = cross_term(contracted, dr_half, dc_half, size_u, leaf.isometry, &range);
    RawMoments {
        s0,
        s1_x4,
        s2_x16,
        t0,
        t1_x4,
        t2,
    }
}

/// `just gate-6`'s f32-vs-f64 divergence measurement: over every leaf that references a
/// domain (`qalfa != 0` — a DC-only leaf's fit does not depend on precision), recompute
/// its fit at both precisions and count how often they disagree on `qalfa` or `qbeta`.
/// Returns `(compared, differed)`.
pub fn f32_f64_divergence(
    image: &Plane,
    leaves: &[Leaf],
    max_alfa: f64,
    bits_alfa: u32,
    bits_beta: u32,
) -> (usize, usize) {
    let contracted = Contracted::build(image);
    let mut compared = 0usize;
    let mut differed = 0usize;
    for leaf in leaves.iter().filter(|l| l.qalfa != 0) {
        let m = moments_for(&contracted, image, leaf);
        let (qa64, qb64, _) = fit_f64(m, max_alfa, bits_alfa, bits_beta);
        let (qa32, qb32, _) = fit_f32(m, max_alfa as f32, bits_alfa, bits_beta);
        compared += 1;
        if qa64 != qa32 || qb64 != qb32 {
            differed += 1;
        }
    }
    (compared, differed)
}

/// Encode one image exhaustively, per §5's partition and §8's fit. Returns the header, the
/// leaves in `parse`'s order, and the total `evals` spent (§M5).
///
/// Step 14: dispatches on `params.lambda`. `None` runs the legacy top-down, `t_rms`
/// threshold-driven `walk` unchanged (bit-identical to every pre-Step-14 caller and
/// test). `Some(lambda)` runs the bottom-up `J = D + λR` walk instead -- see
/// [`walk_rd`]/`crate::rate`'s module doc for the rate-estimation approximation this
/// requires.
pub fn encode_image(image: &Plane, params: &EncodeParams) -> (Header, Vec<Leaf>, u64) {
    let (hdr, leaves, evals, _stats) = encode_image_rd(image, params);
    (hdr, leaves, evals)
}

/// [`encode_image`]'s superset: also returns Step 15's [`ModeStats`] mode-usage histogram
/// (all zero on the legacy `lambda: None` path, which never runs [`walk_rd`] and so never
/// makes a mode decision to count). Additive-only -- [`encode_image`] itself is unchanged
/// and every pre-existing caller keeps compiling against its original three-tuple return.
pub fn encode_image_rd(
    image: &Plane,
    params: &EncodeParams,
) -> (Header, Vec<Leaf>, u64, ModeStats) {
    encode_image_rd_with_modes(image, params, [true; 4])
}

/// [`encode_image_rd`] with an explicit mode mask -- Step 15's own comparison tool.
/// `[true, false, true, false]` restricts `walk_rd`'s leaf decision to modes 0 (flat) and
/// 2 (fractal) only, exactly reproducing Step 14's decision rule (flat-refit-or-fractal,
/// whichever has the smaller `J`) on the *current* codebase, so `mars-bench`'s Step-15-vs-
/// Step-14 BD-rate comparison is a true "modes added, nothing else changed" A/B rather
/// than a diff against a separate git revision (see [`Ctx::allowed_modes`]'s doc for why
/// that is the fairer comparison). Every ordinary caller goes through [`encode_image_rd`]
/// / [`encode_image`], which always pass `[true; 4]`.
pub fn encode_image_rd_with_modes(
    image: &Plane,
    params: &EncodeParams,
    allowed_modes: [bool; 4],
) -> (Header, Vec<Leaf>, u64, ModeStats) {
    encode_image_rd_with_modes_and_density(image, params, allowed_modes, false)
}

/// [`encode_image_rd_with_modes`]'s superset: also takes Step 16's `adaptive_density` flag
/// -- Step 16's own comparison tool, mirroring [`encode_image_rd_with_modes`]'s own
/// same-codebase A/B pattern exactly (`docs/decisions.md`'s D40 has the reasoning for why
/// this is the fairer comparison than a separate git revision). `false` reproduces every
/// prior step's behaviour exactly, byte-for-byte -- [`encode_image_rd_with_modes`] always
/// passes `false`, so this function is purely additive. `true` makes [`walk_rd`] compute a
/// per-block domain-search stride from [`block_rms`]/[`adaptive_shift`] instead of the
/// fixed `params.shift`, on the RD (`lambda: Some`) path only -- meaningless on the legacy
/// `lambda: None` path, which never consults `Ctx::adaptive_density` at all.
pub fn encode_image_rd_with_modes_and_density(
    image: &Plane,
    params: &EncodeParams,
    allowed_modes: [bool; 4],
    adaptive_density: bool,
) -> (Header, Vec<Leaf>, u64, ModeStats) {
    let hdr = Header {
        bits_alfa: params.bits_alfa,
        bits_beta: params.bits_beta,
        min_size: params.min_size,
        max_size: params.max_size,
        shift: params.shift,
        width: image.width() as u32,
        height: image.height() as u32,
        int_max_alfa: quantise_f64(params.max_alfa / 8.0 * 256.0, 255),
    };
    let contracted = Contracted::build(image);

    if let Some(lambda) = params.lambda {
        let rate = build_rate_snapshot(image, &hdr, &contracted, params);
        let ctx = Ctx {
            image,
            contracted: &contracted,
            hdr: &hdr,
            params,
            rate: Some(&rate),
            allowed_modes,
            adaptive_density,
        };
        let result = walk_rd(&ctx, 0, 0, hdr.virtual_size(), lambda);
        return (hdr, result.leaves, result.evals, result.stats);
    }

    let ctx = Ctx {
        image,
        contracted: &contracted,
        hdr: &hdr,
        params,
        rate: None,
        allowed_modes,
        adaptive_density: false,
    };
    let (leaves, evals) = walk(&ctx, 0, 0, hdr.virtual_size());
    (hdr, leaves, evals, ModeStats::default())
}

/// Step 14's fixed warm-up threshold, deliberately independent of the run's own `lambda`
/// (`crate::rate`'s module doc explains why a mutable, decision-order-updated model was
/// rejected in favour of a frozen snapshot). Kept at the 1998 default so the snapshot's
/// leaf-size mix is a plausible, moderate partition regardless of which `lambda` is being
/// evaluated -- not this `lambda`'s own eventual partition, which is the documented
/// fidelity gap (`docs/decisions.md`): no fixed-point iteration is attempted this step.
const RD_WARMUP_T_RMS: f64 = 8.0;

/// Run the legacy top-down encoder once, at [`RD_WARMUP_T_RMS`], and replay its leaves'
/// real event stream through [`mars_entropy::build_models`] to obtain the frozen,
/// read-only per-context snapshot [`walk_rd`]'s rate estimates are priced against.
fn build_rate_snapshot(
    image: &Plane,
    hdr: &Header,
    contracted: &Contracted,
    params: &EncodeParams,
) -> RateModels {
    let warmup_params = EncodeParams {
        t_rms: RD_WARMUP_T_RMS,
        lambda: None,
        ..*params
    };
    let warmup_ctx = Ctx {
        image,
        contracted,
        hdr,
        params: &warmup_params,
        rate: None,
        allowed_modes: [true; 4],
        adaptive_density: false,
    };
    let (leaves, _evals) = walk(&warmup_ctx, 0, 0, hdr.virtual_size());
    RateModels::from_leaves(hdr, &leaves)
        .expect("a warm-up partition from `walk` always writes as a valid `.mars` tree")
}

/// The read-only context one `walk` recursion shares — bundled so the recursive calls
/// stay readable instead of threading four parameters through every one. Every field is
/// a shared reference to plain data (no interior mutability), so `Ctx` is `Sync` and safe
/// to share across the `rayon::join` calls below.
///
/// `rate` is `Some` only on the Step 14 RD path ([`walk_rd`]/[`split_rd`]) — a frozen
/// snapshot, never mutated after construction, so sharing it across parallel subtrees
/// introduces no thread-count dependence (see `crate::rate`'s module doc).
struct Ctx<'a> {
    image: &'a Plane,
    contracted: &'a Contracted,
    hdr: &'a Header,
    params: &'a EncodeParams,
    rate: Option<&'a RateModels>,
    /// Step 15: which of modes 0/1/2/3 [`best_mode_leaf`] is allowed to consider, indexed
    /// by mode number. Always `[true; 4]` on every ordinary encode path; the only other
    /// caller is `mars-bench`'s Step-15-vs-Step-14 comparison (`encode_image_rd_with_modes`
    /// below), which sets it to `[true, false, true, false]` to reproduce Step 14's
    /// flat-or-fractal-only decision using the exact same rate-estimation and search
    /// machinery, differing *only* in which modes may compete -- a fairer, more honest A/B
    /// than diffing against a separate git revision would be (no risk of an incidental,
    /// unrelated code difference between commits leaking into the comparison).
    allowed_modes: [bool; 4],
    /// Step 16: when `true`, [`walk_rd`] computes a per-block domain-search stride from
    /// [`block_rms`]/[`adaptive_shift`] instead of always using `params.shift`. `false`
    /// everywhere except the new [`encode_image_rd_with_modes_and_density`] entry point's
    /// own `true` arm, so every pre-Step-16 caller (including [`walk`], which never reads
    /// this field at all) is byte-for-byte unaffected.
    adaptive_density: bool,
}

/// Step 12: below this block size, a `rayon::join`'s task-spawn/steal overhead costs more
/// than the search it would parallelise, so the four quadrants recurse on the calling
/// thread instead. Chosen well under every project config's `min_size` (>= 4) so the
/// parallel/sequential boundary never falls inside the region a config actually searches
/// at — see the Step 12 prediction (P12.2) for the reasoning and the measured scaling
/// curve this cutoff produces.
const PARALLEL_SIZE_CUTOFF: u32 = 8;

/// Search and partition are still one RMS-driven recursion (§5.2's split rule genuinely
/// depends on this block's own search result), but emission is decoupled from it: each
/// call returns its own `(leaves, evals)` instead of pushing into a shared `Vec`, so
/// independent subtrees can be searched in parallel and merged afterwards in the same
/// TL/BL/TR/BR order the sequential walk always used — the merge is pure concatenation,
/// not a sort, so it reproduces the sequential leaf order (and hence bitstream) exactly
/// regardless of how many threads did the searching.
fn walk(ctx: &Ctx, row: u32, col: u32, size: u32) -> (Vec<Leaf>, u64) {
    let hdr = ctx.hdr;
    if row >= hdr.height || col >= hdr.width {
        return (Vec::new(), 0); // §5.1
    }
    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    if forced {
        let half = size / 2;
        return split(ctx, row, col, half);
    }

    // §5.3 / coding_func.c `tip == 0`: a size-1 block is a raw, truncated pixel — no
    // search, no evals, and it can never exceed `min_size` (a power of two >= 2) so it is
    // always a leaf.
    if size == 1 {
        let pixel =
            u32::from(ctx.image.as_slice()[row as usize * ctx.image.width() + col as usize]);
        let leaf = Leaf {
            row,
            col,
            size,
            mode: 0,
            qalfa: 0,
            qbeta: pixel & ((1 << hdr.bits_beta) - 1),
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
            qgx: 0,
            qgy: 0,
            residual: Vec::new(),
        };
        return (vec![leaf], 0);
    }

    let (candidate, block_evals) = search(ctx.image, ctx.contracted, row, col, size, ctx.params);
    let best_rms = candidate.map_or(f64::INFINITY, |c| c.rms);

    if best_rms > ctx.params.t_rms && size > hdr.min_size {
        let half = size / 2;
        let (leaves, sub_evals) = split(ctx, row, col, half);
        return (leaves, sub_evals + block_evals);
    }

    let mut leaf = candidate.map_or(
        Leaf {
            row,
            col,
            size,
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
        |c| Leaf {
            row,
            col,
            size,
            mode: 2,
            qalfa: c.qalfa,
            qbeta: c.qbeta,
            isometry: c.isometry,
            dom_row: c.dom_row,
            dom_col: c.dom_col,
            qgx: 0,
            qgy: 0,
            residual: Vec::new(),
        },
    );
    // §8.1: discard the searched qbeta and refit DC-only whenever qalfa lands within
    // `zero_threshold` of zeroalfa (0) — true whenever `qalfa == 0` at the 1998 default.
    if leaf.qalfa.abs_diff(0) <= ctx.params.zero_threshold {
        let px = ctx.image.as_slice();
        let stride = ctx.image.width();
        let size_u = size as usize;
        let mut range_sum = 0i64;
        for i in 0..size_u {
            let src = (row as usize + i) * stride + col as usize;
            for j in 0..size_u {
                range_sum += i64::from(px[src + j]);
            }
        }
        leaf.qbeta = best_beta(range_sum, i64::from(size) * i64::from(size), hdr.bits_beta);
        leaf.qalfa = 0;
        leaf.mode = 0;
        leaf.isometry = 0;
        leaf.dom_row = 0;
        leaf.dom_col = 0;
    }
    (vec![leaf], block_evals)
}

/// Recurse into the four quadrants of a `2*half x 2*half` region at `(row, col)`, in
/// canonical TL/BL/TR/BR order (§5's tree order, matched by `mars_format`'s writer). Above
/// [`PARALLEL_SIZE_CUTOFF`], the two pairs run via `rayon::join`; below it, sequentially on
/// the calling thread. Either way the four results are concatenated in the same fixed
/// order, so the merge itself introduces no thread-count dependence.
fn split(ctx: &Ctx, row: u32, col: u32, half: u32) -> (Vec<Leaf>, u64) {
    let quadrants = if half >= PARALLEL_SIZE_CUTOFF {
        let ((tl, tl_e), (bl, bl_e)) = rayon::join(
            || walk(ctx, row, col, half),
            || walk(ctx, row + half, col, half),
        );
        let ((tr, tr_e), (br, br_e)) = rayon::join(
            || walk(ctx, row, col + half, half),
            || walk(ctx, row + half, col + half, half),
        );
        [(tl, tl_e), (bl, bl_e), (tr, tr_e), (br, br_e)]
    } else {
        [
            walk(ctx, row, col, half),
            walk(ctx, row + half, col, half),
            walk(ctx, row, col + half, half),
            walk(ctx, row + half, col + half, half),
        ]
    };
    let mut leaves = Vec::new();
    let mut evals = 0u64;
    for (mut q_leaves, q_evals) in quadrants {
        leaves.append(&mut q_leaves);
        evals += q_evals;
    }
    (leaves, evals)
}

/// One subtree's bottom-up RD result: its leaves, the search effort spent producing them,
/// and its total distortion/rate -- `d` is the summed sum-of-squared-error (SSE, matching
/// `fit_f64`'s own `sum` before the final `/s0`.sqrt()), `r` the summed estimated bits
/// (`crate::rate::RateModels::bits_for`), **including** this subtree's own split-flag bit
/// where one was actually emitted (i.e. everywhere `size > min_size` and not `forced`).
/// Distortion is SSE rather than RMS specifically so two branches covering the same pixel
/// area can be compared by plain addition (`docs/decisions.md`'s Step 14 entry has the
/// reasoning): `J = D + λR` with `D` in SSE units is the same convention H.26x-style RDO
/// uses (`SSD + λ·bits`).
struct RdResult {
    leaves: Vec<Leaf>,
    evals: u64,
    d: f64,
    r: f64,
    stats: ModeStats,
}

/// Step 15's mode-usage histogram (the brief's own header finding, more scientifically
/// interesting than the BD-rate number): how often each leaf mode wins the `J`
/// competition, plus how often "subdivide" (mode 4) wins over coding this block as any
/// single leaf at all. `leaf_modes[m]` counts leaves whose final `mode == m`;
/// `split_decisions`/`leaf_decisions` count every point in the walk where a leaf-vs-split
/// choice was actually made (i.e. every non-forced, `size > min_size` node) -- their sum
/// is the total number of such decision points, and `split_decisions as f64 / total` is
/// the mode-4 share.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModeStats {
    pub leaf_modes: [u64; 4],
    pub split_decisions: u64,
    pub leaf_decisions: u64,
}

impl ModeStats {
    fn merge(&mut self, other: ModeStats) {
        for i in 0..4 {
            self.leaf_modes[i] += other.leaf_modes[i];
        }
        self.split_decisions += other.split_decisions;
        self.leaf_decisions += other.leaf_decisions;
    }
}

/// Sum of [`RateModels::bits_for`] over every event a candidate leaf would emit.
fn event_bits(rate: &RateModels, events: &[mars_entropy::Event]) -> f64 {
    events
        .iter()
        .map(|e| rate.bits_for(e.ctx, e.alphabet, e.symbol))
        .sum()
}

/// Step 15's five-way mode competition, per-leaf half (`docs/predictions.md`'s Step 15
/// prediction; R&D plan §4). Evaluates modes 0 (flat), 1 (affine), 2 (fractal), and 3
/// (fractal + residual) for this block, prices each under `rate` via the exact same
/// [`crate::mars_format::leaf_events`] event stream the real entropy coder would emit, and
/// returns whichever has the smallest `J = D + lambda*R`. The remaining mode, 4
/// (subdivide), is decided by [`walk_rd`]'s existing leaf-vs-split comparison one level up
/// -- this function only ever returns a leaf, never a split decision.
///
/// Replaces the legacy `walk`/[`walk`]'s §8.1 zero-alfa override on the RD path only. The
/// legacy top-down path keeps that override unchanged (`lambda: None` never reaches this
/// function), since modes 1/3 have no representation in the raw `.ifs` bitstream that path
/// still targets (`crate::ifs::Leaf`'s doc).
#[allow(clippy::too_many_arguments)]
fn best_mode_leaf(
    ctx: &Ctx,
    row: u32,
    col: u32,
    size: u32,
    candidate: Option<Candidate>,
    rate: &RateModels,
    lambda: f64,
    size_class: u32,
) -> (Leaf, f64, f64) {
    let hdr = ctx.hdr;
    let px = ctx.image.as_slice();
    let stride = ctx.image.width();
    let size_u = size as usize;

    let mut block = vec![0.0f64; size_u * size_u];
    let mut range_sum = 0i64;
    for i in 0..size_u {
        let src = (row as usize + i) * stride + col as usize;
        for j in 0..size_u {
            let v = px[src + j];
            block[i * size_u + j] = f64::from(v);
            range_sum += i64::from(v);
        }
    }
    let s0 = i64::from(size) * i64::from(size);

    let mut candidates: Vec<(Leaf, f64, f64)> = Vec::with_capacity(4);
    let allowed = ctx.allowed_modes;

    // Mode 0 -- flat: exactly §8.1's `best_beta` refit, priced as its own competing mode
    // rather than an override forced onto mode 2's result.
    if allowed[0] {
        let qbeta = best_beta(range_sum, s0, hdr.bits_beta);
        let leaf = Leaf {
            row,
            col,
            size,
            mode: 0,
            qalfa: 0,
            qbeta,
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
            qgx: 0,
            qgy: 0,
            residual: Vec::new(),
        };
        let max_qbeta = (1u32 << hdr.bits_beta) - 1;
        let beta2 = f64::from(qbeta) / f64::from(max_qbeta) * 255.0;
        let sse: f64 = block.iter().map(|&p| (p - beta2).powi(2)).sum();
        let r = event_bits(
            rate,
            &crate::mars_format::leaf_events(hdr, &leaf, size_class),
        );
        candidates.push((leaf, sse, r));
    }

    // Mode 1 -- affine: a spatial-gradient plane fit, no domain search at all.
    if allowed[1] {
        let (qbeta, qgx, qgy, sse) = affine_fit(&block, size_u, hdr.bits_beta);
        let leaf = Leaf {
            row,
            col,
            size,
            mode: 1,
            qalfa: 0,
            qbeta,
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
            qgx,
            qgy,
            residual: Vec::new(),
        };
        let r = event_bits(
            rate,
            &crate::mars_format::leaf_events(hdr, &leaf, size_class),
        );
        candidates.push((leaf, sse, r));
    }

    // Modes 2/3 -- fractal, and fractal + residual -- only when the search actually found
    // a legal domain candidate whose fit did not collapse to alfa == 0 (a genuinely
    // domain-independent block, which mode 0/1 already cover with no domain fields to
    // code at all).
    if let Some(c) = candidate {
        if c.qalfa >= 1 && allowed[2] {
            let leaf2 = Leaf {
                row,
                col,
                size,
                mode: 2,
                qalfa: c.qalfa,
                qbeta: c.qbeta,
                isometry: c.isometry,
                dom_row: c.dom_row,
                dom_col: c.dom_col,
                qgx: 0,
                qgy: 0,
                residual: Vec::new(),
            };
            let sse2 = c.rms * c.rms * f64::from(size) * f64::from(size);
            let r2 = event_bits(
                rate,
                &crate::mars_format::leaf_events(hdr, &leaf2, size_class),
            );
            candidates.push((leaf2, sse2, r2));
        }
        if c.qalfa >= 1 && allowed[3] {
            let (levels, sse3) =
                residual_for_candidate(&block, px, stride, size_u, &c, ctx.params, hdr);
            let leaf3 = Leaf {
                row,
                col,
                size,
                mode: 3,
                qalfa: c.qalfa,
                qbeta: c.qbeta,
                isometry: c.isometry,
                dom_row: c.dom_row,
                dom_col: c.dom_col,
                qgx: 0,
                qgy: 0,
                residual: levels,
            };
            let r3 = event_bits(
                rate,
                &crate::mars_format::leaf_events(hdr, &leaf3, size_class),
            );
            candidates.push((leaf3, sse3, r3));
        }
    }

    if candidates.is_empty() {
        // Defensive fallback only -- unreachable with every mask this project actually
        // uses (mode 0 is always allowed), but cheaper to guarantee here than to let a
        // pathological all-`false` mask panic deep in `min_by`.
        let qbeta = best_beta(range_sum, s0, hdr.bits_beta);
        let leaf = Leaf {
            row,
            col,
            size,
            mode: 0,
            qalfa: 0,
            qbeta,
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
            qgx: 0,
            qgy: 0,
            residual: Vec::new(),
        };
        let max_qbeta = (1u32 << hdr.bits_beta) - 1;
        let beta2 = f64::from(qbeta) / f64::from(max_qbeta) * 255.0;
        let sse: f64 = block.iter().map(|&p| (p - beta2).powi(2)).sum();
        let r = event_bits(
            rate,
            &crate::mars_format::leaf_events(hdr, &leaf, size_class),
        );
        candidates.push((leaf, sse, r));
    }

    let (best_leaf, best_d, best_r) = candidates
        .into_iter()
        .min_by(|a, b| {
            let ja = a.1 + lambda * a.2;
            let jb = b.1 + lambda * b.2;
            ja.partial_cmp(&jb).expect("D and R are always finite")
        })
        .expect("candidates is never empty: either a mode was allowed, or the fallback ran");
    (best_leaf, best_d, best_r)
}

/// Mode 1's least-squares plane fit `b0 + gx*u + gy*v` over a `size x size` block, solved
/// in closed form: the design matrix depends only on `size` (the sample positions are a
/// fixed regular grid), so its normal-equation coefficients are plain sums over `0..size`,
/// independent of the pixel data, and only the right-hand side (`Σp`, `Σp·u`, `Σp·v`)
/// varies per block. Returns `(qbeta, qgx, qgy, sse)`, `sse` computed against the
/// *quantised* plane (matching this module's convention elsewhere of pricing distortion
/// against what the decoder will actually reconstruct's continuous value, before the
/// final per-pixel round/clamp -- see [`fit_f64`]'s own `sum` formula for the precedent).
fn affine_fit(block: &[f64], size: usize, bits_beta: u32) -> (u32, i32, i32, f64) {
    let n = size as f64 * size as f64;
    let (mut t0, mut tu, mut tv) = (0.0, 0.0, 0.0);
    for u in 0..size {
        for v in 0..size {
            let p = block[u * size + v];
            t0 += p;
            tu += p * u as f64;
            tv += p * v as f64;
        }
    }
    let su: f64 = (0..size).map(|u| u as f64).sum();
    let suu: f64 = (0..size).map(|u| (u as f64).powi(2)).sum();
    let s0 = n;
    let s_u = size as f64 * su;
    let s_v = s_u;
    let s_uu = size as f64 * suu;
    let s_vv = s_uu;
    let s_uv = su * su;

    fn det3(m: [[f64; 3]; 3]) -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
    let a = [[s0, s_u, s_v], [s_u, s_uu, s_uv], [s_v, s_uv, s_vv]];
    let det_a = det3(a);
    let (b0, gx, gy) = if det_a.abs() < 1e-9 {
        (t0 / s0, 0.0, 0.0)
    } else {
        let a_b0 = [[t0, s_u, s_v], [tu, s_uu, s_uv], [tv, s_uv, s_vv]];
        let a_gx = [[s0, t0, s_v], [s_u, tu, s_uv], [s_v, tv, s_vv]];
        let a_gy = [[s0, s_u, t0], [s_u, s_uu, tu], [s_v, s_uv, tv]];
        (det3(a_b0) / det_a, det3(a_gx) / det_a, det3(a_gy) / det_a)
    };

    let max_qbeta = (1u32 << bits_beta) - 1;
    let qbeta = quantise_f64((b0 / 255.0 * f64::from(max_qbeta)).max(0.0), max_qbeta);
    let qgx = (gx * AFFINE_GRAD_SCALE)
        .round()
        .clamp(-f64::from(AFFINE_GRAD_CLAMP), f64::from(AFFINE_GRAD_CLAMP)) as i32;
    let qgy = (gy * AFFINE_GRAD_SCALE)
        .round()
        .clamp(-f64::from(AFFINE_GRAD_CLAMP), f64::from(AFFINE_GRAD_CLAMP)) as i32;

    let b0q = f64::from(qbeta) / f64::from(max_qbeta) * 255.0;
    let gxq = f64::from(qgx) / AFFINE_GRAD_SCALE;
    let gyq = f64::from(qgy) / AFFINE_GRAD_SCALE;
    let mut sse = 0.0;
    for u in 0..size {
        for v in 0..size {
            let pred = b0q + gxq * u as f64 + gyq * v as f64;
            let diff = block[u * size + v] - pred;
            sse += diff * diff;
        }
    }
    (qbeta, qgx, qgy, sse)
}

/// Mode 3's residual: the mode-2 fractal prediction's continuous reconstruction error,
/// forward-DCT'd, dead-zone quantised (`crate::quant`, at the fixed
/// [`RESIDUAL_QSTEP_DEFAULT`]/[`RESIDUAL_DEAD_ZONE`]), and inverse-transformed back so the
/// exact SSE the decoder will actually see can be priced -- the same "price the quantised
/// reconstruction, not the ideal one" discipline [`affine_fit`] and [`fit_f64`] both
/// follow. Returns `(levels, sse)`; `levels` are the clamped integer coefficients
/// [`crate::residual::encode_events`] will code.
#[allow(clippy::too_many_arguments)]
fn residual_for_candidate(
    block: &[f64],
    px: &[u8],
    stride: usize,
    size: usize,
    c: &Candidate,
    params: &EncodeParams,
    hdr: &Header,
) -> (Vec<i32>, f64) {
    let alfa = f64::from(c.qalfa) / f64::from(1u32 << params.bits_alfa) * params.max_alfa;
    let mut beta =
        f64::from(c.qbeta) / f64::from((1u32 << hdr.bits_beta) - 1) * ((1.0 + alfa.abs()) * 255.0);
    if alfa > 0.0 {
        beta -= alfa * 255.0;
    }

    let mut resid = vec![0.0f64; size * size];
    for u in 0..size {
        for v in 0..size {
            let dr = c.dom_row as usize + 2 * u;
            let dc = c.dom_col as usize + 2 * v;
            let d = (f64::from(px[dr * stride + dc])
                + f64::from(px[(dr + 1) * stride + dc])
                + f64::from(px[dr * stride + dc + 1])
                + f64::from(px[(dr + 1) * stride + dc + 1]))
                / 4.0;
            let (i, j) = isometry::map(c.isometry, u, v, size);
            let pred = 0.5 + d * alfa + beta;
            resid[i * size + j] = block[i * size + j] - pred;
        }
    }

    let coeffs = crate::dct::forward_dct2d(&resid, size);
    let levels: Vec<i32> = coeffs
        .iter()
        .map(|&x| {
            crate::quant::dead_zone_quantize(x, RESIDUAL_QSTEP_DEFAULT, RESIDUAL_DEAD_ZONE)
                .clamp(-crate::residual::LEVEL_CLAMP, crate::residual::LEVEL_CLAMP)
        })
        .collect();
    let deq: Vec<f64> = levels
        .iter()
        .map(|&l| crate::quant::dead_zone_dequantize(l, RESIDUAL_QSTEP_DEFAULT, RESIDUAL_DEAD_ZONE))
        .collect();
    let recon_resid = crate::dct::inverse_dct2d(&deq, size);
    let sse: f64 = (0..size * size)
        .map(|k| (resid[k] - recon_resid[k]).powi(2))
        .sum();
    (levels, sse)
}

/// Step 14's bottom-up `J = D + λR` walk: search this block, then recursively resolve its
/// four children (each already deciding leaf-vs-further-split), then keep whichever of
/// "this block as one leaf" or "the four children as they stand" has the smaller
/// `D + λR`. Strictly more informed than [`walk`]'s top-down RMS threshold, which decides
/// before any child has been searched (`implementation-plan.md` Step 14's own framing) --
/// at the cost of visiting every node of the full quadtree down to `min_size` regardless
/// of outcome, not just the nodes a threshold would have stopped early at (`evals` and
/// wall time both reflect this; see `docs/predictions.md`'s Step 14 entry).
fn walk_rd(ctx: &Ctx, row: u32, col: u32, size: u32, lambda: f64) -> RdResult {
    let hdr = ctx.hdr;
    if row >= hdr.height || col >= hdr.width {
        return RdResult {
            leaves: Vec::new(),
            evals: 0,
            d: 0.0,
            r: 0.0,
            stats: ModeStats::default(),
        };
    }
    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    if forced {
        let half = size / 2;
        return split_rd(ctx, row, col, half, lambda);
    }

    let rate = ctx
        .rate
        .expect("walk_rd always runs with a rate snapshot (params.lambda is Some)");
    let size_class = size.trailing_zeros();

    // §5.3: a size-1 block is a raw, truncated pixel -- no search, no evals, always a
    // leaf. Its truncation error is technically nonzero whenever `bits_beta < 8`, but
    // this project's configs all keep `min_size >= 2` (docs/decisions.md), so size == 1
    // is never actually reached; D is approximated as 0 here rather than computed exactly
    // for a path that never executes.
    if size == 1 {
        let pixel =
            u32::from(ctx.image.as_slice()[row as usize * ctx.image.width() + col as usize]);
        let leaf = Leaf {
            row,
            col,
            size,
            mode: 0,
            qalfa: 0,
            qbeta: pixel & ((1 << hdr.bits_beta) - 1),
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
            qgx: 0,
            qgy: 0,
            residual: Vec::new(),
        };
        let r = event_bits(rate, &leaf_events(hdr, &leaf, size_class));
        let mut stats = ModeStats::default();
        stats.leaf_modes[0] += 1;
        return RdResult {
            leaves: vec![leaf],
            evals: 0,
            d: 0.0,
            r,
            stats,
        };
    }

    // Step 16: content-adaptive domain-pool density. Computed before the search itself so
    // the density decision never depends on the search's own result (which would make the
    // "which stride did we search at" question circular). `false` (every pre-Step-16
    // caller) preserves `search(..)`'s original fixed-`params.shift` behaviour exactly.
    let (candidate, block_evals) = if ctx.adaptive_density {
        let rms = block_rms(ctx.image, row, col, size);
        let shift = adaptive_shift(ctx.params.shift, rms);
        search_with_shift(ctx.image, ctx.contracted, row, col, size, ctx.params, shift)
    } else {
        search(ctx.image, ctx.contracted, row, col, size, ctx.params)
    };
    let (leaf, leaf_d, leaf_r) =
        best_mode_leaf(ctx, row, col, size, candidate, rate, lambda, size_class);

    if size <= hdr.min_size {
        // Never split below min_size -- mars_format never emits a split flag here either.
        let mut stats = ModeStats::default();
        stats.leaf_modes[leaf.mode as usize] += 1;
        return RdResult {
            leaves: vec![leaf],
            evals: block_evals,
            d: leaf_d,
            r: leaf_r,
            stats,
        };
    }

    let half = size / 2;
    let children = split_rd(ctx, row, col, half, lambda);
    let total_evals = block_evals + children.evals;

    let split_flag_bits = |symbol: u32| rate.bits_for((FIELD_SPLIT, size_class), 2, symbol);
    let leaf_j = leaf_d + lambda * (leaf_r + split_flag_bits(0));
    let split_j = children.d + lambda * (children.r + split_flag_bits(1));

    if leaf_j <= split_j {
        let mut stats = ModeStats::default();
        stats.leaf_modes[leaf.mode as usize] += 1;
        stats.leaf_decisions += 1;
        RdResult {
            leaves: vec![leaf],
            evals: total_evals,
            d: leaf_d,
            r: leaf_r + split_flag_bits(0),
            stats,
        }
    } else {
        let mut stats = children.stats;
        stats.split_decisions += 1;
        RdResult {
            leaves: children.leaves,
            evals: total_evals,
            d: children.d,
            r: children.r + split_flag_bits(1),
            stats,
        }
    }
}

/// [`split`]'s Step 14 counterpart: recurse into the four quadrants via [`walk_rd`],
/// preserving the exact same `rayon::join` structure and fixed TL/BL/TR/BR merge order
/// (so RD-driven output is bit-identical across thread counts too), summing each child's
/// `(leaves, evals, d, r)` rather than just `(leaves, evals)`.
fn split_rd(ctx: &Ctx, row: u32, col: u32, half: u32, lambda: f64) -> RdResult {
    let quadrants = if half >= PARALLEL_SIZE_CUTOFF {
        let (tl, bl) = rayon::join(
            || walk_rd(ctx, row, col, half, lambda),
            || walk_rd(ctx, row + half, col, half, lambda),
        );
        let (tr, br) = rayon::join(
            || walk_rd(ctx, row, col + half, half, lambda),
            || walk_rd(ctx, row + half, col + half, half, lambda),
        );
        [tl, bl, tr, br]
    } else {
        [
            walk_rd(ctx, row, col, half, lambda),
            walk_rd(ctx, row + half, col, half, lambda),
            walk_rd(ctx, row, col + half, half, lambda),
            walk_rd(ctx, row + half, col + half, half, lambda),
        ]
    };
    let mut leaves = Vec::new();
    let mut evals = 0u64;
    let mut d = 0.0;
    let mut r = 0.0;
    let mut stats = ModeStats::default();
    for q in quadrants {
        leaves.extend(q.leaves);
        evals += q.evals;
        d += q.d;
        r += q.r;
        stats.merge(q.stats);
    }
    RdResult {
        leaves,
        evals,
        d,
        r,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny xorshift PRNG — deterministic and dependency-free, which is all a
    /// reproducible differential test needs.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn byte(&mut self) -> u8 {
            (self.next_u64() & 0xff) as u8
        }
        fn size(&mut self) -> usize {
            [2usize, 4, 8, 16][(self.next_u64() % 4) as usize]
        }
        fn isometry(&mut self) -> u8 {
            (self.next_u64() % 8) as u8
        }
    }

    /// The integer-exact moments of one synthetic `(range, domain)` pair, computed the
    /// same way `search` computes them, but standalone so the differential test below
    /// does not need a full `Plane`/`Contracted` image.
    fn integer_moments(range: &[u8], domain: &[u8], size: usize, isom: u8) -> RawMoments {
        let dstride = 2 * size;
        let mut d = vec![0i64; size * size];
        let (mut s1, mut s2) = (0i64, 0i64);
        for u in 0..size {
            for v in 0..size {
                let (r0, c0) = (2 * u, 2 * v);
                let sum = i64::from(domain[r0 * dstride + c0])
                    + i64::from(domain[r0 * dstride + c0 + 1])
                    + i64::from(domain[(r0 + 1) * dstride + c0])
                    + i64::from(domain[(r0 + 1) * dstride + c0 + 1]);
                d[u * size + v] = sum;
                s1 += sum;
                s2 += sum * sum;
            }
        }
        let mut t1 = 0i64;
        for u in 0..size {
            for v in 0..size {
                let (i, j) = isometry::map(isom, u, v, size);
                t1 += i64::from(range[i * size + j]) * d[u * size + v];
            }
        }
        let (mut t0, mut t2) = (0i64, 0i64);
        for &r in range {
            t0 += i64::from(r);
            t2 += i64::from(r) * i64::from(r);
        }
        RawMoments {
            s0: (size * size) as i64,
            s1_x4: s1,
            s2_x16: s2,
            t0,
            t1_x4: t1,
            t2,
        }
    }

    /// The same six moments, computed by the naive real-valued definition of §8 (`d` is
    /// the floating-point mean of the four pixels, not `D = 4d`) — a reference
    /// implementation that shares no code with `integer_moments`.
    fn f64_reference_moments(
        range: &[u8],
        domain: &[u8],
        size: usize,
        isom: u8,
    ) -> (f64, f64, f64, f64, f64) {
        let dstride = 2 * size;
        let mut d = vec![0.0f64; size * size];
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for u in 0..size {
            for v in 0..size {
                let (r0, c0) = (2 * u, 2 * v);
                let mean = (f64::from(domain[r0 * dstride + c0])
                    + f64::from(domain[r0 * dstride + c0 + 1])
                    + f64::from(domain[(r0 + 1) * dstride + c0])
                    + f64::from(domain[(r0 + 1) * dstride + c0 + 1]))
                    / 4.0;
                d[u * size + v] = mean;
                s1 += mean;
                s2 += mean * mean;
            }
        }
        let mut t1 = 0.0f64;
        for u in 0..size {
            for v in 0..size {
                let (i, j) = isometry::map(isom, u, v, size);
                t1 += f64::from(range[i * size + j]) * d[u * size + v];
            }
        }
        let (mut t0, mut t2) = (0.0f64, 0.0f64);
        for &r in range {
            t0 += f64::from(r);
            t2 += f64::from(r) * f64::from(r);
        }
        (s1, s2, t0, t1, t2)
    }

    /// Step 6's exit criterion: the integer moments must be *exactly* equal to a naive
    /// `f64` reference over 1e6 random blocks — equality, not closeness (A7).
    #[test]
    fn integer_moments_are_exact_over_a_million_random_blocks() {
        let mut rng = Rng(0x5EED_C0DE_1234_5678);
        for _ in 0..1_000_000 {
            let size = rng.size();
            let mut range = vec![0u8; size * size];
            for b in &mut range {
                *b = rng.byte();
            }
            let mut domain = vec![0u8; 4 * size * size];
            for b in &mut domain {
                *b = rng.byte();
            }
            let isom = rng.isometry();

            let m = integer_moments(&range, &domain, size, isom);
            let (s1, s2, t0, t1, t2) = f64_reference_moments(&range, &domain, size, isom);

            assert_eq!(m.s1_x4 as f64 / 4.0, s1, "s1 at size {size}");
            assert_eq!(m.s2_x16 as f64 / 16.0, s2, "s2 at size {size}");
            assert_eq!(m.t0 as f64, t0, "t0 at size {size}");
            assert_eq!(m.t1_x4 as f64 / 4.0, t1, "t1 at size {size}");
            assert_eq!(m.t2 as f64, t2, "t2 at size {size}");
        }
    }

    #[test]
    fn fit_f64_matches_the_flat128_worked_example() {
        // docs/mars1-format.md §11: a constant 128 block has det == 0, so alfa == 0 and
        // the search never even needs a domain — this exercises fit_f64's det==0 branch
        // and the exact qbeta the spec derives by hand.
        let size = 16i64;
        let m = RawMoments {
            s0: size * size,
            s1_x4: 128 * 4 * size * size, // every domain sample D = 4*128 = 512
            s2_x16: 512 * 512 * size * size,
            t0: 128 * size * size,
            t1_x4: 128 * 512 * size * size,
            t2: 128 * 128 * size * size,
        };
        let (qalfa, qbeta, rms) = fit_f64(m, 1.0, 4, 7);
        assert_eq!(qalfa, 0);
        assert_eq!(
            qbeta, 64,
            "not the zero-alfa refit yet, just the raw search fit"
        );
        // §11: sqrt((t2 - 2*beta*t0 + s0*beta^2)/s0) with beta = 64/127*255 = 128.50...,
        // not zero — det==0 forces alfa=0, but qbeta's own quantisation still leaves a
        // residual against the true mean of 128.
        assert!((rms - 0.503_937_007_873_617_4).abs() < 1e-12, "rms={rms}");
    }

    #[test]
    fn zero_alfa_override_matches_best_beta() {
        // §8.1: once qalfa == 0, qbeta is discarded and refit from the block mean alone.
        assert_eq!(
            best_beta(128 * 256, 256, 7),
            64,
            "matches §11's worked example"
        );
    }

    /// A small textured synthetic image -- enough spatial structure that both "leaf" and
    /// "split further" are genuinely live options at every level, unlike a flat or
    /// perfectly self-similar fixture where one side trivially always wins.
    fn textured_image(w: usize, h: usize) -> Plane {
        let mut data = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                let v =
                    ((r * 37 + c * 19) % 256) as i32 + (((r / 8) as i32 * (c / 8) as i32 * 3) % 64);
                data[r * w + c] = v.clamp(0, 255) as u8;
            }
        }
        Plane::from_vec(w, h, data)
    }

    /// Half flat, half noisy -- unlike [`textured_image`] (whose block-periodic texture
    /// makes every same-size block statistically identical, so the whole image tends to
    /// flip leaf size in lockstep), this deliberately gives different regions genuinely
    /// different local complexity, so a *single* lambda can legitimately keep the flat
    /// half as large leaves while the noisy half keeps splitting to `min_size`.
    fn half_flat_half_noisy_image(w: usize, h: usize) -> Plane {
        let mut data = vec![0u8; w * h];
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for r in 0..h {
            for c in 0..w {
                data[r * w + c] = if c < w / 2 {
                    128
                } else {
                    (next() & 0xff) as u8
                };
            }
        }
        Plane::from_vec(w, h, data)
    }

    fn rd_params(lambda: f64) -> EncodeParams {
        EncodeParams {
            min_size: 4,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            t_rms: 8.0,
            zero_threshold: 0,
            lambda: Some(lambda),
        }
    }

    /// Step 14's P14.3 harness-sanity check: at a moderate lambda, the RD walk must not
    /// degenerate to "every block is one leaf" or "every block splits to min_size" --
    /// either would indicate a sign error or unit mismatch in `J = D + lambda*R`
    /// (`docs/predictions.md`'s Step 14 prediction), not a genuine partition decision.
    #[test]
    fn moderate_lambda_produces_a_real_mixed_partition_not_a_degenerate_one() {
        let image = half_flat_half_noisy_image(64, 64);
        let (hdr, leaves, evals) = encode_image(&image, &rd_params(1500.0));
        assert!(evals > 0, "RD walk must actually search");
        let sizes: std::collections::HashSet<u32> = leaves.iter().map(|l| l.size).collect();
        assert!(
            sizes.len() > 1,
            "expected a mix of leaf sizes at a moderate lambda, got only {sizes:?}"
        );
        assert!(
            leaves
                .iter()
                .all(|l| l.size >= hdr.min_size && l.size <= hdr.max_size),
            "every leaf must respect min/max_size"
        );
    }

    /// Very low lambda (rate essentially free) should push the RD search toward the
    /// smallest blocks the format allows far more often than a very high lambda (rate
    /// dominates, large cheap-to-code blocks win) -- a basic monotonicity sanity check
    /// independent of the exact BD-rate number.
    #[test]
    fn higher_lambda_yields_fewer_larger_leaves_than_lower_lambda() {
        let image = half_flat_half_noisy_image(64, 64);
        let (_hdr, low_leaves, _) = encode_image(&image, &rd_params(1.0));
        let (_hdr, high_leaves, _) = encode_image(&image, &rd_params(5000.0));
        assert!(
            high_leaves.len() < low_leaves.len(),
            "high lambda ({}) leaves should be fewer than low lambda ({})",
            high_leaves.len(),
            low_leaves.len()
        );
    }

    /// Step 12/14's determinism discipline: the RD walk shares `split`'s exact
    /// `rayon::join` structure, so its output must also be bit-identical regardless of
    /// how many threads searched it -- the frozen, read-only `RateModels` snapshot is
    /// what makes this possible without a mutable-model race.
    #[test]
    fn rd_walk_is_bit_identical_across_thread_counts() {
        let image = textured_image(96, 96);
        let params = rd_params(0.05);
        let mut reference: Option<(Header, Vec<Leaf>)> = None;
        for threads in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (hdr, mut leaves, _evals) = pool.install(|| encode_image(&image, &params));
            leaves.sort_by_key(|l| (l.row, l.col, l.size));
            match &reference {
                None => reference = Some((hdr, leaves)),
                Some((ref_hdr, ref_leaves)) => {
                    assert_eq!(&hdr, ref_hdr, "header differs at {threads} threads");
                    assert_eq!(&leaves, ref_leaves, "leaf set differs at {threads} threads");
                }
            }
        }
    }

    // ------------------------------------------------------------------------ Step 16

    /// [`adaptive_shift`]'s three-way routing, checked directly rather than only through
    /// an end-to-end encode -- a cheap, exact-equality oracle for the density-selection
    /// logic itself.
    #[test]
    fn adaptive_shift_routes_high_low_and_mid_rms_correctly() {
        assert_eq!(
            adaptive_shift(4, 20.0),
            2,
            "high RMS should halve the base shift"
        );
        assert_eq!(
            adaptive_shift(4, 2.0),
            8,
            "low RMS should double the base shift"
        );
        assert_eq!(
            adaptive_shift(4, 10.0),
            4,
            "mid RMS should leave the base shift unchanged"
        );
        assert_eq!(
            adaptive_shift(1, 20.0),
            1,
            "halved shift is floored at 1, never 0"
        );
    }

    /// `block_rms` must actually distinguish a flat block from a noisy one -- otherwise
    /// [`adaptive_shift`]'s routing would never fire in practice regardless of the
    /// thresholds chosen.
    #[test]
    fn block_rms_is_zero_on_flat_and_nonzero_on_noisy() {
        let flat = Plane::from_vec(16, 16, vec![100u8; 16 * 16]);
        assert_eq!(block_rms(&flat, 0, 0, 8), 0.0);

        let image = half_flat_half_noisy_image(16, 16);
        let flat_half = block_rms(&image, 0, 0, 4);
        let noisy_half = block_rms(&image, 0, 12, 4);
        assert_eq!(flat_half, 0.0, "left half is constant 128");
        assert!(
            noisy_half > 20.0,
            "right half is uniform random bytes: rms={noisy_half}"
        );
    }

    /// Step 16's own A/B: `adaptive_density: true` must actually change the domain
    /// positions searched (and hence, generically, the resulting leaves) relative to
    /// `false` on an image with genuinely non-uniform local complexity -- otherwise the
    /// density knob would be wired in but inert. Not a BD-rate claim, just a harness-
    /// sanity check that the feature does something observable.
    #[test]
    fn adaptive_density_changes_the_partition_on_a_mixed_complexity_image() {
        let image = half_flat_half_noisy_image(64, 64);
        let params = rd_params(200.0);
        let (_hdr, base_leaves, _evals, _stats) =
            encode_image_rd_with_modes_and_density(&image, &params, [true; 4], false);
        let (_hdr, adaptive_leaves, _evals, _stats) =
            encode_image_rd_with_modes_and_density(&image, &params, [true; 4], true);
        assert_ne!(
            base_leaves, adaptive_leaves,
            "adaptive density should change at least one leaf's chosen domain/partition \
             on an image with genuinely non-uniform local complexity"
        );
    }

    /// Step 12/14's determinism discipline, extended to Step 16's new density path:
    /// `block_rms`/`adaptive_shift` are pure functions of already-read pixel data, so the
    /// adaptive-density RD walk must be bit-identical across thread counts for exactly the
    /// same reason `rd_walk_is_bit_identical_across_thread_counts` holds for the
    /// fixed-density path.
    #[test]
    fn adaptive_density_rd_walk_is_bit_identical_across_thread_counts() {
        let image = textured_image(96, 96);
        let params = rd_params(0.05);
        let mut reference: Option<(Header, Vec<Leaf>)> = None;
        for threads in [1, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (hdr, mut leaves, _evals, _stats) = pool.install(|| {
                encode_image_rd_with_modes_and_density(&image, &params, [true; 4], true)
            });
            leaves.sort_by_key(|l| (l.row, l.col, l.size));
            match &reference {
                None => reference = Some((hdr, leaves)),
                Some((ref_hdr, ref_leaves)) => {
                    assert_eq!(&hdr, ref_hdr, "header differs at {threads} threads");
                    assert_eq!(&leaves, ref_leaves, "leaf set differs at {threads} threads");
                }
            }
        }
    }

    /// `adaptive_density: false` (Step 16's default arm) must reproduce
    /// `encode_image_rd_with_modes`'s output exactly -- the whole point of threading the
    /// flag through as an additive superset rather than changing existing behaviour.
    #[test]
    fn adaptive_density_false_is_byte_identical_to_the_pre_step16_path() {
        let image = half_flat_half_noisy_image(64, 64);
        let params = rd_params(200.0);
        let via_old = encode_image_rd_with_modes(&image, &params, [true; 4]);
        let via_new = encode_image_rd_with_modes_and_density(&image, &params, [true; 4], false);
        assert_eq!(via_old.0, via_new.0, "header differs");
        assert_eq!(via_old.1, via_new.1, "leaves differ");
        assert_eq!(via_old.2, via_new.2, "evals differ");
        assert_eq!(via_old.3, via_new.3, "mode stats differ");
    }

    // ------------------------------------------------------------------------ Step 15

    /// [`ModeStats`]'s own internal-consistency oracle: the number of leaves reported by
    /// [`encode_image_rd`] must equal the sum of its mode histogram, and every leaf-vs-
    /// split decision point counted must be non-negative and actually add up over a real
    /// image -- cheap, exact-equality checks that would catch a bookkeeping bug in
    /// [`walk_rd`]'s stats threading long before it showed up as a wrong BD-rate number.
    #[test]
    fn mode_stats_leaf_count_matches_the_actual_leaf_count() {
        let image = half_flat_half_noisy_image(64, 64);
        let (_hdr, leaves, _evals, stats) = encode_image_rd(&image, &rd_params(200.0));
        let histogram_total: u64 = stats.leaf_modes.iter().sum();
        assert_eq!(
            histogram_total,
            leaves.len() as u64,
            "mode histogram total must equal leaf count"
        );
        assert!(
            stats.split_decisions + stats.leaf_decisions > 0,
            "a 64x64 image with min_size 4 must visit at least one leaf-vs-split decision point"
        );
    }

    /// The legacy `lambda: None` path never runs [`walk_rd`] and so never makes a mode
    /// decision -- [`encode_image_rd`]'s stats must reflect that honestly (all zero)
    /// rather than reporting misleading nonzero counts for a path that has no modes.
    #[test]
    fn legacy_path_reports_all_zero_mode_stats() {
        let image = textured_image(64, 64);
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
        let (_hdr, _leaves, _evals, stats) = encode_image_rd(&image, &params);
        assert_eq!(stats, ModeStats::default());
    }

    /// A block that is genuinely just a linear intensity ramp (no texture at all) should
    /// have mode 1 (affine) available and competitive -- not necessarily always winning
    /// (mode 2/3 can still win if a matching domain happens to exist), but present in the
    /// candidate set with a near-zero SSE, which is what [`affine_fit`] exists to capture
    /// that mode 0 (flat) structurally cannot.
    #[test]
    fn affine_fit_recovers_a_pure_gradient_almost_exactly() {
        let size = 8usize;
        let mut block = vec![0.0f64; size * size];
        for u in 0..size {
            for v in 0..size {
                block[u * size + v] = 10.0 + 3.0 * u as f64 + 1.5 * v as f64;
            }
        }
        let (_qbeta, qgx, qgy, sse) = affine_fit(&block, size, 7);
        assert!(
            sse < 5.0,
            "a pure ramp should fit almost exactly, got sse={sse}"
        );
        assert!(
            qgx > 0,
            "positive u-gradient should quantise to a positive qgx, got {qgx}"
        );
        assert!(
            qgy > 0,
            "positive v-gradient should quantise to a positive qgy, got {qgy}"
        );
    }

    /// A DCT residual's own round trip (encode-time quantise/dequantise, matching
    /// [`crate::dct`]/[`crate::quant`]'s own unit tests but exercised through
    /// [`residual_for_candidate`]'s actual call shape): coding the *same* prediction as
    /// the true pixels should drive every residual coefficient toward zero, since a
    /// perfect prediction has nothing left to correct.
    #[test]
    fn zero_residual_when_prediction_is_exact() {
        let size = 8usize;
        // qalfa == 0 collapses mode 2's fractal prediction to a flat `beta` (§8's own
        // alfa2 == 0 branch) -- picking `qbeta` and the block's constant value to be the
        // *exact* same real number (rather than an independently chosen pixel value)
        // means the prediction error is exactly 0.0 in every pixel, not merely close,
        // which is what makes this an equality assertion rather than a tolerance.
        let qbeta = 50u32;
        let max_qbeta = 127.0;
        let beta = f64::from(qbeta) / max_qbeta * 255.0;
        // `residual_for_candidate`'s prediction is `0.5 + d*alfa + beta` unconditionally
        // (§10.1's own convention, alfa == 0 here just zeroes the domain term) -- so the
        // block value that makes the residual exactly 0.0 is `beta + 0.5`, not `beta`.
        let block = vec![beta + 0.5; size * size];
        let px = vec![0u8; 32 * 32]; // unused: qalfa == 0 means no domain read happens.
        let c = Candidate {
            dom_row: 0,
            dom_col: 0,
            isometry: 0,
            qalfa: 0,
            qbeta,
            rms: 0.0,
            moments: RawMoments {
                s0: 1,
                s1_x4: 0,
                s2_x16: 0,
                t0: 0,
                t1_x4: 0,
                t2: 0,
            },
        };
        let hdr = Header {
            bits_alfa: 4,
            bits_beta: 7,
            min_size: 4,
            max_size: 16,
            shift: 4,
            width: 32,
            height: 32,
            int_max_alfa: 32,
        };
        let (levels, sse) =
            residual_for_candidate(&block, &px, 32, size, &c, &rd_params(1.0), &hdr);
        assert!(
            sse < 1e-6,
            "an exact prediction should leave ~zero residual SSE, got {sse}"
        );
        assert!(
            levels.iter().all(|&l| l == 0),
            "an exact prediction's residual should quantise entirely to zero, got {levels:?}"
        );
    }

    /// **Decisive diagnostic for the gate-15 BD-rate regression investigation
    /// (`docs/decisions.md`'s D40).** The coordinator's hypothesis was that the
    /// step14-equivalent and step15 mode-mask arms might be priced against *different*
    /// frozen `RateModels` snapshots (an apples-to-oranges comparison) -- ruled out by
    /// inspection ([`build_rate_snapshot`] calls the legacy [`walk`], which never reads
    /// `ctx.allowed_modes` at all, so both arms' snapshots are constructed identically).
    ///
    /// This test checks the *provable* mathematical property directly, reading
    /// [`walk_rd`]'s own internal `RdResult.r`/`.d` -- the exact quantity `best_mode_leaf`/
    /// `walk_rd` minimise, in the exact fresh-predictor-per-leaf convention they use
    /// internally (an externally-reconstructed estimate via `events_for_leaves`'s *real*
    /// sequential-predictor convention is a **different** quantity -- `crate::rate`'s own
    /// module doc already documents this fresh-vs-sequential predictor mismatch as a
    /// distinct, narrower approximation, and conflating the two is a test-methodology bug,
    /// not a codec bug; an earlier version of this test made exactly that mistake). Under
    /// the identical frozen snapshot, allowing strictly more leaf-mode candidates at every
    /// node can only ever produce an internal aggregate estimate `<=` the more restricted
    /// mask's -- per-node superset minimisation, summed bottom-up by induction. If the
    /// *real* entropy-coded bpp diverges from that ordering, the divergence is proof that
    /// the estimate itself -- not the mode-competition minimisation -- is where reality and
    /// the frozen snapshot disagree, exactly the already-documented Step 14 rate-estimation
    /// gap (`crate::rate`'s own module doc), not a new Step 15 defect.
    #[test]
    fn aggregate_estimated_cost_is_provably_no_worse_under_more_modes_even_though_real_bpp_can_be()
    {
        let image = textured_image(96, 96);
        let params = EncodeParams {
            min_size: 4,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            t_rms: 8.0,
            zero_threshold: 0,
            lambda: Some(200.0),
        };

        let hdr = Header {
            bits_alfa: params.bits_alfa,
            bits_beta: params.bits_beta,
            min_size: params.min_size,
            max_size: params.max_size,
            shift: params.shift,
            width: image.width() as u32,
            height: image.height() as u32,
            int_max_alfa: quantise_f64(params.max_alfa / 8.0 * 256.0, 255),
        };
        let contracted = Contracted::build(&image);
        // The exact same snapshot-construction call both mode-mask arms use internally
        // (`encode_image_rd_with_modes`) -- built once, shared explicitly here so there is
        // no possibility of the two arms below seeing different snapshots.
        let rate = build_rate_snapshot(&image, &hdr, &contracted, &params);
        let lambda = params.lambda.unwrap();

        let run = |allowed_modes: [bool; 4]| -> RdResult {
            let ctx = Ctx {
                image: &image,
                contracted: &contracted,
                hdr: &hdr,
                params: &params,
                rate: Some(&rate),
                allowed_modes,
                adaptive_density: false,
            };
            walk_rd(&ctx, 0, 0, hdr.virtual_size(), lambda)
        };
        let real_bytes = |leaves: &[Leaf]| -> usize {
            crate::mars_format::write(&hdr, leaves)
                .expect("a valid partition always writes")
                .len()
        };

        let result_2mode = run([true, false, true, false]);
        let result_4mode = run([true; 4]);
        let real_2mode = real_bytes(&result_2mode.leaves) * 8;
        let real_4mode = real_bytes(&result_4mode.leaves) * 8;

        eprintln!(
            "2-mode: internal d={:.1} r={:.1} bits, real={real_2mode} bits ({} leaves)\n\
             4-mode: internal d={:.1} r={:.1} bits, real={real_4mode} bits ({} leaves)",
            result_2mode.d,
            result_2mode.r,
            result_2mode.leaves.len(),
            result_4mode.d,
            result_4mode.r,
            result_4mode.leaves.len(),
        );

        // The provable half: allowing more modes under the *same* frozen snapshot can
        // only lower (or tie) the internal aggregate J = D + lambda*R -- every node's own
        // candidate set is a strict superset, and `min_by` over a superset (summed
        // bottom-up by induction across the whole tree) is never worse.
        let j_2mode = result_2mode.d + lambda * result_2mode.r;
        let j_4mode = result_4mode.d + lambda * result_4mode.r;
        assert!(
            j_4mode <= j_2mode + 1e-6,
            "4-mode internal J ({j_4mode:.1}) must never exceed the 2-mode internal J \
             ({j_2mode:.1}) under the identical frozen snapshot -- if this fails, the bug \
             is in best_mode_leaf's/walk_rd's minimisation itself, not the estimate"
        );

        // The diagnostic half, reported (not a hard pass/fail -- it is the *expected*
        // signature of the known rate-estimation gap, not a new assertion this test
        // exists to enforce): the internal estimate can be more optimistic, relative to
        // what the real entropy coder actually charges, for the 4-mode arm than for the
        // 2-mode arm -- exactly what "the frozen snapshot never observed modes 1/3's
        // fields at all" predicts, and the mechanism `docs/decisions.md`'s D40 names as
        // the BD-rate regression's root cause.
        let gap_2mode = real_2mode as f64 - result_2mode.r;
        let gap_4mode = real_4mode as f64 - result_4mode.r;
        eprintln!(
            "estimate-vs-real gap (real bits - internal r estimate): 2-mode={gap_2mode:.1}, \
             4-mode={gap_4mode:.1} (4-mode's larger gap is the expected signature)"
        );
    }
}
