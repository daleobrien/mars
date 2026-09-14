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
    /// larger than `min_size`.
    pub t_rms: f64,
    /// `-z`. The 1998 default is 0, meaning the override (§8.1) fires exactly when
    /// `qalfa == 0`.
    pub zero_threshold: u32,
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
                let t1_x4 =
                    cross_term_permuted(contracted, dr_half, dc_half, size_u, &range_by_iso[k as usize]);
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
            dom_col += params.shift;
        }
        dom_row += params.shift;
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
fn permute_range(range: &[u8], k: u8, size: usize) -> Vec<u8> {
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
/// fixed isometry (see [`permute_range`]'s doc).
fn cross_term_permuted(contracted: &Contracted, dr: usize, dc: usize, size: usize, range_k: &[u8]) -> i64 {
    mars_simd::moments::dot_u8_i32_window(range_k, &contracted.data, contracted.stride, dr, dc, size)
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
pub fn encode_image(image: &Plane, params: &EncodeParams) -> (Header, Vec<Leaf>, u64) {
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
    let ctx = Ctx {
        image,
        contracted: &contracted,
        hdr: &hdr,
        params,
    };
    let (leaves, evals) = walk(&ctx, 0, 0, hdr.virtual_size());
    (hdr, leaves, evals)
}

/// The read-only context one `walk` recursion shares — bundled so the recursive calls
/// stay readable instead of threading four parameters through every one. Every field is
/// a shared reference to plain data (no interior mutability), so `Ctx` is `Sync` and safe
/// to share across the `rayon::join` calls below.
struct Ctx<'a> {
    image: &'a Plane,
    contracted: &'a Contracted,
    hdr: &'a Header,
    params: &'a EncodeParams,
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
            qalfa: 0,
            qbeta: pixel & ((1 << hdr.bits_beta) - 1),
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
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
            qalfa: 0,
            qbeta: 0,
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
        },
        |c| Leaf {
            row,
            col,
            size,
            qalfa: c.qalfa,
            qbeta: c.qbeta,
            isometry: c.isometry,
            dom_row: c.dom_row,
            dom_col: c.dom_col,
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
}
