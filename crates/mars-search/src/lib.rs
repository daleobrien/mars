//! Mars 1's six classical fractal-search speed-up methods (`reference/mars1/index_func.c`,
//! `coding_func.c`), behind one [`CandidateRetriever`] trait — Step 9. `mars-codec`'s
//! Step 6/7 exhaustive encoder evaluates every `(domain, isometry)` pair for every range
//! block; every method here instead restricts that search to a small candidate subset,
//! chosen by an index built once per block size ahead of the quadtree walk (mirroring
//! Mars 1's own `*Indexing`/`*Coding` function-pair structure, noted in its own README).
//!
//! **Framing (`implementation-plan.md` Step 9).** Without a bit-exactness requirement
//! these are no longer compatibility targets — they are baselines for the recall
//! comparison against Step 8's oracle, which is what they were always scientifically for.
//! `evals` (§M5) is incremented at exactly one site per method, [`search_block`], mirroring
//! `coding_func.c`'s six `comparisons++` sites (one per method, at the point a candidate's
//! affine fit + RMS is actually computed) — see that function's doc for the exact
//! bookkeeping.

pub mod classify;
pub mod exhaustive;
pub mod fisher;
pub mod hurtgen;
pub mod kdtree;
pub mod masscenter;
pub mod mc_saupe;
pub mod saupe;
pub mod saupe_fisher;
pub mod tables;

use mars_codec::encode::{cross_term, domain_sums, Contracted, EncodeParams, RawMoments};
use mars_codec::ifs::{Header, Leaf};
use mars_core::Plane;

/// What one [`CandidateRetriever::index`] call needs: the whole image's 2:1 box-sum plane
/// (§9 of the format doc; shared with `mars-codec`'s exhaustive search, never
/// recomputed), the block `size` being indexed, and the domain scan geometry (`shift`,
/// image dimensions) that determines which domain positions exist at all.
///
/// One `DomainPool` corresponds to one `(image, size)` pair — Mars 1 calls each method's
/// `*Indexing(size, tip)` once per size before coding begins at any size, and this is that
/// call's argument.
pub struct DomainPool<'a> {
    pub contracted: &'a Contracted,
    pub size: u32,
    pub shift: u32,
    pub image_width: u32,
    pub image_height: u32,
}

impl<'a> DomainPool<'a> {
    /// Every valid domain top-left `(row, col)` at this pool's size and shift — the same
    /// legal range the exhaustive search in `mars_codec::encode` scans, and what every
    /// method's indexing function iterates over (`index_func.c`'s
    /// `i < image_height - 2*size + 1; i += SHIFT`).
    pub fn domain_positions(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let max_row = self.image_height.saturating_sub(2 * self.size);
        let max_col = self.image_width.saturating_sub(2 * self.size);
        let shift = self.shift.max(1);
        (0..=max_row)
            .step_by(shift as usize)
            .flat_map(move |r| (0..=max_col).step_by(shift as usize).map(move |c| (r, c)))
    }

    /// The `size x size` block of contracted (`D = 4x` box-sum) samples at one domain
    /// position, as an `f64` [`classify::Block`] — the common input every classifier
    /// needs.
    pub fn domain_block(&self, row: u32, col: u32) -> classify::Block {
        let (dr, dc) = ((row / 2) as usize, (col / 2) as usize);
        let size = self.size as usize;
        let mut data = vec![0.0f64; size * size];
        for u in 0..size {
            for v in 0..size {
                data[u * size + v] = f64::from(self.contracted.at(dr + u, dc + v));
            }
        }
        classify::Block::new(size, data)
    }
}

/// One range block being coded: its position/size and raw pixel samples — everything a
/// [`CandidateRetriever::candidates`] call needs to classify the range and restrict the
/// search, and everything [`search_block`] needs to run the shared affine fit.
pub struct RangeBlock<'a> {
    pub row: u32,
    pub col: u32,
    pub size: u32,
    /// `size x size`, row-major, raw range pixels.
    pub pixels: &'a [u8],
}

impl RangeBlock<'_> {
    pub fn t0_t2(&self) -> (i64, i64) {
        let mut t0 = 0i64;
        let mut t2 = 0i64;
        for &p in self.pixels {
            t0 += i64::from(p);
            t2 += i64::from(p) * i64::from(p);
        }
        (t0, t2)
    }

    /// The range block's pixels as an `f64` [`classify::Block`] — used by every method
    /// that classifies (or canonicalises the orientation of) the range itself.
    pub fn as_block(&self) -> classify::Block {
        let data: Vec<f64> = self.pixels.iter().map(|&p| f64::from(p)).collect();
        classify::Block::new(self.size as usize, data)
    }
}

/// One `(domain, isometry)` candidate a [`CandidateRetriever`] proposes for a range block
/// — full-image domain coordinates (matching `Leaf::dom_row`/`dom_col`) and an isometry
/// code in `mars_codec::isometry`'s convention (identical numbering to Mars 1's, verified
/// against `coding_func.c`'s per-isometry `t1` accumulation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub dom_row: u32,
    pub dom_col: u32,
    pub isometry: u8,
}

/// An iterator over one range block's candidate set. Owns its contents (built fresh per
/// `candidates()` call from whatever bucket/tree lookup the method performs) rather than
/// borrowing from `&self`, but is named with a lifetime to match the trait's sketch in
/// the Step 9 brief.
pub struct CandidateIter<'a> {
    iter: std::vec::IntoIter<Candidate>,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl CandidateIter<'_> {
    fn new(v: Vec<Candidate>) -> Self {
        Self {
            iter: v.into_iter(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl Iterator for CandidateIter<'_> {
    type Item = Candidate;
    fn next(&mut self) -> Option<Candidate> {
        self.iter.next()
    }
}

/// The extension point (Step 9 brief): "a new method needs only two functions — an
/// indexing function and a coding function" (Mars 1's own README), made type-safe.
/// `index` is called once per `(image, size)` before any range block of that size is
/// coded; `candidates` is called once per range block and must restrict the search to
/// whatever subset the method's index supports — `Exhaustive` is the trivial case that
/// returns every legal domain position x isometry.
pub trait CandidateRetriever {
    fn index(&mut self, pool: &DomainPool);
    fn candidates(&self, range: &RangeBlock) -> CandidateIter<'_>;
}

/// Run one range block's search through a [`CandidateRetriever`]: the shared driver every
/// method uses, so the affine fit and the `evals` bookkeeping (§M5) live in exactly one
/// place. `comparisons++`'s analogue: `evals` increments once per candidate that reaches
/// [`mars_codec::encode::fit_f64`], mirroring `coding_func.c`'s six `comparisons++` sites,
/// each of which sits immediately before its own affine-fit block.
///
/// Returns the winning `(candidate, qalfa, qbeta, rms)` (`None` if the retriever proposed
/// no candidates at all) and the number of evals spent.
pub fn search_block(
    retriever: &dyn CandidateRetriever,
    contracted: &Contracted,
    range: &RangeBlock,
    max_alfa: f64,
    bits_alfa: u32,
    bits_beta: u32,
) -> (Option<(Candidate, u32, u32, f64)>, u64) {
    let (t0, t2) = range.t0_t2();
    let size = range.size as usize;
    let s0 = i64::from(range.size) * i64::from(range.size);

    let mut best: Option<(Candidate, u32, u32, f64)> = None;
    let mut evals = 0u64;
    for c in retriever.candidates(range) {
        let (dr_half, dc_half) = ((c.dom_row / 2) as usize, (c.dom_col / 2) as usize);
        let (s1_x4, s2_x16) = domain_sums(contracted, dr_half, dc_half, size);
        let t1_x4 = cross_term(contracted, dr_half, dc_half, size, c.isometry, range.pixels);
        let moments = RawMoments {
            s0,
            s1_x4,
            s2_x16,
            t0,
            t1_x4,
            t2,
        };
        let (qalfa, qbeta, rms) =
            mars_codec::encode::fit_f64(moments, max_alfa, bits_alfa, bits_beta);
        evals += 1;
        let better = match &best {
            None => true,
            Some((_, _, _, brms)) => rms < *brms,
        };
        if better {
            best = Some((c, qalfa, qbeta, rms));
        }
    }
    (best, evals)
}

/// The seven methods this crate provides, `Exhaustive` included as the sanity baseline
/// (Step 9 brief). Naming matches `mars_bench::mars1::Method::key()` so a result row can
/// be joined against the Mars 1 baseline by string key without a translation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodName {
    Exhaustive,
    Fisher,
    Hurtgen,
    MassCenter,
    Saupe,
    SaupeFisher,
    McSaupe,
}

impl MethodName {
    pub const ALL: [MethodName; 7] = [
        MethodName::Exhaustive,
        MethodName::Fisher,
        MethodName::Hurtgen,
        MethodName::MassCenter,
        MethodName::Saupe,
        MethodName::SaupeFisher,
        MethodName::McSaupe,
    ];

    pub fn key(self) -> &'static str {
        match self {
            MethodName::Exhaustive => "exhaustive",
            MethodName::Fisher => "fisher",
            MethodName::Hurtgen => "hurtgen",
            MethodName::MassCenter => "masscenter",
            MethodName::Saupe => "saupe",
            MethodName::SaupeFisher => "saupe-fisher",
            MethodName::McSaupe => "mc-saupe",
        }
    }

    /// A fresh, not-yet-indexed retriever — call `index()` once per size before coding.
    pub fn new_retriever(self) -> Box<dyn CandidateRetriever> {
        match self {
            MethodName::Exhaustive => Box::new(exhaustive::Exhaustive::default()),
            MethodName::Fisher => Box::new(fisher::Fisher::default()),
            MethodName::Hurtgen => Box::new(hurtgen::Hurtgen::default()),
            MethodName::MassCenter => Box::new(masscenter::MassCenter::default()),
            MethodName::Saupe => Box::new(saupe::Saupe::default()),
            MethodName::SaupeFisher => Box::new(saupe_fisher::SaupeFisher::default()),
            MethodName::McSaupe => Box::new(mc_saupe::McSaupe::default()),
        }
    }
}

/// One method, indexed once per size that a partition might ever produce a leaf at
/// (`min_size..=max_size`, every power of two). Built once per image by
/// [`encode_image`]/the `mars-bench` driver, then reused across every range block of a
/// matching size during the quadtree walk.
pub struct SizedRetrievers {
    /// Indexed by `tip = log2(size)`; `0` is never populated (size 1 never searches).
    by_tip: Vec<Option<Box<dyn CandidateRetriever>>>,
}

impl SizedRetrievers {
    /// Build and index one fresh retriever per size in `[min_size, max_size]`, via
    /// `make(size) -> Box<dyn CandidateRetriever>` (not yet indexed) — this crate's
    /// method constructors (`fisher::Fisher::new()`, etc.) all provide one.
    pub fn build(
        contracted: &Contracted,
        image_width: u32,
        image_height: u32,
        shift: u32,
        min_size: u32,
        max_size: u32,
        mut make: impl FnMut() -> Box<dyn CandidateRetriever>,
    ) -> Self {
        let max_tip = (max_size as f64).log2().round() as usize;
        let mut by_tip: Vec<Option<Box<dyn CandidateRetriever>>> =
            (0..=max_tip).map(|_| None).collect();
        let mut size = min_size;
        while size <= max_size {
            let tip = (size as f64).log2().round() as usize;
            let pool = DomainPool {
                contracted,
                size,
                shift,
                image_width,
                image_height,
            };
            let mut r = make();
            r.index(&pool);
            by_tip[tip] = Some(r);
            size *= 2;
        }
        Self { by_tip }
    }

    pub fn get(&self, size: u32) -> &dyn CandidateRetriever {
        let tip = (size as f64).log2().round() as usize;
        self.by_tip[tip]
            .as_deref()
            .unwrap_or_else(|| panic!("no retriever indexed for size {size} (tip {tip})"))
    }
}

/// One leaf's chosen encoding, in the same shape `mars_bench::recall::MethodPick` uses
/// (row/col/size locate the range block, the rest is the method's chosen fit) — kept as
/// a plain struct here rather than a tuple (`clippy::type_complexity`) and rather than a
/// dependency on `mars-bench` (wrong direction: `mars-bench` depends on this crate, not
/// the reverse). `mars-bench`'s driver converts one-for-one into `MethodPick`.
#[derive(Debug, Clone, Copy)]
pub struct Pick {
    pub row: u32,
    pub col: u32,
    pub size: u32,
    pub dom_row: u32,
    pub dom_col: u32,
    pub isometry: u8,
    pub qalfa: u32,
    pub qbeta: u32,
    pub rms: f64,
}

/// Encode one image with a `SizedRetrievers` set, following exactly the same quadtree
/// partition and §8.1 zero-alfa override as `mars_codec::encode::encode_image` (so RD
/// curves are comparable) but delegating each block's search to the matching
/// [`CandidateRetriever`] instead of the exhaustive scan. Also returns one
/// [`mars_bench`-shaped] pick per leaf as `(row, col, size, dom_row, dom_col, isometry,
/// qalfa, qbeta, rms)`, for recall scoring against the Step 8 oracle.
pub fn encode_image(
    image: &Plane,
    params: &EncodeParams,
    retrievers: &SizedRetrievers,
) -> (Header, Vec<Leaf>, u64, Vec<Pick>) {
    let hdr = Header {
        bits_alfa: params.bits_alfa,
        bits_beta: params.bits_beta,
        min_size: params.min_size,
        max_size: params.max_size,
        shift: params.shift,
        width: image.width() as u32,
        height: image.height() as u32,
        int_max_alfa: quantise(params.max_alfa / 8.0 * 256.0, 255),
    };
    let contracted = mars_codec::encode::build_contracted(image);
    let mut leaves = Vec::new();
    let mut evals = 0u64;
    let mut picks = Vec::new();
    walk(
        image,
        &contracted,
        &hdr,
        params,
        retrievers,
        0,
        0,
        hdr.virtual_size(),
        &mut leaves,
        &mut evals,
        &mut picks,
    );
    (hdr, leaves, evals, picks)
}

#[allow(clippy::too_many_arguments)]
fn walk(
    image: &Plane,
    contracted: &Contracted,
    hdr: &Header,
    params: &EncodeParams,
    retrievers: &SizedRetrievers,
    row: u32,
    col: u32,
    size: u32,
    leaves: &mut Vec<Leaf>,
    evals: &mut u64,
    picks: &mut Vec<Pick>,
) {
    if row >= hdr.height || col >= hdr.width {
        return;
    }
    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    if forced {
        let half = size / 2;
        for (r, c) in quad(row, col, half) {
            walk(
                image, contracted, hdr, params, retrievers, r, c, half, leaves, evals, picks,
            );
        }
        return;
    }

    if size == 1 {
        let pixel = u32::from(image.as_slice()[row as usize * image.width() + col as usize]);
        leaves.push(Leaf {
            row,
            col,
            size,
            qalfa: 0,
            qbeta: pixel & ((1 << hdr.bits_beta) - 1),
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
        });
        return;
    }

    let size_u = size as usize;
    let px = image.as_slice();
    let stride = image.width();
    let mut pixels = vec![0u8; size_u * size_u];
    for i in 0..size_u {
        let src = (row as usize + i) * stride + col as usize;
        pixels[i * size_u..(i + 1) * size_u].copy_from_slice(&px[src..src + size_u]);
    }
    let range = RangeBlock {
        row,
        col,
        size,
        pixels: &pixels,
    };
    let retriever = retrievers.get(size);
    let (best, block_evals) = search_block(
        retriever,
        contracted,
        &range,
        params.max_alfa,
        params.bits_alfa,
        params.bits_beta,
    );
    *evals += block_evals;
    let best_rms = best.map_or(f64::INFINITY, |(_, _, _, rms)| rms);

    if best_rms > params.t_rms && size > hdr.min_size {
        let half = size / 2;
        for (r, c) in quad(row, col, half) {
            walk(
                image, contracted, hdr, params, retrievers, r, c, half, leaves, evals, picks,
            );
        }
        return;
    }

    let mut leaf = best.map_or(
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
        |(c, qalfa, qbeta, _)| Leaf {
            row,
            col,
            size,
            qalfa,
            qbeta,
            isometry: c.isometry,
            dom_row: c.dom_row,
            dom_col: c.dom_col,
        },
    );

    if let Some((c, qalfa, qbeta, rms)) = best {
        picks.push(Pick {
            row,
            col,
            size,
            dom_row: c.dom_row,
            dom_col: c.dom_col,
            isometry: c.isometry,
            qalfa,
            qbeta,
            rms,
        });
    }

    if leaf.qalfa.abs_diff(0) <= params.zero_threshold {
        let mut range_sum = 0i64;
        for i in 0..size_u {
            let src = (row as usize + i) * stride + col as usize;
            for j in 0..size_u {
                range_sum += i64::from(px[src + j]);
            }
        }
        let mean = range_sum as f64 / (size_u * size_u) as f64;
        let max_qbeta = (1u32 << params.bits_beta) - 1;
        leaf.qbeta = quantise(mean / 255.0 * f64::from(max_qbeta), max_qbeta);
        leaf.qalfa = 0;
        leaf.isometry = 0;
        leaf.dom_row = 0;
        leaf.dom_col = 0;
    }
    leaves.push(leaf);
}

fn quad(row: u32, col: u32, half: u32) -> [(u32, u32); 4] {
    [
        (row, col),
        (row + half, col),
        (row, col + half),
        (row + half, col + half),
    ]
}

fn quantise(x: f64, max: u32) -> u32 {
    (0.5 + x).trunc().clamp(0.0, f64::from(max)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use exhaustive::Exhaustive;
    use mars_codec::encode::build_contracted;

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

    /// The harness sanity check (P9.4): the new `mars-search` driver's `Exhaustive`
    /// method must reproduce `mars_codec::encode`'s own exhaustive search exactly —
    /// same winner per block, same evals — since it is meant to be a thin wrapper, not a
    /// re-implementation.
    #[test]
    fn exhaustive_matches_mars_codec_search_block_exactly() {
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
        };

        for size in [4u32, 8, 16] {
            let pool = DomainPool {
                contracted: &contracted,
                size,
                shift: params.shift,
                image_width: image.width() as u32,
                image_height: image.height() as u32,
            };
            let mut retriever = Exhaustive::default();
            retriever.index(&pool);

            for row in (0..image.height() as u32 - size + 1).step_by(size as usize) {
                for col in (0..image.width() as u32 - size + 1).step_by(size as usize) {
                    let size_u = size as usize;
                    let px = image.as_slice();
                    let stride = image.width();
                    let mut pixels = vec![0u8; size_u * size_u];
                    for i in 0..size_u {
                        let src = (row as usize + i) * stride + col as usize;
                        pixels[i * size_u..(i + 1) * size_u]
                            .copy_from_slice(&px[src..src + size_u]);
                    }
                    let range = RangeBlock {
                        row,
                        col,
                        size,
                        pixels: &pixels,
                    };

                    let (got_best, got_evals) = search_block(
                        &retriever,
                        &contracted,
                        &range,
                        params.max_alfa,
                        params.bits_alfa,
                        params.bits_beta,
                    );
                    let (want_best, want_evals) = mars_codec::encode::search_block(
                        &image,
                        &contracted,
                        row,
                        col,
                        size,
                        &params,
                    );

                    assert_eq!(got_evals, want_evals, "evals at ({row},{col},{size})");
                    match (got_best, want_best) {
                        (None, None) => {}
                        (Some((c, qa, qb, rms)), Some(w)) => {
                            assert_eq!(
                                (c.dom_row, c.dom_col, c.isometry, qa, qb),
                                (w.dom_row, w.dom_col, w.isometry, w.qalfa, w.qbeta)
                            );
                            assert!(
                                (rms - w.rms).abs() < 1e-9,
                                "rms mismatch at ({row},{col},{size}): {rms} vs {}",
                                w.rms
                            );
                        }
                        other => panic!("mismatched Option at ({row},{col},{size}): {other:?}"),
                    }
                }
            }
        }
    }
}
