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

pub mod apcc;
pub mod classify;
pub mod exhaustive;
pub mod fisher;
pub mod funnel;
pub mod hurtgen;
pub mod kdtree;
pub mod learned;
pub mod masscenter;
pub mod mc_saupe;
pub mod random;
pub mod saupe;
pub mod saupe_fisher;
pub mod tables;

use mars_codec::encode::{
    cross_term_permuted, domain_sums, permute_range, Candidate as FittedCandidate, Contracted,
    EncodeOptions, EncodeParams, RawMoments, SearchOutcome, SearchProvider, SearchRequest,
};
use mars_codec::ifs::{Header, Leaf};
use mars_codec::isometry;
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
        let bounds = self.size.checked_mul(2).and_then(|diameter| {
            Some((
                self.image_height.checked_sub(diameter)?,
                self.image_width.checked_sub(diameter)?,
            ))
        });
        let shift = self.shift.max(1) as usize;
        bounds.into_iter().flat_map(move |(max_row, max_col)| {
            (0..=max_row)
                .step_by(shift)
                .flat_map(move |r| (0..=max_col).step_by(shift).map(move |c| (r, c)))
        })
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
///
/// `Sync` (Gate C / the parallel `walk` below): `index` runs once, single-threaded, before
/// any `candidates` call; every method's index is plain owned data (no interior
/// mutability) by the time `candidates(&self, ..)` — read-only — is shared across
/// `rayon::join` tasks.
pub trait CandidateRetriever: Sync {
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
    let outcome = search_block_fitted(retriever, contracted, range, max_alfa, bits_alfa, bits_beta);
    (
        outcome.candidate.map(|c| {
            (
                Candidate {
                    dom_row: c.dom_row,
                    dom_col: c.dom_col,
                    isometry: c.isometry,
                },
                c.qalfa,
                c.qbeta,
                c.rms,
            )
        }),
        outcome.evals,
    )
}

/// Search with the codec's exact fitter, retaining the winning fit's integer moments.
/// Candidate order and strict minimum-RMS tie breaking are unchanged from
/// [`search_block`]. Each candidate, including duplicates, counts as one evaluation;
/// an empty candidate set returns no fit and zero evaluations.
pub fn search_block_fitted(
    retriever: &dyn CandidateRetriever,
    contracted: &Contracted,
    range: &RangeBlock,
    max_alfa: f64,
    bits_alfa: u32,
    bits_beta: u32,
) -> SearchOutcome {
    let (t0, t2) = range.t0_t2();
    let size = range.size as usize;
    let s0 = i64::from(range.size) * i64::from(range.size);

    // Permuted once per isometry here, not once per candidate inside the loop below --
    // mirrors `mars_codec::encode::search`'s own fix for the identical regression (D31):
    // `cross_term`'s NEON dot product only pays off once the permutation it needs is
    // amortised across many calls sharing an isometry, and a range block's candidates
    // (at most 8 distinct isometries, `MAPPING[isom][dom_iso]` over `dom_iso in 0..8`)
    // reuse each permutation many times over the whole eval loop (D35).
    let range_by_iso: [Vec<u8>; 8] =
        std::array::from_fn(|k| permute_range(range.pixels, k as u8, size));
    debug_assert_eq!(isometry::ALL, [0, 1, 2, 3, 4, 5, 6, 7]);

    let timing = diag::enabled();
    let cand_start = timing.then(std::time::Instant::now);
    let candidates = retriever.candidates(range);
    if let Some(s) = cand_start {
        diag::CANDIDATES_NS.fetch_add(
            s.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    if timing {
        diag::BLOCKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    let eval_start = timing.then(std::time::Instant::now);
    let mut best: Option<FittedCandidate> = None;
    let mut evals = 0u64;
    for c in candidates {
        let (dr_half, dc_half) = ((c.dom_row / 2) as usize, (c.dom_col / 2) as usize);
        let (s1_x4, s2_x16) = domain_sums(contracted, dr_half, dc_half, size);
        let t1_x4 = cross_term_permuted(
            contracted,
            dr_half,
            dc_half,
            size,
            &range_by_iso[c.isometry as usize],
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
            mars_codec::encode::fit_f64(moments, max_alfa, bits_alfa, bits_beta);
        evals += 1;
        let better = match &best {
            None => true,
            Some(b) => rms < b.rms,
        };
        if better {
            best = Some(FittedCandidate {
                dom_row: c.dom_row,
                dom_col: c.dom_col,
                isometry: c.isometry,
                qalfa,
                qbeta,
                rms,
                moments,
            });
        }
    }
    if let Some(s) = eval_start {
        diag::EVAL_NS.fetch_add(
            s.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        diag::EVALS.fetch_add(evals, std::sync::atomic::Ordering::Relaxed);
    }
    SearchOutcome {
        candidate: best,
        evals,
    }
}

/// A root-causing diagnostic for Gate C's D33 finding (single-threaded Fisher encode
/// measured ~2x slower than Mars 1's C Fisher despite matching evals/transform) — opt-in
/// via `MARS_SEARCH_TIMING=1`, same pattern as `mars-gpu`'s `MARS_GPU_TIMING` (D24). Not
/// on the hot path unless enabled: `diag::enabled()` is a `OnceLock` read, and the timers
/// are plain atomics with no lock contention.
pub mod diag {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;

    pub static CANDIDATES_NS: AtomicU64 = AtomicU64::new(0);
    pub static EVAL_NS: AtomicU64 = AtomicU64::new(0);
    pub static BLOCKS: AtomicU64 = AtomicU64::new(0);
    pub static EVALS: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("MARS_SEARCH_TIMING").is_ok())
    }

    pub fn reset() {
        CANDIDATES_NS.store(0, Ordering::Relaxed);
        EVAL_NS.store(0, Ordering::Relaxed);
        BLOCKS.store(0, Ordering::Relaxed);
        EVALS.store(0, Ordering::Relaxed);
    }

    /// Deliberately coarse: one timer pair per *block* (not per eval), so this adds one
    /// `Instant::now()` pair per `search_block` call rather than per candidate. An earlier
    /// version of this diagnostic timed `domain_sums`/`cross_term`/`fit_f64` individually
    /// (two extra `Instant::now()` calls per *eval*) and measured its own overhead as ~26%
    /// of the loop it was trying to measure at ~15M evals/run -- accurate enough to see
    /// that `domain_sums`+`cross_term` dominates `fit_f64`, but not to trust the absolute
    /// ns/eval split. Kept at this granularity instead: still isolates "candidate-list
    /// construction" from "the per-candidate domain_sums/cross_term/fit_f64 loop" cleanly,
    /// which is what mattered for Gate C's D33 finding.
    pub fn summary() -> String {
        let cand_ms = CANDIDATES_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let eval_ms = EVAL_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let blocks = BLOCKS.load(Ordering::Relaxed);
        let evals = EVALS.load(Ordering::Relaxed);
        format!(
            "blocks={blocks} evals={evals} candidates()={cand_ms:.2}ms eval-loop={eval_ms:.2}ms \
             ({:.1}ns/eval)",
            if evals > 0 {
                eval_ms * 1e6 / evals as f64
            } else {
                0.0
            }
        )
    }
}

/// The methods this crate provides: Step 9's six classical ports plus `Exhaustive` as the
/// sanity baseline, and Step 13's `Funnel`. Naming matches `mars_bench::mars1::Method::key()`
/// for the seven Step 9 names, so a result row can be joined against the Mars 1 baseline by
/// string key without a translation table — `Funnel` has no Mars 1 counterpart (it is a
/// novel Step 13 method, not a 1998 port) and simply has no matching baseline row to join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MethodName {
    Exhaustive,
    Fisher,
    Hurtgen,
    MassCenter,
    Saupe,
    SaupeFisher,
    McSaupe,
    Funnel,
    Learned,
}

impl MethodName {
    pub const ALL: [MethodName; 9] = [
        MethodName::Exhaustive,
        MethodName::Fisher,
        MethodName::Hurtgen,
        MethodName::MassCenter,
        MethodName::Saupe,
        MethodName::SaupeFisher,
        MethodName::McSaupe,
        MethodName::Funnel,
        MethodName::Learned,
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
            MethodName::Funnel => "funnel",
            MethodName::Learned => "learned",
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
            MethodName::Funnel => Box::new(funnel::Funnel::default()),
            MethodName::Learned => Box::new(learned::Learned::default()),
        }
    }
}

/// One method, indexed once per size that a partition might ever produce a leaf at
/// (`min_size..=max_size`, plus smaller sizes needed by border splits). Built once per image by
/// [`encode_image`]/the `mars-bench` driver, then reused across every range block of a
/// matching size during the quadtree walk.
pub struct SizedRetrievers {
    /// Indexed by `tip = log2(size)`; `0` is never populated (size 1 never searches).
    by_tip: Vec<Option<Box<dyn CandidateRetriever>>>,
}

impl SizedRetrievers {
    /// Build and index one fresh retriever per power-of-two size in
    /// `[min_size, max_size]`, plus sizes down to 2 when image borders are not aligned
    /// to `min_size`. Size 1 is always coded directly, without searching.
    /// `make() -> Box<dyn CandidateRetriever>` supplies each unindexed retriever.
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
        let mut size = if image_width % min_size != 0 || image_height % min_size != 0 {
            2
        } else {
            min_size.max(2)
        };
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

/// Owned, prebuilt search indexes for the codec's threshold and RD production walks.
/// All indexing finishes in [`Self::build`]; concurrent searches only read the indexes.
/// Partitioning, DC fallback, mode selection, residuals, and warmup remain codec-owned.
pub struct IndexedSearchProvider {
    base_shift: u32,
    base: SizedRetrievers,
    doubled: Option<(u32, SizedRetrievers)>,
}

impl IndexedSearchProvider {
    /// Prepare the configured and border-split sizes for one image and method.
    /// Adaptive RD density gets a separate doubled-stride index: filtering an already
    /// pruned base-stride candidate list would not perform the same coarse-pool search.
    /// The result owns its indexes and borrows neither the image nor the contracted plane.
    /// Use it only with the image, geometry, and options supplied here; sizes must be
    /// legal powers of two and the base shift must be positive.
    pub fn build(
        image: &Plane,
        contracted: &Contracted,
        params: &EncodeParams,
        options: &EncodeOptions,
        method: MethodName,
    ) -> Self {
        let build = |shift| {
            SizedRetrievers::build(
                contracted,
                image.width() as u32,
                image.height() as u32,
                shift,
                params.min_size,
                params.max_size,
                || method.new_retriever(),
            )
        };
        let base = build(params.shift);
        let doubled = (params.lambda.is_some() && options.adaptive_density).then(|| {
            let shift = params.shift.saturating_mul(2);
            (shift, build(shift))
        });
        Self {
            base_shift: params.shift,
            base,
            doubled,
        }
    }
}

impl SearchProvider for IndexedSearchProvider {
    fn search(&self, request: &SearchRequest<'_>) -> SearchOutcome {
        let indexes = if request.shift == self.base_shift {
            &self.base
        } else {
            let (shift, indexes) = self
                .doubled
                .as_ref()
                .expect("search requested a stride not prepared by IndexedSearchProvider::build");
            assert_eq!(
                request.shift, *shift,
                "search stride differs from the prepared index"
            );
            indexes
        };
        let size = request.size as usize;
        let stride = request.image.width();
        let mut pixels = Vec::with_capacity(size * size);
        for row in request.row as usize..request.row as usize + size {
            let start = row * stride + request.col as usize;
            pixels.extend_from_slice(&request.image.as_slice()[start..start + size]);
        }
        let range = RangeBlock {
            row: request.row,
            col: request.col,
            size: request.size,
            pixels: &pixels,
        };
        search_block_fitted(
            indexes.get(request.size),
            request.contracted,
            &range,
            request.params.max_alfa,
            request.params.bits_alfa,
            request.params.bits_beta,
        )
    }
}

/// One leaf's chosen encoding, in the same shape `mars_bench::recall::MethodPick` uses
/// (row/col/size locate the range block, the rest is the method's chosen fit) — kept as
/// a plain struct here rather than a tuple (`clippy::type_complexity`) and rather than a
/// dependency on `mars-bench` (wrong direction: `mars-bench` depends on this crate, not
/// the reverse). `mars-bench`'s driver converts one-for-one into `MethodPick`.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    let ctx = Ctx {
        image,
        contracted: &contracted,
        hdr: &hdr,
        params,
        retrievers,
    };
    let (leaves, evals, picks) = walk(&ctx, 0, 0, hdr.virtual_size());
    (hdr, leaves, evals, picks)
}

/// The read-only context one `walk` recursion shares (mirrors `mars_codec::encode::Ctx`
/// from Step 12): every field is a shared reference to plain, already-indexed data, so
/// `Ctx` is `Sync` and safe to share across the `rayon::join` calls in [`split`].
struct Ctx<'a> {
    image: &'a Plane,
    contracted: &'a Contracted,
    hdr: &'a Header,
    params: &'a EncodeParams,
    retrievers: &'a SizedRetrievers,
}

/// Gate C: below this block size, a `rayon::join`'s task-spawn/steal overhead costs more
/// than the classified search it would parallelise. Same value and reasoning as
/// `mars_codec::encode::PARALLEL_SIZE_CUTOFF` (Step 12) — chosen well under every
/// project config's `min_size` (>= 4) so the boundary never falls inside a config's own
/// search range.
const PARALLEL_SIZE_CUTOFF: u32 = 8;

/// Search and partition are one RMS-driven recursion, but emission is decoupled from it
/// (Step 12's pattern, ported here for Gate C): each call returns its own
/// `(leaves, evals, picks)` instead of pushing into shared `Vec`s, so independent
/// subtrees can be searched in parallel and merged afterwards in the same TL/BL/TR/BR
/// order the original sequential walk always used.
fn walk(ctx: &Ctx, row: u32, col: u32, size: u32) -> (Vec<Leaf>, u64, Vec<Pick>) {
    let hdr = ctx.hdr;
    if row >= hdr.height || col >= hdr.width {
        return (Vec::new(), 0, Vec::new());
    }
    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    if forced {
        let half = size / 2;
        return split(ctx, row, col, half);
    }

    if size == 1 {
        let pixel =
            u32::from(ctx.image.as_slice()[row as usize * ctx.image.width() + col as usize]);
        let max_qbeta = (1u32 << hdr.bits_beta) - 1;
        let leaf = Leaf {
            row,
            col,
            size,
            mode: 0,
            qalfa: 0,
            qbeta: quantise(f64::from(pixel) / 255.0 * f64::from(max_qbeta), max_qbeta),
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
            qgx: 0,
            qgy: 0,
            residual: Vec::new(),
        };
        return (vec![leaf], 0, Vec::new());
    }

    let size_u = size as usize;
    let px = ctx.image.as_slice();
    let stride = ctx.image.width();
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
    let retriever = ctx.retrievers.get(size);
    let (best, block_evals) = search_block(
        retriever,
        ctx.contracted,
        &range,
        ctx.params.max_alfa,
        ctx.params.bits_alfa,
        ctx.params.bits_beta,
    );
    let best_rms = best.map_or(f64::INFINITY, |(_, _, _, rms)| rms);

    if best_rms > ctx.params.t_rms && size > hdr.min_size {
        let half = size / 2;
        let (leaves, sub_evals, picks) = split(ctx, row, col, half);
        return (leaves, sub_evals + block_evals, picks);
    }

    let mut leaf = best.map_or(
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
        |(c, qalfa, qbeta, _)| Leaf {
            row,
            col,
            size,
            mode: if qalfa == 0 { 0 } else { 2 },
            qalfa,
            qbeta,
            isometry: c.isometry,
            dom_row: c.dom_row,
            dom_col: c.dom_col,
            qgx: 0,
            qgy: 0,
            residual: Vec::new(),
        },
    );

    let mut picks = Vec::new();
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

    if leaf.qalfa.abs_diff(0) <= ctx.params.zero_threshold {
        let mut range_sum = 0i64;
        for i in 0..size_u {
            let src = (row as usize + i) * stride + col as usize;
            for j in 0..size_u {
                range_sum += i64::from(px[src + j]);
            }
        }
        let mean = range_sum as f64 / (size_u * size_u) as f64;
        let max_qbeta = (1u32 << ctx.params.bits_beta) - 1;
        leaf.qbeta = quantise(mean / 255.0 * f64::from(max_qbeta), max_qbeta);
        leaf.qalfa = 0;
        leaf.mode = 0;
        leaf.isometry = 0;
        leaf.dom_row = 0;
        leaf.dom_col = 0;
    }
    (vec![leaf], block_evals, picks)
}

/// Recurse into the four quadrants of a `2*half x 2*half` region at `(row, col)`, in
/// canonical TL/BL/TR/BR order. Above [`PARALLEL_SIZE_CUTOFF`], the two pairs run via
/// `rayon::join`; below it, sequentially on the calling thread. Either way the four
/// results are concatenated in the same fixed order, so the merge introduces no
/// thread-count dependence (mirrors `mars_codec::encode::split`, Step 12).
fn split(ctx: &Ctx, row: u32, col: u32, half: u32) -> (Vec<Leaf>, u64, Vec<Pick>) {
    let quadrants = if half >= PARALLEL_SIZE_CUTOFF {
        let ((tl, tl_e, tl_p), (bl, bl_e, bl_p)) = rayon::join(
            || walk(ctx, row, col, half),
            || walk(ctx, row + half, col, half),
        );
        let ((tr, tr_e, tr_p), (br, br_e, br_p)) = rayon::join(
            || walk(ctx, row, col + half, half),
            || walk(ctx, row + half, col + half, half),
        );
        [
            (tl, tl_e, tl_p),
            (bl, bl_e, bl_p),
            (tr, tr_e, tr_p),
            (br, br_e, br_p),
        ]
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
    let mut picks = Vec::new();
    for (mut q_leaves, q_evals, mut q_picks) in quadrants {
        leaves.append(&mut q_leaves);
        evals += q_evals;
        picks.append(&mut q_picks);
    }
    (leaves, evals, picks)
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
