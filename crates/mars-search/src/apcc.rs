//! Train-free v1 absolute-Pearson-correlation-coefficient (APCC) retrieval.
//!
//! Fisher canonicalization selects one `(clas, variance_class)` bucket, with no
//! empty-bucket or exhaustive fallback. Within it, domains sort by increasing
//! absolute Pearson correlation to a fixed synthetic reference, then `(row, col)`.
//! This reference is not offline-trained and implies no paper-faithful speedup.
//!
//! For size `n >= 2`, let `h = n/2`. The reference at `(r, c)` is
//! `4 * ranks[quadrant] + ((r % h) * h + c % h) / h^2`, with quadrant ranks
//! `[3,2,1,0]`, `[3,2,0,1]`, `[3,1,0,2]` for classes 0, 1, 2 respectively.
//! These are the three canonical quadrant orders, with a fixed intra-quadrant ramp.
//!
//! Queries use the lower bound (first key >= query key). With
//! `k = min(budget, bucket_len)`, take `floor(k/2)` positions before that insertion
//! point and `k - floor(k/2)` starting at it, wrapping modulo the bucket length.
//! Finally restore bucket order. Thus budgets are nested, positions are unique,
//! and a full budget scans the entire matching bucket in sorted order.
//! Each position emits the Fisher-composed isometry first, then the remaining
//! isometries in numeric order. The shared fitter evaluates all eight, retaining
//! real moments and the first minimum-RMS fit; absolute correlation never supplies
//! a contrast sign to the codec's nonnegative quantized affine model.

use mars_codec::encode::{
    Contracted, EncodeOptions, EncodeParams, SearchOutcome, SearchProvider, SearchRequest,
};
use mars_core::Plane;

use crate::classify::{flips, newclass, variance_class, Block};
use crate::tables::MAPPING;
use crate::{
    search_block_fitted, Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock,
};

/// Deterministic domain-position budget; each inspected position costs eight fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApccConfig {
    /// Maximum positions per query. Zero returns no candidates (codec-owned DC).
    pub budget: usize,
}

/// Absolute Pearson key: `|sum((x-mean(x))*(y-mean(y)))| /
/// (sqrt(sum((x-mean(x))^2)) * sqrt(sum((y-mean(y))^2)))`, clamped to `[0, 1]`
/// for rounding error. Both indexing and querying use this same helper.
///
/// Returns `None` for mismatched/empty blocks, zero variance in either block, or
/// non-finite inputs/intermediates. Undefined correlation is never a sortable key.
/// Positive scaling (including the contracted plane's factor of four) and offsets
/// do not change the mathematical key; negating either block does not change it.
pub fn apcc_key(block: &Block, reference: &Block) -> Option<f64> {
    if block.size != reference.size
        || block.size.checked_mul(block.size) != Some(block.data.len())
        || block.data.len() != reference.data.len()
        || block.data.is_empty()
    {
        return None;
    }
    let n = block.data.len() as f64;
    let mean_x = block.data.iter().sum::<f64>() / n;
    let mean_y = reference.data.iter().sum::<f64>() / n;
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    for (&x, &y) in block.data.iter().zip(&reference.data) {
        let (x, y) = (x - mean_x, y - mean_y);
        xy += x * y;
        xx += x * x;
        yy += y * y;
    }
    let denominator = xx.sqrt() * yy.sqrt();
    if !xy.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let key = xy.abs() / denominator;
    key.is_finite().then(|| key.clamp(0.0, 1.0))
}

fn reference(size: usize, clas: usize) -> Block {
    let ranks = [[3, 2, 1, 0], [3, 2, 0, 1], [3, 1, 0, 2]][clas];
    let h = size / 2;
    Block::new(
        size,
        (0..size * size)
            .map(|i| {
                let (r, c) = (i / size, i % size);
                let quadrant = 2 * (r / h) + c / h;
                4.0 * f64::from(ranks[quadrant]) + ((r % h) * h + c % h) as f64 / (h * h) as f64
            })
            .collect(),
    )
}

struct Domain {
    key: f64,
    row: u32,
    col: u32,
    iso: u8,
}

/// One owned APCC size/stride index implementing [`CandidateRetriever`].
/// Build via [`Self::new`] then [`CandidateRetriever::index`]; queries must match
/// the indexed size. No image borrow, query-order state, or random seed is retained.
pub struct Apcc {
    config: ApccConfig,
    size: u32,
    references: Vec<Block>,
    buckets: Vec<Vec<Domain>>,
}

impl Apcc {
    /// Create an unindexed retriever. Until indexed, queries return no candidates.
    pub fn new(config: ApccConfig) -> Self {
        Self {
            config,
            size: 0,
            references: Vec::new(),
            buckets: Vec::new(),
        }
    }
}

impl CandidateRetriever for Apcc {
    fn index(&mut self, pool: &DomainPool) {
        assert!(pool.size >= 2 && pool.size.is_power_of_two());
        self.size = pool.size;
        self.references = (0..3)
            .map(|clas| reference(pool.size as usize, clas))
            .collect();
        self.buckets = (0..72).map(|_| Vec::new()).collect();
        for (row, col) in pool.domain_positions() {
            let block = pool.domain_block(row, col);
            let (iso, clas) = newclass(&block);
            let canonical = flips(&block, iso);
            if let Some(key) = apcc_key(&canonical, &self.references[clas as usize]) {
                let var_class = variance_class(&canonical);
                self.buckets[clas as usize * 24 + var_class as usize].push(Domain {
                    key,
                    row,
                    col,
                    iso,
                });
            }
        }
        for bucket in &mut self.buckets {
            bucket.sort_unstable_by(|a, b| {
                a.key
                    .total_cmp(&b.key)
                    .then_with(|| (a.row, a.col).cmp(&(b.row, b.col)))
            });
        }
    }

    fn candidates(&self, range: &RangeBlock) -> CandidateIter<'_> {
        let mut out = Vec::new();
        if self.size == 0 || self.config.budget == 0 {
            return CandidateIter::new(out);
        }
        assert_eq!(range.size, self.size, "APCC range size differs from index");
        let block = range.as_block();
        let (isom, clas) = newclass(&block);
        let canonical = flips(&block, isom);
        let Some(key) = apcc_key(&canonical, &self.references[clas as usize]) else {
            return CandidateIter::new(out);
        };
        let bucket = &self.buckets[clas as usize * 24 + variance_class(&canonical) as usize];
        let len = bucket.len();
        let count = self.config.budget.min(len);
        if count == 0 {
            return CandidateIter::new(out);
        }
        let selected = if count == len {
            (0..len).collect::<Vec<_>>()
        } else {
            let insertion = bucket.partition_point(|domain| domain.key < key) % len;
            let before = count / 2;
            let start = if insertion >= before {
                insertion - before
            } else {
                len - (before - insertion)
            };
            let mut indices = (start..len).chain(0..start).take(count).collect::<Vec<_>>();
            indices.sort_unstable();
            indices
        };
        for index in selected {
            let domain = &bucket[index];
            let first = MAPPING[isom as usize][domain.iso as usize];
            for isometry in std::iter::once(first).chain((0..8).filter(|&iso| iso != first)) {
                out.push(Candidate {
                    dom_row: domain.row,
                    dom_col: domain.col,
                    isometry,
                });
            }
        }
        CandidateIter::new(out)
    }
}

struct Pools {
    shift: u32,
    by_tip: Vec<Option<Apcc>>,
}

/// Owned per-size/stride APCC indexes for the grayscale production encoder.
/// Indexing completes before queries; concurrent [`SearchProvider`] calls only read.
/// Partitioning, DC fallback, density decisions, modes, and residuals stay codec-owned.
pub struct ApccSearchProvider {
    base: Pools,
    doubled: Option<Pools>,
}

impl ApccSearchProvider {
    /// Index every power-of-two size from 2 through `max_size`, including sizes
    /// below `min_size` needed at image borders. Also index doubled stride exactly
    /// when `lambda.is_some() && options.adaptive_density`.
    ///
    /// Panics unless sizes are positive ordered powers of two and shift is positive.
    /// Requests must use the same image/configuration and a prepared size/stride;
    /// size 1 remains codec-owned DC. Neither input image nor contraction is borrowed.
    pub fn build(
        image: &Plane,
        contracted: &Contracted,
        params: &EncodeParams,
        options: &EncodeOptions,
        config: ApccConfig,
    ) -> Self {
        assert!(params.min_size.is_power_of_two() && params.max_size.is_power_of_two());
        assert!(params.min_size <= params.max_size && params.shift > 0);
        let build = |shift| {
            let max_tip = params.max_size.trailing_zeros();
            let mut by_tip = Vec::with_capacity(max_tip as usize + 1);
            by_tip.push(None);
            for tip in 1..=max_tip {
                let mut index = Apcc::new(config);
                index.index(&DomainPool {
                    contracted,
                    size: 1 << tip,
                    shift,
                    image_width: image.width() as u32,
                    image_height: image.height() as u32,
                });
                by_tip.push(Some(index));
            }
            Pools { shift, by_tip }
        };
        Self {
            base: build(params.shift),
            doubled: (params.lambda.is_some() && options.adaptive_density)
                .then(|| build(params.shift.saturating_mul(2))),
        }
    }
}

impl SearchProvider for ApccSearchProvider {
    fn search(&self, request: &SearchRequest<'_>) -> SearchOutcome {
        let pools = if request.shift == self.base.shift {
            &self.base
        } else {
            self.doubled
                .as_ref()
                .filter(|p| p.shift == request.shift)
                .expect("APCC search stride was not indexed")
        };
        assert!(request.size >= 2 && request.size.is_power_of_two());
        let index = pools
            .by_tip
            .get(request.size.trailing_zeros() as usize)
            .and_then(Option::as_ref)
            .expect("APCC search size was not indexed");
        let size = request.size as usize;
        let mut pixels = Vec::with_capacity(size * size);
        for row in request.row as usize..request.row as usize + size {
            let start = row * request.image.width() + request.col as usize;
            pixels.extend_from_slice(&request.image.as_slice()[start..start + size]);
        }
        let range = RangeBlock {
            row: request.row,
            col: request.col,
            size: request.size,
            pixels: &pixels,
        };
        search_block_fitted(
            index,
            request.contracted,
            &range,
            request.params.max_alfa,
            request.params.bits_alfa,
            request.params.bits_beta,
        )
    }
}
