//! Versioned, query-local random domain sampling for grayscale production encoding.
//!
//! Sampling uses sparse partial Fisher–Yates with unbiased rejection draws, followed
//! by reference-order sorting. Budgets select prefixes of the same permutation, so
//! sampled sets are nested. No shared RNG, exhaustive fallback, or early stopping.

use std::collections::HashMap;

use mars_codec::encode::{
    Contracted, EncodeOptions, EncodeParams, SearchOutcome, SearchProvider, SearchRequest,
};
use mars_core::Plane;

use crate::{
    search_block_fitted, Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock,
};

/// Algorithm identity: FNV-1a framing, SplitMix64, rejection draws, and sparse forward
/// Fisher–Yates. Changing any seed framing or sampling rule requires a new version.
pub const RNG_VERSION: &str = "mars-random-gray-fnv1a64-splitmix64-fy-v1";

/// Reproducible position budget (each position evaluates all eight isometries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RandomConfig {
    /// Maximum unique domain positions per query. Zero explicitly returns no fit;
    /// production CLI callers should reject zero rather than silently encode DC only.
    pub budget: usize,
    /// User seed; independent of query scheduling and thread count.
    pub seed: u64,
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Stable, non-cryptographic grayscale identity: FNV-1a over `RNG_VERSION`, a zero
/// delimiter, `gray\0`, width/height as little-endian u64, then row-major pixel bytes.
/// RGB/Y/Cb/Cr identities are deliberately not supported by this version.
pub fn image_fingerprint(image: &Plane) -> u64 {
    let mut hash = hash_bytes(FNV_OFFSET, RNG_VERSION.as_bytes());
    hash = hash_bytes(hash, b"\0gray\0");
    hash = hash_bytes(hash, &(image.width() as u64).to_le_bytes());
    hash = hash_bytes(hash, &(image.height() as u64).to_le_bytes());
    hash_bytes(hash, image.as_slice())
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold {
                return value % bound;
            }
        }
    }
}

struct Pools {
    shift: u32,
    by_tip: Vec<Vec<(u32, u32)>>,
}

/// Owned legal-position indexes and immutable query seed material. Build once per
/// grayscale image, then share via `SearchProvider: Sync`. No image borrow is retained.
pub struct RandomSearchProvider {
    config: RandomConfig,
    fingerprint: u64,
    base: Pools,
    doubled: Option<Pools>,
}

impl RandomSearchProvider {
    /// Index all power-of-two sizes from 2 through `max_size`, including boundary
    /// sizes below `min_size`. Adaptive RD density also indexes the doubled stride.
    /// Parameters must have positive, ordered power-of-two sizes and a positive shift.
    /// Requests must use the same image/configuration; size 1 remains codec-owned DC.
    pub fn build(
        image: &Plane,
        contracted: &Contracted,
        params: &EncodeParams,
        options: &EncodeOptions,
        config: RandomConfig,
    ) -> Self {
        assert!(params.min_size.is_power_of_two() && params.max_size.is_power_of_two());
        assert!(params.min_size <= params.max_size && params.shift > 0);
        let build = |shift| {
            let max_tip = params.max_size.trailing_zeros();
            let mut by_tip = vec![Vec::new(); max_tip as usize + 1];
            for tip in 1..=max_tip {
                let pool = DomainPool {
                    contracted,
                    size: 1 << tip,
                    shift,
                    image_width: image.width() as u32,
                    image_height: image.height() as u32,
                };
                by_tip[tip as usize] = pool.domain_positions().collect();
            }
            Pools { shift, by_tip }
        };
        Self {
            config,
            fingerprint: image_fingerprint(image),
            base: build(params.shift),
            doubled: (params.lambda.is_some() && options.adaptive_density)
                .then(|| build(params.shift.saturating_mul(2))),
        }
    }

    /// Query seed is FNV-1a over `query\0`, image fingerprint and user seed (LE u64),
    /// then row, column, size, effective stride (LE u32). Budget is intentionally absent.
    pub fn query_seed(&self, row: u32, col: u32, size: u32, shift: u32) -> u64 {
        let mut hash = hash_bytes(FNV_OFFSET, b"query\0");
        hash = hash_bytes(hash, &self.fingerprint.to_le_bytes());
        hash = hash_bytes(hash, &self.config.seed.to_le_bytes());
        for value in [row, col, size, shift] {
            hash = hash_bytes(hash, &value.to_le_bytes());
        }
        hash
    }

    /// Unique sampled domain positions in row-major reference order. Full budget
    /// returns the whole pool without consuming RNG. Selection takes expected O(K)
    /// time/space, plus O(K log K) sorting, where K = min(budget, pool length).
    /// Hash-map iteration order is never consulted; the map only stores sparse swaps.
    /// Panics for an unprepared size/stride (a caller configuration mismatch).
    pub fn sampled_positions(&self, row: u32, col: u32, size: u32, shift: u32) -> Vec<(u32, u32)> {
        let pools = if shift == self.base.shift {
            &self.base
        } else {
            self.doubled
                .as_ref()
                .filter(|p| p.shift == shift)
                .expect("random search stride was not indexed")
        };
        assert!(
            size >= 2 && size.is_power_of_two(),
            "random search size must be a power of two >= 2"
        );
        let positions = pools
            .by_tip
            .get(size.trailing_zeros() as usize)
            .expect("random search size was not indexed");
        let count = self.config.budget.min(positions.len());
        if count == positions.len() {
            return positions.clone();
        }
        let mut rng = SplitMix64(self.query_seed(row, col, size, shift));
        let mut swaps = HashMap::with_capacity(count);
        let mut selected = Vec::with_capacity(count);
        for i in 0..count {
            let j = i + rng.below((positions.len() - i) as u64) as usize;
            let at_j = swaps.remove(&j).unwrap_or(j);
            let at_i = swaps.remove(&i).unwrap_or(i);
            if j != i {
                swaps.insert(j, at_i);
            }
            selected.push(at_j);
        }
        selected.sort_unstable();
        selected.into_iter().map(|index| positions[index]).collect()
    }
}

struct SampledDomains(Vec<(u32, u32)>);

impl CandidateRetriever for SampledDomains {
    fn index(&mut self, pool: &DomainPool) {
        self.0 = pool.domain_positions().collect();
    }

    fn candidates(&self, _range: &RangeBlock) -> CandidateIter<'_> {
        CandidateIter::new(
            self.0
                .iter()
                .flat_map(|&(dom_row, dom_col)| {
                    (0..8).map(move |isometry| Candidate {
                        dom_row,
                        dom_col,
                        isometry,
                    })
                })
                .collect(),
        )
    }
}

impl SearchProvider for RandomSearchProvider {
    fn search(&self, request: &SearchRequest<'_>) -> SearchOutcome {
        let domains = SampledDomains(self.sampled_positions(
            request.row,
            request.col,
            request.size,
            request.shift,
        ));
        if domains.0.is_empty() {
            return SearchOutcome {
                candidate: None,
                evals: 0,
            };
        }
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
            &domains,
            request.contracted,
            &range,
            request.params.max_alfa,
            request.params.bits_alfa,
            request.params.bits_beta,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn splitmix64_reference_stream_and_rejection() {
        let mut rng = SplitMix64(0);
        assert_eq!(rng.next(), 0xe220a8397b1dcdaf);
        assert_eq!(rng.next(), 0x6e789e6aa1b965f4);
        assert_eq!(rng.next(), 0x06c45d188009454f);
        // Large bound forces a rejection for the second draw of the seed-zero stream.
        let mut rng = SplitMix64(0);
        let bound = (1u64 << 63) + 1;
        assert_eq!(rng.below(bound), 0x6220a8397b1dcdae);
        let mut reference = SplitMix64(0);
        reference.next();
        let expected = loop {
            let n = reference.next();
            if n >= bound.wrapping_neg() % bound {
                break n % bound;
            }
        };
        assert_eq!(rng.below(bound), expected);
        assert_eq!(rng.next(), reference.next());
    }
}
