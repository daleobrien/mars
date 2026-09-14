//! `Exhaustive`: every legal domain position x every isometry — the trivial
//! `CandidateRetriever`, and the sanity baseline every real method's recall is measured
//! against. Not a speed-up method; `mars_codec::encode`'s Step 6/7 search already
//! implements this exact search directly (for performance), so this wrapper exists only
//! to give the trait a reference point and a differential test
//! (`exhaustive_matches_mars_codec_search_block_exactly`, `src/lib.rs`).

use crate::{Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock};

#[derive(Default)]
pub struct Exhaustive {
    domains: Vec<(u32, u32)>,
}

impl CandidateRetriever for Exhaustive {
    fn index(&mut self, pool: &DomainPool) {
        self.domains = pool.domain_positions().collect();
    }

    fn candidates(&self, _range: &RangeBlock) -> CandidateIter<'_> {
        let mut out = Vec::with_capacity(self.domains.len() * 8);
        for &(dom_row, dom_col) in &self.domains {
            for isometry in 0u8..8 {
                out.push(Candidate {
                    dom_row,
                    dom_col,
                    isometry,
                });
            }
        }
        CandidateIter::new(out)
    }
}
