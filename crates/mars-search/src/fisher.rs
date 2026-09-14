//! Fisher (`reference/mars1/index_func.c::FisherIndexing`,
//! `coding_func.c::FisherCoding`).
//!
//! **Indexing.** Every domain is canonicalised (`newclass`) into one of 3 shape classes
//! `clas`, sub-classed by its (now-canonical) quadrant-variance order `var_class` (one of
//! 24) — `class_fisher[clas][var_class]`, a bucket of `(dom_row, dom_col, dom_iso)` where
//! `dom_iso` is the isometry that canonicalised *this* domain (needed to compose with the
//! range's own canonicalising isometry at coding time).
//!
//! **Coding.** The range block is canonicalised the same way, giving its own
//! `(isom, clas)` and (from the canonicalised range) `var_class`; only bucket
//! `[clas][var_class]` is scanned. The isometry actually applied to a candidate is
//! `MAPPING[isom][dom_iso]` (`globals.h`'s composition table).
//!
//! **Simplification (`docs/decisions.md`).** Mars 1's `-full1`/`-full2` flags widen this
//! to a full 3x24 bucket scan; this port implements only the restricted one-bucket
//! search, which is the method's whole point per the Step 9 brief.

use crate::classify::{newclass, variance_class, Block};
use crate::tables::MAPPING;
use crate::{Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock};

#[derive(Default)]
pub struct Fisher {
    // [clas][var_class] -> domains, each (dom_row, dom_col, dom_iso).
    buckets: Vec<Vec<Vec<(u32, u32, u8)>>>,
}

impl CandidateRetriever for Fisher {
    fn index(&mut self, pool: &DomainPool) {
        let mut buckets: Vec<Vec<Vec<(u32, u32, u8)>>> = (0..3)
            .map(|_| (0..24).map(|_| Vec::new()).collect())
            .collect();
        for (dom_row, dom_col) in pool.domain_positions() {
            let block: Block = pool.domain_block(dom_row, dom_col);
            let (iso, clas) = newclass(&block);
            let flipped = crate::classify::flips(&block, iso);
            let var_class = variance_class(&flipped);
            buckets[clas as usize][var_class as usize].push((dom_row, dom_col, iso));
        }

        // "Make sure no class is empty" (`FisherIndexing`): every empty bucket is
        // replaced by the first non-empty one, in `(clas, var_class)` row-major order.
        let fallback: Option<Vec<(u32, u32, u8)>> =
            buckets.iter().flatten().find(|b| !b.is_empty()).cloned();
        if let Some(fb) = fallback {
            for row in &mut buckets {
                for b in row.iter_mut() {
                    if b.is_empty() {
                        *b = fb.clone();
                    }
                }
            }
        }
        self.buckets = buckets;
    }

    fn candidates(&self, range: &RangeBlock) -> CandidateIter<'_> {
        let block = range.as_block();
        let (isom, clas) = newclass(&block);
        let flipped = crate::classify::flips(&block, isom);
        let var_class = variance_class(&flipped);

        let mut out = Vec::new();
        if let Some(row) = self.buckets.get(clas as usize) {
            if let Some(bucket) = row.get(var_class as usize) {
                for &(dom_row, dom_col, dom_iso) in bucket {
                    let isometry = MAPPING[isom as usize][dom_iso as usize];
                    out.push(Candidate {
                        dom_row,
                        dom_col,
                        isometry,
                    });
                }
            }
        }
        CandidateIter::new(out)
    }
}
