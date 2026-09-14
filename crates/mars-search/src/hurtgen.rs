//! Hurtgen (`index_func.c::HurtgenIndexing`, `coding_func.c::HurtgenCoding`).
//!
//! **Indexing.** Domains are *not* canonicalised — each is classified as-is into
//! `class_hurtgen[m_class_1][m_class_2]`, `m_class_1` a 4-bit code from
//! [`crate::classify::hurtgen_class`] (0..16) and `m_class_2` the same
//! [`crate::classify::variance_class`] (0..24) Fisher uses.
//!
//! **Coding.** Since domains carry no canonical orientation, the range is instead tried
//! at all 8 isometries; for each, the *flipped range's own* `(m_class_1, m_class_2)`
//! selects the bucket to scan, and that same outer isometry is what gets applied to every
//! candidate found (no composition table — domains were never canonicalised).
//!
//! **The isom/flip swap.** `HurtgenCoding`'s `switch(isom)` uses `R_ROTATE90`'s flip when
//! `isom == L_ROTATE90` and vice versa, for every other case using `flips(..., isom)`
//! directly. Ported exactly (`docs/decisions.md` notes this looks like a reference bug
//! but is preserved per the Step 9 brief, since it is directly observable in which
//! candidates get considered — the RD-curve gate would catch silently "fixing" it).

use crate::classify::{hurtgen_class, variance_class, Block};
use crate::{Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock};

/// The isometry whose `flips()` result is used to *classify* under outer isometry `isom`
/// — identity for every isometry except the L/R 90-degree rotations, which are swapped.
fn flip_iso_for_classification(isom: u8) -> u8 {
    match isom {
        1 => 2, // L_ROTATE90 -> classify under R_ROTATE90's flip
        2 => 1, // R_ROTATE90 -> classify under L_ROTATE90's flip
        other => other,
    }
}

#[derive(Default)]
pub struct Hurtgen {
    // [m_class_1: 0..16][m_class_2: 0..24] -> domains.
    buckets: Vec<Vec<Vec<(u32, u32)>>>,
}

impl CandidateRetriever for Hurtgen {
    fn index(&mut self, pool: &DomainPool) {
        let mut buckets: Vec<Vec<Vec<(u32, u32)>>> = (0..16)
            .map(|_| (0..24).map(|_| Vec::new()).collect())
            .collect();
        for (dom_row, dom_col) in pool.domain_positions() {
            let block: Block = pool.domain_block(dom_row, dom_col);
            let m1 = hurtgen_class(&block) as usize;
            let m2 = variance_class(&block) as usize;
            buckets[m1][m2].push((dom_row, dom_col));
        }

        let fallback: Option<Vec<(u32, u32)>> =
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
        let mut out = Vec::new();
        for isom in 0u8..8 {
            let flip_iso = flip_iso_for_classification(isom);
            let flipped = crate::classify::flips(&block, flip_iso);
            let m1 = hurtgen_class(&flipped) as usize;
            let m2 = variance_class(&flipped) as usize;
            if let Some(bucket) = self.buckets.get(m1).and_then(|row| row.get(m2)) {
                for &(dom_row, dom_col) in bucket {
                    out.push(Candidate {
                        dom_row,
                        dom_col,
                        isometry: isom,
                    });
                }
            }
        }
        CandidateIter::new(out)
    }
}
