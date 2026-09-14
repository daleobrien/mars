//! Saupe-Fisher (`index_func.c::Saupe_FisherIndexing`,
//! `coding_func.c::Saupe_FisherCoding`).
//!
//! Like [`crate::saupe::Saupe`], but domains are canonicalised (`newclass` + `flips`,
//! exactly like [`crate::fisher::Fisher`]) *before* their feature vector is computed and
//! indexed into one shared k-d tree per size — so, unlike plain Saupe, the range is
//! canonicalised and queried **once** at coding time rather than looped over all 8
//! isometries. The isometry actually applied to a candidate is
//! `MAPPING[isom][dom_iso]`, the same composition-table pattern as Fisher.

use crate::classify::{flips, newclass};
use crate::kdtree::{self, KdNode};
use crate::saupe::{EPS, MATCHES, N_FEATURES};
use crate::tables::MAPPING;
use crate::{Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock};

#[derive(Default)]
pub struct SaupeFisher {
    average_factor: usize,
    points: Vec<Vec<f32>>,
    domains: Vec<(u32, u32, u8)>, // (dom_row, dom_col, dom_iso)
    tree: Option<KdNode>,
}

impl CandidateRetriever for SaupeFisher {
    fn index(&mut self, pool: &DomainPool) {
        let (dim, average_factor) = crate::classify::saupe_dim_and_factor(pool.size, N_FEATURES);
        self.average_factor = average_factor;

        let mut points = Vec::new();
        let mut domains = Vec::new();
        for (dom_row, dom_col) in pool.domain_positions() {
            let block = pool.domain_block(dom_row, dom_col);
            let (iso, _clas) = newclass(&block);
            let flipped = flips(&block, iso);
            points.push(crate::classify::compute_saupe_vector(
                &flipped,
                average_factor,
            ));
            domains.push((dom_row, dom_col, iso));
        }
        self.tree = kdtree::build(&points, dim);
        self.points = points;
        self.domains = domains;
    }

    fn candidates(&self, range: &RangeBlock) -> CandidateIter<'_> {
        let mut out = Vec::new();
        let Some(tree) = &self.tree else {
            return CandidateIter::new(out);
        };
        let block = range.as_block();
        let (isom, _clas) = newclass(&block);
        let flipped = flips(&block, isom);
        let query = crate::classify::compute_saupe_vector(&flipped, self.average_factor);
        let found = kdtree::search(&query, &self.points, tree, EPS, MATCHES);
        for idx in found {
            let (dom_row, dom_col, dom_iso) = self.domains[idx];
            let isometry = MAPPING[isom as usize][dom_iso as usize];
            out.push(Candidate {
                dom_row,
                dom_col,
                isometry,
            });
        }
        CandidateIter::new(out)
    }
}
