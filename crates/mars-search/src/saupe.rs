//! Saupe (`index_func.c::SaupeIndexing`, `coding_func.c::SaupeCoding`).
//!
//! **Indexing.** Every domain's Saupe feature vector
//! ([`crate::classify::compute_saupe_vector`], with the size-dependent `ShrunkBlock`
//! factor from [`crate::classify::saupe_dim_and_factor`]) goes into one k-d tree per
//! size ([`crate::kdtree`]).
//!
//! **Coding.** For each of the 8 range isometries (L/R 90-degree flip swapped, same
//! pattern as [`crate::hurtgen`]/[`crate::masscenter`]), compute the flipped range's own
//! feature vector and run an approximate k-NN search (`eps = 2.0`, up to `matches = 50`
//! neighbours — `globals.h`'s defaults) against the shared tree.

use crate::kdtree::{self, KdNode};
use crate::{Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock};

/// `globals.h` defaults.
pub const EPS: f32 = 2.0;
pub const MATCHES: usize = 50;
pub const N_FEATURES: u32 = 16;

fn flip_iso_for_classification(isom: u8) -> u8 {
    match isom {
        1 => 2,
        2 => 1,
        other => other,
    }
}

#[derive(Default)]
pub struct Saupe {
    dim: usize,
    average_factor: usize,
    points: Vec<Vec<f32>>,
    domains: Vec<(u32, u32)>,
    tree: Option<KdNode>,
}

impl CandidateRetriever for Saupe {
    fn index(&mut self, pool: &DomainPool) {
        let (dim, average_factor) = crate::classify::saupe_dim_and_factor(pool.size, N_FEATURES);
        self.dim = dim;
        self.average_factor = average_factor;

        let mut points = Vec::new();
        let mut domains = Vec::new();
        for (dom_row, dom_col) in pool.domain_positions() {
            let block = pool.domain_block(dom_row, dom_col);
            points.push(crate::classify::compute_saupe_vector(
                &block,
                average_factor,
            ));
            domains.push((dom_row, dom_col));
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
        for isom in 0u8..8 {
            let flip_iso = flip_iso_for_classification(isom);
            let flipped = crate::classify::flips(&block, flip_iso);
            let query = crate::classify::compute_saupe_vector(&flipped, self.average_factor);
            let found = kdtree::search(&query, &self.points, tree, EPS, MATCHES);
            for idx in found {
                let (dom_row, dom_col) = self.domains[idx];
                out.push(Candidate {
                    dom_row,
                    dom_col,
                    isometry: isom,
                });
            }
        }
        CandidateIter::new(out)
    }
}
