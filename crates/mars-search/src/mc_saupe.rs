//! Mc-Saupe (`index_func.c::Mc_SaupeIndexing`, `coding_func.c::Mc_SaupeCoding`).
//!
//! Combines [`crate::masscenter::MassCenter`]'s polar grid with a per-cell k-d tree over
//! Saupe feature vectors ([`crate::saupe::Saupe`]'s feature, one tree per occupied
//! `(cx, cy)` cell instead of one shared tree).
//!
//! **Coding.** For each of the 8 range isometries (L/R swap, same pattern as the other
//! four non-Fisher-family methods), the *same* flipped range gives both the Mc angle bin
//! and the Saupe feature vector (`Mc_SaupeCoding` computes both from one `flips()` call —
//! ported that way here rather than duplicating the flip). The ring search starts at
//! `r=0, t=1` — **not** MassCenter's `r=1, t=3`; the Step 9 brief calls this out
//! explicitly as a place not to assume the two methods share a start value.

use crate::classify::compute_saupe_vector;
use crate::kdtree::{self, KdNode};
use crate::masscenter::{expanding_ring, N_P_CLASS};
use crate::saupe::{EPS, MATCHES, N_FEATURES};
use crate::{Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock};

fn flip_iso_for_classification(isom: u8) -> u8 {
    match isom {
        1 => 2,
        2 => 1,
        other => other,
    }
}

fn bin(theta: f64, n: usize) -> usize {
    let idx = (theta / std::f64::consts::TAU * n as f64) as isize;
    idx.clamp(0, n as isize - 1) as usize
}

struct Cell {
    tree: KdNode,
    points: Vec<Vec<f32>>,
    domains: Vec<(u32, u32)>,
}

#[derive(Default)]
pub struct McSaupe {
    average_factor: usize,
    cells: Vec<Vec<Option<Cell>>>, // [cx][cy]
}

impl CandidateRetriever for McSaupe {
    fn index(&mut self, pool: &DomainPool) {
        let (dim, average_factor) = crate::classify::saupe_dim_and_factor(pool.size, N_FEATURES);
        self.average_factor = average_factor;

        let mut grouped: Vec<Vec<Vec<(u32, u32)>>> = (0..N_P_CLASS)
            .map(|_| (0..N_P_CLASS).map(|_| Vec::new()).collect())
            .collect();
        for (dom_row, dom_col) in pool.domain_positions() {
            let block = pool.domain_block(dom_row, dom_col);
            let (theta0, theta1) = crate::classify::compute_mc_vectors(&block);
            grouped[bin(theta0, N_P_CLASS)][bin(theta1, N_P_CLASS)].push((dom_row, dom_col));
        }

        let mut cells: Vec<Vec<Option<Cell>>> = (0..N_P_CLASS)
            .map(|_| (0..N_P_CLASS).map(|_| None).collect())
            .collect();
        for (cx, row) in grouped.into_iter().enumerate() {
            for (cy, domains) in row.into_iter().enumerate() {
                if domains.is_empty() {
                    continue;
                }
                let points: Vec<Vec<f32>> = domains
                    .iter()
                    .map(|&(r, c)| compute_saupe_vector(&pool.domain_block(r, c), average_factor))
                    .collect();
                if let Some(tree) = kdtree::build(&points, dim) {
                    cells[cx][cy] = Some(Cell {
                        tree,
                        points,
                        domains,
                    });
                }
            }
        }
        self.cells = cells;
    }

    fn candidates(&self, range: &RangeBlock) -> CandidateIter<'_> {
        let mut out = Vec::new();
        let block = range.as_block();
        for isom in 0u8..8 {
            let flip_iso = flip_iso_for_classification(isom);
            let flipped = crate::classify::flips(&block, flip_iso);
            let (theta0, theta1) = crate::classify::compute_mc_vectors(&flipped);
            let query = compute_saupe_vector(&flipped, self.average_factor);
            let cx = bin(theta0, N_P_CLASS);
            let cy = bin(theta1, N_P_CLASS);

            let ring = expanding_ring(cx, cy, N_P_CLASS, 0, 1, |a, b| self.cells[a][b].is_some());
            for (a, b) in ring {
                if let Some(cell) = &self.cells[a][b] {
                    let found = kdtree::search(&query, &cell.points, &cell.tree, EPS, MATCHES);
                    for idx in found {
                        let (dom_row, dom_col) = cell.domains[idx];
                        out.push(Candidate {
                            dom_row,
                            dom_col,
                            isometry: isom,
                        });
                    }
                }
            }
        }
        CandidateIter::new(out)
    }
}
