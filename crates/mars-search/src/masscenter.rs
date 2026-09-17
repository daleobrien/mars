//! MassCenter (`index_func.c::MassCenterIndexing`, `coding_func.c::MassCenterCoding`).
//!
//! **Indexing.** Every domain's two centre-of-mass polar angles
//! ([`crate::classify::compute_mc_vectors`]) are binned into an `n_p_class x n_p_class`
//! grid `class_polar[cx][cy]`.
//!
//! **Coding.** For each of the 8 range isometries (L/R 90-degree flip swapped, matching
//! [`crate::hurtgen`]'s pattern — `MassCenterCoding`'s own `switch` has the identical
//! swap), compute the range's own `(cx, cy)` cell, then expand a ring outward from it
//! (starting at `r=1, t=3` — a 3x3 box) with modular wraparound until a non-empty cell is
//! found, and scan every domain in every cell of that *final* ring (not just the first
//! non-empty cell).

use crate::{Candidate, CandidateIter, CandidateRetriever, DomainPool, RangeBlock};

/// `globals.h`: `EXTERN int n_p_class INIT(= 50)`.
pub const N_P_CLASS: usize = 50;

fn bin(theta: f64, n: usize) -> usize {
    let idx = (theta / std::f64::consts::TAU * n as f64) as isize;
    idx.clamp(0, n as isize - 1) as usize
}

/// Isometry whose `flips()` is used for classification purposes under outer isometry
/// `isom` — same L/R-rotate90 swap as [`crate::hurtgen::flip_iso_for_classification`].
fn flip_iso_for_classification(isom: u8) -> u8 {
    match isom {
        1 => 2,
        2 => 1,
        other => other,
    }
}

/// Ring cells for the expanding search: keeps widening `(r, t)` — starting from the
/// caller's own `(r0, t0)` — until at least one `(a, b)` in the `t x t` ring satisfies
/// `nonempty`, then returns every `(a, b)` in that final ring. Returns an empty vector
/// if a full-grid probe finds no occupied cell. Mirrors
/// `MassCenterCoding`'s / `Mc_SaupeCoding`'s identical two-pass (probe, then gather) loop
/// structure, parameterised over each method's own starting `(r0, t0)` (§ Step 9 brief:
/// "port each method's own start values exactly, don't assume they match").
pub fn expanding_ring(
    cx: usize,
    cy: usize,
    n_p_class: usize,
    r0: i64,
    t0: i64,
    mut nonempty: impl FnMut(usize, usize) -> bool,
) -> Vec<(usize, usize)> {
    if n_p_class == 0 {
        return Vec::new();
    }
    let n = n_p_class as i64;
    let mut r = r0;
    let mut t = t0;
    loop {
        let mut found = false;
        'outer: for aa in 0..t {
            let a = (cx as i64 + n - r + aa).rem_euclid(n) as usize;
            for bb in 0..t {
                let b = (cy as i64 + n - r + bb).rem_euclid(n) as usize;
                if nonempty(a, b) {
                    found = true;
                    break 'outer;
                }
            }
        }
        if found {
            break;
        }
        // Once the wrapped window spans every bin, no wider search can find a domain.
        if t >= n {
            return Vec::new();
        }
        r += 1;
        t += 1;
    }
    let mut cells = Vec::with_capacity((t * t) as usize);
    for aa in 0..t {
        let a = (cx as i64 + n - r + aa).rem_euclid(n) as usize;
        for bb in 0..t {
            let b = (cy as i64 + n - r + bb).rem_euclid(n) as usize;
            cells.push((a, b));
        }
    }
    cells
}

#[derive(Default)]
pub struct MassCenter {
    buckets: Vec<Vec<Vec<(u32, u32)>>>, // [cx][cy] -> domains
}

impl CandidateRetriever for MassCenter {
    fn index(&mut self, pool: &DomainPool) {
        let mut buckets: Vec<Vec<Vec<(u32, u32)>>> = (0..N_P_CLASS)
            .map(|_| (0..N_P_CLASS).map(|_| Vec::new()).collect())
            .collect();
        for (dom_row, dom_col) in pool.domain_positions() {
            let block = pool.domain_block(dom_row, dom_col);
            let (theta0, theta1) = crate::classify::compute_mc_vectors(&block);
            let cx = bin(theta0, N_P_CLASS);
            let cy = bin(theta1, N_P_CLASS);
            buckets[cx][cy].push((dom_row, dom_col));
        }
        self.buckets = buckets;
    }

    fn candidates(&self, range: &RangeBlock) -> CandidateIter<'_> {
        let block = range.as_block();
        let mut out = Vec::new();
        for isom in 0u8..8 {
            let flip_iso = flip_iso_for_classification(isom);
            let flipped = crate::classify::flips(&block, flip_iso);
            let (theta0, theta1) = crate::classify::compute_mc_vectors(&flipped);
            let cx = bin(theta0, N_P_CLASS);
            let cy = bin(theta1, N_P_CLASS);

            let cells = expanding_ring(cx, cy, N_P_CLASS, 1, 3, |a, b| {
                !self.buckets[a][b].is_empty()
            });
            for (a, b) in cells {
                for &(dom_row, dom_col) in &self.buckets[a][b] {
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
