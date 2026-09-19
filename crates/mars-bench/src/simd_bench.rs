//! Step 11's speedup report -- reported, not gated (`just gate-11` runs the exact-equality
//! differential tests and the bitstream-unchanged check, which *are* the pass/fail bar;
//! see that step's own low verification burden). Per the `benchmark-protocol` skill: A/B
//! interleaved, N >= 5, median + MAD, machine fingerprint recorded.
//!
//! The "scalar" side of every comparison here is a literal copy of what
//! `mars_codec::encode::domain_sums`/`cross_term` computed before Step 11 rewired them
//! onto `mars-simd` -- reimplemented locally against `Contracted`'s already-public `.at`
//! rather than exposing new internals from `mars-codec`, so this file is a faithful "what
//! changed" comparison, not an approximation of one.

use std::time::{Duration, Instant};

use mars_codec::encode::{domain_sums, Contracted};
use mars_codec::isometry;
use mars_simd::moments::dot_u8_i32_window;

pub use crate::gpu_search::machine_fingerprint_line;

fn median_and_mad(mut samples: Vec<Duration>) -> (Duration, Duration) {
    samples.sort();
    let median = samples[samples.len() / 2];
    let mut dev: Vec<Duration> = samples.iter().map(|&s| s.abs_diff(median)).collect();
    dev.sort();
    (median, dev[dev.len() / 2])
}

/// Pre-Step-11 `domain_sums`, byte-for-byte: a strided double loop over `Contracted::at`.
fn scalar_domain_sums(contracted: &Contracted, dr: usize, dc: usize, size: usize) -> (i64, i64) {
    let (mut s1, mut s2) = (0i64, 0i64);
    for u in 0..size {
        for v in 0..size {
            let d = i64::from(contracted.at(dr + u, dc + v));
            s1 += d;
            s2 += d * d;
        }
    }
    (s1, s2)
}

/// Pre-Step-11 `cross_term`, byte-for-byte: `isometry::map` looked up per element rather
/// than a permuted buffer handed to a vectorised dot product.
fn scalar_cross_term(
    contracted: &Contracted,
    dr: usize,
    dc: usize,
    size: usize,
    k: u8,
    range: &[u8],
) -> i64 {
    let mut t1 = 0i64;
    for u in 0..size {
        for v in 0..size {
            let d = i64::from(contracted.at(dr + u, dc + v));
            let (i, j) = isometry::map(k, u, v, size);
            t1 += i64::from(range[i * size + j]) * d;
        }
    }
    t1
}

pub struct KernelResult {
    pub name: String,
    pub size: u32,
    pub positions: usize,
    pub scalar_median: Duration,
    pub scalar_mad: Duration,
    pub neon_median: Duration,
    pub neon_mad: Duration,
    pub speedup: f64,
}

fn legal_domain_positions(contracted: &Contracted, size: usize) -> Vec<(usize, usize)> {
    let max_dr = contracted.height().saturating_sub(size);
    let max_dc = contracted.width().saturating_sub(size);
    (0..=max_dr)
        .step_by(2)
        .flat_map(|dr| (0..=max_dc).step_by(2).map(move |dc| (dr, dc)))
        .collect()
}

/// Times `domain_sums` over every legal domain position at `size` -- the full workload
/// `search()`'s outer `while` loops run for one range block, not a single call -- so this
/// stands in for end-to-end impact without a second full encoder copy: Step 7 already
/// established this double loop (domain positions x isometries), not the floating-point
/// fit, dominates search cost.
pub fn ab_domain_sums(contracted: &Contracted, size: u32, runs: usize) -> KernelResult {
    let size_u = size as usize;
    let positions = legal_domain_positions(contracted, size_u);

    let mut scalar_times = Vec::with_capacity(runs);
    let mut neon_times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let mut acc = 0i64;
        for &(dr, dc) in &positions {
            let (s1, s2) = scalar_domain_sums(contracted, dr, dc, size_u);
            acc ^= s1 ^ s2;
        }
        scalar_times.push(t0.elapsed());
        std::hint::black_box(acc);

        let t1 = Instant::now();
        let mut acc = 0i64;
        for &(dr, dc) in &positions {
            let (s1, s2) = domain_sums(contracted, dr, dc, size_u);
            acc ^= s1 ^ s2;
        }
        neon_times.push(t1.elapsed());
        std::hint::black_box(acc);
    }

    let (scalar_median, scalar_mad) = median_and_mad(scalar_times);
    let (neon_median, neon_mad) = median_and_mad(neon_times);
    KernelResult {
        name: "domain_sums".into(),
        size,
        positions: positions.len(),
        scalar_median,
        scalar_mad,
        neon_median,
        neon_mad,
        speedup: scalar_median.as_secs_f64() / neon_median.as_secs_f64(),
    }
}

/// Times `cross_term` over every legal domain position x all 8 isometries at `size` --
/// `search()`'s complete inner loop for one range block. `range` is a synthetic but
/// realistic (non-constant) block so neither the scalar nor NEON path is measuring a
/// degenerate all-zero case.
pub fn ab_cross_term(contracted: &Contracted, size: u32, runs: usize) -> KernelResult {
    let size_u = size as usize;
    let positions = legal_domain_positions(contracted, size_u);
    let range: Vec<u8> = (0..size_u * size_u).map(|i| (i * 31 % 256) as u8).collect();
    let (plane, stride) = contracted.raw();

    let mut scalar_times = Vec::with_capacity(runs);
    let mut neon_times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let mut acc = 0i64;
        for &(dr, dc) in &positions {
            for &k in &isometry::ALL {
                acc ^= scalar_cross_term(contracted, dr, dc, size_u, k, &range);
            }
        }
        scalar_times.push(t0.elapsed());
        std::hint::black_box(acc);

        // Matches `search()`'s own structure: permute once per isometry, amortised
        // across every domain position -- not once per `(domain, isometry)` pair. The
        // first version of this benchmark permuted inside the domain loop, which is what
        // caught the regression this amortisation fixes (docs/decisions.md's Step 11
        // entry) -- kept fixed here since re-measuring the wrong code path is worse than
        // not benchmarking at all.
        let range_by_iso: Vec<Vec<u8>> = isometry::ALL
            .iter()
            .map(|&k| {
                let mut range_k = vec![0u8; size_u * size_u];
                for u in 0..size_u {
                    for v in 0..size_u {
                        let (i, j) = isometry::map(k, u, v, size_u);
                        range_k[u * size_u + v] = range[i * size_u + j];
                    }
                }
                range_k
            })
            .collect();
        let t1 = Instant::now();
        let mut acc = 0i64;
        for &(dr, dc) in &positions {
            for range_k in &range_by_iso {
                acc ^= dot_u8_i32_window(range_k, plane, stride, dr, dc, size_u);
            }
        }
        neon_times.push(t1.elapsed());
        std::hint::black_box(acc);
    }

    let (scalar_median, scalar_mad) = median_and_mad(scalar_times);
    let (neon_median, neon_mad) = median_and_mad(neon_times);
    KernelResult {
        name: "cross_term (x8 isometries)".into(),
        size,
        positions: positions.len(),
        scalar_median,
        scalar_mad,
        neon_median,
        neon_mad,
        speedup: scalar_median.as_secs_f64() / neon_median.as_secs_f64(),
    }
}
