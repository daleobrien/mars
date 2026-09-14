//! A direct (non-fast) 2D DCT-II / DCT-III pair for Step 15's residual mode.
//!
//! The brief is explicit that Step 15 should "start here rather than with a wavelet -- it
//! is simpler". A direct O(N^3) per-block separable transform (two O(N^2) 1D passes) is
//! simpler still than a fast DCT, and every leaf this project codes is small (`size <= 16`
//! in every config `implementation-plan.md` names), so the O(N^3) cost is negligible next
//! to the fractal search that dominates encode time -- there is no speed reason to reach
//! for a fast transform here, and a direct summation is far easier to keep deterministic
//! (no butterfly-stage reordering to get bit-identical across thread counts) and to verify
//! (the orthonormality/round-trip property below is a strong, cheap, symbolic check).
//!
//! `f64` throughout: the residual path is float by nature (unlike the affine-fit moments,
//! which are integer-exact by construction) since it quantises real-valued transform
//! coefficients, not raw pixel sums. Determinism still holds: no wall-clock, no unordered
//! reduction, plain nested loops over a fixed axis order.

use std::f64::consts::PI;

/// The orthonormal DCT-II basis value for frequency `k`, sample `n`, block size `n_size`:
/// `sqrt(2/N) * C(k) * cos(pi/N * (n + 0.5) * k)`, with `C(0) = 1/sqrt(2)` and `C(k) = 1`
/// otherwise -- the normalisation that makes DCT-II and DCT-III exact inverses of one
/// another (an orthonormal transform, not just "a" DCT convention).
fn basis(n_size: usize, k: usize, n: usize) -> f64 {
    let c_k = if k == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
    (2.0 / n_size as f64).sqrt() * c_k * (PI / n_size as f64 * (n as f64 + 0.5) * k as f64).cos()
}

/// 1D orthonormal DCT-II of `x` (length `n`).
fn dct1d(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|k| (0..n).map(|i| x[i] * basis(n, k, i)).sum())
        .collect()
}

/// 1D orthonormal DCT-III (the exact inverse of [`dct1d`]) of `coeffs` (length `n`).
fn idct1d(coeffs: &[f64]) -> Vec<f64> {
    let n = coeffs.len();
    (0..n)
        .map(|i| (0..n).map(|k| coeffs[k] * basis(n, k, i)).sum())
        .collect()
}

/// Forward 2D DCT-II of a `size x size` row-major block: rows then columns, each an
/// independent 1D orthonormal transform -- separability is what makes the 2D orthonormal
/// DCT a plain composition of the 1D one along each axis.
pub fn forward_dct2d(block: &[f64], size: usize) -> Vec<f64> {
    assert_eq!(block.len(), size * size, "block must be size x size");
    let mut tmp = vec![0.0; size * size];
    for r in 0..size {
        let row = &block[r * size..(r + 1) * size];
        let t = dct1d(row);
        tmp[r * size..(r + 1) * size].copy_from_slice(&t);
    }
    let mut out = vec![0.0; size * size];
    let mut col = vec![0.0; size];
    for c in 0..size {
        for r in 0..size {
            col[r] = tmp[r * size + c];
        }
        let t = dct1d(&col);
        for r in 0..size {
            out[r * size + c] = t[r];
        }
    }
    out
}

/// Inverse of [`forward_dct2d`]: columns then rows (either axis order round-trips exactly
/// since the two 1D passes commute for a separable transform; this order is simply the
/// mirror of the forward pass).
pub fn inverse_dct2d(coeffs: &[f64], size: usize) -> Vec<f64> {
    assert_eq!(coeffs.len(), size * size, "coeffs must be size x size");
    let mut tmp = vec![0.0; size * size];
    let mut col = vec![0.0; size];
    for c in 0..size {
        for r in 0..size {
            col[r] = coeffs[r * size + c];
        }
        let t = idct1d(&col);
        for r in 0..size {
            tmp[r * size + c] = t[r];
        }
    }
    let mut out = vec![0.0; size * size];
    for r in 0..size {
        let row = &tmp[r * size..(r + 1) * size];
        let t = idct1d(row);
        out[r * size..(r + 1) * size].copy_from_slice(&t);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// The exact-inverse property (verification-discipline: prefer a property that holds
    /// regardless of implementation over an eyeballed tolerance) -- forward then inverse
    /// must recover the input to within ordinary f64 round-off, for every block size this
    /// project's configs actually use.
    #[test]
    fn forward_then_inverse_round_trips_within_float_tolerance() {
        let mut rng = Rng(0xD17C_1234_5678_9ABC);
        for &size in &[2usize, 4, 8, 16] {
            for _ in 0..200 {
                let block: Vec<f64> = (0..size * size).map(|_| rng.next_f64() * 255.0).collect();
                let coeffs = forward_dct2d(&block, size);
                let back = inverse_dct2d(&coeffs, size);
                for (a, b) in block.iter().zip(back.iter()) {
                    assert!(
                        (a - b).abs() < 1e-9,
                        "size {size}: {a} vs {b} (diff {})",
                        (a - b).abs()
                    );
                }
            }
        }
    }

    /// A constant block's energy must land entirely in the DC coefficient (index 0) --
    /// the textbook property that catches a basis/normalisation bug cheaply.
    #[test]
    fn constant_block_has_energy_only_in_dc() {
        let size = 8;
        let block = vec![100.0; size * size];
        let coeffs = forward_dct2d(&block, size);
        assert!((coeffs[0] - 100.0 * size as f64).abs() < 1e-9, "dc={}", coeffs[0]);
        for &c in &coeffs[1..] {
            assert!(c.abs() < 1e-9, "expected 0 AC energy, got {c}");
        }
    }
}
