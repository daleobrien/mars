//! Moment-accumulation kernels -- the inner loop Step 6 made the encoder's hot path and
//! Step 7 proved dominates search cost (implementation-plan.md Step 11's priority order
//! names this pair first, ahead of SAD/downsample/gradient/DCT).
//!
//! Both kernels operate on plain slices, not `mars_codec`'s own `Contracted`/`Plane`
//! types, so this crate has no dependency on `mars-codec` -- `mars-codec` depends on
//! *this* crate instead, and adapts its own strided-plane types to these signatures at
//! the call site (`crates/mars-codec/src/encode.rs`).
//!
//! Every sum below is over `i32`/`i64` integers with no floating point involved, so unlike
//! a typical SIMD kernel there is no reassociation hazard to tolerate: integer addition
//! and multiplication are exactly associative and commutative regardless of instruction
//! order, as long as no intermediate overflows. Each kernel therefore has an **exact**
//! differential test against its scalar reference (`tests` below), not an epsilon
//! comparison -- the stronger check Step 11's exit criteria ask for precisely because nothing
//! here needs a weaker one.

use std::arch::aarch64::*;

/// Scalar reference for [`domain_sums`].
pub fn domain_sums_scalar(
    plane: &[i32],
    stride: usize,
    dr: usize,
    dc: usize,
    size: usize,
) -> (i64, i64) {
    let (mut s1, mut s2) = (0i64, 0i64);
    for u in 0..size {
        let start = (dr + u) * stride + dc;
        for &d in &plane[start..start + size] {
            let d = i64::from(d);
            s1 += d;
            s2 += d * d;
        }
    }
    (s1, s2)
}

/// `(ΣD, ΣD²)` over one `size x size` window of a row-major, `stride`-wide `i32` plane --
/// `mars_codec::encode::Contracted`'s own 2:1 box-sum plane, from the caller's side.
///
/// # Panics
/// If the window `[dr, dr+size) x [dc, dc+size)` runs past `plane`'s bounds for the given
/// `stride`. Same contract as [`domain_sums_scalar`] (a direct slice index would panic the
/// same way); the NEON path does not add its own bounds requirement.
pub fn domain_sums(plane: &[i32], stride: usize, dr: usize, dc: usize, size: usize) -> (i64, i64) {
    // SAFETY: NEON is unconditional on aarch64 (implementation-plan.md §2.2 -- this
    // project targets one machine, Apple Silicon) -- no runtime feature check needed.
    unsafe { domain_sums_neon(plane, stride, dr, dc, size) }
}

#[target_feature(enable = "neon")]
unsafe fn domain_sums_neon(
    plane: &[i32],
    stride: usize,
    dr: usize,
    dc: usize,
    size: usize,
) -> (i64, i64) {
    let mut acc_s1 = vdupq_n_s64(0);
    let mut acc_s2 = vdupq_n_s64(0);
    let mut tail_s1 = 0i64;
    let mut tail_s2 = 0i64;

    for u in 0..size {
        let start = (dr + u) * stride + dc;
        let row = &plane[start..start + size];
        let mut v = 0usize;
        while v + 4 <= size {
            let d = vld1q_s32(row.as_ptr().add(v));
            let lo = vget_low_s32(d);
            let hi = vget_high_s32(d);
            acc_s1 = vaddq_s64(acc_s1, vmovl_s32(lo));
            acc_s1 = vaddq_s64(acc_s1, vmovl_s32(hi));
            acc_s2 = vaddq_s64(acc_s2, vmull_s32(lo, lo));
            acc_s2 = vaddq_s64(acc_s2, vmull_s32(hi, hi));
            v += 4;
        }
        while v < size {
            let d = i64::from(row[v]);
            tail_s1 += d;
            tail_s2 += d * d;
            v += 1;
        }
    }

    let s1 = vgetq_lane_s64(acc_s1, 0) + vgetq_lane_s64(acc_s1, 1) + tail_s1;
    let s2 = vgetq_lane_s64(acc_s2, 0) + vgetq_lane_s64(acc_s2, 1) + tail_s2;
    (s1, s2)
}

/// Scalar reference for [`dot_u8_i32`].
pub fn dot_u8_i32_scalar(a: &[u8], b: &[i32]) -> i64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&a, &b)| i64::from(a) * i64::from(b))
        .sum()
}

/// `Σ a[i] * b[i]` for two equal-length contiguous slices -- `mars_codec::encode`'s
/// `cross_term` (`Σ r·D` under one isometry), once the range block has been permuted into
/// the domain's raster order so both operands are contiguous (see that module's call
/// site). `a` is range pixels (`u8`), `b` is domain box-sums (`i32`, up to 1020 -- see
/// `Contracted`'s own doc).
///
/// **On `UDOT`:** Step 11's brief suggests expanding this into four `u8 x u8` dot products
/// (`D = a+b+c+e`, the four un-summed corner pixels `Contracted` averages together) so
/// `UDOT` can accumulate `u8 x u8 -> u32` four-at-a-time, which is a real, larger
/// vectorisation width than the `i32` multiply below manages. Not implemented this
/// session: it needs a second plane-splitting builder (four half-resolution `u8` planes
/// instead of `Contracted`'s one summed `i32` plane) so the four corner pixels are
/// contiguously addressable, which is a real structural addition on top of this step's
/// time budget, not a small kernel change -- recorded in `docs/decisions.md` rather than
/// silently dropped.
///
/// # Panics
/// If `a.len() != b.len()`.
pub fn dot_u8_i32(a: &[u8], b: &[i32]) -> i64 {
    assert_eq!(a.len(), b.len());
    // SAFETY: NEON is unconditional on aarch64 (§2.2).
    unsafe { dot_u8_i32_neon(a, b) }
}

#[target_feature(enable = "neon")]
unsafe fn dot_u8_i32_neon(a: &[u8], b: &[i32]) -> i64 {
    let n = a.len();
    let mut acc = vdupq_n_s64(0);
    let mut tail = 0i64;
    let mut i = 0usize;

    while i + 8 <= n {
        let av = vld1_u8(a.as_ptr().add(i));
        let av16 = vmovl_u8(av);
        let a_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(av16)));
        let a_hi = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(av16)));
        let b_lo = vld1q_s32(b.as_ptr().add(i));
        let b_hi = vld1q_s32(b.as_ptr().add(i + 4));
        let p_lo = vmulq_s32(a_lo, b_lo);
        let p_hi = vmulq_s32(a_hi, b_hi);
        acc = vaddq_s64(acc, vmovl_s32(vget_low_s32(p_lo)));
        acc = vaddq_s64(acc, vmovl_s32(vget_high_s32(p_lo)));
        acc = vaddq_s64(acc, vmovl_s32(vget_low_s32(p_hi)));
        acc = vaddq_s64(acc, vmovl_s32(vget_high_s32(p_hi)));
        i += 8;
    }
    while i < n {
        tail += i64::from(a[i]) * i64::from(b[i]);
        i += 1;
    }

    vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1) + tail
}

/// Scalar reference for [`dot_u8_i32_window`].
pub fn dot_u8_i32_window_scalar(
    a: &[u8],
    b: &[i32],
    b_stride: usize,
    br: usize,
    bc: usize,
    size: usize,
) -> i64 {
    assert_eq!(a.len(), size * size);
    (0..size)
        .map(|u| {
            let a_row = &a[u * size..u * size + size];
            let start = (br + u) * b_stride + bc;
            dot_u8_i32_scalar(a_row, &b[start..start + size])
        })
        .sum()
}

/// `Σ_(u,v) a[u*size+v] * b[(br+u)*b_stride+bc+v]` -- `mars_codec::encode`'s `cross_term`
/// (`Σ r·D` under one isometry), with `a` the range block already permuted into the
/// domain's raster order (once per isometry, not per domain position -- see the call
/// site) and `b` addressed the same strided way [`domain_sums`] is, so no per-domain-
/// position copy is needed on either operand: each of the `size` rows is contiguous in
/// both `a` and `b`, which is exactly what [`dot_u8_i32`] wants.
///
/// # Panics
/// If `a.len() != size * size`, or if the windowed region of `b` runs past its bounds.
pub fn dot_u8_i32_window(
    a: &[u8],
    b: &[i32],
    b_stride: usize,
    br: usize,
    bc: usize,
    size: usize,
) -> i64 {
    assert_eq!(a.len(), size * size);
    (0..size)
        .map(|u| {
            let a_row = &a[u * size..u * size + size];
            let start = (br + u) * b_stride + bc;
            dot_u8_i32(a_row, &b[start..start + size])
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn domain_sums_matches_scalar_exactly() {
        let mut rng = StdRng::seed_from_u64(0xD0_A1_A1_A1);
        for _ in 0..500 {
            let stride = rng.gen_range(1..40);
            let height = rng.gen_range(1..40);
            let plane: Vec<i32> = (0..stride * height)
                .map(|_| rng.gen_range(0..=1020))
                .collect();
            let size = rng.gen_range(1..=stride.min(height));
            let dr = rng.gen_range(0..=height - size);
            let dc = rng.gen_range(0..=stride - size);

            let want = domain_sums_scalar(&plane, stride, dr, dc, size);
            let got = domain_sums(&plane, stride, dr, dc, size);
            assert_eq!(want, got, "stride={stride} dr={dr} dc={dc} size={size}");
        }
    }

    #[test]
    fn domain_sums_handles_every_remainder_class() {
        // Sizes 1..=16 exercise every possible 4-lane tail (0, 1, 2, 3 leftover elements).
        for size in 1..=16usize {
            let stride = size + 3; // a non-trivial stride, distinct from size
            let plane: Vec<i32> = (0..stride * size).map(|v| v as i32).collect();
            let want = domain_sums_scalar(&plane, stride, 0, 0, size);
            let got = domain_sums(&plane, stride, 0, 0, size);
            assert_eq!(want, got, "size={size}");
        }
    }

    #[test]
    fn dot_u8_i32_matches_scalar_exactly() {
        let mut rng = StdRng::seed_from_u64(0x5EED_5EED);
        for _ in 0..500 {
            let n = rng.gen_range(0..=200);
            let a: Vec<u8> = (0..n).map(|_| rng.gen()).collect();
            let b: Vec<i32> = (0..n).map(|_| rng.gen_range(0..=1020)).collect();
            assert_eq!(dot_u8_i32_scalar(&a, &b), dot_u8_i32(&a, &b), "n={n}");
        }
    }

    #[test]
    fn dot_u8_i32_handles_every_remainder_class() {
        for n in 0..=24usize {
            let a: Vec<u8> = (0..n).map(|i| (i * 7) as u8).collect();
            let b: Vec<i32> = (0..n).map(|i| i as i32 * 3).collect();
            assert_eq!(dot_u8_i32_scalar(&a, &b), dot_u8_i32(&a, &b), "n={n}");
        }
    }

    #[test]
    #[should_panic]
    fn dot_u8_i32_rejects_mismatched_lengths() {
        let _ = dot_u8_i32(&[1, 2, 3], &[1, 2]);
    }

    #[test]
    fn dot_u8_i32_window_matches_scalar_exactly() {
        let mut rng = StdRng::seed_from_u64(0x0710_0710);
        for _ in 0..500 {
            let b_stride = rng.gen_range(1..40);
            let b_height = rng.gen_range(1..40);
            let b: Vec<i32> = (0..b_stride * b_height)
                .map(|_| rng.gen_range(0..=1020))
                .collect();
            let size = rng.gen_range(1..=b_stride.min(b_height));
            let br = rng.gen_range(0..=b_height - size);
            let bc = rng.gen_range(0..=b_stride - size);
            let a: Vec<u8> = (0..size * size).map(|_| rng.gen()).collect();

            let want = dot_u8_i32_window_scalar(&a, &b, b_stride, br, bc, size);
            let got = dot_u8_i32_window(&a, &b, b_stride, br, bc, size);
            assert_eq!(want, got, "b_stride={b_stride} br={br} bc={bc} size={size}");
        }
    }
}
