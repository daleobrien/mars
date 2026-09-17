//! A scalar dead-zone quantiser for Step 15's residual-mode DCT coefficients.
//!
//! "Dead-zone" means the bin around zero is wider than the other bins (`dz_frac` controls
//! how much wider) -- the standard trick transform coders use to push small, noise-like
//! coefficients to exactly zero rather than +-1, which is what makes a residual block's
//! coefficient sequence sparse (and hence cheap to entropy-code: `crate::residual`'s
//! nonzero/magnitude event scheme relies on most positions being zero).
//!
//! Pure integer/float functions, no interior state -- deterministic by construction (no
//! iteration order to depend on), and the round-trip property below is the exact-equality
//! oracle this module ships instead of a numeric tolerance.

/// A positive, finite binary32 residual step in `[1, 65535]`.
///
/// Stored as bits so headers retain exact equality. Encoding, distortion scoring and
/// reconstruction all use the same rounded value, not the unrounded lambda mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidualQstep(u32);

impl ResidualQstep {
    /// The pre-O7 fixed step, retained for diagnostic comparisons and `.ifs` metadata.
    pub const LEGACY: Self = Self(8.0f32.to_bits());
    /// Minimum step: one intensity unit in the orthonormal DCT domain.
    pub const MIN: f64 = 1.0;
    /// Maximum step: the largest unsigned 16-bit value (exact in binary32).
    pub const MAX: f64 = 65535.0;

    /// Validate and round a diagnostic fixed step to its wire representation.
    pub fn new(step: f64) -> Option<Self> {
        if step.is_finite() && (Self::MIN..=Self::MAX).contains(&step) {
            Some(Self((step as f32).to_bits()))
        } else {
            None
        }
    }

    /// Closed-form high-rate mapping, chosen before O7/Step22 measurement.
    ///
    /// Orthonormal DCT preserves SSE. With per-coefficient `D ≈ q²/12` and
    /// `R ≈ C - log2(q)` bits, minimising `D + lambda*R` gives
    /// `q = sqrt(6*lambda/ln(2))`. Clipping, sparse levels and the frozen rate
    /// estimator violate this approximation; this is not a claim of RD optimality.
    /// The bounds are representational, not fitted to a corpus. Square-rooting
    /// lambda first avoids overflow at finite extreme inputs. Nonpositive/NaN
    /// lambda maps to MIN, positive infinity to MAX; encoders reject invalid lambda.
    pub fn from_lambda(lambda: f64) -> Self {
        let step = (lambda.max(0.0).sqrt() * (6.0 / std::f64::consts::LN_2).sqrt())
            .clamp(Self::MIN, Self::MAX);
        Self::new(step).expect("bounded finite step")
    }

    /// Exact wire value promoted to binary64 for quantisation and reconstruction.
    pub fn get(self) -> f64 {
        f64::from(f32::from_bits(self.0))
    }

    pub(crate) fn to_le_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    pub(crate) fn from_le_bytes(bytes: [u8; 4]) -> Option<Self> {
        Self::new(f64::from(f32::from_le_bytes(bytes)))
    }
}

/// Quantise `x` (a DCT coefficient) to an integer level with step `step` and dead-zone
/// fraction `dz_frac` (`0.0` = plain mid-tread uniform quantiser, `>0.0` widens the zero
/// bin): `level = sign(x) * floor(|x|/step - dz_frac + 1)` when `|x| > dz_frac * step`,
/// else `0`. Equivalent to comparing `|x|` against the dead-zone threshold up front and
/// otherwise rounding as usual, which is how the expression below is actually written.
pub fn dead_zone_quantize(x: f64, step: f64, dz_frac: f64) -> i32 {
    debug_assert!(step > 0.0, "quantisation step must be positive");
    debug_assert!((0.0..1.0).contains(&dz_frac), "dz_frac must be in [0, 1)");
    let threshold = dz_frac * step;
    let ax = x.abs();
    if ax <= threshold {
        return 0;
    }
    let level = ((ax - threshold) / step).floor() + 1.0;
    let signed = if x < 0.0 { -level } else { level };
    signed as i32
}

/// Dequantise a level back to a representative real value: the centre of its bin,
/// `sign(level) * (dz_frac * step + (|level| - 1 + 0.5) * step)` for `level != 0`, else
/// `0.0`. Matches [`dead_zone_quantize`]'s bin boundaries exactly.
pub fn dead_zone_dequantize(level: i32, step: f64, dz_frac: f64) -> f64 {
    if level == 0 {
        return 0.0;
    }
    let threshold = dz_frac * step;
    let mag = threshold + (level.unsigned_abs() as f64 - 1.0 + 0.5) * step;
    if level < 0 {
        -mag
    } else {
        mag
    }
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
            (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) * 2000.0 - 1000.0
        }
    }

    #[test]
    fn zero_maps_to_zero() {
        assert_eq!(dead_zone_quantize(0.0, 4.0, 0.5), 0);
        assert_eq!(dead_zone_dequantize(0, 4.0, 0.5), 0.0);
    }

    /// The exact-equality oracle this module relies on instead of a tolerance
    /// (verification-discipline): quantising the dequantised value of any level must
    /// reproduce that exact level -- the quantiser and dequantiser must agree on where
    /// every bin boundary is, for any step/dead-zone/level combination.
    #[test]
    fn requantising_a_dequantised_level_is_idempotent() {
        let mut rng = Rng(0xFEED_1357_2468_ACE0);
        for _ in 0..200_000 {
            let step = (rng.next_f64().abs() / 40.0).max(0.05);
            let dz = (rng.next_f64().abs() / 2000.0).min(0.49);
            let level = ((rng.next_f64() / 20.0) as i32).clamp(-63, 63);
            let x = dead_zone_dequantize(level, step, dz);
            let requantized = dead_zone_quantize(x, step, dz);
            assert_eq!(
                requantized, level,
                "step={step} dz={dz} level={level} x={x} requantized={requantized}"
            );
        }
    }

    #[test]
    fn larger_step_never_increases_magnitude_of_the_quantised_level() {
        let mut rng = Rng(0x1234_5678_ABCD_EF00);
        for _ in 0..50_000 {
            let x = rng.next_f64();
            let small_step = (rng.next_f64().abs() / 40.0).max(0.05);
            let big_step = small_step * 3.0;
            let a = dead_zone_quantize(x, small_step, 0.5).unsigned_abs();
            let b = dead_zone_quantize(x, big_step, 0.5).unsigned_abs();
            assert!(b <= a, "x={x} small_step={small_step} a={a} b={b}");
        }
    }
}
