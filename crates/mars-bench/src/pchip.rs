//! Piecewise cubic Hermite interpolation with Fritsch-Carlson slopes (PCHIP), plus an
//! exact closed-form integral.
//!
//! PCHIP rather than a natural cubic spline, deliberately: with only four or five RD
//! points a natural spline overshoots between knots, and the overshoot lands directly
//! in the BD-rate integral. PCHIP is shape-preserving, so a monotone RD curve
//! interpolates to a monotone curve. The slope rules are the ones MATLAB's `pchip` and
//! `scipy.interpolate.PchipInterpolator` implement, so the result is cross-checkable.
//!
//! The integral is evaluated analytically from the Hermite basis rather than by
//! quadrature. With a closed form available, a Simpson rule would only add an error
//! term to argue about.

/// A PCHIP interpolant over strictly increasing knots.
#[derive(Debug, Clone)]
pub struct Pchip {
    x: Vec<f64>,
    y: Vec<f64>,
    d: Vec<f64>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PchipError {
    #[error("need at least 2 points, got {0}")]
    TooFewPoints(usize),
    #[error("x values must be strictly increasing (violated at index {0})")]
    NotIncreasing(usize),
    #[error("x and y must have equal length ({0} vs {1})")]
    LengthMismatch(usize, usize),
}

impl Pchip {
    pub fn new(x: &[f64], y: &[f64]) -> Result<Self, PchipError> {
        if x.len() != y.len() {
            return Err(PchipError::LengthMismatch(x.len(), y.len()));
        }
        if x.len() < 2 {
            return Err(PchipError::TooFewPoints(x.len()));
        }
        for i in 1..x.len() {
            // NaN must fail this check, so the test is "is it greater?" rather than
            // "is it not less-or-equal" -- the latter silently accepts NaN.
            if !is_greater(x[i], x[i - 1]) {
                return Err(PchipError::NotIncreasing(i));
            }
        }
        let n = x.len();
        let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
        let delta: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();

        let mut d = vec![0.0; n];
        if n == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
        } else {
            for i in 1..n - 1 {
                if delta[i - 1] * delta[i] > 0.0 {
                    let w1 = 2.0 * h[i] + h[i - 1];
                    let w2 = h[i] + 2.0 * h[i - 1];
                    d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
                } else {
                    d[i] = 0.0;
                }
            }
            d[0] = edge_slope(h[0], h[1], delta[0], delta[1]);
            d[n - 1] = edge_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        }
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            d,
        })
    }

    pub fn domain(&self) -> (f64, f64) {
        (self.x[0], self.x[self.x.len() - 1])
    }

    /// Index of the interval containing `t`, clamped to the domain.
    fn interval(&self, t: f64) -> usize {
        let n = self.x.len();
        match self.x.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
            Ok(i) => i.min(n - 2),
            Err(0) => 0,
            Err(i) => (i - 1).min(n - 2),
        }
    }

    /// Evaluate, extrapolating with the end cubic outside the domain.
    pub fn eval(&self, t: f64) -> f64 {
        let i = self.interval(t);
        let h = self.x[i + 1] - self.x[i];
        let s = (t - self.x[i]) / h;
        let (h00, h10, h01, h11) = hermite(s);
        self.y[i] * h00 + h * self.d[i] * h10 + self.y[i + 1] * h01 + h * self.d[i + 1] * h11
    }

    /// Exact `integral_a^b f(t) dt`.
    ///
    /// # Panics
    /// If `a > b`, or if either bound lies outside the interpolation domain — extending
    /// the integral past the data would be extrapolation dressed up as measurement.
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        let (lo, hi) = self.domain();
        assert!(a <= b, "integration bounds reversed");
        assert!(
            a >= lo - 1e-12 && b <= hi + 1e-12,
            "integration range [{a}, {b}] is outside the data domain [{lo}, {hi}]"
        );
        let a = a.max(lo);
        let b = b.min(hi);
        if b <= a {
            return 0.0;
        }
        let mut total = 0.0;
        for i in 0..self.x.len() - 1 {
            let (x0, x1) = (self.x[i], self.x[i + 1]);
            let seg_lo = a.max(x0);
            let seg_hi = b.min(x1);
            if seg_hi <= seg_lo {
                continue;
            }
            let h = x1 - x0;
            let s0 = (seg_lo - x0) / h;
            let s1 = (seg_hi - x0) / h;
            let (f00, f10, f01, f11) = (
                anti_h00(s1) - anti_h00(s0),
                anti_h10(s1) - anti_h10(s0),
                anti_h01(s1) - anti_h01(s0),
                anti_h11(s1) - anti_h11(s0),
            );
            total += h
                * (self.y[i] * f00
                    + h * self.d[i] * f10
                    + self.y[i + 1] * f01
                    + h * self.d[i + 1] * f11);
        }
        total
    }
}

/// One-sided three-point slope with the shape-preserving clamp, matching
/// `scipy.interpolate.PchipInterpolator`'s `_edge_case`.
fn edge_slope(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let mut d = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if d.signum() != d0.signum() {
        d = 0.0;
    } else if d0.signum() != d1.signum() && d.abs() > (3.0 * d0).abs() {
        d = 3.0 * d0;
    }
    d
}

/// `a > b`, as a named function so that NaN-rejecting comparisons read as intent rather
/// than as a negated operator clippy would rather we did not write.
#[inline]
pub(crate) fn is_greater(a: f64, b: f64) -> bool {
    matches!(a.partial_cmp(&b), Some(std::cmp::Ordering::Greater))
}

fn hermite(s: f64) -> (f64, f64, f64, f64) {
    let (s2, s3) = (s * s, s * s * s);
    (
        2.0 * s3 - 3.0 * s2 + 1.0,
        s3 - 2.0 * s2 + s,
        -2.0 * s3 + 3.0 * s2,
        s3 - s2,
    )
}

fn anti_h00(s: f64) -> f64 {
    s.powi(4) / 2.0 - s.powi(3) + s
}
fn anti_h10(s: f64) -> f64 {
    s.powi(4) / 4.0 - 2.0 * s.powi(3) / 3.0 + s * s / 2.0
}
fn anti_h01(s: f64) -> f64 {
    -s.powi(4) / 2.0 + s.powi(3)
}
fn anti_h11(s: f64) -> f64 {
    s.powi(4) / 4.0 - s.powi(3) / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_through_the_knots() {
        let x = [0.0, 1.0, 2.5, 4.0];
        let y = [1.0, 3.0, 2.0, 5.0];
        let p = Pchip::new(&x, &y).unwrap();
        for (&xi, &yi) in x.iter().zip(&y) {
            assert!(
                (p.eval(xi) - yi).abs() < 1e-12,
                "knot {xi} -> {}",
                p.eval(xi)
            );
        }
    }

    #[test]
    fn integrates_a_straight_line_exactly() {
        // PCHIP of collinear points is the line itself, so the integral is analytic.
        let x = [0.0, 1.0, 2.0, 3.0];
        let y = [0.0, 2.0, 4.0, 6.0];
        let p = Pchip::new(&x, &y).unwrap();
        assert!((p.integrate(0.0, 3.0) - 9.0).abs() < 1e-12);
        assert!((p.integrate(0.5, 1.5) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn is_shape_preserving() {
        // The classic overshoot case: a natural cubic spline exceeds 1.0 here.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y = [0.0, 0.0, 0.0, 1.0, 1.0];
        let p = Pchip::new(&x, &y).unwrap();
        for i in 0..=400 {
            let v = p.eval(i as f64 / 100.0);
            assert!((-1e-12..=1.0 + 1e-12).contains(&v), "overshoot: {v}");
        }
    }

    #[test]
    fn rejects_unsorted_input() {
        assert_eq!(
            Pchip::new(&[0.0, 2.0, 1.0], &[0.0, 1.0, 2.0]).unwrap_err(),
            PchipError::NotIncreasing(2)
        );
        assert_eq!(
            Pchip::new(&[0.0, f64::NAN, 1.0], &[0.0, 1.0, 2.0]).unwrap_err(),
            PchipError::NotIncreasing(1)
        );
    }
}
