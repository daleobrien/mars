//! BD-rate and BD-PSNR (§M3).
//!
//! §M3 is a hard rule: a single `(bpp, PSNR)` pair is **not** a comparison between two
//! codecs, and a BD-rate quoted without the interval it was integrated over is
//! meaningless. This module therefore cannot produce a number without also producing
//! the interval, and it refuses inputs that do not meet the contract rather than
//! returning something quotable.

use mars_core::metrics::bpp;
use serde::{Deserialize, Serialize};

use crate::pchip::{is_greater, Pchip, PchipError};

/// §M3: at least this many quality points per curve.
pub const MIN_POINTS_PER_CURVE: usize = 4;

/// One operating point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RdPoint {
    pub bpp: f64,
    pub psnr: f64,
}

impl RdPoint {
    pub fn from_size(file_size_bytes: u64, width: usize, height: usize, psnr: f64) -> Self {
        Self {
            bpp: bpp(file_size_bytes, width, height),
            psnr,
        }
    }
}

/// A rate-distortion curve: a named set of operating points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RdCurve {
    pub label: String,
    pub points: Vec<RdPoint>,
}

impl RdCurve {
    pub fn new(label: impl Into<String>, points: Vec<RdPoint>) -> Self {
        Self {
            label: label.into(),
            points,
        }
    }

    /// Points sorted by increasing bpp, with the §M3 preconditions checked.
    fn prepared(&self) -> Result<Vec<RdPoint>, BdError> {
        if self.points.len() < MIN_POINTS_PER_CURVE {
            return Err(BdError::TooFewPoints {
                curve: self.label.clone(),
                got: self.points.len(),
                need: MIN_POINTS_PER_CURVE,
            });
        }
        if let Some(p) = self
            .points
            .iter()
            .find(|p| !is_greater(p.bpp, 0.0) || !p.psnr.is_finite())
        {
            return Err(BdError::NonFinitePoint {
                curve: self.label.clone(),
                bpp: p.bpp,
                psnr: p.psnr,
            });
        }
        let mut pts = self.points.clone();
        pts.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).expect("finite"));
        // A curve where more bits buy less quality is a broken measurement, not a
        // shape to interpolate through (§A7: anomalies halt).
        for w in pts.windows(2) {
            if !is_greater(w[1].bpp, w[0].bpp) || !is_greater(w[1].psnr, w[0].psnr) {
                return Err(BdError::NonMonotonic {
                    curve: self.label.clone(),
                    a: w[0],
                    b: w[1],
                });
            }
        }
        Ok(pts)
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BdError {
    #[error("curve '{curve}' has {got} points; §M3 requires at least {need}")]
    TooFewPoints {
        curve: String,
        got: usize,
        need: usize,
    },
    #[error(
        "curve '{curve}' contains a non-finite or non-positive point (bpp={bpp}, psnr={psnr})"
    )]
    NonFinitePoint { curve: String, bpp: f64, psnr: f64 },
    #[error("curve '{curve}' is not monotonically increasing: {a:?} then {b:?}")]
    NonMonotonic {
        curve: String,
        a: RdPoint,
        b: RdPoint,
    },
    #[error("curves '{a}' and '{b}' have no overlapping {axis} range; BD metrics are undefined")]
    NoOverlap {
        a: String,
        b: String,
        axis: &'static str,
    },
    #[error("interpolation failed: {0}")]
    Interpolation(#[from] PchipError),
}

/// A BD result, inseparable from the interval it was computed over (§M3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BdResult {
    pub reference: String,
    pub test: String,
    /// Average bitrate change of `test` relative to `reference`, in percent.
    /// Negative means the test curve needs fewer bits for the same quality.
    pub bd_rate_pct: f64,
    /// Average PSNR change of `test` relative to `reference`, in dB.
    pub bd_psnr_db: f64,
    /// The PSNR range the BD-rate integral was taken over, in dB.
    pub psnr_interval_db: (f64, f64),
    /// The bpp range spanned by the two curves across `psnr_interval_db`. This is the
    /// "bpp interval the number was computed over" §M3 requires to be quoted.
    pub bpp_interval: (f64, f64),
    /// The log-bpp overlap the BD-PSNR integral was taken over, expressed as bpp.
    pub bd_psnr_bpp_interval: (f64, f64),
    pub points_reference: usize,
    pub points_test: usize,
    pub interpolation: String,
}

/// BD-rate and BD-PSNR of `test` against `reference`.
///
/// Both are computed in the `(log10 bpp, PSNR)` plane per §M3:
/// - **BD-rate** integrates `log10(bpp)` as a function of PSNR over the PSNR overlap,
///   then `10^(mean difference) - 1`.
/// - **BD-PSNR** integrates PSNR as a function of `log10(bpp)` over the rate overlap.
pub fn bd_metrics(reference: &RdCurve, test: &RdCurve) -> Result<BdResult, BdError> {
    let a = reference.prepared()?;
    let b = test.prepared()?;

    // --- BD-rate: x = log10(bpp) as a function of y = PSNR -------------------
    let (ay, ax): (Vec<f64>, Vec<f64>) = a.iter().map(|p| (p.psnr, p.bpp.log10())).unzip();
    let (by, bx): (Vec<f64>, Vec<f64>) = b.iter().map(|p| (p.psnr, p.bpp.log10())).unzip();

    let rate_a = Pchip::new(&ay, &ax)?;
    let rate_b = Pchip::new(&by, &bx)?;

    let y_lo = ay[0].max(by[0]);
    let y_hi = ay[ay.len() - 1].min(by[by.len() - 1]);
    if !is_greater(y_hi, y_lo) {
        return Err(BdError::NoOverlap {
            a: reference.label.clone(),
            b: test.label.clone(),
            axis: "PSNR",
        });
    }
    let span = y_hi - y_lo;
    let int_a = rate_a.integrate(y_lo, y_hi);
    let int_b = rate_b.integrate(y_lo, y_hi);
    let bd_rate_pct = (10f64.powf((int_b - int_a) / span) - 1.0) * 100.0;

    // The rate span actually traversed, for the mandatory interval report.
    let bpp_lo = rate_a
        .eval(y_lo)
        .min(rate_b.eval(y_lo))
        .min(rate_a.eval(y_hi))
        .min(rate_b.eval(y_hi));
    let bpp_hi = rate_a
        .eval(y_lo)
        .max(rate_b.eval(y_lo))
        .max(rate_a.eval(y_hi))
        .max(rate_b.eval(y_hi));

    // --- BD-PSNR: y = PSNR as a function of x = log10(bpp) ------------------
    let dist_a = Pchip::new(&ax, &ay)?;
    let dist_b = Pchip::new(&bx, &by)?;
    let x_lo = ax[0].max(bx[0]);
    let x_hi = ax[ax.len() - 1].min(bx[bx.len() - 1]);
    if !is_greater(x_hi, x_lo) {
        return Err(BdError::NoOverlap {
            a: reference.label.clone(),
            b: test.label.clone(),
            axis: "bpp",
        });
    }
    let bd_psnr_db = (dist_b.integrate(x_lo, x_hi) - dist_a.integrate(x_lo, x_hi)) / (x_hi - x_lo);

    Ok(BdResult {
        reference: reference.label.clone(),
        test: test.label.clone(),
        bd_rate_pct,
        bd_psnr_db,
        psnr_interval_db: (y_lo, y_hi),
        bpp_interval: (10f64.powf(bpp_lo), 10f64.powf(bpp_hi)),
        bd_psnr_bpp_interval: (10f64.powf(x_lo), 10f64.powf(x_hi)),
        points_reference: a.len(),
        points_test: b.len(),
        interpolation: "pchip(log10 bpp, psnr)".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mars_core::tolerance::{BDRATE_ANALYTIC_PCT, BDRATE_SELF_PCT};

    fn curve(label: &str, bpps: &[f64], psnrs: &[f64]) -> RdCurve {
        RdCurve::new(
            label,
            bpps.iter()
                .zip(psnrs)
                .map(|(&bpp, &psnr)| RdPoint { bpp, psnr })
                .collect(),
        )
    }

    fn sample(label: &str) -> RdCurve {
        curve(
            label,
            &[0.125, 0.25, 0.5, 1.0, 2.0],
            &[26.1, 29.4, 32.8, 36.0, 39.7],
        )
    }

    #[test]
    fn a_curve_against_itself_is_zero() {
        let r = bd_metrics(&sample("a"), &sample("a")).unwrap();
        assert!(
            r.bd_rate_pct.abs() < BDRATE_SELF_PCT,
            "bd-rate {} should be 0",
            r.bd_rate_pct
        );
        assert!(r.bd_psnr_db.abs() < 1e-12, "bd-psnr {}", r.bd_psnr_db);
    }

    #[test]
    fn a_uniformly_scaled_curve_gives_the_analytic_answer() {
        // If every rate is multiplied by k at unchanged quality, then
        // log10(bpp_B)(y) = log10(bpp_A)(y) + log10(k) for all y, so the mean
        // difference is exactly log10(k) and BD-rate is exactly (k - 1) * 100%.
        // This holds for ANY interpolation scheme, which is what makes it a real test
        // of the integration rather than of the spline.
        for k in [0.5, 0.8, 1.0, 1.25, 2.0] {
            let a = sample("ref");
            let b = RdCurve::new(
                "scaled",
                a.points
                    .iter()
                    .map(|p| RdPoint {
                        bpp: p.bpp * k,
                        psnr: p.psnr,
                    })
                    .collect(),
            );
            let r = bd_metrics(&a, &b).unwrap();
            let expected = (k - 1.0) * 100.0;
            assert!(
                (r.bd_rate_pct - expected).abs() < BDRATE_ANALYTIC_PCT,
                "k={k}: got {}, expected {expected}",
                r.bd_rate_pct
            );
        }
    }

    #[test]
    fn a_uniformly_shifted_curve_gives_the_analytic_bd_psnr() {
        // PSNR_B(x) = PSNR_A(x) + delta for all x => BD-PSNR = delta exactly.
        for delta in [-1.5, 0.0, 0.75] {
            let a = sample("ref");
            let b = RdCurve::new(
                "shifted",
                a.points
                    .iter()
                    .map(|p| RdPoint {
                        bpp: p.bpp,
                        psnr: p.psnr + delta,
                    })
                    .collect(),
            );
            let r = bd_metrics(&a, &b).unwrap();
            assert!(
                (r.bd_psnr_db - delta).abs() < 1e-12,
                "delta={delta}: got {}",
                r.bd_psnr_db
            );
        }
    }

    #[test]
    fn the_reported_interval_is_the_psnr_overlap() {
        let a = curve("a", &[0.1, 0.2, 0.4, 0.8], &[25.0, 28.0, 31.0, 34.0]);
        let b = curve("b", &[0.2, 0.4, 0.8, 1.6], &[27.0, 30.0, 33.0, 36.0]);
        let r = bd_metrics(&a, &b).unwrap();
        assert_eq!(r.psnr_interval_db, (27.0, 34.0));
        assert!(r.bpp_interval.0 > 0.0 && r.bpp_interval.1 > r.bpp_interval.0);
    }

    #[test]
    fn three_points_is_rejected() {
        let a = curve("short", &[0.1, 0.2, 0.4], &[25.0, 28.0, 31.0]);
        assert!(matches!(
            bd_metrics(&a, &sample("b")),
            Err(BdError::TooFewPoints {
                got: 3,
                need: 4,
                ..
            })
        ));
    }

    #[test]
    fn disjoint_curves_are_rejected_rather_than_extrapolated() {
        let a = curve("low", &[0.1, 0.2, 0.3, 0.4], &[20.0, 21.0, 22.0, 23.0]);
        let b = curve("high", &[4.0, 5.0, 6.0, 7.0], &[40.0, 41.0, 42.0, 43.0]);
        assert!(matches!(
            bd_metrics(&a, &b),
            Err(BdError::NoOverlap { axis: "PSNR", .. })
        ));
    }

    #[test]
    fn a_non_monotonic_curve_is_an_error_not_a_shape() {
        let a = curve("broken", &[0.1, 0.2, 0.4, 0.8], &[25.0, 28.0, 27.0, 34.0]);
        assert!(matches!(
            bd_metrics(&a, &sample("b")),
            Err(BdError::NonMonotonic { .. })
        ));
    }
}
