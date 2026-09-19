//! Step 14's λ sweep, BD-rate against the Step 9 `Exhaustive` reference, and the
//! convexity/monotonicity check the brief calls "the fastest available diagnostic" for a
//! rate-estimation bug.
//!
//! "Matched search effort" (the exit criterion's own phrase) means comparing against
//! `mars_codec::encode`'s exhaustive per-block domain x isometry search -- the same full
//! search Step 14's own bottom-up walk uses at every node -- rather than against one of
//! Step 9's *restricted*-candidate methods (Fisher, Saupe, ...), which spend deliberately
//! less search effort per block. The reference curve here is exactly `rust_encoder.rs`'s
//! own `RMS_GRID` sweep of the legacy top-down `walk` (`lambda: None`), which is Step 9's
//! own "Exhaustive" baseline; the test curve is the same encoder with `lambda: Some(_)`.
//! Both curves measure `.mars` v0 (entropy-coded) bpp, since Step 14 optimises the actual
//! coded rate, not the raw `.ifs` bit count.

use mars_codec::encode::{encode_image, EncodeParams};
use mars_codec::ifs::{decode_iterative, Leaf};
use mars_codec::mars_format;
use mars_core::metrics::psnr;
use mars_core::Plane;

use crate::bdrate::{RdCurve, RdPoint};

/// One operating point plus the search cost that produced it -- Step 16's own "report the
/// cost alongside the gain" precedent, applied here since bottom-up RD pruning visits
/// every node of the quadtree rather than stopping early (`docs/decisions.md`).
#[derive(Debug, Clone, Copy)]
pub struct RdSample {
    pub point: RdPoint,
    pub evals: u64,
    pub leaves: usize,
}

/// Encode `image` with `params`, and measure the resulting `.mars` v0 bpp and decode
/// PSNR -- one point on an RD curve, plus its search cost.
pub fn sample(image: &Plane, params: &EncodeParams) -> RdSample {
    let (hdr, leaves, evals) = encode_image(image, params);
    let bytes = mars_format::write(&hdr, &leaves)
        .expect("a partition `encode_image` produced must always be a writable `.mars` v0 tree");
    sample_from_bytes(image, &bytes, evals)
        .expect("the serialized `encode_image` partition must be readable")
        .0
}

/// Measure an in-memory `.mars` inner stream, not a full CLI file-to-file round trip.
/// Only parsed reconstruction metadata and leaves may contribute to decoded quality.
pub(crate) fn sample_from_bytes(
    image: &Plane,
    bytes: &[u8],
    evals: u64,
) -> Result<(RdSample, Vec<Leaf>), mars_format::MarsFormatError> {
    let (hdr, leaves) = mars_format::read(bytes)?;
    // Keep the full parsed Header: coercing to ifs::Header would lose residual qstep.
    let decoded = decode_iterative(&hdr, &leaves, 10);
    let psnr_db = psnr(image, &decoded).unwrap_or(f64::INFINITY); // None = lossless (§M2)
    Ok((
        RdSample {
            point: RdPoint::from_size(bytes.len() as u64, image.width(), image.height(), psnr_db),
            evals,
            leaves: leaves.len(),
        },
        leaves,
    ))
}

/// The Step 9 `Exhaustive` reference curve: the legacy top-down, `t_rms`-threshold walk
/// (`lambda: None`) swept over `t_rms_grid`, measured at `.mars` v0 bpp (not raw `.ifs`,
/// so the comparison is against the same entropy-coded rate Step 14 is optimising).
pub fn exhaustive_reference_curve(
    label: impl Into<String>,
    image: &Plane,
    base: &EncodeParams,
    t_rms_grid: &[f64],
) -> (RdCurve, Vec<RdSample>) {
    let mut samples = Vec::with_capacity(t_rms_grid.len());
    for &t_rms in t_rms_grid {
        let params = EncodeParams {
            t_rms,
            lambda: None,
            ..*base
        };
        samples.push(sample(image, &params));
    }
    let curve = RdCurve::new(label, samples.iter().map(|s| s.point).collect());
    (curve, samples)
}

/// Step 14's own RD curve: the bottom-up `J = D + λR` walk swept over `lambda_grid`.
pub fn lambda_curve(
    label: impl Into<String>,
    image: &Plane,
    base: &EncodeParams,
    lambda_grid: &[f64],
) -> (RdCurve, Vec<RdSample>) {
    let mut samples = Vec::with_capacity(lambda_grid.len());
    for &lambda in lambda_grid {
        let params = EncodeParams {
            lambda: Some(lambda),
            ..*base
        };
        samples.push(sample(image, &params));
    }
    let curve = RdCurve::new(label, samples.iter().map(|s| s.point).collect());
    (curve, samples)
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ConvexityError {
    #[error("need at least 3 points to check convexity, got {0}")]
    TooFewPoints(usize),
    #[error(
        "curve is not monotonic: bpp {bpp_a} -> {bpp_b} but psnr {psnr_a} -> {psnr_b} \
         (higher rate must not buy lower quality)"
    )]
    NotMonotonic {
        bpp_a: f64,
        bpp_b: f64,
        psnr_a: f64,
        psnr_b: f64,
    },
    #[error(
        "curve is not convex between bpp {bpp_lo:.4}/{bpp_mid:.4}/{bpp_hi:.4}: slope \
         increased from {slope_a:.3} to {slope_b:.3} dB/bpp (diminishing returns violated \
         -- per the Step 14 brief, this is the signature of a rate-estimation bug, not a \
         result to report as-is)"
    )]
    NotConvex {
        bpp_lo: f64,
        bpp_mid: f64,
        bpp_hi: f64,
        slope_a: f64,
        slope_b: f64,
    },
}

/// The brief's own diagnostic: sort by bpp, require PSNR strictly increasing with bpp
/// (monotonicity), and require the secant slopes `dPSNR/dbpp` to be non-increasing
/// (convexity -- diminishing returns, the standard shape of a real RD curve). `slack`
/// tolerates the ordinary numerical noise of a coarse, few-point sweep without hiding a
/// real sign-flip; it is declared once here, not scattered inline (§A2).
pub fn check_convex_and_monotonic(points: &[RdPoint], slack: f64) -> Result<(), ConvexityError> {
    if points.len() < 3 {
        return Err(ConvexityError::TooFewPoints(points.len()));
    }
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).expect("finite bpp"));

    for w in pts.windows(2) {
        if w[1].psnr <= w[0].psnr {
            return Err(ConvexityError::NotMonotonic {
                bpp_a: w[0].bpp,
                bpp_b: w[1].bpp,
                psnr_a: w[0].psnr,
                psnr_b: w[1].psnr,
            });
        }
    }

    let slopes: Vec<f64> = pts
        .windows(2)
        .map(|w| (w[1].psnr - w[0].psnr) / (w[1].bpp - w[0].bpp))
        .collect();
    for i in 0..slopes.len() - 1 {
        if slopes[i + 1] > slopes[i] + slack {
            return Err(ConvexityError::NotConvex {
                bpp_lo: pts[i].bpp,
                bpp_mid: pts[i + 1].bpp,
                bpp_hi: pts[i + 2].bpp,
                slope_a: slopes[i],
                slope_b: slopes[i + 1],
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual_stream() -> (mars_format::Header, Vec<Leaf>) {
        use mars_codec::ifs::Header;
        use mars_codec::quant::ResidualQstep;

        let hdr = mars_format::Header {
            geometry: Header {
                bits_alfa: 4,
                bits_beta: 7,
                min_size: 4,
                max_size: 4,
                shift: 4,
                width: 8,
                height: 8,
                int_max_alfa: 32,
            },
            residual_qstep: ResidualQstep::LEGACY,
        };
        let leaves = [(0, 0), (4, 0), (0, 4), (4, 4)]
            .into_iter()
            .map(|(row, col)| {
                let mut residual = vec![0; 16];
                residual[0] = 4;
                residual[1] = -2;
                residual[5] = 1;
                Leaf {
                    row,
                    col,
                    size: 4,
                    mode: 3,
                    qalfa: 4,
                    qbeta: 64,
                    isometry: 0,
                    dom_row: 0,
                    dom_col: 0,
                    qgx: 0,
                    qgy: 0,
                    residual,
                }
            })
            .collect();
        (hdr, leaves)
    }

    #[test]
    fn serialized_measurement_uses_wire_qstep_and_leaves() {
        use mars_codec::quant::ResidualQstep;

        let (mut hdr, mut leaves) = residual_stream();
        let image = Plane::from_vec(8, 8, vec![128; 64]);
        let mut bytes = mars_format::write(&hdr, &leaves).unwrap();
        let (legacy, _) = sample_from_bytes(&image, &bytes, 17).unwrap();

        // Change only the serialized qstep: reconstruction must not default to fixed8.
        let step = ResidualQstep::new(17.123456789).unwrap();
        bytes[20..24].copy_from_slice(&(step.get() as f32).to_le_bytes());
        hdr.residual_qstep = step;
        let expected = decode_iterative(&hdr, &leaves, 10);
        assert_ne!(expected, decode_iterative(&hdr.geometry, &leaves, 10));
        let (sample, parsed_leaves) = sample_from_bytes(&image, &bytes, 17).unwrap();
        assert_eq!(parsed_leaves, leaves);
        assert_eq!(sample.evals, 17);
        assert_eq!(sample.leaves, leaves.len());
        assert_eq!(
            sample.point,
            RdPoint::from_size(bytes.len() as u64, 8, 8, psnr(&image, &expected).unwrap())
        );
        assert_ne!(sample.point.psnr, legacy.point.psnr);
        assert_eq!(mars_format::read(&bytes).unwrap().0, hdr);

        // A different serialized leaf payload must change the measured reconstruction.
        for leaf in &mut leaves {
            leaf.residual.fill(0);
        }
        let changed_bytes = mars_format::write(&hdr, &leaves).unwrap();
        let changed_image = decode_iterative(&hdr, &leaves, 10);
        assert_ne!(changed_image, expected);
        let (changed, parsed_leaves) = sample_from_bytes(&image, &changed_bytes, 17).unwrap();
        assert_eq!(parsed_leaves, leaves);
        assert_eq!(
            changed.point,
            RdPoint::from_size(
                changed_bytes.len() as u64,
                8,
                8,
                psnr(&image, &changed_image).unwrap_or(f64::INFINITY),
            )
        );
        assert_ne!(changed.point.psnr, sample.point.psnr);

        let (lossless, _) = sample_from_bytes(&changed_image, &changed_bytes, 0).unwrap();
        assert_eq!(lossless.point.psnr, f64::INFINITY);
    }

    #[test]
    fn serialized_measurement_rejects_invalid_streams_without_fallback() {
        use mars_format::MarsFormatError;

        let (hdr, leaves) = residual_stream();
        let image = Plane::from_vec(8, 8, vec![128; 64]);
        let mut bytes = mars_format::write(&hdr, &leaves).unwrap();
        assert!(matches!(
            sample_from_bytes(&image, &bytes[..10], 0),
            Err(MarsFormatError::Truncated)
        ));
        bytes[20..24].copy_from_slice(&0.0_f32.to_le_bytes());
        assert!(matches!(
            sample_from_bytes(&image, &bytes, 0),
            Err(MarsFormatError::InvalidResidualQstep)
        ));
    }

    fn pt(bpp: f64, psnr: f64) -> RdPoint {
        RdPoint { bpp, psnr }
    }

    #[test]
    fn a_textbook_convex_curve_passes() {
        let pts = vec![pt(0.1, 20.0), pt(0.2, 26.0), pt(0.4, 30.0), pt(0.8, 32.0)];
        assert!(check_convex_and_monotonic(&pts, 1e-9).is_ok());
    }

    #[test]
    fn a_non_monotonic_curve_is_rejected() {
        let pts = vec![pt(0.1, 20.0), pt(0.2, 18.0), pt(0.4, 30.0)];
        assert!(matches!(
            check_convex_and_monotonic(&pts, 1e-9),
            Err(ConvexityError::NotMonotonic { .. })
        ));
    }

    #[test]
    fn a_non_convex_curve_is_rejected() {
        // Slope increases from the first to the second segment -- more bits buying
        // *more* marginal quality than fewer bits did, the rate-estimation-bug shape.
        let pts = vec![pt(0.1, 20.0), pt(0.2, 21.0), pt(0.4, 30.0)];
        assert!(matches!(
            check_convex_and_monotonic(&pts, 1e-9),
            Err(ConvexityError::NotConvex { .. })
        ));
    }
}
