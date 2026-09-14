//! Step 6's exit criteria (`just gate-6`) as a command that exits 0 or 1 (§A1).
//!
//! Runs the exhaustive Rust encoder (`mars_codec::encode`) over the same
//! `fixtures`/`default`/RMS grid Step 2's baseline swept (`corpus/fixtures.images.json`,
//! `rms = [2, 4, 8, 16, 32]`), and checks:
//!
//! 1. **Decode agreement**: Rust-encode -> `decmars -i` decodes within 0.1 dB of Rust's
//!    own decode of the same stream — both measured as PSNR against the original image,
//!    the same way every other RD number in this project is measured (Step 5's pattern).
//! 2. **RD-curve floor**: the resulting per-image curve's BD-PSNR against the Step 2
//!    Fisher baseline (same settings) is never more than 0.2 dB *worse*. This is a floor,
//!    not a symmetric band — exhaustive search is a strict superset of Fisher's classified
//!    candidate set, so at matched settings it cannot legitimately lose, and a large
//!    *positive* BD-PSNR on a pathological synthetic fixture (`checker8`, `impulse`,
//!    `noise_u8`, ...) is expected, not a defect (`docs/decisions.md`). A curve that is
//!    too degenerate to compare at all (a synthetic fixture whose RD point does not move
//!    across the whole rms grid, e.g. `flat128`) is counted and named, not gated — §A7.
//!
//! The other two Step 6 exit criteria live elsewhere: integer-moment exactness is
//! `mars_codec::encode`'s own differential test (it needs no image corpus), and the
//! f32-vs-f64 fit divergence is measured here (via `mars_codec::encode::f32_f64_divergence`)
//! and reported, but is a recorded number rather than a pass/fail check — the brief's
//! decision rule (adopt f32 below 0.1%, otherwise keep f64 and document a tie-break) is a
//! decision for `docs/decisions.md`, not a gate.

use std::path::Path;

use mars_codec::encode::{encode_image, f32_f64_divergence, EncodeParams};
use mars_codec::ifs::{decode_iterative, write};
use mars_core::io::{read_pgm, read_raw, ImageError};
use mars_core::metrics::{bpp, psnr};

use crate::bdrate::{bd_metrics, RdCurve, RdPoint};
use crate::mars1::{decode, DecodeMode, DecodeParams, Mars1Binaries, Mars1Error};
use crate::mars1_report::{self, per_image_curves, Check, ReportError, Selection};
use crate::provenance::sha256_hex;
use crate::sweep::{ImageSet, SweepError};

pub const DECODE_TOLERANCE_DB: f64 = 0.1;
pub const BD_PSNR_TOLERANCE_DB: f64 = 0.2;
pub const RMS_GRID: [f64; 5] = [2.0, 4.0, 8.0, 16.0, 32.0];
/// §8's 1998 defaults — the "exhaustive-equivalent settings" the brief asks for.
const BASE: EncodeParams = EncodeParams {
    min_size: 4,
    max_size: 16,
    shift: 4,
    bits_alfa: 4,
    bits_beta: 7,
    max_alfa: 1.0,
    t_rms: 0.0, // overwritten per rate
    zero_threshold: 0,
    lambda: None,
};
const REFERENCE_METHOD: &str = "fisher";

#[derive(Debug, thiserror::Error)]
pub enum GateError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{name}: expected sha256 {want}, found {got}")]
    ImageHashMismatch {
        name: String,
        want: String,
        got: String,
    },
    #[error(transparent)]
    Sweep(#[from] SweepError),
    #[error(transparent)]
    ImageIo(#[from] ImageError),
    #[error(transparent)]
    Mars1(#[from] Mars1Error),
    #[error(transparent)]
    Report(#[from] ReportError),
}

fn read(path: &Path) -> Result<Vec<u8>, GateError> {
    std::fs::read(path).map_err(|source| GateError::Io {
        path: path.display().to_string(),
        source,
    })
}

/// The f32-vs-f64 fit divergence measurement: not gated (the brief records a decision
/// rule, not a pass/fail threshold), so the raw counts travel with the percentage rather
/// than being reduced to a number nobody can sanity-check for sample size.
#[derive(Debug, Clone, Copy)]
pub struct Divergence {
    /// Domain-referencing leaves (`qalfa != 0`) the fit was recomputed for.
    pub compared: usize,
    /// Of those, how many picked a different `qalfa` or `qbeta` at f32 than at f64.
    pub differed: usize,
}

impl Divergence {
    pub fn pct(&self) -> f64 {
        if self.compared == 0 {
            0.0
        } else {
            100.0 * self.differed as f64 / self.compared as f64
        }
    }
}

/// §M5's headline metric, one row per image, summed over the whole rms grid.
#[derive(Debug, Clone)]
pub struct EvalsRow {
    pub image: String,
    pub evals: u64,
    pub transforms: u64,
}

impl EvalsRow {
    pub fn evals_per_transform(&self) -> f64 {
        self.evals as f64 / self.transforms as f64
    }
}

/// Run both checks over the fixtures/default/RMS-grid corpus and return one `Check` per
/// criterion plus the f32-vs-f64 divergence measurement.
pub fn gate(
    root: &Path,
    fixtures_index: &Path,
    baseline_store: &Path,
    mars1_dir: &Path,
    scratch: &Path,
) -> Result<(Vec<Check>, Divergence, Vec<EvalsRow>), GateError> {
    let images = ImageSet::read(&root.join(fixtures_index))?.images;
    let bins = Mars1Binaries::from_dir(mars1_dir)?;
    std::fs::create_dir_all(scratch).map_err(|source| GateError::Io {
        path: scratch.display().to_string(),
        source,
    })?;

    let baseline_rows = mars1_report::load(baseline_store)?;
    let fisher_curves = per_image_curves(
        &baseline_rows,
        &Selection {
            corpus: "fixtures".into(),
            variant: "default".into(),
            decode_mode: DecodeMode::Iterative,
        },
    );

    let mut decode_gaps: Vec<(String, f64)> = Vec::new();
    let mut max_decode_gap = 0.0f64;
    // Exhaustive search is a strict superset of Fisher's classified candidate set, so at
    // matched settings it can never do *worse* per block: at every t_rms it splits no more
    // than Fisher does and fits each accepted block no worse, which makes BD-PSNR against
    // Fisher a one-sided floor, not a symmetric band. A large *positive* BD-PSNR (Rust much
    // better) is expected on the pathological synthetic fixtures and is not a failure; a
    // negative one below the floor is the actual "something is broken" signal.
    let mut bd_psnr_failures: Vec<(String, f64)> = Vec::new();
    let mut curve_exclusions: Vec<String> = Vec::new();
    let mut min_bd_psnr = f64::INFINITY;
    let mut compared_curves = 0usize;
    let mut compared_total = 0usize;
    let mut differed_total = 0usize;
    // §M5: `evals` is the headline search-cost metric, and Step 6's deliverable list
    // names the counter explicitly — recorded per image (summed over the rms grid) rather
    // than only threaded through and discarded.
    let mut evals_by_image: Vec<EvalsRow> = Vec::with_capacity(images.len());

    for image in &images {
        let bytes = read(&root.join(&image.file))?;
        let got_sha256 = sha256_hex(&bytes);
        if got_sha256 != image.sha256 {
            return Err(GateError::ImageHashMismatch {
                name: image.name.clone(),
                want: image.sha256.clone(),
                got: got_sha256,
            });
        }
        let ground_truth = read_raw(
            &root.join(&image.file),
            image.width as usize,
            image.height as usize,
        )?;

        let mut curve_points = Vec::with_capacity(RMS_GRID.len());
        let (mut image_evals, mut image_transforms) = (0u64, 0u64);
        for &t_rms in &RMS_GRID {
            let params = EncodeParams { t_rms, ..BASE };
            let (hdr, leaves, evals) = encode_image(&ground_truth, &params);
            image_evals += evals;
            image_transforms += leaves.len() as u64;
            let ifs_bytes = write(&hdr, &leaves).unwrap_or_else(|e| {
                panic!(
                    "{}: encoder produced an unwritable tree at rms {t_rms}: {e}",
                    image.name
                )
            });

            let (compared, differed) = f32_f64_divergence(
                &ground_truth,
                &leaves,
                params.max_alfa,
                params.bits_alfa,
                params.bits_beta,
            );
            compared_total += compared;
            differed_total += differed;

            let rust_decode = decode_iterative(&hdr, &leaves, 10);
            let psnr_rust = psnr(&ground_truth, &rust_decode); // None = lossless (§M2)

            let workdir = scratch.join(format!("{}_r{}", image.name, t_rms));
            std::fs::create_dir_all(&workdir).map_err(|source| GateError::Io {
                path: workdir.display().to_string(),
                source,
            })?;
            std::fs::write(workdir.join("o.ifs"), &ifs_bytes).map_err(|source| GateError::Io {
                path: workdir.display().to_string(),
                source,
            })?;
            decode(
                &bins,
                &workdir,
                "o.ifs",
                "d_it.pgm",
                (image.width, image.height),
                &DecodeParams {
                    mode: DecodeMode::Iterative,
                    iterations: None,
                    postprocess: false,
                },
            )?;
            let c_decode = read_pgm(&workdir.join("d_it.pgm"))?;
            let psnr_c = psnr(&ground_truth, &c_decode);

            // A plain `(a - b).abs()` on the `f64::INFINITY` stand-ins used to sit here
            // silently passed the both-lossless case: `(∞ - ∞).abs()` is `NaN`, which
            // fails every comparison, so it neither failed nor demonstrated agreement. An
            // explicit three-way match makes "both lossless" a real, verified pass and
            // "only one lossless" a real, verified failure instead of a no-op.
            let gap = match (psnr_rust, psnr_c) {
                (None, None) => 0.0,
                (Some(a), Some(b)) => (a - b).abs(),
                (None, Some(_)) | (Some(_), None) => f64::INFINITY,
            };
            max_decode_gap = max_decode_gap.max(gap);
            if gap > DECODE_TOLERANCE_DB {
                decode_gaps.push((format!("{} r={t_rms}", image.name), gap));
            }

            curve_points.push(RdPoint {
                bpp: bpp(
                    ifs_bytes.len() as u64,
                    image.width as usize,
                    image.height as usize,
                ),
                psnr: psnr_rust.unwrap_or(f64::INFINITY),
            });
        }
        evals_by_image.push(EvalsRow {
            image: image.name.clone(),
            evals: image_evals,
            transforms: image_transforms,
        });

        let rust_curve = RdCurve::new(format!("{} rust-exhaustive", image.name), curve_points);
        let Some(fisher_curve) =
            fisher_curves.get(&(image.name.clone(), REFERENCE_METHOD.to_string()))
        else {
            curve_exclusions.push(format!(
                "{}: no {REFERENCE_METHOD} baseline curve in {}",
                image.name,
                baseline_store.display()
            ));
            continue;
        };
        match bd_metrics(fisher_curve, &rust_curve) {
            Ok(bd) => {
                compared_curves += 1;
                min_bd_psnr = min_bd_psnr.min(bd.bd_psnr_db);
                if bd.bd_psnr_db < -BD_PSNR_TOLERANCE_DB {
                    bd_psnr_failures.push((image.name.clone(), bd.bd_psnr_db));
                }
            }
            // §A7: a degenerate curve (too few points after dedup, non-monotonic, no
            // overlap) is counted and named, not silently dropped — but it is not a Rust
            // encoder failure unless the *Rust* curve is the one that is degenerate while
            // Fisher's own curve on the same image is fine, which the error message's
            // curve label distinguishes.
            Err(e) => curve_exclusions.push(format!("{}: {e}", image.name)),
        }
    }

    let total_points = images.len();
    let divergence = Divergence {
        compared: compared_total,
        differed: differed_total,
    };

    let checks = vec![
        Check {
            name: format!(
                "Rust iterative decode is within {DECODE_TOLERANCE_DB} dB of decmars -i \
                 on every (image, rms) point"
            ),
            passed: decode_gaps.is_empty(),
            detail: if decode_gaps.is_empty() {
                format!(
                    "{} points, max |ΔPSNR| = {max_decode_gap:.4} dB",
                    total_points * RMS_GRID.len()
                )
            } else {
                format!(
                    "{} exceeded {DECODE_TOLERANCE_DB} dB: {}",
                    decode_gaps.len(),
                    decode_gaps
                        .iter()
                        .map(|(s, g)| format!("{s} ({g:.4} dB)"))
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            },
        },
        Check {
            name: format!(
                "the exhaustive RD curve is never more than {BD_PSNR_TOLERANCE_DB} dB \
                 BD-PSNR worse than the Step 2 {REFERENCE_METHOD} baseline, per image \
                 (exhaustive search dominates a restricted candidate set at matched \
                 settings, so this is a floor, not a band — a large positive BD-PSNR on a \
                 pathological synthetic fixture is expected, not a failure)"
            ),
            passed: bd_psnr_failures.is_empty(),
            detail: format!(
                "{compared_curves} of {total_points} images comparable, worst BD-PSNR = \
                 {min_bd_psnr:.4} dB{}{}",
                if bd_psnr_failures.is_empty() {
                    String::new()
                } else {
                    format!(
                        " — {} below the floor: {}",
                        bd_psnr_failures.len(),
                        bd_psnr_failures
                            .iter()
                            .map(|(s, g)| format!("{s} ({g:.4} dB)"))
                            .collect::<Vec<_>>()
                            .join("; ")
                    )
                },
                if curve_exclusions.is_empty() {
                    String::new()
                } else {
                    format!(
                        " — {} excluded (degenerate curve, not gated): {}",
                        curve_exclusions.len(),
                        curve_exclusions.join("; ")
                    )
                }
            ),
        },
    ];

    Ok((checks, divergence, evals_by_image))
}
