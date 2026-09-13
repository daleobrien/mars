//! Turning `results/anchors.jsonl` (plus the Step 2 Mars 1 baseline) into the Step 4
//! report: a combined Kodak RD plot and a BD-rate table against JPEG.
//!
//! Two rules carried over from `mars1_report`, because they are correct there for the
//! same reason here: BD-rate is computed **per image and then summarised** (a BD-rate
//! between two *averaged* curves is a different, less honest number), and nothing that
//! fails to compare is silently dropped — it is named (§A7).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::anchors::{AnchorCodec, AnchorRow, ROW_KIND};
use crate::bdrate::{bd_metrics, RdCurve, RdPoint};
use crate::mars1_report::Check;
use crate::store::{read_rows, StoreError};
use crate::sweep::BaselineRow;

/// §ⁱ the plan's "≥ 6 points per codec per image over roughly 0.1–2.0 bpp".
pub const TARGET_BPP_LO: f64 = 0.1;
pub const TARGET_BPP_HI: f64 = 2.0;
pub const MIN_POINTS_IN_RANGE: usize = 6;

#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("{path} row {row}: an anchors row did not deserialise: {source}")]
    Payload {
        path: String,
        row: usize,
        #[source]
        source: serde_json::Error,
    },
}

pub fn load(path: &Path) -> Result<Vec<AnchorRow>, ReportError> {
    let mut out = Vec::new();
    for (i, row) in read_rows(path)?.into_iter().enumerate() {
        if row.kind != ROW_KIND {
            continue;
        }
        out.push(
            serde_json::from_value(row.data).map_err(|source| ReportError::Payload {
                path: path.display().to_string(),
                row: i + 1,
                source,
            })?,
        );
    }
    Ok(out)
}

fn point(r: &AnchorRow) -> Option<RdPoint> {
    Some(RdPoint {
        bpp: r.quality.bpp?,
        psnr: r.quality.psnr_y?,
    })
}

/// Per-image RD curves, keyed by `(image, codec)`.
pub fn per_image_curves(rows: &[AnchorRow]) -> BTreeMap<(String, String), RdCurve> {
    let mut out: BTreeMap<(String, String), RdCurve> = BTreeMap::new();
    for r in rows {
        let Some(p) = point(r) else { continue };
        out.entry((r.image.clone(), r.codec.clone()))
            .or_insert_with(|| RdCurve::new(format!("{} {}", r.image, r.codec), Vec::new()))
            .points
            .push(p);
    }
    out
}

/// One corpus-averaged curve per codec: at each swept param, the mean bpp and mean
/// PSNR-Y over images. **For plotting only** — quoted BD-rates are per-image (below).
pub fn corpus_curves(rows: &[AnchorRow]) -> BTreeMap<String, RdCurve> {
    let mut acc: BTreeMap<(String, u64), (f64, f64, usize)> = BTreeMap::new();
    for r in rows {
        let Some(p) = point(r) else { continue };
        let e = acc
            .entry((r.codec.clone(), r.param.to_bits()))
            .or_insert((0.0, 0.0, 0));
        e.0 += p.bpp;
        e.1 += p.psnr;
        e.2 += 1;
    }
    let mut out: BTreeMap<String, RdCurve> = BTreeMap::new();
    for ((codec, _), (bpp, psnr, n)) in acc {
        out.entry(codec.clone())
            .or_insert_with(|| RdCurve::new(codec, Vec::new()))
            .points
            .push(RdPoint {
                bpp: bpp / n as f64,
                psnr: psnr / n as f64,
            });
    }
    out
}

/// BD-rate of `test` codec against `reference` codec, summarised per image (D7-style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BdSummary {
    pub reference: String,
    pub test: String,
    pub n: usize,
    pub mean_pct: f64,
    pub median_pct: f64,
    pub min_pct: f64,
    pub max_pct: f64,
    pub psnr_interval_db: (f64, f64),
    pub excluded: Vec<Exclusion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exclusion {
    pub image: String,
    pub reason: String,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

pub fn bd_summary(rows: &[AnchorRow], reference: &str, test: &str) -> BdSummary {
    let curves = per_image_curves(rows);
    let images: std::collections::BTreeSet<&String> = curves.keys().map(|(i, _)| i).collect();

    let mut pcts = Vec::new();
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut excluded = Vec::new();

    for image in images {
        let (Some(a), Some(b)) = (
            curves.get(&(image.clone(), reference.to_string())),
            curves.get(&(image.clone(), test.to_string())),
        ) else {
            excluded.push(Exclusion {
                image: image.clone(),
                reason: "one of the two codecs has no curve for this image".into(),
            });
            continue;
        };
        match bd_metrics(a, b) {
            Ok(r) => {
                pcts.push(r.bd_rate_pct);
                lo = lo.min(r.psnr_interval_db.0);
                hi = hi.max(r.psnr_interval_db.1);
            }
            Err(e) => excluded.push(Exclusion {
                image: image.clone(),
                reason: e.to_string(),
            }),
        }
    }

    let n = pcts.len();
    BdSummary {
        reference: reference.into(),
        test: test.into(),
        n,
        mean_pct: if n == 0 {
            f64::NAN
        } else {
            pcts.iter().sum::<f64>() / n as f64
        },
        median_pct: median(pcts.clone()),
        min_pct: pcts.iter().copied().fold(f64::INFINITY, f64::min),
        max_pct: pcts.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        psnr_interval_db: (lo, hi),
        excluded,
    }
}

/// Every anchor codec (bar the reference) against `reference`.
pub fn bd_matrix(rows: &[AnchorRow], reference: &str) -> Vec<BdSummary> {
    let mut codecs: Vec<String> = rows.iter().map(|r| r.codec.clone()).collect();
    codecs.sort();
    codecs.dedup();
    codecs
        .into_iter()
        .filter(|c| c != reference)
        .map(|c| bd_summary(rows, reference, &c))
        .collect()
}

/// Count of `(bpp in range)` points per `(image, codec)`, for the ≥6-points-in-range gate.
pub fn points_in_range(rows: &[AnchorRow], lo: f64, hi: f64) -> BTreeMap<(String, String), usize> {
    let mut out: BTreeMap<(String, String), usize> = BTreeMap::new();
    for r in rows {
        let Some(bpp) = r.quality.bpp else { continue };
        let key = (r.image.clone(), r.codec.clone());
        let e = out.entry(key).or_insert(0);
        if bpp >= lo && bpp <= hi {
            *e += 1;
        }
    }
    out
}

/// Every combined RD curve for the Step 4 plot: one per anchor codec, plus one per Mars 1
/// method from the Step 2 baseline (labelled `mars1-<method>` so the legend disambiguates
/// them from any future Mars 2 curve on the same axes).
pub fn combined_plot_curves(anchor_rows: &[AnchorRow], mars1_rows: &[BaselineRow]) -> Vec<RdCurve> {
    let mut curves: Vec<RdCurve> = corpus_curves(anchor_rows).into_values().collect();

    let sel = crate::mars1_report::Selection {
        corpus: "standard".into(),
        variant: "default".into(),
        decode_mode: crate::mars1::DecodeMode::Pyramidal,
    };
    for (method, mut curve) in crate::mars1_report::corpus_curves(mars1_rows, &sel) {
        curve.label = format!("mars1-{method}");
        curves.push(curve);
    }
    curves.sort_by(|a, b| a.label.cmp(&b.label));
    curves
}

// --------------------------------------------------------------------------- the gate

fn check(name: &str, passed: bool, detail: impl Into<String>) -> Check {
    Check {
        name: name.into(),
        passed,
        detail: detail.into(),
    }
}

/// Step 4's exit criteria, evaluated against the store (§A1: a command that exits 0 or 1).
pub fn gate(
    rows: &[AnchorRow],
    expected_codecs: &[String],
    reference: &str,
    report_files: &[(&str, &Path)],
) -> Vec<Check> {
    let mut out = Vec::new();

    let images: std::collections::BTreeSet<&str> = rows.iter().map(|r| r.image.as_str()).collect();
    out.push(check(
        "corpus is non-empty",
        !images.is_empty(),
        format!("{} images", images.len()),
    ));

    // --- all five codecs present --------------------------------------------
    let mut have: Vec<&str> = rows.iter().map(|r| r.codec.as_str()).collect();
    have.sort_unstable();
    have.dedup();
    let missing: Vec<&String> = expected_codecs
        .iter()
        .filter(|c| !have.contains(&c.as_str()))
        .collect();
    out.push(check(
        "all five anchor codecs are present",
        missing.is_empty(),
        if missing.is_empty() {
            format!("{} codecs: {}", have.len(), have.join(", "))
        } else {
            format!("missing: {missing:?}")
        },
    ));

    // --- >= 6 points per (codec, image) inside the target bpp range ---------
    let counts = points_in_range(rows, TARGET_BPP_LO, TARGET_BPP_HI);
    let short: Vec<String> = counts
        .iter()
        .filter(|(_, &n)| n < MIN_POINTS_IN_RANGE)
        .map(|((img, codec), n)| format!("{img}/{codec}: {n} points"))
        .collect();
    let complete_pairs = images.len() * expected_codecs.len();
    out.push(check(
        &format!(
            "every (codec, image) has >= {MIN_POINTS_IN_RANGE} points in {TARGET_BPP_LO}-{TARGET_BPP_HI} bpp"
        ),
        short.is_empty() && counts.len() == complete_pairs,
        if short.is_empty() && counts.len() == complete_pairs {
            format!(
                "{} (codec, image) pairs, all >= {MIN_POINTS_IN_RANGE} points in range",
                counts.len()
            )
        } else if counts.len() != complete_pairs {
            format!(
                "expected {complete_pairs} (codec, image) pairs, found {}",
                counts.len()
            )
        } else {
            short.join("; ")
        },
    ));

    // --- BD-rate vs JPEG is defined and finite for every other anchor -------
    let mut bd_ok = true;
    let mut bd_detail = Vec::new();
    for c in expected_codecs.iter().filter(|c| c.as_str() != reference) {
        let s = bd_summary(rows, reference, c);
        let finite = s.mean_pct.is_finite();
        if !finite || !s.excluded.is_empty() || s.n != images.len() {
            bd_ok = false;
        }
        bd_detail.push(format!(
            "{c}: {:+.2}% over {}/{} images{}",
            s.mean_pct,
            s.n,
            images.len(),
            if s.excluded.is_empty() {
                String::new()
            } else {
                format!(
                    " -- EXCLUDED {}",
                    s.excluded
                        .iter()
                        .map(|e| format!("{} ({})", e.image, e.reason))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        ));
    }
    out.push(check(
        "BD-rate against JPEG is defined and finite for every other anchor, on every image",
        bd_ok,
        bd_detail.join(" | "),
    ));

    // --- output files present ------------------------------------------------
    let missing_files: Vec<&str> = report_files
        .iter()
        .filter(|(_, p)| !p.is_file())
        .map(|(name, _)| *name)
        .collect();
    out.push(check(
        "report output files exist",
        missing_files.is_empty(),
        if missing_files.is_empty() {
            report_files
                .iter()
                .map(|(n, _)| *n)
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            format!("missing: {}", missing_files.join(", "))
        },
    ));

    out
}

#[allow(dead_code)]
fn _doc_anchor(_c: AnchorCodec) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::measure::{Definitions, Measurement};

    fn row(image: &str, codec: &str, bpp: f64, psnr: f64) -> AnchorRow {
        AnchorRow {
            sweep: "t".into(),
            corpus: "standard".into(),
            image: image.into(),
            image_sha256: "0".repeat(64),
            codec: codec.into(),
            param: bpp * 100.0,
            param_kind: "quality".into(),
            coded_bytes: 1,
            quality: Measurement {
                original: "o".into(),
                decoded: "d".into(),
                width: 768,
                height: 512,
                planes: 3,
                mse: vec![1.0],
                psnr: vec![Some(psnr)],
                psnr_y: Some(psnr),
                psnr_cb: Some(psnr),
                psnr_cr: Some(psnr),
                psnr_yuv: Some(psnr),
                ssim: None,
                ssim_unavailable: None,
                ms_ssim: None,
                ms_ssim_unavailable: None,
                coded_bytes: Some(1),
                bpp: Some(bpp),
                definitions: Definitions {
                    psnr: String::new(),
                    ssim: String::new(),
                    ms_ssim: String::new(),
                    bpp: String::new(),
                },
            },
            indicative_encode_seconds: 0.0,
            indicative_decode_seconds: 0.0,
            timing_protocol: crate::anchors::TIMING_PROTOCOL.to_string(),
            encode_cmd: String::new(),
            decode_cmd: String::new(),
            codec_build_info: String::new(),
            jobs: 1,
        }
    }

    fn corpus() -> Vec<AnchorRow> {
        let mut rows = Vec::new();
        for image in ["a", "b"] {
            for (i, bpp) in [0.2, 0.4, 0.8, 1.2, 1.6].iter().enumerate() {
                rows.push(row(image, "jpeg", *bpp, 30.0 + i as f64));
                // avif: same quality, fewer bits -> negative BD-rate vs jpeg.
                rows.push(row(image, "avif", bpp * 0.8, 30.0 + i as f64));
            }
        }
        rows
    }

    #[test]
    fn per_image_curves_are_kept_separate() {
        let c = per_image_curves(&corpus());
        assert_eq!(c.len(), 4); // 2 images x 2 codecs
        assert_eq!(c[&("a".into(), "jpeg".into())].points.len(), 5);
    }

    #[test]
    fn a_20_percent_rate_cut_reads_as_about_minus_20_percent_bd_rate() {
        let s = bd_summary(&corpus(), "jpeg", "avif");
        assert_eq!(s.n, 2);
        assert!(s.excluded.is_empty());
        assert!(
            (s.mean_pct - (-20.0)).abs() < 1e-6,
            "mean BD-rate was {}",
            s.mean_pct
        );
    }

    #[test]
    fn points_in_range_counts_correctly() {
        let counts = points_in_range(&corpus(), 0.1, 1.0);
        // jpeg on image a: 0.2, 0.4, 0.8 in range (3); 1.2, 1.6 out.
        assert_eq!(counts[&("a".into(), "jpeg".into())], 3);
    }

    #[test]
    fn gate_fails_when_a_codec_is_missing() {
        let rows = corpus();
        let checks = gate(
            &rows,
            &["jpeg".into(), "avif".into(), "webp".into()],
            "jpeg",
            &[],
        );
        let codecs_check = checks
            .iter()
            .find(|c| c.name.contains("five anchor codecs"))
            .unwrap();
        assert!(!codecs_check.passed);
    }
}
