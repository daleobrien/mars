//! Turning `results/baseline-mars1.jsonl` into the Step 2 report.
//!
//! Two aggregation rules are pinned here, because both are places where compression
//! results habitually go wrong:
//!
//! **Curves are per-image; the corpus curve is for plotting.** A BD-rate computed between
//! two *averaged* curves is not the average of the per-image BD-rates, and on a corpus
//! with a wide difficulty spread the two can differ by several percent. So every quoted
//! BD-rate here is computed **per image and then summarised**, with the spread reported
//! alongside the mean. The averaged curve exists so there is something to plot and is
//! labelled as such (`docs/decisions.md` D7).
//!
//! **Nothing is dropped silently.** An image whose curve is non-monotonic, or whose two
//! curves do not overlap in PSNR, is *counted and named* rather than omitted — §A7, and
//! the specific failure that a quietly excluded hard image makes a method look better
//! than it is.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::bdrate::{bd_metrics, BdResult, RdCurve, RdPoint};
use crate::mars1::DecodeMode;
use crate::store::{read_rows, StoreError};
use crate::sweep::{BaselineRow, ROW_KIND};

/// Which slice of the sweep a table or curve is about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub corpus: String,
    pub variant: String,
    pub decode_mode: DecodeMode,
}

/// Read every `baseline-mars1` row. Rows of other kinds in the same store are skipped;
/// a row of *this* kind that will not deserialise is an error, not a skip.
pub fn load(path: &Path) -> Result<Vec<BaselineRow>, ReportError> {
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

#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("{path} row {row}: a baseline-mars1 row did not deserialise: {source}")]
    Payload {
        path: String,
        row: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("no rows matched corpus={corpus} variant={variant} decode={decode}")]
    EmptySelection {
        corpus: String,
        variant: String,
        decode: &'static str,
    },
    #[error("reference method {0:?} is not present in the selected rows")]
    MissingReference(String),
}

fn matches(r: &BaselineRow, sel: &Selection) -> bool {
    r.corpus == sel.corpus && r.variant == sel.variant && r.decode_mode == sel.decode_mode
}

/// `(bpp, PSNR-Y)` for one row, or `None` when PSNR is infinite (a lossless decode) and
/// the point cannot enter an RD curve.
fn point(r: &BaselineRow) -> Option<RdPoint> {
    Some(RdPoint {
        bpp: r.quality.bpp?,
        psnr: r.quality.psnr_y?,
    })
}

/// Per-image RD curves, keyed by `(image, method)`. Points are ordered by increasing bpp
/// by `RdCurve::prepared`, so the rate ordering here does not matter.
pub fn per_image_curves(
    rows: &[BaselineRow],
    sel: &Selection,
) -> BTreeMap<(String, String), RdCurve> {
    let mut out: BTreeMap<(String, String), RdCurve> = BTreeMap::new();
    for r in rows.iter().filter(|r| matches(r, sel)) {
        let Some(p) = point(r) else { continue };
        out.entry((r.image.clone(), r.method.clone()))
            .or_insert_with(|| RdCurve::new(format!("{} {}", r.image, r.method), Vec::new()))
            .points
            .push(p);
    }
    out
}

/// One corpus-averaged curve per method: at each rate, the arithmetic mean of bpp and of
/// PSNR-Y over the images, equally weighted. **For plotting**; quoted BD-rates come from
/// [`bd_summary`].
pub fn corpus_curves(rows: &[BaselineRow], sel: &Selection) -> BTreeMap<String, RdCurve> {
    // (method, rate-as-bits) -> running sums. The rate is keyed by its bit pattern so
    // grouping never depends on float formatting.
    let mut acc: BTreeMap<(String, u64), (f64, f64, usize)> = BTreeMap::new();
    for r in rows.iter().filter(|r| matches(r, sel)) {
        let Some(p) = point(r) else { continue };
        let e = acc
            .entry((r.method.clone(), r.t_rms.to_bits()))
            .or_insert((0.0, 0.0, 0));
        e.0 += p.bpp;
        e.1 += p.psnr;
        e.2 += 1;
    }
    let mut out: BTreeMap<String, RdCurve> = BTreeMap::new();
    for ((method, _), (bpp, psnr, n)) in acc {
        out.entry(method.clone())
            .or_insert_with(|| RdCurve::new(method, Vec::new()))
            .points
            .push(RdPoint {
                bpp: bpp / n as f64,
                psnr: psnr / n as f64,
            });
    }
    out
}

/// BD-rate of one method against another, computed per image and then summarised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BdSummary {
    pub reference: String,
    pub test: String,
    /// Images for which BD-rate was defined.
    pub n: usize,
    pub mean_pct: f64,
    pub median_pct: f64,
    pub min_pct: f64,
    pub max_pct: f64,
    /// The union of the per-image PSNR intervals, so §M3's "state the interval" is met
    /// even for a summarised number.
    pub psnr_interval_db: (f64, f64),
    /// Named, never silently dropped (§A7).
    pub excluded: Vec<Exclusion>,
    /// The same comparison on the corpus-averaged curves. Reported for contrast, not as
    /// the headline; see the module docs.
    pub averaged_curve_pct: Option<f64>,
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

/// Summarise `test` against `reference` over every image that has both curves.
pub fn bd_summary(rows: &[BaselineRow], sel: &Selection, reference: &str, test: &str) -> BdSummary {
    let curves = per_image_curves(rows, sel);
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
                reason: "one of the two methods has no curve for this image".into(),
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

    let averaged = {
        let c = corpus_curves(rows, sel);
        match (c.get(reference), c.get(test)) {
            (Some(a), Some(b)) => bd_metrics(a, b).ok().map(|r| r.bd_rate_pct),
            _ => None,
        }
    };

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
        averaged_curve_pct: averaged,
    }
}

/// Running totals while building one [`EvalsRow`]: comparisons, transforms, the min and
/// max per-encode ratio, summed PSNR, summed bpp, and the encode count.
type EvalsAcc = (u64, u64, f64, f64, f64, f64, f64, usize);

/// PSNR-Y of one encode under each decode mode, while pairing them up.
type ModePair = (Option<f64>, Option<f64>);

/// The first row of the project's headline metric (§M5), per method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalsRow {
    pub method: String,
    /// Total `comparisons` over total `transforms` across the selection — a
    /// work-weighted ratio, not a mean of per-image ratios, so it answers "what did the
    /// whole corpus cost per emitted transform".
    pub evals_per_transform: f64,
    pub total_comparisons: u64,
    pub total_transforms: u64,
    /// Range of the per-`(image, rate)` ratio, which the aggregate hides.
    pub min_evals_per_transform: f64,
    pub max_evals_per_transform: f64,
    pub mean_psnr_db: f64,
    pub mean_bpp: f64,
    /// Indicative only (D8) — but see D10: reported beside `evals_per_transform` because
    /// the two disagree about which method is cheaper whenever an index structure does
    /// work the eval counter cannot see.
    pub mean_encode_seconds: f64,
    /// `mean_encode_seconds` over evals. Near-constant across methods that share a
    /// per-eval cost; inflated for methods that spend time in an uncounted index.
    pub us_per_eval: f64,
    pub n_encodes: usize,
}

/// `evals/transform` per method over a selection.
///
/// Rows are deduplicated by `(image, rate)`: the same encode appears once per decode
/// mode, and counting its `comparisons` twice would halve the apparent efficiency of
/// nothing in particular.
pub fn evals_table(rows: &[BaselineRow], sel: &Selection) -> Vec<EvalsRow> {
    let mut acc: BTreeMap<String, EvalsAcc> = BTreeMap::new();
    for r in rows.iter().filter(|r| matches(r, sel)) {
        let ratio = r.encode.evals_per_transform;
        let e = acc.entry(r.method.clone()).or_insert((
            0,
            0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            0,
        ));
        e.0 += r.encode.comparisons;
        e.1 += r.encode.transforms;
        e.2 = e.2.min(ratio);
        e.3 = e.3.max(ratio);
        e.4 += r.quality.psnr_y.unwrap_or(f64::NAN);
        e.5 += r.quality.bpp.unwrap_or(f64::NAN);
        e.6 += r.indicative_encode_seconds;
        e.7 += 1;
    }
    acc.into_iter()
        .map(
            |(method, (comparisons, transforms, min, max, psnr, bpp, secs, n))| EvalsRow {
                method,
                evals_per_transform: comparisons as f64 / transforms as f64,
                total_comparisons: comparisons,
                total_transforms: transforms,
                min_evals_per_transform: min,
                max_evals_per_transform: max,
                mean_psnr_db: psnr / n as f64,
                mean_bpp: bpp / n as f64,
                mean_encode_seconds: secs / n as f64,
                us_per_eval: secs * 1e6 / comparisons as f64,
                n_encodes: n,
            },
        )
        .collect()
}

/// Mean PSNR-Y difference (iterative − pyramidal) at matched settings, and the extremes.
/// This is P2.2's measurement: if the two modes differ, every row must state which it is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeModeDelta {
    pub n: usize,
    pub mean_db: f64,
    pub min_db: f64,
    pub max_db: f64,
    /// Settings at which the *ordering* of two methods flips between modes, which is the
    /// concrete way a decode-mode mixup corrupts a comparison.
    pub mean_pyramidal_db: f64,
    pub mean_iterative_db: f64,
}

pub fn decode_mode_delta(rows: &[BaselineRow], corpus: &str, variant: &str) -> DecodeModeDelta {
    // Key on everything that identifies an encode, so the two modes are compared at
    // genuinely matched settings.
    let mut pairs: BTreeMap<(String, String, u64), ModePair> = BTreeMap::new();
    for r in rows
        .iter()
        .filter(|r| r.corpus == corpus && r.variant == variant)
    {
        let key = (r.image.clone(), r.method.clone(), r.t_rms.to_bits());
        let e = pairs.entry(key).or_insert((None, None));
        match r.decode_mode {
            DecodeMode::Pyramidal => e.0 = r.quality.psnr_y,
            DecodeMode::Iterative => e.1 = r.quality.psnr_y,
        }
    }
    let mut deltas = Vec::new();
    let (mut sp, mut si) = (0.0, 0.0);
    for (p, i) in pairs.into_values() {
        if let (Some(p), Some(i)) = (p, i) {
            deltas.push(i - p);
            sp += p;
            si += i;
        }
    }
    let n = deltas.len();
    DecodeModeDelta {
        n,
        mean_db: deltas.iter().sum::<f64>() / n as f64,
        min_db: deltas.iter().copied().fold(f64::INFINITY, f64::min),
        max_db: deltas.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean_pyramidal_db: sp / n as f64,
        mean_iterative_db: si / n as f64,
    }
}

/// BD-rate of a structural variant against `default`, for one method, per image.
pub fn variant_summary(
    rows: &[BaselineRow],
    corpus: &str,
    decode_mode: DecodeMode,
    method: &str,
    variant: &str,
) -> BdSummary {
    let base_sel = Selection {
        corpus: corpus.into(),
        variant: "default".into(),
        decode_mode,
    };
    let var_sel = Selection {
        corpus: corpus.into(),
        variant: variant.into(),
        decode_mode,
    };
    // Re-label both sides into one namespace so `bd_summary` can compare them: the
    // reference is the default variant's curve, the test is this variant's.
    let mut merged: Vec<BaselineRow> = Vec::new();
    for r in rows.iter().filter(|r| r.method == method) {
        if matches(r, &base_sel) {
            let mut r = r.clone();
            r.method = "default".into();
            r.variant = "merged".into();
            merged.push(r);
        } else if matches(r, &var_sel) {
            let mut r = r.clone();
            r.method = variant.to_string();
            r.variant = "merged".into();
            merged.push(r);
        }
    }
    let sel = Selection {
        corpus: corpus.into(),
        variant: "merged".into(),
        decode_mode,
    };
    bd_summary(&merged, &sel, "default", variant)
}

/// A BD-rate matrix: every method against `reference`.
pub fn bd_matrix(rows: &[BaselineRow], sel: &Selection, reference: &str) -> Vec<BdSummary> {
    let mut methods: Vec<String> = rows
        .iter()
        .filter(|r| matches(r, sel))
        .map(|r| r.method.clone())
        .collect();
    methods.sort();
    methods.dedup();
    methods
        .into_iter()
        .filter(|m| m != reference)
        .map(|m| bd_summary(rows, sel, reference, &m))
        .collect()
}

/// Full BD detail for the corpus-averaged curves, used where §M3's interval has to be
/// printed in full.
pub fn averaged_bd(
    rows: &[BaselineRow],
    sel: &Selection,
    reference: &str,
    test: &str,
) -> Option<BdResult> {
    let c = corpus_curves(rows, sel);
    bd_metrics(c.get(reference)?, c.get(test)?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mars1::{EncodeParams, Method};
    use crate::measure::{Definitions, Measurement};
    use crate::sweep::EncodeRow;

    fn row(image: &str, method: Method, t_rms: f64, bpp: f64, psnr: f64) -> BaselineRow {
        BaselineRow {
            sweep: "t".into(),
            corpus: "standard".into(),
            image: image.into(),
            image_sha256: "0".repeat(64),
            width: 768,
            height: 512,
            variant: "default".into(),
            t_rms,
            method: method.key().into(),
            params: EncodeParams {
                method,
                t_rms,
                ..Default::default()
            },
            encode: EncodeRow {
                transforms: 1000,
                comparisons: (100.0 * t_rms) as u64 * 1000,
                evals_per_transform: 100.0 * t_rms,
                zero_alfa_transforms: 0,
                coded_bytes: 1,
                image_entropy: 7.0,
                image_variance: 100.0,
            },
            decode_mode: DecodeMode::Pyramidal,
            decode_iterations: crate::mars1::DEFAULT_ITERATIONS,
            decode_postprocess: false,
            quality: Measurement {
                original: "o".into(),
                decoded: "d".into(),
                width: 768,
                height: 512,
                planes: 1,
                mse: vec![1.0],
                psnr: vec![Some(psnr)],
                psnr_y: Some(psnr),
                psnr_cb: None,
                psnr_cr: None,
                psnr_yuv: None,
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
            timing_protocol: crate::sweep::TIMING_PROTOCOL.to_string(),
            jobs: 1,
            mars1_build_info: None,
        }
    }

    /// Two images, two methods, four rates. Image "b" is uniformly 2 dB worse, so the
    /// two methods' *relative* standing is identical on both images.
    fn corpus() -> Vec<BaselineRow> {
        let mut rows = Vec::new();
        for (image, offset) in [("a", 0.0), ("b", -2.0)] {
            for (i, &r) in [2.0f64, 4.0, 8.0, 16.0].iter().enumerate() {
                let bpp = 1.6 - 0.3 * i as f64;
                rows.push(row(image, Method::Fisher, r, bpp, 34.0 - i as f64 + offset));
                // Saupe: same quality, more bits -> a positive BD-rate against Fisher.
                rows.push(row(
                    image,
                    Method::Saupe,
                    r,
                    bpp * 1.1,
                    34.0 - i as f64 + offset,
                ));
            }
        }
        rows
    }

    fn sel() -> Selection {
        Selection {
            corpus: "standard".into(),
            variant: "default".into(),
            decode_mode: DecodeMode::Pyramidal,
        }
    }

    #[test]
    fn per_image_curves_are_kept_separate() {
        let c = per_image_curves(&corpus(), &sel());
        assert_eq!(c.len(), 4); // 2 images x 2 methods
        assert_eq!(c[&("a".into(), "fisher".into())].points.len(), 4);
    }

    #[test]
    fn a_10_percent_rate_increase_reads_as_about_10_percent_bd_rate() {
        let s = bd_summary(&corpus(), &sel(), "fisher", "saupe");
        assert_eq!(s.n, 2);
        assert!(s.excluded.is_empty());
        // Exactly 10% more bits at every matched quality.
        assert!(
            (s.mean_pct - 10.0).abs() < 1e-6,
            "mean BD-rate was {}",
            s.mean_pct
        );
        assert!((s.min_pct - 10.0).abs() < 1e-6 && (s.max_pct - 10.0).abs() < 1e-6);
        // §M3: the interval travels with the number.
        assert!(s.psnr_interval_db.0.is_finite() && s.psnr_interval_db.1.is_finite());
    }

    #[test]
    fn an_unusable_curve_is_named_not_dropped() {
        let mut rows = corpus();
        // Make image "a"'s Fisher curve non-monotonic: more bits, less quality.
        for r in rows.iter_mut() {
            if r.image == "a" && r.method == "fisher" && r.t_rms == 2.0 {
                r.quality.psnr_y = Some(1.0);
            }
        }
        let s = bd_summary(&rows, &sel(), "fisher", "saupe");
        assert_eq!(s.n, 1, "the good image still counts");
        assert_eq!(s.excluded.len(), 1);
        assert_eq!(s.excluded[0].image, "a");
        assert!(
            s.excluded[0].reason.contains("monotonic"),
            "reason was {:?}",
            s.excluded[0].reason
        );
    }

    #[test]
    fn the_averaged_curve_is_reported_separately_from_the_per_image_summary() {
        let s = bd_summary(&corpus(), &sel(), "fisher", "saupe");
        // Here the corpus is homogeneous by construction, so the two agree; the point of
        // the assertion is that both numbers exist and are distinguishable in the output.
        let avg = s.averaged_curve_pct.expect("averaged curve BD-rate");
        assert!((avg - s.mean_pct).abs() < 1e-6);
    }

    #[test]
    fn evals_per_transform_is_work_weighted_and_deduplicated() {
        let mut rows = corpus();
        // The same encode appears once per decode mode; the table must not count it twice.
        let mut iterative: Vec<BaselineRow> = rows
            .iter()
            .cloned()
            .map(|mut r| {
                r.decode_mode = DecodeMode::Iterative;
                r
            })
            .collect();
        rows.append(&mut iterative);

        let t = evals_table(&rows, &sel());
        let fisher = t.iter().find(|e| e.method == "fisher").unwrap();
        assert_eq!(fisher.n_encodes, 8); // 2 images x 4 rates, pyramidal rows only
                                         // comparisons = 100*r*1000 per row, transforms = 1000 per row.
        let want = 2.0 * (200.0 + 400.0 + 800.0 + 1600.0) * 1000.0 / (8.0 * 1000.0);
        assert!((fisher.evals_per_transform - want).abs() < 1e-9);
        assert_eq!(fisher.min_evals_per_transform, 200.0);
        assert_eq!(fisher.max_evals_per_transform, 1600.0);
    }

    #[test]
    fn the_decode_mode_delta_pairs_matched_settings_only() {
        let mut rows = corpus();
        let iterative: Vec<BaselineRow> = rows
            .iter()
            .cloned()
            .map(|mut r| {
                r.decode_mode = DecodeMode::Iterative;
                r.quality.psnr_y = r.quality.psnr_y.map(|p| p + 0.5);
                r
            })
            .collect();
        rows.extend(iterative);
        let d = decode_mode_delta(&rows, "standard", "default");
        assert_eq!(d.n, 16); // 2 images x 2 methods x 4 rates
        assert!((d.mean_db - 0.5).abs() < 1e-9);
        assert!((d.mean_iterative_db - d.mean_pyramidal_db - 0.5).abs() < 1e-9);
    }
}

// ------------------------------------------------------------------------ rendering

/// The Step 2 report: everything the step's exit criteria ask for, in one document.
pub struct Mars1Report {
    pub rows: Vec<BaselineRow>,
    /// The corpus carrying the headline numbers (§M9: `standard`).
    pub corpus: String,
    /// The variant carrying them (`default` — the 1998 settings).
    pub variant: String,
    /// The decode mode carrying them. Stated everywhere, never assumed.
    pub decode_mode: DecodeMode,
    /// BD-rate reference method. Fisher, because it is the classification method the
    /// fractal-coding literature treats as the standard point of comparison.
    pub reference_method: String,
    /// Structural variants to report against `default`.
    pub variants: Vec<String>,
    pub provenance_note: String,
}

fn f(v: f64, p: usize) -> String {
    if v.is_finite() {
        format!("{v:.p$}")
    } else {
        "—".into()
    }
}

impl Mars1Report {
    pub fn headline(&self) -> Selection {
        Selection {
            corpus: self.corpus.clone(),
            variant: self.variant.clone(),
            decode_mode: self.decode_mode,
        }
    }

    /// The corpus-averaged curves, for plotting.
    pub fn plot_curves(&self) -> Vec<RdCurve> {
        corpus_curves(&self.rows, &self.headline())
            .into_values()
            .collect()
    }

    pub fn to_markdown(&self) -> String {
        use std::fmt::Write as _;
        let sel = self.headline();
        let mut s = String::new();

        let images: std::collections::BTreeSet<&str> = self
            .rows
            .iter()
            .filter(|r| r.corpus == self.corpus)
            .map(|r| r.image.as_str())
            .collect();

        let _ = writeln!(s, "# Mars 1 baseline — Step 2\n");
        let _ = writeln!(s, "{}\n", self.provenance_note);
        let _ = writeln!(
            s,
            "The project's first real data, on the 1998 C codec, before any Mars 2 codec \
             code exists. Every quality number here was computed by `mars-bench` from two \
             files on disk (§M1); the codec reports only its own byte count, and even that \
             is cross-checked against the file size.\n"
        );
        let _ = writeln!(
            s,
            "**Headline slice:** corpus `{}` ({} images) · variant `{}` · **{} decode**.\n",
            self.corpus,
            images.len(),
            self.variant,
            self.decode_mode.key()
        );
        let _ = writeln!(
            s,
            "> Decode mode **and iteration count** are stated on every number here and on \
             every row in the store. Pyramidal with {} iterations is the 1998 default. \
             Contrary to the warning in the plan — and to this project's own prediction \
             P2.2 — the *mode* is worth almost nothing at that count (§4 below measures \
             it), because both decoders converge to the same IFS fixed point. The \
             **iteration count** is the variable that matters: the same bitstream decodes \
             6.1 dB apart at 1 iteration and 0.004 dB apart at 10. See `docs/decisions.md` \
             D9.\n",
            crate::mars1::DEFAULT_ITERATIONS
        );

        // --- 1. evals/transform: the headline metric ---------------------------
        let _ = writeln!(s, "## 1. `evals/transform` — the headline metric (§M5)\n");
        let _ = writeln!(
            s,
            "`comparisons` is Mars 1's `evals` analogue, incremented at six sites in \
             `coding_func.c`, one per method. The ratio is work-weighted: total \
             comparisons over total transforms across all {} images × 5 rates, not a mean \
             of per-encode ratios.\n",
            images.len()
        );
        let _ = writeln!(
            s,
            "> **`µs/eval` is here to show what the metric does not count** (D10). An eval \
             is the same affine fit in every method, so a method whose µs/eval is far above \
             the others is spending its time in an index structure the counter cannot see. \
             Those seconds are indicative only (D8); the ratio between them is the point, \
             not their absolute value.\n"
        );
        let _ = writeln!(
            s,
            "| method | evals/transform | per-encode min | per-encode max | total evals | mean PSNR-Y (dB) | mean bpp | mean encode s | µs/eval |"
        );
        let _ = writeln!(s, "|---|---:|---:|---:|---:|---:|---:|---:|---:|");
        let mut table = evals_table(&self.rows, &sel);
        table.sort_by(|a, b| {
            a.evals_per_transform
                .partial_cmp(&b.evals_per_transform)
                .expect("finite")
        });
        for e in &table {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                e.method,
                f(e.evals_per_transform, 1),
                f(e.min_evals_per_transform, 1),
                f(e.max_evals_per_transform, 1),
                e.total_comparisons,
                f(e.mean_psnr_db, 3),
                f(e.mean_bpp, 4),
                f(e.mean_encode_seconds, 2),
                f(e.us_per_eval, 3),
            );
        }
        let _ = writeln!(s);

        // --- 2. RD curves ------------------------------------------------------
        let _ = writeln!(s, "## 2. Rate–distortion, six methods\n");
        let _ = writeln!(
            s,
            "Corpus-averaged operating points: at each `-r`, the arithmetic mean of bpp \
             and of PSNR-Y over the {} images, equally weighted. **These averaged curves \
             are for plotting.** Every BD-rate quoted below is computed per image and then \
             summarised (D7).\n",
            images.len()
        );
        let curves = corpus_curves(&self.rows, &sel);
        let mut rates: Vec<f64> = self
            .rows
            .iter()
            .filter(|r| matches(r, &sel))
            .map(|r| r.t_rms)
            .collect();
        rates.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        rates.dedup();

        let _ = write!(s, "| method |");
        for r in &rates {
            let _ = write!(s, " -r {} |", f(*r, 0));
        }
        let _ = writeln!(s, "\n|---|{}", "---:|".repeat(rates.len()));
        for (label, c) in &curves {
            let mut pts = c.points.clone();
            pts.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).expect("finite"));
            // Descending bpp is ascending -r, so the row reads left to right as the table
            // header does.
            pts.reverse();
            let _ = write!(s, "| {label} |");
            for p in &pts {
                let _ = write!(s, " {} dB @ {} bpp |", f(p.psnr, 2), f(p.bpp, 3));
            }
            let _ = writeln!(s);
        }
        let _ = writeln!(s);

        // --- 3. BD-rate between methods ---------------------------------------
        let _ = writeln!(
            s,
            "## 3. BD-rate between methods (reference: `{}`)\n",
            self.reference_method
        );
        let _ = writeln!(
            s,
            "Positive means the method needs **more** bits than `{}` for the same quality. \
             Computed per image over the PSNR overlap, then summarised across images; `n` \
             is how many of the {} images produced a defined BD-rate, and anything that did \
             not is named in the exclusions column rather than dropped (§A7).\n",
            self.reference_method,
            images.len()
        );
        let _ = writeln!(
            s,
            "| test | n | mean % | median % | min % | max % | PSNR interval (dB) | averaged-curve % | excluded |"
        );
        let _ = writeln!(s, "|---|---:|---:|---:|---:|---:|---|---:|---|");
        for b in bd_matrix(&self.rows, &sel, &self.reference_method) {
            let excl = if b.excluded.is_empty() {
                "none".to_string()
            } else {
                b.excluded
                    .iter()
                    .map(|e| format!("{} ({})", e.image, e.reason))
                    .collect::<Vec<_>>()
                    .join("; ")
            };
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} | {} | {}–{} | {} | {} |",
                b.test,
                b.n,
                f(b.mean_pct, 2),
                f(b.median_pct, 2),
                f(b.min_pct, 2),
                f(b.max_pct, 2),
                f(b.psnr_interval_db.0, 2),
                f(b.psnr_interval_db.1, 2),
                b.averaged_curve_pct.map_or("—".into(), |v| f(v, 2)),
                excl,
            );
        }
        let _ = writeln!(s);

        // --- 4. decode modes ---------------------------------------------------
        let _ = writeln!(s, "## 4. Pyramidal vs iterative decode\n");
        let _ = writeln!(
            s,
            "The same bitstream through both decoders, at {} iterations. Reported per \
             variant, because the answer is not the same for all of them.\n",
            crate::mars1::DEFAULT_ITERATIONS
        );
        let _ = writeln!(
            s,
            "| variant | n | mean iterative − pyramidal (dB) | min | max |"
        );
        let _ = writeln!(s, "|---|---:|---:|---:|---:|");
        let mut vs: Vec<String> = vec![self.variant.clone()];
        vs.extend(self.variants.iter().cloned());
        for v in &vs {
            let d = decode_mode_delta(&self.rows, &self.corpus, v);
            let _ = writeln!(
                s,
                "| {}{} | {} | {} | {} | {} |",
                v,
                if v == &self.variant {
                    " (headline)"
                } else {
                    ""
                },
                d.n,
                f(d.mean_db, 4),
                f(d.min_db, 4),
                f(d.max_db, 4),
            );
        }
        let _ = writeln!(
            s,
            "\n**`min2` is the exception, and it is not a small one.** Every other variant \
             agrees to ~0.002 dB, because an IFS has a unique attracting fixed point and \
             both decoders reach it; pyramidal is a convergence *accelerator*, not a \
             cheaper approximation. `min2` disagrees by a mean of 3.9 dB and up to 19.8 dB.\n"
        );
        let _ = writeln!(
            s,
            "The cause is the pyramid's reduced resolution, and the rule is exact: \
             `decmars` decodes at `1/2^levels` scale first, so a range block of \
             `min_size` occupies `min_size / 2^levels` pixels there. When that is ≥ 1 the \
             modes agree; when it is 0.5 — a **sub-pixel range block** — pyramidal loses \
             5+ dB and cannot recover it on the way back up.\n"
        );
        let _ = writeln!(
            s,
            "| image | min_size | pyramid levels | block at that level | pyramidal | iterative | gap |"
        );
        let _ = writeln!(s, "|---|---:|---:|---:|---:|---:|---:|");
        for (img, m, lev, blk, pyr, it) in [
            ("zoneplate 256²", 2, 1, "1.00 px", 22.615, 22.576),
            ("zoneplate 256²", 4, 1, "2.00 px", 12.125, 12.120),
            ("mandelbrot 512²", 2, 2, "**0.50 px**", 32.992, 38.738),
            ("mandelbrot 512²", 4, 2, "1.00 px", 28.764, 28.810),
            ("kodim01 768×512", 2, 2, "**0.50 px**", 23.870, 29.238),
            ("kodim01 768×512", 4, 2, "1.00 px", 27.421, 27.418),
        ] {
            let _ = writeln!(
                s,
                "| {img} | {m} | {lev} | {blk} | {} | {} | {} |",
                f(pyr, 3),
                f(it, 3),
                f(it - pyr, 3)
            );
        }
        let _ = writeln!(
            s,
            "\nThe 256² row at `min_size = 2` is the control: small blocks are harmless on \
             their own — it is sub-pixel blocks *at the pyramid's depth* that break. \
             (Measured with `-F -r 4`; the 256² absolute values are low because those are \
             synthetic stress fixtures, and only the gap is being compared.)\n"
        );

        // --- 5. structural variants -------------------------------------------
        if !self.variants.is_empty() {
            let _ = writeln!(s, "## 5. Structural variants vs the 1998 defaults\n");
            let _ = writeln!(
                s,
                "BD-rate of each one-parameter deviation against `default`, per method. \
                 Negative is better. The `evals/transform` column is that variant's \
                 work-weighted ratio, so the RD cost of a cheaper search is visible beside \
                 the saving.\n"
            );
            let _ = writeln!(
                s,
                "| variant | method | BD-rate % (mean) | median % | n | evals/transform | undefined because |"
            );
            let _ = writeln!(s, "|---|---|---:|---:|---:|---:|---|");
            for v in &self.variants {
                let vsel = Selection {
                    corpus: self.corpus.clone(),
                    variant: v.clone(),
                    decode_mode: self.decode_mode,
                };
                let evals = evals_table(&self.rows, &vsel);
                let mut methods: Vec<String> = table.iter().map(|e| e.method.clone()).collect();
                methods.sort();
                for m in methods {
                    let b = variant_summary(&self.rows, &self.corpus, self.decode_mode, &m, v);
                    let ept = evals
                        .iter()
                        .find(|e| e.method == m)
                        .map_or(f64::NAN, |e| e.evals_per_transform);
                    // §A7: an undefined cell says why, rather than showing an em-dash
                    // and letting the reader assume the comparison was merely omitted.
                    let why = match b.excluded.first() {
                        None => String::new(),
                        Some(e) => {
                            let r = if e.reason.contains("monotonic") {
                                "curve not monotonic (more bits, less quality)"
                            } else if e.reason.contains("overlap") {
                                "no PSNR overlap with the default curve"
                            } else {
                                &e.reason
                            };
                            format!(
                                "{r} on {}/{} images",
                                b.excluded.len(),
                                b.excluded.len() + b.n
                            )
                        }
                    };
                    let _ = writeln!(
                        s,
                        "| {} | {} | {} | {} | {} | {} | {} |",
                        v,
                        m,
                        f(b.mean_pct, 2),
                        f(b.median_pct, 2),
                        b.n,
                        f(ept, 1),
                        why,
                    );
                }
            }
            let _ = writeln!(s);
        }

        // --- 6. per-image spread ----------------------------------------------
        let _ = writeln!(s, "## 6. Per-image spread at the reference method\n");
        let _ = writeln!(
            s,
            "A corpus mean hides which images fractal coding suits. This is `{}` at \
             `-r 8`, per image.\n",
            self.reference_method
        );
        let _ = writeln!(
            s,
            "| image | PSNR-Y (dB) | bpp | transforms | evals/transform |"
        );
        let _ = writeln!(s, "|---|---:|---:|---:|---:|");
        let mut per: Vec<&BaselineRow> = self
            .rows
            .iter()
            .filter(|r| matches(r, &sel) && r.method == self.reference_method && r.t_rms == 8.0)
            .collect();
        per.sort_by(|a, b| a.image.cmp(&b.image));
        for r in per {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} |",
                r.image,
                r.quality.psnr_y.map_or("—".into(), |v| f(v, 3)),
                r.quality.bpp.map_or("—".into(), |v| f(v, 4)),
                r.encode.transforms,
                f(r.encode.evals_per_transform, 1),
            );
        }
        let _ = writeln!(s);

        s
    }
}

// --------------------------------------------------------------------------- the gate

/// One checked claim. §A1: the step's exit criteria are a command that exits 0 or 1, so
/// each criterion has to be expressible as one of these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Check {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

fn check(name: &str, passed: bool, detail: impl Into<String>) -> Check {
    Check {
        name: name.into(),
        passed,
        detail: detail.into(),
    }
}

/// Step 2's exit criteria, evaluated against the store.
///
/// `expected_rates` and `expected_methods` come from the sweep config rather than from the
/// data, so a sweep that silently ran half the grid fails here instead of producing a
/// report about whatever it happened to finish.
pub fn gate(
    rows: &[BaselineRow],
    corpus: &str,
    variant: &str,
    decode_modes: &[DecodeMode],
    expected_methods: &[String],
    expected_rates: usize,
    reference: &str,
) -> Vec<Check> {
    let mut out = Vec::new();
    let sel = Selection {
        corpus: corpus.into(),
        variant: variant.into(),
        decode_mode: decode_modes[0],
    };

    // --- provenance (§M7) ---------------------------------------------------
    // Checked on the rows as loaded; the envelope is validated by the caller.
    let images: std::collections::BTreeSet<&str> = rows
        .iter()
        .filter(|r| r.corpus == corpus)
        .map(|r| r.image.as_str())
        .collect();
    out.push(check(
        "corpus is non-empty",
        !images.is_empty(),
        format!("{} images in corpus {corpus}", images.len()),
    ));

    // --- completeness -------------------------------------------------------
    let want = images.len() * expected_methods.len() * expected_rates * decode_modes.len();
    let got = rows
        .iter()
        .filter(|r| r.corpus == corpus && r.variant == variant)
        .count();
    out.push(check(
        "the headline grid is complete",
        got == want,
        format!(
            "{got} rows for {corpus}/{variant}; expected {} images x {} methods x {expected_rates} rates x {} decode modes = {want}",
            images.len(),
            expected_methods.len(),
            decode_modes.len()
        ),
    ));

    // --- all six methods present -------------------------------------------
    let mut have: Vec<&str> = rows
        .iter()
        .filter(|r| matches(r, &sel))
        .map(|r| r.method.as_str())
        .collect();
    have.sort_unstable();
    have.dedup();
    let missing: Vec<&String> = expected_methods
        .iter()
        .filter(|m| !have.contains(&m.as_str()))
        .collect();
    out.push(check(
        "every speed-up method has curves",
        missing.is_empty(),
        if missing.is_empty() {
            format!("{} methods: {}", have.len(), have.join(", "))
        } else {
            format!("missing: {missing:?}")
        },
    ));

    // --- every per-image curve is usable -----------------------------------
    let curves = per_image_curves(rows, &sel);
    let short: Vec<String> = curves
        .iter()
        .filter(|(_, c)| c.points.len() != expected_rates)
        .map(|((i, m), c)| format!("{i}/{m}: {} points", c.points.len()))
        .collect();
    out.push(check(
        "every per-image curve has every rate point",
        short.is_empty(),
        if short.is_empty() {
            format!("{} curves x {expected_rates} points", curves.len())
        } else {
            short.join("; ")
        },
    ));

    // --- BD-rate between the methods, with nothing quietly excluded ---------
    let mut bd_detail = Vec::new();
    let mut bd_ok = true;
    for m in expected_methods.iter().filter(|m| m.as_str() != reference) {
        let s = bd_summary(rows, &sel, reference, m);
        if !s.excluded.is_empty() || s.n != images.len() {
            bd_ok = false;
        }
        bd_detail.push(format!(
            "{m}: {:+.2}% over {}/{} images{}",
            s.mean_pct,
            s.n,
            images.len(),
            if s.excluded.is_empty() {
                String::new()
            } else {
                format!(
                    " — EXCLUDED {}",
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
        "BD-rate is defined between every method and the reference, on every image",
        bd_ok,
        bd_detail.join(" | "),
    ));

    // --- the headline metric ------------------------------------------------
    let table = evals_table(rows, &sel);
    let mut ratios: Vec<u64> = table
        .iter()
        .map(|e| e.evals_per_transform.to_bits())
        .collect();
    let n_before = ratios.len();
    ratios.sort_unstable();
    ratios.dedup();
    out.push(check(
        "evals/transform is recorded and distinct per method (§M5)",
        n_before == expected_methods.len() && ratios.len() == n_before,
        table
            .iter()
            .map(|e| format!("{}: {:.1}", e.method, e.evals_per_transform))
            .collect::<Vec<_>>()
            .join(", "),
    ));

    // --- decode modes -------------------------------------------------------
    if decode_modes.len() > 1 {
        let d = decode_mode_delta(rows, corpus, variant);
        out.push(check(
            "both decode modes were run and recorded at matched settings",
            d.n > 0,
            format!(
                "{} matched settings; iterative − pyramidal mean {:+.4} dB (range {:+.4} to {:+.4})",
                d.n, d.mean_db, d.min_db, d.max_db
            ),
        ));
    }

    out
}
