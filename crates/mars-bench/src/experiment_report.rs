//! The strict experiment payload and its report (research plan §3 P0.3).
//!
//! One append-only JSONL row per planned case, appended the moment that case
//! completes. The row is the record of record: it carries the case's stable identity
//! (`case_id`), the config/build hashes resume compares against, the exact codec and
//! decoder arguments, the codec/arm/decoder parameters, the raw per-repetition timing
//! samples, and either the measured quality or an explicit non-success status. Missing
//! results are never implied by an absent row — a planned case always produces a row,
//! including `unsupported` and `missing_prerequisite` outcomes.
//!
//! This payload is versioned separately from the `Row` envelope (`store::SCHEMA`) and
//! from the older Step22 JSONL: legacy readers keep working, and this schema is not a
//! reinterpretation of theirs.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bdrate::{bd_metrics, RdCurve, RdPoint};
use crate::experiment_config::{Arm, CodecSpec, DecoderSpec};
use crate::measure::Measurement;
use crate::provenance::Provenance;
use crate::store::{read_rows, ResultStore, Row, StoreError};

/// `Row.kind` for an experiment case. Distinct from `baseline-mars1`, `quality`, etc.
pub const EXPERIMENT_ROW_KIND: &str = "experiment";
/// Payload schema. Bump when the payload shape changes; readers key off it.
pub const EXPERIMENT_SCHEMA: u32 = 1;

/// A file's bytes and content hash. Deserialisable, so a row can be read back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileIdentity {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

/// The source/build identity that resume compares and that a reader needs to decide
/// whether two rows are even comparable (§M7).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildIdentity {
    pub git_sha: String,
    /// Uncommitted tracked changes were present when the row was written.
    pub git_dirty: bool,
    /// SHA-256 of `git diff HEAD` when [`Self::git_dirty`]; `None` when clean. A dirty
    /// row is recorded rather than suppressed, but two dirty builds are not assumed equal.
    pub git_patch_sha256: Option<String>,
    pub lockfile_sha256: Option<String>,
    pub rustc_version: String,
}

/// The outcome of one planned case. Every planned case gets exactly one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    /// Encode, decode and metrics all completed.
    Succeeded,
    /// A codec process exited non-zero, or a produced artifact was invalid.
    Failed,
    /// A codec process exceeded the stage timeout and was killed and reaped.
    TimedOut,
    /// A required file or binary was absent when the case was about to run.
    MissingPrerequisite,
    /// The combination is refused by the codec for this image (e.g. `--method` on a
    /// colour image). Recorded explicitly rather than silently dropped.
    Unsupported,
}

impl CaseStatus {
    pub fn is_succeeded(self) -> bool {
        matches!(self, Self::Succeeded)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::TimedOut => "timed_out",
            Self::MissingPrerequisite => "missing_prerequisite",
            Self::Unsupported => "unsupported",
        }
    }
}

/// One encode+decode repetition. Raw samples, never averaged into the row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Repetition {
    pub index: u32,
    pub encode_wall_secs: f64,
    pub decode_wall_secs: f64,
    pub stream: FileIdentity,
    pub decoded: FileIdentity,
    pub decode_iterations: u32,
    pub encode_status: String,
    pub decode_status: String,
}

/// Encoder phase work counters, parsed from the encoder's own summary line where that
/// line carries them. `None` means the codec reported nothing parseable — an explicitly
/// unavailable diagnostic, not a zero.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EncodeCounters {
    pub search_evals: Option<u64>,
    pub warmup_evals: Option<u64>,
    pub total_evals: Option<u64>,
    pub transforms: Option<u64>,
}

fn digits_after(line: &str, marker: &str) -> Option<u64> {
    let idx = line.find(marker)?;
    let digits: String = line[idx + marker.len()..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

fn digits_before(line: &str, marker: &str) -> Option<u64> {
    let idx = line.find(marker)?;
    let digits: String = line[..idx]
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    digits.parse().ok()
}

impl EncodeCounters {
    /// Parse the `encmars` summary line. Recognises the method, plain-grayscale and
    /// colour shapes the CLI emits; anything else yields `None` rather than a guess.
    pub fn parse(stdout: &str) -> Option<Self> {
        let line = stdout.lines().rev().find(|line| line.contains(" bpp"))?;
        let method = line.contains("method ") || line.contains("search_evals=");
        let search_evals = if method {
            digits_after(line, "search_evals=")
        } else if line.contains("evals Y ") {
            digits_after(line, "evals Y ")
        } else {
            digits_before(line, " evals)")
        };
        Some(Self {
            search_evals,
            warmup_evals: digits_after(line, "warmup_evals="),
            total_evals: digits_after(line, "total_evals="),
            transforms: method.then(|| digits_before(line, " transforms")).flatten(),
        })
    }
}

/// The payload of one `experiment` row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentCaseRow {
    pub schema: u32,
    /// Config `name`.
    pub experiment: String,
    /// Stable id of this experiment run definition: `<name>-<config-sha-prefix>`.
    pub experiment_id: String,
    /// SHA-256 of the config's canonical JSON. Resume requires an exact match.
    pub config_sha256: String,
    pub stage: String,
    /// Stable per-case identity; the resume key.
    pub case_id: String,
    /// Position in the deterministic plan.
    pub case_index: usize,
    pub total_cases: usize,
    pub arm: String,
    pub arm_description: String,
    pub image: String,
    pub image_file: String,
    /// Source image-set index, when the image came from one.
    pub corpus_index: Option<String>,
    /// `None` when the input is a self-describing container (PNG/PNM).
    pub raw_dims: Option<(usize, usize)>,
    pub planes: usize,
    /// Where this attempt's artifacts live. `None` when nothing ran.
    pub artifact_dir: Option<PathBuf>,
    /// `MARC` (single-layer) or `MPRG` (progressive), read from the coded file's magic.
    pub container: Option<String>,
    pub codec: CodecSpec,
    pub arm_params: Arm,
    pub decoder: DecoderSpec,
    pub encode_args: Vec<String>,
    pub decode_args: Vec<String>,
    /// The input file as read at execution time (bytes and hash), independent of any
    /// index-recorded hash.
    pub input: Option<FileIdentity>,
    pub encoder_binary: Option<FileIdentity>,
    pub decoder_binary: Option<FileIdentity>,
    pub build: BuildIdentity,
    pub status: CaseStatus,
    pub error: Option<String>,
    /// Why an `unsupported` case was not run.
    pub unsupported_reason: Option<String>,
    pub repetitions: Vec<Repetition>,
    /// Nominal (first repetition) wall times, for convenient grouping.
    pub encode_wall_secs: Option<f64>,
    pub decode_wall_secs: Option<f64>,
    pub stream: Option<FileIdentity>,
    pub decoded: Option<FileIdentity>,
    pub metrics: Option<Measurement>,
    /// Why metrics are absent, when the case otherwise succeeded. Distinguishes an
    /// unavailable optional metric from a missing experiment.
    pub metrics_unavailable: Option<String>,
    /// The encoder's summary line, retained verbatim.
    pub encode_summary: Option<String>,
    /// Phase work counters, where the encoder's line carries them.
    pub encode_counters: Option<EncodeCounters>,
    /// Search diagnostics the subprocess codec does not expose (per-block candidate
    /// coverage/regret, leaf/mode histograms). Non-`None` states the reason, so an
    /// unavailable diagnostic is never mistaken for an unrun experiment.
    pub diagnostics_unavailable: Option<String>,
    pub timeout_secs: u64,
    pub notes: String,
}

impl ExperimentCaseRow {
    /// `(bpp, PSNR-Y)` when the case produced a finite, positive point.
    pub fn rd_point(&self) -> Option<RdPoint> {
        let m = self.metrics.as_ref()?;
        let psnr = m
            .psnr_y
            .or_else(|| m.psnr.iter().flatten().next().copied())?;
        let bpp = m.bpp?;
        (bpp.is_finite() && bpp > 0.0 && psnr.is_finite()).then_some(RdPoint { bpp, psnr })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("{path}: row {line} has an unreadable experiment payload: {source}")]
    Payload {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path}: payload schema {got} is not {need}")]
    Schema { path: PathBuf, got: u32, need: u32 },
}

/// Append one case row. Flushed before returning, so an interrupted run loses at most
/// the case in flight.
pub fn append_case(
    store: &mut ResultStore,
    provenance: &Provenance,
    row: &ExperimentCaseRow,
) -> Result<(), StoreError> {
    store.append(&Row::new(EXPERIMENT_ROW_KIND, provenance.clone(), row)?)
}

/// Read every experiment row in order. Returns the payloads, not the envelopes.
pub fn read_cases(
    path: &Path,
    expected_id: Option<&str>,
) -> Result<Vec<ExperimentCaseRow>, ReportError> {
    let rows = read_rows(path)?;
    let mut cases = Vec::new();
    for (line, row) in rows.iter().enumerate() {
        if row.kind != EXPERIMENT_ROW_KIND {
            continue;
        }
        let payload: ExperimentCaseRow =
            serde_json::from_value(row.data.clone()).map_err(|source| ReportError::Payload {
                path: path.to_path_buf(),
                line: line + 1,
                source,
            })?;
        if payload.schema != EXPERIMENT_SCHEMA {
            return Err(ReportError::Schema {
                path: path.to_path_buf(),
                got: payload.schema,
                need: EXPERIMENT_SCHEMA,
            });
        }
        if expected_id.is_some_and(|id| id != payload.experiment_id) {
            continue;
        }
        cases.push(payload);
    }
    Ok(cases)
}

/// The latest row per `case_id`, in deterministic plan order.
pub fn latest_by_case(cases: &[ExperimentCaseRow]) -> Vec<ExperimentCaseRow> {
    let mut latest: BTreeMap<&str, &ExperimentCaseRow> = BTreeMap::new();
    for case in cases {
        latest.insert(case.case_id.as_str(), case);
    }
    let mut out: Vec<ExperimentCaseRow> = latest.into_values().cloned().collect();
    out.sort_by(|a, b| {
        a.case_index
            .cmp(&b.case_index)
            .then_with(|| a.case_id.cmp(&b.case_id))
    });
    out
}

/// Counts of every status, so no planned case can go unaccounted for.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusTally {
    pub succeeded: usize,
    pub failed: usize,
    pub timed_out: usize,
    pub missing_prerequisite: usize,
    pub unsupported: usize,
}

impl StatusTally {
    pub fn add(&mut self, status: CaseStatus) {
        match status {
            CaseStatus::Succeeded => self.succeeded += 1,
            CaseStatus::Failed => self.failed += 1,
            CaseStatus::TimedOut => self.timed_out += 1,
            CaseStatus::MissingPrerequisite => self.missing_prerequisite += 1,
            CaseStatus::Unsupported => self.unsupported += 1,
        }
    }

    pub fn total(&self) -> usize {
        self.succeeded + self.failed + self.timed_out + self.missing_prerequisite + self.unsupported
    }

    pub fn is_clean(&self) -> bool {
        self.failed == 0
            && self.timed_out == 0
            && self.missing_prerequisite == 0
            && self.unsupported == 0
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

/// The analysis group of a case: its arm's `group`, or the arm label when unset.
fn group_of(case: &ExperimentCaseRow) -> &str {
    case.arm_params
        .group
        .as_deref()
        .unwrap_or(case.arm.as_str())
}

/// Render a Markdown report from a store's latest-per-case rows. `reference_group`
/// names the arm `group` BD-rate is computed against.
pub fn render_markdown(
    experiment_id: &str,
    cases: &[ExperimentCaseRow],
    reference_group: Option<&str>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Experiment `{experiment_id}`\n\n"));
    if cases.is_empty() {
        out.push_str("No experiment rows found.\n");
        return out;
    }
    let total_cases = cases
        .iter()
        .map(|c| c.total_cases)
        .max()
        .unwrap_or(cases.len());
    let config_sha = cases[0].config_sha256.chars().take(16).collect::<String>();
    let build = &cases[0].build;
    out.push_str(&format!(
        "- config: `{}` (sha256 `{config_sha}…`)\n- stage: `{}`\n- planned cases: {total_cases}\n- build: `{}`{}{}\n- rustc: `{}`{}\n\n",
        cases[0].experiment,
        cases[0].stage,
        build.git_sha.chars().take(12).collect::<String>(),
        if build.git_dirty { "-dirty" } else { "" },
        build
            .git_patch_sha256
            .as_ref()
            .map(|h| format!(" (patch sha256 `{}…`)", h.chars().take(12).collect::<String>()))
            .unwrap_or_default(),
        build.rustc_version,
        build
            .lockfile_sha256
            .as_ref()
            .map(|h| format!(", Cargo.lock sha256 `{}…`", h.chars().take(12).collect::<String>()))
            .unwrap_or_else(|| ", Cargo.lock: unavailable".to_string()),
    ));

    let mut tally = StatusTally::default();
    for case in cases {
        tally.add(case.status);
    }
    out.push_str("## Status\n\n| status | cases |\n|---|---:|\n");
    for (label, count) in [
        ("succeeded", tally.succeeded),
        ("failed", tally.failed),
        ("timed_out", tally.timed_out),
        ("missing_prerequisite", tally.missing_prerequisite),
        ("unsupported", tally.unsupported),
    ] {
        out.push_str(&format!("| {label} | {count} |\n"));
    }
    out.push_str(&format!("| **total** | **{}** |\n\n", tally.total()));

    // Per-arm aggregation over the latest row per case.
    let mut arms: BTreeMap<&str, Vec<&ExperimentCaseRow>> = BTreeMap::new();
    for case in cases {
        arms.entry(case.arm.as_str()).or_default().push(case);
    }
    out.push_str("## Per-arm summary\n\n| arm | cases | ok | mean PSNR-Y dB | mean bpp | mean encode s | mean decode s |\n|---|---:|---:|---:|---:|---:|---:|\n");
    for (arm, rows) in &arms {
        let points: Vec<RdPoint> = rows.iter().filter_map(|r| r.rd_point()).collect();
        let encodes: Vec<f64> = rows.iter().filter_map(|r| r.encode_wall_secs).collect();
        let decodes: Vec<f64> = rows.iter().filter_map(|r| r.decode_wall_secs).collect();
        let ok = rows.iter().filter(|r| r.status.is_succeeded()).count();
        out.push_str(&format!(
            "| {arm} | {} | {ok} | {} | {} | {} | {} |\n",
            rows.len(),
            mean(&points.iter().map(|p| p.psnr).collect::<Vec<_>>())
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "—".into()),
            mean(&points.iter().map(|p| p.bpp).collect::<Vec<_>>())
                .map(|v| format!("{v:.5}"))
                .unwrap_or_else(|| "—".into()),
            mean(&encodes)
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "—".into()),
            mean(&decodes)
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "—".into()),
        ));
    }
    out.push('\n');

    // BD-rate per image between analysis groups, only where both curves qualify.
    if let Some(reference) = reference_group {
        out.push_str(&format!("## BD-rate vs group `{reference}`\n\n"));
        let mut groups: BTreeMap<&str, Vec<&ExperimentCaseRow>> = BTreeMap::new();
        for case in cases {
            groups.entry(group_of(case)).or_default().push(case);
        }
        let Some(reference_rows) = groups.get(reference) else {
            out.push_str(&format!(
                "Reference group `{reference}` is not present in this store (groups: {:?}).\n\n",
                groups.keys().collect::<Vec<_>>()
            ));
            let non_succeeded: Vec<&ExperimentCaseRow> =
                cases.iter().filter(|c| !c.status.is_succeeded()).collect();
            push_non_succeeded(&mut out, &non_succeeded);
            return out;
        };
        let mut reference_per_image: BTreeMap<&str, Vec<RdPoint>> = BTreeMap::new();
        for row in reference_rows {
            if let Some(point) = row.rd_point() {
                reference_per_image
                    .entry(row.image.as_str())
                    .or_default()
                    .push(point);
            }
        }
        out.push_str("| group | image | BD-rate % | PSNR interval dB |\n|---|---|---:|---|\n");
        let mut any = false;
        for (group, rows) in &groups {
            if group == &reference {
                continue;
            }
            let mut per_image: BTreeMap<&str, Vec<RdPoint>> = BTreeMap::new();
            for row in rows {
                if let Some(point) = row.rd_point() {
                    per_image.entry(row.image.as_str()).or_default().push(point);
                }
            }
            for (image, test_points) in per_image {
                let Some(reference_points) = reference_per_image.get(image) else {
                    continue;
                };
                if reference_points.len() < 4 || test_points.len() < 4 {
                    out.push_str(&format!(
                        "| {group} | {image} | insufficient points ({} vs {}) | — |\n",
                        reference_points.len(),
                        test_points.len()
                    ));
                    any = true;
                    continue;
                }
                let reference_curve = RdCurve::new(reference.to_string(), reference_points.clone());
                let test_curve = RdCurve::new(group.to_string(), test_points.clone());
                match bd_metrics(&reference_curve, &test_curve) {
                    Ok(result) => {
                        out.push_str(&format!(
                            "| {group} | {image} | {:.4} | {:.6}–{:.6} |\n",
                            result.bd_rate_pct,
                            result.psnr_interval_db.0,
                            result.psnr_interval_db.1
                        ));
                        any = true;
                    }
                    Err(error) => {
                        out.push_str(&format!(
                            "| {group} | {image} | not computable: {error} | — |\n"
                        ));
                        any = true;
                    }
                }
            }
        }
        if !any {
            out.push_str("| — | — | no image had two complete curves | — |\n");
        }
        out.push('\n');
    }

    let non_succeeded: Vec<&ExperimentCaseRow> =
        cases.iter().filter(|c| !c.status.is_succeeded()).collect();
    push_non_succeeded(&mut out, &non_succeeded);
    out
}

fn push_non_succeeded(out: &mut String, non_succeeded: &[&ExperimentCaseRow]) {
    if non_succeeded.is_empty() {
        return;
    }
    out.push_str("## Non-succeeded cases\n\n| case | arm | image | status | reason |\n|---|---|---|---|---|\n");
    for case in non_succeeded {
        let reason = case
            .error
            .as_deref()
            .or(case.unsupported_reason.as_deref())
            .unwrap_or("")
            .replace('|', "\\|")
            .replace('\n', " ");
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {reason} |\n",
            &case.case_id[..case.case_id.len().min(12)],
            case.arm,
            case.image,
            case.status.as_str()
        ));
    }
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment_config::{Arm, CodecSpec, DecoderSpec};
    use crate::provenance::Provenance;

    fn codec() -> CodecSpec {
        CodecSpec {
            min_size: 4,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            zero_threshold: 0,
            subsampling: "444".into(),
            t_rms: 8.0,
            chroma_t_rms: 8.0,
            threads: 1,
        }
    }

    fn arm(label: &str) -> Arm {
        Arm {
            label: label.into(),
            description: "d".into(),
            group: None,
            lambda: Some(200.0),
            modes: Some(vec![0, 2]),
            method: None,
            budget: None,
            seed: None,
            adaptive_density: false,
            adaptive_residual: false,
            progressive: false,
            iterations: None,
            smooth: None,
        }
    }

    fn row(
        case_id: &str,
        case_index: usize,
        arm: &str,
        image: &str,
        psnr: f64,
        bpp: f64,
    ) -> ExperimentCaseRow {
        ExperimentCaseRow {
            schema: EXPERIMENT_SCHEMA,
            experiment: "e".into(),
            experiment_id: "e-abc".into(),
            config_sha256: "f".repeat(64),
            stage: "smoke".into(),
            case_id: case_id.into(),
            case_index,
            total_cases: 2,
            arm: arm.into(),
            arm_description: "d".into(),
            image: image.into(),
            image_file: "x.raw".into(),
            corpus_index: None,
            raw_dims: Some((8, 8)),
            planes: 1,
            artifact_dir: Some(PathBuf::from("/tmp/x")),
            container: Some("MARC".into()),
            codec: codec(),
            arm_params: self::arm(arm),
            decoder: DecoderSpec::default(),
            encode_args: vec!["a".into()],
            decode_args: vec!["b".into()],
            input: None,
            encoder_binary: None,
            decoder_binary: None,
            build: BuildIdentity {
                git_sha: "deadbeef".into(),
                git_dirty: false,
                git_patch_sha256: None,
                lockfile_sha256: Some("ab".repeat(32)),
                rustc_version: "rustc 1.85.0".into(),
            },
            status: CaseStatus::Succeeded,
            error: None,
            unsupported_reason: None,
            repetitions: Vec::new(),
            encode_wall_secs: Some(1.0),
            decode_wall_secs: Some(0.5),
            stream: None,
            decoded: None,
            metrics: Some(Measurement {
                original: "a".into(),
                decoded: "b".into(),
                width: 8,
                height: 8,
                planes: 1,
                mse: vec![1.0],
                psnr: vec![Some(psnr)],
                psnr_y: Some(psnr),
                psnr_cb: None,
                psnr_cr: None,
                psnr_yuv: None,
                ssim: Some(0.9),
                ssim_unavailable: None,
                ms_ssim: None,
                ms_ssim_unavailable: Some("too small".into()),
                coded_bytes: Some(10),
                bpp: Some(bpp),
                definitions: crate::measure::definitions(
                    &mars_core::metrics::SsimConfig::default(),
                    &mars_core::metrics::MsSsimConfig::default(),
                ),
            }),
            metrics_unavailable: None,
            encode_summary: Some("8x8 gray -> x (10 bytes, 1.250 bpp, 7 evals)".into()),
            encode_counters: Some(EncodeCounters {
                search_evals: Some(7),
                warmup_evals: None,
                total_evals: None,
                transforms: None,
            }),
            diagnostics_unavailable: Some("subprocess codec exposes no per-block picks".into()),
            timeout_secs: 30,
            notes: String::new(),
        }
    }

    #[test]
    fn encode_counters_parse_every_summary_shape_the_codec_emits() {
        let gray = "8x8 gray -> out.mars (10 bytes, 1.250 bpp, 7 evals)\n";
        assert_eq!(
            EncodeCounters::parse(gray),
            Some(EncodeCounters {
                search_evals: Some(7),
                ..Default::default()
            })
        );
        let method = "8x8 gray -> out.mars (10 bytes, 1.250 bpp, method fisher, 7 evals, \
                      search_evals=7, warmup_evals=11, total_evals=18, 4 transforms, 1.75 evals/transform)\n";
        assert_eq!(
            EncodeCounters::parse(method),
            Some(EncodeCounters {
                search_evals: Some(7),
                warmup_evals: Some(11),
                total_evals: Some(18),
                transforms: Some(4),
            })
        );
        let rgb = "8x8 rgb (Yuv444) -> out.mars (10 bytes total: Y 4 + Cb 3 + Cr 3, 1.250 bpp, evals Y 7 / Cb 5 / Cr 5)\n";
        assert_eq!(
            EncodeCounters::parse(rgb),
            Some(EncodeCounters {
                search_evals: Some(7),
                ..Default::default()
            })
        );
        assert_eq!(EncodeCounters::parse("no summary here\n"), None);
    }

    #[test]
    fn rd_point_requires_a_finite_positive_point() {
        assert!(row("a", 0, "arm", "img", 30.0, 1.0).rd_point().is_some());
        let mut infinite = row("a", 0, "arm", "img", 30.0, 1.0);
        if let Some(metrics) = infinite.metrics.as_mut() {
            metrics.psnr_y = None;
            metrics.psnr = vec![None];
        }
        assert!(infinite.rd_point().is_none());
        let mut no_metrics = row("a", 0, "arm", "img", 30.0, 1.0);
        no_metrics.metrics = None;
        assert!(no_metrics.rd_point().is_none());
    }

    #[test]
    fn bd_rate_is_computed_per_group_across_arm_points() {
        let mut cases = Vec::new();
        for (i, (bpp, psnr)) in [(0.5, 28.0), (1.0, 31.0), (2.0, 34.0), (4.0, 37.0)]
            .into_iter()
            .enumerate()
        {
            let mut reference = row(&format!("r{i}"), i, "ref-lambda", "img", psnr, bpp);
            reference.arm_params.group = Some("reference".into());
            let mut test = row(
                &format!("t{i}"),
                10 + i,
                "test-lambda",
                "img",
                psnr - 1.0,
                bpp * 1.1,
            );
            test.arm_params.group = Some("test".into());
            cases.push(reference);
            cases.push(test);
        }
        let text = render_markdown("e-abc", &cases, Some("reference"));
        assert!(text.contains("## BD-rate vs group `reference`"), "{text}");
        assert!(text.contains("| test | img |"), "{text}");
        assert!(!text.contains("insufficient points"), "{text}");
    }

    #[test]
    fn a_missing_reference_group_is_named_rather_than_ignored() {
        let text = render_markdown(
            "e-abc",
            &[row("a", 0, "arm", "img", 30.0, 1.0)],
            Some("absent"),
        );
        assert!(text.contains("Reference group `absent` is not present"));
    }

    #[test]
    fn latest_by_case_keeps_the_last_attempt_in_plan_order() {
        let a = row("a", 0, "arm", "img", 30.0, 1.0);
        let mut a_retry = a.clone();
        a_retry.status = CaseStatus::Failed;
        a_retry.error = Some("transient".into());
        let b = row("b", 1, "arm", "img", 31.0, 1.1);
        let latest = latest_by_case(&[a, a_retry, b]);
        assert_eq!(latest.len(), 2);
        assert_eq!(latest[0].case_id, "a");
        assert_eq!(latest[0].status, CaseStatus::Failed);
        assert_eq!(latest[1].case_id, "b");
    }

    #[test]
    fn plan_order_is_independent_of_row_insertion_order() {
        let a = row("a", 0, "arm", "img", 30.0, 1.0);
        let b = row("b", 1, "arm", "img", 31.0, 1.1);
        let forward = latest_by_case(&[a.clone(), b.clone()]);
        let backward = latest_by_case(&[b, a]);
        assert_eq!(forward, backward);
    }

    #[test]
    fn status_tally_never_loses_a_case() {
        let mut tally = StatusTally::default();
        for status in [
            CaseStatus::Succeeded,
            CaseStatus::Failed,
            CaseStatus::TimedOut,
            CaseStatus::MissingPrerequisite,
            CaseStatus::Unsupported,
        ] {
            tally.add(status);
        }
        assert_eq!(tally.total(), 5);
        assert!(!tally.is_clean());
    }

    #[test]
    fn tally_adds_each_status_to_its_own_bucket() {
        let mut tally = StatusTally::default();
        tally.add(CaseStatus::Succeeded);
        tally.add(CaseStatus::Succeeded);
        tally.add(CaseStatus::TimedOut);
        assert_eq!(tally.succeeded, 2);
        assert_eq!(tally.timed_out, 1);
        assert_eq!(tally.failed, 0);
        assert_eq!(tally.total(), 3);
    }

    #[test]
    fn markdown_report_states_every_written_payload_without_ourown_nondeterminism() {
        let cases = vec![
            row("a", 0, "exhaustive", "img", 30.0, 1.0),
            row("b", 1, "random", "img", 29.0, 0.9),
        ];
        let text = render_markdown("e-abc", &cases, Some("exhaustive"));
        assert!(text.contains("Experiment `e-abc`"));
        assert!(text.contains("| exhaustive | 1 | 1 |"));
        assert!(text.contains("| random | 1 | 1 |"));
        assert!(text.contains("## Status"));
        assert!(text.contains("| **total** | **2** |"));
    }

    #[test]
    fn report_lists_the_reason_for_every_non_succeeded_case() {
        let mut failed = row("a", 0, "exhaustive", "img", 30.0, 1.0);
        failed.status = CaseStatus::Failed;
        failed.error = Some("codec exited with 7".into());
        let mut unsupported = row("b", 1, "random", "img", 30.0, 1.0);
        unsupported.status = CaseStatus::Unsupported;
        unsupported.unsupported_reason = Some("--method is grayscale only".into());
        let text = render_markdown("e-abc", &[failed, unsupported], None);
        assert!(text.contains("## Non-succeeded cases"));
        assert!(text.contains("codec exited with 7"));
        assert!(text.contains("--method is grayscale only"));
        assert!(text.contains("| failed | 1 |"));
        assert!(text.contains("| unsupported | 1 |"));
    }

    #[test]
    fn round_trip_through_the_append_only_store_preserves_the_payload() {
        let dir =
            std::env::temp_dir().join(format!("mars-experiment-report-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("results.jsonl");
        let cases = vec![
            row("a", 0, "exhaustive", "img", 30.0, 1.0),
            row("b", 1, "random", "img", 29.0, 0.9),
        ];
        {
            let mut store = ResultStore::open(&path).unwrap();
            for case in &cases {
                append_case(&mut store, &Provenance::detect(0), case).unwrap();
            }
        }
        // Reopening must append, not truncate.
        let mut store = ResultStore::open(&path).unwrap();
        append_case(&mut store, &Provenance::detect(0), &cases[0]).unwrap();
        drop(store);

        let read = read_cases(&path, Some("e-abc")).unwrap();
        assert_eq!(read.len(), 3);
        assert_eq!(latest_by_case(&read).len(), 2);
        // A non-matching experiment id filters every row out, not error.
        assert!(read_cases(&path, Some("other")).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
