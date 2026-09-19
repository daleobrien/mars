//! P5a's rate audit: the RD surrogate's estimated versus actual bits, measured on real
//! images (research plan §8 P5a).
//!
//! `crates/mars-codec/src/audit.rs` prices one partition twice -- against the frozen
//! warm-up snapshot the RD search decides with, and against the live models the stream is
//! really coded with -- and attributes both to the plan's five cost categories. This module
//! turns that into a measurement: it runs the encoder over an image set and a partition
//! grid, computes the quality each partition actually achieves, and appends one
//! self-describing row per case to the append-only store.
//!
//! Two partition families are audited against the *same* surrogate, which is what makes
//! them comparable (P5a's "same retrieval provider and mode set"): RD (`lambda = Some`) and
//! the legacy threshold walk (`lambda = None`). Only the RD family's estimate is an
//! objective the encoder optimised; a threshold row's estimate is a reference price that
//! walk never consulted, and the row says so.
//!
//! Bits come from the inner `.mars` plane stream. The whole-container (MARC) size and its
//! bpp are recorded separately, because the inner stream omits the CLI wrapper and the two
//! must not be conflated (§M2).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mars_codec::audit::{Category, EstimatedProvenance, RateAudit};
use mars_codec::encode::{
    audit_rd, AuditedEncode, EncodeOptions, EncodeParams, ResidualQuantisation,
};
use mars_codec::mars_format;
use mars_core::io::{read_image, write_pnm};
use serde::{Deserialize, Serialize};

use crate::measure::{measure, MeasureRequest, Measurement};
use crate::provenance::{sha256_hex, Provenance};
use crate::store::{ResultStore, Row};
use crate::sweep::ImageSet;

/// `Row.kind` for one audited partition.
pub const ROW_KIND: &str = "rate-audit";

/// Which partition family a row describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionKind {
    /// Step 14's bottom-up `J = D + λR` walk; its objective *is* the audited estimate.
    Rd,
    /// The legacy top-down `t_rms` threshold walk; the estimate is a reference price only.
    Threshold,
}

impl PartitionKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Rd => "rd",
            Self::Threshold => "threshold",
        }
    }
}

/// Estimated and actual bits for one category.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryRow {
    pub category: String,
    pub events: u64,
    pub estimated_bits: f64,
    pub actual_bits: f64,
}

/// One audited `(image, partition)` case. Self-contained per §M7.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateAuditRow {
    pub image: String,
    pub image_sha256: String,
    pub width: u32,
    pub height: u32,
    pub partition: PartitionKind,
    /// Present for [`PartitionKind::Rd`].
    pub lambda: Option<f64>,
    /// Present for [`PartitionKind::Threshold`].
    pub t_rms: Option<f64>,
    pub modes: Vec<u8>,
    pub iterations: u32,
    /// The five categories in [`Category::ALL`] order, so a reader never has to guess.
    pub categories: Vec<CategoryRow>,
    pub leaves: usize,
    pub events: u64,
    pub unattributed_events: u64,
    /// Frozen-snapshot price of the final event stream.
    pub estimated_bits: f64,
    /// Live-model information content of the same events.
    pub actual_event_bits: f64,
    /// The inner `.mars` plane stream, header and sections included.
    pub inner_stream_bytes: u64,
    pub inner_stream_bits: f64,
    /// `inner_stream_bits - actual_event_bits`: header, section table, alignment, coder flush.
    pub overhead_bits: f64,
    pub estimate_error_bits: f64,
    pub estimate_error_pct: f64,
    pub estimated_provenance: String,
    pub search_evals: u64,
    pub warmup_evals: u64,
    pub quality: Measurement,
    pub notes: String,
}

/// One image to audit, resolved from a corpus index or an explicit input.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditImage {
    pub name: String,
    /// Repo-relative path to the headerless 8-bit grayscale raw.
    pub file: PathBuf,
    pub width: u32,
    pub height: u32,
    pub sha256: String,
}

/// Encoder parameters held fixed across the whole sweep.
#[derive(Debug, Clone, Copy)]
pub struct AuditCodec {
    pub min_size: u32,
    pub max_size: u32,
    pub shift: u32,
    pub bits_alfa: u32,
    pub bits_beta: u32,
    pub max_alfa: f64,
}

impl Default for AuditCodec {
    /// The project's default profile: sizes 4-16, stride 4, 4/7 quantiser bits, contrast 1.
    fn default() -> Self {
        Self {
            min_size: 4,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
        }
    }
}

/// What to audit: images, the partition grid, the mode mask, and scratch space.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub images: Vec<AuditImage>,
    /// RD policies, in the order they should be run.
    pub lambdas: Vec<f64>,
    /// Legacy threshold policies, in the order they should be run.
    pub thresholds: Vec<f64>,
    /// P5a starts at modes 0/2; the mask is explicit so that stays a decision, not a default.
    pub modes: [bool; 4],
    pub mode_numbers: Vec<u8>,
    pub codec: AuditCodec,
    pub iterations: u32,
    pub scratch: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{0}")]
    Invalid(String),
    #[error("{name}: the audit is grayscale-only, but this image has {planes} planes")]
    NotGray { name: String, planes: usize },
    #[error(transparent)]
    Format(#[from] mars_format::MarsFormatError),
    #[error("decoding the audited stream: {0}")]
    Decode(String),
    #[error(transparent)]
    Measure(#[from] crate::measure::MeasureError),
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
    #[error("serialising a row: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Resolve images from a corpus image-set index, verifying each file against its recorded
/// hash before any encode runs.
pub fn images_from_index(
    root: &Path,
    index: &Path,
    subset: &[String],
) -> Result<Vec<AuditImage>, AuditError> {
    let path = root.join(index);
    let set = ImageSet::read(&path).map_err(|source| AuditError::Invalid(source.to_string()))?;
    let wanted: std::collections::BTreeSet<&str> = subset.iter().map(String::as_str).collect();
    let mut images = Vec::new();
    for entry in &set.images {
        if !wanted.is_empty() && !wanted.contains(entry.name.as_str()) {
            continue;
        }
        let absolute = root.join(&entry.file);
        let bytes = std::fs::read(&absolute).map_err(|source| AuditError::Io {
            path: absolute.display().to_string(),
            source,
        })?;
        let actual = sha256_hex(&bytes);
        if actual != entry.sha256 {
            return Err(AuditError::Invalid(format!(
                "{}: sha256 {actual} does not match the index's {}",
                entry.file.display(),
                entry.sha256
            )));
        }
        images.push(AuditImage {
            name: entry.name.clone(),
            file: entry.file.clone(),
            width: entry.width,
            height: entry.height,
            sha256: entry.sha256.clone(),
        });
    }
    for name in &wanted {
        if !images.iter().any(|image| image.name == *name) {
            return Err(AuditError::Invalid(format!(
                "requested image {name:?} is not in {}",
                index.display()
            )));
        }
    }
    if images.is_empty() {
        return Err(AuditError::Invalid(format!(
            "{} resolved to zero images",
            index.display()
        )));
    }
    Ok(images)
}

/// Resolve one explicit headerless raw input, verifying its size and recording its hash.
pub fn image_from_input(
    root: &Path,
    file: &Path,
    width: usize,
    height: usize,
    name: &str,
) -> Result<AuditImage, AuditError> {
    let absolute = root.join(file);
    let bytes = std::fs::read(&absolute).map_err(|source| AuditError::Io {
        path: absolute.display().to_string(),
        source,
    })?;
    let expected = width
        .checked_mul(height)
        .ok_or_else(|| AuditError::Invalid("dimensions overflow".into()))?;
    if bytes.len() != expected {
        return Err(AuditError::Invalid(format!(
            "{}: {} bytes does not match {}x{} ({expected})",
            file.display(),
            bytes.len(),
            width,
            height
        )));
    }
    Ok(AuditImage {
        name: name.to_string(),
        file: file.to_path_buf(),
        width: width as u32,
        height: height as u32,
        sha256: sha256_hex(&bytes),
    })
}

/// What a sweep produced.
#[derive(Debug, Clone, Serialize)]
pub struct AuditSummary {
    pub rows: usize,
    pub store: PathBuf,
    pub markdown: String,
}

struct RowBuild {
    partition: PartitionKind,
    lambda: Option<f64>,
    t_rms: Option<f64>,
    modes: Vec<u8>,
    iterations: u32,
}

fn provenance_label(provenance: EstimatedProvenance) -> &'static str {
    match provenance {
        EstimatedProvenance::RdWarmupSnapshot => "rd_warmup_snapshot",
        EstimatedProvenance::ReferenceOnly => "reference_only",
    }
}

fn category_rows(audit: &RateAudit) -> Vec<CategoryRow> {
    Category::ALL
        .iter()
        .map(|&category| {
            let bits = audit.category(category);
            CategoryRow {
                category: category.label().to_string(),
                events: bits.events,
                estimated_bits: bits.estimated_bits,
                actual_bits: bits.actual_bits,
            }
        })
        .collect()
}

fn build_row(
    image: &AuditImage,
    build: RowBuild,
    audited: &AuditedEncode,
    quality: Measurement,
) -> RateAuditRow {
    let audit = &audited.audit;
    let notes = match audit.estimated_provenance {
        EstimatedProvenance::RdWarmupSnapshot => {
            "estimate is the objective this partition minimised".to_string()
        }
        EstimatedProvenance::ReferenceOnly => {
            "estimate is a reference price the threshold walk never consulted".to_string()
        }
    };
    RateAuditRow {
        image: image.name.clone(),
        image_sha256: image.sha256.clone(),
        width: image.width,
        height: image.height,
        partition: build.partition,
        lambda: build.lambda,
        t_rms: build.t_rms,
        modes: build.modes,
        iterations: build.iterations,
        categories: category_rows(audit),
        leaves: audit.leaves,
        events: audit.events,
        unattributed_events: audit.unattributed_events,
        estimated_bits: audit.estimated_bits,
        actual_event_bits: audit.actual_event_bits,
        inner_stream_bytes: audit.serialized_bytes,
        inner_stream_bits: audit.serialized_bits,
        overhead_bits: audit.overhead_bits,
        estimate_error_bits: audit.estimate_error_bits(),
        estimate_error_pct: audit.estimate_error_pct(),
        estimated_provenance: provenance_label(audit.estimated_provenance).to_string(),
        search_evals: audited.outcome.counters.search_evals,
        warmup_evals: audited.outcome.counters.warmup_evals,
        quality,
        notes,
    }
}

fn params(codec: &AuditCodec, lambda: Option<f64>, t_rms: f64) -> EncodeParams {
    EncodeParams {
        min_size: codec.min_size,
        max_size: codec.max_size,
        shift: codec.shift,
        bits_alfa: codec.bits_alfa,
        bits_beta: codec.bits_beta,
        max_alfa: codec.max_alfa,
        t_rms,
        zero_threshold: 0,
        lambda,
    }
}

/// Encode one partition, write the container and its decode, and measure the pair with the
/// harness's one metrics implementation (§M1) rather than anything the codec reports.
fn audit_one(
    root: &Path,
    config: &AuditConfig,
    image: &AuditImage,
    build: RowBuild,
) -> Result<RateAuditRow, AuditError> {
    let lambda = build.lambda;
    let t_rms = build.t_rms.unwrap_or(8.0);
    let original = root.join(&image.file);
    let input = read_image(
        &original,
        Some((image.width as usize, image.height as usize)),
    )
    .map_err(|source| AuditError::Invalid(format!("{}: {source}", image.file.display())))?;
    if input.planes().len() != 1 {
        return Err(AuditError::NotGray {
            name: image.name.clone(),
            planes: input.planes().len(),
        });
    }
    let plane = &input.planes()[0];

    let options = EncodeOptions {
        allowed_modes: config.modes,
        adaptive_density: false,
        residual_quantisation: ResidualQuantisation::default(),
    };
    let audited = audit_rd(plane, &params(&config.codec, lambda, t_rms), &options)?;

    let inner = mars_format::write(&audited.outcome.header, &audited.outcome.leaves)?;
    let container = mars_codec::color::wrap_gray_stream(inner);
    let case_dir = config.scratch.join(&image.name).join(match lambda {
        Some(value) => format!("rd-l{value}"),
        None => format!("threshold-t{t_rms}"),
    });
    std::fs::create_dir_all(&case_dir).map_err(|source| AuditError::Io {
        path: case_dir.display().to_string(),
        source,
    })?;
    let coded = case_dir.join("coded.mars");
    let decoded = case_dir.join("decoded.pgm");
    std::fs::write(&coded, &container).map_err(|source| AuditError::Io {
        path: coded.display().to_string(),
        source,
    })?;
    let image_out =
        mars_codec::color::decode_color_image_zoomed(&container, config.iterations, 1.0)
            .map_err(|error| AuditError::Decode(error.to_string()))?;
    write_pnm(&decoded, &image_out)
        .map_err(|source| AuditError::Invalid(format!("{}: {source}", decoded.display())))?;

    let request = MeasureRequest::new(&original, &decoded)
        .with_raw_dims(image.width as usize, image.height as usize)
        .with_coded_bytes(container.len() as u64);
    let quality = measure(&request)?;

    Ok(build_row(image, build, &audited, quality))
}

/// Run the whole grid (image-major, then RD lambdas, then thresholds), appending each row
/// as it completes. A failure stops the sweep with an error; rows already written stay.
pub fn run(
    root: &Path,
    config: &AuditConfig,
    store_path: &Path,
) -> Result<AuditSummary, AuditError> {
    if config.lambdas.is_empty() && config.thresholds.is_empty() {
        return Err(AuditError::Invalid(
            "no partition policies requested: give at least one lambda or threshold".into(),
        ));
    }
    for value in config.lambdas.iter().chain(config.thresholds.iter()) {
        if !value.is_finite() || *value < 0.0 {
            return Err(AuditError::Invalid(format!(
                "partition parameters must be finite and nonnegative, got {value}"
            )));
        }
    }
    let mut store = ResultStore::open(store_path)?;
    let mut rows = Vec::new();
    for image in &config.images {
        for &lambda in &config.lambdas {
            let build = RowBuild {
                partition: PartitionKind::Rd,
                lambda: Some(lambda),
                t_rms: None,
                modes: config.mode_numbers.clone(),
                iterations: config.iterations,
            };
            let row = audit_one(root, config, image, build)?;
            store.append(&Row::new(ROW_KIND, Provenance::detect(0), &row)?)?;
            rows.push(row);
        }
        for &t_rms in &config.thresholds {
            let build = RowBuild {
                partition: PartitionKind::Threshold,
                lambda: None,
                t_rms: Some(t_rms),
                modes: config.mode_numbers.clone(),
                iterations: config.iterations,
            };
            let row = audit_one(root, config, image, build)?;
            store.append(&Row::new(ROW_KIND, Provenance::detect(0), &row)?)?;
            rows.push(row);
        }
    }
    let markdown = render_markdown(&rows);
    Ok(AuditSummary {
        rows: rows.len(),
        store: store_path.to_path_buf(),
        markdown,
    })
}

/// Key that makes the report independent of the order rows happened to be produced in.
fn sort_key(row: &RateAuditRow) -> (String, u8, f64) {
    let parameter = row.lambda.or(row.t_rms).unwrap_or(0.0);
    let kind = match row.partition {
        PartitionKind::Rd => 0,
        PartitionKind::Threshold => 1,
    };
    (row.image.clone(), kind, parameter)
}

/// Render the audit as Markdown. Rows are sorted, so the same row set always renders the
/// same text regardless of insertion order.
pub fn render_markdown(rows: &[RateAuditRow]) -> String {
    let mut sorted: Vec<&RateAuditRow> = rows.iter().collect();
    // `total_cmp`, not `partial_cmp`: a stored row could carry a non-finite parameter, and
    // a report must never panic on data it can simply order a different way.
    sorted.sort_by(|a, b| {
        let (a_image, a_kind, a_parameter) = sort_key(a);
        let (b_image, b_kind, b_parameter) = sort_key(b);
        a_image
            .cmp(&b_image)
            .then(a_kind.cmp(&b_kind))
            .then(a_parameter.total_cmp(&b_parameter))
    });

    let mut out = String::new();
    out.push_str("# P5a rate audit — RD surrogate vs actual coded bits\n\n");
    out.push_str(
        "`est` is the frozen `t_rms = 8` warm-up snapshot's price for the final event stream; \
         `actual` is the same events' information content under the live, evolving models the \
         stream is really coded with. Both partition families are priced against the identical \
         snapshot, so they are comparable; an `rd` row's estimate is the objective that \
         partition minimised, while a `threshold` row's is a reference price the walk never \
         consulted. `overhead` is everything the payload does not carry: inner header, section \
         table, word alignment and the coder's final state.\n\n",
    );

    out.push_str("| image | partition | parameter | leaves | bpp | PSNR-Y dB | est bits | actual bits | error % | inner B | container B | overhead bits |\n");
    out.push_str("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for row in &sorted {
        let parameter = match (row.lambda, row.t_rms) {
            (Some(lambda), _) => format!("λ={lambda}"),
            (None, Some(t_rms)) => format!("t_rms={t_rms}"),
            (None, None) => "—".to_string(),
        };
        out.push_str(&format!(
            "| {image} | {partition} | {parameter} | {leaves} | {bpp} | {psnr} | {est:.1} | {actual:.1} | {error:.3} | {inner} | {container} | {overhead:.1} |\n",
            image = row.image,
            partition = row.partition.label(),
            leaves = row.leaves,
            bpp = row
                .quality
                .bpp
                .map(|v| format!("{v:.5}"))
                .unwrap_or_else(|| "—".into()),
            psnr = row
                .quality
                .psnr_y
                .map(|v| format!("{v:.4}"))
                .unwrap_or_else(|| "—".into()),
            est = row.estimated_bits,
            actual = row.actual_event_bits,
            error = row.estimate_error_pct,
            inner = row.inner_stream_bytes,
            container = row.container_bytes(),
            overhead = row.overhead_bits,
        ));
    }
    out.push('\n');

    out.push_str("## Category breakdown\n\n");
    for row in &sorted {
        let parameter = match (row.lambda, row.t_rms) {
            (Some(lambda), _) => format!("λ={lambda}"),
            (None, Some(t_rms)) => format!("t_rms={t_rms}"),
            (None, None) => "—".to_string(),
        };
        out.push_str(&format!(
            "### {} · {} · {parameter}\n\n| category | events | est bits | actual bits | error % |\n|---|---:|---:|---:|---:|\n",
            row.image,
            row.partition.label()
        ));
        for category in &row.categories {
            let error = if category.actual_bits > 0.0 {
                (category.estimated_bits - category.actual_bits) / category.actual_bits * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "| {} | {} | {:.1} | {:.1} | {:.3} |\n",
                category.category,
                category.events,
                category.estimated_bits,
                category.actual_bits,
                error
            ));
        }
        out.push('\n');
    }

    out.push_str("## Notes\n\n");
    out.push_str(
        "- The estimate deliberately excludes container overhead; a stream is always larger \
         than its events' information content by at least that much.\n",
    );
    out.push_str(
        "- Only `rd` rows were optimised against their `est` column. A `threshold` row's \
         `est` is reported for comparability, never as an objective it pursued.\n",
    );
    out.push_str(
        "- Category bits are the inner plane stream's own; `container B` includes the MARC \
         wrapper and is what §M2's bpp is computed from.\n",
    );
    out
}

impl RateAuditRow {
    /// The whole-container (MARC) size. Read from the measurement rather than recomputed,
    /// so the container figure and the bpp it came from can never drift apart.
    pub fn container_bytes(&self) -> u64 {
        self.quality.coded_bytes.unwrap_or(0)
    }
}

/// Group rows by image, for callers that want per-image summaries.
pub fn by_image(rows: &[RateAuditRow]) -> BTreeMap<&str, Vec<&RateAuditRow>> {
    let mut map: BTreeMap<&str, Vec<&RateAuditRow>> = BTreeMap::new();
    for row in rows {
        map.entry(row.image.as_str()).or_default().push(row);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use mars_core::Plane;

    fn image() -> AuditImage {
        AuditImage {
            name: "synthetic".into(),
            file: PathBuf::from("synthetic.raw"),
            width: 64,
            height: 64,
            sha256: "a".repeat(64),
        }
    }

    fn synthetic_plane() -> Plane {
        let mut data = vec![0u8; 64 * 64];
        for r in 0..64 {
            for c in 0..64 {
                data[r * 64 + c] = if c < 32 {
                    128
                } else {
                    ((r * 37 + c * 19) % 256) as u8
                };
            }
        }
        Plane::from_vec(64, 64, data)
    }

    fn row_for(partition: PartitionKind, lambda: Option<f64>, t_rms: Option<f64>) -> RateAuditRow {
        let plane = synthetic_plane();
        let options = EncodeOptions {
            allowed_modes: [true, false, true, false],
            ..EncodeOptions::default()
        };
        let audited = audit_rd(
            &plane,
            &EncodeParams {
                min_size: 4,
                max_size: 16,
                shift: 4,
                bits_alfa: 4,
                bits_beta: 7,
                max_alfa: 1.0,
                t_rms: t_rms.unwrap_or(8.0),
                zero_threshold: 0,
                lambda,
            },
            &options,
        )
        .unwrap();
        let quality = Measurement {
            original: "a".into(),
            decoded: "b".into(),
            width: 64,
            height: 64,
            planes: 1,
            mse: vec![1.0],
            psnr: vec![Some(30.0)],
            psnr_y: Some(30.0),
            psnr_cb: None,
            psnr_cr: None,
            psnr_yuv: None,
            ssim: Some(0.9),
            ssim_unavailable: None,
            ms_ssim: None,
            ms_ssim_unavailable: Some("too small".into()),
            coded_bytes: Some(100),
            bpp: Some(0.5),
            definitions: crate::measure::definitions(
                &mars_core::metrics::SsimConfig::default(),
                &mars_core::metrics::MsSsimConfig::default(),
            ),
        };
        build_row(
            &image(),
            RowBuild {
                partition,
                lambda,
                t_rms,
                modes: vec![0, 2],
                iterations: 10,
            },
            &audited,
            quality,
        )
    }

    #[test]
    fn categories_come_out_in_canonical_order_for_every_row() {
        let row = row_for(PartitionKind::Rd, Some(200.0), None);
        let labels: Vec<&str> = row.categories.iter().map(|c| c.category.as_str()).collect();
        let expected: Vec<&str> = Category::ALL.iter().map(|c| c.label()).collect();
        assert_eq!(labels, expected);
        assert_eq!(row.categories.len(), 5);
        // Every event the stream emitted is attributed to exactly one category.
        let events: u64 = row.categories.iter().map(|c| c.events).sum();
        assert_eq!(events + row.unattributed_events, row.events);
        assert_eq!(row.unattributed_events, 0);
    }

    #[test]
    fn a_row_records_the_estimate_error_and_the_objective_it_belongs_to() {
        let rd = row_for(PartitionKind::Rd, Some(200.0), None);
        assert_eq!(rd.estimated_provenance, "rd_warmup_snapshot");
        assert!((rd.estimate_error_bits - (rd.estimated_bits - rd.actual_event_bits)).abs() < 1e-9);
        assert!(rd.estimated_bits > 0.0);
        assert!(rd.inner_stream_bits >= rd.actual_event_bits);

        let threshold = row_for(PartitionKind::Threshold, None, Some(8.0));
        assert_eq!(threshold.estimated_provenance, "reference_only");
        assert_eq!(
            threshold.warmup_evals, 0,
            "the threshold walk pays no warm-up"
        );
        assert!(threshold.notes.contains("never consulted"));
    }

    #[test]
    fn markdown_is_identical_for_any_insertion_order_and_names_both_partitions() {
        let rd = row_for(PartitionKind::Rd, Some(200.0), None);
        let threshold = row_for(PartitionKind::Threshold, None, Some(8.0));
        let forward = render_markdown(&[rd.clone(), threshold.clone()]);
        let backward = render_markdown(&[threshold, rd]);
        assert_eq!(forward, backward);
        assert!(forward.contains("| synthetic | rd | λ=200 |"));
        assert!(forward.contains("| synthetic | threshold | t_rms=8 |"));
        assert!(forward.contains("## Category breakdown"));
        assert!(forward.contains("### synthetic · rd · λ=200"));
        assert!(forward.contains("| partition |"));
        assert!(forward.contains("| category | events | est bits | actual bits | error % |"));
    }

    #[test]
    fn by_image_groups_rows_in_a_stable_order() {
        let rd = row_for(PartitionKind::Rd, Some(200.0), None);
        let threshold = row_for(PartitionKind::Threshold, None, Some(8.0));
        let rows = [rd, threshold];
        let grouped = by_image(&rows);
        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped["synthetic"].len(), 2);
    }

    #[test]
    fn empty_grid_and_invalid_parameters_are_refused() {
        let config = AuditConfig {
            images: vec![image()],
            lambdas: Vec::new(),
            thresholds: Vec::new(),
            modes: [true, false, true, false],
            mode_numbers: vec![0, 2],
            codec: AuditCodec::default(),
            iterations: 10,
            scratch: std::env::temp_dir().join("mars-rate-audit-empty"),
        };
        let error = run(Path::new("."), &config, Path::new("/tmp/unused.jsonl"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("no partition policies"), "{error}");

        let config = AuditConfig {
            lambdas: vec![f64::NAN],
            thresholds: Vec::new(),
            ..config
        };
        let error = run(Path::new("."), &config, Path::new("/tmp/unused.jsonl"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("finite and nonnegative"), "{error}");
    }
}
