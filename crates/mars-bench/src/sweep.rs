//! The Mars 1 baseline sweep (Step 2).
//!
//! §2.3: *every experiment is a config file, not a code edit*. The grid lives in
//! `configs/baseline-mars1.json`; this module reads it, expands it into a deterministic
//! list of jobs, runs them, and appends one self-contained row per
//! `(image, variant, method, rate, decode mode)` to the append-only store.
//!
//! **The grid is one-at-a-time around a base point, not a full cross product.** The plan's
//! table reads "`(4,16)` default; also `(2,16)`, `(4,32)`" — alternatives to a default,
//! not axes to multiply. Crossing all of them would be 720 configurations per image for
//! no extra information: the structural parameters are being probed for their individual
//! effect on the RD curve, and the six methods × five rates already carry the headline.
//! See `docs/decisions.md` D6.
//!
//! **Row order is deterministic.** Jobs are executed by a pool of worker threads, but
//! results are buffered and appended in job order, so re-running the sweep on the same
//! inputs produces a byte-comparable file regardless of `--jobs` (§2.3).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::mars1::{
    decode, encode, DecodeMode, DecodeParams, EncodeParams, Mars1Binaries, Mars1Error, Method,
};
use crate::measure::{measure, MeasureRequest};
use crate::provenance::Provenance;
use crate::store::{ResultStore, Row};

/// The `kind` every baseline row carries.
pub const ROW_KIND: &str = "baseline-mars1";

/// Recorded verbatim in every row so no reader can mistake these seconds for a §M4
/// timing result: one run, no interleaving, no median, siblings on other cores.
pub const TIMING_PROTOCOL: &str =
    "indicative-single-run; NOT a §M4 timing result (no interleaving, no median, concurrent jobs)";

// ---------------------------------------------------------------- the config file

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SweepConfig {
    pub name: String,
    pub description: String,
    /// Both modes are run for every encode; the mode is recorded on the row because
    /// comparing a pyramidal decode against an iterative one is a silent 0.2–1 dB error.
    pub decode_modes: Vec<DecodeMode>,
    pub methods: Vec<Method>,
    /// The `-r` values that make the RD curve.
    pub rms: Vec<f64>,
    pub base: BaseParams,
    pub variants: Vec<Variant>,
    pub corpora: Vec<CorpusSpec>,
}

/// The structural parameters at their 1998 defaults, from which variants deviate.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct BaseParams {
    pub min_size: u32,
    pub max_size: u32,
    pub shift: u32,
    pub bits_alfa: u32,
    pub bits_beta: u32,
    pub max_alfa: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Variant {
    pub label: String,
    pub description: String,
    #[serde(default)]
    pub min_size: Option<u32>,
    #[serde(default)]
    pub max_size: Option<u32>,
    #[serde(default)]
    pub shift: Option<u32>,
    #[serde(default)]
    pub bits_alfa: Option<u32>,
    #[serde(default)]
    pub bits_beta: Option<u32>,
    #[serde(default)]
    pub max_alfa: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CorpusSpec {
    /// Which image-set index to read, e.g. `corpus/standard.images.json`.
    pub index: PathBuf,
    /// Variant labels to run on this corpus. The fixtures set runs the default point
    /// only: it exists to exercise edge cases, not to carry RD numbers (§M9).
    pub variants: Vec<String>,
}

impl SweepConfig {
    pub fn read(path: &Path) -> Result<Self, SweepError> {
        let text = std::fs::read_to_string(path).map_err(|source| SweepError::Io {
            path: path.display().to_string(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| SweepError::Config {
            path: path.display().to_string(),
            source,
        })
    }

    fn variant(&self, label: &str) -> Option<&Variant> {
        self.variants.iter().find(|v| v.label == label)
    }

    fn params(&self, variant: &Variant, method: Method, t_rms: f64) -> EncodeParams {
        EncodeParams {
            method,
            t_rms,
            min_size: variant.min_size.unwrap_or(self.base.min_size),
            max_size: variant.max_size.unwrap_or(self.base.max_size),
            shift: variant.shift.unwrap_or(self.base.shift),
            bits_alfa: variant.bits_alfa.unwrap_or(self.base.bits_alfa),
            bits_beta: variant.bits_beta.unwrap_or(self.base.bits_beta),
            max_alfa: variant.max_alfa.unwrap_or(self.base.max_alfa),
            // Left at the 1998 defaults throughout. P2.1 predicts they are inert there;
            // `tests/mars1_reference.rs` tests that claim directly rather than spending
            // 4320 encodes on it.
            t_ent: None,
            t_var: None,
        }
    }

    /// Reject a config that cannot produce a §M3-valid comparison, before spending hours
    /// discovering it. A sweep that silently yields three-point curves is worse than one
    /// that refuses to start.
    pub fn validate(&self) -> Result<(), SweepError> {
        if self.rms.len() < crate::bdrate::MIN_POINTS_PER_CURVE {
            return Err(SweepError::TooFewRates {
                got: self.rms.len(),
                need: crate::bdrate::MIN_POINTS_PER_CURVE,
            });
        }
        if self.methods.is_empty() || self.decode_modes.is_empty() || self.corpora.is_empty() {
            return Err(SweepError::EmptyAxis);
        }
        let labels: BTreeSet<&str> = self.variants.iter().map(|v| v.label.as_str()).collect();
        if labels.len() != self.variants.len() {
            return Err(SweepError::DuplicateVariant);
        }
        for c in &self.corpora {
            for want in &c.variants {
                if !labels.contains(want.as_str()) {
                    return Err(SweepError::UnknownVariant {
                        label: want.clone(),
                        index: c.index.display().to_string(),
                    });
                }
            }
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ the image sets

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageSet {
    pub set: String,
    pub role: String,
    pub provenance: serde_json::Value,
    pub images: Vec<ImageEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageEntry {
    pub name: String,
    /// Repo-relative path to the headerless 8-bit grayscale raw.
    pub file: PathBuf,
    pub width: u32,
    pub height: u32,
    pub sha256: String,
}

impl ImageSet {
    pub fn read(path: &Path) -> Result<Self, SweepError> {
        let text = std::fs::read_to_string(path).map_err(|source| SweepError::Io {
            path: path.display().to_string(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| SweepError::Config {
            path: path.display().to_string(),
            source,
        })
    }
}

// ------------------------------------------------------------------------- planning

/// One encode, plus the decodes that hang off it.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: usize,
    pub corpus: String,
    pub image: ImageEntry,
    pub variant: String,
    pub params: EncodeParams,
}

/// Expand the config into jobs, in a fixed order: corpus, then image, then variant, then
/// method, then rate. The order is part of the output (row order is this order), so it
/// is spelled out rather than left to whatever iteration happens to produce.
pub fn plan(config: &SweepConfig, sets: &[(CorpusSpec, ImageSet)]) -> Result<Vec<Job>, SweepError> {
    config.validate()?;
    let mut jobs = Vec::new();
    for (spec, set) in sets {
        for image in &set.images {
            for label in &spec.variants {
                let variant = config
                    .variant(label)
                    .ok_or_else(|| SweepError::UnknownVariant {
                        label: label.clone(),
                        index: spec.index.display().to_string(),
                    })?;
                for &method in &config.methods {
                    for &t_rms in &config.rms {
                        jobs.push(Job {
                            id: jobs.len(),
                            corpus: set.set.clone(),
                            image: image.clone(),
                            variant: label.clone(),
                            params: config.params(variant, method, t_rms),
                        });
                    }
                }
            }
        }
    }
    Ok(jobs)
}

// -------------------------------------------------------------------- the row payload

/// The payload of a `baseline-mars1` row. Self-contained per §M7: everything needed to
/// re-run this exact measurement is here, including the image hash and the compiler that
/// built the reference binaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineRow {
    pub sweep: String,
    pub corpus: String,
    pub image: String,
    pub image_sha256: String,
    pub width: u32,
    pub height: u32,
    pub variant: String,
    pub params: EncodeParams,
    /// `-r`, lifted out of `params` so a reader can group by rate without digging.
    pub t_rms: f64,
    /// The method key, lifted out for the same reason.
    pub method: String,
    pub encode: EncodeRow,
    pub decode_mode: DecodeMode,
    pub quality: crate::measure::Measurement,
    pub indicative_encode_seconds: f64,
    pub indicative_decode_seconds: f64,
    pub timing_protocol: String,
    /// Concurrent jobs at the time, so a reader knows how contended the machine was.
    pub jobs: usize,
    pub mars1_build_info: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeRow {
    /// Leaf blocks emitted (§M5).
    pub transforms: u64,
    /// Mars 1's `evals` analogue (§M5).
    pub comparisons: u64,
    /// The headline metric (§M5), computed from the two integers above.
    pub evals_per_transform: f64,
    pub zero_alfa_transforms: u64,
    /// Whole-file size; the bpp in `quality` is derived from this (§M2).
    pub coded_bytes: u64,
    pub image_entropy: f64,
    pub image_variance: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum SweepError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing {path}: {source}")]
    Config {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("the sweep declares {got} rate points; §M3 requires at least {need} per curve")]
    TooFewRates { got: usize, need: usize },
    #[error("the sweep has an empty methods, decode_modes or corpora list")]
    EmptyAxis,
    #[error("two variants share a label")]
    DuplicateVariant,
    #[error("corpus {index} asks for variant {label:?}, which the config does not define")]
    UnknownVariant { label: String, index: String },
    #[error("job {id} produced no rows and reported no error; the sweep is incomplete")]
    MissingResult { id: usize },
    #[error(transparent)]
    Mars1(#[from] Mars1Error),
    #[error(transparent)]
    Measure(#[from] crate::measure::MeasureError),
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
    #[error("serialising a row: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("job {id} ({image} {variant} {method} r={t_rms}) failed: {source}")]
    Job {
        id: usize,
        image: String,
        variant: String,
        method: String,
        t_rms: f64,
        #[source]
        source: Box<SweepError>,
    },
}

// ------------------------------------------------------------------------ execution

/// Everything a worker needs that does not vary per job.
pub struct RunContext<'a> {
    pub bins: &'a Mars1Binaries,
    pub repo_root: &'a Path,
    /// A short directory (D3) under which each worker gets its own subdirectory.
    pub scratch: &'a Path,
    pub decode_modes: &'a [DecodeMode],
    pub jobs: usize,
    pub sweep_name: &'a str,
}

/// Run one job: one encode, then one decode per mode, then one measurement per decode.
///
/// Filenames inside the worker directory are deliberately tiny (`i.raw`, `o.ifs`) —
/// `globals.h` gives the reference a `char[50]` for each, and an overflow presents as
/// "Can't open output file" rather than as a crash (D3).
fn run_job(ctx: &RunContext, workdir: &Path, job: &Job) -> Result<Vec<BaselineRow>, SweepError> {
    let src = ctx.repo_root.join(&job.image.file);
    let input = workdir.join("i.raw");
    std::fs::copy(&src, &input).map_err(|source| SweepError::Io {
        path: src.display().to_string(),
        source,
    })?;

    let enc = encode(
        ctx.bins,
        workdir,
        "i.raw",
        "o.ifs",
        job.image.width,
        job.image.height,
        &job.params,
    )?;

    let mut rows = Vec::with_capacity(ctx.decode_modes.len());
    for &mode in ctx.decode_modes {
        let dec_params = DecodeParams {
            mode,
            iterations: None,
            postprocess: false,
        };
        let out_name = "d.pgm";
        let dec = decode(
            ctx.bins,
            workdir,
            "o.ifs",
            out_name,
            (job.image.width, job.image.height),
            &dec_params,
        )?;

        // §M1: quality is computed by the harness from the two files on disk. The codec
        // reports its own size, not its own PSNR — and even the size is cross-checked.
        let req = MeasureRequest::new(&input, workdir.join(out_name))
            .with_raw_dims(job.image.width as usize, job.image.height as usize)
            .with_coded_bytes(enc.coded_bytes);
        let quality = measure(&req)?;

        rows.push(BaselineRow {
            sweep: ctx.sweep_name.to_string(),
            corpus: job.corpus.clone(),
            image: job.image.name.clone(),
            image_sha256: job.image.sha256.clone(),
            width: job.image.width,
            height: job.image.height,
            variant: job.variant.clone(),
            t_rms: job.params.t_rms,
            method: job.params.method.key().to_string(),
            params: job.params.clone(),
            encode: EncodeRow {
                transforms: enc.stats.transforms,
                comparisons: enc.stats.comparisons,
                evals_per_transform: enc.stats.evals_per_transform(),
                zero_alfa_transforms: enc.stats.zero_alfa_transforms,
                coded_bytes: enc.coded_bytes,
                image_entropy: enc.stats.image_entropy,
                image_variance: enc.stats.image_variance,
            },
            decode_mode: mode,
            quality,
            indicative_encode_seconds: enc.indicative_seconds,
            indicative_decode_seconds: dec.indicative_seconds,
            timing_protocol: TIMING_PROTOCOL.to_string(),
            jobs: ctx.jobs,
            mars1_build_info: ctx.bins.build_info.clone(),
        });
    }
    Ok(rows)
}

/// Run every job across `ctx.jobs` worker threads and append the rows **in job order**.
///
/// Buffering rather than streaming costs memory proportional to the sweep and buys a
/// result file that does not depend on thread scheduling — the same property §2.3
/// demands of the codec, applied to the harness that judges it.
pub fn run(
    ctx: &RunContext,
    jobs: &[Job],
    provenance: &Provenance,
    store: &mut ResultStore,
    progress: &(dyn Fn(usize, usize, &Job) + Sync),
) -> Result<usize, SweepError> {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Vec<BaselineRow>>>> =
        Mutex::new((0..jobs.len()).map(|_| None).collect());
    let first_error: Mutex<Option<SweepError>> = Mutex::new(None);

    std::thread::scope(|scope| {
        for worker in 0..ctx.jobs.max(1) {
            let next = &next;
            let done = &done;
            let results = &results;
            let first_error = &first_error;
            scope.spawn(move || {
                let workdir = ctx.scratch.join(format!("w{worker}"));
                if let Err(source) = std::fs::create_dir_all(&workdir) {
                    // Returning quietly here would leave this worker's share of the jobs
                    // unrun and the result file merely *short* — the exact shape of
                    // silent loss the completeness check exists to catch, arriving an
                    // hour later. Fail the sweep instead.
                    let mut slot = first_error.lock().expect("poisoned");
                    if slot.is_none() {
                        *slot = Some(SweepError::Io {
                            path: workdir.display().to_string(),
                            source,
                        });
                    }
                    return;
                }
                loop {
                    // A stop flag rather than a race: once one job has failed, finishing
                    // the other 4000 only buries the message.
                    if first_error.lock().expect("poisoned").is_some() {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(job) = jobs.get(i) else { return };
                    match run_job(ctx, &workdir, job) {
                        Ok(rows) => {
                            results.lock().expect("poisoned")[i] = Some(rows);
                            let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                            progress(n, jobs.len(), job);
                        }
                        Err(source) => {
                            let mut slot = first_error.lock().expect("poisoned");
                            if slot.is_none() {
                                *slot = Some(SweepError::Job {
                                    id: job.id,
                                    image: job.image.name.clone(),
                                    variant: job.variant.clone(),
                                    method: job.params.method.key().to_string(),
                                    t_rms: job.params.t_rms,
                                    source: Box::new(source),
                                });
                            }
                            return;
                        }
                    }
                }
            });
        }
    });

    if let Some(e) = first_error.lock().expect("poisoned").take() {
        return Err(e);
    }

    let results = results.into_inner().expect("poisoned");
    let mut written = 0;
    for (i, slot) in results.into_iter().enumerate() {
        // No error was recorded, so every job must have produced rows. A `None` here
        // would mean a job vanished, and writing the rest would produce a file that looks
        // complete and is not.
        let payloads = slot.ok_or(SweepError::MissingResult { id: i })?;
        for payload in payloads {
            store.append(&Row::new(ROW_KIND, provenance.clone(), &payload)?)?;
            written += 1;
        }
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SweepConfig {
        serde_json::from_str(
            r#"{
              "name": "t", "description": "d",
              "decode_modes": ["pyramidal", "iterative"],
              "methods": ["fisher", "saupe"],
              "rms": [2, 4, 8, 16],
              "base": {"min_size":4,"max_size":16,"shift":4,"bits_alfa":4,"bits_beta":7,"max_alfa":1.0},
              "variants": [
                {"label":"default","description":"1998 defaults"},
                {"label":"min2","description":"min 2","min_size":2}
              ],
              "corpora": [{"index":"corpus/standard.images.json","variants":["default","min2"]}]
            }"#,
        )
        .unwrap()
    }

    fn image_set(names: &[&str]) -> ImageSet {
        ImageSet {
            set: "standard".into(),
            role: "r".into(),
            provenance: serde_json::json!({}),
            images: names
                .iter()
                .map(|n| ImageEntry {
                    name: (*n).into(),
                    file: PathBuf::from(format!("{n}.raw")),
                    width: 768,
                    height: 512,
                    sha256: "0".repeat(64),
                })
                .collect(),
        }
    }

    #[test]
    fn the_plan_is_the_full_grid_in_a_fixed_order() {
        let c = config();
        let sets = vec![(c.corpora[0].clone(), image_set(&["a", "b"]))];
        let jobs = plan(&c, &sets).unwrap();
        // 2 images x 2 variants x 2 methods x 4 rates.
        assert_eq!(jobs.len(), 32);
        // Order: image, variant, method, rate.
        assert_eq!(jobs[0].image.name, "a");
        assert_eq!(jobs[0].variant, "default");
        assert_eq!(jobs[0].params.method, Method::Fisher);
        assert_eq!(jobs[0].params.t_rms, 2.0);
        assert_eq!(jobs[3].params.t_rms, 16.0);
        assert_eq!(jobs[4].params.method, Method::Saupe);
        assert_eq!(jobs[8].variant, "min2");
        assert_eq!(jobs[16].image.name, "b");
        // Ids are dense and ascending, since row order is job order.
        assert!(jobs.iter().enumerate().all(|(i, j)| j.id == i));
        // Planning is a pure function of the inputs.
        let again = plan(&c, &sets).unwrap();
        assert!(jobs
            .iter()
            .zip(&again)
            .all(|(x, y)| x.params == y.params && x.image.name == y.image.name));
    }

    #[test]
    fn a_variant_overrides_only_what_it_names() {
        let c = config();
        let sets = vec![(c.corpora[0].clone(), image_set(&["a"]))];
        let jobs = plan(&c, &sets).unwrap();
        let d = &jobs.iter().find(|j| j.variant == "default").unwrap().params;
        let m = &jobs.iter().find(|j| j.variant == "min2").unwrap().params;
        assert_eq!(d.min_size, 4);
        assert_eq!(m.min_size, 2);
        // Everything else is untouched — a variant is a one-parameter deviation (D6).
        assert_eq!(
            (m.max_size, m.shift, m.bits_alfa, m.bits_beta),
            (16, 4, 4, 7)
        );
        // The pre-split thresholds stay at the 1998 defaults throughout the sweep.
        assert_eq!(d.t_ent, None);
        assert_eq!(d.t_var, None);
    }

    #[test]
    fn a_config_that_cannot_make_a_valid_curve_is_refused() {
        // §M3 needs four points; three would produce a file full of unquotable numbers.
        let mut c = config();
        c.rms = vec![2.0, 8.0, 32.0];
        assert!(matches!(
            c.validate(),
            Err(SweepError::TooFewRates { got: 3, need: 4 })
        ));
    }

    #[test]
    fn a_corpus_naming_an_undefined_variant_is_refused() {
        let mut c = config();
        c.corpora[0].variants.push("nope".into());
        assert!(matches!(
            c.validate(),
            Err(SweepError::UnknownVariant { .. })
        ));
    }

    #[test]
    fn duplicate_variant_labels_are_refused() {
        let mut c = config();
        let dup = c.variants[0].clone();
        c.variants.push(dup);
        assert!(matches!(c.validate(), Err(SweepError::DuplicateVariant)));
    }
}
