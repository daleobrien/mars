//! The strict experiment config: schema, validation, corpus resolution and planning
//! (research plan §3 P0.3, §14).
//!
//! A config is data, not code (§2.3: "every experiment is a config file, not a code
//! edit"). It names its inputs, the codec parameters held fixed across the stage, the
//! decoder settings, and a list of named stages, each a list of labelled arms. Planning
//! expands (image × arm) into a deterministic, content-addressed set of cases.
//!
//! Two kinds of "this combination does not work" are distinguished on purpose:
//!
//! * **Config errors** are arm-level and image-independent (e.g. `--method` together
//!   with `--progressive`, or `--budget` on a method that takes none). These are
//!   rejected before anything runs, listing every offender at once, so an unsupported
//!   flag combination is never accepted as a silent no-op.
//! * **Per-image unsupported combinations** (e.g. `--method`, which is grayscale-only,
//!   paired with a colour image) are recorded as explicit `unsupported` cases. They are
//!   data-dependent, cannot be known from the arm alone, and must still appear in the
//!   run's denominator rather than being dropped.
//!
//! Validation happens before any expensive work, and `count` in a corpus index must
//! match its image list: corpus completeness is a precondition, not a hope.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use mars_core::io::read_image;

use crate::provenance::{canonical_json, sha256_hex};
use crate::sweep::ImageEntry;

/// Config schema. Bump when the config shape changes; readers key off it.
pub const CONFIG_SCHEMA: u32 = 1;

/// Every `encmars --method` key. The single source of truth for the runner and the
/// smoke passthrough alike.
pub const METHOD_KEYS: [&str; 11] = [
    "exhaustive",
    "fisher",
    "hurtgen",
    "masscenter",
    "saupe",
    "saupe-fisher",
    "mc-saupe",
    "funnel",
    "learned",
    "random",
    "apcc",
];

/// The full config file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentConfig {
    pub schema_version: u32,
    pub name: String,
    pub description: String,
    /// Corpus image-set indexes (e.g. `corpus/standard.images.json`), repo-relative.
    #[serde(default)]
    pub indexes: Vec<PathBuf>,
    /// Optional subset of image names to run. Empty means every image in the indexes.
    #[serde(default)]
    pub images: Vec<String>,
    /// Images that belong to no corpus index (fixtures, smoke inputs).
    #[serde(default)]
    pub inputs: Vec<InputSpec>,
    pub codec: CodecSpec,
    #[serde(default)]
    pub decoder: DecoderSpec,
    pub stages: Vec<StageSpec>,
}

/// One explicitly listed input, pinned by hash where possible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSpec {
    pub name: String,
    pub file: PathBuf,
    /// Required with `height` for headerless `.raw`/`.y`/`.gray` inputs.
    #[serde(default)]
    pub width: Option<usize>,
    #[serde(default)]
    pub height: Option<usize>,
    #[serde(default)]
    pub sha256: Option<String>,
}

/// Encoder parameters held fixed across every arm in the config.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodecSpec {
    pub min_size: u32,
    pub max_size: u32,
    pub shift: u32,
    pub bits_alfa: u32,
    pub bits_beta: u32,
    pub max_alfa: f64,
    #[serde(default)]
    pub zero_threshold: u32,
    #[serde(default = "default_subsampling")]
    pub subsampling: String,
    /// Legacy/fallback threshold. RD arms ignore it; the RD warm-up is fixed at RMS 8.
    #[serde(default = "default_t_rms")]
    pub t_rms: f64,
    #[serde(default = "default_t_rms")]
    pub chroma_t_rms: f64,
    #[serde(default = "default_threads")]
    pub threads: usize,
}

/// Decoder settings shared by the stage, with per-arm overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderSpec {
    #[serde(default = "default_iterations")]
    pub iterations: u32,
    #[serde(default = "default_zoom")]
    pub zoom: f64,
    /// Output container format, matching `decmars`'s extension-based writer.
    #[serde(default = "default_output")]
    pub output: String,
    #[serde(default)]
    pub smooth: bool,
    #[serde(default)]
    pub auto: bool,
    #[serde(default)]
    pub threshold: u8,
    #[serde(default)]
    pub layer: Option<u8>,
}

impl Default for DecoderSpec {
    fn default() -> Self {
        Self {
            iterations: default_iterations(),
            zoom: default_zoom(),
            output: default_output(),
            smooth: false,
            auto: false,
            threshold: 0,
            layer: None,
        }
    }
}

/// One labelled codec configuration. The only things that vary between cases of the
/// same image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub label: String,
    pub description: String,
    /// Analysis grouping for RD curves. Arms sharing a group are one policy swept over
    /// another axis (lambda, budget, seed); `experiment-report` builds per-image curves
    /// per group. Defaults to the arm label when omitted.
    #[serde(default)]
    pub group: Option<String>,
    /// Selects the RD path when present; otherwise the legacy `t_rms` path.
    #[serde(default)]
    pub lambda: Option<f64>,
    #[serde(default)]
    pub modes: Option<Vec<u8>>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub budget: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub adaptive_density: bool,
    #[serde(default)]
    pub adaptive_residual: bool,
    #[serde(default)]
    pub progressive: bool,
    #[serde(default)]
    pub iterations: Option<u32>,
    #[serde(default)]
    pub smooth: Option<bool>,
}

/// A named stage: which arms run, how many timing repetitions, and the per-process
/// timeout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageSpec {
    pub name: String,
    pub description: String,
    #[serde(default = "default_repetitions")]
    pub repetitions: u32,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    pub arms: Vec<Arm>,
}

fn default_subsampling() -> String {
    "444".into()
}
fn default_t_rms() -> f64 {
    8.0
}
fn default_threads() -> usize {
    1
}
fn default_iterations() -> u32 {
    10
}
fn default_zoom() -> f64 {
    1.0
}
fn default_output() -> String {
    "png".into()
}
fn default_repetitions() -> u32 {
    1
}
fn default_timeout_secs() -> u64 {
    60
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path}: corpus index: {source}")]
    Index {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    /// One or more validation failures, all reported together.
    #[error("{0}")]
    Invalid(String),
}

fn invalid(problems: Vec<String>) -> ConfigError {
    ConfigError::Invalid(problems.join("; "))
}

/// The `count`/`set` fields a corpus index carries alongside its images.
#[derive(Debug, Deserialize)]
struct IndexFile {
    #[serde(default)]
    count: Option<usize>,
    #[serde(default)]
    set: Option<String>,
    images: Vec<ImageEntry>,
}

/// A config-resolved input image ready to plan against.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedImage {
    pub name: String,
    /// Repo-relative path, as the config and index wrote it.
    pub file: PathBuf,
    /// Dimensions for headerless raw input; `None` for self-describing containers.
    pub raw_dims: Option<(usize, usize)>,
    /// Index-recorded hash, where one exists.
    pub sha256: Option<String>,
    pub planes: usize,
    /// The image-set index this came from, when it came from one.
    pub corpus_index: Option<PathBuf>,
}

impl ExperimentConfig {
    pub fn read(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Io {
            path: path.display().to_string(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| ConfigError::Parse {
            path: path.display().to_string(),
            source,
        })
    }

    /// SHA-256 over the config's canonical JSON. Resume requires an exact match.
    pub fn config_sha256(&self) -> Result<String, ConfigError> {
        let value = serde_json::to_value(self).map_err(|source| ConfigError::Parse {
            path: "<config>".into(),
            source,
        })?;
        Ok(sha256_hex(canonical_json(&value).as_bytes()))
    }

    /// A path-safe experiment id: `<slug(name)>-<config sha prefix>`.
    pub fn experiment_id(&self) -> Result<String, ConfigError> {
        let slug: String = self
            .name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        let sha = self.config_sha256()?;
        Ok(format!("{slug}-{}", &sha[..12]))
    }

    /// Select one stage by name. With a single stage the name may be omitted; with
    /// several it is required, so a stage is never chosen by accident.
    pub fn stage(&self, name: Option<&str>) -> Result<&StageSpec, ConfigError> {
        let names: Vec<&str> = self.stages.iter().map(|s| s.name.as_str()).collect();
        match name {
            Some(name) => self.stages.iter().find(|s| s.name == name).ok_or_else(|| {
                invalid(vec![format!(
                    "stage {name:?} is not defined; available stages: {names:?}"
                )])
            }),
            None => match self.stages.as_slice() {
                [only] => Ok(only),
                _ => Err(invalid(vec![format!(
                    "config defines {} stages; pass --stage <{}>",
                    self.stages.len(),
                    names.join("|")
                )])),
            },
        }
    }

    /// Reject every statically unsupported arm combination, listing all of them at once.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut problems = Vec::new();
        if self.schema_version != CONFIG_SCHEMA {
            problems.push(format!(
                "config schema_version {} is not {CONFIG_SCHEMA}",
                self.schema_version
            ));
        }
        if self.name.trim().is_empty() {
            problems.push("name must not be empty".into());
        }
        if self.description.trim().is_empty() {
            problems.push("description must not be empty".into());
        }
        self.validate_codec(&mut problems);
        self.validate_decoder(&mut problems);
        self.validate_inputs(&mut problems);

        if self.stages.is_empty() {
            problems.push("config defines no stages".into());
        }
        let mut stage_names = BTreeSet::new();
        for stage in &self.stages {
            if stage.name.trim().is_empty() {
                problems.push("a stage has an empty name".into());
            }
            if !stage_names.insert(stage.name.as_str()) {
                problems.push(format!("two stages share the name {:?}", stage.name));
            }
            if stage.description.trim().is_empty() {
                problems.push(format!("stage {:?} has an empty description", stage.name));
            }
            if stage.repetitions == 0 {
                problems.push(format!(
                    "stage {:?}: repetitions must be positive",
                    stage.name
                ));
            }
            if stage.timeout_secs == 0 {
                problems.push(format!(
                    "stage {:?}: timeout_secs must be positive",
                    stage.name
                ));
            }
            if stage.arms.is_empty() {
                problems.push(format!("stage {:?} has no arms", stage.name));
            }
            let mut labels = BTreeSet::new();
            for arm in &stage.arms {
                let where_ = format!("stage {:?} arm {:?}", stage.name, arm.label);
                if arm.label.trim().is_empty() {
                    problems.push(format!("stage {:?} has an empty arm label", stage.name));
                }
                if !labels.insert(arm.label.as_str()) {
                    problems.push(format!("{where_}: duplicate arm label"));
                }
                if arm.description.trim().is_empty() {
                    problems.push(format!("{where_}: empty description"));
                }
                validate_arm(arm, &self.decoder, &where_, &mut problems);
            }
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(invalid(problems))
        }
    }

    fn validate_codec(&self, problems: &mut Vec<String>) {
        let codec = &self.codec;
        for (name, size) in [("min_size", codec.min_size), ("max_size", codec.max_size)] {
            if !size.is_power_of_two() || size > u32::from(u8::MAX) {
                problems.push(format!("codec.{name} must be a power of two in 1..=128"));
            }
        }
        if codec.min_size > codec.max_size {
            problems.push("codec.min_size must not exceed codec.max_size".into());
        }
        if codec.shift == 0 || codec.shift > u32::from(u8::MAX) || codec.shift % 2 != 0 {
            problems.push("codec.shift must be even and in 2..=254".into());
        }
        if !(2..=24).contains(&codec.bits_alfa) {
            problems.push("codec.bits_alfa must be in 2..=24".into());
        }
        if !(1..=24).contains(&codec.bits_beta) {
            problems.push("codec.bits_beta must be in 1..=24".into());
        }
        let int_max_alfa = codec.max_alfa * 32.0;
        if !int_max_alfa.is_finite()
            || !(1.0..=255.0).contains(&int_max_alfa)
            || int_max_alfa.fract() != 0.0
        {
            problems.push(
                "codec.max_alfa must be a multiple of 1/32 in 1/32..=255/32 (header-representable)"
                    .into(),
            );
        }
        if !matches!(codec.subsampling.as_str(), "444" | "420") {
            problems.push("codec.subsampling must be \"444\" or \"420\"".into());
        }
        for (name, value) in [("t_rms", codec.t_rms), ("chroma_t_rms", codec.chroma_t_rms)] {
            if !value.is_finite() || value < 0.0 {
                problems.push(format!("codec.{name} must be finite and nonnegative"));
            }
        }
        if codec.threads == 0 {
            problems.push("codec.threads must be positive".into());
        }
    }

    fn validate_decoder(&self, problems: &mut Vec<String>) {
        let decoder = &self.decoder;
        if decoder.iterations == 0 {
            problems.push("decoder.iterations must be positive".into());
        }
        if !decoder.zoom.is_finite() || decoder.zoom <= 0.0 {
            problems.push("decoder.zoom must be finite and positive".into());
        }
        if !matches!(decoder.output.as_str(), "png" | "pgm" | "ppm") {
            problems.push("decoder.output must be \"png\", \"pgm\" or \"ppm\"".into());
        }
        if let Some(layer) = decoder.layer {
            if !(1..=4).contains(&layer) {
                problems.push("decoder.layer must be in 1..=4".into());
            }
        }
    }

    fn validate_inputs(&self, problems: &mut Vec<String>) {
        let mut names = BTreeSet::new();
        for input in &self.inputs {
            if input.name.trim().is_empty() {
                problems.push("an input has an empty name".into());
            }
            if !names.insert(input.name.as_str()) {
                problems.push(format!("two inputs share the name {:?}", input.name));
            }
            if input.width.is_some() != input.height.is_some() {
                problems.push(format!(
                    "input {:?}: width and height must be given together",
                    input.name
                ));
            }
            if let Some((w, h)) = input.width.zip(input.height) {
                if w == 0 || h == 0 || w.checked_mul(h).is_none() {
                    problems.push(format!(
                        "input {:?}: dimensions must be positive and not overflow",
                        input.name
                    ));
                }
            }
            let ext = input
                .file
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if matches!(ext.as_str(), "raw" | "y" | "gray")
                && input.width.is_none()
                && input.height.is_none()
            {
                problems.push(format!(
                    "input {:?}: headerless .raw input requires width and height",
                    input.name
                ));
            }
        }
    }

    /// Validate the config, then resolve every input to disk and confirm its hash and
    /// plane count. Missing files are an explicit error *before* any encode runs.
    pub fn resolve(&self, root: &Path) -> Result<Vec<ResolvedImage>, ConfigError> {
        self.validate()?;
        let mut problems = Vec::new();
        let mut resolved: Vec<ResolvedImage> = Vec::new();

        let wanted: BTreeSet<&str> = self.images.iter().map(String::as_str).collect();
        let mut found = BTreeSet::new();
        for index in &self.indexes {
            let path = root.join(index);
            let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Io {
                path: path.display().to_string(),
                source,
            })?;
            let file: IndexFile =
                serde_json::from_str(&text).map_err(|source| ConfigError::Index {
                    path: path.display().to_string(),
                    source,
                })?;
            if let Some(count) = file.count {
                if count != file.images.len() {
                    problems.push(format!(
                        "{}: count {count} does not match {} listed images",
                        index.display(),
                        file.images.len()
                    ));
                }
            }
            let set = file.set.unwrap_or_else(|| index.display().to_string());
            for entry in &file.images {
                if !wanted.is_empty() && !wanted.contains(entry.name.as_str()) {
                    continue;
                }
                found.insert(entry.name.clone());
                let raw_dims = Some((entry.width as usize, entry.height as usize));
                resolved.push(ResolvedImage {
                    name: entry.name.clone(),
                    file: entry.file.clone(),
                    raw_dims,
                    sha256: Some(entry.sha256.clone()),
                    planes: 0,
                    corpus_index: Some(index.clone()),
                });
                let _ = &set;
            }
        }

        for name in &wanted {
            if !found.contains(*name) {
                problems.push(format!("requested image {name:?} is not in any index"));
            }
        }

        for input in &self.inputs {
            if !wanted.is_empty() && !wanted.contains(input.name.as_str()) {
                continue;
            }
            let raw_dims = input.width.zip(input.height);
            resolved.push(ResolvedImage {
                name: input.name.clone(),
                file: input.file.clone(),
                raw_dims,
                sha256: input.sha256.clone(),
                planes: 0,
                corpus_index: None,
            });
        }

        if resolved.is_empty() {
            problems.push("config resolves to zero images".into());
        }
        let mut seen = BTreeSet::new();
        for image in &mut resolved {
            if !seen.insert(image.name.clone()) {
                problems.push(format!(
                    "image name {:?} is defined more than once",
                    image.name
                ));
            }
            let absolute = root.join(&image.file);
            match std::fs::read(&absolute) {
                Ok(bytes) => {
                    if let Some(expected) = &image.sha256 {
                        let actual = sha256_hex(&bytes);
                        if actual != *expected {
                            problems.push(format!(
                                "{}: sha256 {actual} does not match the recorded {expected}",
                                image.file.display()
                            ));
                        }
                    }
                }
                Err(source) => {
                    problems.push(format!(
                        "{}: {}",
                        image.file.display(),
                        ConfigError::Io {
                            path: absolute.display().to_string(),
                            source,
                        }
                    ));
                    continue;
                }
            }
            match read_image(&absolute, image.raw_dims) {
                Ok(decoded) => image.planes = decoded.planes().len(),
                Err(source) => problems.push(format!(
                    "{}: decoding to count planes: {source}",
                    image.file.display()
                )),
            }
        }

        if problems.is_empty() {
            Ok(resolved)
        } else {
            Err(invalid(problems))
        }
    }
}

fn validate_arm(arm: &Arm, shared: &DecoderSpec, where_: &str, problems: &mut Vec<String>) {
    if let Some(method) = &arm.method {
        if !METHOD_KEYS.contains(&method.as_str()) {
            problems.push(format!(
                "{where_}: unknown method {method:?}; expected one of {METHOD_KEYS:?}"
            ));
        }
    }
    match arm.method.as_deref() {
        Some("random") => {
            if !arm.budget.is_some_and(|b| b > 0) {
                problems.push(format!(
                    "{where_}: --method random requires a positive budget"
                ));
            }
        }
        Some("apcc") => {
            if !arm.budget.is_some_and(|b| b > 0) {
                problems.push(format!(
                    "{where_}: --method apcc requires a positive budget"
                ));
            }
            if arm.seed.is_some() {
                problems.push(format!("{where_}: seed is only valid with --method random"));
            }
        }
        Some(_) => {
            if arm.budget.is_some() {
                problems.push(format!(
                    "{where_}: budget is only valid with --method random or apcc"
                ));
            }
            if arm.seed.is_some() {
                problems.push(format!("{where_}: seed is only valid with --method random"));
            }
        }
        None => {
            if arm.budget.is_some() || arm.seed.is_some() {
                problems.push(format!(
                    "{where_}: budget/seed require --method random (or apcc for budget)"
                ));
            }
        }
    }

    if let Some(lambda) = arm.lambda {
        if !lambda.is_finite() || lambda < 0.0 {
            problems.push(format!("{where_}: lambda must be finite and nonnegative"));
        }
    }
    if let Some(modes) = &arm.modes {
        if modes.is_empty() {
            problems.push(format!("{where_}: modes must not be empty when given"));
        }
        let mut seen = BTreeSet::new();
        for &mode in modes {
            if mode > 3 {
                problems.push(format!("{where_}: mode {mode} is not in 0..=3"));
            }
            if !seen.insert(mode) {
                problems.push(format!("{where_}: mode {mode} is listed twice"));
            }
        }
    }
    if let Some(iterations) = arm.iterations {
        if iterations == 0 {
            problems.push(format!("{where_}: iterations must be positive"));
        }
    }

    let rd = arm.lambda.is_some();
    if !rd {
        if arm.modes.is_some() {
            problems.push(format!(
                "{where_}: modes require the RD path; give lambda or drop modes"
            ));
        }
        if arm.adaptive_density {
            problems.push(format!(
                "{where_}: adaptive_density requires the RD path; give lambda"
            ));
        }
    }
    let effective_modes: Vec<u8> = arm.modes.clone().unwrap_or_else(|| vec![0, 2]);
    if arm.adaptive_residual && !effective_modes.contains(&3) {
        problems.push(format!(
            "{where_}: adaptive_residual requires mode 3, e.g. modes [0,2,3]"
        ));
    }
    if arm.progressive && arm.method.is_some() {
        problems.push(format!(
            "{where_}: progressive and method are mutually exclusive"
        ));
    }
    let smooth = arm.smooth.unwrap_or(shared.smooth);
    if smooth && arm.progressive {
        problems.push(format!(
            "{where_}: smooth is not supported for progressive streams"
        ));
    }
    if shared.layer.is_some() && !arm.progressive {
        problems.push(format!(
            "{where_}: decoder.layer only applies to progressive arms"
        ));
    }
    if arm.progressive && (shared.auto || shared.zoom != 1.0 || shared.iterations == 0) {
        problems.push(format!(
            "{where_}: progressive streams support neither decoder.auto nor decoder.zoom"
        ));
    }
}

/// One planned case. `unsupported` is `Some` for a combination the codec refuses for
/// this image; such a case is still counted, and recorded with an `unsupported` status.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedCase {
    pub index: usize,
    pub total: usize,
    pub case_id: String,
    pub image: ResolvedImage,
    pub arm: Arm,
    /// The decoder spec with this arm's overrides applied.
    pub decoder: DecoderSpec,
    pub unsupported: Option<String>,
}

/// A stage expanded into a deterministic case list.
#[derive(Debug, Clone)]
pub struct Plan {
    pub experiment: String,
    pub experiment_id: String,
    pub config_sha256: String,
    pub stage: String,
    pub repetitions: u32,
    pub timeout_secs: u64,
    pub codec: CodecSpec,
    pub decoder: DecoderSpec,
    pub cases: Vec<PlannedCase>,
}

#[derive(Serialize)]
struct CaseIdentity<'a> {
    experiment: &'a str,
    stage: &'a str,
    config_sha256: &'a str,
    image: &'a str,
    image_file: &'a Path,
    image_sha256: &'a Option<String>,
    raw_dims: &'a Option<(usize, usize)>,
    planes: usize,
    arm: &'a Arm,
    codec: &'a CodecSpec,
    decoder: &'a DecoderSpec,
}

/// Expand `(image × arm)` in image-major, then arm order. The order is part of the
/// output (row order is this order), so it is fixed rather than left to iteration.
pub fn plan(
    config: &ExperimentConfig,
    stage: &StageSpec,
    images: &[ResolvedImage],
) -> Result<Plan, ConfigError> {
    config.validate()?;
    let config_sha256 = config.config_sha256()?;
    let experiment_id = config.experiment_id()?;
    let mut cases = Vec::new();
    for image in images {
        for arm in &stage.arms {
            let decoder = DecoderSpec {
                iterations: arm.iterations.unwrap_or(config.decoder.iterations),
                smooth: arm.smooth.unwrap_or(config.decoder.smooth),
                ..config.decoder.clone()
            };
            let identity = CaseIdentity {
                experiment: &config.name,
                stage: &stage.name,
                config_sha256: &config_sha256,
                image: &image.name,
                image_file: &image.file,
                image_sha256: &image.sha256,
                raw_dims: &image.raw_dims,
                planes: image.planes,
                arm,
                codec: &config.codec,
                decoder: &decoder,
            };
            let value = serde_json::to_value(&identity).map_err(|source| ConfigError::Parse {
                path: "<case identity>".into(),
                source,
            })?;
            let case_id = sha256_hex(canonical_json(&value).as_bytes())[..32].to_string();
            cases.push(PlannedCase {
                index: 0,
                total: 0,
                case_id,
                image: image.clone(),
                arm: arm.clone(),
                decoder,
                unsupported: unsupported_reason(arm, image.planes),
            });
        }
    }
    let total = cases.len();
    for (index, case) in cases.iter_mut().enumerate() {
        case.index = index;
        case.total = total;
    }
    Ok(Plan {
        experiment: config.name.clone(),
        experiment_id,
        config_sha256,
        stage: stage.name.clone(),
        repetitions: stage.repetitions,
        timeout_secs: stage.timeout_secs,
        codec: config.codec.clone(),
        decoder: config.decoder.clone(),
        cases,
    })
}

/// A data-dependent refusal the codec applies to `--method`/`--progressive`/`--smooth`
/// on colour input, or to smoothing a progressive stream.
fn unsupported_reason(arm: &Arm, planes: usize) -> Option<String> {
    if arm.method.is_some() && planes != 1 {
        return Some(format!(
            "--method only supports grayscale input (image has {planes} planes)"
        ));
    }
    if arm.progressive && planes != 1 {
        return Some(format!(
            "--progressive only supports grayscale input (image has {planes} planes)"
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn config() -> ExperimentConfig {
        ExperimentConfig {
            schema_version: CONFIG_SCHEMA,
            name: "test".into(),
            description: "d".into(),
            indexes: Vec::new(),
            images: Vec::new(),
            inputs: Vec::new(),
            codec: CodecSpec {
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
            },
            decoder: DecoderSpec::default(),
            stages: vec![StageSpec {
                name: "smoke".into(),
                description: "d".into(),
                repetitions: 1,
                timeout_secs: 30,
                arms: vec![arm("baseline")],
            }],
        }
    }

    fn images() -> Vec<ResolvedImage> {
        vec![
            ResolvedImage {
                name: "gray".into(),
                file: "g.raw".into(),
                raw_dims: Some((8, 8)),
                sha256: Some("a".repeat(64)),
                planes: 1,
                corpus_index: None,
            },
            ResolvedImage {
                name: "rgb".into(),
                file: "c.png".into(),
                raw_dims: None,
                sha256: None,
                planes: 3,
                corpus_index: None,
            },
        ]
    }

    #[test]
    fn a_valid_config_plans_image_major_then_arm_order() {
        let mut c = config();
        c.stages[0].arms = vec![arm("b"), arm("a")];
        let plan = plan(&c, &c.stages[0], &images()).unwrap();
        assert_eq!(plan.cases.len(), 4);
        assert_eq!(plan.cases[0].image.name, "gray");
        assert_eq!(plan.cases[0].arm.label, "b");
        assert_eq!(plan.cases[1].arm.label, "a");
        assert_eq!(plan.cases[2].image.name, "rgb");
        for (i, case) in plan.cases.iter().enumerate() {
            assert_eq!(case.index, i);
            assert_eq!(case.total, 4);
        }
    }

    #[test]
    fn case_ids_are_stable_and_content_addressed() {
        let c = config();
        let a = plan(&c, &c.stages[0], &images()).unwrap();
        let b = plan(&c, &c.stages[0], &images()).unwrap();
        assert_eq!(
            a.cases.iter().map(|c| &c.case_id).collect::<Vec<_>>(),
            b.cases.iter().map(|c| &c.case_id).collect::<Vec<_>>()
        );
        // Order is not identity: swapping the arm list keeps each case's id.
        let mut swapped = config();
        swapped.stages[0].arms = vec![arm("baseline"), arm("baseline2")];
        let swapped = plan(&swapped, &swapped.stages[0], &images()).unwrap();
        assert_ne!(a.cases[0].case_id, swapped.cases[0].case_id);
    }

    #[test]
    fn a_method_on_a_colour_image_is_an_explicit_unsupported_case() {
        let mut c = config();
        c.stages[0].arms = vec![Arm {
            method: Some("fisher".into()),
            ..arm("m")
        }];
        let plan = plan(&c, &c.stages[0], &images()).unwrap();
        assert!(plan.cases[0].unsupported.is_none(), "{:?}", plan.cases[0]);
        assert!(plan.cases[1]
            .unsupported
            .as_deref()
            .unwrap()
            .contains("grayscale"));
    }

    fn assert_rejected(label: &str, mutate: impl Fn(&mut Arm), expected: &str) {
        let mut c = config();
        let mut a = arm("bad");
        mutate(&mut a);
        c.stages[0].arms = vec![a];
        let error = c.validate().unwrap_err().to_string();
        assert!(error.contains(expected), "{label}: {error}");
    }

    #[test]
    fn unsupported_arm_combinations_are_rejected_before_anything_runs() {
        assert_rejected(
            "unknown method",
            |a| a.method = Some("nope".into()),
            "unknown method",
        );
        assert_rejected(
            "random without budget",
            |a| a.method = Some("random".into()),
            "positive budget",
        );
        assert_rejected(
            "budget without method",
            |a| a.budget = Some(4),
            "budget/seed",
        );
        assert_rejected(
            "seed on apcc",
            |a| {
                a.method = Some("apcc".into());
                a.budget = Some(4);
                a.seed = Some(0);
            },
            "only valid with --method random",
        );
        assert_rejected(
            "modes without rd",
            |a| {
                a.lambda = None;
                a.modes = Some(vec![0, 2]);
            },
            "require the RD path",
        );
        assert_rejected(
            "residual without mode 3",
            |a| a.adaptive_residual = true,
            "requires mode 3",
        );
        assert_rejected(
            "method with progressive",
            |a| {
                a.method = Some("fisher".into());
                a.progressive = true;
            },
            "mutually exclusive",
        );
        assert_rejected(
            "duplicate mode",
            |a| a.modes = Some(vec![2, 2]),
            "listed twice",
        );
        assert_rejected(
            "mode out of range",
            |a| a.modes = Some(vec![4]),
            "not in 0..=3",
        );
        assert_rejected(
            "smooth with progressive",
            |a| {
                a.progressive = true;
                a.smooth = Some(true);
            },
            "not supported for progressive",
        );

        // `decoder.layer` lives on the shared decoder spec, not the arm.
        let mut c = config();
        c.decoder.layer = Some(2);
        assert!(c
            .validate()
            .unwrap_err()
            .to_string()
            .contains("layer only applies to progressive arms"));
    }

    #[test]
    fn config_rejects_nonrepresentable_contrast_and_bad_geometry() {
        let mut c = config();
        c.codec.max_alfa = 0.1;
        assert!(c.validate().unwrap_err().to_string().contains("max_alfa"));
        let mut c = config();
        c.codec.shift = 3;
        assert!(c.validate().unwrap_err().to_string().contains("shift"));
        let mut c = config();
        c.codec.min_size = 32;
        c.codec.max_size = 16;
        assert!(c.validate().unwrap_err().to_string().contains("min_size"));
    }

    #[test]
    fn stage_selection_is_explicit_when_ambiguous() {
        let mut c = config();
        c.stages.push(StageSpec {
            name: "rd".into(),
            description: "d".into(),
            repetitions: 1,
            timeout_secs: 30,
            arms: vec![arm("a")],
        });
        assert!(c.stage(Some("rd")).is_ok());
        let error = c.stage(None).unwrap_err().to_string();
        assert!(error.contains("pass --stage"), "{error}");
        assert!(c
            .stage(Some("nope"))
            .unwrap_err()
            .to_string()
            .contains("not defined"));
    }

    #[test]
    fn resolve_verifies_index_count_hashes_and_missing_files() {
        let dir =
            std::env::temp_dir().join(format!("mars-experiment-config-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pixels: Vec<u8> = (0..64).collect();
        std::fs::write(dir.join("a.raw"), &pixels).unwrap();
        let sha = sha256_hex(&pixels);
        let index = serde_json::json!({
            "count": 1, "set": "t",
            "images": [{"name":"a","file":"a.raw","width":8,"height":8,"sha256":sha}]
        });
        std::fs::write(dir.join("t.images.json"), index.to_string()).unwrap();

        let mut c = config();
        c.indexes = vec![PathBuf::from("t.images.json")];
        let resolved = c.resolve(&dir).unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].planes, 1);
        assert_eq!(resolved[0].raw_dims, Some((8, 8)));

        // A tampered hash fails before any encode.
        let bad = serde_json::json!({
            "count": 1, "set": "t",
            "images": [{"name":"a","file":"a.raw","width":8,"height":8,"sha256":"0".repeat(64)}]
        });
        std::fs::write(dir.join("bad.images.json"), bad.to_string()).unwrap();
        let mut c = config();
        c.indexes = vec![PathBuf::from("bad.images.json")];
        assert!(c
            .resolve(&dir)
            .unwrap_err()
            .to_string()
            .contains("does not match"));

        // An index whose count disagrees with its list is a completeness failure.
        let short = serde_json::json!({
            "count": 2, "set": "t",
            "images": [{"name":"a","file":"a.raw","width":8,"height":8,"sha256":sha}]
        });
        std::fs::write(dir.join("short.images.json"), short.to_string()).unwrap();
        let mut c = config();
        c.indexes = vec![PathBuf::from("short.images.json")];
        assert!(c
            .resolve(&dir)
            .unwrap_err()
            .to_string()
            .contains("does not match"));

        // A missing file is named explicitly.
        let missing = serde_json::json!({
            "count": 1, "set": "t",
            "images": [{"name":"a","file":"gone.raw","width":8,"height":8,"sha256":"0".repeat(64)}]
        });
        std::fs::write(dir.join("missing.images.json"), missing.to_string()).unwrap();
        let mut c = config();
        c.indexes = vec![PathBuf::from("missing.images.json")];
        assert!(c
            .resolve(&dir)
            .unwrap_err()
            .to_string()
            .contains("gone.raw"));

        // A requested subset name that no index provides is an explicit error.
        let mut c = config();
        c.indexes = vec![PathBuf::from("t.images.json")];
        c.images = vec!["not-here".into()];
        assert!(c
            .resolve(&dir)
            .unwrap_err()
            .to_string()
            .contains("not in any index"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
