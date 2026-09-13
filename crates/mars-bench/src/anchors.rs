//! The anchor-codec sweep (Step 4).
//!
//! Five CLI-driven codecs — JPEG, JPEG 2000, WebP, AVIF, JPEG XL — swept over their own
//! quality knob and measured by the same `mars-bench` machinery as everything else (§M1).
//! No codec here ever reports its own PSNR; each job encodes with the anchor's CLI,
//! decodes back to a PNG, and hands both files to [`crate::measure::measure`].
//!
//! **Quality is swept, not bpp.** These tools take a quality/distance/ratio knob, not a
//! target bpp, so the config lists quality points per codec (`configs/anchors.json`) and
//! the resulting bpp is recorded and later checked to fall inside 0.1–2.0 bpp for at least
//! six points per (codec, image) — see `anchors_report::gate`. Forcing an exact bpp target
//! by bisection was considered and rejected: it would silently paper over a codec that
//! structurally cannot reach part of the range, which is exactly the kind of thing §A2
//! says must be a recorded decision, not a tuned-away search (`docs/decisions.md` D17).
//!
//! **Colour, not grayscale.** Unlike the Step 2 sweep (which converts Kodak to grayscale
//! raw for the 1998 codec), anchors run on the corpus's original colour PNGs directly —
//! `mars_core::io::read_image` already reads them, and §M2's PSNR-Y/PSNR-YUV definitions
//! are colour-aware. See `docs/decisions.md` D17.

use std::collections::BTreeSet;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::measure::{measure, MeasureRequest};
use crate::provenance::{sha256_hex, Provenance};
use crate::store::{ResultStore, Row};

pub const ROW_KIND: &str = "anchors";

pub const TIMING_PROTOCOL: &str =
    "indicative-single-run; NOT a §M4 timing result (no interleaving, no median, concurrent jobs)";

// -------------------------------------------------------------------- the anchor codecs

/// The five anchor codecs named in the plan (§ Step 4 / §M10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorCodec {
    Jpeg,
    Jpeg2000,
    Webp,
    Avif,
    Jxl,
}

impl AnchorCodec {
    pub const ALL: [AnchorCodec; 5] = [
        AnchorCodec::Jpeg,
        AnchorCodec::Jpeg2000,
        AnchorCodec::Webp,
        AnchorCodec::Avif,
        AnchorCodec::Jxl,
    ];

    pub fn key(self) -> &'static str {
        match self {
            AnchorCodec::Jpeg => "jpeg",
            AnchorCodec::Jpeg2000 => "jpeg2000",
            AnchorCodec::Webp => "webp",
            AnchorCodec::Avif => "avif",
            AnchorCodec::Jxl => "jpegxl",
        }
    }

    /// What the swept parameter means, recorded on every row so a reader never has to
    /// guess whether a bigger number means better or worse quality.
    pub fn param_kind(self) -> &'static str {
        match self {
            AnchorCodec::Jpeg => "quality (0-100, higher is better)",
            AnchorCodec::Jpeg2000 => "compression ratio (lower is better quality)",
            AnchorCodec::Webp => "quality (0-100, higher is better)",
            AnchorCodec::Avif => "quality (0-100, higher is better)",
            AnchorCodec::Jxl => "butteraugli distance (lower is better quality)",
        }
    }

    fn encoded_ext(self) -> &'static str {
        match self {
            AnchorCodec::Jpeg => "jpg",
            AnchorCodec::Jpeg2000 => "j2k",
            AnchorCodec::Webp => "webp",
            AnchorCodec::Avif => "avif",
            AnchorCodec::Jxl => "jxl",
        }
    }

    /// One CLI invocation each way, so a version string is honest about what actually ran
    /// (§M10: "pinned versions with exact CLI invocations recorded in the manifest").
    fn version_probe(self) -> (&'static str, &'static [&'static str]) {
        match self {
            AnchorCodec::Jpeg => ("cjpeg", &["-version"]),
            AnchorCodec::Jpeg2000 => ("opj_compress", &["-h"]),
            AnchorCodec::Webp => ("cwebp", &["-version"]),
            AnchorCodec::Avif => ("avifenc", &["--version"]),
            AnchorCodec::Jxl => ("cjxl", &["--version"]),
        }
    }

    /// Encode `input` (a colour PNG) at `param`, writing into `workdir`. Returns the
    /// encoded file's path, byte size, indicative seconds, and the command line run.
    fn encode(
        self,
        workdir: &Path,
        input: &Path,
        param: f64,
    ) -> Result<(PathBuf, u64, f64, String), AnchorError> {
        let out = workdir.join(format!("enc.{}", self.encoded_ext()));
        let _ = std::fs::remove_file(&out);
        let (bin, args): (&str, Vec<String>) = match self {
            AnchorCodec::Jpeg => (
                "cjpeg",
                vec![
                    "-quality".into(),
                    format!("{}", param.round() as i64),
                    "-outfile".into(),
                    out.display().to_string(),
                    input.display().to_string(),
                ],
            ),
            AnchorCodec::Jpeg2000 => {
                // OpenJPEG's PNG reader honours the input's `gAMA` chunk on read but its
                // writer does not restore it, so a plain PNG round trip through
                // opj_compress/opj_decompress silently darkens every pixel -- even
                // losslessly. Kodak's PNGs carry a gAMA chunk, so we route around
                // OpenJPEG's PNG path entirely: convert to a plain PPM (no
                // colour-management metadata for anything to misapply) and feed that
                // instead. See `docs/decisions.md` D18.
                let ppm_in = workdir.join("in.ppm");
                let image = mars_core::io::read_image(input, None)?;
                mars_core::io::write_pnm(&ppm_in, &image)?;
                (
                    "opj_compress",
                    vec![
                        "-i".into(),
                        ppm_in.display().to_string(),
                        "-o".into(),
                        out.display().to_string(),
                        "-r".into(),
                        fmt_ratio(param),
                    ],
                )
            }
            AnchorCodec::Webp => (
                "cwebp",
                vec![
                    "-quiet".into(),
                    "-q".into(),
                    format!("{}", param.round() as i64),
                    input.display().to_string(),
                    "-o".into(),
                    out.display().to_string(),
                ],
            ),
            AnchorCodec::Avif => (
                "avifenc",
                vec![
                    "-q".into(),
                    format!("{}", param.round() as i64),
                    input.display().to_string(),
                    out.display().to_string(),
                ],
            ),
            AnchorCodec::Jxl => (
                "cjxl",
                vec![
                    "-d".into(),
                    format!("{param}"),
                    "--quiet".into(),
                    input.display().to_string(),
                    out.display().to_string(),
                ],
            ),
        };
        let cmd_str = format!("{bin} {}", args.join(" "));
        let (_out, seconds) = run_cli(bin, workdir, &args)?;
        let bytes = std::fs::metadata(&out)
            .map_err(|source| AnchorError::Io {
                path: out.display().to_string(),
                source,
            })?
            .len();
        Ok((out, bytes, seconds, cmd_str))
    }

    /// Decode `coded` back to a colour PNG in `workdir`. Returns the PNG path, indicative
    /// seconds, and the command line run.
    fn decode(self, workdir: &Path, coded: &Path) -> Result<(PathBuf, f64, String), AnchorError> {
        let out = workdir.join("dec.png");
        let _ = std::fs::remove_file(&out);
        let (bin, args): (&str, Vec<String>) = match self {
            AnchorCodec::Jpeg => (
                "djpeg",
                vec![
                    "-png".into(),
                    "-outfile".into(),
                    out.display().to_string(),
                    coded.display().to_string(),
                ],
            ),
            AnchorCodec::Jpeg2000 => (
                "opj_decompress",
                vec![
                    "-i".into(),
                    coded.display().to_string(),
                    "-o".into(),
                    out.display().to_string(),
                ],
            ),
            AnchorCodec::Webp => (
                "dwebp",
                vec![
                    "-quiet".into(),
                    coded.display().to_string(),
                    "-o".into(),
                    out.display().to_string(),
                ],
            ),
            AnchorCodec::Avif => (
                "avifdec",
                vec![coded.display().to_string(), out.display().to_string()],
            ),
            AnchorCodec::Jxl => (
                "djxl",
                vec![
                    coded.display().to_string(),
                    out.display().to_string(),
                    "--quiet".into(),
                ],
            ),
        };
        let cmd_str = format!("{bin} {}", args.join(" "));
        let (_out, seconds) = run_cli(bin, workdir, &args)?;
        if !out.is_file() {
            return Err(AnchorError::NoOutput {
                bin: bin.into(),
                expected: out.display().to_string(),
            });
        }
        Ok((out, seconds, cmd_str))
    }
}

impl std::str::FromStr for AnchorCodec {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        AnchorCodec::ALL
            .into_iter()
            .find(|c| c.key() == s)
            .ok_or_else(|| format!("unknown anchor codec {s:?}"))
    }
}

/// JPEG 2000's `-r` wants a plain decimal, not Rust's default float formatting (which
/// would print `1` as `1` but `1.3` as `1.3` anyway — this just avoids surprises like
/// `200.0` where `opj_compress` is happier with `200`).
fn fmt_ratio(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn run_cli(bin: &str, workdir: &Path, args: &[String]) -> Result<(Output, f64), AnchorError> {
    let started = Instant::now();
    let out = Command::new(bin)
        .args(args)
        .current_dir(workdir)
        .output()
        .map_err(|source| AnchorError::Spawn {
            bin: bin.into(),
            source,
        })?;
    let seconds = started.elapsed().as_secs_f64();
    if !out.status.success() {
        return Err(AnchorError::ExitStatus {
            bin: bin.into(),
            status: out.status.to_string(),
            stdout: String::from_utf8_lossy(&out.stdout).trim().to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        });
    }
    Ok((out, seconds))
}

/// Run the codec's version probe once and return its combined, trimmed output — the
/// pinned-version part of §M10.
pub fn codec_build_info(codec: AnchorCodec) -> String {
    let (bin, args) = codec.version_probe();
    match Command::new(bin).args(args).output() {
        Ok(out) => {
            let mut s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
            if !err.is_empty() {
                if !s.is_empty() {
                    s.push('\n');
                }
                s.push_str(&err);
            }
            // Version probes commonly emit multi-line help. Most of the five put the
            // version string on the first line; `opj_compress -h` (OpenJPEG has no
            // dedicated version flag) buries it a few lines in as prose, so prefer
            // whichever line actually contains a digit, falling back to the first line.
            s.lines()
                .find(|l| l.chars().any(|c| c.is_ascii_digit()))
                .or_else(|| s.lines().next())
                .unwrap_or("")
                .trim()
                .to_string()
        }
        Err(e) => format!("<{bin} -- probe failed: {e}>"),
    }
}

// ------------------------------------------------------------------------- the config

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnchorsConfig {
    pub name: String,
    pub description: String,
    /// The Kodak fetch manifest (`corpus/kodak.manifest.json`), reused directly rather
    /// than duplicated into a second image-set schema: it already has name + sha256, and
    /// anchors need nothing else (§2.3 — one source of truth for the corpus).
    pub corpus_manifest: PathBuf,
    pub codecs: Vec<CodecSweep>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CodecSweep {
    pub codec: AnchorCodec,
    /// The quality/ratio/distance points to sweep, in the units `codec.param_kind()`
    /// documents. Not necessarily monotonic in bpp order in the file; `plan` does not
    /// require it, and `anchors_report` sorts by measured bpp.
    pub params: Vec<f64>,
}

impl AnchorsConfig {
    pub fn read(path: &Path) -> Result<Self, AnchorError> {
        let text = std::fs::read_to_string(path).map_err(|source| AnchorError::Io {
            path: path.display().to_string(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| AnchorError::Config {
            path: path.display().to_string(),
            source,
        })
    }

    /// A config that cannot conceivably produce a valid curve is refused before spending
    /// any wall-clock time on it — the same instinct as `SweepConfig::validate`.
    pub fn validate(&self) -> Result<(), AnchorError> {
        if self.codecs.is_empty() {
            return Err(AnchorError::EmptyAxis);
        }
        let mut seen = BTreeSet::new();
        for c in &self.codecs {
            if !seen.insert(c.codec) {
                return Err(AnchorError::DuplicateCodec {
                    codec: c.codec.key(),
                });
            }
            if c.params.len() < crate::bdrate::MIN_POINTS_PER_CURVE {
                return Err(AnchorError::TooFewParams {
                    codec: c.codec.key(),
                    got: c.params.len(),
                    need: crate::bdrate::MIN_POINTS_PER_CURVE,
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------- the corpus set

#[derive(Debug, Clone, Deserialize)]
struct KodakManifest {
    set: String,
    dir: String,
    images: Vec<KodakManifestImage>,
}

#[derive(Debug, Clone, Deserialize)]
struct KodakManifestImage {
    name: String,
    sha256: String,
}

/// One Kodak colour PNG, resolved and hash-checked against the fetch manifest.
#[derive(Debug, Clone)]
pub struct AnchorImage {
    pub name: String,
    pub path: PathBuf,
    pub sha256: String,
}

/// Load the corpus named in `manifest_path`, verifying every file is present and its
/// hash matches — the same guarantee `just corpus` gives at fetch time, re-checked here
/// so a row's `image_sha256` is never an assertion nobody verified.
pub fn load_corpus(
    repo_root: &Path,
    manifest_path: &Path,
) -> Result<(String, Vec<AnchorImage>), AnchorError> {
    let text = std::fs::read_to_string(repo_root.join(manifest_path)).map_err(|source| {
        AnchorError::Io {
            path: manifest_path.display().to_string(),
            source,
        }
    })?;
    let manifest: KodakManifest =
        serde_json::from_str(&text).map_err(|source| AnchorError::Config {
            path: manifest_path.display().to_string(),
            source,
        })?;
    let dir = repo_root.join(&manifest.dir);
    let mut images = Vec::with_capacity(manifest.images.len());
    for entry in manifest.images {
        let path = dir.join(&entry.name);
        let mut f = std::fs::File::open(&path).map_err(|source| AnchorError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)
            .map_err(|source| AnchorError::Io {
                path: path.display().to_string(),
                source,
            })?;
        let got = sha256_hex(&bytes);
        if got != entry.sha256 {
            return Err(AnchorError::CorpusHashMismatch {
                image: entry.name,
                want: entry.sha256,
                got,
            });
        }
        images.push(AnchorImage {
            name: entry.name,
            path,
            sha256: entry.sha256,
        });
    }
    Ok((manifest.set, images))
}

// ------------------------------------------------------------------------- planning

#[derive(Debug, Clone)]
pub struct Job {
    pub id: usize,
    pub codec: AnchorCodec,
    pub image: AnchorImage,
    pub param: f64,
}

/// Expand the config into jobs in a fixed order — codec, then image, then param — so row
/// order does not depend on thread scheduling (§2.3).
pub fn plan(config: &AnchorsConfig, images: &[AnchorImage]) -> Result<Vec<Job>, AnchorError> {
    config.validate()?;
    let mut jobs = Vec::new();
    for sweep in &config.codecs {
        for image in images {
            for &param in &sweep.params {
                jobs.push(Job {
                    id: jobs.len(),
                    codec: sweep.codec,
                    image: image.clone(),
                    param,
                });
            }
        }
    }
    Ok(jobs)
}

// -------------------------------------------------------------------- the row payload

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorRow {
    pub sweep: String,
    pub corpus: String,
    pub image: String,
    pub image_sha256: String,
    pub codec: String,
    pub param: f64,
    pub param_kind: String,
    pub coded_bytes: u64,
    pub quality: crate::measure::Measurement,
    pub indicative_encode_seconds: f64,
    pub indicative_decode_seconds: f64,
    pub timing_protocol: String,
    pub encode_cmd: String,
    pub decode_cmd: String,
    pub codec_build_info: String,
    pub jobs: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum AnchorError {
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
    #[error("the anchors config has no codecs")]
    EmptyAxis,
    #[error("codec {codec} appears twice in the config")]
    DuplicateCodec { codec: &'static str },
    #[error("codec {codec} declares {got} params; needs at least {need}")]
    TooFewParams {
        codec: &'static str,
        got: usize,
        need: usize,
    },
    #[error("{image}: sha256 mismatch (manifest says {want}, disk has {got}) -- corpus changed under us")]
    CorpusHashMismatch {
        image: String,
        want: String,
        got: String,
    },
    #[error("spawning {bin}: {source}")]
    Spawn {
        bin: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{bin} exited {status}\nstdout: {stdout}\nstderr: {stderr}")]
    ExitStatus {
        bin: String,
        status: String,
        stdout: String,
        stderr: String,
    },
    #[error("{bin} exited 0 but did not produce {expected}")]
    NoOutput { bin: String, expected: String },
    #[error(transparent)]
    Image(#[from] mars_core::io::ImageError),
    #[error(transparent)]
    Measure(#[from] crate::measure::MeasureError),
    #[error(transparent)]
    Store(#[from] crate::store::StoreError),
    #[error("serialising a row: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("job {id} ({image} {codec} param={param}) failed: {source}")]
    Job {
        id: usize,
        image: String,
        codec: &'static str,
        param: f64,
        #[source]
        source: Box<AnchorError>,
    },
    #[error("job {id} produced no row and reported no error; the sweep is incomplete")]
    MissingResult { id: usize },
}

// ------------------------------------------------------------------------ execution

pub struct RunContext<'a> {
    pub repo_root: &'a Path,
    pub scratch: &'a Path,
    pub jobs: usize,
    pub sweep_name: &'a str,
}

fn run_job(ctx: &RunContext, workdir: &Path, job: &Job) -> Result<AnchorRow, AnchorError> {
    let src = &job.image.path;
    let (coded_path, coded_bytes, enc_seconds, encode_cmd) =
        job.codec.encode(workdir, src, job.param)?;
    let (decoded_path, dec_seconds, decode_cmd) = job.codec.decode(workdir, &coded_path)?;

    // §M1: quality computed by the harness from two files on disk, never self-reported.
    let req = MeasureRequest::new(src, &decoded_path).with_coded_bytes(coded_bytes);
    let quality = measure(&req)?;

    Ok(AnchorRow {
        sweep: ctx.sweep_name.to_string(),
        corpus: "standard".to_string(),
        image: job.image.name.clone(),
        image_sha256: job.image.sha256.clone(),
        codec: job.codec.key().to_string(),
        param: job.param,
        param_kind: job.codec.param_kind().to_string(),
        coded_bytes,
        quality,
        indicative_encode_seconds: enc_seconds,
        indicative_decode_seconds: dec_seconds,
        timing_protocol: TIMING_PROTOCOL.to_string(),
        encode_cmd,
        decode_cmd,
        codec_build_info: codec_build_info(job.codec),
        jobs: ctx.jobs,
    })
}

/// Run every job across `ctx.jobs` worker threads, appending rows **in job order** — the
/// same buffer-then-append discipline as `sweep::run`, for the same reason: a result file
/// that does not depend on thread scheduling.
pub fn run(
    ctx: &RunContext,
    jobs: &[Job],
    provenance: &Provenance,
    store: &mut ResultStore,
    progress: &(dyn Fn(usize, usize, &Job) + Sync),
) -> Result<usize, AnchorError> {
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<AnchorRow>>> =
        Mutex::new((0..jobs.len()).map(|_| None).collect());
    let first_error: Mutex<Option<AnchorError>> = Mutex::new(None);

    std::thread::scope(|scope| {
        for worker in 0..ctx.jobs.max(1) {
            let next = &next;
            let done = &done;
            let results = &results;
            let first_error = &first_error;
            scope.spawn(move || {
                let workdir = ctx.scratch.join(format!("w{worker}"));
                if let Err(source) = std::fs::create_dir_all(&workdir) {
                    let mut slot = first_error.lock().expect("poisoned");
                    if slot.is_none() {
                        *slot = Some(AnchorError::Io {
                            path: workdir.display().to_string(),
                            source,
                        });
                    }
                    return;
                }
                loop {
                    if first_error.lock().expect("poisoned").is_some() {
                        return;
                    }
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(job) = jobs.get(i) else { return };
                    match run_job(ctx, &workdir, job) {
                        Ok(row) => {
                            results.lock().expect("poisoned")[i] = Some(row);
                            let n = done.fetch_add(1, Ordering::SeqCst) + 1;
                            progress(n, jobs.len(), job);
                        }
                        Err(source) => {
                            let mut slot = first_error.lock().expect("poisoned");
                            if slot.is_none() {
                                *slot = Some(AnchorError::Job {
                                    id: job.id,
                                    image: job.image.name.clone(),
                                    codec: job.codec.key(),
                                    param: job.param,
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
        let payload = slot.ok_or(AnchorError::MissingResult { id: i })?;
        store.append(&Row::new(ROW_KIND, provenance.clone(), &payload)?)?;
        written += 1;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AnchorsConfig {
        AnchorsConfig {
            name: "t".into(),
            description: "d".into(),
            corpus_manifest: PathBuf::from("corpus/kodak.manifest.json"),
            codecs: vec![
                CodecSweep {
                    codec: AnchorCodec::Jpeg,
                    params: vec![5.0, 10.0, 30.0, 60.0, 80.0, 90.0],
                },
                CodecSweep {
                    codec: AnchorCodec::Avif,
                    params: vec![5.0, 10.0, 30.0, 60.0, 80.0, 90.0],
                },
            ],
        }
    }

    fn images(names: &[&str]) -> Vec<AnchorImage> {
        names
            .iter()
            .map(|n| AnchorImage {
                name: (*n).into(),
                path: PathBuf::from(format!("{n}.png")),
                sha256: "0".repeat(64),
            })
            .collect()
    }

    #[test]
    fn the_plan_is_the_full_grid_in_a_fixed_order() {
        let c = config();
        let imgs = images(&["a", "b"]);
        let jobs = plan(&c, &imgs).unwrap();
        // 2 codecs x 2 images x 6 params.
        assert_eq!(jobs.len(), 24);
        assert_eq!(jobs[0].codec, AnchorCodec::Jpeg);
        assert_eq!(jobs[0].image.name, "a");
        assert_eq!(jobs[0].param, 5.0);
        assert_eq!(jobs[5].param, 90.0);
        assert_eq!(jobs[6].image.name, "b");
        assert_eq!(jobs[12].codec, AnchorCodec::Avif);
        assert!(jobs.iter().enumerate().all(|(i, j)| j.id == i));
    }

    #[test]
    fn a_config_with_too_few_params_is_refused() {
        let mut c = config();
        c.codecs[0].params = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            c.validate(),
            Err(AnchorError::TooFewParams {
                got: 3,
                need: 4,
                ..
            })
        ));
    }

    #[test]
    fn a_duplicate_codec_is_refused() {
        let mut c = config();
        let dup = c.codecs[0].clone();
        c.codecs.push(dup);
        assert!(matches!(
            c.validate(),
            Err(AnchorError::DuplicateCodec { .. })
        ));
    }

    #[test]
    fn codec_keys_round_trip_through_from_str() {
        for c in AnchorCodec::ALL {
            assert_eq!(c.key().parse::<AnchorCodec>().unwrap(), c);
        }
    }

    #[test]
    fn fmt_ratio_avoids_a_trailing_point_zero() {
        assert_eq!(fmt_ratio(200.0), "200");
        assert_eq!(fmt_ratio(1.3), "1.3");
    }
}
