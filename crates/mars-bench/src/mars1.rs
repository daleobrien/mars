//! The Mars 1 subprocess driver (Step 2).
//!
//! Wraps the unmodified 1998 `encmars` / `decmars` binaries and parses their stdout for
//! the four numbers Step 2 exists to capture: `transforms`, `comparisons`,
//! `zero_alfa_transform`, and bytes written. `comparisons` is Mars 1's `evals` analogue
//! (§M5) and `comparisons / transforms` is the first row of the project's headline metric.
//!
//! **Everything here is checked rather than trusted.** The driver re-derives every number
//! it can from a second source and fails on disagreement:
//!
//! - the method the encoder *reports* must equal the method we asked for (this is not
//!   paranoia — see [`Method`]: passing no flag silently selects MassCenter),
//! - the byte count the encoder prints must equal the size of the file on disk,
//!   `Comparisons/Transformations` must equal `comparisons / transforms` re-derived and
//!   re-formatted, and the echoed thresholds must equal the ones we passed.
//!
//! These are exact equalities, not tolerances, and they cost one string comparison each.
//!
//! **Working directory.** `reference/mars1/globals.h` declares `char filein[50]` and
//! `getopt_enc` `strcpy`s `argv` into it, so a path longer than 49 bytes overflows the
//! buffer and produces a misleading "Can't open output file". Every process is therefore
//! spawned with `current_dir` set to a scratch directory and **relative** filenames
//! (`docs/decisions.md` D3). `Command::current_dir` is used rather than
//! `std::env::set_current_dir` so that the driver stays safe to run concurrently.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// The six speed-up methods of `reference/mars1/def.h`.
///
/// **There is no exhaustive mode.** `globals.h:201` is `EXTERN int method INIT(=
/// MassCenter)`, so invoking `encmars` with no method flag runs MassCenter while printing
/// nothing that looks like a default. Every invocation this driver makes passes an
/// explicit flag, and [`EncodeStats::method_reported`] is checked against it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Method {
    Fisher,
    Hurtgen,
    MassCenter,
    Saupe,
    SaupeFisher,
    McSaupe,
}

impl Method {
    /// In `def.h` order, which is also the order the six `comparisons` increment sites
    /// appear in `coding_func.c`.
    pub const ALL: [Method; 6] = [
        Method::MassCenter,
        Method::SaupeFisher,
        Method::Saupe,
        Method::Fisher,
        Method::Hurtgen,
        Method::McSaupe,
    ];

    /// The `encmars` command-line flag.
    pub fn flag(self) -> &'static str {
        match self {
            Method::Fisher => "-F",
            Method::Hurtgen => "-X",
            Method::MassCenter => "-C",
            Method::Saupe => "-S",
            Method::SaupeFisher => "-Z",
            Method::McSaupe => "-Y",
        }
    }

    /// Stable identifier for config files, result rows and curve labels.
    pub fn key(self) -> &'static str {
        match self {
            Method::Fisher => "fisher",
            Method::Hurtgen => "hurtgen",
            Method::MassCenter => "masscenter",
            Method::Saupe => "saupe",
            Method::SaupeFisher => "saupe-fisher",
            Method::McSaupe => "mc-saupe",
        }
    }

    /// Exactly what `mars_enc.c` prints after "Speed-up method: ". Spelling and
    /// punctuation are the reference's, not ours, and the match is exact.
    pub fn reported_label(self) -> &'static str {
        match self {
            Method::Fisher => "Fisher",
            Method::Hurtgen => "Hurtgen",
            Method::MassCenter => "MassCenter",
            Method::Saupe => "Saupe",
            Method::SaupeFisher => "Saupe-Fisher",
            Method::McSaupe => "Mc-Saupe",
        }
    }
}

// Serialised through `key()` rather than by `#[derive]`, so the config spelling, the
// result-row spelling, and the curve label cannot drift apart into two names for one
// method — which is exactly what a derived `rename_all` produced on the first attempt.
impl Serialize for Method {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.key())
    }
}

impl<'de> Deserialize<'de> for Method {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl std::str::FromStr for Method {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Method::ALL
            .iter()
            .copied()
            .find(|m| m.key() == s)
            .ok_or_else(|| {
                let keys: Vec<&str> = Method::ALL.iter().map(|m| m.key()).collect();
                format!("unknown method {s:?}; expected one of {}", keys.join(", "))
            })
    }
}

/// The encoder parameters Step 2 sweeps. Serialised into every result row, and hashed
/// into the provenance block's `parameter_set_sha256` (§M7).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncodeParams {
    pub method: Method,
    /// `-r`, `T_RMS`.
    pub t_rms: f64,
    /// `-m`, minimum range size.
    pub min_size: u32,
    /// `-M`, maximum range size.
    pub max_size: u32,
    /// `-d`, the domain step `SHIFT`.
    pub shift: u32,
    /// `-A`, `N_BITALFA`.
    pub bits_alfa: u32,
    /// `-B`, `N_BITBETA`.
    pub bits_beta: u32,
    /// `-y`, `MAX_ALFA`.
    pub max_alfa: f64,
    /// `-e`, `T_ENT`. `None` leaves the 1998 default of 8.0 in place.
    pub t_ent: Option<f64>,
    /// `-v`, `T_VAR`. `None` leaves the 1998 default of 1e6 in place.
    pub t_var: Option<f64>,
}

impl Default for EncodeParams {
    /// The 1998 defaults, with the method made explicit because the reference's own
    /// default is a silent MassCenter.
    fn default() -> Self {
        Self {
            method: Method::Fisher,
            t_rms: 8.0,
            min_size: 4,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            t_ent: None,
            t_var: None,
        }
    }
}

impl EncodeParams {
    /// The argument vector, with `input` and `output` last as `encmars` expects.
    pub fn args(&self, width: u32, height: u32, input: &str, output: &str) -> Vec<String> {
        let mut a: Vec<String> = vec![
            self.method.flag().into(),
            "-W".into(),
            width.to_string(),
            "-H".into(),
            height.to_string(),
            "-r".into(),
            fmt_c_float(self.t_rms),
            "-m".into(),
            self.min_size.to_string(),
            "-M".into(),
            self.max_size.to_string(),
            "-d".into(),
            self.shift.to_string(),
            "-A".into(),
            self.bits_alfa.to_string(),
            "-B".into(),
            self.bits_beta.to_string(),
            "-y".into(),
            fmt_c_float(self.max_alfa),
        ];
        if let Some(e) = self.t_ent {
            a.push("-e".into());
            a.push(fmt_c_float(e));
        }
        if let Some(v) = self.t_var {
            a.push("-v".into());
            a.push(fmt_c_float(v));
        }
        a.push(input.into());
        a.push(output.into());
        a
    }
}

/// Render a float the way `atof` will read it back exactly: enough digits to round-trip,
/// and never in a form (`1e6`, `inf`) that would depend on the C library's parser.
fn fmt_c_float(v: f64) -> String {
    let s = format!("{v:.17}");
    // Trim the trailing zeros `{:.17}` adds, but keep at least one decimal digit so the
    // value can never be read as an integer by something downstream.
    let t = s.trim_end_matches('0');
    if t.ends_with('.') {
        format!("{t}0")
    } else {
        t.to_string()
    }
}

/// Which decoder the numbers came from.
///
/// **This is not a detail.** `decmars` defaults to pyramidal (`piramidal INIT(=1)`,
/// `iterations = 10`); comparing a pyramidal decode against an iterative one is a silent
/// 0.2–1 dB error, so the mode travels with every row that has a PSNR in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeMode {
    /// The 1998 default: decode a low-resolution image, then raise resolution.
    Pyramidal,
    /// `-i`: plain iterated-function-system fixed-point iteration.
    Iterative,
}

impl DecodeMode {
    pub fn key(self) -> &'static str {
        match self {
            DecodeMode::Pyramidal => "pyramidal",
            DecodeMode::Iterative => "iterative",
        }
    }
}

impl Serialize for DecodeMode {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.key())
    }
}

impl<'de> Deserialize<'de> for DecodeMode {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl std::str::FromStr for DecodeMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pyramidal" => Ok(DecodeMode::Pyramidal),
            "iterative" => Ok(DecodeMode::Iterative),
            other => Err(format!(
                "unknown decode mode {other:?}; expected pyramidal or iterative"
            )),
        }
    }
}

/// `decmars`'s compiled-in iteration count (`globals.h`: `iterations INIT(= 10)`).
///
/// This is the variable that actually moves decoded PSNR, and it moves it a long way:
/// measured on kodim01, pyramidal and iterative decode of the *same* bitstream differ by
/// 6.1 dB at 1 iteration, 0.26 dB at 5, and 0.004 dB at 10. Both modes converge to the
/// same IFS fixed point, so by the default count the choice of mode is worth almost
/// nothing — see `docs/decisions.md` D9.
pub const DEFAULT_ITERATIONS: u32 = 10;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeParams {
    pub mode: DecodeMode,
    /// `-n`. `None` leaves the 1998 default of 10.
    pub iterations: Option<u32>,
    /// `-p`, the postprocessing filter. Off for every reported number; the baseline is
    /// the codec, not the codec plus a deblocker.
    pub postprocess: bool,
}

impl Default for DecodeParams {
    fn default() -> Self {
        Self {
            mode: DecodeMode::Pyramidal,
            iterations: None,
            postprocess: false,
        }
    }
}

impl DecodeParams {
    /// The count actually in force, with the reference's default resolved. Recorded on
    /// every row rather than left implicit: a PSNR whose iteration count is unstated is
    /// not reproducible, and at low counts it is not even close.
    pub fn effective_iterations(&self) -> u32 {
        self.iterations.unwrap_or(DEFAULT_ITERATIONS)
    }

    pub fn args(&self, input: &str, output: &str) -> Vec<String> {
        let mut a: Vec<String> = Vec::new();
        if self.mode == DecodeMode::Iterative {
            a.push("-i".into());
        }
        if let Some(n) = self.iterations {
            a.push("-n".into());
            a.push(n.to_string());
        }
        if self.postprocess {
            a.push("-p".into());
        }
        a.push(input.into());
        a.push(output.into());
        a
    }
}

/// Everything `encmars` prints about a run, parsed. Field names follow the plan's
/// vocabulary (`transforms`, `comparisons`) rather than the C's printf labels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncodeStats {
    /// Verbatim, as printed after "Speed-up method: ".
    pub method_reported: String,
    pub image_entropy: f64,
    pub image_variance: f64,
    /// `T_ENT` as the encoder echoed it — the value actually in force, not the one we
    /// believe we passed.
    pub t_ent: f64,
    /// `T_VAR` as echoed.
    pub t_var: f64,
    /// `T_RMS` as echoed.
    pub t_rms: f64,
    pub zero_alfa_transforms: u64,
    /// Leaf blocks emitted (§M5).
    pub transforms: u64,
    /// Mars 1's `evals` analogue (§M5): full affine fit + RMS evaluations.
    pub comparisons: u64,
    /// As printed. Re-derived and checked; never used in preference to the ratio of the
    /// two integers above.
    pub comparisons_per_transform_reported: f64,
    pub bytes_written: u64,
    /// The output filename the encoder echoed, used to catch an argument-order slip.
    pub output_name: String,
}

impl EncodeStats {
    /// The headline metric (§M5), from the two integers rather than the printed ratio.
    pub fn evals_per_transform(&self) -> f64 {
        self.comparisons as f64 / self.transforms as f64
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeStats {
    pub mode: DecodeMode,
    pub width: u32,
    pub height: u32,
}

/// One encode, with what it produced.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncodeOutcome {
    pub params: EncodeParams,
    pub stats: EncodeStats,
    /// Size of the `.ifs` on disk. Checked equal to `stats.bytes_written`; this is the
    /// one used for bpp (§M2: whole file, header included).
    pub coded_bytes: u64,
    /// Wall-clock seconds for the subprocess.
    ///
    /// **Not a §M4 timing result.** One run, no interleaving, no median, possibly with
    /// sibling encodes on other cores. It is recorded to size future work, and
    /// [`crate::sweep`] stamps every row with `timing_protocol` saying exactly that.
    pub indicative_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodeOutcome {
    pub params: DecodeParams,
    pub stats: DecodeStats,
    pub indicative_seconds: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum Mars1Error {
    #[error("{what} not found at {path}; run `just mars1` to build the 1998 reference")]
    MissingBinary { what: &'static str, path: PathBuf },

    #[error("spawning {bin}: {source}")]
    Spawn {
        bin: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{bin} exited with {status}\n--- stdout tail ---\n{stdout}\n--- stderr ---\n{stderr}")]
    ExitStatus {
        bin: String,
        status: String,
        stdout: String,
        stderr: String,
    },

    #[error("{bin} printed no `{field}` line\n--- stdout tail ---\n{stdout}")]
    MissingField {
        bin: String,
        field: &'static str,
        stdout: String,
    },

    #[error("{bin}: cannot read {field} from {line:?}")]
    UnparsableField {
        bin: String,
        field: &'static str,
        line: String,
    },

    /// The check that catches the MassCenter trap, and any future flag-table drift.
    #[error("asked encmars for {requested} ({flag}) but it reported {reported:?}; the flag table is wrong")]
    MethodMismatch {
        requested: &'static str,
        flag: &'static str,
        reported: String,
    },

    #[error("encmars echoed {field} = {echoed} but was passed {passed}")]
    ThresholdMismatch {
        field: &'static str,
        echoed: String,
        passed: String,
    },

    #[error("encmars printed {printed} bytes written but {path} is {actual} bytes on disk")]
    ByteCountMismatch {
        printed: u64,
        actual: u64,
        path: String,
    },

    #[error("encmars printed Comparisons/Transformations = {printed} but {comparisons}/{transforms} formats as {derived}")]
    RatioMismatch {
        printed: String,
        derived: String,
        comparisons: u64,
        transforms: u64,
    },

    #[error("encmars emitted zero transformations; the encode produced nothing to measure")]
    NoTransforms,

    #[error("decmars reports the image is {got_w}x{got_h}, expected {want_w}x{want_h}")]
    DecodeSizeMismatch {
        got_w: u32,
        got_h: u32,
        want_w: u32,
        want_h: u32,
    },

    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Located `encmars` / `decmars`.
#[derive(Debug, Clone)]
pub struct Mars1Binaries {
    pub encmars: PathBuf,
    pub decmars: PathBuf,
    /// Contents of `build-info.txt`, so a result row can cite the exact compiler and
    /// flags that produced the baseline (§M7).
    pub build_info: Option<String>,
}

impl Mars1Binaries {
    /// Default location: `target/mars1`, where `scripts/build-mars1.sh` puts them.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, Mars1Error> {
        let dir = dir.as_ref();
        let encmars = dir.join("encmars");
        let decmars = dir.join("decmars");
        if !encmars.is_file() {
            return Err(Mars1Error::MissingBinary {
                what: "encmars",
                path: encmars,
            });
        }
        if !decmars.is_file() {
            return Err(Mars1Error::MissingBinary {
                what: "decmars",
                path: decmars,
            });
        }
        Ok(Self {
            encmars,
            decmars,
            build_info: std::fs::read_to_string(dir.join("build-info.txt")).ok(),
        })
    }
}

/// Run a child in `workdir` with relative filenames (D3) and return its output.
fn run(bin: &Path, workdir: &Path, args: &[String]) -> Result<(Output, f64), Mars1Error> {
    let started = Instant::now();
    let out = Command::new(bin)
        .args(args)
        .current_dir(workdir)
        .output()
        .map_err(|source| Mars1Error::Spawn {
            bin: bin.display().to_string(),
            source,
        })?;
    let seconds = started.elapsed().as_secs_f64();
    if !out.status.success() {
        return Err(Mars1Error::ExitStatus {
            bin: bin.display().to_string(),
            status: out.status.to_string(),
            stdout: tail(&String::from_utf8_lossy(&out.stdout)),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        });
    }
    Ok((out, seconds))
}

/// The reference's progress output is megabytes of carriage-return-terminated spam; only
/// the tail is ever worth showing a human.
fn tail(s: &str) -> String {
    let lines: Vec<&str> = logical_lines(s).collect();
    let start = lines.len().saturating_sub(14);
    lines[start..].join("\n")
}

/// Split on `\n` *and* `\r`: `mars_enc.c` writes its progress counter with a trailing
/// `\r`, so a summary line would otherwise arrive glued to the end of the progress spam.
fn logical_lines(s: &str) -> impl Iterator<Item = &str> {
    s.split(['\n', '\r'])
        .map(str::trim)
        .filter(|l| !l.is_empty())
}

/// Find the first line beginning with `prefix` and return the rest of it, trimmed.
fn after_prefix<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    logical_lines(s).find_map(|l| l.strip_prefix(prefix).map(str::trim))
}

/// Find the first line containing `": "` whose left side, trimmed, equals `label`.
fn labelled<'a>(s: &'a str, label: &str) -> Option<&'a str> {
    logical_lines(s).find_map(|l| {
        let (k, v) = l.split_once(':')?;
        (k.trim() == label).then(|| v.trim())
    })
}

fn field<'a>(
    bin: &'static str,
    stdout: &'a str,
    label: &str,
    field: &'static str,
) -> Result<&'a str, Mars1Error> {
    labelled(stdout, label).ok_or_else(|| Mars1Error::MissingField {
        bin: bin.into(),
        field,
        stdout: tail(stdout),
    })
}

fn parse_f64(bin: &'static str, raw: &str, name: &'static str) -> Result<f64, Mars1Error> {
    raw.parse().map_err(|_| Mars1Error::UnparsableField {
        bin: bin.into(),
        field: name,
        line: raw.to_string(),
    })
}

fn parse_u64(bin: &'static str, raw: &str, name: &'static str) -> Result<u64, Mars1Error> {
    raw.parse().map_err(|_| Mars1Error::UnparsableField {
        bin: bin.into(),
        field: name,
        line: raw.to_string(),
    })
}

/// Parse the encoder's summary block. Pure, so it is unit-tested against committed
/// stdout without needing the binaries.
///
/// Every field is required. A missing field is an error rather than a `None`, because the
/// one thing worse than a failed baseline run is a baseline row with a silently defaulted
/// `comparisons` in it.
pub fn parse_encode_stdout(stdout: &str) -> Result<EncodeStats, Mars1Error> {
    const BIN: &str = "encmars";

    let method_reported = after_prefix(stdout, "Speed-up method:")
        .ok_or_else(|| Mars1Error::MissingField {
            bin: BIN.into(),
            field: "Speed-up method",
            stdout: tail(stdout),
        })?
        .to_string();

    // " 15814 bytes written in l8.ifs"
    let written = logical_lines(stdout)
        .find(|l| l.contains("bytes written in"))
        .ok_or_else(|| Mars1Error::MissingField {
            bin: BIN.into(),
            field: "bytes written",
            stdout: tail(stdout),
        })?;
    let mut w = written.split_whitespace();
    let bytes_written = parse_u64(BIN, w.next().unwrap_or(""), "bytes written")?;
    let output_name = w
        .nth(3)
        .ok_or_else(|| Mars1Error::UnparsableField {
            bin: BIN.into(),
            field: "bytes written",
            line: written.to_string(),
        })?
        .to_string();

    let comparisons_raw = field(BIN, stdout, "Comparisons/Transformations", "Comparisons/Tr")?;

    Ok(EncodeStats {
        method_reported,
        image_entropy: parse_f64(
            BIN,
            field(BIN, stdout, "Image Entropy", "Image Entropy")?,
            "Image Entropy",
        )?,
        image_variance: parse_f64(
            BIN,
            field(BIN, stdout, "Image Variance", "Image Variance")?,
            "Image Variance",
        )?,
        t_ent: parse_f64(
            BIN,
            field(BIN, stdout, "Entropy threshold", "Entropy threshold")?,
            "Entropy threshold",
        )?,
        t_var: parse_f64(
            BIN,
            field(BIN, stdout, "Variance threshold", "Variance threshold")?,
            "Variance threshold",
        )?,
        t_rms: parse_f64(
            BIN,
            field(BIN, stdout, "Rms threshold", "Rms threshold")?,
            "Rms threshold",
        )?,
        zero_alfa_transforms: parse_u64(
            BIN,
            field(
                BIN,
                stdout,
                "Zero_alfa_transformations",
                "Zero_alfa_transformations",
            )?,
            "Zero_alfa_transformations",
        )?,
        transforms: parse_u64(
            BIN,
            field(
                BIN,
                stdout,
                "Number of transformations",
                "Number of transformations",
            )?,
            "Number of transformations",
        )?,
        comparisons: parse_u64(
            BIN,
            field(
                BIN,
                stdout,
                "Number of comparisons",
                "Number of comparisons",
            )?,
            "Number of comparisons",
        )?,
        comparisons_per_transform_reported: parse_f64(BIN, comparisons_raw, "Comparisons/Tr")?,
        bytes_written,
        output_name,
    })
}

/// Parse the decoder's header lines.
pub fn parse_decode_stdout(stdout: &str, mode: DecodeMode) -> Result<DecodeStats, Mars1Error> {
    const BIN: &str = "decmars";
    let raw = field(BIN, stdout, "Original image size", "Original image size")?;
    let (w, h) = raw
        .split_once('x')
        .ok_or_else(|| Mars1Error::UnparsableField {
            bin: BIN.into(),
            field: "Original image size",
            line: raw.to_string(),
        })?;
    Ok(DecodeStats {
        mode,
        width: parse_u64(BIN, w.trim(), "Original image size")? as u32,
        height: parse_u64(BIN, h.trim(), "Original image size")? as u32,
    })
}

/// Run `encmars` on `input` (relative to `workdir`), writing `output`.
///
/// `width` and `height` describe the headerless raw input. Every consistency check
/// described in the module docs runs before this returns.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    bins: &Mars1Binaries,
    workdir: &Path,
    input: &str,
    output: &str,
    width: u32,
    height: u32,
    params: &EncodeParams,
) -> Result<EncodeOutcome, Mars1Error> {
    let args = params.args(width, height, input, output);
    let (out, seconds) = run(&bins.encmars, workdir, &args)?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stats = parse_encode_stdout(&stdout)?;

    // The method check. `method INIT(= MassCenter)` means a wrong or dropped flag does not
    // fail, it silently measures a different algorithm — and every row downstream would be
    // mislabelled rather than missing.
    if stats.method_reported != params.method.reported_label() {
        return Err(Mars1Error::MethodMismatch {
            requested: params.method.reported_label(),
            flag: params.method.flag(),
            reported: stats.method_reported,
        });
    }

    // The encoder echoes the thresholds actually in force. Comparing them to what we
    // passed catches an argument-order slip, which otherwise reads as a real RD result.
    check_echo("T_RMS", stats.t_rms, params.t_rms)?;
    if let Some(e) = params.t_ent {
        check_echo("T_ENT", stats.t_ent, e)?;
    }
    if let Some(v) = params.t_var {
        check_echo("T_VAR", stats.t_var, v)?;
    }

    if stats.transforms == 0 {
        return Err(Mars1Error::NoTransforms);
    }

    // The printed ratio must be the printed integers' ratio. If the three disagree, one of
    // them was misparsed and the headline metric would be wrong by an unknown factor.
    let derived = format!("{:.6}", stats.comparisons as f64 / stats.transforms as f64);
    let printed = format!("{:.6}", stats.comparisons_per_transform_reported);
    if derived != printed {
        return Err(Mars1Error::RatioMismatch {
            printed,
            derived,
            comparisons: stats.comparisons,
            transforms: stats.transforms,
        });
    }

    let path = workdir.join(output);
    let coded_bytes = std::fs::metadata(&path)
        .map_err(|source| Mars1Error::Io {
            path: path.display().to_string(),
            source,
        })?
        .len();
    if coded_bytes != stats.bytes_written {
        return Err(Mars1Error::ByteCountMismatch {
            printed: stats.bytes_written,
            actual: coded_bytes,
            path: path.display().to_string(),
        });
    }

    Ok(EncodeOutcome {
        params: params.clone(),
        stats,
        coded_bytes,
        indicative_seconds: seconds,
    })
}

/// `%f` prints six decimals, so an echoed threshold can only be compared to the value we
/// passed after rounding that value the same way. Equality of the two strings is exact —
/// this is not a tolerance.
fn check_echo(field: &'static str, echoed: f64, passed: f64) -> Result<(), Mars1Error> {
    let e = format!("{echoed:.6}");
    let p = format!("{passed:.6}");
    if e != p {
        return Err(Mars1Error::ThresholdMismatch {
            field,
            echoed: e,
            passed: p,
        });
    }
    Ok(())
}

/// Run `decmars`, writing a PGM (or raw, if the caller asks for it via the filename and
/// `-r` — not used by the baseline sweep, which measures PGM output).
pub fn decode(
    bins: &Mars1Binaries,
    workdir: &Path,
    input: &str,
    output: &str,
    expect: (u32, u32),
    params: &DecodeParams,
) -> Result<DecodeOutcome, Mars1Error> {
    let args = params.args(input, output);
    let (out, seconds) = run(&bins.decmars, workdir, &args)?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stats = parse_decode_stdout(&stdout, params.mode)?;
    if (stats.width, stats.height) != expect {
        return Err(Mars1Error::DecodeSizeMismatch {
            got_w: stats.width,
            got_h: stats.height,
            want_w: expect.0,
            want_h: expect.1,
        });
    }
    Ok(DecodeOutcome {
        params: params.clone(),
        stats,
        indicative_seconds: seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real `encmars` summary, captured verbatim including the `\r`-terminated progress
    /// spam it is embedded in. If the parser can survive this it can survive the sweep.
    const SAMPLE: &str = concat!(
        "\n    /\\        \n   /__\\       Mars -- Version 1.0\n",
        " Reading lena.raw (512x512) ... done\n",
        " Speed-up method: Fisher\n\n",
        " Classifying domain (16x16)  1\r Classifying domain (16x16)  2\r",
        "\n Image Entropy      : 7.448350\n",
        " Image Variance     : 2300.367042\n",
        " Entropy threshold  : 8.000000\n",
        " Variance threshold : 1000000.000000\n",
        " Rms threshold      : 8.000000\n\n",
        " Coding (16x16)  1\r Coding (16x16)  2\r",
        "\n\n Zero_alfa_transformations   : 1\n",
        " Number of transformations   : 4618\n",
        " Number of comparisons       : 1952738\n",
        " Comparisons/Transformations : 422.853616\n",
        " 16504 bytes written in l8f.ifs\n"
    );

    #[test]
    fn the_encoder_summary_parses_out_of_the_progress_spam() {
        let s = parse_encode_stdout(SAMPLE).unwrap();
        assert_eq!(s.method_reported, "Fisher");
        assert_eq!(s.transforms, 4618);
        assert_eq!(s.comparisons, 1_952_738);
        assert_eq!(s.zero_alfa_transforms, 1);
        assert_eq!(s.bytes_written, 16504);
        assert_eq!(s.output_name, "l8f.ifs");
        assert_eq!(s.t_rms, 8.0);
        assert_eq!(s.t_ent, 8.0);
        assert_eq!(s.t_var, 1e6);
        assert_eq!(s.image_entropy, 7.448350);
        // The headline metric comes from the integers, and must agree with the print.
        assert_eq!(
            format!("{:.6}", s.evals_per_transform()),
            format!("{:.6}", s.comparisons_per_transform_reported)
        );
    }

    #[test]
    fn a_missing_field_is_an_error_not_a_default() {
        // Losing one line must not yield a row with a zero in it.
        let mangled = SAMPLE.replace(" Number of comparisons       : 1952738\n", "");
        assert!(matches!(
            parse_encode_stdout(&mangled),
            Err(Mars1Error::MissingField {
                field: "Number of comparisons",
                ..
            })
        ));
    }

    #[test]
    fn every_method_has_a_distinct_flag_key_and_label() {
        let mut flags: Vec<&str> = Method::ALL.iter().map(|m| m.flag()).collect();
        let mut keys: Vec<&str> = Method::ALL.iter().map(|m| m.key()).collect();
        let mut labels: Vec<&str> = Method::ALL.iter().map(|m| m.reported_label()).collect();
        for v in [&mut flags, &mut keys, &mut labels] {
            let before = v.len();
            v.sort_unstable();
            v.dedup();
            assert_eq!(v.len(), before, "duplicate entry in the method table");
        }
        assert_eq!(flags.len(), 6);
        // Round-trip through the config spelling.
        for m in Method::ALL {
            assert_eq!(m.key().parse::<Method>().unwrap(), m);
        }
    }

    #[test]
    fn a_method_has_exactly_one_spelling_everywhere() {
        // The config file, the result row and the curve label must all say the same
        // word. A derived `rename_all` silently produced a second spelling once already.
        for m in Method::ALL {
            let json = serde_json::to_string(&m).unwrap();
            assert_eq!(json, format!("\"{}\"", m.key()));
            assert_eq!(serde_json::from_str::<Method>(&json).unwrap(), m);
        }
        for d in [DecodeMode::Pyramidal, DecodeMode::Iterative] {
            let json = serde_json::to_string(&d).unwrap();
            assert_eq!(json, format!("\"{}\"", d.key()));
            assert_eq!(serde_json::from_str::<DecodeMode>(&json).unwrap(), d);
        }
    }

    #[test]
    fn the_method_flag_is_always_passed_explicitly() {
        // globals.h:201 makes the unflagged default MassCenter, so an EncodeParams that
        // emitted no method flag would silently measure the wrong algorithm.
        for m in Method::ALL {
            let p = EncodeParams {
                method: m,
                ..Default::default()
            };
            let args = p.args(512, 512, "i.raw", "o.ifs");
            assert!(args.contains(&m.flag().to_string()), "{m:?} flag missing");
            assert_eq!(args[args.len() - 2], "i.raw");
            assert_eq!(args[args.len() - 1], "o.ifs");
        }
    }

    #[test]
    fn omitted_presplit_thresholds_are_left_at_the_1998_defaults() {
        let p = EncodeParams::default();
        let args = p.args(512, 512, "i.raw", "o.ifs");
        assert!(!args.contains(&"-e".to_string()));
        assert!(!args.contains(&"-v".to_string()));
        let p = EncodeParams {
            t_ent: Some(4.0),
            t_var: Some(1e3),
            ..Default::default()
        };
        let args = p.args(512, 512, "i.raw", "o.ifs");
        assert!(args.windows(2).any(|w| w[0] == "-e" && w[1] == "4.0"));
        assert!(args.windows(2).any(|w| w[0] == "-v" && w[1] == "1000.0"));
    }

    #[test]
    fn floats_are_rendered_so_atof_reads_them_back_exactly() {
        assert_eq!(fmt_c_float(8.0), "8.0");
        assert_eq!(fmt_c_float(1e6), "1000000.0");
        assert_eq!(fmt_c_float(0.5), "0.5");
        assert_eq!(fmt_c_float(2.0), "2.0");
        for v in [2.0f64, 4.0, 8.0, 16.0, 32.0, 1.0, 0.1, 1e6, 1e9] {
            assert_eq!(fmt_c_float(v).parse::<f64>().unwrap(), v);
        }
    }

    #[test]
    fn decode_flags_match_the_1998_option_table() {
        // Pyramidal is the default and therefore adds no flag; -i selects iterative.
        let p = DecodeParams::default();
        assert_eq!(p.mode, DecodeMode::Pyramidal);
        assert_eq!(p.args("a.ifs", "b.pgm"), vec!["a.ifs", "b.pgm"]);
        let p = DecodeParams {
            mode: DecodeMode::Iterative,
            iterations: Some(12),
            postprocess: false,
        };
        assert_eq!(
            p.args("a.ifs", "b.pgm"),
            vec!["-i", "-n", "12", "a.ifs", "b.pgm"]
        );
    }

    #[test]
    fn the_decoder_header_parses() {
        let s = " Reading a.ifs ... done\n Original image size: 768x512\n\
                 \n Decoding at low resolution (192x128) 0\r Writing b.pgm ... done\n";
        let d = parse_decode_stdout(s, DecodeMode::Pyramidal).unwrap();
        assert_eq!((d.width, d.height), (768, 512));
        assert_eq!(d.mode, DecodeMode::Pyramidal);
    }
}
