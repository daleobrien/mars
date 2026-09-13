//! `marsbench` — the measurement front end.
//!
//! Three subcommands, matching Step 1's deliverable:
//!   `metrics`  one (original, decoded) pair -> every §M2 metric, optionally appended
//!              to the result store.
//!   `bdrate`   two RD curves -> BD-rate and BD-PSNR with their intervals (§M3).
//!   `report`   curves -> Markdown and HTML with RD plots.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use mars_bench::bdrate::{bd_metrics, RdCurve, RdPoint};
use mars_bench::measure::{file_size, measure, MeasureRequest};
use mars_bench::provenance::Provenance;
use mars_bench::report::Report;
use mars_bench::store::{ResultStore, Row};
use mars_core::metrics::Downsample;

#[derive(Parser)]
#[command(
    name = "marsbench",
    about = "Mars 2 measurement harness (§M1-M10 of the measurement contract)",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compute every pinned quality metric for one (original, decoded) pair.
    Metrics(MetricsArgs),
    /// BD-rate and BD-PSNR between two curves given as JSON.
    Bdrate(BdrateArgs),
    /// Render curves as Markdown and/or HTML with RD plots.
    Report(ReportArgs),
}

#[derive(Args)]
struct MetricsArgs {
    /// The reference image.
    original: PathBuf,
    /// The decoded image.
    decoded: PathBuf,
    /// Width and height for headerless raw input, e.g. `--raw-dims 512x512`.
    #[arg(long, value_name = "WxH")]
    raw_dims: Option<String>,
    /// Measure this region only (§M2 — Mars 1's power-of-two padding is never measured).
    #[arg(long, value_name = "WxH")]
    region: Option<String>,
    /// Bitstream whose size gives bpp. §M2: the whole file, header included.
    #[arg(long)]
    coded: Option<PathBuf>,
    /// Append the row to this append-only JSONL store.
    #[arg(long)]
    store: Option<PathBuf>,
    /// Free-text tag recorded with the row, e.g. the parameter set under test.
    #[arg(long)]
    tag: Option<String>,
    /// MS-SSIM inter-scale decimation. The default is the pinned one; `scipy-uniform2`
    /// exists only so the scale chain can be cross-validated against `sewar`, which
    /// uses a half-pixel-shifted window. Never use it for reported numbers.
    #[arg(long, value_enum, default_value_t = DownsampleArg::Box2x2)]
    ms_ssim_downsample: DownsampleArg,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum DownsampleArg {
    Box2x2,
    ScipyUniform2,
}

impl From<DownsampleArg> for Downsample {
    fn from(a: DownsampleArg) -> Self {
        match a {
            DownsampleArg::Box2x2 => Downsample::Box2x2,
            DownsampleArg::ScipyUniform2 => Downsample::ScipyUniform2,
        }
    }
}

#[derive(Args)]
struct BdrateArgs {
    /// JSON file: `{"label": "...", "points": [{"bpp": .., "psnr": ..}, ...]}`.
    reference: PathBuf,
    /// Same shape; compared against the reference.
    test: PathBuf,
}

#[derive(Args)]
struct ReportArgs {
    /// JSON file: an array of curves.
    curves: PathBuf,
    #[arg(long)]
    title: Option<String>,
    /// Curve label to use as the BD-rate reference.
    #[arg(long)]
    reference: Option<String>,
    #[arg(long)]
    markdown_out: Option<PathBuf>,
    #[arg(long)]
    html_out: Option<PathBuf>,
}

fn parse_dims(s: &str) -> Result<(usize, usize)> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .with_context(|| format!("expected WxH, got {s:?}"))?;
    Ok((w.trim().parse()?, h.trim().parse()?))
}

fn read_curve(path: &Path) -> Result<RdCurve> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("parsing {} as an RD curve", path.display()))
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Metrics(a) => metrics(a),
        Cmd::Bdrate(a) => bdrate(a),
        Cmd::Report(a) => report(a),
    }
}

fn metrics(a: MetricsArgs) -> Result<()> {
    let mut req = MeasureRequest::new(&a.original, &a.decoded);
    if let Some(d) = a.raw_dims.as_deref() {
        let (w, h) = parse_dims(d)?;
        req = req.with_raw_dims(w, h);
    }
    if let Some(d) = a.region.as_deref() {
        let (w, h) = parse_dims(d)?;
        req = req.with_region(w, h);
    }
    if let Some(coded) = a.coded.as_deref() {
        req = req.with_coded_bytes(file_size(coded)?);
    }
    req.ms_ssim.downsample = a.ms_ssim_downsample.into();

    let m = measure(&req)?;
    println!("{}", serde_json::to_string_pretty(&m)?);

    if let Some(store_path) = a.store {
        let mut data = serde_json::to_value(&m)?;
        if let Some(tag) = a.tag {
            data["tag"] = serde_json::Value::String(tag);
        }
        let row = Row::new("quality", Provenance::detect(0), &data)?;
        ResultStore::open(&store_path)?.append(&row)?;
        eprintln!("appended 1 row to {}", store_path.display());
    }
    Ok(())
}

fn bdrate(a: BdrateArgs) -> Result<()> {
    let reference = read_curve(&a.reference)?;
    let test = read_curve(&a.test)?;
    match bd_metrics(&reference, &test) {
        Ok(r) => {
            println!("{}", serde_json::to_string_pretty(&r)?);
            Ok(())
        }
        // §M3: refuse rather than emit a quotable number from invalid input.
        Err(e) => bail!("BD metrics undefined: {e}"),
    }
}

fn report(a: ReportArgs) -> Result<()> {
    let text = std::fs::read_to_string(&a.curves)
        .with_context(|| format!("reading {}", a.curves.display()))?;
    let curves: Vec<RdCurve> = serde_json::from_str(&text)
        .with_context(|| format!("parsing {} as an array of RD curves", a.curves.display()))?;
    if curves.is_empty() {
        bail!("no curves to report");
    }

    let p = Provenance::detect(0);
    let mut r = Report::new(
        a.title.unwrap_or_else(|| "Mars 2 — rate-distortion".into()),
        curves,
    )
    .with_provenance_note(format!(
        "{} @ {}{} · harness {} · {} ({} P-cores / {} E-cores) · {}",
        p.build_profile,
        &p.git_sha[..p.git_sha.len().min(12)],
        if p.git_dirty { "-dirty" } else { "" },
        p.harness_version,
        p.machine.cpu_brand,
        p.machine.p_cores.map_or("?".into(), |c| c.to_string()),
        p.machine.e_cores.map_or("?".into(), |c| c.to_string()),
        p.timestamp_utc,
    ));
    if let Some(reference) = a.reference {
        r = r.with_reference(reference);
    }

    let md = r.to_markdown();
    match a.markdown_out {
        Some(path) => {
            std::fs::write(&path, &md)?;
            eprintln!("wrote {}", path.display());
        }
        None => print!("{md}"),
    }
    if let Some(path) = a.html_out {
        std::fs::write(&path, r.to_html())?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

/// Keep `RdPoint` reachable from the binary's docs; the CLI consumes curves as JSON.
#[allow(dead_code)]
fn _doc_anchor(p: RdPoint) -> f64 {
    p.bpp
}
