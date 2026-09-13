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
use mars_bench::mars1::Mars1Binaries;
use mars_bench::mars1_report::{self, Mars1Report};
use mars_bench::measure::{file_size, measure, MeasureRequest};
use mars_bench::provenance::Provenance;
use mars_bench::report::Report;
use mars_bench::store::{read_rows, ResultStore, Row};
use mars_bench::sweep::{plan, run, ImageSet, Job, RunContext, SweepConfig};
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
    /// Step 2: run the Mars 1 parameter sweep and append it to the result store.
    Mars1Sweep(Mars1SweepArgs),
    /// Step 2: render the Mars 1 baseline report from that store.
    Mars1Report(Mars1ReportArgs),
    /// Step 2's exit criteria as a command that exits 0 or 1 (§A1).
    Mars1Check(Mars1CheckArgs),
}

#[derive(Args)]
struct Mars1CheckArgs {
    #[arg(long, default_value = "results/baseline-mars1.jsonl")]
    store: PathBuf,
    /// The grid the sweep was supposed to run. Checked against, so a half-finished sweep
    /// fails here rather than producing a report about whatever it managed.
    #[arg(long, default_value = "configs/baseline-mars1.json")]
    config: PathBuf,
    #[arg(long, default_value = "standard")]
    corpus: String,
    #[arg(long, default_value = "default")]
    variant: String,
    #[arg(long, default_value = "fisher")]
    reference: String,
}

#[derive(Args)]
struct Mars1SweepArgs {
    /// The sweep grid. §2.3: every experiment is a config file, not a code edit.
    #[arg(long, default_value = "configs/baseline-mars1.json")]
    config: PathBuf,
    /// Append-only JSONL to write to (§M7).
    #[arg(long, default_value = "results/baseline-mars1.jsonl")]
    out: PathBuf,
    /// Where `just mars1` put the 1998 binaries, relative to the repo root.
    #[arg(long, default_value = "target/mars1")]
    mars1_dir: PathBuf,
    /// Worker processes. Defaults to the P-core count (§M4); row order does not depend
    /// on it.
    #[arg(long)]
    jobs: Option<usize>,
    /// Short scratch directory for the subprocesses (D3).
    #[arg(long)]
    scratch: Option<PathBuf>,
    /// Run only the first N jobs. For smoke-testing the pipeline, never for results.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Args)]
struct Mars1ReportArgs {
    #[arg(long, default_value = "results/baseline-mars1.jsonl")]
    store: PathBuf,
    /// Corpus carrying the headline numbers.
    #[arg(long, default_value = "standard")]
    corpus: String,
    /// Variant carrying them.
    #[arg(long, default_value = "default")]
    variant: String,
    /// Decode mode carrying them. Never defaulted silently — it is printed on every table.
    #[arg(long, default_value = "pyramidal")]
    decode: String,
    /// BD-rate reference method.
    #[arg(long, default_value = "fisher")]
    reference: String,
    #[arg(long)]
    markdown_out: Option<PathBuf>,
    #[arg(long)]
    html_out: Option<PathBuf>,
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
        Cmd::Mars1Sweep(a) => mars1_sweep(a),
        Cmd::Mars1Report(a) => mars1_report(a),
        Cmd::Mars1Check(a) => mars1_check(a),
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

// ------------------------------------------------------------- Mars 1 baseline (Step 2)

fn mars1_sweep(a: Mars1SweepArgs) -> Result<()> {
    let root = repo_root(&a.config)?;
    let config = SweepConfig::read(&a.config)?;
    config.validate()?;

    let mut sets = Vec::new();
    for spec in &config.corpora {
        let path = root.join(&spec.index);
        let set = ImageSet::read(&path)?;
        eprintln!(
            "corpus {:<10} {:>3} images  {}",
            set.set,
            set.images.len(),
            spec.index.display()
        );
        sets.push((spec.clone(), set));
    }

    let mut jobs = plan(&config, &sets)?;
    if let Some(limit) = a.limit {
        jobs.truncate(limit);
    }
    let bins = Mars1Binaries::from_dir(root.join(&a.mars1_dir))?;

    // D3: the reference stores filenames in char[50], so every child runs in a short
    // directory with relative names. One directory per worker keeps them from colliding.
    let scratch = a
        .scratch
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/mars1-sweep-{}", std::process::id())));
    std::fs::create_dir_all(&scratch)
        .with_context(|| format!("creating scratch dir {}", scratch.display()))?;

    // §M4: pin concurrency to the P-core count. Defaulting to every core would put half
    // the jobs on efficiency cores and bend even the indicative timings.
    let workers = a.jobs.unwrap_or_else(p_cores).max(1);

    let provenance = Provenance::detect(0)
        .with_corpus_manifest(&std::fs::read(root.join(&config.corpora[0].index))?)
        .with_parameter_set(&config);

    eprintln!(
        "{} jobs x {} decode mode(s) on {} worker(s) -> {}",
        jobs.len(),
        config.decode_modes.len(),
        workers,
        a.out.display()
    );

    let ctx = RunContext {
        bins: &bins,
        repo_root: &root,
        scratch: &scratch,
        decode_modes: &config.decode_modes,
        jobs: workers,
        sweep_name: &config.name,
    };

    let started = std::time::Instant::now();
    let mut store = ResultStore::open(&a.out)?;
    let progress = move |done: usize, total: usize, job: &Job| {
        if done % 25 == 0 || done == total {
            let el = started.elapsed().as_secs_f64();
            let eta = el / done as f64 * (total - done) as f64;
            eprintln!(
                "  {done:>5}/{total}  {:>5.0}s elapsed, ~{eta:.0}s left   (last: {} {} {} r={})",
                el,
                job.image.name,
                job.variant,
                job.params.method.key(),
                job.params.t_rms
            );
        }
    };
    let written = run(&ctx, &jobs, &provenance, &mut store, &progress)?;

    std::fs::remove_dir_all(&scratch).ok();
    eprintln!(
        "appended {written} rows to {} in {:.0}s",
        a.out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Performance-core count on Apple Silicon; a conservative guess elsewhere.
fn p_cores() -> usize {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.perflevel0.physicalcpu"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
        .unwrap_or(4)
}

/// Walk up from the config file to the directory holding `Cargo.toml`, so image paths
/// recorded relative to the repo root resolve wherever the command is run from.
fn repo_root(from: &Path) -> Result<PathBuf> {
    let mut dir = std::fs::canonicalize(from)
        .with_context(|| format!("resolving {}", from.display()))?
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("crates").is_dir() {
            return Ok(dir);
        }
        if !dir.pop() {
            bail!(
                "could not find the repository root above {}",
                from.display()
            );
        }
    }
}

fn mars1_report(a: Mars1ReportArgs) -> Result<()> {
    let rows = mars1_report::load(&a.store)?;
    if rows.is_empty() {
        bail!("{} contains no baseline-mars1 rows", a.store.display());
    }

    let p = Provenance::detect(0);
    let build = rows
        .iter()
        .find_map(|r| r.mars1_build_info.clone())
        .unwrap_or_default();
    let sweep_sha: Vec<String> = {
        let mut v: Vec<String> = read_rows(&a.store)?
            .iter()
            .map(|r| r.provenance.git_sha[..r.provenance.git_sha.len().min(12)].to_string())
            .collect();
        v.sort();
        v.dedup();
        v
    };

    let note = format!(
        "{} rows from `{}` · sweep at git {} · report at {}{} · harness {} · {} ({} P / {} E cores) · {}\n\n\
         <details><summary>Mars 1 reference build</summary>\n\n```\n{}```\n\n</details>",
        rows.len(),
        a.store.display(),
        sweep_sha.join(", "),
        &p.git_sha[..p.git_sha.len().min(12)],
        if p.git_dirty { "-dirty" } else { "" },
        p.harness_version,
        p.machine.cpu_brand,
        p.machine.p_cores.map_or("?".into(), |c| c.to_string()),
        p.machine.e_cores.map_or("?".into(), |c| c.to_string()),
        p.timestamp_utc,
        build,
    );

    let variants: Vec<String> = {
        let mut v: Vec<String> = rows
            .iter()
            .filter(|r| r.corpus == a.corpus && r.variant != a.variant)
            .map(|r| r.variant.clone())
            .collect();
        v.sort();
        v.dedup();
        v
    };

    let report = Mars1Report {
        corpus: a.corpus.clone(),
        variant: a.variant.clone(),
        decode_mode: a.decode.parse().map_err(|e: String| anyhow::anyhow!(e))?,
        reference_method: a.reference.clone(),
        variants,
        provenance_note: note.clone(),
        rows,
    };

    let markdown = report.to_markdown();
    if let Some(out) = &a.markdown_out {
        std::fs::write(out, &markdown)?;
        eprintln!("wrote {}", out.display());
    }
    if let Some(out) = &a.html_out {
        let plot = Report::new(
            format!(
                "Mars 1 baseline — {} · {} · {} decode",
                report.corpus,
                report.variant,
                report.decode_mode.key()
            ),
            report.plot_curves(),
        )
        .with_reference(report.reference_method.clone())
        .with_provenance_note(note);
        std::fs::write(out, plot.to_html())?;
        eprintln!("wrote {}", out.display());
    }
    if a.markdown_out.is_none() && a.html_out.is_none() {
        print!("{markdown}");
    }
    Ok(())
}

fn mars1_check(a: Mars1CheckArgs) -> Result<()> {
    let config = SweepConfig::read(&a.config)?;
    config.validate()?;
    let rows = mars1_report::load(&a.store)?;

    // §M7: a row without provenance is not a result. Checked on the envelope, which the
    // payload loader discards.
    let envelopes = read_rows(&a.store)?;
    let unprovenanced = envelopes
        .iter()
        .filter(|r| {
            r.provenance.git_sha == "unknown"
                || r.provenance.corpus_manifest_sha256.is_none()
                || r.provenance.parameter_set_sha256.is_none()
        })
        .count();

    let methods: Vec<String> = config.methods.iter().map(|m| m.key().to_string()).collect();
    let mut checks = mars1_report::gate(
        &rows,
        &a.corpus,
        &a.variant,
        &config.decode_modes,
        &methods,
        config.rms.len(),
        &a.reference,
    );
    checks.insert(
        0,
        mars1_report::Check {
            name: "every row carries a full provenance block (§M7)".into(),
            passed: unprovenanced == 0,
            detail: format!("{} of {} rows incomplete", unprovenanced, envelopes.len()),
        },
    );

    let mut failed = 0;
    for c in &checks {
        let mark = if c.passed {
            "PASS"
        } else {
            failed += 1;
            "FAIL"
        };
        println!("{mark}  {}\n      {}", c.name, c.detail);
    }
    if failed > 0 {
        bail!("gate-2: {failed} of {} checks failed", checks.len());
    }
    println!("\ngate-2: PASS ({} checks)", checks.len());
    Ok(())
}
