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
use mars_bench::anchors::{self, AnchorsConfig};
use mars_bench::anchors_report;
use mars_bench::bdrate::{bd_metrics, RdCurve, RdPoint};
use mars_bench::mars1::Mars1Binaries;
use mars_bench::mars1_fixtures::{self, FixtureConfig, FIXTURE_DIR};
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
    /// Step 3: regenerate the golden `.ifs` fixtures and their manifest.
    Mars1Fixtures(Mars1FixturesArgs),
    /// Step 4: run the anchor-codec quality sweeps and append them to the result store.
    AnchorsSweep(AnchorsSweepArgs),
    /// Step 4: render the anchors + Mars 1 combined RD report from the stores.
    AnchorsReport(AnchorsReportArgs),
    /// Step 4's exit criteria as a command that exits 0 or 1 (§A1, gate-4).
    AnchorsCheck(AnchorsCheckArgs),
    /// Step 5's exit criteria as a command that exits 0 or 1 (§A1, gate-5).
    IfsCheck(IfsCheckArgs),
    /// Step 6's exit criteria as a command that exits 0 or 1 (§A1, gate-6).
    RustEncoderCheck(RustEncoderCheckArgs),
    /// Step 7's bit-identical + speedup exit criteria as a command that exits 0 or 1
    /// (§A1, gate-7).
    GpuSearchCheck(GpuSearchCheckArgs),
    /// Step 7's A/B-interleaved GPU-vs-Rayon-CPU speed comparison, reported only (not
    /// gated on its own -- `gpu-search-check` folds the speedup threshold into gate-7).
    GpuSearchBench(GpuSearchBenchArgs),
}

#[derive(Args)]
struct AnchorsSweepArgs {
    /// The codec/param grid. §2.3: every experiment is a config file, not a code edit.
    #[arg(long, default_value = "configs/anchors.json")]
    config: PathBuf,
    /// Append-only JSONL to write to (§M7).
    #[arg(long, default_value = "results/anchors.jsonl")]
    out: PathBuf,
    /// Worker threads. Defaults to the P-core count (§M4); row order does not depend on it.
    #[arg(long)]
    jobs: Option<usize>,
    #[arg(long)]
    scratch: Option<PathBuf>,
    /// Run only the first N jobs. For smoke-testing the pipeline, never for results.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Args)]
struct AnchorsReportArgs {
    #[arg(long, default_value = "results/anchors.jsonl")]
    store: PathBuf,
    /// The Step 2 Mars 1 baseline, whose six methods share the plot and the BD-rate table.
    #[arg(long, default_value = "results/baseline-mars1.jsonl")]
    mars1_store: PathBuf,
    /// BD-rate reference codec -- JPEG, the anchor of record (plan's Step 4 exit criteria).
    #[arg(long, default_value = "jpeg")]
    reference: String,
    #[arg(long)]
    markdown_out: Option<PathBuf>,
    #[arg(long)]
    html_out: Option<PathBuf>,
}

#[derive(Args)]
struct AnchorsCheckArgs {
    #[arg(long, default_value = "results/anchors.jsonl")]
    store: PathBuf,
    #[arg(long, default_value = "configs/anchors.json")]
    config: PathBuf,
    #[arg(long, default_value = "jpeg")]
    reference: String,
    #[arg(long, default_value = "results/anchors.md")]
    markdown_out: PathBuf,
    #[arg(long, default_value = "results/anchors.html")]
    html_out: PathBuf,
}

#[derive(Args)]
struct IfsCheckArgs {
    #[arg(long, default_value = "fixtures/mars1/manifest.toml")]
    manifest: PathBuf,
    #[arg(long, default_value = "configs/mars1-fixtures.json")]
    config: PathBuf,
    /// Where `just mars1` put the 1998 binaries; only `decmars` is used.
    #[arg(long, default_value = "target/mars1")]
    mars1_dir: PathBuf,
    #[arg(long, default_value = "target/ifs-check")]
    scratch: PathBuf,
}

#[derive(Args)]
struct RustEncoderCheckArgs {
    #[arg(long, default_value = "corpus/fixtures.images.json")]
    fixtures_index: PathBuf,
    /// Step 2's baseline store, for the Fisher reference curves.
    #[arg(long, default_value = "results/baseline-mars1.jsonl")]
    baseline_store: PathBuf,
    /// Where `just mars1` put the 1998 binaries; only `decmars` is used.
    #[arg(long, default_value = "target/mars1")]
    mars1_dir: PathBuf,
    #[arg(long, default_value = "target/rust-encoder-check")]
    scratch: PathBuf,
}

#[derive(Args)]
struct GpuSearchCheckArgs {
    #[arg(long, default_value = "corpus/fixtures.images.json")]
    fixtures_index: PathBuf,
    #[arg(long, default_value = "corpus/standard.images.json")]
    standard_index: PathBuf,
    /// A/B-interleaved timing repetitions for the speedup check folded into this gate.
    #[arg(long, default_value_t = 5)]
    bench_runs: usize,
    /// Skip the timing half (speed varies by machine load; the bit-identical half is the
    /// hard requirement and this flag lets it be checked on its own, e.g. in a sandboxed
    /// environment with a software GPU adapter).
    #[arg(long)]
    skip_bench: bool,
}

#[derive(Args)]
struct GpuSearchBenchArgs {
    #[arg(long, default_value = "corpus/standard.images.json")]
    standard_index: PathBuf,
    /// Image names to benchmark. Defaults to a handful of Kodak images spanning the
    /// sweep the differential test already exercises.
    #[arg(long, value_delimiter = ',')]
    images: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "16,32")]
    sizes: Vec<u32>,
    #[arg(long, default_value_t = 5)]
    runs: usize,
}

#[derive(Args)]
struct Mars1FixturesArgs {
    /// The fixture set. §2.3: the set is a config file, not a code edit.
    #[arg(long, default_value = "configs/mars1-fixtures.json")]
    config: PathBuf,
    /// Where `just mars1` put the 1998 binaries.
    #[arg(long, default_value = "target/mars1")]
    mars1_dir: PathBuf,
    /// Scratch space for the encodes and decodes; nothing here is committed.
    #[arg(long, default_value = "target/mars1-fixtures")]
    scratch: PathBuf,
    /// Plan the set and print it without running anything.
    #[arg(long)]
    dry_run: bool,
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
        Cmd::Mars1Fixtures(a) => mars1_fixtures(a),
        Cmd::AnchorsSweep(a) => anchors_sweep(a),
        Cmd::AnchorsReport(a) => anchors_report_cmd(a),
        Cmd::AnchorsCheck(a) => anchors_check(a),
        Cmd::IfsCheck(a) => ifs_check(a),
        Cmd::RustEncoderCheck(a) => rust_encoder_check(a),
        Cmd::GpuSearchCheck(a) => gpu_search_check(a),
        Cmd::GpuSearchBench(a) => gpu_search_bench(a),
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

// ------------------------------------------------------------------ GPU search (Step 7)

fn gpu_search_check(a: GpuSearchCheckArgs) -> Result<()> {
    let root = Path::new(".");
    let scope = mars_bench::gpu_search::DEFAULT_SCOPE;
    eprintln!(
        "gpu-search-check: {}",
        mars_bench::gpu_search::machine_fingerprint_line()
    );
    // Pin Rayon's global pool before *any* `par_iter()` use in this process (including the
    // differential test's CPU reference below) -- the pool can only be built once, and
    // Rayon lazily builds an unpinned default pool on first use if this isn't done first.
    if !a.skip_bench {
        let threads = p_cores().max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("pinning Rayon to the P-core count")?;
        eprintln!("  (Rayon pinned to {threads} P-core threads)");
    }
    let outcome = mars_bench::gpu_search::gate(root, &a.fixtures_index, &a.standard_index, &scope)
        .context("running Step 7 differential test")?;

    // D25 (CONTRACT-CHANGE, user-directed): bit-identical equality is no longer required.
    // A block where one side found a domain and the other found none is a structural
    // disagreement and always fails; among blocks where both sides found a candidate,
    // divergence is bounded by rate and by each diverging block's relative rms gap
    // ("close in compression", since rms is exactly the distortion term being minimised).
    const MAX_DIVERGENCE_RATE: f64 = 0.001; // D23 observed 0.00297% -- >30x margin
    const MAX_RELATIVE_RMS_DELTA: f64 = 0.01; // D23's worst case was ~0.067% -- >14x margin

    let mut hard_divergences = 0usize;
    let mut max_relative_rms_delta = 0.0f64;
    for d in &outcome.divergences {
        match (&d.cpu, &d.gpu) {
            (Some(c), Some(g)) => {
                let (rc, rg) = (c.rms, f64::from(g.rms));
                let rel = (rc - rg).abs() / rc.max(rg).max(f64::EPSILON);
                max_relative_rms_delta = max_relative_rms_delta.max(rel);
            }
            _ => hard_divergences += 1,
        }
    }
    let divergence_rate = if outcome.compared == 0 {
        0.0
    } else {
        outcome.divergences.len() as f64 / outcome.compared as f64
    };
    let identical = hard_divergences == 0
        && divergence_rate <= MAX_DIVERGENCE_RATE
        && max_relative_rms_delta <= MAX_RELATIVE_RMS_DELTA;

    println!(
        "{}  close to CPU search (D25): {} blocks compared, {} divergence(s) ({:.4}%, \
         limit {:.2}%), {} structural, worst relative |delta rms| {:.4}% (limit {:.2}%)",
        if identical { "PASS" } else { "FAIL" },
        outcome.compared,
        outcome.divergences.len(),
        divergence_rate * 100.0,
        MAX_DIVERGENCE_RATE * 100.0,
        hard_divergences,
        max_relative_rms_delta * 100.0,
        MAX_RELATIVE_RMS_DELTA * 100.0,
    );
    if !outcome.divergences.is_empty() {
        println!("\n  divergences (full diagnostic detail -- §A7, never averaged away):");
        for d in &outcome.divergences {
            println!("  {} size={} block=({}, {})", d.image, d.size, d.row, d.col);
            match &d.cpu {
                Some(c) => println!(
                    "      cpu: dom=({}, {}) iso={} qalfa={} qbeta={} rms={:.10} moments={:?}",
                    c.dom_row, c.dom_col, c.isometry, c.qalfa, c.qbeta, c.rms, c.moments
                ),
                None => println!("      cpu: no valid domain position"),
            }
            match &d.gpu {
                Some(g) => println!(
                    "      gpu: dom=({}, {}) iso={} qalfa={} qbeta={} rms={:.10}",
                    g.dom_row, g.dom_col, g.isometry, g.qalfa, g.qbeta, g.rms
                ),
                None => println!("      gpu: no valid domain position"),
            }
        }
    }

    println!(
        "\nu32-accumulator check (P7.1): largest raw moment magnitude observed across all \
         compared blocks = {} (u32::MAX = {})",
        outcome.max_u32_moment_observed,
        u32::MAX
    );

    let mut bench_ok = true;
    let mut bench_summary = String::from("skipped (--skip-bench)");
    if !a.skip_bench {
        // §M4 / benchmark-protocol: Rayon is already pinned to the P-core count above (before
        // the differential test's own `par_iter()` use), so the CPU baseline in this
        // comparison is not artificially fast from scheduling onto efficiency cores.
        let images = mars_bench::sweep::ImageSet::read(&root.join(&a.standard_index))?;
        let mut speedups = Vec::new();
        let gpu = mars_gpu::GpuSearcher::new().context("initialising GPU for speed bench")?;
        let bench_images = ["kodim05", "kodim13"];
        for entry in images
            .images
            .iter()
            .filter(|e| bench_images.contains(&e.name.as_str()))
        {
            let image = mars_core::io::read_raw(
                &root.join(&entry.file),
                entry.width as usize,
                entry.height as usize,
            )?;
            for &size in &[16u32, 32] {
                let params = mars_codec::encode::EncodeParams {
                    min_size: 4,
                    max_size: 32,
                    shift: 4,
                    bits_alfa: 4,
                    bits_beta: 7,
                    max_alfa: 1.0,
                    t_rms: 0.0,
                    zero_threshold: 0,
                };
                let r =
                    mars_bench::gpu_search::ab_compare(&image, size, &params, &gpu, a.bench_runs);
                println!(
                    "  {:<10} size={:<3} blocks={:<6} cpu={:>8.1?} (+/-{:.1?})  \
                     gpu={:>8.1?} (+/-{:.1?})  speedup={:.1}x",
                    entry.name,
                    size,
                    r.blocks,
                    r.cpu_median,
                    r.cpu_mad,
                    r.gpu_median,
                    r.gpu_mad,
                    r.speedup
                );
                speedups.push(r.speedup);
            }
        }
        // D25 (CONTRACT-CHANGE, user-directed): floor lowered from the brief's >= 50x to
        // >= 8x, below every point D24 measured (11.4-22.5x) -- see that entry for why a
        // demonstrated, understood 11-22x still satisfies the brief's actual goal ("hours
        // not weeks" for a full oracle build) even though it misses the number the brief
        // guessed at before anything was measured.
        const MIN_SPEEDUP: f64 = 8.0;
        let min_speedup = speedups.iter().cloned().fold(f64::INFINITY, f64::min);
        bench_ok = min_speedup >= MIN_SPEEDUP;
        bench_summary = format!(
            "worst observed speedup {min_speedup:.1}x across {} (image, size) points \
             (target >= {MIN_SPEEDUP}x per docs/decisions.md D25)",
            speedups.len()
        );
    }
    println!(
        "\n{}  speed: {bench_summary}",
        if bench_ok { "PASS" } else { "FAIL" }
    );

    if !identical || !bench_ok {
        bail!("gate-7: FAIL (bit-identical={identical}, speed={bench_ok})");
    }
    println!("\ngate-7: PASS");
    Ok(())
}

fn gpu_search_bench(a: GpuSearchBenchArgs) -> Result<()> {
    // §M4 / benchmark-protocol: pin Rayon to the P-core count for the CPU side of a
    // timing comparison. Defaulting to every logical core (E-cores included) would make
    // the CPU baseline look faster for scheduling reasons unrelated to the algorithm,
    // understating the GPU's real speedup margin rather than overstating it -- still the
    // wrong number to report.
    let threads = p_cores().max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("pinning Rayon to the P-core count")?;
    eprintln!(
        "{} (Rayon pinned to {threads} P-core threads)",
        mars_bench::gpu_search::machine_fingerprint_line()
    );
    let root = Path::new(".");
    let images = mars_bench::sweep::ImageSet::read(&root.join(&a.standard_index))?;
    let gpu = mars_gpu::GpuSearcher::new().context("initialising GPU")?;
    let want: Vec<&str> = if a.images.is_empty() {
        vec!["kodim01", "kodim05", "kodim13", "kodim19"]
    } else {
        a.images.iter().map(String::as_str).collect()
    };
    let params = mars_codec::encode::EncodeParams {
        min_size: 4,
        max_size: 32,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 0.0,
        zero_threshold: 0,
    };
    for entry in images
        .images
        .iter()
        .filter(|e| want.contains(&e.name.as_str()))
    {
        let image = mars_core::io::read_raw(
            &root.join(&entry.file),
            entry.width as usize,
            entry.height as usize,
        )?;
        for &size in &a.sizes {
            let r = mars_bench::gpu_search::ab_compare(&image, size, &params, &gpu, a.runs);
            println!(
                "{:<10} size={:<3} blocks={:<6} cpu={:>9.1?} (+/-{:.1?})  gpu={:>9.1?} \
                 (+/-{:.1?})  speedup={:.1}x",
                entry.name,
                size,
                r.blocks,
                r.cpu_median,
                r.cpu_mad,
                r.gpu_median,
                r.gpu_mad,
                r.speedup
            );
        }
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

/// Step 3: regenerate the golden `.ifs` fixtures.
///
/// Serial on purpose. It takes a few minutes, it runs once when the set changes, and a
/// concurrent version would buy nothing except a way for two cases to disagree about which
/// of them wrote `quadtree.pgm`.
fn mars1_fixtures(a: Mars1FixturesArgs) -> Result<()> {
    let root = Path::new(".");
    let config = FixtureConfig::read(&a.config)
        .with_context(|| format!("reading {}", a.config.display()))?;
    let cases = config.plan(root)?;

    if a.dry_run {
        for c in &cases {
            println!("{:>4}  {:<12} {}", c.id, c.group, c.stem());
        }
        println!("{} fixtures", cases.len());
        return Ok(());
    }

    let bins = Mars1Binaries::from_dir(&a.mars1_dir)?;
    std::fs::create_dir_all(&a.scratch)?;

    let mut records = Vec::with_capacity(cases.len());
    for c in &cases {
        let r = mars1_fixtures::generate(&bins, root, &a.scratch, c)
            .with_context(|| format!("generating fixture {}", c.stem()))?;
        println!(
            "{:>4}/{}  {:<48} {:>7} B  {:>6} transforms ({} DC)",
            c.id + 1,
            cases.len(),
            r.stem,
            r.ifs_bytes,
            r.transforms,
            r.zero_alfa_transforms
        );
        records.push(r);
    }

    let manifest = mars1_fixtures::manifest_toml(&config, &records, bins.build_info.as_deref());
    let path = root.join(FIXTURE_DIR).join("manifest.toml");
    std::fs::write(&path, manifest).with_context(|| format!("writing {}", path.display()))?;
    println!("\nwrote {} ({} fixtures)", path.display(), records.len());
    Ok(())
}

// ------------------------------------------------------------------------ anchors (Step 4)

fn anchors_sweep(a: AnchorsSweepArgs) -> Result<()> {
    let root = repo_root(&a.config)?;
    let config = AnchorsConfig::read(&a.config)?;
    config.validate()?;

    let (corpus_name, images) = anchors::load_corpus(&root, &config.corpus_manifest)?;
    eprintln!(
        "corpus {:<10} {:>3} images  {}",
        corpus_name,
        images.len(),
        config.corpus_manifest.display()
    );

    let mut jobs = anchors::plan(&config, &images)?;
    if let Some(limit) = a.limit {
        jobs.truncate(limit);
    }

    let scratch = a
        .scratch
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/anchors-sweep-{}", std::process::id())));
    std::fs::create_dir_all(&scratch)
        .with_context(|| format!("creating scratch dir {}", scratch.display()))?;

    // §M4: pin concurrency to the P-core count, as the Step 2 sweep does.
    let workers = a.jobs.unwrap_or_else(p_cores).max(1);

    let manifest_path = root.join(&config.corpus_manifest);
    let provenance = Provenance::detect(0)
        .with_corpus_manifest(&std::fs::read(&manifest_path)?)
        .with_parameter_set(&config);

    eprintln!(
        "{} jobs on {} worker(s) -> {}",
        jobs.len(),
        workers,
        a.out.display()
    );

    let ctx = anchors::RunContext {
        repo_root: &root,
        scratch: &scratch,
        jobs: workers,
        sweep_name: &config.name,
    };

    let started = std::time::Instant::now();
    let mut store = ResultStore::open(&a.out)?;
    let progress = move |done: usize, total: usize, job: &anchors::Job| {
        if done % 25 == 0 || done == total {
            let el = started.elapsed().as_secs_f64();
            let eta = el / done as f64 * (total - done) as f64;
            eprintln!(
                "  {done:>5}/{total}  {:>5.0}s elapsed, ~{eta:.0}s left   (last: {} {} param={})",
                el,
                job.image.name,
                job.codec.key(),
                job.param
            );
        }
    };
    let written = anchors::run(&ctx, &jobs, &provenance, &mut store, &progress)?;

    std::fs::remove_dir_all(&scratch).ok();
    eprintln!(
        "appended {written} rows to {} in {:.0}s",
        a.out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn anchors_report_cmd(a: AnchorsReportArgs) -> Result<()> {
    let anchor_rows = anchors_report::load(&a.store)?;
    if anchor_rows.is_empty() {
        bail!("{} contains no anchors rows", a.store.display());
    }
    let mars1_rows = if a.mars1_store.is_file() {
        mars1_report::load(&a.mars1_store)?
    } else {
        eprintln!(
            "note: {} not found; the plot will show anchors only, no Mars 1 curves",
            a.mars1_store.display()
        );
        Vec::new()
    };

    let p = Provenance::detect(0);
    let build_infos: std::collections::BTreeSet<String> = anchor_rows
        .iter()
        .map(|r| format!("{}: {}", r.codec, r.codec_build_info))
        .collect();
    let note = format!(
        "{} anchor rows from `{}` (+ {} Mars 1 rows from `{}`) · report at {}{} · harness {} · {} ({} P / {} E cores) · {}\n\n\
         <details><summary>Anchor codec versions</summary>\n\n```\n{}\n```\n\n</details>",
        anchor_rows.len(),
        a.store.display(),
        mars1_rows.len(),
        a.mars1_store.display(),
        &p.git_sha[..p.git_sha.len().min(12)],
        if p.git_dirty { "-dirty" } else { "" },
        p.harness_version,
        p.machine.cpu_brand,
        p.machine.p_cores.map_or("?".into(), |c| c.to_string()),
        p.machine.e_cores.map_or("?".into(), |c| c.to_string()),
        p.timestamp_utc,
        build_infos.into_iter().collect::<Vec<_>>().join("\n"),
    );

    let curves = anchors_report::combined_plot_curves(&anchor_rows, &mars1_rows);
    let mut report = Report::new(
        "Mars 2 -- Kodak rate-distortion: five anchor codecs + six Mars 1 methods".to_string(),
        curves,
    )
    .with_provenance_note(note.clone());
    report = report.with_reference(a.reference.clone());

    let bd_matrix = anchors_report::bd_matrix(&anchor_rows, &a.reference);

    let mut md = report.to_markdown();
    md.push_str(&format!(
        "\n## Per-image BD-rate vs `{}` (anchor codecs only)\n\n\
         Computed per image over the PSNR overlap, then summarised across images -- \
         §M3/D7-style: a BD-rate between two *averaged* curves is a different, less \
         honest number. Nothing is dropped silently; exclusions are named (§A7).\n\n\
         | codec | n | mean % | median % | min % | max % | PSNR interval (dB) | excluded |\n\
         |---|---:|---:|---:|---:|---:|---|---|\n",
        a.reference
    ));
    for b in &bd_matrix {
        let excl = if b.excluded.is_empty() {
            "none".to_string()
        } else {
            b.excluded
                .iter()
                .map(|e| format!("{} ({})", e.image, e.reason))
                .collect::<Vec<_>>()
                .join("; ")
        };
        md.push_str(&format!(
            "| {} | {} | {:+.2} | {:+.2} | {:+.2} | {:+.2} | {:.2}-{:.2} | {} |\n",
            b.test,
            b.n,
            b.mean_pct,
            b.median_pct,
            b.min_pct,
            b.max_pct,
            b.psnr_interval_db.0,
            b.psnr_interval_db.1,
            excl,
        ));
    }

    match &a.markdown_out {
        Some(path) => {
            std::fs::write(path, &md)?;
            eprintln!("wrote {}", path.display());
        }
        None => print!("{md}"),
    }
    if let Some(path) = &a.html_out {
        std::fs::write(path, report.to_html())?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn anchors_check(a: AnchorsCheckArgs) -> Result<()> {
    let rows = anchors_report::load(&a.store)?;
    let envelopes = read_rows(&a.store)?;
    let unprovenanced = envelopes
        .iter()
        .filter(|r| {
            r.provenance.git_sha == "unknown"
                || r.provenance.corpus_manifest_sha256.is_none()
                || r.provenance.parameter_set_sha256.is_none()
        })
        .count();

    let expected_codecs: Vec<String> = mars_bench::anchors::AnchorCodec::ALL
        .iter()
        .map(|c| c.key().to_string())
        .collect();

    let report_files: Vec<(&str, &Path)> = vec![
        ("markdown report", a.markdown_out.as_path()),
        ("html report", a.html_out.as_path()),
    ];

    let mut checks = anchors_report::gate(&rows, &expected_codecs, &a.reference, &report_files);
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
        bail!("gate-4: {failed} of {} checks failed", checks.len());
    }
    println!("\ngate-4: PASS ({} checks)", checks.len());
    Ok(())
}

// -------------------------------------------------------------------- .ifs reader (Step 5)

fn ifs_check(a: IfsCheckArgs) -> Result<()> {
    let root = Path::new(".");
    let checks =
        mars_bench::ifs_check::gate(root, &a.manifest, &a.config, &a.mars1_dir, &a.scratch)?;

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
        bail!("gate-5: {failed} of {} checks failed", checks.len());
    }
    println!("\ngate-5: PASS ({} checks)", checks.len());
    Ok(())
}

// ----------------------------------------------------------------- exhaustive encoder (Step 6)

fn rust_encoder_check(a: RustEncoderCheckArgs) -> Result<()> {
    let root = Path::new(".");
    let (checks, divergence, evals_by_image) = mars_bench::rust_encoder::gate(
        root,
        &a.fixtures_index,
        &a.baseline_store,
        &a.mars1_dir,
        &a.scratch,
    )?;

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

    println!(
        "\nevals/transform (§M5), summed over rms = {:?}:",
        mars_bench::rust_encoder::RMS_GRID
    );
    let (mut total_evals, mut total_transforms) = (0u64, 0u64);
    for row in &evals_by_image {
        total_evals += row.evals;
        total_transforms += row.transforms;
        println!(
            "  {:<16} {:>12} evals  {:>6} transforms  {:>8.1} evals/transform",
            row.image,
            row.evals,
            row.transforms,
            row.evals_per_transform()
        );
    }
    if total_transforms > 0 {
        println!(
            "  {:<16} {:>12} evals  {:>6} transforms  {:>8.1} evals/transform",
            "(all fixtures)",
            total_evals,
            total_transforms,
            total_evals as f64 / total_transforms as f64
        );
    }

    println!(
        "\nf32-vs-f64 fit divergence: {:.4}% ({} of {} domain-referencing leaves picked a \
         different qalfa or qbeta; recorded, not gated — see docs/decisions.md)",
        divergence.pct(),
        divergence.differed,
        divergence.compared
    );
    if failed > 0 {
        bail!("gate-6: {failed} of {} checks failed", checks.len());
    }
    println!("\ngate-6: PASS ({} checks)", checks.len());
    Ok(())
}
