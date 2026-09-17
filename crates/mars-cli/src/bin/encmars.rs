//! `encmars` -- compress an image into Mars 2's `.mars` container (Step 10 format), via
//! the Step 6 exhaustive Rust encoder. Colour input (Step 18) is encoded as independent
//! Y/Cb/Cr planes, each via the same single-plane encoder, wrapped in the small colour
//! container `mars_codec::color` defines -- see that module's doc for the design.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use mars_codec::color::{encode_color_image_with_residual_quantisation, wrap_gray_stream, ColorEncodeParams, Subsampling};
use mars_codec::encode::{EncodeOptions, EncodeParams, ResidualQuantisation};
use mars_core::io::read_image;

/// `mars-search`'s nine candidate-restriction methods (Step 9's six classical ports plus
/// `Exhaustive`, and Step 13's `Funnel`), named to match `mars_search::MethodName::key()`
/// exactly so a script can join a `--method` run against `marsbench`'s own output by the
/// same string. **Not** the same code path as `mars-codec`'s own exhaustive walk --
/// `mars-codec::encode`'s own search (what every other `encmars` invocation runs) is a
/// distinct, independently-implemented exhaustive scan kept for cross-validation, not an
/// instance of `mars_search::MethodName::Exhaustive`. `--help` says which is which so a
/// reader is not left to guess from the shared name.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum MethodArg {
    Exhaustive,
    Fisher,
    Hurtgen,
    Masscenter,
    Saupe,
    #[value(name = "saupe-fisher")]
    SaupeFisher,
    #[value(name = "mc-saupe")]
    McSaupe,
    Funnel,
    Learned,
}

impl From<MethodArg> for mars_search::MethodName {
    fn from(m: MethodArg) -> Self {
        match m {
            MethodArg::Exhaustive => mars_search::MethodName::Exhaustive,
            MethodArg::Fisher => mars_search::MethodName::Fisher,
            MethodArg::Hurtgen => mars_search::MethodName::Hurtgen,
            MethodArg::Masscenter => mars_search::MethodName::MassCenter,
            MethodArg::Saupe => mars_search::MethodName::Saupe,
            MethodArg::SaupeFisher => mars_search::MethodName::SaupeFisher,
            MethodArg::McSaupe => mars_search::MethodName::McSaupe,
            MethodArg::Funnel => mars_search::MethodName::Funnel,
            MethodArg::Learned => mars_search::MethodName::Learned,
        }
    }
}

/// Chroma subsampling mode (Step 18); ignored for grayscale input.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum SubsamplingArg {
    #[value(name = "444")]
    Yuv444,
    #[value(name = "420")]
    Yuv420,
}

impl From<SubsamplingArg> for Subsampling {
    fn from(v: SubsamplingArg) -> Self {
        match v {
            SubsamplingArg::Yuv444 => Subsampling::Yuv444,
            SubsamplingArg::Yuv420 => Subsampling::Yuv420,
        }
    }
}

/// Compress an image into a `.mars` bitstream.
#[derive(Parser)]
struct Cli {
    /// Input image: .png, .pgm, .ppm, or headerless .raw/.y/.gray (needs --raw-width/--raw-height).
    input: PathBuf,
    /// Output `.mars` bitstream path.
    output: PathBuf,

    /// Select legacy threshold partitioning: split blocks whose best-fit RMS exceeds
    /// this value. Higher = smaller files/lower quality. Without a threshold or --method,
    /// the default is RD at lambda=200. With explicit --lambda, this instead sets the
    /// rate-estimation warm-up threshold. The implicit legacy/warm-up threshold is 8.
    #[arg(short = 'r', long)]
    t_rms: Option<f64>,

    /// Split threshold for the Cb/Cr planes of a colour image (Step 18: independent
    /// per-plane quality control). Like --t-rms, selects legacy partitioning unless
    /// --lambda is explicitly supplied. Defaults to the luma threshold (8 if omitted).
    /// Chroma very
    /// commonly wants a looser (higher) threshold than luma, but the encoder never picks
    /// that automatically; the caller sets it explicitly.
    #[arg(long)]
    chroma_t_rms: Option<f64>,

    /// Chroma subsampling for colour input: `444` (no subsampling) or `420`
    /// (half-resolution Cb/Cr, box-filtered). Ignored for grayscale input.
    #[arg(long, value_enum, default_value = "444")]
    subsampling: SubsamplingArg,

    /// Step 14's `J = D + lambda*R` quality knob: the encoder searches every candidate
    /// block bottom-up and keeps whichever of "one leaf here" or "the four children" has
    /// the smaller distortion-plus-lambda-times-estimated-bits. This is the RD-optimal
    /// replacement for `--t-rms`/`-r`'s top-down threshold -- an RD curve is a lambda
    /// sweep, not a t_rms sweep. When given, `--t-rms` is ignored for the split decision
    /// (it still seeds the internal rate-estimation warm-up pass, unaffected by this
    /// flag). Larger lambda = more weight on rate = fewer/larger blocks. Defaults to 200
    /// unless --t-rms, --chroma-t-rms or --method selects the legacy path.
    #[arg(long)]
    lambda: Option<f64>,

    /// Experimental lambda-derived residual step. Fixed step 8 remains the default:
    /// the initial Kodak measurement regressed. Requires --lambda; stored in the stream
    /// so decmars needs no matching option. Enable residual mode too, e.g. --modes 0,2,3.
    /// Applies to grayscale, colour and progressive.
    #[arg(long, requires = "lambda", conflicts_with = "method")]
    adaptive_residual: bool,

    /// Smallest range-block size.
    #[arg(long, default_value_t = 4)]
    min_size: u32,
    /// Largest range-block size.
    #[arg(long, default_value_t = 16)]
    max_size: u32,
    /// Domain-block downsampling shift.
    #[arg(long, default_value_t = 4)]
    shift: u32,
    /// Bits for the quantised contrast (alfa) coefficient.
    #[arg(long, default_value_t = 4)]
    bits_alfa: u32,
    /// Bits for the quantised brightness (beta) coefficient.
    #[arg(long, default_value_t = 7)]
    bits_beta: u32,
    /// Maximum contrast magnitude before quantisation.
    #[arg(long, default_value_t = 1.0)]
    max_alfa: f64,
    /// `-z`: qalfa threshold for the zero-alfa (flat block) override.
    #[arg(long, default_value_t = 0)]
    zero_threshold: u32,

    /// Width for headerless raw input.
    #[arg(long)]
    raw_width: Option<usize>,
    /// Height for headerless raw input.
    #[arg(long)]
    raw_height: Option<usize>,

    /// Step 16's content-adaptive domain-pool density: search a sparser domain-pool
    /// stride where a block's local pixel-domain RMS is low (near-flat), instead of
    /// always using the run's fixed `--shift`. An eval/wall-clock optimisation, not a
    /// quality knob: measured essentially BD-rate-neutral (mean -0.14% on kodim01/
    /// kodim02) but genuinely faster (~0.87x encode time) -- `docs/decisions.md` D48,
    /// which also documents why an earlier "denser where RMS is high" branch was removed
    /// (it could corrupt the bitstream; D43's originally-claimed -6.82% BD-rate number is
    /// withdrawn, see D48). Default off, matching every `encmars` invocation before this
    /// flag existed -- byte-identical output either way when omitted.
    #[arg(long, default_value_t = false)]
    adaptive_density: bool,

    /// Step 15's per-leaf mode mask, a diagnostic/comparison knob: comma-separated mode
    /// numbers to allow, from 0 (flat), 1 (affine), 2 (fractal), 3 (fractal + residual) --
    /// e.g. `--modes 0,1,2` disables mode 3, `--modes 2` forces fractal-only. Only affects
    /// the RD path (the default); explicit --t-rms without --lambda uses the legacy
    /// split and ignores the mask. Default: 0,2 (flat and fractal), the better measured
    /// combination on kodim01/02. Use --modes 0,1,2,3 to enable all four modes.
    #[arg(long, value_delimiter = ',', default_value = "0,2")]
    modes: Vec<u8>,

    /// Run `mars-search`'s candidate-restriction search (Step 9's classical speed-ups
    /// plus Step 13's novel `funnel` method) instead of `mars-codec`'s own exhaustive
    /// walk -- a diagnostic/reproduction tool (isolating one method's candidate set,
    /// reproducing `marsbench`'s own evals/transforms numbers from the command line), not
    /// a quality control most users reach for; `exhaustive` here is a *different*
    /// implementation of the same idea as the default path, kept for cross-validation,
    /// not a faster or slower version of it. **Mutually exclusive with `--lambda`**:
    /// `mars-search::encode_image` only runs the legacy top-down `--t-rms` partition (the
    /// same one selected by explicit --t-rms without --lambda) -- it has no
    /// RD-pruning implementation of its own, so `--method X --lambda Y` would silently
    /// ignore one of the two; this is refused explicitly rather than left as a silent
    /// interaction for a user to discover. **Grayscale input only** for now -- colour's
    /// per-plane YCbCr/subsampling wrapping (`mars_codec::color`) is not wired to this
    /// path; given a colour image with `--method` set, `encmars` refuses rather than
    /// silently encoding only the luma plane. `learned`'s weights are baked into the
    /// `mars-search` binary already (Step 17); no separate training step is needed to use
    /// it here.
    #[arg(long, value_enum)]
    method: Option<MethodArg>,

    /// Pin Rayon's global thread pool to this many threads instead of Rayon's own
    /// default (`std::thread::available_parallelism()`). Matches `marsbench`'s own
    /// `rayon::ThreadPoolBuilder` usage exactly, so the two binaries' thread-count
    /// semantics don't silently diverge. Must be set before any parallel work runs
    /// (Rayon lazily builds an unpinned default pool on first use otherwise), so this is
    /// applied as the very first thing `main` does. Determinism (Step 12/14's own
    /// discipline): every encode path in this crate is bit-identical across thread
    /// counts by construction, so this flag exists for controlling wall-clock/CPU usage,
    /// never for reproducing a different result.
    #[arg(long)]
    threads: Option<usize>,

    /// Write Step 19's 4-layer progressive container (base -> partition refinement ->
    /// fractal refinement -> residual refinement) instead of the single-layer `.mars` v0
    /// format -- so a `decmars --layer N` can decode any prefix independently, the whole
    /// point of a progressive stream. **Grayscale only** for this first cut: whether/how
    /// a progressive container composes with `mars_codec::color`'s per-plane YCbCr/
    /// subsampling wrapping (`progressive.rs` was not designed against it) is unresolved
    /// and explicitly out of scope here -- refused on colour input rather than silently
    /// encoding only the luma plane. Mutually exclusive with `--method` (the progressive
    /// encoder takes an already-built `(Header, Vec<Leaf>)`, which `--progressive` gets
    /// from the same RD/legacy partition every other `encmars` invocation uses --
    /// composing it with `mars-search`'s own candidate-restricted partition is unattempted
    /// and not wired). Composes with `--lambda`/`--t-rms`/`--adaptive-density`/`--modes`
    /// normally -- those choose the partition; `--progressive` only changes how that same
    /// partition is serialised.
    #[arg(long)]
    progressive: bool,
}

impl Cli {
    fn effective_lambda(&self) -> Option<f64> {
        // Only an explicitly selected legacy path suppresses the RD default. Keeping
        // the parsed lambda optional preserves --method conflicts and warm-up overrides.
        self.lambda.or_else(|| {
            (self.t_rms.is_none() && self.chroma_t_rms.is_none() && self.method.is_none())
                .then_some(200.0)
        })
    }

    fn effective_t_rms(&self) -> f64 {
        self.t_rms.unwrap_or(8.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(["encmars", "input.png", "output.mars"].into_iter().chain(args.iter().copied()))
            .expect("valid arguments")
    }

    #[test]
    fn defaults_select_rd_and_measured_modes_without_other_experiments() {
        let cli = parse(&[]);
        assert_eq!(cli.effective_lambda(), Some(200.0));
        assert_eq!(cli.effective_t_rms(), 8.0);
        assert_eq!(cli.modes, [0, 2]);
        assert!(matches!(cli.subsampling, SubsamplingArg::Yuv444));
        assert!(!cli.adaptive_density && !cli.adaptive_residual && !cli.progressive);
    }

    #[test]
    fn explicit_thresholds_and_methods_keep_the_legacy_path() {
        for args in [vec!["--t-rms", "8"], vec!["-r", "4"],
            vec!["--chroma-t-rms", "16"], vec!["--method", "fisher"]] {
            assert_eq!(parse(&args).effective_lambda(), None);
        }
        assert_eq!(parse(&["-r", "4"]).effective_t_rms(), 4.0);
    }

    #[test]
    fn explicit_lambda_overrides_threshold_and_modes_replace_defaults() {
        let cli = parse(&["--lambda", "50", "--t-rms", "0", "--chroma-t-rms", "16",
            "--modes", "0,1,2,3"]);
        assert_eq!(cli.effective_lambda(), Some(50.0));
        assert_eq!(cli.effective_t_rms(), 0.0);
        assert_eq!(cli.modes, [0, 1, 2, 3]);
        assert_eq!(parse(&["--lambda", "0"]).effective_lambda(), Some(0.0));
        assert_eq!(parse(&["--modes", "3"]).modes, [3]);
        assert_eq!(parse(&["--progressive"]).effective_lambda(), Some(200.0));
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.lambda.is_some_and(|value| !value.is_finite() || value < 0.0) {
        bail!("--lambda must be finite and nonnegative");
    }

    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("pinning Rayon's global thread pool to --threads")?;
    }

    let raw_dims = match (cli.raw_width, cli.raw_height) {
        (Some(w), Some(h)) => Some((w, h)),
        (None, None) => None,
        _ => bail!("--raw-width and --raw-height must be given together"),
    };
    let image = read_image(&cli.input, raw_dims)
        .with_context(|| format!("reading {}", cli.input.display()))?;
    if image.planes().len() != 1 && image.planes().len() != 3 {
        bail!(
            "{}: expected 1 (gray) or 3 (RGB) planes, got {}",
            cli.input.display(),
            image.planes().len()
        );
    }

    if cli.progressive && cli.method.is_some() {
        bail!(
            "--progressive and --method are mutually exclusive: --progressive serialises \
             the same RD/legacy partition every other encmars invocation produces, not a \
             mars-search-restricted one -- see --help for --progressive"
        );
    }

    if let Some(method_arg) = cli.method {
        if cli.lambda.is_some() {
            bail!(
                "--method and --lambda are mutually exclusive: mars-search's methods only \
                 run mars-codec's legacy top-down --t-rms partition, which has no RD \
                 pruning of its own -- see --help for --method"
            );
        }
        if image.planes().len() != 1 {
            bail!(
                "{}: --method only supports grayscale input for now (got {} planes) -- \
                 see --help for --method",
                cli.input.display(),
                image.planes().len()
            );
        }
        return run_with_method(&cli, method_arg, &image);
    }

    if cli.progressive && image.planes().len() != 1 {
        bail!(
            "{}: --progressive only supports grayscale input for now (got {} planes) -- \
             see --help for --progressive",
            cli.input.display(),
            image.planes().len()
        );
    }

    let base = EncodeParams {
        min_size: cli.min_size,
        max_size: cli.max_size,
        shift: cli.shift,
        bits_alfa: cli.bits_alfa,
        bits_beta: cli.bits_beta,
        max_alfa: cli.max_alfa,
        t_rms: cli.effective_t_rms(),
        zero_threshold: cli.zero_threshold,
        lambda: cli.effective_lambda(),
    };
    let chroma = EncodeParams {
        t_rms: cli.chroma_t_rms.unwrap_or(cli.effective_t_rms()),
        ..base
    };
    let mut allowed_modes = [true; 4];
    if !cli.modes.is_empty() {
        allowed_modes = [false; 4];
        for &m in &cli.modes {
            let idx = usize::from(m);
            match allowed_modes.get_mut(idx) {
                Some(slot) => *slot = true,
                None => bail!("--modes: {m} is not a valid mode (expected 0-3)"),
            }
        }
    }

    if cli.progressive {
        return run_progressive(&cli, &base, allowed_modes, &image);
    }

    let params = ColorEncodeParams {
        y: base,
        chroma,
        subsampling: cli.subsampling.into(),
        adaptive_density: cli.adaptive_density,
        allowed_modes,
    };

    let (width, height) = (image.width(), image.height());
    let (bytes, stats) = encode_color_image_with_residual_quantisation(
        &image, &params, residual_policy(&cli),
    );

    std::fs::write(&cli.output, &bytes)
        .with_context(|| format!("writing {}", cli.output.display()))?;

    let bpp = 8.0 * bytes.len() as f64 / (width * height) as f64;
    if image.planes().len() == 1 {
        println!(
            "{width}x{height} gray -> {} ({} bytes, {bpp:.3} bpp, {} evals)",
            cli.output.display(),
            bytes.len(),
            stats.y_evals,
        );
    } else {
        println!(
            "{width}x{height} rgb ({:?}) -> {} ({} bytes total: Y {} + Cb {} + Cr {}, \
             {bpp:.3} bpp, evals Y {} / Cb {} / Cr {})",
            cli.subsampling,
            cli.output.display(),
            bytes.len(),
            stats.y_bytes,
            stats.cb_bytes,
            stats.cr_bytes,
            stats.y_evals,
            stats.cb_evals,
            stats.cr_evals,
        );
    }
    Ok(())
}

/// CLI-C's own path: `mars_search::encode_image` instead of `mars_codec::encode`'s own
/// walk. Grayscale-only (validated by the caller), legacy top-down `--t-rms` partition
/// only (`--lambda` validated absent by the caller) -- `mars-search`'s `encode_image` has
/// no RD-pruning implementation to switch to. Writes the same `MARC` single-plane
/// container every other grayscale `encmars` invocation does
/// (`mars_codec::color::wrap_gray_stream`), so `decmars` reads it identically either way.
fn run_with_method(
    cli: &Cli,
    method_arg: MethodArg,
    image: &mars_core::image::Image,
) -> Result<()> {
    let method: mars_search::MethodName = method_arg.into();
    let plane = &image.planes()[0];
    let params = EncodeParams {
        min_size: cli.min_size,
        max_size: cli.max_size,
        shift: cli.shift,
        bits_alfa: cli.bits_alfa,
        bits_beta: cli.bits_beta,
        max_alfa: cli.max_alfa,
        t_rms: cli.effective_t_rms(),
        zero_threshold: cli.zero_threshold,
        lambda: None,
    };

    let contracted = mars_codec::encode::build_contracted(plane);
    let retrievers = mars_search::SizedRetrievers::build(
        &contracted,
        plane.width() as u32,
        plane.height() as u32,
        params.shift,
        params.min_size,
        params.max_size,
        || method.new_retriever(),
    );
    let (hdr, leaves, evals, _picks) = mars_search::encode_image(plane, &params, &retrievers);
    let transforms = leaves.len() as u64;

    let plane_bytes = mars_codec::mars_format::write(&hdr, &leaves)
        .context("mars-search's encode_image produced a header mars_format::write rejected")?;
    let bytes = wrap_gray_stream(plane_bytes);

    std::fs::write(&cli.output, &bytes)
        .with_context(|| format!("writing {}", cli.output.display()))?;

    let (width, height) = (image.width(), image.height());
    let bpp = 8.0 * bytes.len() as f64 / (width * height) as f64;
    println!(
        "{width}x{height} gray -> {} ({} bytes, {bpp:.3} bpp, method {}, {evals} evals, \
         {transforms} transforms, {:.2} evals/transform)",
        cli.output.display(),
        bytes.len(),
        method.key(),
        evals as f64 / transforms.max(1) as f64,
    );
    Ok(())
}

fn residual_policy(cli: &Cli) -> ResidualQuantisation {
    if cli.adaptive_residual {
        ResidualQuantisation::LambdaAdaptive
    } else {
        ResidualQuantisation::default()
    }
}

/// Encode the same RD/legacy partition, including the selected residual policy,
/// into the progressive container instead of the single-layer representation.
fn run_progressive(
    cli: &Cli,
    base: &EncodeParams,
    allowed_modes: [bool; 4],
    image: &mars_core::image::Image,
) -> Result<()> {
    let plane = &image.planes()[0];
    let (hdr, leaves, evals, _stats) = mars_codec::encode::encode_image_with_options(
        plane,
        base,
        &EncodeOptions {
            allowed_modes,
            adaptive_density: cli.adaptive_density,
            residual_quantisation: residual_policy(cli),
        },
    );
    let bytes = mars_codec::progressive::encode(plane, &hdr, &leaves)
        .context("building the progressive container")?;

    std::fs::write(&cli.output, &bytes)
        .with_context(|| format!("writing {}", cli.output.display()))?;

    let offsets = mars_codec::progressive::layer_end_offsets(&bytes)
        .expect("bytes this call just produced are always a valid progressive stream");
    let (width, height) = (image.width(), image.height());
    let bpp = 8.0 * bytes.len() as f64 / (width * height) as f64;
    println!(
        "{width}x{height} gray -> {} ({} bytes, {bpp:.3} bpp, progressive, {evals} evals, \
         {} leaves, layer end-offsets {offsets:?})",
        cli.output.display(),
        bytes.len(),
        leaves.len(),
    );
    Ok(())
}
