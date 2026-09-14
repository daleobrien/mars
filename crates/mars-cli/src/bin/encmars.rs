//! `encmars` -- compress an image into Mars 2's `.mars` container (Step 10 format), via
//! the Step 6 exhaustive Rust encoder. Colour input (Step 18) is encoded as independent
//! Y/Cb/Cr planes, each via the same single-plane encoder, wrapped in the small colour
//! container `mars_codec::color` defines -- see that module's doc for the design.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use mars_codec::color::{encode_color_image, wrap_gray_stream, ColorEncodeParams, Subsampling};
use mars_codec::encode::EncodeParams;
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

    /// Split threshold (Mars 1's `-r`) for the luma (or the only, for grayscale) plane: a
    /// block splits when its best-fit RMS exceeds this. Higher = fewer/larger blocks =
    /// more compression, lower quality. Superseded by `--lambda` (Step 14) as the
    /// recommended quality knob; kept for the legacy top-down split rule, used whenever
    /// `--lambda` is not given.
    #[arg(short = 'r', long, default_value_t = 8.0)]
    t_rms: f64,

    /// Split threshold for the Cb/Cr planes of a colour image (Step 18: independent
    /// per-plane quality control). Defaults to `--t-rms` if not given -- chroma very
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
    /// flag). Larger lambda = more weight on rate = fewer/larger blocks.
    #[arg(long)]
    lambda: Option<f64>,

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
    /// the RD path (`--lambda`); the legacy top-down split (`--t-rms` alone) never runs
    /// Step 15's mode competition, so this flag is a no-op without `--lambda`. Default
    /// (omitted): all four modes allowed, today's behaviour, byte-identical either way.
    #[arg(long, value_delimiter = ',')]
    modes: Vec<u8>,

    /// Run `mars-search`'s candidate-restriction search (Step 9's classical speed-ups
    /// plus Step 13's novel `funnel` method) instead of `mars-codec`'s own exhaustive
    /// walk -- a diagnostic/reproduction tool (isolating one method's candidate set,
    /// reproducing `marsbench`'s own evals/transforms numbers from the command line), not
    /// a quality control most users reach for; `exhaustive` here is a *different*
    /// implementation of the same idea as the default path, kept for cross-validation,
    /// not a faster or slower version of it. **Mutually exclusive with `--lambda`**:
    /// `mars-search::encode_image` only runs the legacy top-down `--t-rms` partition (the
    /// same one `mars-codec`'s own encoder uses when `--lambda` is omitted) -- it has no
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

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

    let base = EncodeParams {
        min_size: cli.min_size,
        max_size: cli.max_size,
        shift: cli.shift,
        bits_alfa: cli.bits_alfa,
        bits_beta: cli.bits_beta,
        max_alfa: cli.max_alfa,
        t_rms: cli.t_rms,
        zero_threshold: cli.zero_threshold,
        lambda: cli.lambda,
    };
    let chroma = EncodeParams {
        t_rms: cli.chroma_t_rms.unwrap_or(cli.t_rms),
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

    let params = ColorEncodeParams {
        y: base,
        chroma,
        subsampling: cli.subsampling.into(),
        adaptive_density: cli.adaptive_density,
        allowed_modes,
    };

    let (width, height) = (image.width(), image.height());
    let (bytes, stats) = encode_color_image(&image, &params);

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
        t_rms: cli.t_rms,
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
