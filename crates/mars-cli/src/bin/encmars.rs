//! `encmars` -- compress an image into Mars 2's `.mars` container (Step 10 format), via
//! the Step 6 exhaustive Rust encoder. Colour input (Step 18) is encoded as independent
//! Y/Cb/Cr planes, each via the same single-plane encoder, wrapped in the small colour
//! container `mars_codec::color` defines -- see that module's doc for the design.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use mars_codec::color::{encode_color_image, ColorEncodeParams, Subsampling};
use mars_codec::encode::EncodeParams;
use mars_core::io::read_image;

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
    let params = ColorEncodeParams {
        y: base,
        chroma,
        subsampling: cli.subsampling.into(),
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
