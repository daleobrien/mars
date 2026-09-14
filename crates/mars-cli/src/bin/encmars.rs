//! `encmars` -- compress an image into Mars 2's `.mars` container (Step 10 format), via
//! the Step 6 exhaustive Rust encoder.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use mars_codec::encode::{encode_image, EncodeParams};
use mars_codec::mars_format;
use mars_core::io::read_image;

/// Compress an image into a `.mars` bitstream.
#[derive(Parser)]
struct Cli {
    /// Input image: .png, .pgm, .ppm, or headerless .raw/.y/.gray (needs --raw-width/--raw-height).
    input: PathBuf,
    /// Output `.mars` bitstream path.
    output: PathBuf,

    /// Split threshold (Mars 1's `-r`): a block splits when its best-fit RMS exceeds
    /// this. Higher = fewer/larger blocks = more compression, lower quality. Superseded
    /// by `--lambda` (Step 14) as the recommended quality knob; kept for the legacy
    /// top-down split rule, used whenever `--lambda` is not given.
    #[arg(short = 'r', long, default_value_t = 8.0)]
    t_rms: f64,

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
    if image.planes().len() != 1 {
        bail!(
            "{}: colour input has {} planes, but the encoder only fills the luma plane \
             so far (colour arrives in Step 18) -- pass a grayscale image",
            cli.input.display(),
            image.planes().len()
        );
    }

    let params = EncodeParams {
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

    let (width, height) = (image.width(), image.height());
    let (hdr, leaves, evals) = encode_image(&image.planes()[0], &params);
    let bytes = mars_format::write(&hdr, &leaves)
        .context("serialising the encoded image to the .mars container")?;

    std::fs::write(&cli.output, &bytes)
        .with_context(|| format!("writing {}", cli.output.display()))?;

    let bpp = 8.0 * bytes.len() as f64 / (width * height) as f64;
    println!(
        "{width}x{height} -> {} ({} bytes, {bpp:.3} bpp, {} leaves, {evals} evals)",
        cli.output.display(),
        bytes.len(),
        leaves.len(),
    );
    Ok(())
}
