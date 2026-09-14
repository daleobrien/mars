//! `decmars` -- decompress a `.mars` bitstream back to an image, via the Step 5/6 fixed-
//! point iterative decoder, for visual before/after inspection. Handles both grayscale
//! and colour (Step 18) containers written by `encmars` -- see `mars_codec::color`.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use mars_codec::color::{decode_color_image, decode_color_image_auto};
use mars_core::io::{write_png, write_pnm};

/// Decompress a `.mars` bitstream to an image.
#[derive(Parser)]
struct Cli {
    /// Input `.mars` bitstream.
    input: PathBuf,
    /// Output image path. `.png` writes a viewable PNG; `.pgm`/`.ppm` writes a raw PNM
    /// (single-plane for grayscale input, P6 colour for RGB input).
    output: PathBuf,

    /// Fixed-point iterations to run from the flat grey (128) seed. With `--auto`, this is
    /// instead the maximum number of iterations before giving up on convergence.
    #[arg(short = 'i', long, default_value_t = 10)]
    iterations: u32,

    /// Iterate until the image stabilises instead of running a fixed count: stop once no
    /// pixel moves by more than `--threshold` between successive iterations (or
    /// `--iterations` is reached, whichever comes first).
    #[arg(short = 'a', long)]
    auto: bool,

    /// Max per-pixel change (0-255) between iterations to call the image stable. Only used
    /// with `--auto`.
    #[arg(long, default_value_t = 0)]
    threshold: u8,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let bytes =
        std::fs::read(&cli.input).with_context(|| format!("reading {}", cli.input.display()))?;
    let (image, iterations_used) = if cli.auto {
        let (image, used) = decode_color_image_auto(&bytes, cli.threshold, cli.iterations)
            .with_context(|| format!("parsing {} as a .mars container", cli.input.display()))?;
        (image, used)
    } else {
        let image = decode_color_image(&bytes, cli.iterations)
            .with_context(|| format!("parsing {} as a .mars container", cli.input.display()))?;
        (image, cli.iterations)
    };
    let (width, height) = (image.width(), image.height());

    let ext = cli
        .output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => write_png(&cli.output, &image),
        "pgm" | "ppm" => write_pnm(&cli.output, &image),
        _ => bail!(
            "{}: unsupported output extension; use .png, .pgm or .ppm",
            cli.output.display()
        ),
    }
    .with_context(|| format!("writing {}", cli.output.display()))?;

    println!(
        "{} -> {width}x{height} {} ({} iterations{}, {} plane(s))",
        cli.input.display(),
        cli.output.display(),
        iterations_used,
        if cli.auto { ", auto" } else { "" },
        image.planes().len(),
    );
    Ok(())
}
