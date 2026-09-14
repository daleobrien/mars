//! `decmars` -- decompress a `.mars` bitstream back to an image, via the Step 5/6 fixed-
//! point iterative decoder, for visual before/after inspection.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use mars_codec::ifs::decode_iterative;
use mars_codec::mars_format;
use mars_core::image::Image;
use mars_core::io::{write_png, write_pnm};

/// Decompress a `.mars` bitstream to an image.
#[derive(Parser)]
struct Cli {
    /// Input `.mars` bitstream.
    input: PathBuf,
    /// Output image path. `.png` writes a viewable PNG; `.pgm` writes a raw PNM (Mars 2
    /// is grayscale-only so far, so both come out single-plane).
    output: PathBuf,

    /// Fixed-point iterations to run from the flat grey (128) seed.
    #[arg(short = 'i', long, default_value_t = 10)]
    iterations: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let bytes = std::fs::read(&cli.input)
        .with_context(|| format!("reading {}", cli.input.display()))?;
    let (hdr, leaves) = mars_format::read(&bytes)
        .with_context(|| format!("parsing {} as a .mars container", cli.input.display()))?;

    let plane = decode_iterative(&hdr, &leaves, cli.iterations);
    let (width, height) = (plane.width(), plane.height());
    let image = Image::gray(plane);

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
            "{}: unsupported output extension; use .png or .pgm",
            cli.output.display()
        ),
    }
    .with_context(|| format!("writing {}", cli.output.display()))?;

    println!(
        "{} -> {width}x{height} {} ({} iterations)",
        cli.input.display(),
        cli.output.display(),
        cli.iterations,
    );
    Ok(())
}
