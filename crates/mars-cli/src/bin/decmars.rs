//! `decmars` -- decompress a `.mars` bitstream back to an image, via the Step 5/6 fixed-
//! point iterative decoder, for visual before/after inspection. Handles both grayscale
//! and colour (Step 18) containers written by `encmars` -- see `mars_codec::color`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use mars_codec::color::{
    decode_color_image_auto, decode_color_image_progression, decode_color_image_zoomed,
};
use mars_core::image::Image;
use mars_core::io::{write_png, write_pnm, ImageError};

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

    /// Also write one image per iteration into this directory (created if missing), named
    /// `iter-0001.<ext>`, `iter-0002.<ext>`, ... — for watching the fractal decode converge.
    /// Frames use the same format as `output`. Combine with `--auto` to stop the
    /// progression (and the final `output`) as soon as the image stabilises.
    #[arg(long)]
    progression: Option<PathBuf>,

    /// Decode at this multiple of the bitstream's encoded resolution instead of its native
    /// size (e.g. `2.0` decodes at double width/height). Fractal "zoom": the same
    /// contractive map is iterated over a larger or smaller canvas, synthesizing detail at
    /// a resolution the encoder never saw, rather than resampling a finished decode. `1.0`
    /// (the default) is the bitstream's native resolution.
    #[arg(short = 'z', long, default_value_t = 1.0)]
    zoom: f64,
}

fn image_writer(ext: &str) -> Result<fn(&Path, &Image) -> Result<(), ImageError>> {
    match ext {
        "png" => Ok(write_png),
        "pgm" | "ppm" => Ok(write_pnm),
        _ => bail!("unsupported output extension {ext:?}; use .png, .pgm or .ppm"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let bytes =
        std::fs::read(&cli.input).with_context(|| format!("reading {}", cli.input.display()))?;
    let ext = cli
        .output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let write = image_writer(&ext)?;

    let (image, iterations_used) = if let Some(dir) = &cli.progression {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating {}", dir.display()))?;
        let width = cli.iterations.max(1).to_string().len();
        let stable_threshold = cli.auto.then_some(cli.threshold);
        decode_color_image_progression(
            &bytes,
            cli.iterations,
            stable_threshold,
            cli.zoom,
            |n, frame| {
                let frame_path = dir.join(format!("iter-{n:0width$}.{ext}"));
                if let Err(e) = write(&frame_path, frame) {
                    eprintln!("warning: failed to write {}: {e:#}", frame_path.display());
                }
            },
        )
        .with_context(|| format!("parsing {} as a .mars container", cli.input.display()))?
    } else if cli.auto {
        decode_color_image_auto(&bytes, cli.threshold, cli.iterations, cli.zoom)
            .with_context(|| format!("parsing {} as a .mars container", cli.input.display()))?
    } else {
        let image = decode_color_image_zoomed(&bytes, cli.iterations, cli.zoom)
            .with_context(|| format!("parsing {} as a .mars container", cli.input.display()))?;
        (image, cli.iterations)
    };
    let (width, height) = (image.width(), image.height());

    write(&cli.output, &image).with_context(|| format!("writing {}", cli.output.display()))?;

    println!(
        "{} -> {width}x{height} {} ({} iterations{}{}, {} plane(s)){}",
        cli.input.display(),
        cli.output.display(),
        iterations_used,
        if cli.auto { ", auto" } else { "" },
        if cli.zoom != 1.0 {
            format!(", {}x zoom", cli.zoom)
        } else {
            String::new()
        },
        image.planes().len(),
        cli.progression
            .as_ref()
            .map(|d| format!(", {iterations_used} frame(s) in {}", d.display()))
            .unwrap_or_default(),
    );
    Ok(())
}
