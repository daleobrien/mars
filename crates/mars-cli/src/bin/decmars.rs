//! `decmars` -- decompress a `.mars` bitstream back to an image, via the Step 5/6 fixed-
//! point iterative decoder, for visual before/after inspection. Handles both grayscale
//! and colour (Step 18) containers written by `encmars` -- see `mars_codec::color`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;
use mars_codec::color::{
    decode_color_image_auto, decode_color_image_progression, decode_color_image_zoomed,
    quadtree_image,
};
use mars_core::image::Image;
use mars_core::io::{ImageError, write_png, write_pnm};

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
    #[arg(short = 'i', long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..))]
    iterations: u32,

    /// Iterate until the image stabilises instead of running a fixed count: stop once no
    /// pixel moves by more than `--threshold` between successive iterations (or
    /// `--iterations` is reached, whichever comes first).
    #[arg(short = 'a', long)]
    auto: bool,

    /// Max per-pixel change (0-255) between iterations to call the image stable. Only used
    /// with `--auto`.
    #[arg(long, default_value_t = 0, requires = "auto")]
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

    /// Decode only the first N of a progressive stream's 4 layers (1=base, 2=+partition
    /// refinement, 3=+fractal refinement, 4=+residual refinement -- `mars_codec::
    /// progressive`'s own doc has the full layer semantics), instead of every layer
    /// present. Only valid for a progressive `.mars` file (`encmars --progressive`);
    /// refused on a single-layer container. Default (omitted): decode every layer
    /// present, equivalent to today's output for a progressive file. `--auto`/
    /// `--progression`/`--zoom` are not supported together with a progressive input this
    /// first cut -- refused rather than silently ignored. `--iterations` controls the
    /// fixed-point reconstruction for every layer (default: 10).
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=4))]
    layer: Option<u8>,

    /// Also write a quadtree debug image: white background, black lines at every
    /// range-block boundary -- mirrors reference Mars 1's `encmars -Q` (`quadtree.pgm`).
    /// Drawn from the stream's luma/gray leaves, at native bitstream resolution
    /// (independent of `--zoom`). Extension picks the format, same as `output`. Not
    /// supported for a progressive `.mars` input.
    #[arg(long)]
    debug_rects: Option<PathBuf>,
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
    if !cli.zoom.is_finite() || cli.zoom <= 0.0 {
        bail!("--zoom must be finite and positive");
    }

    let bytes =
        std::fs::read(&cli.input).with_context(|| format!("reading {}", cli.input.display()))?;
    let ext = cli
        .output
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let write = image_writer(&ext)?;

    if mars_codec::progressive::is_progressive(&bytes) {
        if cli.debug_rects.is_some() {
            bail!(
                "{}: --debug-rects is not supported for a progressive stream yet",
                cli.input.display()
            );
        }

        return decode_progressive(&cli, &bytes, &write);
    }
    if cli.layer.is_some() {
        bail!(
            "{}: --layer only applies to a progressive stream (encmars --progressive) -- \
             this is a single-layer .mars container",
            cli.input.display()
        );
    }

    let (image, iterations_used) = if let Some(dir) = &cli.progression {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let width = cli.iterations.to_string().len();
        let stable_threshold = cli.auto.then_some(cli.threshold);
        let mut frame_error = None;
        let decoded = decode_color_image_progression(
            &bytes,
            cli.iterations,
            stable_threshold,
            cli.zoom,
            |n, frame| {
                // The codec callback cannot abort decoding. Preserve the first failure
                // and stop writing frames; report it before writing the final output.
                if frame_error.is_none() {
                    let frame_path = dir.join(format!("iter-{n:0width$}.{ext}"));
                    frame_error = write(&frame_path, frame)
                        .with_context(|| {
                            format!("writing progression frame {}", frame_path.display())
                        })
                        .err();
                }
            },
        )
        .with_context(|| format!("parsing {} as a .mars container", cli.input.display()))?;
        if let Some(error) = frame_error {
            return Err(error);
        }
        decoded
    } else if cli.auto {
        decode_color_image_auto(&bytes, cli.threshold, cli.iterations, cli.zoom)
            .with_context(|| format!("parsing {} as a .mars container", cli.input.display()))?
    } else {
        let image = decode_color_image_zoomed(&bytes, cli.iterations, cli.zoom)
            .with_context(|| format!("parsing {} as a .mars container", cli.input.display()))?;
        (image, cli.iterations)
    };
    let (width, height) = (image.width(), image.height());

    if let Some(path) = &cli.debug_rects {
        let debug_ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let debug_write = image_writer(&debug_ext)?;
        let debug = quadtree_image(&bytes)
            .with_context(|| format!("building quadtree image from {}", cli.input.display()))?;
        debug_write(path, &debug).with_context(|| format!("writing {}", path.display()))?;
    }
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

/// CLI-E's own path: `--layer N` truncates `bytes` to that layer's own end offset
/// (`mars_codec::progressive::layer_end_offsets`) before handing it to `progressive::
/// decode_with_iterations`, which decodes "however many layers are fully present" -- the mechanism P19.1
/// exercises internally is exposed here exactly as-is, just driven by a CLI flag instead
/// of a test harness. `--auto`/`--progression`/`--zoom` have no progressive-stream
/// equivalent yet -- refused rather than
/// silently ignored.
fn decode_progressive(
    cli: &Cli,
    bytes: &[u8],
    write: &fn(&Path, &Image) -> Result<(), ImageError>,
) -> Result<()> {
    if cli.auto || cli.progression.is_some() || cli.zoom != 1.0 {
        bail!(
            "{}: --auto/--progression/--zoom are not supported for a progressive stream yet \
             -- decmars --layer only",
            cli.input.display()
        );
    }

    let offsets = mars_codec::progressive::layer_end_offsets(bytes).with_context(|| {
        format!(
            "parsing {} as a progressive .mars stream",
            cli.input.display()
        )
    })?;
    let truncated = match cli.layer {
        Some(n) => bytes.get(..offsets[usize::from(n) - 1]).with_context(|| {
            format!(
                "{}: --layer {n} is unavailable: requested layer is not fully present",
                cli.input.display()
            )
        })?,
        None => bytes,
    };
    let decoded = mars_codec::progressive::decode_with_iterations(truncated, cli.iterations)
        .with_context(|| {
            format!(
                "parsing {} as a progressive .mars stream",
                cli.input.display()
            )
        })?;
    let image = Image::gray(decoded.image);
    let (width, height) = (image.width(), image.height());

    write(&cli.output, &image).with_context(|| format!("writing {}", cli.output.display()))?;

    println!(
        "{} -> {width}x{height} {} (progressive, layer {}/4 decoded, {} iterations, layer end-offsets {offsets:?})",
        cli.input.display(),
        cli.output.display(),
        decoded.layers,
        cli.iterations,
    );
    Ok(())
}
