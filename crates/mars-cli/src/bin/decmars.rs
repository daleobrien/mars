//! `decmars` -- decompress a `.mars` bitstream back to an image, via the Step 5/6 fixed-
//! point iterative decoder, for visual before/after inspection. Handles both grayscale
//! and colour (Step 18) containers written by `encmars` -- see `mars_codec::color`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use mars_codec::color::{
    decode_color_image_auto, decode_color_image_progression, decode_color_image_zoomed,
    quadtree_image,
};
use mars_codec::postprocess::smooth_boundaries;
use mars_core::image::Image;
use mars_core::io::{
    write_jpeg_with_quality, write_png, write_pnm, write_tga, write_tiff, write_webp, ImageError,
};

/// Decompress a `.mars` bitstream to an image.
#[derive(Parser)]
struct Cli {
    /// Input `.mars` bitstream.
    input: PathBuf,
    /// Output image path; the extension picks the format. `.png`, `.tga`, `.tif`/`.tiff`
    /// and `.webp` (lossless) write those formats; `.jpg`/`.jpeg` a lossy JPEG (inspection
    /// only -- it will not match the decode pixel-for-pixel); `.pgm`/`.ppm` a raw PNM
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

    /// Smooth leaf boundaries after decoding a regular grayscale stream. Strong edges
    /// are protected; progressive/color containers are refused. Iteration frames stay
    /// unsmoothed, and only the final output is filtered.
    #[arg(long)]
    smooth: bool,

    /// JPEG quality (1-100, higher is better) for a `.jpg`/`.jpeg` `output`; defaults to
    /// 75. Supplying it with a non-JPEG output is refused rather than silently ignored,
    /// and `--debug-rects` always uses the default quality regardless of this flag.
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=100))]
    quality: Option<u8>,
}

/// The `image` crate's own default JPEG quality, used when `--quality` is omitted.
const DEFAULT_JPEG_QUALITY: u8 = 75;

/// A writer for one output image format, selected from the output filename's extension by
/// `output_writer`. It is a boxed closure rather than a plain function pointer because the
/// JPEG writer carries a quality setting.
type ImageWriter = Box<dyn Fn(&Path, &Image) -> Result<(), ImageError>>;

/// `true` for the extensions `--quality` applies to.
fn is_jpeg(ext: &str) -> bool {
    matches!(ext, "jpg" | "jpeg")
}

/// Pick the writer for `ext`, carrying `quality` for JPEG output. A quality given with a
/// format that is not JPEG is refused by `main` before this is called, so it is never
/// silently dropped here.
fn output_writer(ext: &str, quality: u8) -> Result<ImageWriter> {
    match ext {
        "png" => Ok(Box::new(write_png)),
        "tga" => Ok(Box::new(write_tga)),
        "tif" | "tiff" => Ok(Box::new(write_tiff)),
        "webp" => Ok(Box::new(write_webp)),
        "pgm" | "ppm" => Ok(Box::new(write_pnm)),
        "jpg" | "jpeg" => Ok(Box::new(move |path, image| {
            write_jpeg_with_quality(path, image, quality)
        })),
        _ => bail!(
            "unsupported output extension {ext:?}; use .png, .jpg, .jpeg, .tga, .tif, .tiff, \
             .webp, .pgm or .ppm"
        ),
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
    if cli.quality.is_some() && !is_jpeg(&ext) {
        bail!(
            "--quality only applies to .jpg/.jpeg output, not {}",
            cli.output.display()
        );
    }
    let write = output_writer(&ext, cli.quality.unwrap_or(DEFAULT_JPEG_QUALITY))?;

    if mars_codec::progressive::is_progressive(&bytes) {
        if cli.smooth {
            bail!(
                "--smooth is not supported for progressive streams; use a regular grayscale container"
            );
        }
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

    let boundaries = cli
        .smooth
        .then(|| smoothing_leaves(&bytes, cli.zoom))
        .transpose()?;
    let (mut image, iterations_used) = if let Some(dir) = &cli.progression {
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
    if let Some(leaves) = boundaries {
        image = Image::gray(smooth_boundaries(&image.planes()[0], &leaves));
    }
    let (width, height) = (image.width(), image.height());

    if let Some(path) = &cli.debug_rects {
        let debug_ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let debug_write = output_writer(&debug_ext, DEFAULT_JPEG_QUALITY)?;
        let debug = quadtree_image(&bytes)
            .with_context(|| format!("building quadtree image from {}", cli.input.display()))?;
        debug_write(path, &debug).with_context(|| format!("writing {}", path.display()))?;
    }
    write(&cli.output, &image).with_context(|| format!("writing {}", cli.output.display()))?;

    println!(
        "{} -> {width}x{height} {} ({} iterations{}{}, {} plane(s)){}{}",
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
        if cli.smooth { ", smoothed" } else { "" },
    );
    Ok(())
}

// color::quadtree_image uses the same inner-stream parser, but its container reader
// is private. Keep this deliberately restricted to the known MARC v0 gray envelope;
// do not infer boundaries from a rendered debug image or accept unknown layouts.
fn smoothing_leaves(bytes: &[u8], zoom: f64) -> Result<Vec<(usize, usize, usize)>> {
    if bytes.get(..5) != Some(b"MARC\0") {
        bail!("--smooth cannot expose leaves: expected a MARC v0 grayscale container");
    }
    if matches!(bytes.get(5), Some(1 | 2)) {
        bail!("--smooth is grayscale only; color containers are not supported");
    }
    if bytes.get(5..7) != Some(&[0, 1]) {
        bail!("--smooth cannot expose leaves: expected one grayscale section");
    }
    let length: [u8; 4] = bytes
        .get(7..11)
        .context("--smooth: truncated grayscale section length")?
        .try_into()?;
    let end = 11usize
        .checked_add(u32::from_le_bytes(length) as usize)
        .context("--smooth: grayscale section length overflow")?;
    let stream = bytes
        .get(11..end)
        .context("--smooth: truncated grayscale section")?;
    if end != bytes.len() {
        bail!("--smooth cannot expose leaves: unexpected trailing container data");
    }
    let (header, leaves) =
        mars_codec::mars_format::read(stream).context("--smooth: parsing grayscale leaves")?;
    let (_, leaves) = mars_codec::ifs::zoom_leaves(&header, &leaves, zoom);
    Ok(leaves
        .into_iter()
        .map(|leaf| (leaf.row as usize, leaf.col as usize, leaf.size as usize))
        .collect())
}

/// CLI-E's own path: `--layer N` truncates `bytes` to that layer's own end offset
/// (`mars_codec::progressive::layer_end_offsets`) before handing it to `progressive::
/// decode_with_iterations`, which decodes "however many layers are fully present" -- the mechanism P19.1
/// exercises internally is exposed here exactly as-is, just driven by a CLI flag instead
/// of a test harness. `--auto`/`--progression`/`--zoom` have no progressive-stream
/// equivalent yet -- refused rather than
/// silently ignored.
fn decode_progressive(cli: &Cli, bytes: &[u8], write: &ImageWriter) -> Result<()> {
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
