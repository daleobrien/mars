//! Colour encode/decode plumbing — Step 18.
//!
//! **Entry-condition note (see `docs/decisions.md`).** Step 18's own brief text says
//! "Entry condition: Gate D passed", but §5's dependency graph and "what may run
//! concurrently" table both place Step 18 right after Gate B, concurrent with Steps
//! 10/11/13/14. This module was built under the latter reading, at the user's explicit
//! request, alongside a concurrently-developed Step 14. Everything here that does not
//! depend on Gate D's residual/adaptive-partitioning machinery is built and validated
//! now; the brief's own exit criterion (an apples-to-apples BD-rate against the anchors)
//! implicitly assumes the Gate-D-complete codec, so any BD-rate numbers produced against
//! *today's* pre-Gate-D encoder are provisional and will need re-measurement once Gate D
//! actually passes.
//!
//! **Design.** Rather than teach the shared search/partition core
//! ([`crate::encode::encode_image`]) about multiple planes, colour is layered on top: a
//! colour image is converted to BT.601 YCbCr ([`mars_core::metrics::ycbcr`]), chroma is
//! optionally subsampled, and each of Y/Cb/Cr is encoded independently by calling
//! `encode_image` three times with three independent [`EncodeParams`] — chroma very
//! commonly wants a looser quality setting than luma, hence "independent quality control
//! per plane" being a caller-supplied pair of params, never something the encoder infers.
//! Decode does the mirror image: three independent [`crate::ifs::decode_iterative`] calls,
//! chroma upsampled back to luma resolution, then the inverse YCbCr transform
//! ([`mars_core::metrics::rgb_from_ycbcr`]).
//!
//! **What this deliberately does not do.** No shared/optimiser-decided λ allocation
//! across planes — the brief calls that "eventually", i.e. explicitly out of scope for
//! this step. No residual layer, no adaptive partitioning (both Gate-D machinery, not
//! built yet). No progressive/prefix decoding (Step 19).

use mars_core::image::{ColorSpace, Image};
use mars_core::metrics::{rgb_from_ycbcr, ycbcr};
use mars_core::Plane;

use crate::encode::{
    encode_image_with_options, EncodeOptions, EncodeParams, LambdaRegion, ResidualQuantisation,
};
use crate::ifs::{
    decode_iterative, decode_step, decode_until_stable, max_pixel_delta, zoom_leaves,
};
use crate::mars_format::{self, MarsFormatError};

// ---------------------------------------------------------------------------
// Chroma subsampling
// ---------------------------------------------------------------------------

/// 4:4:4 (no subsampling) or 4:2:0 (half-resolution Cb/Cr) — Step 18 brief's minimum
/// pair. JPEG/video's usual third option, 4:2:2, is not implemented: nothing in the
/// brief requires it and adding it now would be scope the step does not ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    Yuv444,
    Yuv420,
}

/// Downsample one chroma plane 2x in both axes with a box filter (simple 2x2 pixel
/// averaging, rounded half-away-from-zero) — the standard, simplest-correct choice
/// (documented in `docs/decisions.md`, not left implicit). Odd dimensions replicate the
/// last row/column so every output pixel still averages a full 2x2 neighbourhood; this is
/// the same edge convention libjpeg's own chroma subsampler uses.
pub fn downsample_box(plane: &Plane) -> Plane {
    let (w, h) = (plane.width(), plane.height());
    let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
    let mut out = vec![0u8; ow * oh];
    for oy in 0..oh {
        for ox in 0..ow {
            let x0 = 2 * ox;
            let y0 = 2 * oy;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let sum = u32::from(plane.get(x0, y0))
                + u32::from(plane.get(x1, y0))
                + u32::from(plane.get(x0, y1))
                + u32::from(plane.get(x1, y1));
            out[oy * ow + ox] = ((sum + 2) / 4) as u8;
        }
    }
    Plane::from_vec(ow, oh, out)
}

/// Upsample one chroma plane back to `(width, height)` by nearest-neighbour pixel
/// replication — the exact inverse operation of [`downsample_box`]'s 2x2 grouping (each
/// output pixel takes its enclosing downsampled sample), and deterministic/branch-free
/// unlike bilinear, which is why it was chosen over a smoother filter (see
/// `docs/decisions.md`).
pub fn upsample_nearest(plane: &Plane, width: usize, height: usize) -> Plane {
    let mut out = vec![0u8; width * height];
    for y in 0..height {
        let sy = (y / 2).min(plane.height() - 1);
        for x in 0..width {
            let sx = (x / 2).min(plane.width() - 1);
            out[y * width + x] = plane.get(sx, sy);
        }
    }
    Plane::from_vec(width, height, out)
}

// ---------------------------------------------------------------------------
// Region-limited colour (chroma masking)
// ---------------------------------------------------------------------------

/// BT.601's neutral chroma value: `Cb == Cr == 128` inverts to `R == G == B`, i.e. a
/// grayscale pixel. [`mars_core::metrics::ycbcr`] emits exactly this for gray input.
const NEUTRAL_CHROMA: u8 = 128;

/// Force `cb` and `cr` to neutral outside every rectangle in `regions`, keeping the colour
/// inside them. This is the "colour only inside the face boxes, grayscale elsewhere"
/// transform: the neutral area compresses to almost nothing, and because `Cb == Cr == 128`
/// inverts to `R == G == B`, [`decode_color_image`] reconstructs those pixels as grayscale
/// with no format or decoder change at all.
///
/// The rectangles are in luma pixel coordinates, which are the chroma planes' coordinates
/// at this point too: the caller masks *before* any 4:2:0 subsampling, so a neutralised area
/// stays neutral through the box filter. A pixel covered by at least one region keeps its
/// colour, so overlapping or adjacent boxes compose by union. An empty list is a no-op,
/// leaving the planes exactly as `ycbcr` produced them.
fn neutralise_chroma_outside(cb: &mut Plane, cr: &mut Plane, regions: &[LambdaRegion]) {
    if regions.is_empty() {
        return;
    }
    let (width, height) = (cb.width(), cb.height());
    let colour_cb = cb.as_slice().to_vec();
    let colour_cr = cr.as_slice().to_vec();
    let cb_out = cb.as_mut_slice();
    let cr_out = cr.as_mut_slice();
    cb_out.fill(NEUTRAL_CHROMA);
    cr_out.fill(NEUTRAL_CHROMA);
    for region in regions {
        let x0 = (region.col as usize).min(width);
        let y0 = (region.row as usize).min(height);
        let x1 = (region.col as usize)
            .saturating_add(region.width as usize)
            .min(width);
        let y1 = (region.row as usize)
            .saturating_add(region.height as usize)
            .min(height);
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        for y in y0..y1 {
            let start = y * width + x0;
            let end = y * width + x1;
            cb_out[start..end].copy_from_slice(&colour_cb[start..end]);
            cr_out[start..end].copy_from_slice(&colour_cr[start..end]);
        }
    }
}

// ---------------------------------------------------------------------------
// Independent per-plane quality control
// ---------------------------------------------------------------------------

/// Encode parameters for a colour image: one [`EncodeParams`] for Y, one shared by both
/// Cb and Cr (chroma is very commonly coded at a single, looser quality than luma; this
/// project has no per-Cb/per-Cr use case yet, so one set of params for "chroma" rather
/// than two independent ones is the least-invasive choice covering the brief's actual
/// ask — see `docs/decisions.md`).
#[derive(Debug, Clone)]
pub struct ColorEncodeParams {
    pub y: EncodeParams,
    pub chroma: EncodeParams,
    pub subsampling: Subsampling,
    /// Step 16's content-adaptive domain-pool density (`docs/decisions.md` D43, measured
    /// -6.82% mean BD-rate on kodim01/kodim02). Applies to every plane (Y, Cb, Cr alike) --
    /// a codec-wide encode choice, not a luma/chroma-specific one like `t_rms`. `false`
    /// (the default) reproduces every pre-Step-16 caller's behaviour byte-for-byte.
    pub adaptive_density: bool,
    /// Step 15's per-leaf mode mask (flat, affine, fractal, fractal+residual), applied to
    /// every plane alike -- a diagnostic/comparison knob (isolating one mode's effect,
    /// reproducing a mode-usage histogram), not a quality control. Only consulted on the
    /// RD (`EncodeParams::lambda: Some`) path -- the legacy top-down path never reaches
    /// mode competition at all. `[true; 4]` (the
    /// default) reproduces every pre-existing caller's behaviour exactly.
    pub allowed_modes: [bool; 4],
    /// P5c: how many of the search's best-RMS domain candidates compete for modes 2/3 on
    /// the RD path, per plane. `1` (the default) is the historical single-winner
    /// behaviour and is byte-identical to it. See [`EncodeOptions::rd_candidates`].
    pub rd_candidates: usize,
    /// Spatially varying lambda for the **luma** plane (human-adaptive encoding): each
    /// [`LambdaRegion`] is in luma pixel coordinates and scales `params.y.lambda` for blocks
    /// overlapping it. Chroma keeps the run's uniform lambda -- human sensitivity to chroma
    /// detail is low, and the regions are defined from luma-space detections. Empty (the
    /// default) reproduces every pre-existing caller byte-for-byte.
    pub lambda_regions: Vec<LambdaRegion>,
    /// Regions (luma pixel coordinates) that **keep their colour**; chroma outside their
    /// union is forced to neutral before encoding ([`neutralise_chroma_outside`]), so the
    /// decoder reconstructs those pixels as grayscale (`R == G == B`). This is what
    /// `encmars --human-adaptive --color-faces-only` (whole face boxes) and
    /// `--color-features-only` (eyes/nose/mouth boxes) fill: colour inside those regions, a
    /// luma-only (grayscale) image with the same detail everywhere else. Empty (the
    /// default) leaves chroma untouched and reproduces every pre-existing caller
    /// byte-for-byte. An empty set because nothing was detected is deliberately a no-op, not
    /// "grayscale everywhere" -- see `docs/decisions.md` D52. Ignored for grayscale input.
    /// When non-empty, the chroma planes' `bits_beta` is raised to at least 8 (in
    /// [`encode_color_image_with_residual_quantisation`]) so that neutral reconstructs
    /// exactly; below 8 bits the DC step skips 128 and the background would keep a
    /// one-level colour cast. Luma is unaffected.
    pub color_regions: Vec<LambdaRegion>,
}

/// Per-plane stats from a colour encode, for measurement (bpp attribution, evals).
#[derive(Debug, Clone, Copy)]
pub struct ColorEncodeStats {
    pub y_bytes: usize,
    pub cb_bytes: usize,
    pub cr_bytes: usize,
    pub y_evals: u64,
    pub cb_evals: u64,
    pub cr_evals: u64,
}

impl ColorEncodeStats {
    pub fn total_bytes(&self) -> usize {
        self.y_bytes + self.cb_bytes + self.cr_bytes
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ColorFormatError {
    #[error("not a .mars colour container: bad magic")]
    BadMagic,
    #[error("unsupported colour container version {0}")]
    UnsupportedVersion(u8),
    #[error("container truncated")]
    Truncated,
    #[error("unknown colour mode {0}")]
    UnknownMode(u8),
    #[error("section table claims {claimed} bytes but only {available} remain")]
    SectionOutOfBounds { claimed: usize, available: usize },
    #[error("plane stream error: {0}")]
    Plane(#[from] MarsFormatError),
}

const COLOR_MAGIC: [u8; 4] = *b"MARC";
const COLOR_VERSION: u8 = 0;
const MODE_GRAY: u8 = 0;
const MODE_RGB_444: u8 = 1;
const MODE_RGB_420: u8 = 2;

/// Wrap one already-written grayscale `mars_format` plane stream in the same `MARC`
/// single-plane container [`encode_color_image`]'s own `Gray` arm produces -- for callers
/// (CLI-C's `encmars --method`, `mars-search`-driven grayscale-only encodes) that build
/// the plane stream themselves via a different encoder (`mars_search::encode_image`, not
/// this module's own `encode_image_with_options`) but still need to hand
/// `decmars` something it can read. Colour is not supported through this path -- callers
/// needing colour go through [`encode_color_image`] itself.
pub fn wrap_gray_stream(plane_bytes: Vec<u8>) -> Vec<u8> {
    write_container(MODE_GRAY, &[plane_bytes])
}

/// Encode an [`Image`] (gray or RGB) to a colour `.mars` container with fixed residual step 8.
/// Use [`encode_color_image_with_residual_quantisation`] for explicit experiments.
///
/// For `Gray` input, `params.y` is used and `params.chroma`/`params.subsampling` are
/// ignored (there is no chroma to encode) — this keeps a single call site working for
/// both colour spaces rather than forcing every caller to branch.
pub fn encode_color_image(img: &Image, params: &ColorEncodeParams) -> (Vec<u8>, ColorEncodeStats) {
    encode_color_image_with_residual_quantisation(img, params, ResidualQuantisation::default())
}

/// Encode gray or RGB with an explicit residual quantisation policy, without changing
/// `ColorEncodeParams` literals. `LambdaAdaptive` derives each plane's step from its
/// own lambda (`params.y` for Y, `params.chroma` for Cb/Cr); `Fixed` uses one step
/// across all planes. For gray input only `params.y` is used.
pub fn encode_color_image_with_residual_quantisation(
    img: &Image,
    params: &ColorEncodeParams,
    policy: ResidualQuantisation,
) -> (Vec<u8>, ColorEncodeStats) {
    let y_options = EncodeOptions {
        allowed_modes: params.allowed_modes,
        adaptive_density: params.adaptive_density,
        residual_quantisation: policy,
        rd_candidates: params.rd_candidates,
        lambda_regions: params.lambda_regions.clone(),
    };
    // Chroma intentionally drops the luma-space regions (see `ColorEncodeParams`).
    let chroma_options = EncodeOptions {
        lambda_regions: Vec::new(),
        ..y_options.clone()
    };
    match img.color() {
        ColorSpace::Gray => {
            let (hdr, leaves, evals, _stats) =
                encode_image_with_options(&img.planes()[0], &params.y, &y_options);
            let bytes = mars_format::write(&hdr, &leaves)
                .expect("encode_image always produces a header valid for mars_format::write");
            let y_bytes = bytes.len();
            let container = write_container(MODE_GRAY, &[bytes]);
            let stats = ColorEncodeStats {
                y_bytes,
                cb_bytes: 0,
                cr_bytes: 0,
                y_evals: evals,
                cb_evals: 0,
                cr_evals: 0,
            };
            (container, stats)
        }
        ColorSpace::Rgb => {
            let [y, mut cb, mut cr] = ycbcr(img);
            neutralise_chroma_outside(&mut cb, &mut cr, &params.color_regions);
            // Exact neutral chroma needs a DC step of 1.0. Below 8 bits,
            // `qbeta/((1<<bits_beta)-1)*255` steps past 128 (127 -> 127, 128 -> 129), so the
            // neutralised background would reconstruct with a one-level colour cast instead
            // of being gray. Raise only the chroma planes, and only when a mask is in use.
            let chroma_params = if params.color_regions.is_empty() {
                params.chroma
            } else {
                EncodeParams {
                    bits_beta: params.chroma.bits_beta.max(8),
                    ..params.chroma
                }
            };
            let (mode, cb_enc, cr_enc) = match params.subsampling {
                Subsampling::Yuv444 => (MODE_RGB_444, cb, cr),
                Subsampling::Yuv420 => (MODE_RGB_420, downsample_box(&cb), downsample_box(&cr)),
            };

            let (y_hdr, y_leaves, y_evals, _) =
                encode_image_with_options(&y, &params.y, &y_options);
            let (cb_hdr, cb_leaves, cb_evals, _) =
                encode_image_with_options(&cb_enc, &chroma_params, &chroma_options);
            let (cr_hdr, cr_leaves, cr_evals, _) =
                encode_image_with_options(&cr_enc, &chroma_params, &chroma_options);

            let y_bytes = mars_format::write(&y_hdr, &y_leaves).expect("valid header");
            let cb_bytes = mars_format::write(&cb_hdr, &cb_leaves).expect("valid header");
            let cr_bytes = mars_format::write(&cr_hdr, &cr_leaves).expect("valid header");

            let stats = ColorEncodeStats {
                y_bytes: y_bytes.len(),
                cb_bytes: cb_bytes.len(),
                cr_bytes: cr_bytes.len(),
                y_evals,
                cb_evals,
                cr_evals,
            };
            let container = write_container(mode, &[y_bytes, cb_bytes, cr_bytes]);
            (container, stats)
        }
    }
}

/// Decode a colour `.mars` container back to an [`Image`].
pub fn decode_color_image(data: &[u8], iterations: u32) -> Result<Image, ColorFormatError> {
    let (mode, streams) = read_container(data)?;
    match mode {
        MODE_GRAY => {
            let (hdr, leaves) = mars_format::read(&streams[0])?;
            let plane = decode_iterative(&hdr, &leaves, iterations);
            Ok(Image::gray(plane))
        }
        MODE_RGB_444 | MODE_RGB_420 => {
            let (y_hdr, y_leaves) = mars_format::read(&streams[0])?;
            let (cb_hdr, cb_leaves) = mars_format::read(&streams[1])?;
            let (cr_hdr, cr_leaves) = mars_format::read(&streams[2])?;

            let y = decode_iterative(&y_hdr, &y_leaves, iterations);
            let cb_small = decode_iterative(&cb_hdr, &cb_leaves, iterations);
            let cr_small = decode_iterative(&cr_hdr, &cr_leaves, iterations);

            let (w, h) = (y.width(), y.height());
            let (cb, cr) = if mode == MODE_RGB_420 {
                (
                    upsample_nearest(&cb_small, w, h),
                    upsample_nearest(&cr_small, w, h),
                )
            } else {
                (cb_small, cr_small)
            };
            Ok(rgb_from_ycbcr(&y, &cb, &cr))
        }
        other => Err(ColorFormatError::UnknownMode(other)),
    }
}

/// Mirror of [`decode_color_image`] that decodes at `zoom`x the encoded resolution instead
/// of the bitstream's native size — see [`crate::ifs::zoom_leaves`] for what "zoom" means
/// for a fractal decode (enlarging or shrinking the canvas the same contractive map runs
/// over, not resampling a finished image). `zoom == 1.0` behaves exactly like
/// `decode_color_image`. Each plane is scaled independently by its own header, so this
/// works the same for 4:2:0 chroma as for 4:4:4.
pub fn decode_color_image_zoomed(
    data: &[u8],
    iterations: u32,
    zoom: f64,
) -> Result<Image, ColorFormatError> {
    let (mode, streams) = read_container(data)?;
    match mode {
        MODE_GRAY => {
            let (hdr, leaves) = mars_format::read(&streams[0])?;
            let (hdr, leaves) = zoom_leaves(&hdr, &leaves, zoom);
            let plane = decode_iterative(&hdr, &leaves, iterations);
            Ok(Image::gray(plane))
        }
        MODE_RGB_444 | MODE_RGB_420 => {
            let (y_hdr, y_leaves) = mars_format::read(&streams[0])?;
            let (cb_hdr, cb_leaves) = mars_format::read(&streams[1])?;
            let (cr_hdr, cr_leaves) = mars_format::read(&streams[2])?;
            let (y_hdr, y_leaves) = zoom_leaves(&y_hdr, &y_leaves, zoom);
            let (cb_hdr, cb_leaves) = zoom_leaves(&cb_hdr, &cb_leaves, zoom);
            let (cr_hdr, cr_leaves) = zoom_leaves(&cr_hdr, &cr_leaves, zoom);

            let y = decode_iterative(&y_hdr, &y_leaves, iterations);
            let cb_small = decode_iterative(&cb_hdr, &cb_leaves, iterations);
            let cr_small = decode_iterative(&cr_hdr, &cr_leaves, iterations);

            let (w, h) = (y.width(), y.height());
            let (cb, cr) = if mode == MODE_RGB_420 {
                (
                    upsample_nearest(&cb_small, w, h),
                    upsample_nearest(&cr_small, w, h),
                )
            } else {
                (cb_small, cr_small)
            };
            Ok(rgb_from_ycbcr(&y, &cb, &cr))
        }
        other => Err(ColorFormatError::UnknownMode(other)),
    }
}

/// Mirror of [`decode_color_image`] using [`decode_until_stable`] instead of a fixed
/// iteration count for each plane. Each plane converges independently (luma and chroma
/// stabilise at different rates), so the return value reports the worst-case (maximum)
/// iteration count across planes — the number a caller would need to reproduce this
/// decode with the fixed-count API. `zoom` behaves as in [`decode_color_image_zoomed`].
pub fn decode_color_image_auto(
    data: &[u8],
    threshold: u8,
    max_iterations: u32,
    zoom: f64,
) -> Result<(Image, u32), ColorFormatError> {
    let (mode, streams) = read_container(data)?;
    match mode {
        MODE_GRAY => {
            let (hdr, leaves) = mars_format::read(&streams[0])?;
            let (hdr, leaves) = zoom_leaves(&hdr, &leaves, zoom);
            let (plane, used) = decode_until_stable(&hdr, &leaves, threshold, max_iterations);
            Ok((Image::gray(plane), used))
        }
        MODE_RGB_444 | MODE_RGB_420 => {
            let (y_hdr, y_leaves) = mars_format::read(&streams[0])?;
            let (cb_hdr, cb_leaves) = mars_format::read(&streams[1])?;
            let (cr_hdr, cr_leaves) = mars_format::read(&streams[2])?;
            let (y_hdr, y_leaves) = zoom_leaves(&y_hdr, &y_leaves, zoom);
            let (cb_hdr, cb_leaves) = zoom_leaves(&cb_hdr, &cb_leaves, zoom);
            let (cr_hdr, cr_leaves) = zoom_leaves(&cr_hdr, &cr_leaves, zoom);

            let (y, y_used) = decode_until_stable(&y_hdr, &y_leaves, threshold, max_iterations);
            let (cb_small, cb_used) =
                decode_until_stable(&cb_hdr, &cb_leaves, threshold, max_iterations);
            let (cr_small, cr_used) =
                decode_until_stable(&cr_hdr, &cr_leaves, threshold, max_iterations);
            let used = y_used.max(cb_used).max(cr_used);

            let (w, h) = (y.width(), y.height());
            let (cb, cr) = if mode == MODE_RGB_420 {
                (
                    upsample_nearest(&cb_small, w, h),
                    upsample_nearest(&cr_small, w, h),
                )
            } else {
                (cb_small, cr_small)
            };
            Ok((rgb_from_ycbcr(&y, &cb, &cr), used))
        }
        other => Err(ColorFormatError::UnknownMode(other)),
    }
}

/// Decodes a colour `.mars` container one iteration at a time, calling `on_iteration` with
/// the 1-indexed iteration number and the reconstructed [`Image`] after each step, so a
/// caller can save a frame per iteration and watch the fractal decode converge. Planes are
/// advanced in lock-step (unlike [`decode_color_image`], which runs each plane to
/// completion before moving to the next) so every emitted frame is a single coherent
/// image, not a partially-updated one.
///
/// Stops after `max_iterations`, or earlier once `stable_threshold` is given and every
/// plane's worst-case pixel movement drops to that value or below — mirroring
/// [`decode_until_stable`]'s convergence test, applied across all planes at once. `zoom`
/// behaves as in [`decode_color_image_zoomed`] — each frame emitted is at the zoomed
/// resolution. Returns the final image and the iteration count actually run.
pub fn decode_color_image_progression(
    data: &[u8],
    max_iterations: u32,
    stable_threshold: Option<u8>,
    zoom: f64,
    mut on_iteration: impl FnMut(u32, &Image),
) -> Result<(Image, u32), ColorFormatError> {
    let (mode, streams) = read_container(data)?;
    match mode {
        MODE_GRAY => {
            let (hdr, leaves) = mars_format::read(&streams[0])?;
            let (hdr, leaves) = zoom_leaves(&hdr, &leaves, zoom);
            let (w, h) = (hdr.width as usize, hdr.height as usize);
            let mut img = vec![128u8; w * h];
            let mut used = 0;
            let mut image = Image::gray(Plane::from_vec(w, h, img.clone()));
            for i in 0..max_iterations.max(1) {
                let next = decode_step(&hdr, &leaves, &img);
                let delta = max_pixel_delta(&img, &next);
                img = next;
                used = i + 1;
                image = Image::gray(Plane::from_vec(w, h, img.clone()));
                on_iteration(used, &image);
                if stable_threshold.is_some_and(|t| delta <= t) {
                    break;
                }
            }
            Ok((image, used))
        }
        MODE_RGB_444 | MODE_RGB_420 => {
            let (y_hdr, y_leaves) = mars_format::read(&streams[0])?;
            let (cb_hdr, cb_leaves) = mars_format::read(&streams[1])?;
            let (cr_hdr, cr_leaves) = mars_format::read(&streams[2])?;
            let (y_hdr, y_leaves) = zoom_leaves(&y_hdr, &y_leaves, zoom);
            let (cb_hdr, cb_leaves) = zoom_leaves(&cb_hdr, &cb_leaves, zoom);
            let (cr_hdr, cr_leaves) = zoom_leaves(&cr_hdr, &cr_leaves, zoom);

            let (yw, yh) = (y_hdr.width as usize, y_hdr.height as usize);
            let (cw, ch) = (cb_hdr.width as usize, cb_hdr.height as usize);
            let mut y_img = vec![128u8; yw * yh];
            let mut cb_img = vec![128u8; cw * ch];
            let mut cr_img = vec![128u8; cw * ch];
            let mut used = 0;
            // Bugfix: the flat-grey seed image must go through the same 4:2:0 upsample
            // the loop below applies to every subsequent iteration -- `rgb_from_ycbcr`
            // asserts its three planes share one size, and cb_img/cr_img are only
            // (yw, yh)-sized post-upsample; at their native (cw, ch) they are half that
            // in each dimension for 4:2:0, which panicked the assertion (not merely
            // producing a wrong seed) before this pre-loop value was ever overwritten by
            // the loop's own first iteration.
            let (cb_seed, cr_seed) = if mode == MODE_RGB_420 {
                (
                    upsample_nearest(&Plane::from_vec(cw, ch, cb_img.clone()), yw, yh),
                    upsample_nearest(&Plane::from_vec(cw, ch, cr_img.clone()), yw, yh),
                )
            } else {
                (
                    Plane::from_vec(cw, ch, cb_img.clone()),
                    Plane::from_vec(cw, ch, cr_img.clone()),
                )
            };
            let mut image =
                rgb_from_ycbcr(&Plane::from_vec(yw, yh, y_img.clone()), &cb_seed, &cr_seed);
            for i in 0..max_iterations.max(1) {
                let y_next = decode_step(&y_hdr, &y_leaves, &y_img);
                let cb_next = decode_step(&cb_hdr, &cb_leaves, &cb_img);
                let cr_next = decode_step(&cr_hdr, &cr_leaves, &cr_img);
                let delta = max_pixel_delta(&y_img, &y_next)
                    .max(max_pixel_delta(&cb_img, &cb_next))
                    .max(max_pixel_delta(&cr_img, &cr_next));
                y_img = y_next;
                cb_img = cb_next;
                cr_img = cr_next;
                used = i + 1;

                let y_plane = Plane::from_vec(yw, yh, y_img.clone());
                let cb_plane = Plane::from_vec(cw, ch, cb_img.clone());
                let cr_plane = Plane::from_vec(cw, ch, cr_img.clone());
                let (cb_up, cr_up) = if mode == MODE_RGB_420 {
                    (
                        upsample_nearest(&cb_plane, yw, yh),
                        upsample_nearest(&cr_plane, yw, yh),
                    )
                } else {
                    (cb_plane, cr_plane)
                };
                image = rgb_from_ycbcr(&y_plane, &cb_up, &cr_up);
                on_iteration(used, &image);
                if stable_threshold.is_some_and(|t| delta <= t) {
                    break;
                }
            }
            Ok((image, used))
        }
        other => Err(ColorFormatError::UnknownMode(other)),
    }
}

/// Renders the quadtree partition as a standalone black-on-white image — the debug view
/// reference Mars 1's `encmars -Q` writes to `quadtree.pgm` (`mars_enc.c`: white canvas,
/// black lines at every split boundary), reconstructed here from the leaves `decmars`
/// already parses out of the luma (or grayscale) stream. Native bitstream resolution,
/// independent of any `--zoom`; chroma streams share the same partition shape up to
/// subsampling and are not drawn separately.
pub fn quadtree_image(data: &[u8]) -> Result<Image, ColorFormatError> {
    let (_mode, streams) = read_container(data)?;
    let (hdr, leaves) = mars_format::read(&streams[0])?;
    Ok(Image::gray(quadtree_plane(&hdr, &leaves)))
}

fn quadtree_plane(hdr: &crate::ifs::Header, leaves: &[crate::ifs::Leaf]) -> Plane {
    let (width, height) = (hdr.width as usize, hdr.height as usize);
    let mut pixels = vec![255u8; width * height];
    for leaf in leaves {
        let (row, col, size) = (leaf.row as usize, leaf.col as usize, leaf.size as usize);
        for dx in 0..size {
            let x = col + dx;
            if x >= width {
                continue;
            }
            if row < height {
                pixels[row * width + x] = 0;
            }
            let bottom = row + size - 1;
            if bottom < height {
                pixels[bottom * width + x] = 0;
            }
        }
        for dy in 0..size {
            let y = row + dy;
            if y >= height {
                continue;
            }
            if col < width {
                pixels[y * width + col] = 0;
            }
            let right = col + size - 1;
            if right < width {
                pixels[y * width + right] = 0;
            }
        }
    }
    Plane::from_vec(width, height, pixels)
}

const COLOR_HEADER_LEN: usize = 4 /* magic */ + 1 /* version */ + 1 /* mode */ + 1 /* section_count */;

fn write_container(mode: u8, streams: &[Vec<u8>]) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(COLOR_HEADER_LEN + streams.iter().map(|s| 4 + s.len()).sum::<usize>());
    out.extend_from_slice(&COLOR_MAGIC);
    out.push(COLOR_VERSION);
    out.push(mode);
    out.push(streams.len() as u8);
    for s in streams {
        out.extend_from_slice(&(u32::try_from(s.len()).expect("stream fits in u32")).to_le_bytes());
        out.extend_from_slice(s);
    }
    out
}

fn read_container(data: &[u8]) -> Result<(u8, Vec<Vec<u8>>), ColorFormatError> {
    if data.len() < COLOR_HEADER_LEN {
        return Err(ColorFormatError::Truncated);
    }
    if data[0..4] != COLOR_MAGIC {
        return Err(ColorFormatError::BadMagic);
    }
    let version = data[4];
    if version != COLOR_VERSION {
        return Err(ColorFormatError::UnsupportedVersion(version));
    }
    let mode = data[5];
    let section_count = data[6] as usize;
    let mut offset = COLOR_HEADER_LEN;
    let mut out = Vec::with_capacity(section_count);
    for _ in 0..section_count {
        if offset + 4 > data.len() {
            return Err(ColorFormatError::Truncated);
        }
        let len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        if offset + len > data.len() {
            return Err(ColorFormatError::SectionOutOfBounds {
                claimed: len,
                available: data.len() - offset,
            });
        }
        out.push(data[offset..offset + len].to_vec());
        offset += len;
    }
    Ok((mode, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mars_core::image::Plane as CorePlane;

    fn gradient_image(w: usize, h: usize) -> Image {
        let mut r = Vec::with_capacity(w * h);
        let mut g = Vec::with_capacity(w * h);
        let mut b = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                r.push(((x * 5) % 256) as u8);
                g.push(((y * 5) % 256) as u8);
                b.push((((x + y) * 3) % 256) as u8);
            }
        }
        Image::rgb(
            CorePlane::from_vec(w, h, r),
            CorePlane::from_vec(w, h, g),
            CorePlane::from_vec(w, h, b),
        )
    }

    fn params(t_rms: f64) -> EncodeParams {
        EncodeParams {
            min_size: 4,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            t_rms,
            zero_threshold: 0,
            lambda: None,
        }
    }

    #[test]
    fn downsample_upsample_round_trip_dimensions() {
        let p = CorePlane::from_vec(5, 3, (0..15).map(|v| v as u8).collect());
        let down = downsample_box(&p);
        assert_eq!((down.width(), down.height()), (3, 2));
        let up = upsample_nearest(&down, 5, 3);
        assert_eq!((up.width(), up.height()), (5, 3));
    }

    #[test]
    fn downsample_box_averages_a_flat_block_exactly() {
        let p = CorePlane::filled(4, 4, 100);
        let down = downsample_box(&p);
        assert!(down.as_slice().iter().all(|&v| v == 100));
    }

    #[test]
    fn gray_round_trips_through_the_color_container() {
        let img = Image::gray(CorePlane::from_vec(
            16,
            16,
            (0..256).map(|v| v as u8).collect(),
        ));
        let cfg = ColorEncodeParams {
            y: params(1.0),
            chroma: params(1.0),
            subsampling: Subsampling::Yuv444,
            adaptive_density: false,
            allowed_modes: [true; 4],
            rd_candidates: 1,
            lambda_regions: Vec::new(),
            color_regions: Vec::new(),
        };
        let (bytes, _stats) = encode_color_image(&img, &cfg);
        let decoded = decode_color_image(&bytes, 10).unwrap();
        assert_eq!(decoded.color(), ColorSpace::Gray);
        assert_eq!(decoded.width(), 16);
        assert_eq!(decoded.height(), 16);
    }

    #[test]
    fn rgb_444_round_trips_with_reasonable_quality() {
        let img = gradient_image(64, 64);
        let cfg = ColorEncodeParams {
            y: params(4.0),
            chroma: params(4.0),
            subsampling: Subsampling::Yuv444,
            adaptive_density: false,
            allowed_modes: [true; 4],
            rd_candidates: 1,
            lambda_regions: Vec::new(),
            color_regions: Vec::new(),
        };
        let (bytes, stats) = encode_color_image(&img, &cfg);
        let decoded = decode_color_image(&bytes, 10).unwrap();
        assert_eq!(decoded.color(), ColorSpace::Rgb);
        assert_eq!((decoded.width(), decoded.height()), (64, 64));
        assert!(stats.total_bytes() > 0);

        let quality = mars_core::metrics::quality(
            &img,
            &decoded,
            None,
            &mars_core::metrics::SsimConfig::default(),
            &mars_core::metrics::MsSsimConfig::default(),
        );
        let psnr_y = quality
            .psnr_y
            .expect("non-identical images give a finite PSNR");
        assert!(
            psnr_y > 20.0,
            "PSNR-Y {psnr_y} looks too low for a round trip"
        );
    }

    /// P5c: `ColorEncodeParams::rd_candidates` must reach every plane's RD walk -- a
    /// grayscale-only implementation would silently drop the field on this path -- and must
    /// not break the container round trip or determinism.
    #[test]
    fn colour_rd_candidates_plumb_through_every_plane_and_round_trip() {
        let img = gradient_image(64, 64);
        let rd = EncodeParams {
            lambda: Some(200.0),
            ..params(8.0)
        };
        let cfg = ColorEncodeParams {
            y: rd,
            chroma: rd,
            subsampling: Subsampling::Yuv444,
            adaptive_density: false,
            allowed_modes: [true; 4],
            rd_candidates: 3,
            lambda_regions: Vec::new(),
            color_regions: Vec::new(),
        };
        let (bytes, stats) = encode_color_image(&img, &cfg);
        let (bytes_again, _) = encode_color_image(&img, &cfg);
        assert_eq!(
            bytes, bytes_again,
            "a wider candidate list must be deterministic"
        );
        assert!(stats.cb_bytes > 0 && stats.cr_bytes > 0);
        let decoded = decode_color_image(&bytes, 10).unwrap();
        assert_eq!(decoded.color(), ColorSpace::Rgb);
        assert_eq!((decoded.width(), decoded.height()), (64, 64));
    }

    #[test]
    fn rgb_420_round_trips_and_produces_smaller_chroma_streams_than_444() {
        let img = gradient_image(64, 64);
        let cfg_444 = ColorEncodeParams {
            y: params(4.0),
            chroma: params(4.0),
            subsampling: Subsampling::Yuv444,
            adaptive_density: false,
            allowed_modes: [true; 4],
            rd_candidates: 1,
            lambda_regions: Vec::new(),
            color_regions: Vec::new(),
        };
        let cfg_420 = ColorEncodeParams {
            y: params(4.0),
            chroma: params(4.0),
            subsampling: Subsampling::Yuv420,
            adaptive_density: false,
            allowed_modes: [true; 4],
            rd_candidates: 1,
            lambda_regions: Vec::new(),
            color_regions: Vec::new(),
        };
        let (bytes_444, stats_444) = encode_color_image(&img, &cfg_444);
        let (bytes_420, stats_420) = encode_color_image(&img, &cfg_420);

        let decoded_420 = decode_color_image(&bytes_420, 10).unwrap();
        assert_eq!((decoded_420.width(), decoded_420.height()), (64, 64));

        // 4:2:0's chroma planes are a quarter the pixel count, so at matched t_rms they
        // should not cost more bits than 4:4:4's chroma -- a very weak sanity check, not
        // the BD-rate comparison itself (see docs/predictions.md P18.1 for that).
        assert!(
            stats_420.cb_bytes + stats_420.cr_bytes <= stats_444.cb_bytes + stats_444.cr_bytes,
            "420 chroma ({}) should not exceed 444 chroma ({}) at matched t_rms",
            stats_420.cb_bytes + stats_420.cr_bytes,
            stats_444.cb_bytes + stats_444.cr_bytes
        );
        assert!(
            bytes_420.len() < bytes_444.len() || stats_420.total_bytes() <= stats_444.total_bytes()
        );
    }

    /// **Regression test for a real bug**, reported as a panic running `decmars -a
    /// --progression <dir>` on a 4:2:0 file: `assertion left == right failed: left:
    /// (768, 512) right: (384, 256)` -- `rgb_from_ycbcr` requires its three planes share
    /// one size, but `decode_color_image_progression`'s pre-loop seed image built Cb/Cr
    /// at their native (half-resolution, for 4:2:0) size instead of upsampling to Y's
    /// size first, the way every other iteration in the same loop already does. The seed
    /// value is always overwritten by the loop's own first iteration before being
    /// returned, so this was a pure construction bug, not a real need for a differently-
    /// sized seed.
    #[test]
    fn progression_decode_does_not_panic_on_420_input() {
        let img = gradient_image(64, 64);
        let cfg_420 = ColorEncodeParams {
            y: params(4.0),
            chroma: params(4.0),
            subsampling: Subsampling::Yuv420,
            adaptive_density: false,
            allowed_modes: [true; 4],
            rd_candidates: 1,
            lambda_regions: Vec::new(),
            color_regions: Vec::new(),
        };
        let (bytes_420, _stats) = encode_color_image(&img, &cfg_420);

        let mut frame_count = 0;
        let (decoded, used) =
            decode_color_image_progression(&bytes_420, 5, None, 1.0, |_n, _frame| {
                frame_count += 1;
            })
            .expect("progression decode of a 4:2:0 stream must not panic or error");
        assert_eq!(used, 5);
        assert_eq!(frame_count, 5);
        assert_eq!((decoded.width(), decoded.height()), (64, 64));
    }

    /// The colour-region mask's core contract (shared by `--color-faces-only` and
    /// `--color-features-only`): chroma outside the given regions is neutral, so those
    /// pixels decode to grayscale, while colour inside the regions survives -- with no
    /// decoder or format change. The region is dyadic (exactly half a 64x64 image), so no
    /// quadtree leaf straddles its edge; a straddling leaf is the documented "approximately"
    /// caveat, and asserting on it would test leaf geometry rather than this contract.
    #[test]
    fn color_regions_keep_colour_inside_and_force_grayscale_outside() {
        let img = gradient_image(64, 64);
        let cfg = ColorEncodeParams {
            y: params(1.0),
            chroma: params(1.0),
            subsampling: Subsampling::Yuv444,
            adaptive_density: false,
            allowed_modes: [true; 4],
            rd_candidates: 1,
            lambda_regions: Vec::new(),
            color_regions: vec![LambdaRegion {
                row: 0,
                col: 0,
                height: 32,
                width: 64,
                scale: 1.0,
                min_size: None,
            }],
        };
        let (bytes, stats) = encode_color_image(&img, &cfg);
        let decoded = decode_color_image(&bytes, 10).unwrap();
        let [r, g, b] = decoded.planes() else {
            panic!("expected an RGB decode");
        };

        // Outside the region (the lower half): exactly neutral, so R == G == B.
        for y in 32..64 {
            for x in 0..64 {
                assert!(
                    r.get(x, y) == g.get(x, y) && g.get(x, y) == b.get(x, y),
                    "pixel ({x}, {y}) outside the colour region should be grayscale, got \
                     ({}, {}, {})",
                    r.get(x, y),
                    g.get(x, y),
                    b.get(x, y)
                );
            }
        }
        // Inside it, colour must have survived -- otherwise the assertion above is vacuous.
        let colourful_inside = (0..32).any(|y| {
            (0..64).any(|x| r.get(x, y) != g.get(x, y) || g.get(x, y) != b.get(x, y))
        });
        assert!(colourful_inside, "the colour region lost all its colour");

        // Masking chroma must not cost more chroma bits than keeping it everywhere.
        let plain = ColorEncodeParams {
            color_regions: Vec::new(),
            ..cfg.clone()
        };
        let (_, baseline) = encode_color_image(&img, &plain);
        assert!(
            stats.cb_bytes + stats.cr_bytes <= baseline.cb_bytes + baseline.cr_bytes,
            "neutralising chroma should not cost more: {}+{} vs {}+{}",
            stats.cb_bytes,
            stats.cr_bytes,
            baseline.cb_bytes,
            baseline.cr_bytes
        );
    }

    #[test]
    fn container_rejects_bad_magic() {
        let err = decode_color_image(&[0u8; 16], 1).unwrap_err();
        assert!(matches!(err, ColorFormatError::BadMagic));
    }

    #[test]
    fn container_rejects_truncated_input() {
        let err = decode_color_image(b"MARC", 1).unwrap_err();
        assert!(matches!(err, ColorFormatError::Truncated));
    }
}
