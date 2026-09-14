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

use crate::encode::{encode_image_rd_with_modes_and_density, EncodeParams};
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
/// this module's own `encode_image_rd_with_modes_and_density`) but still need to hand
/// `decmars` something it can read. Colour is not supported through this path -- callers
/// needing colour go through [`encode_color_image`] itself.
pub fn wrap_gray_stream(plane_bytes: Vec<u8>) -> Vec<u8> {
    write_container(MODE_GRAY, &[plane_bytes])
}

/// Encode an [`Image`] (gray or RGB) to a colour `.mars` container.
///
/// For `Gray` input, `params.y` is used and `params.chroma`/`params.subsampling` are
/// ignored (there is no chroma to encode) — this keeps a single call site working for
/// both colour spaces rather than forcing every caller to branch.
pub fn encode_color_image(img: &Image, params: &ColorEncodeParams) -> (Vec<u8>, ColorEncodeStats) {
    match img.color() {
        ColorSpace::Gray => {
            let (hdr, leaves, evals, _stats) = encode_image_rd_with_modes_and_density(
                &img.planes()[0],
                &params.y,
                params.allowed_modes,
                params.adaptive_density,
            );
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
            let [y, cb, cr] = ycbcr(img);
            let (mode, cb_enc, cr_enc) = match params.subsampling {
                Subsampling::Yuv444 => (MODE_RGB_444, cb, cr),
                Subsampling::Yuv420 => (MODE_RGB_420, downsample_box(&cb), downsample_box(&cr)),
            };

            let (y_hdr, y_leaves, y_evals, _) = encode_image_rd_with_modes_and_density(
                &y,
                &params.y,
                params.allowed_modes,
                params.adaptive_density,
            );
            let (cb_hdr, cb_leaves, cb_evals, _) = encode_image_rd_with_modes_and_density(
                &cb_enc,
                &params.chroma,
                params.allowed_modes,
                params.adaptive_density,
            );
            let (cr_hdr, cr_leaves, cr_evals, _) = encode_image_rd_with_modes_and_density(
                &cr_enc,
                &params.chroma,
                params.allowed_modes,
                params.adaptive_density,
            );

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
            let mut image = rgb_from_ycbcr(
                &Plane::from_vec(yw, yh, y_img.clone()),
                &Plane::from_vec(cw, ch, cb_img.clone()),
                &Plane::from_vec(cw, ch, cr_img.clone()),
            );
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

    #[test]
    fn rgb_420_round_trips_and_produces_smaller_chroma_streams_than_444() {
        let img = gradient_image(64, 64);
        let cfg_444 = ColorEncodeParams {
            y: params(4.0),
            chroma: params(4.0),
            subsampling: Subsampling::Yuv444,
            adaptive_density: false,
            allowed_modes: [true; 4],
        };
        let cfg_420 = ColorEncodeParams {
            y: params(4.0),
            chroma: params(4.0),
            subsampling: Subsampling::Yuv420,
            adaptive_density: false,
            allowed_modes: [true; 4],
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
