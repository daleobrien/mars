//! **The** quality metrics (§M1, §M2).
//!
//! Every number the project reports about image quality comes from this file. No codec
//! ever reports its own PSNR; the harness computes it from two files on disk.
//!
//! The definitions are pinned, and the pinning is the point — each one names the exact
//! convention it follows, because the cheap variants (padded instead of valid windows,
//! sample instead of population covariance, overlapping instead of non-overlapping
//! decimation) all produce plausible numbers that differ in the third decimal place.

use crate::image::{ColorSpace, Image, Plane};

/// 8-bit dynamic range. The project is 8-bit throughout; readers reject anything else.
pub const PEAK: f64 = 255.0;

// ---------------------------------------------------------------------------
// MSE / PSNR
// ---------------------------------------------------------------------------

/// Mean squared error between two equally sized planes.
///
/// §M2: the mean is taken over the plane as given. Callers are responsible for having
/// already cropped to the original `W x H` region — Mars 1 pads to `virtual_size` and
/// that padding must never be measured. [`Plane::crop_top_left`] is the tool.
///
/// # Panics
/// If the planes differ in size. Comparing mismatched images is always a harness bug,
/// never something to paper over with a resize.
pub fn mse(a: &Plane, b: &Plane) -> f64 {
    assert_eq!(
        (a.width(), a.height()),
        (b.width(), b.height()),
        "MSE requires identical dimensions"
    );
    let n = a.as_slice().len();
    assert!(n > 0, "MSE of an empty plane is undefined");
    let sum: u64 = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| {
            let d = i32::from(x) - i32::from(y);
            (d * d) as u64
        })
        .sum();
    // Exact: the numerator is an integer and fits comfortably in u64 for any image we
    // will ever read (255^2 * 2^32 pixels < 2^64), so there is no accumulation error.
    sum as f64 / n as f64
}

/// `10 * log10(255^2 / MSE)`.
///
/// Returns `None` for identical images. §M2 requires infinity to be emitted as JSON
/// `null` rather than a large float: a sentinel like 99.0 or 1e9 silently poisons any
/// average or BD-rate it reaches.
pub fn psnr_from_mse(mse: f64) -> Option<f64> {
    (mse > 0.0).then(|| 10.0 * (PEAK * PEAK / mse).log10())
}

/// Convenience: MSE then PSNR for one plane pair.
pub fn psnr(a: &Plane, b: &Plane) -> Option<f64> {
    psnr_from_mse(mse(a, b))
}

/// `8 * total_file_size_bytes / (W * H)`.
///
/// §M2: the **whole file**, header included. Payload-only bitrates are forbidden
/// because they make a format look better than it is by exactly the amount of overhead
/// it actually has.
pub fn bpp(file_size_bytes: u64, width: usize, height: usize) -> f64 {
    assert!(width > 0 && height > 0, "bpp needs a non-empty image");
    8.0 * file_size_bytes as f64 / (width * height) as f64
}

// ---------------------------------------------------------------------------
// SSIM
// ---------------------------------------------------------------------------

/// The pinned SSIM parameters (§M2): 11x11 Gaussian, sigma 1.5, K1 0.01, K2 0.03.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsimConfig {
    pub window: usize,
    pub sigma: f64,
    pub k1: f64,
    pub k2: f64,
    pub peak: f64,
}

impl Default for SsimConfig {
    fn default() -> Self {
        Self {
            window: 11,
            sigma: 1.5,
            k1: 0.01,
            k2: 0.03,
            peak: PEAK,
        }
    }
}

/// Which 2x decimation MS-SSIM uses between scales.
///
/// This is a parameter only because it is the one place where otherwise-equivalent
/// published implementations disagree, and pretending otherwise would mean declaring a
/// cross-validation failure "noise".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Downsample {
    /// **The pinned choice.** Non-overlapping 2x2 box average: `out[j] = mean(in[2j],
    /// in[2j+1])` per axis. This is Wang's reference `msssim.m`
    /// (`imfilter(ones(2)/4,'same')` then `1:2:end`) and is identical to the
    /// `avg_pool2d(2)` used by the PyTorch and TensorFlow implementations.
    #[default]
    Box2x2,
    /// `scipy.ndimage.uniform_filter(im, 2)` followed by `[::2, ::2]`, i.e. a window
    /// shifted half a pixel backwards with reflected edges. Present **only** so the
    /// MS-SSIM scale chain can be cross-validated against `sewar`, which uses it.
    /// Never used for reported numbers.
    ScipyUniform2,
}

/// MS-SSIM parameters (§M2): 5 scales, Wang weights.
///
/// The scale count *is* `weights.len()`, rather than a separate field that could
/// disagree with it. A single-element weight vector therefore reduces MS-SSIM to plain
/// SSIM, which is the identity the test suite uses to tie the two together.
#[derive(Debug, Clone, PartialEq)]
pub struct MsSsimConfig {
    pub ssim: SsimConfig,
    pub weights: Vec<f64>,
    pub downsample: Downsample,
}

/// The pinned weights from Wang, Simoncelli & Bovik (2003).
pub const WANG_WEIGHTS: [f64; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

impl Default for MsSsimConfig {
    fn default() -> Self {
        Self {
            ssim: SsimConfig::default(),
            weights: WANG_WEIGHTS.to_vec(),
            downsample: Downsample::Box2x2,
        }
    }
}

/// The smallest dimension MS-SSIM can be computed on: the 5th scale is a 16x decimation
/// and still needs a full 11-tap window, so `11 * 16 = 176`.
pub const MSSSIM_MIN_DIM: usize = 176;

/// Why a metric could not be computed. §A7: these are **flags, not fallbacks** — the
/// harness records the reason and excludes the row, and never silently rescales an
/// image to make a number appear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unavailable {
    /// Image smaller than the metric's window requires.
    TooSmall,
    /// A contrast-structure term went non-positive, so the weighted geometric mean is
    /// not real-valued. Only reachable on pathological synthetic input.
    NonPositiveContrast,
}

/// SSIM and its contrast-structure component, both as means over the **valid** region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsimResult {
    pub ssim: f64,
    pub cs: f64,
}

/// SSIM on one plane pair.
///
/// Pinned conventions, each of which changes the third decimal place if flipped:
/// - **valid** region only: no padding, and the mean is over the `(W-10) x (H-10)`
///   interior, as in Wang's `filter2(..., 'valid')`.
/// - **population** covariance (no `N/(N-1)` correction), as Wang's reference does.
/// - separable normalised Gaussian, which is exactly `fspecial('gaussian', 11, 1.5)`.
pub fn ssim(a: &Plane, b: &Plane, cfg: &SsimConfig) -> Result<SsimResult, Unavailable> {
    assert_eq!(
        (a.width(), a.height()),
        (b.width(), b.height()),
        "SSIM requires identical dimensions"
    );
    ssim_f64(&a.to_f64(), &b.to_f64(), a.width(), a.height(), cfg)
}

fn ssim_f64(
    x: &[f64],
    y: &[f64],
    w: usize,
    h: usize,
    cfg: &SsimConfig,
) -> Result<SsimResult, Unavailable> {
    if w < cfg.window || h < cfg.window {
        return Err(Unavailable::TooSmall);
    }
    let kernel = gaussian_kernel_1d(cfg.window, cfg.sigma);
    let c1 = (cfg.k1 * cfg.peak).powi(2);
    let c2 = (cfg.k2 * cfg.peak).powi(2);

    let xx: Vec<f64> = x.iter().map(|v| v * v).collect();
    let yy: Vec<f64> = y.iter().map(|v| v * v).collect();
    let xy: Vec<f64> = x.iter().zip(y).map(|(a, b)| a * b).collect();

    let mu_x = filter_valid(x, w, h, &kernel);
    let mu_y = filter_valid(y, w, h, &kernel);
    let f_xx = filter_valid(&xx, w, h, &kernel);
    let f_yy = filter_valid(&yy, w, h, &kernel);
    let f_xy = filter_valid(&xy, w, h, &kernel);

    let n = mu_x.len() as f64;
    let mut ssim_sum = 0.0;
    let mut cs_sum = 0.0;
    for i in 0..mu_x.len() {
        let (mx, my) = (mu_x[i], mu_y[i]);
        let sxx = f_xx[i] - mx * mx;
        let syy = f_yy[i] - my * my;
        let sxy = f_xy[i] - mx * my;
        let luminance = (2.0 * mx * my + c1) / (mx * mx + my * my + c1);
        let contrast = (2.0 * sxy + c2) / (sxx + syy + c2);
        ssim_sum += luminance * contrast;
        cs_sum += contrast;
    }
    Ok(SsimResult {
        ssim: ssim_sum / n,
        cs: cs_sum / n,
    })
}

/// MS-SSIM over 5 scales (§M2).
///
/// `MS-SSIM = prod_{j<M} cs_j^{w_j} * (ssim_M)^{w_M}`, where the final scale
/// contributes the luminance term and earlier scales contribute contrast-structure
/// only. Images whose smaller dimension is below [`MSSSIM_MIN_DIM`] are **flagged and
/// excluded**, never rescaled.
pub fn ms_ssim(a: &Plane, b: &Plane, cfg: &MsSsimConfig) -> Result<f64, Unavailable> {
    assert_eq!(
        (a.width(), a.height()),
        (b.width(), b.height()),
        "MS-SSIM requires identical dimensions"
    );
    let scales = cfg.weights.len();
    assert!(scales >= 1, "MS-SSIM needs at least one scale weight");
    let needed = cfg.ssim.window << (scales - 1);
    if a.width().min(a.height()) < needed {
        return Err(Unavailable::TooSmall);
    }

    let mut x = a.to_f64();
    let mut y = b.to_f64();
    let (mut w, mut h) = (a.width(), a.height());

    let mut product = 1.0;
    for (j, &weight) in cfg.weights.iter().enumerate() {
        let r = ssim_f64(&x, &y, w, h, &cfg.ssim)?;
        let last = j + 1 == scales;
        let term = if last { r.ssim } else { r.cs };
        if term <= 0.0 {
            return Err(Unavailable::NonPositiveContrast);
        }
        product *= term.powf(weight);
        if !last {
            let (nx, nw, nh) = downsample(&x, w, h, cfg.downsample);
            let (ny, _, _) = downsample(&y, w, h, cfg.downsample);
            x = nx;
            y = ny;
            w = nw;
            h = nh;
        }
    }
    Ok(product)
}

/// Normalised 1D Gaussian. The 2D kernel is the outer product, whose entries sum to 1
/// because each 1D kernel does; that is bit-for-bit `fspecial('gaussian', ws, sigma)`
/// up to the separable evaluation order.
fn gaussian_kernel_1d(size: usize, sigma: f64) -> Vec<f64> {
    assert!(size % 2 == 1, "window size must be odd");
    let centre = (size / 2) as f64;
    let mut k: Vec<f64> = (0..size)
        .map(|i| {
            let d = i as f64 - centre;
            (-(d * d) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum: f64 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Separable convolution, **valid** region only: output is `(w - size + 1) x (h - size + 1)`.
///
/// The kernel is symmetric, so convolution and correlation coincide and there is no
/// flip to get wrong.
fn filter_valid(src: &[f64], w: usize, h: usize, kernel: &[f64]) -> Vec<f64> {
    let k = kernel.len();
    debug_assert!(w >= k && h >= k);
    let ow = w - k + 1;
    let oh = h - k + 1;

    // Horizontal pass: full height, valid width.
    let mut tmp = vec![0.0; ow * h];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        for ox in 0..ow {
            let mut acc = 0.0;
            for (t, &kv) in kernel.iter().enumerate() {
                acc += kv * row[ox + t];
            }
            tmp[y * ow + ox] = acc;
        }
    }
    // Vertical pass: valid height.
    let mut out = vec![0.0; ow * oh];
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = 0.0;
            for (t, &kv) in kernel.iter().enumerate() {
                acc += kv * tmp[(oy + t) * ow + ox];
            }
            out[oy * ow + ox] = acc;
        }
    }
    out
}

fn downsample(src: &[f64], w: usize, h: usize, mode: Downsample) -> (Vec<f64>, usize, usize) {
    match mode {
        Downsample::Box2x2 => {
            let (ow, oh) = (w / 2, h / 2);
            let mut out = vec![0.0; ow * oh];
            for oy in 0..oh {
                for ox in 0..ow {
                    let (x0, y0) = (2 * ox, 2 * oy);
                    out[oy * ow + ox] = 0.25
                        * (src[y0 * w + x0]
                            + src[y0 * w + x0 + 1]
                            + src[(y0 + 1) * w + x0]
                            + src[(y0 + 1) * w + x0 + 1]);
                }
            }
            (out, ow, oh)
        }
        Downsample::ScipyUniform2 => {
            // uniform_filter(im, 2): out[i] = (in[i-1] + in[i]) / 2 with 'reflect'
            // edges, where reflect maps index -1 to index 0. Then [::2, ::2].
            let filt = |idx: isize| -> usize { idx.max(0) as usize };
            let mut full = vec![0.0; w * h];
            for y in 0..h {
                for x in 0..w {
                    let y0 = filt(y as isize - 1);
                    let x0 = filt(x as isize - 1);
                    full[y * w + x] = 0.25
                        * (src[y0 * w + x0] + src[y0 * w + x] + src[y * w + x0] + src[y * w + x]);
                }
            }
            let (ow, oh) = (w.div_ceil(2), h.div_ceil(2));
            let mut out = vec![0.0; ow * oh];
            for oy in 0..oh {
                for ox in 0..ow {
                    out[oy * ow + ox] = full[(2 * oy) * w + 2 * ox];
                }
            }
            (out, ow, oh)
        }
    }
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/// BT.601 full-range ("JPEG") Y, Cb, Cr planes, rounded half-away-from-zero.
///
/// Used only to report PSNR-Y and PSNR-YUV for colour input; the codec path is
/// grayscale until Step 18.
pub fn ycbcr(img: &Image) -> [Plane; 3] {
    match img.color() {
        ColorSpace::Gray => {
            let y = img.planes()[0].clone();
            let neutral = Plane::filled(y.width(), y.height(), 128);
            [y, neutral.clone(), neutral]
        }
        ColorSpace::Rgb => {
            let (rp, gp, bp) = (&img.planes()[0], &img.planes()[1], &img.planes()[2]);
            let n = rp.as_slice().len();
            let (mut y, mut cb, mut cr) = (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            );
            for i in 0..n {
                let r = f64::from(rp.as_slice()[i]);
                let g = f64::from(gp.as_slice()[i]);
                let b = f64::from(bp.as_slice()[i]);
                let yy = 0.299 * r + 0.587 * g + 0.114 * b;
                y.push(yy.round().clamp(0.0, 255.0) as u8);
                cb.push(
                    (128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                );
                cr.push(
                    (128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                );
            }
            let (w, h) = (rp.width(), rp.height());
            [
                Plane::from_vec(w, h, y),
                Plane::from_vec(w, h, cb),
                Plane::from_vec(w, h, cr),
            ]
        }
    }
}

/// Everything §M2 asks for, for one (original, decoded) pair.
#[derive(Debug, Clone, PartialEq)]
pub struct Quality {
    pub width: usize,
    pub height: usize,
    /// MSE per native plane (one entry for gray, R/G/B for colour).
    pub mse_per_plane: Vec<f64>,
    /// PSNR per native plane; `None` where the planes are identical.
    pub psnr_per_plane: Vec<Option<f64>>,
    pub psnr_y: Option<f64>,
    /// Present only for colour input.
    pub psnr_cb: Option<f64>,
    pub psnr_cr: Option<f64>,
    /// `(6*Y + Cb + Cr) / 8`. `None` if any component is infinite, i.e. if the images
    /// are identical; a weighted mean that silently drops an infinite term would read
    /// as a finite, wrong number.
    pub psnr_yuv: Option<f64>,
    /// SSIM on Y, or the reason it is unavailable.
    pub ssim: Result<f64, Unavailable>,
    pub ms_ssim: Result<f64, Unavailable>,
}

/// Compute every §M2 metric for a pair of images.
///
/// Both images are cropped to `region` first — §M2's "original W x H region only" rule.
/// Pass `None` to measure the full (already-matched) extent.
///
/// # Panics
/// If the images have different dimensions, or different colour spaces.
pub fn quality(
    original: &Image,
    decoded: &Image,
    region: Option<(usize, usize)>,
    ssim_cfg: &SsimConfig,
    ms_cfg: &MsSsimConfig,
) -> Quality {
    assert_eq!(
        original.color(),
        decoded.color(),
        "cannot compare images in different colour spaces"
    );
    let (w, h) = region.unwrap_or((
        original.width().min(decoded.width()),
        original.height().min(decoded.height()),
    ));
    let a = original.crop_top_left(w, h);
    let b = decoded.crop_top_left(w, h);

    let mse_per_plane: Vec<f64> = a
        .planes()
        .iter()
        .zip(b.planes())
        .map(|(p, q)| mse(p, q))
        .collect();
    let psnr_per_plane = mse_per_plane.iter().copied().map(psnr_from_mse).collect();

    let [ya, cba, cra] = ycbcr(&a);
    let [yb, cbb, crb] = ycbcr(&b);
    let psnr_y = psnr(&ya, &yb);
    let colour = a.color() == ColorSpace::Rgb;
    let psnr_cb = colour.then(|| psnr(&cba, &cbb)).flatten();
    let psnr_cr = colour.then(|| psnr(&cra, &crb)).flatten();
    let psnr_yuv = if colour {
        match (psnr_y, psnr_cb, psnr_cr) {
            (Some(y), Some(cb), Some(cr)) => Some((6.0 * y + cb + cr) / 8.0),
            _ => None,
        }
    } else {
        None
    };

    Quality {
        width: w,
        height: h,
        mse_per_plane,
        psnr_per_plane,
        psnr_y,
        psnr_cb,
        psnr_cr,
        psnr_yuv,
        ssim: ssim(&ya, &yb, ssim_cfg).map(|r| r.ssim),
        ms_ssim: ms_ssim(&ya, &yb, ms_cfg),
    }
}
