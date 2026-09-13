//! Measuring one (original, decoded) file pair — §M1's "two files on disk" in code.

use std::path::{Path, PathBuf};

use mars_core::image::Image;
use mars_core::io::read_image;
use mars_core::metrics::{quality, MsSsimConfig, Quality, SsimConfig, Unavailable, MSSSIM_MIN_DIM};
use serde::{Deserialize, Serialize};

/// How to read a pair of files. Raw input has no header, so the dimensions travel with
/// the request rather than being inferred.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureRequest {
    pub original: PathBuf,
    pub decoded: PathBuf,
    /// Dimensions for headerless raw inputs.
    pub raw_dims: Option<(usize, usize)>,
    /// §M2: measure the original `W x H` region only. Mars 1 pads to the next power of
    /// two and that padding must never be measured, so when the decoded file is the
    /// padded `virtual_size` this is the original extent.
    pub region: Option<(usize, usize)>,
    /// Bitstream size for bpp. `None` means bpp is not reported for this pair.
    pub coded_bytes: Option<u64>,
    pub ssim: SsimConfig,
    pub ms_ssim: MsSsimConfig,
}

impl MeasureRequest {
    pub fn new(original: impl Into<PathBuf>, decoded: impl Into<PathBuf>) -> Self {
        Self {
            original: original.into(),
            decoded: decoded.into(),
            raw_dims: None,
            region: None,
            coded_bytes: None,
            ssim: SsimConfig::default(),
            ms_ssim: MsSsimConfig::default(),
        }
    }

    pub fn with_raw_dims(mut self, w: usize, h: usize) -> Self {
        self.raw_dims = Some((w, h));
        self
    }

    pub fn with_region(mut self, w: usize, h: usize) -> Self {
        self.region = Some((w, h));
        self
    }

    pub fn with_coded_bytes(mut self, bytes: u64) -> Self {
        self.coded_bytes = Some(bytes);
        self
    }
}

/// A measurement, shaped for JSON. Infinite PSNR is `null` per §M2, and an unavailable
/// structural metric carries the reason rather than a substitute value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub original: String,
    pub decoded: String,
    pub width: usize,
    pub height: usize,
    pub planes: usize,
    pub mse: Vec<f64>,
    pub psnr: Vec<Option<f64>>,
    pub psnr_y: Option<f64>,
    pub psnr_cb: Option<f64>,
    pub psnr_cr: Option<f64>,
    pub psnr_yuv: Option<f64>,
    pub ssim: Option<f64>,
    pub ssim_unavailable: Option<String>,
    pub ms_ssim: Option<f64>,
    pub ms_ssim_unavailable: Option<String>,
    pub coded_bytes: Option<u64>,
    pub bpp: Option<f64>,
    /// Pinned metric definitions, recorded inline so a row is self-describing.
    pub definitions: Definitions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Definitions {
    pub psnr: String,
    pub ssim: String,
    pub ms_ssim: String,
    pub bpp: String,
}

fn reason(u: Unavailable) -> String {
    match u {
        Unavailable::TooSmall => {
            format!("image smaller than the metric window (MS-SSIM needs min dimension >= {MSSSIM_MIN_DIM})")
        }
        Unavailable::NonPositiveContrast => {
            "contrast-structure term was non-positive; weighted geometric mean is not real".into()
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MeasureError {
    #[error(transparent)]
    Io(#[from] mars_core::io::ImageError),
    #[error("{orig_path} is {ow}x{oh} but {dec_path} is {dw}x{dh}; refusing to compare (set a region explicitly if the decode is padded)")]
    SizeMismatch {
        orig_path: String,
        dec_path: String,
        ow: usize,
        oh: usize,
        dw: usize,
        dh: usize,
    },
    #[error("cannot stat {path}: {source}")]
    Stat {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Read both files and compute every §M2 metric.
pub fn measure(req: &MeasureRequest) -> Result<Measurement, MeasureError> {
    let original = read_image(&req.original, req.raw_dims)?;
    let decoded = read_image(&req.decoded, req.raw_dims)?;

    let region = match req.region {
        Some(r) => r,
        None => {
            // Without an explicit region, differing sizes are a harness bug. §M2 allows
            // measuring a sub-region of a padded decode, but only when asked for.
            if original.width() != decoded.width() || original.height() != decoded.height() {
                return Err(MeasureError::SizeMismatch {
                    orig_path: req.original.display().to_string(),
                    dec_path: req.decoded.display().to_string(),
                    ow: original.width(),
                    oh: original.height(),
                    dw: decoded.width(),
                    dh: decoded.height(),
                });
            }
            (original.width(), original.height())
        }
    };

    let q = quality(&original, &decoded, Some(region), &req.ssim, &req.ms_ssim);
    let bpp = req
        .coded_bytes
        .map(|b| mars_core::metrics::bpp(b, q.width, q.height));

    Ok(Measurement {
        original: req.original.display().to_string(),
        decoded: req.decoded.display().to_string(),
        width: q.width,
        height: q.height,
        planes: original.planes().len(),
        mse: q.mse_per_plane.clone(),
        psnr: q.psnr_per_plane.clone(),
        psnr_y: q.psnr_y,
        psnr_cb: q.psnr_cb,
        psnr_cr: q.psnr_cr,
        psnr_yuv: q.psnr_yuv,
        ssim: q.ssim.ok(),
        ssim_unavailable: q.ssim.err().map(reason),
        ms_ssim: q.ms_ssim.ok(),
        ms_ssim_unavailable: q.ms_ssim.err().map(reason),
        coded_bytes: req.coded_bytes,
        bpp,
        definitions: definitions(&req.ssim, &req.ms_ssim),
    })
}

pub fn definitions(ssim: &SsimConfig, ms: &MsSsimConfig) -> Definitions {
    Definitions {
        psnr: "10*log10(255^2/MSE); MSE over the stated region only; null for identical images"
            .into(),
        ssim: format!(
            "{w}x{w} Gaussian sigma={s}, K1={k1}, K2={k2}, valid region, population covariance, on Y",
            w = ssim.window,
            s = ssim.sigma,
            k1 = ssim.k1,
            k2 = ssim.k2
        ),
        ms_ssim: format!(
            "{n} scales, weights {wts:?}, downsample={ds:?}",
            n = ms.weights.len(),
            wts = ms.weights,
            ds = ms.downsample
        ),
        bpp: "8 * whole_file_bytes / (W*H)".into(),
    }
}

/// File size in bytes, for bpp.
pub fn file_size(path: &Path) -> Result<u64, MeasureError> {
    std::fs::metadata(path)
        .map(|m| m.len())
        .map_err(|source| MeasureError::Stat {
            path: path.display().to_string(),
            source,
        })
}

/// Re-exported so callers can compute quality on in-memory images without a round trip
/// through the filesystem (used by tests that synthesise input).
pub fn quality_of(original: &Image, decoded: &Image, region: Option<(usize, usize)>) -> Quality {
    quality(
        original,
        decoded,
        region,
        &SsimConfig::default(),
        &MsSsimConfig::default(),
    )
}
