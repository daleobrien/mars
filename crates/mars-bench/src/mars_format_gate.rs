//! Step 10's exit criteria (`just gate-10`) as a command that exits 0 or 1 (§A1).
//!
//! Over every fixture in `corpus/fixtures.images.json`, at Step 2's 1998-default
//! settings (`min_size=4, max_size=16, shift=4, bits_alfa=4, bits_beta=7, max_alfa=1.0`)
//! and the same rms grid Step 6's own gate swept ([`crate::rust_encoder::RMS_GRID`]):
//!
//! 1. **Lossless round-trip.** `mars_format::write` -> `mars_format::read` recovers
//!    exactly the leaf list Step 6's encoder produced (compared as a sorted set, since
//!    leaf order is not semantically meaningful). Because `mars_format` never touches
//!    pixel reconstruction — it only serialises `(Header, Vec<Leaf>)` — an exact leaf-list
//!    match makes the decoded image byte-identical to Step 6's own decode of the same
//!    leaves, satisfying the brief's "reconstruction must be byte-identical" requirement
//!    without a second, redundant pixel-level decode.
//! 2. **Attribution.** bpp of the entropy-coded `.mars` bytes vs. the raw `.ifs` bytes for
//!    the *same* leaves — a pure rate comparison at identical reconstruction, per the
//!    brief's "report bpp reduction from entropy coding alone."

use std::path::Path;

use mars_codec::encode::{encode_image, EncodeParams};
use mars_codec::ifs;
use mars_codec::mars_format::{self, MarsFormatError};
use mars_core::io::{read_raw, ImageError};
use mars_core::metrics::bpp;

use crate::mars1_report::Check;
use crate::provenance::sha256_hex;
use crate::rust_encoder::RMS_GRID;
use crate::sweep::{ImageSet, SweepError};

/// §8's 1998 defaults, matching `rust_encoder::gate`'s `BASE` — the same settings Step
/// 6's own gate measured, so this gate's bpp comparison is apples to apples with it.
const BASE: EncodeParams = EncodeParams {
    min_size: 4,
    max_size: 16,
    shift: 4,
    bits_alfa: 4,
    bits_beta: 7,
    max_alfa: 1.0,
    t_rms: 0.0, // overwritten per rate
    zero_threshold: 0,
    lambda: None,
};

#[derive(Debug, thiserror::Error)]
pub enum GateError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{name}: expected sha256 {want}, found {got}")]
    ImageHashMismatch {
        name: String,
        want: String,
        got: String,
    },
    #[error(transparent)]
    Sweep(#[from] SweepError),
    #[error(transparent)]
    ImageIo(#[from] ImageError),
    #[error("{image} r={t_rms}: {source}")]
    Format {
        image: String,
        t_rms: f64,
        #[source]
        source: MarsFormatError,
    },
}

/// One (image, rms) point's rate comparison, holding distortion exactly constant.
#[derive(Debug, Clone)]
pub struct BppRow {
    pub image: String,
    pub t_rms: f64,
    pub raw_bpp: f64,
    pub mars_bpp: f64,
}

impl BppRow {
    pub fn reduction_pct(&self) -> f64 {
        100.0 * (self.raw_bpp - self.mars_bpp) / self.raw_bpp
    }
}

pub fn gate(root: &Path, fixtures_index: &Path) -> Result<(Vec<Check>, Vec<BppRow>), GateError> {
    let images = ImageSet::read(&root.join(fixtures_index))?.images;

    let mut mismatches: Vec<String> = Vec::new();
    let mut rows = Vec::new();

    for image in &images {
        let path = root.join(&image.file);
        let bytes = std::fs::read(&path).map_err(|source| GateError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let got_sha256 = sha256_hex(&bytes);
        if got_sha256 != image.sha256 {
            return Err(GateError::ImageHashMismatch {
                name: image.name.clone(),
                want: image.sha256.clone(),
                got: got_sha256,
            });
        }
        let ground_truth = read_raw(&path, image.width as usize, image.height as usize)?;

        for &t_rms in &RMS_GRID {
            let params = EncodeParams { t_rms, ..BASE };
            let (hdr, leaves, _evals) = encode_image(&ground_truth, &params);

            let raw_bytes = ifs::write(&hdr, &leaves).unwrap_or_else(|e| {
                panic!("{}: raw ifs write failed at rms {t_rms}: {e}", image.name)
            });
            let mars_bytes =
                mars_format::write(&hdr, &leaves).map_err(|source| GateError::Format {
                    image: image.name.clone(),
                    t_rms,
                    source,
                })?;
            let (hdr2, mut leaves2) =
                mars_format::read(&mars_bytes).map_err(|source| GateError::Format {
                    image: image.name.clone(),
                    t_rms,
                    source,
                })?;

            let mut leaves_sorted = leaves.clone();
            leaves_sorted.sort_by_key(|l| (l.row, l.col, l.size));
            leaves2.sort_by_key(|l| (l.row, l.col, l.size));
            if hdr != hdr2 || leaves_sorted != leaves2 {
                mismatches.push(format!("{} r={t_rms}", image.name));
            }

            rows.push(BppRow {
                image: image.name.clone(),
                t_rms,
                raw_bpp: bpp(
                    raw_bytes.len() as u64,
                    image.width as usize,
                    image.height as usize,
                ),
                mars_bpp: bpp(
                    mars_bytes.len() as u64,
                    image.width as usize,
                    image.height as usize,
                ),
            });
        }
    }

    let mean_reduction = if rows.is_empty() {
        0.0
    } else {
        rows.iter().map(BppRow::reduction_pct).sum::<f64>() / rows.len() as f64
    };

    let checks = vec![
        Check {
            name: "`.mars` v0 round-trips every fixture losslessly (write -> read recovers \
                   the exact leaf list Step 6's encoder produced)"
                .into(),
            passed: mismatches.is_empty(),
            detail: if mismatches.is_empty() {
                format!("{} (image, rms) points round-tripped exactly", rows.len())
            } else {
                format!("{} mismatched: {}", mismatches.len(), mismatches.join("; "))
            },
        },
        Check {
            name: "entropy coding reduces bpp vs. the raw `.ifs` bitstream at identical \
                   reconstruction (attribution: same leaves, only the serialisation differs)"
                .into(),
            passed: mean_reduction > 0.0,
            detail: format!(
                "mean reduction {mean_reduction:.1}% across {} points (brief expects \
                 8-20% from split flags and domain deltas alone)",
                rows.len()
            ),
        },
    ];

    Ok((checks, rows))
}
