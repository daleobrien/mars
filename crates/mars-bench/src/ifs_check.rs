//! Step 5's exit criteria (`just gate-5`) as a command that exits 0 or 1 (§A1).
//!
//! Two checks, run over every golden `.ifs` fixture from Step 3:
//!
//! 1. `mars_codec::ifs::parse` recovers exactly the transform count `encmars` printed
//!    (`fixtures/mars1/manifest.toml`'s `transforms`, an exact integer equality).
//! 2. A Rust iterative decode (`mars_codec::ifs::decode_iterative`) agrees with
//!    `decmars -i` to within 0.1 dB — measured the way every other RD number in this
//!    project is measured, as PSNR against the original image, not as a direct
//!    pixel-difference between the two decodes. A gap here that check 1 didn't already
//!    explain means the reader parsed the right transforms into the wrong arithmetic.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use mars_codec::ifs;
use mars_core::io::{read_pgm, read_raw, ImageError};
use mars_core::metrics::psnr;

use crate::mars1::{decode, DecodeMode, DecodeParams, Mars1Binaries, Mars1Error};
use crate::mars1_fixtures::{FixtureConfig, FixtureError, FIXTURE_DIR};
use crate::mars1_report::Check;
use crate::sweep::ImageEntry;

pub const DECODE_ITERATIONS: u32 = 10;
pub const DECODE_TOLERANCE_DB: f64 = 0.1;

#[derive(Debug, Deserialize)]
struct Manifest {
    #[serde(default)]
    fixture: Vec<ManifestEntry>,
}

/// Only the fields gate-5 needs. `manifest.toml` carries more (method, params, hashes);
/// serde ignores the rest by default.
#[derive(Debug, Deserialize)]
struct ManifestEntry {
    stem: String,
    image: String,
    width: u32,
    height: u32,
    transforms: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum IfsCheckError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot parse {path}: {source}")]
    Manifest {
        path: String,
        #[source]
        source: toml::de::Error,
    },
    #[error("{path}: {source}")]
    Ifs {
        path: String,
        #[source]
        source: ifs::IfsError,
    },
    #[error("fixture {stem:?} names image {image:?}, which is in no image index")]
    UnknownImage { stem: String, image: String },
    #[error(transparent)]
    Fixture(#[from] FixtureError),
    #[error(transparent)]
    ImageIo(#[from] ImageError),
    #[error(transparent)]
    Mars1(#[from] Mars1Error),
}

fn read(path: &Path) -> Result<Vec<u8>, IfsCheckError> {
    std::fs::read(path).map_err(|source| IfsCheckError::Io {
        path: path.display().to_string(),
        source,
    })
}

/// Run both gate-5 checks over every fixture in `manifest_path` and return one `Check` per
/// criterion, in the style of `mars1_report::gate` and `anchors_report`'s gate.
pub fn gate(
    root: &Path,
    manifest_path: &Path,
    fixture_config: &Path,
    mars1_dir: &Path,
    scratch: &Path,
) -> Result<Vec<Check>, IfsCheckError> {
    let manifest_text =
        std::fs::read_to_string(manifest_path).map_err(|source| IfsCheckError::Io {
            path: manifest_path.display().to_string(),
            source,
        })?;
    let manifest: Manifest =
        toml::from_str(&manifest_text).map_err(|source| IfsCheckError::Manifest {
            path: manifest_path.display().to_string(),
            source,
        })?;

    let config = FixtureConfig::read(fixture_config)?;
    let images: BTreeMap<String, ImageEntry> = config
        .image_index(root)?
        .into_iter()
        .map(|i| (i.name.clone(), i))
        .collect();

    let bins = Mars1Binaries::from_dir(mars1_dir)?;
    std::fs::create_dir_all(scratch).map_err(|source| IfsCheckError::Io {
        path: scratch.display().to_string(),
        source,
    })?;

    let mut transform_mismatches: Vec<String> = Vec::new();
    let mut decode_gaps: Vec<(String, f64)> = Vec::new();
    let mut max_gap = 0.0f64;

    for entry in &manifest.fixture {
        let ifs_path = root.join(FIXTURE_DIR).join(format!("{}.ifs", entry.stem));
        let bytes = read(&ifs_path)?;
        let (hdr, leaves) = ifs::parse(&bytes).map_err(|source| IfsCheckError::Ifs {
            path: ifs_path.display().to_string(),
            source,
        })?;

        if leaves.len() as u64 != entry.transforms {
            transform_mismatches.push(format!(
                "{}: parsed {} leaves, encmars printed {}",
                entry.stem,
                leaves.len(),
                entry.transforms
            ));
            continue; // a wrong transform count makes the decode check meaningless here
        }

        let image = images
            .get(&entry.image)
            .ok_or_else(|| IfsCheckError::UnknownImage {
                stem: entry.stem.clone(),
                image: entry.image.clone(),
            })?;
        let ground_truth = read_raw(
            &root.join(&image.file),
            image.width as usize,
            image.height as usize,
        )?;

        let rust_decode = ifs::decode_iterative(&hdr, &leaves, DECODE_ITERATIONS);

        let workdir = scratch.join(&entry.stem);
        std::fs::create_dir_all(&workdir).map_err(|source| IfsCheckError::Io {
            path: workdir.display().to_string(),
            source,
        })?;
        std::fs::copy(&ifs_path, workdir.join("o.ifs")).map_err(|source| IfsCheckError::Io {
            path: workdir.display().to_string(),
            source,
        })?;
        decode(
            &bins,
            &workdir,
            "o.ifs",
            "d_it.pgm",
            (entry.width, entry.height),
            &DecodeParams {
                mode: DecodeMode::Iterative,
                iterations: None,
                postprocess: false,
            },
        )?;
        let c_decode = read_pgm(&workdir.join("d_it.pgm"))?;

        let psnr_rust = psnr(&ground_truth, &rust_decode).unwrap_or(f64::INFINITY);
        let psnr_c = psnr(&ground_truth, &c_decode).unwrap_or(f64::INFINITY);
        let gap = (psnr_rust - psnr_c).abs();
        max_gap = max_gap.max(gap);
        if gap > DECODE_TOLERANCE_DB {
            decode_gaps.push((entry.stem.clone(), gap));
        }
    }

    let total = manifest.fixture.len();
    Ok(vec![
        Check {
            name: "every golden .ifs's parsed transform count matches encmars exactly"
                .into(),
            passed: transform_mismatches.is_empty(),
            detail: if transform_mismatches.is_empty() {
                format!("{total}/{total} fixtures")
            } else {
                format!(
                    "{} of {total} mismatched: {}",
                    transform_mismatches.len(),
                    transform_mismatches.join("; ")
                )
            },
        },
        Check {
            name: format!(
                "Rust iterative decode is within {DECODE_TOLERANCE_DB} dB of decmars -i on every fixture"
            ),
            passed: decode_gaps.is_empty(),
            detail: if decode_gaps.is_empty() {
                format!("{total}/{total} fixtures, max |ΔPSNR| = {max_gap:.4} dB")
            } else {
                format!(
                    "{} of {total} exceeded {DECODE_TOLERANCE_DB} dB: {}",
                    decode_gaps.len(),
                    decode_gaps
                        .iter()
                        .map(|(s, g)| format!("{s} ({g:.4} dB)"))
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            },
        },
    ])
}
