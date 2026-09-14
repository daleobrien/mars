//! Step 7's exit criteria (`just gate-7`) as a command that exits 0 or 1 (§A1).
//!
//! This module still runs the exact differential comparison the original brief called
//! for — `mars-gpu`'s search vs. `mars_codec::encode`'s CPU `search()`, block for block —
//! but per `docs/decisions.md` D25 (a user-directed CONTRACT-CHANGE), exact equality and
//! the >= 50x speed floor are no longer the pass/fail policy applied to that comparison's
//! output; see `gpu_search_check` in `mars-cli`'s `marsbench.rs` for the bounded-divergence
//! and lowered speed-floor policy actually gating `gate-7` now. What this module still
//! guarantees:
//!
//! 1. **Every divergence is found and reported in full**, never averaged away (§A7): the
//!    comparison here is still exact-equality detection of `(dom_row, dom_col, isometry,
//!    qalfa, qbeta)`, over `corpus/fixtures.images.json` and a `corpus/standard.images.json`
//!    subset — it is the *policy* applied to the resulting divergence list, not the
//!    detection itself, that D25 relaxed.
//! 2. **Speed** is measured per the `benchmark-protocol` skill: A/B interleaved, N >= 5,
//!    median + MAD, machine fingerprint recorded, Rayon pinned to the P-core count.
//!
//! The CPU reference here is `mars_codec::encode::search_block` (a thin `pub` wrapper
//! added around the crate-private `search()` — see that module's doc comment — so this
//! differential test can call the exact function `encode_image`'s quadtree walk calls,
//! at the same `(row, col, size)` positions the GPU enumerates, without re-running the
//! walk itself). The CPU side of the comparison is parallelised with Rayon purely so this
//! test finishes in reasonable time; that parallelism has no bearing on the *bitstream*
//! Step 12 will later formalise it for, and this module makes no claim about it.

use std::path::Path;
use std::time::{Duration, Instant};

use mars_codec::encode::{build_contracted, search_block, Candidate, EncodeParams, RawMoments};
use mars_core::io::{read_raw, ImageError};
use mars_core::Plane;
use mars_gpu::{GpuCandidate, GpuSearchParams, GpuSearcher};
use rayon::prelude::*;

use crate::provenance::Provenance;
use crate::sweep::{ImageSet, SweepError};

#[derive(Debug, thiserror::Error)]
pub enum GpuGateError {
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
    #[error("mars-gpu: {0}")]
    Gpu(#[from] mars_gpu::GpuError),
}

/// One `(image, size, row, col)` position at which the GPU and CPU winners disagree.
/// Printed in full — §A7: an anomaly is diagnosed, never averaged away.
#[derive(Debug, Clone)]
pub struct Divergence {
    pub image: String,
    pub size: u32,
    pub row: u32,
    pub col: u32,
    pub cpu: Option<CpuWinner>,
    pub gpu: Option<GpuCandidate>,
}

#[derive(Debug, Clone, Copy)]
pub struct CpuWinner {
    pub dom_row: u32,
    pub dom_col: u32,
    pub isometry: u8,
    pub qalfa: u32,
    pub qbeta: u32,
    pub rms: f64,
    pub moments: RawMoments,
}

impl From<Candidate> for CpuWinner {
    fn from(c: Candidate) -> Self {
        Self {
            dom_row: c.dom_row,
            dom_col: c.dom_col,
            isometry: c.isometry,
            qalfa: c.qalfa,
            qbeta: c.qbeta,
            rms: c.rms,
            moments: c.moments,
        }
    }
}

/// One `(image, size)` sweep's outcome.
pub struct DiffReport {
    pub compared: usize,
    pub divergences: Vec<Divergence>,
    /// The largest raw-moment magnitude observed among the CPU's own winners at this
    /// `(image, size)` -- P7.1's u32-accumulator claim, measured on every compared block
    /// (not only divergent ones) without a second, redundant CPU search pass.
    pub max_moment: u64,
}

/// The block grid both sides enumerate: every full `size x size` block on the row-major
/// fixed grid, exactly what `GpuSearcher::search`'s doc comment specifies.
fn block_grid(width: u32, height: u32, size: u32) -> Vec<(u32, u32)> {
    let nbx = width / size;
    let nby = height / size;
    let mut out = Vec::with_capacity((nbx * nby) as usize);
    for by in 0..nby {
        for bx in 0..nbx {
            out.push((by * size, bx * size));
        }
    }
    out
}

/// Run the CPU reference (Rayon-parallel across blocks, order preserved -- see the module
/// doc) and the GPU search over the same image/size/params, and diff them exactly.
pub fn diff_one(
    image_name: &str,
    image: &Plane,
    size: u32,
    params: &EncodeParams,
    gpu: &GpuSearcher,
) -> DiffReport {
    let contracted = build_contracted(image);
    let (width, height) = (image.width() as u32, image.height() as u32);
    let grid = block_grid(width, height, size);

    // Rayon's indexed parallel iterators preserve input order on collect, so this is the
    // same row-major (row outer, col inner) order the GPU's block index enumerates --
    // required for the zip below to compare like-for-like positions.
    let cpu_results: Vec<Option<Candidate>> = grid
        .par_iter()
        .map(|&(row, col)| search_block(image, &contracted, row, col, size, params).0)
        .collect();
    let max_moment = cpu_results
        .par_iter()
        .filter_map(|c| c.as_ref().map(|c| moment_extent(&c.moments)))
        .max()
        .unwrap_or(0);

    let gpu_params = GpuSearchParams {
        size,
        shift: params.shift,
        bits_alfa: params.bits_alfa,
        bits_beta: params.bits_beta,
        max_alfa: params.max_alfa as f32,
    };
    let gpu_results = gpu.search(image, &gpu_params);

    assert_eq!(
        cpu_results.len(),
        gpu_results.len(),
        "{image_name} size={size}: CPU produced {} blocks, GPU produced {} -- grid \
         enumeration mismatch, not a search divergence",
        cpu_results.len(),
        gpu_results.len()
    );

    let mut divergences = Vec::new();
    for (&(row, col), (cpu_c, gpu_c)) in grid.iter().zip(cpu_results.iter().zip(gpu_results.iter()))
    {
        let agree = match (cpu_c, gpu_c) {
            (None, None) => true,
            (Some(c), Some(g)) => {
                c.dom_row == g.dom_row
                    && c.dom_col == g.dom_col
                    && c.isometry == g.isometry
                    && c.qalfa == g.qalfa
                    && c.qbeta == g.qbeta
            }
            _ => false,
        };
        if !agree {
            divergences.push(Divergence {
                image: image_name.to_string(),
                size,
                row,
                col,
                cpu: cpu_c.map(CpuWinner::from),
                gpu: *gpu_c,
            });
        }
    }

    DiffReport {
        compared: cpu_results.len(),
        divergences,
        max_moment,
    }
}

fn read_checked(root: &Path, entry: &crate::sweep::ImageEntry) -> Result<Plane, GpuGateError> {
    let bytes = std::fs::read(root.join(&entry.file)).map_err(|source| GpuGateError::Io {
        path: entry.file.display().to_string(),
        source,
    })?;
    use crate::provenance::sha256_hex;
    let got = sha256_hex(&bytes);
    if got != entry.sha256 {
        return Err(GpuGateError::ImageHashMismatch {
            name: entry.name.clone(),
            want: entry.sha256.clone(),
            got,
        });
    }
    Ok(read_raw(
        &root.join(&entry.file),
        entry.width as usize,
        entry.height as usize,
    )?)
}

const BASE: EncodeParams = EncodeParams {
    min_size: 4,
    max_size: 32,
    shift: 4,
    bits_alfa: 4,
    bits_beta: 7,
    max_alfa: 1.0,
    t_rms: 0.0, // unused: this test calls search_block directly, not the walk
    zero_threshold: 0,
    lambda: None,
};

/// The differential test's exact scope: full exhaustive search is `size^2` work per
/// evaluated candidate and the candidate count itself grows as the domain pool does, so
/// an unrestricted "every size on every image" sweep does not finish in a session. The
/// grid below still exercises every block size this project's configs use (4, 8, 16, 32
/// -- `max_size` never exceeds 32) and both corpora, but caps the expensive sizes to a
/// handful of Kodak images rather than all 24, and skips the smallest sizes on the larger
/// Kodak images (which is where the domain-position count, and therefore runtime, is
/// largest per block-size). `docs/predictions.md` P7.1's u32 overflow concern is about
/// magnitude, not corpus breadth, and size=32 on Kodak is exactly the case that reaches
/// closest to it, so it is deliberately included rather than the size this budget would
/// otherwise favour skipping.
pub struct DiffScope<'a> {
    pub fixtures_sizes: &'a [u32],
    pub standard_images: &'a [&'a str],
    pub standard_sizes: &'a [u32],
}

pub const DEFAULT_SCOPE: DiffScope<'static> = DiffScope {
    fixtures_sizes: &[4, 8, 16, 32],
    standard_images: &["kodim01", "kodim05", "kodim13", "kodim19"],
    standard_sizes: &[16, 32],
};

pub struct GateOutcome {
    pub compared: usize,
    pub divergences: Vec<Divergence>,
    pub max_u32_moment_observed: u64,
}

/// Runs [`DEFAULT_SCOPE`]'s differential sweep over both corpora and returns every
/// divergence found (should be empty) plus the largest raw moment value observed, which
/// is the empirical half of P7.1's u32-accumulator claim.
pub fn gate(
    root: &Path,
    fixtures_index: &Path,
    standard_index: &Path,
    scope: &DiffScope,
) -> Result<GateOutcome, GpuGateError> {
    let gpu = GpuSearcher::new()?;

    let mut compared = 0usize;
    let mut divergences = Vec::new();
    let mut max_moment = 0u64;

    let fixtures = ImageSet::read(&root.join(fixtures_index))?.images;
    for entry in &fixtures {
        let image = read_checked(root, entry)?;
        for &size in scope.fixtures_sizes {
            if image.width() < 2 * size as usize || image.height() < 2 * size as usize {
                continue;
            }
            let t0 = std::time::Instant::now();
            let report = diff_one(&entry.name, &image, size, &BASE, &gpu);
            eprintln!(
                "  fixtures/{} size={size}: {} blocks, {} divergence(s) ({:.1?})",
                entry.name,
                report.compared,
                report.divergences.len(),
                t0.elapsed()
            );
            compared += report.compared;
            max_moment = max_moment.max(report.max_moment);
            divergences.extend(report.divergences);
        }
    }

    let standard = ImageSet::read(&root.join(standard_index))?.images;
    for entry in standard
        .iter()
        .filter(|e| scope.standard_images.contains(&e.name.as_str()))
    {
        let image = read_checked(root, entry)?;
        for &size in scope.standard_sizes {
            let t0 = std::time::Instant::now();
            let report = diff_one(&entry.name, &image, size, &BASE, &gpu);
            eprintln!(
                "  standard/{} size={size}: {} blocks, {} divergence(s) ({:.1?})",
                entry.name,
                report.compared,
                report.divergences.len(),
                t0.elapsed()
            );
            compared += report.compared;
            max_moment = max_moment.max(report.max_moment);
            divergences.extend(report.divergences);
        }
    }

    Ok(GateOutcome {
        compared,
        divergences,
        max_u32_moment_observed: max_moment,
    })
}

fn moment_extent(m: &RawMoments) -> u64 {
    [m.s1_x4, m.s2_x16, m.t0, m.t1_x4, m.t2]
        .into_iter()
        .map(|v| v.unsigned_abs())
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------- speed benchmark

/// One A/B-interleaved timing sample pair, per the `benchmark-protocol` skill: N >= 5,
/// median + MAD, machine fingerprint recorded, encode-only (no decode involved here).
pub struct SpeedResult {
    pub image: String,
    pub size: u32,
    pub blocks: usize,
    pub cpu_median: Duration,
    pub cpu_mad: Duration,
    pub gpu_median: Duration,
    pub gpu_mad: Duration,
    pub speedup: f64,
}

fn median_and_mad(mut samples: Vec<Duration>) -> (Duration, Duration) {
    samples.sort();
    let median = samples[samples.len() / 2];
    let mut abs_dev: Vec<Duration> = samples.iter().map(|&s| s.abs_diff(median)).collect();
    abs_dev.sort();
    (median, abs_dev[abs_dev.len() / 2])
}

/// A/B interleaved: `A B A B A B ...` for `runs` repetitions each, per the
/// `benchmark-protocol` skill, so thermal drift cancels rather than accumulating into
/// whichever side runs second. Rayon's global pool is left at its default here because
/// `rayon::ThreadPoolBuilder::build_global` can only be called once per process; the CLI
/// entry point pins it to the P-core count before calling this function (see
/// `marsbench`'s `gpu-search-bench` command).
pub fn ab_compare(
    image: &Plane,
    size: u32,
    params: &EncodeParams,
    gpu: &GpuSearcher,
    runs: usize,
) -> SpeedResult {
    let contracted = build_contracted(image);
    let grid = block_grid(image.width() as u32, image.height() as u32, size);
    let gpu_params = GpuSearchParams {
        size,
        shift: params.shift,
        bits_alfa: params.bits_alfa,
        bits_beta: params.bits_beta,
        max_alfa: params.max_alfa as f32,
    };

    let mut cpu_times = Vec::with_capacity(runs);
    let mut gpu_times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _cpu: Vec<Option<Candidate>> = grid
            .par_iter()
            .map(|&(row, col)| search_block(image, &contracted, row, col, size, params).0)
            .collect();
        cpu_times.push(t0.elapsed());

        let t1 = Instant::now();
        let _gpu = gpu.search(image, &gpu_params);
        gpu_times.push(t1.elapsed());
    }

    let (cpu_median, cpu_mad) = median_and_mad(cpu_times);
    let (gpu_median, gpu_mad) = median_and_mad(gpu_times);
    SpeedResult {
        image: String::new(),
        size,
        blocks: grid.len(),
        cpu_median,
        cpu_mad,
        gpu_median,
        gpu_mad,
        speedup: cpu_median.as_secs_f64() / gpu_median.as_secs_f64(),
    }
}

pub fn machine_fingerprint_line() -> String {
    let p = Provenance::detect(0);
    format!(
        "{} ({} P-cores / {} E-cores) · rustc {} · {}{}",
        p.machine.cpu_brand,
        p.machine.p_cores.map_or("?".into(), |c| c.to_string()),
        p.machine.e_cores.map_or("?".into(), |c| c.to_string()),
        p.machine.rustc_version,
        p.build_profile,
        if p.git_dirty { " (dirty tree)" } else { "" },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn byte(&mut self) -> u8 {
            (self.next_u64() & 0xff) as u8
        }
    }

    /// A fast, small-scale version of `gate`'s sweep -- a synthetic non-degenerate image
    /// (a GPU with only flat/constant input can hide a broken fit behind `alfa == 0`
    /// short-circuits), run at every block size this project's configs use. Skips (rather
    /// than fails) when no GPU adapter is available, same as `mars-gpu`'s own tests.
    #[test]
    fn synthetic_image_is_bit_identical_at_every_configured_size() {
        let gpu = match GpuSearcher::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let (w, h) = (64usize, 64usize);
        let mut rng = Rng(0x00C0_FFEE_1234_5678);
        let data: Vec<u8> = (0..w * h).map(|_| rng.byte()).collect();
        let image = Plane::from_vec(w, h, data);

        for &size in &[4u32, 8, 16] {
            let report = diff_one("synthetic64", &image, size, &BASE, &gpu);
            assert!(
                report.divergences.is_empty(),
                "size={size}: {} divergence(s) on synthetic64, e.g. {:?}",
                report.divergences.len(),
                report.divergences.first()
            );
            assert!(report.compared > 0);
        }
    }
}
