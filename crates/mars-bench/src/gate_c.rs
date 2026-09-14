//! Gate C — CPU encode speedup vs. Mars 1 at matched RD (`implementation-plan.md`,
//! `just gate-c`), GPU excluded per the brief.
//!
//! The apples-to-apples comparison is Mars 2's Fisher-method encode
//! (`mars_search::encode_image`) against `encmars -M f`: identical `EncodeParams` (the
//! 1998 defaults both sides already use — `mars1::EncodeParams::default()` and this
//! project's own `default` baseline-mars1 variant agree on min/max 4/16, shift 4, bits
//! 4/7, `max_alfa` 1.0), so both sides run the *same classified search* over the *same*
//! candidate set — the same algorithm Mars 1's own Fisher method restricts to, not
//! `mars_codec`'s exhaustive search (which does far more work per block and would not be
//! a "matched" comparison of search cost). `mars_search::walk` only gained Rayon
//! parallelism this session (porting Step 12's split/rayon::join pattern, previously only
//! in `mars_codec::encode`) — see `parallel_determinism` (mars-search) for the
//! determinism check that makes timing it at multiple thread counts meaningful at all.
//!
//! Threads' contribution is attributed directly (1-thread vs. P-core-thread Mars 2 runs,
//! same-implementation comparison, not A/B interleaved against Mars 1 — mirrors Step 12's
//! own `parallel_bench`). NEON's contribution is *not* re-ablated here: the kernels have
//! no runtime scalar/NEON switch (`implementation-plan.md` §M4 — aarch64 gives NEON
//! unconditionally, `#[target_feature(enable = "neon")]` compiled in), so this module
//! cites Step 11's own per-kernel measurement (D31) instead of fabricating a scalar build.

use std::path::Path;
use std::time::{Duration, Instant};

use mars_codec::encode::build_contracted;
use mars_codec::ifs::{decode_iterative, write as ifs_write};
use mars_core::io::{read_pgm, read_raw, ImageError};
use mars_core::metrics::{bpp, psnr};
use mars_search::fisher::Fisher;
use mars_search::SizedRetrievers;

use crate::mars1::{
    decode as mars1_decode, encode as mars1_encode, DecodeMode, DecodeParams,
    EncodeParams as Mars1EncodeParams, Mars1Binaries, Mars1Error,
};

#[derive(Debug, thiserror::Error)]
pub enum GateCError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    Mars1(#[from] Mars1Error),
    #[error(transparent)]
    ImageIo(#[from] ImageError),
}

fn median_and_mad(mut samples: Vec<Duration>) -> (Duration, Duration) {
    samples.sort();
    let median = samples[samples.len() / 2];
    let mut dev: Vec<Duration> = samples.iter().map(|&s| s.abs_diff(median)).collect();
    dev.sort();
    (median, dev[dev.len() / 2])
}

#[derive(Debug, Clone)]
pub struct GateCPoint {
    pub image: String,
    pub mars1_encode_median: Duration,
    pub mars1_encode_mad: Duration,
    pub mars2_encode_median: Duration,
    pub mars2_encode_mad: Duration,
    pub mars2_1thread_median: Duration,
    pub mars2_1thread_mad: Duration,
    /// `mars1_encode_median / mars2_encode_median`, full-thread Mars 2 vs. Mars 1.
    pub encode_speedup: f64,
    /// `mars2_1thread_median / mars2_encode_median` — threads' own contribution, isolated
    /// from the Rust/NEON implementation factor.
    pub thread_speedup: f64,
    pub mars1_bpp: f64,
    pub mars2_bpp: f64,
    pub mars1_psnr: f64,
    pub mars2_psnr: f64,
    pub mars1_decode_seconds: f64,
    pub mars2_decode_median: Duration,
}

/// Run one image through both encoders, A/B interleaved (`runs` of each, alternating),
/// per the `benchmark-protocol` skill. `full_threads` should be the machine's P-core
/// count (§M4); `workdir` is scratch, not committed.
#[allow(clippy::too_many_arguments)]
pub fn run_one(
    bins: &Mars1Binaries,
    workdir: &Path,
    image_path: &Path,
    width: u32,
    height: u32,
    runs: usize,
    full_threads: usize,
) -> Result<GateCPoint, GateCError> {
    std::fs::create_dir_all(workdir).map_err(|source| GateCError::Io {
        path: workdir.display().to_string(),
        source,
    })?;
    let input = workdir.join("i.raw");
    std::fs::copy(image_path, &input).map_err(|source| GateCError::Io {
        path: image_path.display().to_string(),
        source,
    })?;

    let plane = read_raw(image_path, width as usize, height as usize)?;
    let contracted = build_contracted(&plane);
    let m1_params = Mars1EncodeParams::default();
    let mars2_params = mars_codec::encode::EncodeParams {
        min_size: m1_params.min_size,
        max_size: m1_params.max_size,
        shift: m1_params.shift,
        bits_alfa: m1_params.bits_alfa,
        bits_beta: m1_params.bits_beta,
        max_alfa: m1_params.max_alfa,
        t_rms: m1_params.t_rms,
        zero_threshold: 0,
    };
    let build_retrievers = || {
        SizedRetrievers::build(
            &contracted,
            width,
            height,
            mars2_params.shift,
            mars2_params.min_size,
            mars2_params.max_size,
            || Box::new(Fisher::default()),
        )
    };

    let full_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(full_threads)
        .build()
        .expect("building the full-thread pool");
    let one_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("building the 1-thread pool");

    // A/B interleaved encode timing.
    let mut mars1_samples = Vec::with_capacity(runs);
    let mut mars2_samples = Vec::with_capacity(runs);
    let mut last_mars1_ifs_name = String::new();
    let mut last_mars2_hdr = None;
    let mut last_mars2_leaves = None;
    for i in 0..runs {
        let out_name = format!("o{i}.ifs");
        let start = Instant::now();
        mars1_encode(
            bins,
            workdir,
            "i.raw",
            &out_name,
            width,
            height,
            &m1_params,
            &[],
        )?;
        mars1_samples.push(start.elapsed());
        last_mars1_ifs_name = out_name;

        let retrievers = build_retrievers();
        let start = Instant::now();
        let (hdr, leaves, _evals, _picks) = full_pool.install(|| {
            std::hint::black_box(mars_search::encode_image(
                std::hint::black_box(&plane),
                &mars2_params,
                &retrievers,
            ))
        });
        mars2_samples.push(start.elapsed());
        last_mars2_hdr = Some(hdr);
        last_mars2_leaves = Some(leaves);
    }

    // 1-thread ablation: same implementation, not interleaved against Mars 1 (Step 12's
    // `parallel_bench` precedent — there is only one Mars-2 code path here to compare
    // across thread counts, not two implementations to alternate between).
    let mut mars2_1t_samples = Vec::with_capacity(runs);
    mars_search::diag::reset();
    for _ in 0..runs {
        let retrievers = build_retrievers();
        let start = Instant::now();
        one_pool.install(|| {
            std::hint::black_box(mars_search::encode_image(
                std::hint::black_box(&plane),
                &mars2_params,
                &retrievers,
            ))
        });
        mars2_1t_samples.push(start.elapsed());
    }
    if mars_search::diag::enabled() {
        eprintln!(
            "  [MARS_SEARCH_TIMING] 1-thread, {runs} runs summed: {}",
            mars_search::diag::summary()
        );
    }

    let (mars1_median, mars1_mad) = median_and_mad(mars1_samples);
    let (mars2_median, mars2_mad) = median_and_mad(mars2_samples);
    let (mars2_1t_median, mars2_1t_mad) = median_and_mad(mars2_1t_samples);

    let hdr = last_mars2_hdr.expect("at least one run");
    let leaves = last_mars2_leaves.expect("at least one run");
    let mars2_bytes = ifs_write(&hdr, &leaves).expect("encoder produced a writable tree");

    // Decode + matched-RD quality, both sides measured against the same ground truth
    // (§M1: the harness computes quality, not the codec).
    let dec_params = DecodeParams {
        mode: DecodeMode::Iterative,
        iterations: None,
        postprocess: false,
    };
    let dec_outcome = mars1_decode(
        bins,
        workdir,
        &last_mars1_ifs_name,
        "d.pgm",
        (width, height),
        &dec_params,
    )?;
    let mars1_decoded = read_pgm(&workdir.join("d.pgm"))?;
    let mars1_psnr = psnr(&plane, &mars1_decoded).unwrap_or(f64::INFINITY);
    let mars1_ifs_bytes = std::fs::metadata(workdir.join(&last_mars1_ifs_name))
        .map_err(|source| GateCError::Io {
            path: last_mars1_ifs_name.clone(),
            source,
        })?
        .len();
    let mars1_bpp = bpp(mars1_ifs_bytes, width as usize, height as usize);

    let mars2_decode_start = Instant::now();
    let mars2_decoded = std::hint::black_box(decode_iterative(&hdr, &leaves, 10));
    let mars2_decode_once = mars2_decode_start.elapsed();
    let mut mars2_decode_samples = vec![mars2_decode_once];
    for _ in 1..runs {
        let start = Instant::now();
        std::hint::black_box(decode_iterative(&hdr, &leaves, 10));
        mars2_decode_samples.push(start.elapsed());
    }
    let (mars2_decode_median, _mars2_decode_mad) = median_and_mad(mars2_decode_samples);
    let mars2_psnr = psnr(&plane, &mars2_decoded).unwrap_or(f64::INFINITY);
    let mars2_bpp = bpp(mars2_bytes.len() as u64, width as usize, height as usize);

    Ok(GateCPoint {
        image: image_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        mars1_encode_median: mars1_median,
        mars1_encode_mad: mars1_mad,
        mars2_encode_median: mars2_median,
        mars2_encode_mad: mars2_mad,
        mars2_1thread_median: mars2_1t_median,
        mars2_1thread_mad: mars2_1t_mad,
        encode_speedup: mars1_median.as_secs_f64() / mars2_median.as_secs_f64(),
        thread_speedup: mars2_1t_median.as_secs_f64() / mars2_median.as_secs_f64(),
        mars1_bpp,
        mars2_bpp,
        mars1_psnr,
        mars2_psnr,
        mars1_decode_seconds: dec_outcome.indicative_seconds,
        mars2_decode_median,
    })
}
