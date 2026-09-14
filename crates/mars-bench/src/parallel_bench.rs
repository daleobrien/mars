//! Step 12's scaling-curve report -- reported, not gated (the exit bar is the
//! `parallel_determinism` differential test in `mars-codec`, which either passes or does
//! not per the step's low verification burden). Per the `benchmark-protocol` skill: A/B
//! interleaved is not applicable here (there is only one implementation, run at different
//! thread counts), so each thread count is measured with its own scoped `rayon::ThreadPool`
//! -- the global pool can only be built once per process -- median + MAD over N runs,
//! machine fingerprint recorded.

use std::time::{Duration, Instant};

use mars_codec::encode::{encode_image, EncodeParams};
use mars_core::Plane;

pub use crate::gpu_search::machine_fingerprint_line;

fn median_and_mad(mut samples: Vec<Duration>) -> (Duration, Duration) {
    samples.sort();
    let median = samples[samples.len() / 2];
    let mut dev: Vec<Duration> = samples.iter().map(|&s| s.abs_diff(median)).collect();
    dev.sort();
    (median, dev[dev.len() / 2])
}

pub struct ScalingPoint {
    pub threads: usize,
    pub median: Duration,
    pub mad: Duration,
    /// `1_thread_median / this_median`, relative to the same 1-thread baseline for every
    /// point in the sweep (not to a separately-measured single-threaded binary).
    pub speedup: f64,
    /// `speedup / threads` -- the fraction of ideal linear scaling actually achieved.
    pub efficiency: f64,
}

/// Encode `image` `runs` times at each of `thread_counts`, in the order given (not
/// A/B-interleaved -- there is one code path here, not two to alternate between), and
/// report the scaling curve relative to the first entry's median. Per the step brief:
/// "sub-linear scaling is expected and fine; report it rather than quoting only the best
/// number" -- every point is returned, not just the fastest.
pub fn scaling_curve(
    image: &Plane,
    params: &EncodeParams,
    thread_counts: &[usize],
    runs: usize,
) -> Vec<ScalingPoint> {
    let mut points = Vec::new();
    let mut baseline: Option<Duration> = None;
    for &threads in thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("building a scoped Rayon pool");
        let samples: Vec<Duration> = (0..runs)
            .map(|_| {
                let start = Instant::now();
                pool.install(|| {
                    std::hint::black_box(encode_image(std::hint::black_box(image), params))
                });
                start.elapsed()
            })
            .collect();
        let (median, mad) = median_and_mad(samples);
        let baseline = *baseline.get_or_insert(median);
        let speedup = baseline.as_secs_f64() / median.as_secs_f64();
        points.push(ScalingPoint {
            threads,
            median,
            mad,
            speedup,
            efficiency: speedup / threads as f64,
        });
    }
    points
}
