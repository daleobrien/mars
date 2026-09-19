//! Step 12's hard exit criterion: the bitstream (equivalently, the leaf list and eval
//! count `encode_image` produces) must be bit-identical to the single-threaded run at
//! every thread count. Each thread count gets its own scoped `rayon::ThreadPool` — the
//! global pool can only be built once per process, and this test needs several counts in
//! one run.

use mars_codec::encode::{encode_image, EncodeParams};
use mars_core::io::read_raw;

const MANDELBROT: &str = "../../fixtures/images/mandelbrot.raw";
const MIXED_129X127: &str = "../../fixtures/images/mixed_129x127.raw";

/// Larger fixtures are cropped to this square before encoding. The invariant under test --
/// bit-identical leaves and eval count at every thread count, with both the parallel
/// (`>= PARALLEL_SIZE_CUTOFF`) and sequential-cutoff branches exercised -- does not depend
/// on frame size, and the committed 512x512 frame cost ~4x the wall time for no additional
/// coverage of that invariant.
const CROP: usize = 256;

/// `min_size` at 4 (below the encoder's `PARALLEL_SIZE_CUTOFF` of 8) so the recursion
/// exercises both the parallel and the sequential-cutoff branches, not just one.
fn params() -> EncodeParams {
    EncodeParams {
        min_size: 4,
        max_size: 32,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda: None,
    }
}

fn assert_identical_across_thread_counts(path: &str, width: usize, height: usize) {
    let image = read_raw(std::path::Path::new(path), width, height).expect("fixture committed");
    let image = if width > CROP && height > CROP {
        image.crop_top_left(CROP, CROP)
    } else {
        image
    };
    let p = params();

    let (baseline_hdr, baseline_leaves, baseline_evals) = encode_image(&image, &p);
    assert!(
        baseline_evals > 0,
        "sanity: the search must actually run for this fixture"
    );

    for threads in [1usize, 2, 4, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("building a scoped pool");
        let (hdr, leaves, evals) = pool.install(|| encode_image(&image, &p));
        assert_eq!(hdr, baseline_hdr, "{path}: header differs at {threads} threads");
        assert_eq!(
            evals, baseline_evals,
            "{path}: evals differs at {threads} threads"
        );
        assert_eq!(
            leaves, baseline_leaves,
            "{path}: leaf list (bitstream content) differs at {threads} threads"
        );
    }
}

#[test]
fn mandelbrot_bitstream_identical_at_every_thread_count() {
    assert_identical_across_thread_counts(MANDELBROT, 512, 512);
}

/// The forced-subdivision fixture (non-multiple-of-`min_size` dimensions), so the
/// determinism check also covers the `forced` split path, not only the RMS-driven one.
#[test]
fn mixed_129x127_bitstream_identical_at_every_thread_count() {
    assert_identical_across_thread_counts(MIXED_129X127, 129, 127);
}
