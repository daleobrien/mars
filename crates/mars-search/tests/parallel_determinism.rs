//! Gate C's threading port (mirroring `mars_codec`'s Step 12 test): the classical Fisher
//! search's leaf list, eval count, and per-block picks must be bit-identical to the
//! single-threaded run at every thread count. Each thread count gets its own scoped
//! `rayon::ThreadPool` -- the global pool can only be built once per process, and this
//! test needs several counts in one run.

use mars_codec::encode::{build_contracted, EncodeParams};
use mars_core::io::read_raw;
use mars_search::fisher::Fisher;
use mars_search::{encode_image, SizedRetrievers};

const MANDELBROT: &str = "../../fixtures/images/mandelbrot.raw";

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
    }
}

fn assert_identical_across_thread_counts(path: &str, width: usize, height: usize) {
    let image = read_raw(std::path::Path::new(path), width, height).expect("fixture committed");
    let p = params();
    let contracted = build_contracted(&image);
    let retrievers = SizedRetrievers::build(
        &contracted,
        image.width() as u32,
        image.height() as u32,
        p.shift,
        p.min_size,
        p.max_size,
        || Box::new(Fisher::default()),
    );

    let (baseline_hdr, baseline_leaves, baseline_evals, baseline_picks) =
        encode_image(&image, &p, &retrievers);
    assert!(
        baseline_evals > 0,
        "sanity: the search must actually run for this fixture"
    );

    for threads in [1usize, 2, 4, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("building a scoped pool");
        let (hdr, leaves, evals, picks) = pool.install(|| encode_image(&image, &p, &retrievers));
        assert_eq!(
            hdr, baseline_hdr,
            "{path}: header differs at {threads} threads"
        );
        assert_eq!(
            evals, baseline_evals,
            "{path}: evals differs at {threads} threads"
        );
        assert_eq!(
            leaves, baseline_leaves,
            "{path}: leaf list (bitstream content) differs at {threads} threads"
        );
        assert_eq!(
            picks, baseline_picks,
            "{path}: picks differ at {threads} threads"
        );
    }
}

#[test]
fn mandelbrot_512_bitstream_identical_at_every_thread_count() {
    assert_identical_across_thread_counts(MANDELBROT, 512, 512);
}

// Unlike `mars_codec::encode`'s exhaustive walk, the `mixed_129x127` forced-subdivision
// fixture is deliberately not exercised here: `SizedRetrievers` only indexes
// `[min_size, max_size]`, and that fixture's non-multiple-of-`min_size` dimensions force
// the walk to recurse below `min_size` (to a size with no indexed retriever) regardless
// of thread count -- a pre-existing gap in `mars_search::walk`'s forced-split path, not a
// determinism bug, and out of scope for Gate C's threading port. See docs/decisions.md.
