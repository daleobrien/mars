# Mars 2 task runner.
#
# §A1: every exit criterion is a command that exits 0 or 1. The `gate-*` recipes are
# those commands; a gate either passes or it does not, and "the curve looks reasonable"
# is not a gate.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# ---------------------------------------------------------------- build & check

build:
    cargo build --workspace --all-targets

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

clippy:
    cargo clippy --workspace --all-targets -- -D warnings

test:
    cargo test --workspace

deny:
    cargo deny check licenses advisories bans sources

# The fast checks, in the order that fails cheapest-first.
check: fmt-check clippy test

# --------------------------------------------------------------- mars 1 reference

# Build the 1998 C reference with pinned flags.
mars1:
    ./scripts/build-mars1.sh

# Fetch the corpus images named in a manifest and verify their hashes.
corpus manifest="corpus/kodak.manifest.json":
    ./scripts/fetch-corpus.sh {{manifest}}

# Build the grayscale image-set indexes the Mars 1 sweep reads. Kodak is colour PNG and
# Mars 1 reads headerless 8-bit gray, so the conversion is pinned in scripts/ppm2raw.py
# rather than left to whichever ImageMagick is installed.
corpus-gray:
    python3 scripts/build-imageset.py --set fixtures
    python3 scripts/build-imageset.py --set standard

# Verify the image sets still hash to what the indexes say.
corpus-gray-check:
    python3 scripts/build-imageset.py --set fixtures --check
    python3 scripts/build-imageset.py --set standard --check

# ------------------------------------------------------------- mars 1 baseline (Step 2)

# The whole Step 2 sweep, from one command. ~35 minutes on 6 P-cores.
baseline-mars1 jobs="":
    cargo build --release -p mars-cli
    ./target/release/marsbench mars1-sweep {{ if jobs == "" { "" } else { "--jobs " + jobs } }}

# Regenerate the committed report from the committed results.
baseline-mars1-report:
    cargo build --release -p mars-cli
    ./target/release/marsbench mars1-report \
        --markdown-out results/baseline-mars1.md \
        --html-out results/baseline-mars1.html

# Claims about the 1998 reference that the sweep depends on but does not itself test.
mars1-claims:
    cargo test -p mars-bench --test mars1_reference -- --ignored --nocapture

# --------------------------------------------------- mars 1 format spec (Step 3)

# Regenerate the golden `.ifs` set and its manifest, then the §12 walkthrough.
# ~4 minutes; needed only when the fixture config or the reference binaries change.
# Fixtures are contract (§M8): committing a change here needs a FORMAT-CHANGE: trailer.
mars1-fixtures:
    cargo build --release -p mars-cli
    python3 scripts/gen-tiny64.py
    ./target/release/marsbench mars1-fixtures
    .venv-crossval/bin/python scripts/validate-ifs.py --walkthrough

# --------------------------------------------------------------- anchor codecs (Step 4)

# Sweep the five anchor codecs (JPEG/JPEG2000/WebP/AVIF/JPEG XL) over quality on Kodak,
# then regenerate the combined report (anchors + the six Mars 1 methods) and BD-rate
# table against JPEG. ~2 minutes on 6 cores. Requires cjpeg/djpeg, opj_compress/
# opj_decompress, cwebp/dwebp, avifenc/avifdec, cjxl/djxl on PATH.
anchors jobs="":
    cargo build --release -p mars-cli
    ./target/release/marsbench anchors-sweep {{ if jobs == "" { "" } else { "--jobs " + jobs } }}
    ./target/release/marsbench anchors-report \
        --markdown-out results/anchors.md \
        --html-out results/anchors.html

# ------------------------------------------------------------------ crossval

# Create the pinned python venv holding the independent reference implementations.
crossval-setup:
    @command -v python3.12 >/dev/null || { echo "python3.12 required (brew install python@3.12)"; exit 1; }
    python3.12 -m venv .venv-crossval
    .venv-crossval/bin/pip install --quiet --upgrade pip
    .venv-crossval/bin/pip install --quiet -r scripts/crossval/requirements.txt
    .venv-crossval/bin/python -c "import skimage, numpy, scipy, sewar, PIL; print('crossval venv ready')"

# Cross-validate every metric against ffmpeg, scikit-image and sewar on 10 Kodak images.
crossval:
    cargo test -p mars-bench --test crossval -- --ignored --nocapture

# ------------------------------------------------------------------- contract

# §A2: fail if a commit loosens an assertion or a tolerance without saying why.
contract-check base="origin/main":
    ./scripts/check-contract.sh {{base}}

# -------------------------------------------------------------------- gates

# Gate A — "we can measure everything, and we have recorded baselines, with zero Mars 2
# codec code written." Steps 0-4. Run it before starting Group B.
gate-a: fmt-check clippy test deny mars1 crossval gate-3 gate-4
    @echo "gate-a: PASS"

# Step 2 — the Mars 1 baseline is captured, complete, and internally consistent.
# Reads the committed results; run `just baseline-mars1` first if they are missing.
gate-2: test mars1 corpus-gray-check mars1-claims
    cargo build --release -p mars-cli
    ./target/release/marsbench mars1-check
    @echo "gate-2: PASS"

# Step 3 — the format document is a specification, not notes.
#
# The whole criterion is that an implementation written from `docs/mars1-format.md` alone
# agrees with the 1998 binaries. It reads only committed data — the `.ifs` files and the
# integers and hashes in the manifest — so it needs neither the C binaries nor the corpus,
# and it will still run when this machine's toolchain is gone.
gate-3:
    python3 scripts/gen-tiny64.py --check
    .venv-crossval/bin/python scripts/validate-ifs.py
    @echo "gate-3: PASS"

# Step 4 — the anchor curves are recorded, complete, and self-consistent: all five codecs
# present, >= 6 points per (codec, image) inside 0.1-2.0 bpp, BD-rate against JPEG defined
# and finite for every other anchor on every image, report files present.
# Reads the committed results; run `just anchors` first if they are missing.
gate-4:
    cargo build --release -p mars-cli
    ./target/release/marsbench anchors-check
    @echo "gate-4: PASS"

# Step 5 — the `.ifs` reader recovers exactly the transform counts `encmars` printed, and a
# Rust iterative decode agrees with `decmars -i` to within 0.1 dB on every golden fixture.
# Needs `just mars1` (for `decmars`) but not the corpus — the golden `.ifs` files and their
# source images are all fixture-committed or hash-pinned.
gate-5:
    cargo build --release -p mars-cli
    ./target/release/marsbench ifs-check
    @echo "gate-5: PASS"

# Step 6 — the exhaustive Rust encoder: Rust-encode -> decmars decodes within 0.1 dB of
# Rust's own decode, and the resulting RD curve is within 0.2 dB BD-PSNR of the Step 2
# Fisher baseline, on the fixtures/default/rms=[2,4,8,16,32] grid. Needs `just mars1` (for
# `decmars`) and `results/baseline-mars1.jsonl` (`just baseline-mars1`). The integer-moment
# exactness and f32-vs-f64 divergence criteria are covered by `cargo test -p mars-codec`
# and this command's own printed percentage, respectively.
gate-6:
    cargo build --release -p mars-cli
    cargo test -p mars-codec --release
    ./target/release/marsbench rust-encoder-check
    @echo "gate-6: PASS"

# Step 7 — the GPU exhaustive search: bit-identical to the CPU exhaustive search on every
# compared block (any divergence is a bug, never a tolerance -- §A7), and >= 50x faster
# than the Rayon CPU exhaustive path, transfer included, per the `benchmark-protocol`
# skill (A/B interleaved, N=5, median+MAD). Needs a real GPU adapter (run on the target
# M-series machine, in the foreground -- timing runs are never background work) and the
# `standard`/Kodak images under `corpus/images/kodak-gray/`.
gate-7:
    cargo build --release -p mars-cli
    cargo test -p mars-gpu -p mars-bench --release
    ./target/release/marsbench gpu-search-check
    @echo "gate-7: PASS"

# Step 8 — the oracle cache + recall harness (§M6). Builds the top-32 exhaustive-search
# cache for `standard/` (Kodak) at the three `configs/oracle.json` variants (min_size=8 in
# all three -- docs/decisions.md D26), skipping any (image, config) pair whose cache
# already validates against the current image+config so a rerun after a partial build
# doesn't redo finished work. Then runs the exhaustive self-test (`oracle-check`): every
# cached block's rank-0 candidate must equal an *independently* GPU-computed top-1 winner
# (a different kernel/workgroup topology, not just "trust rank 0") -- exit criterion is
# 100% agreement, "anything else is a harness bug" (a merge-reduce dropping the true
# minimum), never a tolerance. Needs a real GPU adapter, run in the foreground.
gate-8:
    cargo build --release -p mars-cli
    cargo test -p mars-gpu -p mars-bench --release
    ./target/release/marsbench oracle-build
    ./target/release/marsbench oracle-check
    @echo "gate-8: PASS"

# Step 9 -- the six classical speed-up methods, behind `mars_search::CandidateRetriever`
# (`crates/mars-search`). Checks what this session could actually run end to end
# (docs/decisions.md D28 records the scope cut and why): the harness sanity check
# (`Exhaustive` vs. the Step 8 oracle, ~100% top-1 recall / ~0dB regret expected -- a
# harness bug otherwise) and each method's evals/transform within 5% of the C
# reference's own counter, reusing `results/baseline-mars1.jsonl`'s already-captured
# numbers since this sandbox cannot compile `reference/mars1` (`xcrun` has no working
# `cc` here). Needs a real GPU adapter (oracle-build) and `corpus-gray` run once.
# Does NOT check the RD-curve-within-0.2dB criterion (D28) -- that remains open.
gate-9:
    cargo build --release -p mars-cli
    cargo test -p mars-search -p mars-bench --release
    ./target/release/marsbench oracle-build --images kodim01 --configs default
    cargo test -p mars-bench --release --test classical_methods_gate -- --nocapture
    @echo "gate-9: PASS (scoped per docs/decisions.md D28 -- RD-curve check not yet implemented)"

# Step 10 -- the `.mars` v0 container + rANS entropy coding (`crates/mars-entropy`,
# `mars_codec::mars_format`), deliberately breaking Mars 1 compatibility. Checks: every
# `corpus/fixtures.images.json` fixture at Step 6's rms grid round-trips losslessly
# through write -> read (byte-identical leaf list, so byte-identical reconstruction);
# entropy coding reduces bpp vs. the raw `.ifs` bitstream at that identical
# reconstruction (attribution: same leaves, only the serialisation differs); and the
# decoder never panics, OOMs, or allocates unboundedly on arbitrary bytes (fuzzed
# separately -- see `just fuzz-mars-format`, not part of this gate since fuzzing has no
# natural exit-0 stopping point). Reuses gate-6's exhaustive encoder, so it costs what
# gate-6 costs (dominated by the exhaustive search, not by this step's own code).
gate-10:
    cargo build --release -p mars-cli
    cargo test -p mars-entropy -p mars-codec --release
    ./target/release/marsbench mars-format-check
    @echo "gate-10: PASS"

# Step 10's fuzz target -- run for a bounded time locally (CI would run this
# continuously; `just gate-10` does not depend on it since a fuzz run has no exit-0
# stopping point of its own). Needs nightly + cargo-fuzz.
fuzz-mars-format seconds="60":
    cd crates/mars-codec/fuzz && cargo +nightly fuzz run mars_format_read -- -max_total_time={{seconds}}

# Step 11 -- NEON kernels (`crates/mars-simd`) for the moment-accumulation inner loop
# (`domain_sums`, `cross_term`) `mars_codec::encode::search` now delegates to. The exit
# bar is exact-equality differential tests (integer sums have no reassociation hazard --
# see that crate's module doc), which `cargo test -p mars-simd` runs; `cargo test -p
# mars-codec` re-running clean is the bitstream-unchanged proof (same fixtures/leaves the
# pre-Step-11 scalar code produced, since the new kernels are exact-equal to it by
# construction, not just by this test). Speedup is reported, not gated -- `just
# simd-bench` (needs an idle machine per the benchmark-protocol skill) -- and recorded in
# docs/decisions.md D31, including a regression this step's own benchmark caught and a fix
# for it (permutation amortisation, not the kernel), which is exactly what the report step
# is for.
gate-11:
    cargo test -p mars-simd -p mars-codec --release
    @echo "gate-11: PASS"

# Step 11's A/B-interleaved NEON-vs-scalar speed report (not gated -- see gate-11's own
# comment). Run this on an idle machine, in the foreground, per the benchmark-protocol
# skill -- never from a background session.
simd-bench:
    cargo build --release -p mars-cli
    ./target/release/marsbench simd-bench

# Step 12 -- Rayon parallelism (`mars_codec::encode`): parallel `Contracted::build` (one
# task per output row) and parallel range-block search, decoupled from emission (each
# quadrant of the RMS-driven quadtree walk returns its own leaves/evals, merged by plain
# concatenation in the same TL/BL/TR/BR order the sequential walk always used, above an
# 8-pixel size cutoff below which `rayon::join` overhead would exceed the search it
# parallelises -- see `encode.rs`'s `PARALLEL_SIZE_CUTOFF`). The exit bar is bitstream
# identity across thread counts, not a speed floor (§ the step's low verification burden):
# `parallel_determinism` builds a scoped `rayon::ThreadPool` at 1/2/4/8/16 threads and
# asserts the header, eval count, and full leaf list are byte-identical to the
# single-threaded run, on both the RMS-driven fixture (`mandelbrot`, 512x512) and the
# forced-subdivision one (`mixed_129x127`).
#
# Gate C prep (docs/decisions.md D33) ported the identical split/rayon::join pattern into
# `mars_search::encode_image`'s Fisher walk (previously exhaustive-only), so its own
# `parallel_determinism` test (mandelbrot only -- `mars_search`'s `SizedRetrievers` doesn't
# support the forced-subdivision fixture, see that test's own doc) is included here too.
gate-12:
    cargo test -p mars-codec -p mars-search --release
    @echo "gate-12: PASS"

# Step 12's thread-count scaling curve, reported only (not gated -- see gate-12's own
# comment). Run this on an idle machine, in the foreground, per the benchmark-protocol
# skill -- never from a background session.
parallel-bench:
    cargo build --release -p mars-cli
    ./target/release/marsbench parallel-bench

# Step 13 -- hierarchical funnel search (`mars_search::funnel`), behind the same
# `CandidateRetriever` trait Step 9 built. Checks: the harness sanity check (`Funnel` with
# narrowing disabled must reproduce `Exhaustive`'s ~100% top-1 recall / ~0dB regret against
# the Step 8 oracle -- a harness bug otherwise), and a first real recall/evals/transform
# data point at the default scaled survivor counts on `kodim01`, checked against a floor
# calibrated to Step 9's own measured recall numbers, not the withdrawn 80-95% guess
# (`docs/predictions.md`'s Step 13 prediction and outcome). Needs `corpus-gray` and a real
# GPU adapter (`oracle-build --images kodim01 --configs default`) once.
# Does NOT check the full survival/recall tradeoff curve or the Pareto frontier against all
# six classical methods across the 24-image corpus -- open gaps, recorded in
# `docs/predictions.md`, mirroring D28's Step 9 scope cut.
gate-13:
    cargo build --release -p mars-cli
    cargo test -p mars-search -p mars-bench --release
    ./target/release/marsbench oracle-build --images kodim01 --configs default
    cargo test -p mars-bench --release --test funnel_gate -- --nocapture
    @echo "gate-13: PASS (scoped to kodim01 -- see docs/predictions.md's Step 13 outcome)"

# The subset of Gate A that Step 1 alone is responsible for: the metrics engine is
# correct, pinned, and agrees with implementations we did not write.
gate-step1: test crossval
    @echo "gate-step1: PASS"

# ---------------------------------------------------------------------- misc

clean:
    cargo clean
    rm -rf target/mars1 target/crossval
