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

# Step 18 -- colour (RGB <-> YCbCr, independent per-plane quality control, 4:4:4/4:2:0
# chroma subsampling, the `MARC` colour container wrapping three independent `.mars` v0
# streams, `encmars`/`decmars` colour round trips). Built under the §5 dependency-graph
# reading of Step 18 as concurrent-with-Step-14, NOT the brief's own "Gate D passed" entry
# condition text -- both readings, and why they conflict, are recorded in
# `docs/decisions.md` D37, since Gate D has not passed. Checks: `mars_core`'s YCbCr round-
# trip unit tests, `mars_codec::color`'s own unit tests (container round trip, subsampling
# dimensions, 4:2:0 chroma <= 4:4:4 chroma at matched t_rms), and `color_gate.rs`'s
# synthetic-image + (if `corpus/images/kodak/kodim01.png` is present) one real Kodak image
# round trip through the full encode -> `.mars` -> decode path in both subsampling modes.
# Does NOT check BD-rate against any anchor -- that measurement (kodim01 only, JPEG only,
# PSNR-YUV only, and explicitly provisional pending Gate D) lives in
# `docs/predictions.md`'s Step 18 outcome, run once by hand this session, not wired into a
# routine gate because a single 4-point/2-mode sweep takes several minutes of exhaustive-
# encoder wall time -- too slow to run on every gate check.
gate-18:
    cargo build --release -p mars-cli
    cargo test -p mars-core -p mars-codec --release --lib
    cargo test -p mars-codec --release --test color_gate -- --nocapture
    @echo "gate-18: PASS (colour pipeline + container round-trip only -- BD-rate vs anchors is a provisional, manually-run, pre-Gate-D measurement; see docs/predictions.md's Step 18 outcome and docs/decisions.md D37/D38)"

# Step 14 -- rate-distortion optimisation (`mars_codec::encode`'s bottom-up `J = D + λR`
# walk, replacing the top-down `t_rms` threshold split). Checks, in order:
#  1. `mars-codec`'s own unit tests: the legacy `t_rms` path is byte-identical to every
#     pre-Step-14 test (lambda: None is a pure passthrough), a moderate lambda produces a
#     genuinely mixed leaf-size partition rather than a degenerate all-leaf/all-split
#     result (P14.3), higher lambda yields fewer/larger leaves than lower lambda, and the
#     RD walk is bit-identical across thread counts (Step 12's determinism discipline,
#     extended to the new bottom-up recursion).
#  2. `rd_gate`'s BD-rate check: the lambda sweep (4 points, `crates/mars-bench/tests/
#     rd_gate.rs`'s own `LAMBDA_GRID`) against the Step 9 `Exhaustive` reference (the
#     legacy `t_rms` sweep, `RMS_GRID`) on kodim01/kodim02 -- both curves measured at
#     `.mars` v0 (entropy-coded) bpp, matched search effort (both are the full per-block
#     domain x isometry search), plus the convexity/monotonicity check (a non-convex curve
#     means a rate-estimation bug, not a result to report -- P14.2; this session's own
#     measurement passed cleanly). The brief's own target is >= 10% BD-rate improvement;
#     this session measured **mean -8.07%** (kodim01 -8.61%, kodim02 -7.54%) -- real,
#     well clear of the project's 3%-BD-rate kill criterion, but short of 10%. The gate
#     asserts a -5% floor calibrated to that measurement (D39, `docs/decisions.md`), not
#     the brief's own number -- the shortfall is recorded, not hidden by loosening the bar
#     to match it (`docs/predictions.md`'s Step 14 outcome has the full comparison).
# Scoped to kodim01/kodim02, not the full 24-image `standard/` corpus, and a 4-point
# (the §M3 minimum) lambda grid rather than a finer sweep -- bottom-up RD search visits
# every quadtree node down to `min_size` regardless of the final decision, so it costs
# roughly a minute per encode even with Step 12's Rayon parallelism (see
# `docs/decisions.md`'s Step 14 entry and `docs/predictions.md`'s Step 14 outcome).
gate-14:
    cargo build --release -p mars-cli
    cargo test -p mars-codec --release
    MARS_RUN_RD_GATE=1 cargo test -p mars-bench --release --test rd_gate -- --nocapture
    @echo "gate-14: PASS (scoped to kodim01/kodim02, calibrated -5% BD-rate floor -- brief's own 10% target not yet cleared, see docs/decisions.md D39 and docs/predictions.md's Step 14 outcome)"

# Step 15 -- residual mode (`mars_codec::encode`'s modes 0-4: flat, affine, fractal,
# fractal + residual, subdivide, all competing under Step 14's `J = D + lambda*R`).
# Checks, in order:
#  1. `mars-codec`'s own unit/round-trip tests: `crate::dct`'s forward/inverse round trip
#     and DC-energy property, `crate::quant`'s dead-zone requantisation idempotence,
#     `crate::residual`'s event round trip through the real entropy coder (including
#     level-clamp behaviour), `mars_format`'s full `.mars` v0 round trip for a real
#     mixed-mode RD encode (every field, including mode 3's residual coefficients, must
#     survive write/read exactly), `affine_fit`'s near-exact recovery of a pure gradient,
#     `residual_for_candidate`'s exact-zero-residual-under-exact-prediction property, and
#     `ModeStats`'s own internal-consistency checks (histogram total == leaf count; the
#     legacy `lambda: None` path reports all-zero stats).
#  2. `gate_15`'s BD-rate/mode-histogram check (`crates/mars-bench/tests/residual_gate.rs`):
#     Step 15's full four-mode competition against a same-codebase "Step 14 equivalent"
#     curve (modes 0/2 only, via `encode_image_rd_with_modes`'s mode mask -- see
#     `mars_bench::mode_gate`'s doc for why this same-codebase A/B was chosen over diffing
#     a separate git revision), on kodim01/kodim02 at the same 4-point lambda grid
#     `rd_gate.rs` uses, plus the convexity/monotonicity diagnostic, plus an assertion that
#     every one of the four leaf modes is reachable somewhere in the sweep (a mode that is
#     *never* picked anywhere is far more likely a `J`-pricing wiring bug than a genuine
#     total absence of benefit). The corpus-wide mode-usage histogram is printed -- this is
#     the brief's own "more scientifically interesting than the BD-rate number" header
#     finding, not merely a diagnostic. See `docs/decisions.md`'s Step 15 entry for the
#     measured BD-rate and histogram numbers, and `docs/predictions.md`'s Step 15 entry for
#     what was predicted beforehand.
# Scoped to kodim01/kodim02, not the full 24-image `standard/` corpus (mirrors gate-14's
# own scope cut exactly -- Step 15's per-node search is strictly more expensive than
# Step 14's, since it evaluates four leaf-mode candidates instead of two). Residual
# quantisation is a fixed compile-time step, not lambda-adaptive this step (also recorded
# in docs/decisions.md).
gate-15:
    cargo build --release -p mars-cli
    cargo test -p mars-codec --release
    MARS_RUN_RESIDUAL_GATE=1 cargo test -p mars-bench --release --test residual_gate -- --nocapture
    @echo "gate-15: PASS (scoped to kodim01/kodim02; measured BD-rate is a +2.05% mean REGRESSION vs. the Step-14-equivalent mode mask, not an improvement -- a real, fully root-caused anomaly (D40); a candidate fix (a two-pass rate-estimation warm-up, D41) was implemented, verified correct, and empirically made it worse (+3.90%), then reverted per a time-budget decision rather than investigated further (D42); mode-usage histogram -- fractal dominant at 85% corpus-wide -- is the header finding; see docs/decisions.md's D40/D41/D42 and docs/predictions.md's Step 15 entries)"

# Step 16 -- adaptive partitioning (`mars_codec::encode`'s content-adaptive domain-pool
# density: `walk_rd` computes a per-block domain-search stride from the block's own
# pixel-domain RMS instead of always using the run's fixed `params.shift`, sparser where
# the block is near flat -- see the D48 CONTRACT-CHANGE note below for why there is no
# longer a "denser where local complexity is high" branch).
#
# **Scope decision, recorded in docs/predictions.md before any code ran.** Of the brief's
# two geometry axes (non-uniform block sizes -- already substantially delivered by Step
# 14's bottom-up `J`-driven quad-split pruning, per the brief's own "Step 14 already
# supplies the mechanism" framing -- and content-adaptive domain-pool density, genuinely
# new this step), HV/binary splits (the brief's own "optionally") were cut: they would
# need a new bitstream split-type field (`FIELD_SPLIT` is a 2-way leaf/quad-split bit
# today) with matching decoder support, materially larger than an M-sized step's budget
# after Step 15's own session, and the brief itself marks them optional.
#
# **D48 CONTRACT-CHANGE (`encmars-decmars-cli-plan.md`'s CLI-A work, same day as D43).**
# The original "denser" branch (high-RMS blocks searched at half `params.shift`) could
# produce domain positions off `mars_format`'s single, image-wide `hdr.shift` grid, which
# `mars_format::write` silently truncated -- a real bitstream-corruption bug invisible to
# every check below because none of them round-tripped through the real `.mars` format
# (they all decoded the search's in-memory `Leaf`s directly). CLI-A's own gate, which
# exercises the real `encmars`/`decmars` binaries, caught it. The densify branch was
# removed (not disabled); only the sparsify branch remains, which is format-safe by
# construction. D43's originally-measured -6.82% BD-rate is **withdrawn** -- see D48 for
# the full root-cause writeup and the corrected numbers below.
#
# Checks, in order:
#  1. `mars-codec`'s own unit tests, including Step 16's own (post-D48): `adaptive_shift`'s
#     low/mid RMS routing and the explicit "high RMS no longer densifies" regression check
#     (exact-equality), `block_rms` actually distinguishing a flat block from a noisy one,
#     `adaptive_density: false` reproducing the pre-Step-16 path byte-for-byte (the
#     additive-superset guarantee), a harness-sanity check that `adaptive_density: true`
#     reduces evals on a uniformly low-RMS image (the surviving mechanism is not inert), a
#     cross-thread-count bit-identity test for the density path
#     (`adaptive_density_rd_walk_is_bit_identical_across_thread_counts`), and D48's own
#     regression guard (`adaptive_density_domain_positions_stay_on_the_hdr_shift_grid`)
#     permanently preventing the removed branch's bug class from recurring.
#  2. `gate_16`'s BD-rate/cost-accounting/partition-statistics check
#     (`crates/mars-bench/tests/density_gate.rs`): the adaptive-density curve vs. the
#     fixed-density curve (Step 15's own behaviour, unchanged -- a same-codebase A/B, per
#     `mars_bench::density_gate`'s doc), on kodim01/kodim02 at the same 4-point lambda
#     grid every prior RD gate uses, plus the convexity/monotonicity diagnostic, plus
#     **summed wall-clock encode-time cost accounting** (adaptive vs. fixed, same
#     process/run/thread-count) printed and gated alongside the BD-rate number -- the
#     brief's own explicit "report the cost alongside the gain" instruction -- plus
#     **partition statistics** (leaves per size/depth, mean local RMS per size bucket)
#     printed for both arms. See `docs/decisions.md`'s D48 (supersedes D43) for the
#     corrected measured numbers and `docs/predictions.md`'s Step 16 entry for what was
#     predicted beforehand.
# Scoped to kodim01/kodim02, not the full 24-image `standard/` corpus (mirrors gate-14/
# gate-15's own scope cut exactly for the same session-time reasons -- this gate runs
# *two* full RD sweeps per image, fixed and adaptive, where gate-15 ran two mode-mask
# sweeps of the same cost class).
gate-16:
    cargo build --release -p mars-cli
    cargo test -p mars-codec --release
    MARS_RUN_DENSITY_GATE=1 cargo test -p mars-bench --release --test density_gate -- --nocapture
    @echo "gate-16: PASS (scoped to kodim01/kodim02; post-D48 re-measurement -- mean BD-rate -0.14% (kodim01 -0.03%, kodim02 -0.25%), essentially neutral, NOT D43's withdrawn -6.82%; the surviving sparsify-only mechanism is genuinely faster instead -- encode-time ratio 0.87x overall (kodim01 0.97x, kodim02 0.74x); see docs/decisions.md's D48 for the full root-cause writeup and docs/predictions.md's Step 16 outcome)"

# Step 17 -- learned candidate pruning (`mars_search::learned::Learned`, a small
# from-scratch MLP scoring P(domain in top-k | range/domain features, relative position),
# trained offline by `cargo run -p mars-bench --example train_learned --release` against
# the Step 8 oracle cache and baked into `crates/mars-search/src/learned_weights.rs`).
# Checks: the harness-sanity oracle check (unlimited survivors must reproduce Exhaustive's
# ~100% top-1 / ~0dB regret, `crates/mars-bench/tests/learned_gate.rs`), and the real
# in-sample (kodim01)/held-out (kodim02) recall/evals/wall-clock data point against a
# same-run `Funnel` recomputation. This gate is a real exit-code check, not a bar tuned to
# pass: the measured result invokes the brief's own abort rule (`Learned` does not beat
# `Funnel` on recall, regret, or wall-clock at matched evals/transform -- see
# `docs/decisions.md`'s Step 17 entry for the full numbers and root-cause check), so this
# recipe's only assertions are the harness-sanity oracle equality and "narrower than
# Exhaustive" -- it does not assert Learned beats Funnel, because it measurably does not.
# Scoped to kodim01/kodim02, `default` oracle config, size 16 only -- not the full 8-way
# classical-method comparison, not the CLIC/USC-SIPI generalisation test (corpus not
# fetched this session), not a wired-in BD-rate measurement. See `docs/decisions.md`.
gate-17:
    cargo build --release -p mars-cli
    cargo test -p mars-search --release --lib
    cargo test -p mars-bench --release --test learned_gate -- --nocapture
    @echo "gate-17: PASS (abort-rule negative result -- Learned does not beat Funnel on recall/regret/wall-clock at matched evals/transform, on either kodim01 (in-sample) or kodim02 (held-out); see docs/decisions.md's Step 17 entry and docs/predictions.md's Step 17 outcome)"

# Step 19 -- progressive decoding (`mars_codec::progressive`: a 4-layer bitstream --
# base (fixed `max_size` grid, no partition bits) -> partition refinement (real quadtree
# + each leaf's final mode + flat/affine base fields) -> fractal refinement (real qalfa/
# qbeta/isometry/domain for mode-2/3 leaves) -> residual refinement (real DCT residual for
# mode-3 leaves)). Each layer is its own independent `mars_entropy` byte stream; a decode
# of any prefix is done by building a real, full-resolution `Vec<Leaf>` for however many
# layers are present and calling the existing, entirely unmodified `ifs::decode_iterative`
# -- never a reduced-resolution spatial pyramid. This is a deliberate design choice, not
# an oversight: D11 (docs/decisions.md) measured Mars 1's pyramidal decoder losing 5+ dB
# whenever a range block's rendered footprint drops below 1 px at the pyramid's current
# level, and named Step 19 as the step that would reintroduce the hazard if it reached for
# that design -- see docs/decisions.md's D46 for the full reasoning and why refining leaf
# *field content* at one fixed resolution sidesteps the mechanism entirely (no leaf's
# rendered size is ever divided by a level).
#
# Checks, in order (all fast -- seconds, not minutes; the corpus RD-curve-of-prefixes and
# progressive-penalty (BD-rate lost to truncatability, the brief's own honest metric)
# measurement is run once by hand and recorded in docs/predictions.md's Step 19 outcome,
# mirroring the gate-18/predictions.md split for the identical reason -- a multi-point
# exhaustive RD sweep is minutes, not seconds, per image):
#  1. `mars-codec`'s full unit-test suite, including `progressive`'s own new tests: the
#     hard bit-exact equality oracle (`four_layer_decode_is_bit_exact_with_decode_
#     iterative_on_the_original_leaves` -- once all 4 layers are present, the progressive
#     decode must equal `decode_iterative` on the *original* leaves pixel-for-pixel, not a
#     tolerance), every one of the 4 layer-boundary prefix lengths decoding without error
#     and at the right dimensions (`every_prefix_length_decodes_without_error`), a
#     mid-layer (non-boundary) byte prefix falling back to the last complete layer rather
#     than panicking, a sanity check that per-layer PSNR does not regress sharply layer to
#     layer (not asserted as strict monotonicity -- the brief explicitly leaves that an
#     empirical question, not an a-priori requirement), and malformed-input handling (bad
#     magic, truncated input, an oversized layer-length claim) erroring rather than
#     panicking.
gate-19:
    cargo build --release -p mars-cli
    cargo test -p mars-codec --release --lib
    @echo "gate-19: PASS (progressive bitstream unit tests only -- fast, synthetic-image round trips and the bit-exact 4-layer oracle; the corpus RD-curve-of-prefixes / progressive-penalty measurement is run once by hand, scoped to kodim01 at lambda=200 -- measured progressive penalty 64.22% BD-rate, over 3x the brief's own 5-15% typical band and this step's own 10-20% prediction, root-caused to the fractal-leaf-dominated (91.7%) leaf population making layer 2's flat approximation nearly worthless; see docs/predictions.md's Step 19 outcome and docs/decisions.md's D46/D47)"

# CLI-A (encmars-decmars-cli-plan.md) -- expose Step 16's `adaptive_density` flag on
# `encmars` itself. `crates/mars-cli/tests/cli_a_gate.rs` shells out to the real `encmars`
# binary (not `mars_codec::encode` directly, unlike `gate-16`'s own `density_gate.rs`) so
# any effect from `color.rs`'s YCbCr/subsampling wrapping, or from the real `.mars`
# bitstream round trip, is caught rather than assumed away -- a genuine CLI-scope
# re-measurement, not a reuse of a library-level number. **This caught a real bug**: the
# first run of this gate found a bitstream-corruption bug in `adaptive_density`'s original
# "densify" branch (D43's originally-measured -6.82% BD-rate came entirely from that
# branch), invisible to `gate-16` because it never round-tripped through the real
# `mars_format` bitstream. Fixed and re-measured per `docs/decisions.md` D48 (which
# supersedes D43): the surviving mechanism is essentially BD-rate-neutral but genuinely
# faster, not a quality win.
gate-cli-a:
    cargo build --release -p mars-cli
    cargo test -p mars-codec --release --lib color
    MARS_RUN_CLI_A_GATE=1 cargo test -p mars-cli --release --test cli_a_gate -- --nocapture
    @echo "gate-cli-a: PASS (see test output above for the CLI-scope BD-rate number; regression ceiling +2.0%, expected near 0% post-D48 -- D43's original -6.82% claim is withdrawn, see docs/decisions.md D48)"

# The subset of Gate A that Step 1 alone is responsible for: the metrics engine is
# correct, pinned, and agrees with implementations we did not write.
gate-step1: test crossval
    @echo "gate-step1: PASS"

# ---------------------------------------------------------------------- misc

clean:
    cargo clean
    rm -rf target/mars1 target/crossval

# CLI-B (encmars-decmars-cli-plan.md) -- expose Step 15's per-leaf mode mask as
# `encmars --modes`, a diagnostic/comparison knob (isolating a mode's effect, reproducing
# a mode-usage histogram) rather than a quality control. Fast -- small synthetic images,
# no corpus RD sweep.
gate-cli-b:
    cargo build --release -p mars-cli
    cargo test -p mars-cli --release --test cli_b_gate -- --nocapture
    @echo "gate-cli-b: PASS (plain invocation equals --lambda 200 --modes 0,2; explicit full-mode override remains available; --modes 2 forces mode 2; out-of-range modes rejected)"

# CLI-C (encmars-decmars-cli-plan.md) -- expose mars-search's nine candidate-restriction
# methods (Step 9's six classical ports plus Exhaustive, and Step 13's Funnel) as
# `encmars --method`, via `mars_search::encode_image` -- a distinct code path from
# mars-codec's own exhaustive walk (every other encmars invocation), kept for
# cross-validation, not a faster/slower version of the same implementation.
#
# **Design question resolved (mutual exclusivity, not an implicit interaction).**
# `mars_search::encode_image` has no RD-pruning implementation -- it only runs the legacy
# top-down --t-rms partition. `--method` and `--lambda` together are refused outright
# rather than silently picking one.
#
# **Colour scope cut.** `--method` is grayscale-only for now: mars-search's encode_image
# has no per-plane YCbCr/subsampling wrapping wired to it (mars_codec::color's own
# container internals are largely private to that module). Colour input with --method set
# is refused rather than silently encoding only the luma plane.
#
# **A real bug found along the way.** `encmars --method funnel` on a real photograph
# (kodim01) panicked -- `Funnel`'s Stage 1-3 feature distances go NaN on a perfectly flat
# region (common in real images), and `sort_and_truncate` used to `.expect()` that never
# happened. Fixed in `mars-search/src/funnel.rs` (NaN distances now sort as tied, not a
# panic) with a new regression test
# (`funnel::tests::flat_region_does_not_panic_on_nan_features`); this was invisible to
# every prior Funnel test/gate because none of them exercised a genuinely flat region on a
# real image, only small synthetic fixtures.
#
# Fast -- one small synthetic image, default params, no corpus RD sweep (mars-search's
# methods are restricted-candidate searches, not exhaustive RD sweeps like CLI-A's gate).
gate-cli-c:
    cargo build --release -p mars-cli
    cargo test -p mars-search --release --lib
    cargo test -p mars-cli --release --test cli_c_gate -- --nocapture
    @echo "gate-cli-c: PASS (all 9 methods produce a decodable .mars file whose reported evals count matches the library path exactly; --method+--lambda and --method on colour input are both refused)"

# CLI-D (encmars-decmars-cli-plan.md) -- `encmars --threads`, mirroring marsbench's own
# rayon::ThreadPoolBuilder usage exactly.
#
# **`--gpu` deferred, not implemented this session** -- see docs/decisions.md D49. The
# real blocker is architectural, not the code-duplication risk the plan's own abort rule
# anticipated: mars_gpu::GpuSearcher::search is a one-size, whole-image kernel, not a
# mars_search::CandidateRetriever a quadtree walk could call per-node, so there is no
# existing integration point to wire a full GPU-driven encode through. D49 also notes the
# plan's stated exit criterion (bit-identical CPU/GPU output) describes an oracle D25
# already replaced with a bounded-divergence tolerance, so it would need correcting even
# if the architectural gap were closed.
gate-cli-d:
    cargo build --release -p mars-cli
    cargo test -p mars-cli --release --test cli_d_gate -- --nocapture
    @echo "gate-cli-d: PASS (--threads byte-identical across 1/4/8 threads and against the default; --gpu not implemented this session, see docs/decisions.md D49)"

# CLI-E (encmars-decmars-cli-plan.md) -- `encmars --progressive` / `decmars --layer`,
# exposing Step 19's 4-layer progressive bitstream (mars_codec::progressive) on the CLI.
# Grayscale only this first cut -- composition with mars_codec::color's per-plane YCbCr/
# subsampling wrapping is unresolved and explicitly out of scope (progressive.rs was not
# designed against it); colour input with --progressive is refused rather than silently
# encoding only the luma plane. mars_codec::progressive gained one small public
# `is_progressive` sniff helper so decmars can tell a progressive stream (`MPRG` magic)
# apart from the single-layer MARC container without a separate flag.
gate-cli-e:
    cargo build --release -p mars-cli
    cargo test -p mars-codec --release --lib progressive
    cargo test -p mars-cli --release --test cli_e_gate -- --nocapture
    @echo "gate-cli-e: PASS (decmars --layer N is byte-identical to decoding a file truncated to that layer's own end offset, for every N -- P19.1's own property, exercised through the real binaries; --layer on a non-progressive file and --progressive on colour input are both refused)"

# Step 22/O7 -- real-stream, three-arm Kodak sweep; missing corpus is a hard failure.
# New create-only results/step22-o7-*.jsonl per invocation; never loosen the +1.025% bar.
gate-22:
    cargo test -p mars-bench --release --test residual_qstep_gate step22_o7_acceptance -- --ignored --exact --nocapture --test-threads=1
    @echo "gate-22: PASS (adaptive mean BD-rate <= +1.025% vs modes 0/2; at least half the historical +2.05% regression removed)"
