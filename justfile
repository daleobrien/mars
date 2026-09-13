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

# The subset of Gate A that Step 1 alone is responsible for: the metrics engine is
# correct, pinned, and agrees with implementations we did not write.
gate-step1: test crossval
    @echo "gate-step1: PASS"

# ---------------------------------------------------------------------- misc

clean:
    cargo clean
    rm -rf target/mars1 target/crossval
