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
gate-a: fmt-check clippy test deny mars1 crossval
    @echo "gate-a: PASS"

# The subset of Gate A that Step 1 alone is responsible for: the metrics engine is
# correct, pinned, and agrees with implementations we did not write.
gate-step1: test crossval
    @echo "gate-step1: PASS"

# ---------------------------------------------------------------------- misc

clean:
    cargo clean
    rm -rf target/mars1 target/crossval
