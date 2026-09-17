# Mars 2

A quadtree fractal image codec in Rust, targeting Apple Silicon (aarch64 + Metal).

The 1998 C original by Mario Polvere lives unmodified in [`reference/mars1/`](reference/mars1/)
and serves as the measured baseline. Mars 2 is a clean-room implementation.

**Plans:** [technical-research-and-development-plan.md](technical-research-and-development-plan.md)
is the *why* and *what*. [implementation-plan.md](implementation-plan.md) is the *how*, *in
what order*, and *how we know it worked*.

## Status

Mars 2 implements threshold and rate–distortion quadtree encoding, iterative decoding,
color, progressive streams, nine search methods, SIMD/GPU infrastructure, and benchmark
metrics/anchors. Implementation does not imply that every research acceptance gate passed.

Current work follows [the research implementation plan](mars-research-implementation-plan.md):
correct residual reconstruction, measure serialized output, validate CLI capabilities, then
integrate search with the production RD encoder. See its execution log for completed steps
and [optimisation status](docs/encmars-optimisation-status.md) for historical results and limits.

## Encode and decode

```bash
cargo build --release -p mars-cli
./target/release/encmars input.png output.mars
./target/release/decmars output.mars decoded.png
```

A plain encode uses RD partitioning at `--lambda 200`, modes `0,2` (flat and fractal),
4:4:4 colour, and automatic thread selection. The mode choice follows the existing
kodim01/02 comparisons, not a claim of universal optimality. RD can be slower than
legacy threshold encoding. Library/benchmark defaults are unchanged.

- `--lambda 50`: prioritise quality; `--lambda 800`: prioritise smaller files.
- `--subsampling 420`: optionally trade chroma detail for smaller colour files.
- `--modes 0,1,2,3`: explicitly enable affine and residual modes too.
- `--t-rms 8`: restore legacy threshold partitioning. `--chroma-t-rms` also selects
  that path unless `--lambda` is supplied; explicit lambda takes precedence and ignores
  the thresholds. The rate-estimation warm-up uses fixed RMS 8.
- `--method fisher` (and other search methods): still grayscale/legacy-only;
  cannot be combined with explicit `--lambda`.
- Explicit `--modes` and `--adaptive-density` require RD, rather than silently doing
  nothing on the legacy/method path. Numeric options are checked before reading input;
  domain stride must be even and contrast limits must be representable in the header.
- Adaptive density, progressive output and experimental adaptive residual quantisation
  remain opt-in. To try the latter, include mode 3:
  `--lambda 200 --modes 0,2,3 --adaptive-residual`.

## Measurement setup

```bash
just crossval-setup   # pinned python reference implementations (needs python3.12)
just corpus           # fetch + hash-verify the Kodak corpus
just mars1            # build the 1998 C reference with pinned flags
just gate-a           # the whole Group A gate: exits 0 or 1
```

Measure a pair of images:

```bash
marsbench metrics original.png decoded.png --coded stream.mars
```

## Layout

```
crates/mars-core/     image types, IO, and THE implementation of every quality metric
crates/mars-codec/    partition/search/fit/encode/decode, color and progressive formats
crates/mars-search/   candidate retrieval and classical/learned search methods
crates/mars-bench/    the harness: BD-rate, provenance, append-only result store, reports
crates/mars-cli/      marsbench, encmars, decmars
reference/mars1/      the unmodified 1998 C, plus a pinned build wrapper
Research/            papers and research summaries
corpus/               manifests; images are fetched and hash-verified, never committed
results/              append-only JSONL — committed
docs/                 mars1-format.md, measurement.md, licensing.md, predictions.md, decisions.md
```

## The rules that matter

Three, from §1 and §2.1 of the implementation plan, because they explain why the code looks
the way it does:

- **One metrics implementation, never self-reported** (§M1). Quality is computed by
  `mars-bench` from two files on disk. No codec reports its own PSNR.
- **A BD-rate without its interval is not a number** (§M3). `BdResult` cannot be
  constructed without the interval it was integrated over, and invalid input is refused
  rather than coerced into something quotable.
- **Tests are the contract** (§A2). Tolerances live in one file
  ([`tolerance.rs`](crates/mars-core/src/tolerance.rs)); `scripts/check-contract.sh` fails
  any commit that loosens an assertion or a tolerance without a `CONTRACT-CHANGE:` trailer
  explaining why.

The executor here is AI agents, which inverts the usual risk profile: typing is cheap and
the characteristic failure is not *running out of time* but **silent plausible wrongness** —
a PSNR that looks entirely reasonable, is wrong by 0.4 dB, and is inherited by everything
downstream. The measurement contract is the defence.

## Licence

GPL-2.0-or-later, following Mars 1. See [docs/licensing.md](docs/licensing.md) — there is a
real Apache-2.0 interaction, and it is resolved deliberately rather than discovered later.


## CLI
```bash

encmars input.png output.mars --lambda 200 --modes 0,2 --subsampling 420

encmars input.png output.mars --lambda 50       # Higher quality
encmars input.png output.mars --subsampling 420 # Smaller colour files
encmars input.png output.mars --t-rms 8         # Legacy encoding
```
