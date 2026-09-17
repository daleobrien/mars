# Mars 2

A quadtree fractal image codec in Rust, targeting Apple Silicon (aarch64 + Metal).

The 1998 C original by Mario Polvere lives unmodified in [`reference/mars1/`](reference/mars1/)
and serves as the measured baseline. Mars 2 is a clean-room implementation.

**Plans:** [technical-research-and-development-plan.md](technical-research-and-development-plan.md)
is the *why* and *what*. [implementation-plan.md](implementation-plan.md) is the *how*, *in
what order*, and *how we know it worked*.

## Status

| Group | Steps | State |
|---|---|---|
| **A — measure before building** | 0 Repo/CI/result store · 1 Metrics engine · 2 Mars 1 baselines · 3 Format spec | **done** |
| | 4 Anchor codecs | next |
| B — decoder, encoder, oracle | 5–9 | not started |
| C — optimisation | 10–14 | not started |
| D — research | 15–21 | not started |

No Mars 2 codec code exists yet, and that is the point: Group A's exit condition is that
everything can be measured *before* anything is built. See §0 of the implementation plan.

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
  that path unless `--lambda` is supplied; explicit lambda takes precedence and the
  thresholds then seed the rate-estimation warm-up.
- `--method fisher` (and other search methods): still grayscale/legacy-only;
  cannot be combined with explicit `--lambda`.
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
crates/mars-codec/    partition/search/fit/encode/decode — empty until Step 5
crates/mars-bench/    the harness: BD-rate, provenance, append-only result store, reports
crates/mars-cli/      marsbench (encmars/decmars arrive with the encoder)
reference/mars1/      the unmodified 1998 C, plus a pinned build wrapper
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
