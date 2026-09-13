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

## Quick start

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
