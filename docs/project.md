# The project

[Back to the README](../README.md)

## Origins and plans

The 1998 C original by Mario Polvere lives unmodified in [`reference/mars1/`](../reference/mars1/)
and serves as the measured baseline. Mars 2 is a clean-room implementation.

**Plans:** [technical-research-and-development-plan.md](../technical-research-and-development-plan.md)
is the *why* and *what*. [implementation-plan.md](../implementation-plan.md) is the *how*, *in
what order*, and *how we know it worked*.

## Status

Mars 2 implements threshold and rate–distortion quadtree encoding, iterative decoding,
color, progressive streams, nine search methods, SIMD/GPU infrastructure, and benchmark
metrics/anchors. Implementation does not imply that every research acceptance gate passed.

Current work follows [the research implementation plan](../mars-research-implementation-plan.md):
correct residual reconstruction, measure serialized output, validate CLI capabilities, then
integrate search with the production RD encoder. See its execution log for completed steps
and [optimisation status](encmars-optimisation-status.md) for historical results and limits.

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
docs/                 the documents, listed in the README (the plans are at the top level)
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
  ([`tolerance.rs`](../crates/mars-core/src/tolerance.rs)); `scripts/check-contract.sh` fails
  any commit that loosens an assertion or a tolerance without a `CONTRACT-CHANGE:` trailer
  explaining why.

The executor here is AI agents, which inverts the usual risk profile: typing is cheap and
the characteristic failure is not *running out of time* but **silent plausible wrongness** —
a PSNR that looks entirely reasonable, is wrong by 0.4 dB, and is inherited by everything
downstream. The measurement contract is the defence.
