# Measurement and experiments

[Back to the README](../README.md)

How results are measured is the [measurement contract](measurement.md); this is how to run it.

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

## Research experiments

The research plan's experiments run through a strict, resumable, config-driven runner.
A config names its inputs, the codec parameters held fixed across the stage, the decoder
settings, and one or more named stages. Every planned case is written to an append-only
store the moment it completes.

```bash
# One reproducible smoke case on the pinned fixture, then read it back:
marsbench experiment --config configs/research-smoke.json --stage smoke
marsbench experiment-report --experiment-id <recorded-id> --reference-group modes0-2
```

Artifacts default to `target/experiments/<experiment-id>/` and hold `results.jsonl` plus a
derived `summary.json`. Referenced inputs, their recorded hashes, and the codec binaries
are checked before any encode; statically unsupported arm combinations are rejected
rather than accepted as no-ops, and image-dependent refusals (for example `--method` on a
colour image) are recorded as explicit `unsupported` cases. A re-run resumes cases whose
config and build hashes still match, and the command exits non-zero if any planned case
failed, timed out, or was unsupported — a partial run never looks like a complete one.
`configs/research-search.json`, `research-rd.json` and `research-final.json` are opt-in
Kodak sweeps and are not run by routine tests.
