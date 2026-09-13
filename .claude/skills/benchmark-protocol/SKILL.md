---
name: benchmark-protocol
description: How to run and report timing and search-cost benchmarks for Mars 2 on Apple Silicon — A/B interleaved runs, median plus MAD over N≥5, P-core and QoS rules, explicit Rayon thread counts, the machine fingerprint, and the evals/transform metric that is the project's primary efficiency number. Use when measuring encode or decode speed, claiming a speedup, profiling a SIMD, GPU, or threading change, or counting search work.
---

# Benchmark protocol

Source of record: `implementation-plan.md` §M4 (timing) and §M5 (search work).

Timing is the easiest number in the project to get wrong, and on an M3 laptop most of
the ways to get it wrong produce *plausible* numbers.

## The rule that decides whether a run counts

> **A speedup figure without absolute times and a machine fingerprint is not a result.**

## Protocol

- **N ≥ 5 runs**; report **median + median absolute deviation**. Never a single run.
- **A/B interleaved**: `A B A B A B…`, not `AAAAA BBBBB`, so thermal drift cancels
  instead of accumulating into the second group.
- **MAD above ~5% is a failed measurement**, not noise to average away. Re-run on an
  idle machine.
- Encode and decode reported **separately**, plus normalised `s/megapixel`.
- Warm file cache by default; state it.
- Record with every row: exact chip variant, core counts, compiler + version + flags,
  OS build, and whether the machine was otherwise idle.

## Apple Silicon specifics — these silently corrupt results if ignored

- **Record the chip variant, not "M3".** M3 / M3 Pro / M3 Max span ~8→16 CPU cores and
  10→40 GPU cores. A number without the variant is not comparable to the next one.
- **P-cores vs E-cores.** macOS schedules by QoS class. A benchmark that lands on
  efficiency cores reads 2–3× slow for reasons unrelated to the code. Run timing at
  user-initiated QoS or higher, never under `nice`, and **never from a background agent
  session** — timing runs are foreground work. If you are an agent asked to produce
  timing numbers in the background, say so and defer the run rather than reporting it.
- **Set Rayon's thread count explicitly to the P-core count** for compute benchmarks.
  Defaulting to `num_cpus` includes E-cores and bends the scaling curve for scheduling
  reasons rather than algorithmic ones.
- **Serialise measurement even when development is parallel.** Two agents may build two
  steps concurrently in separate worktrees; their benchmark runs are taken one at a
  time, on an idle machine.
- **One change per measurement.** Landing SIMD and a new search together makes the
  combined speedup unattributable, and unattributable speedups are how a project ends
  up carrying complexity that never helped.

## `evals` — the primary efficiency metric (M5)

**Definition:** `evals` = the number of `(domain, isometry)` pairs for which a full
affine fit + RMS evaluation was performed. Mars 2 must increment at the point *exactly
analogous* to Mars 1's `comparisons` (incremented at six sites in `coding_func.c`, one
per speed-up method) or the numbers are not comparable.

Report:

```
evals                  — total
transforms             — leaf blocks emitted
evals / transform      — the headline number
```

This metric is machine-independent, deterministic, and non-gameable, which makes it the
most trustworthy number the project produces. Wall-clock speedups will be argued about;
`evals/transform` will not. It must be **invariant under thread count** — a variation is
a determinism bug.

When a learned or multi-stage search is being compared, **model inference cost counts in
the eval budget**. A model that costs more than the evaluations it saves is not an
acceleration.

## Targets currently on record

| Gate | Target |
|---|---|
| Gate C | CPU encode ≥ 20× Mars 1 at matched RD (NEON × threads), **measured excluding the GPU** |
| Gate C | Decode no slower than Mars 1 |
| Gate D | `evals/transform` reduced ≥ 10× vs. the best classical method at matched RD |
| Step 7 | GPU exhaustive ≥ 50× the Rayon CPU exhaustive, transfer included; Kodak oracle in hours |
| Kill criterion | After Gate C, CPU encode speedup < 5× stops the project |

The GPU path is deliberately excluded from Gate C: a GPU speedup would mask a broken CPU
kernel, and the CPU path is what Steps 13–16 build on.

## Checklist for a timing claim

- [ ] N ≥ 5, A/B interleaved, median + MAD reported, MAD ≤ 5%.
- [ ] Absolute times given, not only a ratio; `s/megapixel` included.
- [ ] Machine fingerprint recorded, including the exact M3 variant and core counts.
- [ ] Rayon threads set explicitly; run in the foreground at user-initiated QoS.
- [ ] Encode and decode separated.
- [ ] Exactly one change under test since the comparison baseline.
- [ ] `evals/transform` reported alongside wall-clock wherever search work changed.
- [ ] Row appended to `results/` with full provenance (see `measurement-contract`).
