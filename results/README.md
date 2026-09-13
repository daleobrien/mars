# `results/` — the append-only result store

**§M7: a result you cannot regenerate from its own row is not a result.**

Every file here is JSONL: one JSON object per line, never pretty-printed, never edited,
never overwritten. A crashed sweep costs exactly one line, and `wc -l` is a row count.
Files are committed; the images that produced them are not (see `corpus/`).

Append-only is enforced in code, not by convention:
[`ResultStore`](../crates/mars-bench/src/store.rs) opens with `append(true)` and offers no
seek, truncate, or rewrite. If a number turns out to be wrong, the fix is a **new row**
plus an entry in [`docs/decisions.md`](../docs/decisions.md) — not a deletion. The
history of a wrong number is part of the evidence that the right one is right.

## Row envelope (schema 1)

```json
{
  "schema": 1,
  "kind": "quality",
  "provenance": {
    "harness_version": "0.1.0",
    "git_sha": "…40 hex…",
    "git_dirty": false,
    "build_profile": "release",
    "timestamp_utc": "2026-09-13T02:41:07Z",
    "corpus_manifest_sha256": "…or null…",
    "parameter_set_sha256": "…or null…",
    "run_index": 0,
    "machine": {
      "os": "macos",
      "os_build": "27.0.0",
      "arch": "aarch64",
      "cpu_brand": "Apple M3 Pro",
      "physical_cores": 12,
      "p_cores": 6,
      "e_cores": 6,
      "memory_bytes": 38654705664,
      "rustc_version": "…",
      "conditions": "…operator declaration, or null…"
    }
  },
  "data": { "…kind-specific…": null }
}
```

### Fields that exist for a specific reason

| Field | Why |
|---|---|
| `git_dirty` | A dirty row is still recorded. Refusing to record one just means people stop recording. |
| `corpus_manifest_sha256` | §M9. A changed image set cannot silently change a number. |
| `parameter_set_sha256` | Hashed from **key-sorted canonical JSON**, so a semantically identical parameter set hashes identically regardless of field order. |
| `run_index` | §M4 requires N ≥ 5 runs reported as median + MAD. This distinguishes the repeats. |
| `machine.p_cores` / `e_cores` | §M4. "M3" is not a fingerprint: M3 / M3 Pro / M3 Max span 8→16 CPU and 10→40 GPU cores, and a run that lands on E-cores reads 2–3× slow for scheduling reasons. |
| `machine.conditions` | Whether the machine was otherwise idle. The harness cannot detect this, so the operator declares it via `MARS_BENCH_CONDITIONS`. |

## What is here now

`baseline-mars1.jsonl` · `baseline-mars1.md` · `baseline-mars1.html` — **the Step 2
baseline.** 9360 rows: 4680 encodes (24 Kodak images + 12 fixtures × 6 speed-up methods ×
5 RMS thresholds × 6 structural variants) each decoded in both modes. Regenerate with
`just baseline-mars1` (~21 min on 6 P-cores) and `just baseline-mars1-report`; check with
`just gate-2`.

Read the report's §4 before quoting any `min_size = 2` number: under pyramidal decode those
curves run backwards, for the reason in `docs/decisions.md` D11.


`harness-smoke.jsonl` · `harness-smoke.md` · `harness-smoke.html` — **not a baseline.**
Ten rows from real Mars 1 encodes of Lena (MassCenter and Fisher, `-r 2,4,8,16,32`,
pyramidal decode with 10 iterations), produced at Step 1 purely to exercise the harness end
to end: reader → metrics → result store → BD-rate → report. One image and one decode mode
is not a measurement anyone should quote; §M9 puts the headline numbers on Kodak's 24
images, and the real baseline arrives at Step 2.

## Row kinds

| `kind` | `data` payload | Added at |
|---|---|---|
| `quality` | One `Measurement`: MSE/PSNR per plane, PSNR-Y/Cb/Cr/YUV, SSIM, MS-SSIM, coded bytes, bpp, and the pinned metric definitions inline. | Step 1 |
| `baseline-mars1` | Mars 1 encode/decode under the harness: `transforms`, `comparisons`, `zero_alfa_transform`, bytes, decode mode **and iteration count** (D9/D11 — the mode alone does not determine the reconstruction). | Step 2 |
| `timing` | Median + MAD over N ≥ 5 A/B-interleaved runs. | later |

A row carries its own metric definitions in `data.definitions`. That is redundant with
`docs/measurement.md` on purpose: a row extracted from this directory years from now
should not need the repository to be interpretable.

## Reading rows

```bash
jq -r 'select(.kind=="quality") | [.data.bpp, .data.psnr_y, .data.ssim] | @tsv' results/*.jsonl
```

`mars_bench::store::read_rows` fails on the first malformed line rather than skipping it.
