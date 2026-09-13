# The measurement contract, as implemented

§1 of [implementation-plan.md](../implementation-plan.md) states the rules. This document
states where each one lives in the code, so that a reader can check the rule against its
implementation rather than against a promise.

**Version:** harness `0.1.0` · **Last updated:** 2026-09-13 (Steps 0–1)

## M1 — One metrics implementation, never self-reported

Every quality metric is implemented exactly once, in
[`crates/mars-core/src/metrics.rs`](../crates/mars-core/src/metrics.rs), and computed from
two files on disk by [`mars-bench`](../crates/mars-bench/src/measure.rs). No codec has a
path to report its own PSNR, because no codec is given one: `mars-codec` does not depend
on `mars-core::metrics` for output purposes and the CLI's `metrics` subcommand takes file
paths, not numbers.

## M2 — Metric definitions are pinned

| Metric | Where | Pinned choices that would otherwise drift |
|---|---|---|
| MSE | `metrics::mse` | Integer accumulation in `u64`, so the numerator is exact. The mean is over the region the caller passes; `Plane::crop_top_left` is how Mars 1's power-of-two padding is excluded. |
| PSNR | `metrics::psnr_from_mse` | `10·log10(255²/MSE)`. Identical images return `None`, serialised as JSON `null` — never a sentinel such as 99.0, which would survive an average. |
| SSIM | `metrics::ssim` | 11×11 Gaussian σ=1.5; K1=0.01, K2=0.03; **valid** region only (no padding); **population** covariance (no N/(N−1) correction); computed on Y. This is Wang's original and matches `scikit-image` with `gaussian_weights=True, use_sample_covariance=False`. |
| MS-SSIM | `metrics::ms_ssim` | 5 scales, Wang weights; luminance enters **only** at the final scale; decimation is a **non-overlapping 2×2 box average** (`Downsample::Box2x2`), which is Wang's `msssim.m` and equals `avg_pool2d(2)`. Minimum dimension 176; smaller images return `Unavailable::TooSmall` and are excluded, never rescaled. |
| bpp | `metrics::bpp` | `8 · whole_file_bytes / (W·H)`. The CLI takes a path and stats it, so a "payload-only" figure is not expressible. |

Colour: per-plane MSE/PSNR on the native planes, plus PSNR on BT.601 full-range Y/Cb/Cr
and `PSNR-YUV = (6·Y + Cb + Cr)/8`. If any component is infinite the combination is
`null`, because a weighted mean that quietly dropped an infinite term would read as a
finite, wrong number.

For grayscale input no colour transform is applied at all, and a test asserts that
`PSNR-Y` is bit-identical to the plane PSNR — a ±1 LSB conversion drift between two
columns of the same table is the kind of thing nobody checks.

## M3 — RD comparisons are BD-rate, never single points

[`crates/mars-bench/src/bdrate.rs`](../crates/mars-bench/src/bdrate.rs). The type system
carries the rule: `BdResult` has no constructor that omits `psnr_interval_db` and
`bpp_interval`, so a BD-rate cannot be produced without the interval it was computed
over. Inputs are rejected rather than coerced:

- fewer than 4 points → `BdError::TooFewPoints`
- no overlapping range → `BdError::NoOverlap` (never extrapolated)
- a curve where more bits buy less quality → `BdError::NonMonotonic`

Interpolation is PCHIP over `(log10 bpp, PSNR)`, integrated analytically
([`pchip.rs`](../crates/mars-bench/src/pchip.rs)). PCHIP rather than a natural cubic
spline because with four or five RD points a natural spline overshoots between knots and
the overshoot lands in the integral.

## M4 — Timing protocol

Not yet exercised — Steps 0–1 produce no timing numbers. What exists is the machine
fingerprint every row carries (`provenance::Machine`): chip variant string, physical core
count, and **P-core and E-core counts separately**, because macOS schedules by QoS class
and a benchmark that lands on efficiency cores reads 2–3× slow for reasons unrelated to
the code. `MARS_BENCH_CONDITIONS` records the operator's declaration of machine state,
which the harness cannot detect.

## M5 — Search work

Nothing to implement yet; `evals/transform` arrives with the Mars 1 driver at Step 2 and
the Mars 2 encoder at Step 6.

## M7 — Provenance on every row

[`provenance.rs`](../crates/mars-bench/src/provenance.rs) and
[`store.rs`](../crates/mars-bench/src/store.rs). Append-only is enforced by construction:
`ResultStore` opens with `append(true)` and exposes no seek, truncate, or rewrite. See
[results/README.md](../results/README.md) for the row schema.

## M9 — Corpus

Manifests are committed, images are not. [`corpus/kodak.manifest.json`](../corpus/kodak.manifest.json)
pins all 24 Kodak images by SHA-256 and byte size; `scripts/fetch-corpus.sh` verifies
rather than trusts. `reference/mars1/lena.raw` is the one committed fixture (512×512,
headerless, 8-bit).

## What is deliberately *not* here yet

Steps 2–4 add the Mars 1 subprocess driver, the golden fixtures, and the anchor codecs.
Until then the harness has been exercised end to end exactly once, on a real Mars 1
encode/decode of Lena — see [predictions.md](predictions.md).
