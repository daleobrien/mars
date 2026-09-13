# Mars 1 baseline — Step 2

9360 rows from `results/baseline-mars1.jsonl` · sweep at git b19be1407a34 · report at b19be1407a34-dirty · harness 0.1.0 · Apple M3 Pro (6 P / 6 E cores) · 2026-09-13T07:45:24Z

<details><summary>Mars 1 reference build</summary>

```
built:    2026-09-13T02:46:15Z
arch:     arm64
cc:       cc
cc_version: Apple clang version 21.0.0 (clang-2100.3.34.2)
cflags:   -O2 -fno-fast-math -fno-unsafe-math-optimizations -fno-associative-math -ffp-contract=off -std=gnu89 -Wno-implicit-function-declaration -Wno-implicit-int -Wno-return-type
ldlibs:   -lm
src_sha256: eea2bef49b7ea10a2e15c9e37b190af6abd0581353f0607d8a79084260ed12ff
```

</details>

The project's first real data, on the 1998 C codec, before any Mars 2 codec code exists. Every quality number here was computed by `mars-bench` from two files on disk (§M1); the codec reports only its own byte count, and even that is cross-checked against the file size.

**Headline slice:** corpus `standard` (24 images) · variant `default` · **pyramidal decode**.

> Decode mode **and iteration count** are stated on every number here and on every row in the store. Pyramidal with 10 iterations is the 1998 default. Contrary to the warning in the plan — and to this project's own prediction P2.2 — the *mode* is worth almost nothing at that count (§4 below measures it), because both decoders converge to the same IFS fixed point. The **iteration count** is the variable that matters: the same bitstream decodes 6.1 dB apart at 1 iteration and 0.004 dB apart at 10. See `docs/decisions.md` D9.

## 1. `evals/transform` — the headline metric (§M5)

`comparisons` is Mars 1's `evals` analogue, incremented at six sites in `coding_func.c`, one per method. The ratio is work-weighted: total comparisons over total transforms across all 24 images × 5 rates, not a mean of per-encode ratios.

> **`µs/eval` is here to show what the metric does not count** (D10). An eval is the same affine fit in every method, so a method whose µs/eval is far above the others is spending its time in an index structure the counter cannot see. Those seconds are indicative only (D8); the ratio between them is the point, not their absolute value.

| method | evals/transform | per-encode min | per-encode max | total evals | mean PSNR-Y (dB) | mean bpp | mean encode s | µs/eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| saupe-fisher | 64.3 | 50.3 | 65.6 | 82154400 | 29.577 | 0.7964 | 0.73 | 1.066 |
| mc-saupe | 127.9 | 85.8 | 169.8 | 173137010 | 28.319 | 0.8429 | 0.27 | 0.189 |
| saupe | 513.7 | 401.0 | 524.9 | 644051200 | 29.854 | 0.7814 | 4.12 | 0.767 |
| fisher | 626.4 | 491.0 | 808.8 | 822318670 | 28.978 | 0.8178 | 0.54 | 0.079 |
| masscenter | 1162.4 | 798.2 | 1720.7 | 1498905737 | 29.289 | 0.8036 | 0.90 | 0.072 |
| hurtgen | 1506.8 | 1117.3 | 2114.4 | 1953594776 | 29.203 | 0.8079 | 1.14 | 0.070 |

## 2. Rate–distortion, six methods

Corpus-averaged operating points: at each `-r`, the arithmetic mean of bpp and of PSNR-Y over the 24 images, equally weighted. **These averaged curves are for plotting.** Every BD-rate quoted below is computed per image and then summarised (D7).

| method | -r 2 | -r 4 | -r 8 | -r 16 | -r 32 |
|---|---:|---:|---:|---:|---:|
| fisher | 31.02 dB @ 1.476 bpp | 30.84 dB @ 1.185 bpp | 30.02 dB @ 0.837 bpp | 27.79 dB @ 0.417 bpp | 25.23 dB @ 0.173 bpp |
| hurtgen | 31.35 dB @ 1.470 bpp | 31.15 dB @ 1.176 bpp | 30.25 dB @ 0.822 bpp | 27.93 dB @ 0.404 bpp | 25.34 dB @ 0.169 bpp |
| masscenter | 31.46 dB @ 1.462 bpp | 31.26 dB @ 1.170 bpp | 30.33 dB @ 0.816 bpp | 27.99 dB @ 0.401 bpp | 25.41 dB @ 0.169 bpp |
| mc-saupe | 30.02 dB @ 1.495 bpp | 29.89 dB @ 1.208 bpp | 29.25 dB @ 0.870 bpp | 27.41 dB @ 0.452 bpp | 25.03 dB @ 0.188 bpp |
| saupe | 32.37 dB @ 1.444 bpp | 32.10 dB @ 1.147 bpp | 30.94 dB @ 0.784 bpp | 28.27 dB @ 0.374 bpp | 25.58 dB @ 0.158 bpp |
| saupe-fisher | 31.96 dB @ 1.457 bpp | 31.71 dB @ 1.161 bpp | 30.67 dB @ 0.805 bpp | 28.11 dB @ 0.395 bpp | 25.43 dB @ 0.165 bpp |

## 3. BD-rate between methods (reference: `fisher`)

Positive means the method needs **more** bits than `fisher` for the same quality. Computed per image over the PSNR overlap, then summarised across images; `n` is how many of the 24 images produced a defined BD-rate, and anything that did not is named in the exclusions column rather than dropped (§A7).

| test | n | mean % | median % | min % | max % | PSNR interval (dB) | averaged-curve % | excluded |
|---|---:|---:|---:|---:|---:|---|---:|---|
| hurtgen | 24 | -8.58 | -8.74 | -13.42 | -1.64 | 21.09–34.88 | -8.31 | none |
| masscenter | 24 | -11.11 | -10.80 | -17.79 | -5.21 | 21.10–34.88 | -10.87 | none |
| mc-saupe | 24 | 26.37 | 25.84 | 10.34 | 42.39 | 21.00–33.47 | 25.76 | none |
| saupe | 24 | -25.05 | -25.05 | -29.45 | -17.99 | 21.26–34.88 | -24.50 | none |
| saupe-fisher | 24 | -16.65 | -18.54 | -21.68 | 5.08 | 21.23–34.88 | -16.54 | none |

## 4. Pyramidal vs iterative decode

The same bitstream through both decoders, at 10 iterations. Reported per variant, because the answer is not the same for all of them.

| variant | n | mean iterative − pyramidal (dB) | min | max |
|---|---:|---:|---:|---:|
| default (headline) | 720 | 0.0022 | -0.0146 | 0.0476 |
| alfa5 | 720 | 0.0028 | -0.0182 | 0.0547 |
| beta6 | 720 | 0.0026 | -0.0162 | 0.0479 |
| max32 | 720 | 0.0018 | -0.0247 | 0.0676 |
| min2 | 720 | 3.8912 | -0.0046 | 19.7733 |
| step8 | 720 | 0.0019 | -0.0225 | 0.0366 |

**`min2` is the exception, and it is not a small one.** Every other variant agrees to ~0.002 dB, because an IFS has a unique attracting fixed point and both decoders reach it; pyramidal is a convergence *accelerator*, not a cheaper approximation. `min2` disagrees by a mean of 3.9 dB and up to 19.8 dB.

The cause is the pyramid's reduced resolution, and the rule is exact: `decmars` decodes at `1/2^levels` scale first, so a range block of `min_size` occupies `min_size / 2^levels` pixels there. When that is ≥ 1 the modes agree; when it is 0.5 — a **sub-pixel range block** — pyramidal loses 5+ dB and cannot recover it on the way back up.

| image | min_size | pyramid levels | block at that level | pyramidal | iterative | gap |
|---|---:|---:|---:|---:|---:|---:|
| zoneplate 256² | 2 | 1 | 1.00 px | 22.615 | 22.576 | -0.039 |
| zoneplate 256² | 4 | 1 | 2.00 px | 12.125 | 12.120 | -0.005 |
| mandelbrot 512² | 2 | 2 | **0.50 px** | 32.992 | 38.738 | 5.746 |
| mandelbrot 512² | 4 | 2 | 1.00 px | 28.764 | 28.810 | 0.046 |
| kodim01 768×512 | 2 | 2 | **0.50 px** | 23.870 | 29.238 | 5.368 |
| kodim01 768×512 | 4 | 2 | 1.00 px | 27.421 | 27.418 | -0.003 |

The 256² row at `min_size = 2` is the control: small blocks are harmless on their own — it is sub-pixel blocks *at the pyramid's depth* that break. (Measured with `-F -r 4`; the 256² absolute values are low because those are synthetic stress fixtures, and only the gap is being compared.)

## 5. Structural variants vs the 1998 defaults

BD-rate of each one-parameter deviation against `default`, per method. Negative is better. The `evals/transform` column is that variant's work-weighted ratio, so the RD cost of a cheaper search is visible beside the saving.

| variant | method | BD-rate % (mean) | median % | n | evals/transform | undefined because |
|---|---|---:|---:|---:|---:|---|
| alfa5 | fisher | 1.98 | 1.95 | 24 | 626.3 |  |
| alfa5 | hurtgen | 1.92 | 1.95 | 24 | 1506.4 |  |
| alfa5 | masscenter | 1.85 | 1.83 | 24 | 1162.1 |  |
| alfa5 | mc-saupe | 1.76 | 1.82 | 24 | 127.9 |  |
| alfa5 | saupe | 2.06 | 1.96 | 24 | 513.7 |  |
| alfa5 | saupe-fisher | 2.00 | 1.83 | 24 | 64.3 |  |
| beta6 | fisher | -0.39 | -0.27 | 24 | 626.4 |  |
| beta6 | hurtgen | -0.12 | -0.14 | 24 | 1507.1 |  |
| beta6 | masscenter | -0.20 | -0.29 | 24 | 1162.9 |  |
| beta6 | mc-saupe | -0.42 | -0.52 | 24 | 127.9 |  |
| beta6 | saupe | 0.06 | -0.14 | 24 | 513.9 |  |
| beta6 | saupe-fisher | 0.11 | -0.06 | 24 | 64.3 |  |
| max32 | fisher | 7.25 | 6.95 | 24 | 640.3 |  |
| max32 | hurtgen | 7.10 | 7.28 | 24 | 1539.7 |  |
| max32 | masscenter | 7.39 | 6.92 | 24 | 1189.9 |  |
| max32 | mc-saupe | 7.41 | 7.04 | 24 | 130.9 |  |
| max32 | saupe | 5.73 | 6.10 | 24 | 528.2 |  |
| max32 | saupe-fisher | 5.63 | 5.30 | 24 | 66.0 |  |
| min2 | fisher | — | — | 0 | 7619.3 | curve not monotonic (more bits, less quality) on 24/24 images |
| min2 | hurtgen | — | — | 0 | 17356.6 | curve not monotonic (more bits, less quality) on 24/24 images |
| min2 | masscenter | — | — | 0 | 4260.5 | curve not monotonic (more bits, less quality) on 24/24 images |
| min2 | mc-saupe | 39.27 | 39.27 | 1 | 367.5 | curve not monotonic (more bits, less quality) on 23/24 images |
| min2 | saupe | — | — | 0 | 524.8 | curve not monotonic (more bits, less quality) on 24/24 images |
| min2 | saupe-fisher | — | — | 0 | 65.6 | curve not monotonic (more bits, less quality) on 24/24 images |
| step8 | fisher | 21.83 | 21.37 | 24 | 159.7 |  |
| step8 | hurtgen | 17.42 | 17.18 | 24 | 386.5 |  |
| step8 | masscenter | 16.29 | 15.23 | 24 | 298.5 |  |
| step8 | mc-saupe | 24.83 | 24.57 | 24 | 41.5 |  |
| step8 | saupe | 7.95 | 7.66 | 24 | 514.2 |  |
| step8 | saupe-fisher | 6.54 | 7.21 | 24 | 64.3 |  |

## 6. Per-image spread at the reference method

A corpus mean hides which images fractal coding suits. This is `fisher` at `-r 8`, per image.

| image | PSNR-Y (dB) | bpp | transforms | evals/transform |
|---|---:|---:|---:|---:|
| kodim01 | 27.266 | 1.4238 | 19065 | 773.9 |
| kodim02 | 31.967 | 0.5000 | 6654 | 640.4 |
| kodim03 | 32.606 | 0.4875 | 6489 | 567.6 |
| kodim04 | 32.034 | 0.5704 | 7590 | 599.2 |
| kodim05 | 27.601 | 1.3530 | 18102 | 581.9 |
| kodim06 | 29.462 | 1.0891 | 14577 | 578.9 |
| kodim07 | 29.716 | 0.6765 | 9012 | 702.6 |
| kodim08 | 24.658 | 1.3879 | 18585 | 755.2 |
| kodim09 | 33.037 | 0.5528 | 7359 | 647.1 |
| kodim10 | 31.658 | 0.5094 | 6774 | 676.8 |
| kodim11 | 29.585 | 0.7866 | 10503 | 638.5 |
| kodim12 | 32.492 | 0.4892 | 6504 | 648.8 |
| kodim13 | 24.620 | 1.6644 | 22305 | 580.1 |
| kodim14 | 29.210 | 1.1421 | 15258 | 624.7 |
| kodim15 | 29.939 | 0.5508 | 7347 | 638.0 |
| kodim16 | 31.174 | 0.7601 | 10143 | 587.5 |
| kodim17 | 31.493 | 0.5898 | 7842 | 596.0 |
| kodim18 | 28.669 | 1.0870 | 14532 | 558.8 |
| kodim19 | 31.180 | 0.9154 | 12240 | 596.4 |
| kodim20 | 32.027 | 0.5825 | 7929 | 681.0 |
| kodim21 | 29.518 | 0.8493 | 11358 | 583.5 |
| kodim22 | 30.027 | 0.8118 | 10830 | 577.7 |
| kodim23 | 33.249 | 0.3102 | 4116 | 624.8 |
| kodim24 | 27.185 | 1.0092 | 13566 | 667.8 |

