---
name: measurement-contract
description: Binding rules for computing, comparing, and recording any quality or rate number in Mars 2 — pinned PSNR/SSIM/MS-SSIM/bpp definitions, BD-rate requirements, corpus roles, and the provenance block every result row carries. Use when implementing or changing metric code, comparing two codecs or two revisions, producing an RD curve or a BD-rate figure, writing to results/, or quoting any quality number in a report, README, or commit message.
---

# Measurement contract

Source of record: `implementation-plan.md` §1 (M1–M10). These rules are binding for the
life of the project. The project's characteristic failure is **silent plausible
wrongness** — a number that looks reasonable, is wrong by 0.4 dB, and is inherited by
everything downstream. This contract is the primary defence.

## Non-negotiables

1. **No codec reports its own quality.** `mars-bench` computes every metric from two
   files on disk: the original and the decoded image. If a number came from an encoder's
   stdout (other than byte counts and `evals`), it is not a result.
2. **Metrics are pinned** (below). Never re-derive a definition from memory or from
   another implementation's defaults.
3. **A single `(bpp, PSNR)` pair is not a comparison.** RD claims are BD-rate.
4. **Every row is provenanced and append-only.** Nothing under `results/` is ever
   edited or overwritten.

## Pinned definitions (M2)

| Metric | Definition |
|---|---|
| MSE | Mean over the **original W×H region only**. Mars 1 pads to `virtual_size` (next power of two); padding is never measured. |
| PSNR | `10·log10(255²/MSE)`, 8-bit. Grayscale → Y only. Colour → per channel, plus PSNR-Y, plus PSNR-YUV = `(6·Y + Cb + Cr)/8`. |
| SSIM | 11×11 Gaussian window, σ=1.5, K1=0.01, K2=0.03, on Y, mean of the map. Implementation pinned and version-recorded. |
| MS-SSIM | 5 scales, Wang weights `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`. Min dimension ≥ 176; smaller images are **flagged and excluded**, never silently rescaled. |
| bpp | `8·total_file_size_bytes/(W·H)` — whole file, header included. "Payload-only" bitrates are forbidden. |

Identical images → PSNR is emitted as `null`, not a large float.

## BD-rate rules (M3)

A BD-rate number is valid only with all of:

- ≥ 4 quality points per curve,
- an **overlapping bpp range** between the two curves,
- piecewise-cubic interpolation over (log bpp, PSNR),
- the reported **bpp interval the number was computed over**, quoted alongside the %.

A BD-rate without its overlap interval is meaningless and is rejected. The harness
computes it; nobody computes it by hand. Sanity checks that must hold: a curve against
itself → 0.0%; a curve shifted by a known factor → the analytically expected value.

**λ is the quality control** once Step 14 lands. An RD curve is a λ sweep. Sweeping an
RMS threshold while λ stays fixed produces curves that look fine and mean nothing.

## Provenance (M7)

Every row in `results/*.jsonl` carries: git SHA + dirty flag, build profile, corpus
manifest hash, parameter-set hash, ISO-8601 timestamp, run index, harness version.
Plus, for any timing row, the machine fingerprint (see the `benchmark-protocol` skill).

> A result you cannot regenerate from its own row is not a result.

## Corpus roles (M9)

| Set | Use |
|---|---|
| `fixtures/` | lena.raw + 4 synthetic (flat, gradient, checkerboard, noise) — edge cases, golden files, smoke tests |
| `quick/` | 4 images ≤ 256² — CI gate, < 2 min budget |
| `standard/` | Kodak, 24 images, 768×512 — **all headline RD numbers** |
| `extended/` | CLIC professional validation subset; USC-SIPI textures — large photographic and self-similarity stress |

Lena is a fixture, never a headline. Single-image RD curves are how projects talk
themselves into conclusions that do not survive a second image. USC-SIPI textures matter
more here than for a normal codec: fractal coding's premise is block self-similarity, so
a photographs-only corpus understates and misattributes where the method works.

## Anchors are context, not targets (M10)

`mozjpeg`, OpenJPEG, `cwebp`, `libavif`/`aom`, `libjxl`, pinned versions with exact CLI
invocations recorded in the manifest. Fractal coding is **not** going to beat AVIF or
JPEG XL on rate-distortion, and beating them is explicitly not a success criterion.
Anchors answer "where does this sit?". Say so wherever the table appears.

Also: comparisons before Step 18 are grayscale-vs-colour-capable and must be labelled
as such.

## Before reporting any number

- [ ] Computed by `mars-bench` from files on disk, not self-reported.
- [ ] Metric definition matches the table above, including the W×H-only region rule.
- [ ] RD claims are BD-rate with ≥ 4 points and the overlap interval quoted.
- [ ] Row appended to `results/` with the full provenance block.
- [ ] Corpus named, and headline numbers are on `standard/`.
- [ ] Prediction was written **before** the run (`docs/predictions.md`, A4).
- [ ] Any disagreement with a cross-validation was investigated, not averaged (A7).
