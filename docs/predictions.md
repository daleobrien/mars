# Predictions

**Append-only (§A4).** Before any sweep, the step records what it expects. An agent that
measures first and explains afterwards will rationalise almost any result; the prediction
is what makes a surprise legible as a surprise.

Rules: append, never edit. Record the prediction *and* the outcome when it is known, in
place, so the pair stays together. If a prediction was formed during the work rather than
before it, say so — a retrofitted prediction is worse than none, because it reads as
confirmation.

---

## 2026-09-13 · Step 1 · metrics cross-validation

### P1.1 — ffmpeg's `ssim` filter will *not* agree with our SSIM

**Prediction (before running anything):** ffmpeg's `ssim` filter implements the
x264-derived variant — 8×8 **uniform** windows — not the 11×11 Gaussian that §M2 pins.
It will therefore disagree by far more than the 0.001 tolerance, and the disagreement will
be a definitional difference, not an error on either side.

**Consequence if true:** the step's exit criterion as literally written ("SSIM agrees with
an independent implementation, e.g. `ffmpeg -lavfi ssim`") cannot be met with ffmpeg, and
meeting it by widening the tolerance would be exactly the failure §A2 and §A7 exist to
prevent. The right move is a different reference, not a looser bound.

**Outcome: confirmed, and handled.** PSNR is cross-validated against ffmpeg (which has no
such ambiguity). SSIM is cross-validated against `scikit-image`'s
`structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False)`,
which *is* the definition §M2 pins. Recorded in [decisions.md](decisions.md) as D1.

### P1.2 — agreement will be near machine precision, not merely inside tolerance

**Prediction:** if the definitions truly match, agreement will be ~1e-12 or better, not
~1e-4. A disagreement of 1e-4 would sit comfortably inside the 0.001 tolerance while
actually indicating a real definitional difference — a padded instead of valid window, or
a sample instead of population covariance — that would bias every later result in the same
direction.

So: **treat anything worse than ~1e-9 as a finding to investigate, even though it passes.**

**Outcome: confirmed.** Over 10 Kodak images × 2 JPEG qualities, the worst disagreements
were PSNR 3.8e-7 dB (against both numpy and ffmpeg), SSIM 2.7e-14, MS-SSIM 1.1e-16. The
PSNR figure is larger than the others only because ffmpeg prints six decimal places; the
numpy comparison agrees to ~1e-13.

### P1.3 — MS-SSIM will disagree with `sewar` by ~1e-3, and the cause will be the decimation phase

**Prediction:** `sewar` decimates with `scipy.ndimage.uniform_filter(im, 2)` followed by
`[::2, ::2]`, which is a *backward* half-pixel-shifted window with reflected edges. Wang's
reference `msssim.m` uses a forward window at odd (1-based) indices, i.e. a
**non-overlapping** 2×2 box average — the same thing `avg_pool2d(2)` does in the PyTorch
and TensorFlow implementations. The two differ by half a pixel per scale, which should
show up in the third decimal place: large enough to fail a 0.001 tolerance, far too small
to look like a bug.

**Consequence if true:** the cross-validation must be phase-matched. Implementing both
phases, reporting the pinned one and cross-validating the other, tests the entire scale
chain and weighting while isolating the one parameter the references genuinely disagree on.

**Outcome: confirmed.** On kodim01 at q8 the pinned `Box2x2` result is 0.987864 and the
phase-matched one is 0.986641 — a 1.2e-3 difference, just over the tolerance, exactly as
predicted. Phase-matched, agreement is 1.1e-16. A test asserts the two phases have not
collapsed into each other, so the cross-validation cannot become vacuous.

---

## 2026-09-13 · Step 2 (not yet run) · Mars 1 baseline sweep

Recorded now, before the sweep, per §A4.

### P2.1 — the partition will be purely RMS-driven at 1998 defaults

The plan already flags this (§4, Step 2) and the source agrees: with `T_ENT = 8.0` an
8-bit block's entropy cannot exceed 8.0 (and is bounded by 4.0 for a 4×4 block), so the
entropy pre-split can never fire; with `T_VAR = 1e6` against a variance bounded by
~16256, neither can the variance pre-split. **Prediction: `-e` and `-v` at their defaults
have literally zero effect on the output bitstream**, and a sweep over them will produce
byte-identical files. If it does not, the reading of the source is wrong and that matters
more than the sweep.

### P2.2 — pyramidal and iterative decode will differ by 0.2–1 dB

`decmars` defaults to pyramidal (`piramidal INIT(=1)`, `iterations = 10`). Prediction: the
two modes differ enough to reorder RD curves, so every row must state which was used.
Specifically, pyramidal should be *slightly worse* at matched bitrate but much faster.

### P2.3 — `evals/transform` will span more than an order of magnitude across the six methods

Prediction: the kd-tree methods (Saupe, Saupe-Fisher, Mc-Saupe) will show `evals/transform`
an order of magnitude or more below the classification methods (Fisher, Hurtgen,
MassCenter), and the RD cost of that will be small — a few tenths of a dB — because the
search is finding near-optimal rather than optimal matches. If the RD cost is large, the
kd-tree parameters (`-l 50`, `-p 2.0`) are the first suspects, not the method.

### P2.4 — the baseline will be well short of JPEG

Prediction: Mars 1 at its best settings will lose to `mozjpeg` on Kodak by a large
BD-rate margin (order +50% or worse). §M10 is explicit that this is context, not failure,
but the number should be written down before it is seen so that it cannot be quietly
reframed afterwards.

---

## 2026-09-13 · Step 2 · outcomes

Recorded after the sweep (4680 encodes, 9360 rows), against the predictions above.

### P2.1 — **confirmed**, and tested directly rather than inferred

`the_presplit_thresholds_are_inert_at_the_1998_defaults` encodes lena four ways and
compares bitstreams byte-for-byte: raising `-e` to 1e9 or `-v` to 1e9 above their defaults
produces a **byte-identical** file, so neither pre-split can fire at 8.0 / 1e6. Lowering
`-e` to 1.0 or `-v` to 1.0 *does* change the file, so the flags are read and the inertness
result is not the vacuous one. At default settings the Mars 1 partition is purely
RMS-driven, exactly as the plan flagged.

### P2.2 — **miss**, and the mechanism turned out to be more useful than the prediction

Predicted: pyramidal and iterative decode differ by 0.2–1 dB, enough to reorder RD curves,
with pyramidal "slightly worse at matched bitrate but much faster".

Measured over 720 matched settings on Kodak: **mean +0.0022 dB** (iterative marginally
ahead), range −0.0146 to +0.0476 dB. Two orders of magnitude below the prediction, and
nowhere near enough to reorder anything — method differences are 8–26% BD-rate.

The prediction was not merely wrong in magnitude; it had the wrong model. Sweeping the
iteration count on one bitstream (kodim01, Fisher, `-r 8`):

| iterations | pyramidal | iterative | iterative − pyramidal |
|---:|---:|---:|---:|
| 1 | 24.5337 | 18.4271 | −6.1066 |
| 2 | 26.3481 | 21.3691 | −4.9790 |
| 3 | 26.9511 | 24.5337 | −2.4174 |
| 5 | 27.2182 | 26.9533 | −0.2649 |
| **10 (default)** | **27.2659** | **27.2620** | **−0.0039** |
| 20 | 27.2638 | 27.2671 | +0.0033 |

An IFS has a unique attracting fixed point, so both decoders converge to the *same* image.
Pyramidal is a **convergence accelerator** — it iterates at reduced resolution first and
arrives sooner — not a cheaper, worse reconstruction. The plan's "0.2–1 dB" is precisely
what one sees at **5** iterations; at the 1998 default of 10 both have converged.

Note also that pyramidal is *better*, not worse, whenever the two differ — the opposite of
the predicted direction. The prediction had it backwards because it assumed pyramidal
traded quality for speed.

**Consequence.** `decode_iterations` was added to every result row and the sweep re-run,
and the claim is pinned by a test asserting both halves — the modes must be distinguishable
at 1 iteration and converged by 10. See decisions D9.

**Correction, same day, before this was committed.** The paragraph above originally read
"the confound is the iteration count, not the mode", which was an overgeneralisation from
the `default` variant. Checking the `min2` variant — whose BD-rate column came back
undefined — showed the mode *is* a confound there: mean **+3.89 dB**, max **+19.77 dB**.

So P2.2's *number* is wrong at the settings it concerned (0.2–1 dB predicted, 0.002 dB
measured at `min_size = 4`), and its *warning* is right in a regime it did not name. The
governing rule turned out to be neither mode nor iterations alone: `decmars` decodes at
`1/2^levels` scale, and the modes agree exactly when `min_size / 2^levels ≥ 1`. At half a
pixel pyramidal loses 5+ dB. A 256² control at `min_size = 2` — one pyramid level, block
lands on a whole pixel, no penalty — rules out "small blocks" as the cause. Recorded as D11.

The prediction was worth making. Had the sweep simply reported 0.002 dB with no
pre-registered expectation to contradict, there would have been no reason to look further,
and the `min2` column would have been read as a harness limitation rather than a 19.8 dB
decoder defect.

### P2.3 — **half right**: the span is there, the mechanism is not

Predicted: the kd-tree methods (Saupe, Saupe-Fisher, Mc-Saupe) show `evals/transform` an
order of magnitude or more below the classification methods (Fisher, Hurtgen, MassCenter),
at an RD cost of a few tenths of a dB.

Measured on Kodak, 1998 defaults, pyramidal decode — `evals/transform` work-weighted over
24 images × 5 rates, with BD-rate against Fisher computed per image and averaged:

| method | evals/transform | BD-rate vs fisher | mean encode s | µs per eval |
|---|---:|---:|---:|---:|
| saupe-fisher | 64.3 | −16.65% | 0.77 | 1.128 |
| mc-saupe | 127.9 | +26.37% | 0.28 | 0.192 |
| saupe | 513.7 | −25.05% | 4.29 | 0.800 |
| fisher | 626.4 | — | 0.58 | 0.085 |
| masscenter | 1162.4 | −11.11% | 0.95 | 0.076 |
| hurtgen | 1506.8 | −8.58% | 1.19 | 0.073 |

**The span is 23×**, so "more than an order of magnitude" holds. **The predicted split does
not.** Saupe-Fisher (64.3) is ~10× below Fisher, but plain Saupe (513.7) is only 1.2×
below it — the kd-tree/classification dichotomy does not organise the results.

The RD prediction is wrong in **sign**. Cheaper search was expected to cost a few tenths of
a dB; instead Saupe-Fisher is simultaneously **9.7× cheaper in evals and 16.65% better in
BD-rate** than Fisher, and four of the five non-reference methods beat Fisher outright. The
prediction assumed search cost and RD trade against each other within this family. They do
not, because a method that finds different matches also produces a different *partition*
(the RMS threshold drives splitting), so rate and distortion both move.

An unflagged consequence: **Fisher, the reference chosen for the BD-rate table, is the
second-worst method on RD.** That is not a problem for the table — every number states its
reference — but it should be stated out loud rather than left for a reader to infer from
five negative numbers in a column.

### P2.4 — not yet measured

Requires the anchor codecs from Step 4. Deliberately left open rather than answered from
the Mars 1 numbers alone; the prediction stands as recorded.

---

## 2026-09-13 · Step 3 (not yet run) · `.ifs` format spec + golden fixtures

**Provenance of these predictions.** They were formed by reading `image_io.c`,
`mars_enc.c`, `mars_dec.c` and `coding_func.c`, and written down **before** generating a
single golden fixture, before running the independent validator, and before running
`encmars` even once in this step. They are therefore predictions about what the
*measurement* will show, derived from the source — not from the numbers. Several of them
contradict the spec sketch in `implementation-plan.md` §4 Step 3, which is the point:
that sketch is the thing being validated.

### P3.1 — the `dom_x` / `dom_y` bit-width "asymmetry" is not a naming slip

Both the plan (§4 Step 3) and the `mars1-reference` skill say `dom_x` being packed with
`bits_per_coordinate_h` "looks like a naming slip in the original, but encoder and decoder
agree, so it is normative. Preserve it exactly; do not fix it."

**Prediction: there is nothing to preserve, because there is no slip.** Mars 1 uses
`x` for the **row** axis and `y` for the **column** axis everywhere (`image[atx+x][aty+y]`,
`if(atx >= image_height || aty >= image_width)`). A row coordinate must be sized by the
image *height*, so `bits_per_coordinate_h` for `dom_x` is correct, not a bug that happens
to be symmetric across the two binaries.

**Consequence if true:** the instruction "preserve the bug" is actively harmful — a Rust
implementer who believes `dom_x` is an *x* coordinate in the usual sense will write a
transposed decoder and then "preserve" a second bug to compensate. The spec must state the
axis convention once, at the top, and the plan text must be corrected.

**How it will be falsified:** the Python decoder reads `dom_x` as a row offset. If that is
wrong, decoded images will be visibly transposed and will not match `decmars` at all — a
loud failure, not a subtle one.

### P3.2 — leaf blocks smaller than `min_size` exist, and their count is pure geometry

The forced-subdivision branch (`size > max_size || atx+size > image_height ||
aty+size > image_width`) recurses **without consulting `min_size`**, so on an image whose
dimensions are not multiples of `min_size` the walk drives blocks below `min_size` — down
to size 1. At size 1 the coding functions take a special path (`tip == 0`) that sets
`qalfa = zeroalfa` and `qbeta = image[atx][aty]`, i.e. **the raw pixel value**, which is
then packed into `N_BITBETA = 7` bits and silently loses its top bit.

Because the forced branch depends only on the image dimensions, the counts are independent
of method, of RMS threshold, and of `min_size` itself. Predicted exactly, at `-m 4 -M 16`:

| fixture | size-1 leaves | size-2 leaves |
|---|---:|---:|
| `mixed_129x127` (129×127) | **255** | (not predicted) |
| `mixed_250x250` (250×250) | **0** | **249** |
| every 256² and 512² fixture | **0** | **0** |

The 255 is row 126 in full (129 leaves) plus column 128 for rows 0–125 (126 leaves). The
249 is row 248 at even columns (125) plus column 248 at even rows 0–246 (124).

**Consequence if true:** a parser that assumes leaves are at least `min_size` is wrong on
two of the twelve fixtures, and a Rust encoder that reproduces the partition but not the
`tip == 0` path will disagree with Mars 1 on exactly those edge blocks. Neither the plan's
spec sketch nor the skill mentions this case at all.

### P3.3 — `flat128` at 1998 defaults produces a 392-byte file of 256 identical leaves, and decodes to 129

Everything about this file is predictable in closed form, which makes it the worked example
the spec needs:

- A constant block has `det = s0·s2 − s1² = 0`, so `alfa = 0` and `qalfa = 0` — every leaf
  is zero-alfa, so no isometry and no domain coordinates are emitted.
- The zero-alfa branch (`abs(qalfa − zeroalfa) <= zero_threshold`, true at the default
  `zero_threshold = 0`) **discards the searched `qbeta` and recomputes it** as
  `best_beta(…, 0.0)` = `int(0.5 + 128/255 · 127)` = `int(64.249)` = **64**, for every leaf.
- Reconstructed `beta = 64/127 · 255 = 128.5039…`, so each pixel is
  `bound(0.5 + 128.5039…)` truncated to `unsigned char` = **129**, not 128.
- 256×256 with `max_size = 16` gives exactly **256 leaves**, each `1 + 4 + 7 = 12` bits,
  after a 60-bit header: 3132 bits → **392 bytes**.
- MSE = 1 exactly, so PSNR = 10·log10(65025) = **48.1308 dB** in both decode modes.

**Prediction: all six numbers land exactly.** If `bytes_written` is not 392 the bit
accounting in the spec is wrong; if the decode is 128 rather than 129 the `bound()`
truncation has been mis-stated; if `qbeta` is not 64 the `best_beta` override has been
missed. This is one prediction with six independent ways to fail, which is why it is worth
more than the other five put together.

### P3.4 — the last byte is right-padded, not "left-padded"

The plan says "the final byte is left-padded by `pack(-1, …)`". Reading `pack`: after `m`
bits the accumulator holds them at bit positions `1..m` and `ptr == m+1`, so
`fputc(sum << (8-ptr))` shifts the first-written bit to position 7. **The data is
left-aligned and the unused *low-order* bits are zero.** Prediction: `flat128`'s 3132 bits
are 391 whole bytes plus 4, and byte 392 has its low nibble zero.

### P3.5 — a Python decoder written from the spec will match `decmars` byte-for-byte

The iterative decoder is 2:1 averaging, one multiply-add, `bound(0.5 + v)`, truncation.
Python floats are IEEE-754 binary64 and so are the C's, so **prediction: exact equality on
every fixture, zero differing pixels.**

**The one thing that could break it, named in advance:** the pinned build flags
(`-O2 -fno-fast-math -fno-unsafe-math-optimizations`) do **not** disable FP contraction, so
clang on aarch64 is free to emit an `fmadd` for `pixel * trans->alfa + trans->beta`. If it
does, the C result differs from the Python one in the last bit, and `bound(0.5 + x)`
truncation will occasionally turn that into a whole grey level. So the fallback prediction
is: differences, if any, are **exactly ±1**, affect **< 0.1%** of pixels, and disappear when
the Python side uses `math.fma`. Anything else — a difference of 2, or a structured region
of differences — means the geometry is wrong, not the arithmetic.

### P3.6 — no fixture dimension trips the `ceil(log2(·))` floating-point edge

`bits_per_coordinate` is `ceil(log(dim / SHIFT) / log(2.0))` in binary64 with an integer
division first, and `virtual_size` is `1 << ceil(log(max_dim) / log(2.0))`. For exact
powers of two this is one ULP away from returning a value one too large. Prediction: for
every fixture dimension and every `SHIFT` in the golden set, the double-precision result
equals the exact integer ceiling — so the hazard is real but unarmed here, and the spec
should state the closed-form integer rule rather than the float expression.

---

## 2026-09-13 · Step 3 · outcomes

Recorded after generating 142 golden fixtures and validating every one of them with an
independent Python implementation written from `docs/mars1-format.md`.

### P3.1 — **confirmed**

The Python decoder reads `dom_x` as a row offset and reproduces `decmars -i` byte for byte
on all 142 fixtures. Had the axis reading been wrong the images would have been transposed
garbage. The plan's "preserve the bug exactly" is corrected to an axis convention in
`mars1-format.md` §0 and recorded as D12.

### P3.2 — **confirmed exactly, all four numbers**

| fixture | predicted size-1 | measured | predicted size-2 | measured |
|---|---:|---:|---:|---:|
| `mixed_129x127` | 255 | **255** | (not predicted) | 64 |
| `mixed_250x250` | 0 | **0** | 249 | **249** |
| 256² and 512² fixtures | 0 | **0** | 0 | **0** |

And the invariance claim held in the strong form: 255 and 249 are identical across all
three methods, all three rates, and the `alfa5` / `beta6` / `step8` / `max32` variants —
and 255 survives `-m 2`, because the forced branch never consulted `min_size`.

The `tip == 0` path is real: those 255 leaves each carry a raw pixel truncated to 7 bits.

### P3.3 — **confirmed, all six numbers**

`flat128` at 1998 defaults: 392 bytes, 256 transforms, 256 of them DC-only, every `qbeta`
= 64, decodes to a constant **129** in both modes, MSE exactly 1 → PSNR 48.1308 dB. The
prediction had six independent ways to fail and took none of them, which is the strongest
evidence available that the bit accounting, the `best_beta` override and the `bound()`
truncation are all stated correctly. All six are now asserted by `just gate-3`.

### P3.4 — **confirmed**

3132 bits = 391 whole bytes plus 4. The final byte is `0x10` = `0001 0000`: four data bits
left-aligned, low nibble zero. The plan's "left-padded" is backwards.

### P3.5 — **confirmed in the strong form; the named hazard did not fire**

The Python decoder reproduces `decmars -i` **byte for byte on all 142 fixtures**, so the
fallback prediction (±1 differences from `fmadd` contraction) was not needed: clang did not
contract the expression, and the gate asserts exact equality rather than a tolerance, so a
future toolchain that does contract will fail loudly.

One thing the prediction did not anticipate and the implementation forced into the open:
`0.5 + pixel * alfa + beta` associates **left**, so the 0.5 joins the product before beta.
Computing `(pixel·alfa + beta) + 0.5` instead is a different binary64 value, and the
truncation in `bound()` turns some of those into a whole grey level. Written into
`mars1-format.md` §10.1. Getting this wrong would have produced exactly the "off by one on
a handful of pixels" signature the prediction attributed to FMA — that is, the prediction
named the right *symptom* and the wrong *cause*, and would have sent a debugger to check
the compiler before checking the operator precedence.

### P3.6 — **confirmed**

All 35 combinations of `dim ∈ {64, 127, 129, 250, 256, 512, 768}` and
`SHIFT ∈ {2, 4, 8, 16, 32}` agree with the exact integer ceiling. The spec states the
integer rule, so the hazard cannot arm itself later.

### An unpredicted result worth more than most of the predictions

Checks 1–3 of the validator (transform count, DC count, byte-exact re-serialisation) were
argued in §13 to be blind to the child-recursion order. That argument was tested rather
than trusted: a mutant validator using NW,NE,SW,SE instead of NW,SW,NE,SE was run over the
whole set.

- **106 of 142 fixtures caught it.** 92 by the decoded-image check, 46 by the partition
  render, 14 by the domain-bounds check in §6 (overlapping).
- **Checks 1–3 caught it zero times**, exactly as §13 claims.
- **36 fixtures did not catch it at all** — including `sierpinski`, whose partition is
  symmetric under transpose, so swapping the SW and NE children is genuinely invisible.

The last line is the useful one. A single golden fixture would have had a real chance of
passing a transposed parser, and `sierpinski` — the most obviously "fractal" image in the
set — is one of the ones that would have. The size of the set is not decoration.

---

## 2026-09-13 · Step 4 (not yet run) · anchor codecs

**Provenance of these predictions.** Written down by the orchestrating session before the
anchor sweep was run, per the plan's own Step 4 exit criterion ("write down the expected
verdict before looking"). Recorded here (rather than only in the orchestrator's own
context) so the agent executing Step 4 has something concrete to compare its measurements
against, per this project's working agreement that predictions precede measurement (A4).

### P4.1 — Mars 1 sits below JPEG over most of 0.1–2.0 bpp

This is the plan's own stated exit-criterion pre-registration (§ Step 4): fractal coding at
1998 defaults is expected to need more bits than commodity JPEG for the same quality, over
most of the target bpp range, for every one of the six Mars 1 methods.

### P4.2 — anchor ranking by BD-rate against JPEG: JPEG < JPEG2000 ≈ WebP < AVIF ≈ JPEG XL

Expected ordering from worst to best compression efficiency (BD-rate against JPEG as
reference, more negative is better): JPEG (0%, reference) is worst; JPEG 2000 and WebP are
expected to land close to each other, both meaningfully ahead of JPEG; AVIF and JPEG XL are
expected to be the two strongest anchors and roughly tied with each other, consistent with
their shared generation (AV1-intra and a modern DCT/Modular hybrid respectively, both
newer than JPEG 2000/WebP).

### P4.3 — bpp-floor reachability: JPEG and WebP cannot reach 0.1 bpp on Kodak; AVIF, JPEG XL and JPEG 2000 can

`cwebp` and the mozjpeg-equivalent JPEG anchor are expected to be unable to push any Kodak
image down to 0.1 bpp within a sane quality-parameter range, because both are built on an
8×8 block-DCT design without the strong low-bitrate tooling (large transforms, better
entropy coding) the three newer formats have. `avifenc`, `cjxl` and `opj_compress` are
expected to reach 0.1 bpp on at least some, if not most, Kodak images.

---

## 2026-09-13 · Step 4 · outcomes

Recorded after running the full anchor sweep (`configs/anchors.json`, all 5 codecs × 24
Kodak images, 1536 rows in `results/anchors.jsonl`) and generating `results/anchors.md` /
`results/anchors.html`. Two real harness/config issues were found and fixed before these
numbers were trustworthy — see `docs/decisions.md` D18 (OpenJPEG's PNG reader silently
darkens every pixel via the source's `gAMA` chunk; fixed by routing JPEG 2000 through PNM)
and D19 (two extreme-low-quality non-monotonic points in JPEG and AVIF, and a JPEG XL
low-bitrate coverage gap; fixed by adjusting the swept quality/distance floors, not by
touching the BD-rate monotonicity check). Per §2.1's order of investigation, both were
confirmed as real, reproducible, out-of-harness phenomena (byte-for-byte pixel comparisons
outside `mars-bench` entirely) before any code changed.

### P4.1 — **confirmed**

Every one of the six Mars 1 methods has a **positive** per-image-summarised BD-rate
against JPEG (needs more bits for the same quality), over the full 24-image Kodak corpus,
computed by `marsbench anchors-report`:

| Mars 1 method | BD-rate % vs JPEG (corpus-curve) |
|---|---:|
| saupe | +13.39 |
| saupe-fisher | +21.56 |
| masscenter | +26.90 |
| hurtgen | +28.86 |
| fisher | +35.48 |
| mc-saupe | +54.61 |

All positive, all over the PSNR overlap 25–31 dB (stated per §M3), against JPEG's own
27–37 dB range — Mars 1's usable quality range does not even reach as high as JPEG's does
on this corpus at any of the five swept rates. The prediction holds without qualification.

### P4.2 — **confirmed in its coarse shape, refuted in its specific pairings**

Measured per-image BD-rate against JPEG, mean over 24 images:

| codec | mean BD-rate % vs JPEG |
|---|---:|
| AVIF | −49.59 |
| WebP | −41.85 |
| JPEG 2000 | −35.58 |
| JPEG XL | −33.47 |
| JPEG | 0 (reference) |

JPEG being the worst anchor is confirmed. Everything else about the predicted ordering is
not: JPEG 2000 and WebP are not close (a 6.3 percentage-point gap, and WebP is clearly the
stronger of the two, not the weaker as informally expected from format age); and AVIF and
JPEG XL are not tied — AVIF is the single best anchor by a wide margin (16 points ahead of
the next-best), while JPEG XL is actually the **weakest** of the four modern anchors, not
the strongest. This was investigated before being accepted (§ verification-discipline —
"assume the harness, not the codec"): the JPEG XL curve's PSNR/bpp overlap with JPEG was
checked directly and is not an artefact of too few points or a bad reference range (16
points, PSNR overlap 21.8–40.8 dB). The most likely real explanation is that `cjxl`'s
default effort/heuristics target Butteraugli (a perceptual distance), not raw PSNR, so a
PSNR-only BD-rate — which is what §M2 defines and this table reports — is not the metric
JPEG XL's encoder is tuned against, whereas aom's AVIF encoder does comparatively well on
raw PSNR. This is a known, previously reported asymmetry in the codec literature and not
specific to this harness; a future SSIM/MS-SSIM-based BD-rate table (both already computed
per row and available in `results/anchors.jsonl`) would be the natural follow-up were this
ranking to matter for a downstream decision.

### P4.3 — **confirmed for JPEG and AVIF; refuted for WebP; not reached for JPEG 2000 within the tested range; marginal for JPEG XL**

Fraction of the 24 Kodak images on which each anchor's swept range reached ≤ 0.1 bpp:

| codec | images reaching ≤ 0.1 bpp | global minimum bpp |
|---|---:|---:|
| JPEG | 0 / 24 | 0.154 |
| JPEG 2000 | 0 / 24 | 0.118 |
| WebP | 13 / 24 | 0.059 |
| AVIF | 19 / 24 | 0.056 |
| JPEG XL | 1 / 24 | 0.096 |

JPEG's inability to reach 0.1 bpp is confirmed (its lowest usable quality setting on this
codebase's swept range still lands above 0.15 bpp on every image). AVIF reaching it easily
is confirmed. **WebP is refuted**: `cwebp -q 0` reaches well below 0.1 bpp on the majority
of Kodak images (as low as 0.059 bpp), which the prediction did not expect — WebP's
low-bitrate behaviour turns out to be much closer to AVIF's than to JPEG's, another data
point against grouping "older" and "newer" formats the way P4.2 also assumed. **JPEG 2000
did not reach 0.1 bpp anywhere** at the compression ratios swept here (up to `-r 200`);
this is a statement about the tested range, not a structural limit — `opj_compress` almost
certainly could be pushed lower with a higher `-r`, but doing so was not needed to satisfy
the ≥6-points-in-0.1–2.0-bpp gate and was not attempted, so "expected able to reach it" is
recorded as refuted **for the range actually swept**, not as a claim about OpenJPEG's
absolute floor. **JPEG XL is a marginal confirm**: it reaches ≤ 0.1 bpp on exactly one
image, and only after D19 extended the distance sweep up to 21 (its allowed range goes to
25); the prediction was technically right but by a much thinner margin than expected.

---

## 2026-09-14 (not yet run) · Step 5 · `.ifs` reader + baseline bridge

### P5.1 — the Rust parser will reproduce every golden fixture's transform count exactly

`docs/mars1-format.md` was independently validated by a from-scratch Python parser at
Step 3 (`just gate-3`, all 142 golden fixtures, transform count and DC-leaf count both
exact). A Rust port written from the same document, with no reference to the Python or the
C, should therefore also recover `Number of transformations` exactly on all 142 fixtures —
this is now a test of the *port*, not of the *spec*, since the spec's own correctness was
already the Step 3 result.

### P5.2 — the Rust iterative decoder will agree with `decmars -i` to well under the 0.1 dB
budget, not merely inside it

`just gate-3`'s check 5 already reproduces `decmars -i` output **byte for byte** with a
NumPy decoder on this toolchain (no `fmadd` contraction observed). Rust's `f64` arithmetic
should behave identically to C's on the same hardware, so I expect the Rust decode to also
match byte-for-byte on most or all fixtures, which would put the PSNR-vs-ground-truth
agreement at 0.00 dB, not merely under 0.1. The 0.1 dB budget in the brief reads as
headroom for a toolchain difference that Step 3 already showed does not materialise here,
not as an expected gap.

### P5.3 — if any fixture disagrees, it will be one with size-1 leaves

`mixed_129x127` and `mixed_250x250` are the only golden images with leaves below
`min_size` (§5.3, down to size 1), and size-1 leaves store a raw pixel truncated to
`N_BITBETA` bits with no domain reference — the one payload shape that is pure table
lookup rather than arithmetic. If there is a disagreement, `bound()`'s post-clamp
truncation on a value very near a `.5` boundary is the most likely place: `(0.5 + v)`
computed in a different instruction order between `rustc` and `clang` could round a
different way on the rare pixel that lands exactly there. I do not expect this to happen —
P5.2 predicts exact agreement — but if it does, this is where.

---

## 2026-09-14 · Step 5 · outcomes

`just gate-5` runs both checks over all 142 golden fixtures, decoding each one with both
the Rust reader and `decmars -i` and comparing PSNR-against-ground-truth.

### P5.1 — **confirmed exactly**

142/142 fixtures parse to the exact transform count `encmars` printed. No mismatches.

### P5.2 — **confirmed, and at the predicted margin, not merely inside it**

`max |ΔPSNR| = 0.0000 dB` across all 142 fixtures, including `mixed_129x127` and
`mixed_250x250` — i.e. the Rust reconstruction and `decmars -i`'s own reconstruction agree
with the original image to identical PSNR everywhere `f64` printing distinguishes, on this
toolchain (`rustc`, like the pinned `clang`, does not contract `0.5 + d*alfa + beta` into an
`fmadd`; both languages evaluate the double-precision expression term-by-term as written).
This is the outcome P5.2 called: the 0.1 dB budget in the brief was headroom for a
divergence Step 3 already showed does not occur here, and it did not occur in the Rust port
either.

### P5.3 — **not reached**

No fixture disagreed, so there was nothing to localise. Recorded as moot rather than
confirmed or refuted — the size-1-leaf hypothesis was never exercised.

---

## 2026-09-14 (not yet run) · Step 6 · exhaustive Rust encoder

The integer-moment exactness criterion (`cargo test -p mars-codec`, 1e6 random blocks) and
the `flat128`/`mixed_129x127` known-answer tests were already run during implementation —
that is ordinary development iteration, not the measurement this section predicts. What
has **not** yet run is the corpus-level `just gate-6` check: decode agreement against
`decmars` and the RD-curve comparison against the Step 2 Fisher baseline, both over
`corpus/fixtures.images.json` at `rms = [2, 4, 8, 16, 32]`, plus the f32-vs-f64 divergence
measurement.

### P6.1 — decode agreement will again land at ~0.00 dB, not merely under the 0.1 dB budget

Step 5 already established that this Rust toolchain's `f64` arithmetic agrees with the
pinned `clang` build's, exactly, on the iterative decode formula. Encoding with a different
search (exhaustive instead of Fisher) does not change that: gate-6's decode-agreement check
compares two decodes *of the same Rust-produced bitstream* (one via `mars_codec::ifs`, one
via `decmars -i`), so it exercises the identical decode-side arithmetic Step 5 already
found exact. I expect `max |ΔPSNR| = 0.0000 dB` again.

### P6.2 — the RD curve will be *at least* as good as Fisher on every image, and the real
risk is failing the 0.2 dB band from the *favourable* side

Exhaustive search is a strict superset of Fisher's classified candidate set, so at matched
settings it cannot do worse per block. On natural-content images I expect the margin to be
small (a few hundredths of a dB) and comfortably inside ±0.2 dB. But three of the twelve
fixtures — `checker8`, `impulse`, `noise_u8` — are adversarial-for-classification
synthetic images specifically included to stress search methods; Fisher's domain
classifier may do considerably worse than exhaustive on these, which could push BD-PSNR
*above* +0.2 dB (a fail, but in the direction of "Rust is much better," not "Rust is
broken"). If that happens, the fix is not a tolerance change — it is a documented decision
that the criterion checks for regression (Rust worse than Fisher), not for the ceiling of
exhaustive search's legitimate advantage on pathological images.

### P6.3 — the f32 fit will diverge above the 0.1% adoption threshold, not below it

`Σ D²` reaches ~1.07 G at 32x32 (~67 M already at 16x16, the size that dominates this
grid), well past `f32`'s ~16.7 M exact-integer range, so `s2`'s cast to `f32` loses real
low-order bits before the fit even runs. I expect this to move `qbeta` (7 bits, more
sensitive to a small `beta` shift than `qalfa`'s 4 bits are to `alfa`) on enough
domain-referencing leaves to clear 0.1%, meaning the brief's decision rule keeps `f64` on
CPU rather than adopting `f32`. I do not have a specific percentage prediction beyond
"more than 0.1%, plausibly a few percent."

---

## 2026-09-14 · Step 6 · outcomes

`just gate-6` runs the exhaustive encoder over `corpus/fixtures.images.json` at
`rms = [2, 4, 8, 16, 32]`, cross-checks every point's decode against `decmars -i`, compares
the resulting RD curve against the Step 2 Fisher baseline, and measures f32-vs-f64 fit
divergence over every domain-referencing leaf produced.

### P6.1 — **confirmed exactly**

`max |ΔPSNR| = 0.0000 dB` across all 60 (image, rms) points. Same finding as Step 5, for
the same reason: this checks two decodes of the same Rust-produced bitstream, and this
toolchain's `f64` arithmetic already agreed with `decmars`'s at Step 5.

### P6.2 — **confirmed on the mechanism, wrong on the specific failure mode**

The predicted risk was real but showed up differently than expected: rather than Fisher
losing badly enough on `checker8`/`impulse`/`noise_u8` to blow through +0.2 dB, **11 of the
12 fixtures' Fisher curves turned out to be non-monotonic or absent** — several synthetic
images never cross their own split threshold across the whole rms grid, so Fisher's curve
has repeated identical points and fails `bdrate`'s §M3 precondition before BD-PSNR can even
be computed. Only `mandelbrot` produced a comparable curve: **+1.1577 dB**, comfortably
inside the (one-sided) floor, and positive as the dominance argument predicts. See
`docs/decisions.md` D20 for the full finding and the resulting floor-not-band gate design.

### P6.3 — **refuted, and by a wide margin**

**0 of 69,573** domain-referencing leaves picked a different `qalfa` or `qbeta` between
`fit_f32` and `fit_f64` — not "under 0.1%," an exact zero over a six-figure sample. The
predicted mechanism (`Σ D²`'s cast to `f32` losing real bits) is real and was not the
question that mattered: the quantisers are only 4 and 7 bits wide, and `f32`'s relative
precision is many orders of magnitude finer than either quantisation step, so the lost
bits never cross a rounding boundary. `docs/decisions.md` D21 records the finding and why
the actual precision switch is deferred to Step 7 regardless (Metal has no fp64, so Step 7
needs to answer the closely related but distinct question of whether an f32-*driven
search* — not just an f32 refit of an f64-found winner — agrees with f64 on a real
corpus).

---

## 2026-09-14 · Step 7 (not yet run) · GPU exhaustive search

### P7.1 — 32-bit unsigned accumulators suffice; no 64-bit emulation is needed

The brief flags 64-bit accumulator support as a risk to check "before writing the
kernel," and WGSL has no native 64-bit integer type, so the fallback it names is
splitting `s2`/`t1`/`t2` across two 32-bit lanes. I predict that fallback will not be
needed at all: every one of the six moments is a sum of non-negative terms (`D >= 0`,
`r >= 0`), so there is no cancellation to worry about, and the project's actual configs
(`configs/*.json`) cap `max_size` at 32 — exactly the block size the brief's own table
uses for its worst case. At that size the largest moment, `s2_x16 ~= 1.07e9`, still sits
comfortably under `u32::MAX` (`~4.29e9`), and every other moment is smaller. I expect a
single `u32` accumulator per moment on the GPU to match the CPU's `i64` values exactly
(as integers, not merely as compared floats) over the full fixtures/standard corpus, with
room to spare rather than a near-miss.

### P7.2 — the f32-driven GPU search will match the f64 CPU search's winning candidate
on effectively 100% of range blocks, extending D21 rather than reversing it

D21 measured zero divergence for an f32 *refit* of an f64-found winner, but flagged the
full f32-*driven search* on real photographic content as the genuinely open question,
since a search compares thousands of near-tied candidates per block rather than
recomputing one. I predict the same mechanism D21 found still dominates: `qalfa`/`qbeta`
are only 4 and 7 bits wide, and `f32`'s relative precision is many orders of magnitude
finer than either quantisation step, so an `f32` rounding difference essentially never
flips which of two candidates has the lower `rms` closely enough to change which one a
search keeps as its running best — even across a full exhaustive sweep on Kodak/`standard`,
not just the fixtures corpus. I expect the GPU's chosen `(dom_row, dom_col, isometry,
qalfa, qbeta)` to match the CPU's on every block, gated on the search enumerating
candidates in the same order (`dom_row` outer, `dom_col` inner, isometry innermost,
strict `<` keeps the first candidate on an exact tie) so a genuine exact tie resolves
identically rather than becoming a coin flip. If any block does diverge, I expect it to be
a handful out of the corpus, not a systemic fraction, and traceable to an actual near-exact
tie rather than to widespread `f32` drift.

### P7.3 — the speedup clears 50x and the Kodak oracle build lands in the "hours" band,
not merely "faster than weeks"

This machine's GPU has no host<->device transfer to amortise (unified memory), and the
exhaustive search's regular, branch-free access pattern (a dense `range x domain x
isometry` sweep) is close to the ideal case for a GPU. I expect the measured speedup
against the Rayon CPU exhaustive path, transfer included, to clear the 50x floor with
margin (not land just above it), and a full Kodak oracle build (24 images, the sizes/rates
Step 8 needs) to complete in low single-digit hours rather than needing the full "hours"
budget the exit criterion allows.

---

## 2026-09-14 · Step 8 (not yet run) · Oracle cache + recall harness

### P8.1 — the smallest block size dominates oracle-build cost by an order of magnitude,
not evenly across sizes

M6's config tuple is `(image, min_size, max_size, SHIFT, ...)`, and the actual encoder
partition can emit leaves at every power-of-two size in `[min_size, max_size]`, so "build
the oracle for a config" means an exhaustive sweep at every such size, not one. Per-size
work is roughly `domain_positions(size) x image_area` (range-block count and per-candidate
`size^2` cost cancel), and `domain_positions(size)` shrinks only mildly as `size` grows
(the domain window is `2*size`, so a larger window rules out proportionally fewer positions
than the drop in candidate count from having a coarser grid would suggest) — meaning the
*smallest* configured size dominates total cost by roughly the ratio of domain-position
counts between the smallest and largest size in the range, which back-of-envelope math on
Kodak's 768x512 puts at roughly an order of magnitude between `size=4` and `size=16`. I
therefore expect min_size=4 to make a full-corpus, three-config oracle build impractical
within this session, and the practical choice (to be recorded as a decision if adopted) is
to fix `min_size=8` across all three configs, trading away the smallest size's oracle
coverage for a build that actually finishes.

### P8.2 — the top-32 kernel costs meaningfully more than Step 7's top-1 kernel, but not
proportionally to 32x

Per-thread bounded insertion into a 32-element sorted array, plus a workgroup-level
pairwise merge-reduce, is more register/shared-memory pressure than Step 7's single
running-best comparison, and could reduce occupancy further on top of D24's already-below-
peak throughput. But the dominant per-candidate cost (the moment accumulation and fit) is
unchanged, and insertion into an already-sorted-descending array is O(1) amortised once
the array fills with any reasonably selective early candidates (only candidates better
than the current worst-of-32 do any work beyond a single comparison). I expect the top-32
kernel to run 1.5-3x slower than Step 7's top-1 kernel at the same image/size, not 32x
slower and not within noise of Step 7's numbers.

### P8.3 — the self-test passes trivially, because it has no external ground truth to
diverge from

Unlike Step 7's differential test (GPU vs. an independently-computed CPU reference),
gate-8's "exhaustive self-test" checks the GPU oracle's own top-32 list against its own
separately-reported top-1 winner (candidate at merge-reduce rank 0 equals the
independently tracked global-best winner) — there is no second, independent computation in
this check. I expect this to pass at exactly 100% top-1 recall and 0 dB regret by
construction *unless* the top-32 merge-reduce itself has a bug (e.g. losing the true
minimum during a pairwise merge step), in which case I'd expect a very small number of
blocks to fail outright (rank-0 not matching the tracked global best) rather than a
graded/partial regret — a merge-reduce topology bug is a binary "the true best survived the
tree or it didn't" failure mode, not a numerical near-miss like D23's.

---

## 2026-09-14 · Step 8 · outcomes

`marsbench oracle-build` ran the full `standard/` corpus (24 Kodak images) at all three
`configs/oracle.json` variants; `marsbench oracle-check` re-ran an independent top-1 GPU
computation (a different kernel entry point/workgroup topology, `main` vs. `main_top32`)
against every cached block's rank-0 entry; `marsbench recall`'s self-test scored the
oracle's own top-1 picks against itself.

### P8.1 — **confirmed on the mechanism (min_size=8 was the right call), wrong on the
magnitude of "impractical"**

`min_size=8` across all three configs was adopted as predicted (`configs/oracle.json`,
D26 below). But the actual cost was nowhere near the "hours per config" the brief's own
estimate suggested, or even the "low-to-mid single-digit hours" framing this document
used for Step 7's oracle-build projection: **the full 24-image x 3-config build (69
`(image, config)` pairs, every size in each config's range) completed in 197.5 seconds** —
under 3.5 minutes wall-clock, on the same M3 Pro. `oracle-check`'s independent
re-verification of all 562,176 cached blocks took a further ~156s. The 481MB cache is well
within reason. I did not test whether `min_size=4` would in fact have been "impractical" —
that remains untested — but the specific claim "the full-corpus 3-config build needs
scope-narrowing to finish in a session" was true only in the weak sense that *some*
narrowing (min_size=8, already the plan) sufficed; the "practical" build is fast enough
that a future step revisiting `min_size=4` should just measure it rather than assume it is
still out of reach — D24's 11-23x GPU speedup is doing more work here than this document
gave it credit for.

### P8.2 — **directionally confirmed, at the low end of the predicted range**

Comparing `oracle-build`'s per-size top-32 timings against `gate-7`'s own top-1 numbers on
the same machine: at `size=16`, top-32 took ~1.4-1.5s vs. top-1's ~1.1-1.2s (~1.2-1.4x, just
under the predicted 1.5x floor); at `size=32`, top-32 took ~1.4s (in the `max32` config)
vs. top-1's ~1.0s (~1.4x, also just under 1.5x). Neither point reached the predicted
1.5-3x band, but both are close to its lower edge and clearly not "32x" or "within noise" —
the prediction's *shape* (meaningfully slower, not proportionally-to-32 slower) held; its
specific numeric floor was set a little high.

### P8.3 — **confirmed exactly**

`oracle-check`: **562,176 of 562,176 blocks (100%) agree** between the cached top-32
rank-0 entry and an independently-computed top-1 GPU result — 0 mismatches, exactly as
predicted for a correct merge-reduce. `marsbench recall`'s self-test (oracle's own top-1
picks scored against itself) reported 100.00% top-1/top-5/top-32 recall and exactly 0.0000
dB mean/median/p95 regret on every cache checked — also exactly as predicted.
