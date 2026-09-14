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

---

## 2026-09-14 · Step 9 · prediction (written before any classical-method code runs)

Six classical speed-up methods (Fisher, Hurtgen, MassCenter, Saupe, Saupe-Fisher,
Mc-Saupe) plus `Exhaustive` are being ported into a new `mars-search` crate behind
`CandidateRetriever`, driven against `standard/`'s oracle cache (Step 8) and against
Mars 1's own six `-F/-X/-C/-S/-Z/-Y` binaries (already wrapped by `mars-bench::mars1`,
Step 2). Predictions, per the classical fractal-search literature this brief cites:

### P9.1 — recall/cost ranking

Expect, from best to worst top-1 recall *at comparable evals/transform*:
**Saupe-Fisher ≈ Fisher > Saupe > Mc-Saupe > MassCenter > Hurtgen**, with `Exhaustive`
at 100%/0dB by construction (a harness bug otherwise, same as gate-8's self-test). Fisher
and Saupe-Fisher canonicalise the domain's orientation before indexing, which lets both
compare directly to the range's own canonical class — I expect this buys them the best
recall per candidate examined, since the restriction is "same shape class" rather than a
geometric proximity heuristic. Hurtgen's 4-bit quadrant-mean-sign classifier is coarse (16
buckets vs. Fisher's 3×24 = 72, and no domain-orientation canonicalisation, so it burns
8x the isometry loop at coding time) — I expect it to have the worst recall of the six,
though not necessarily the worst evals/transform, since a coarse classifier with few
buckets can still produce short candidate lists. MassCenter's expanding-ring polar search
is a geometric proxy (center-of-mass angle), not a shape-equivalence class, so I expect
middling recall, similar to or slightly worse than Saupe. Mc-Saupe combines MassCenter's
ring search with a per-cell k-d tree — I expect this to land between MassCenter and Saupe
on recall (finer-grained matching than MassCenter's raw ring, but fragmented per-cell
trees are individually weaker than Saupe's one shared tree).

### P9.2 — evals/transform ranking

Expect **Fisher and Saupe-Fisher to have the lowest evals/transform** (single restricted
bucket / single k-d tree query per range block, no per-isometry loop), **Hurtgen,
MassCenter, and Mc-Saupe higher** (8x isometry loop at coding time each), and **plain
Saupe** in between (8x isometry loop, but each iteration only costs one k-d tree query
against a global tree rather than a linked-list scan). I expect all six to sit well below
`Exhaustive`'s evals/transform (which scans every legal domain position at every size),
plausibly by 1-2 orders of magnitude, consistent with why these methods existed at all in
1998 on much slower hardware.

### P9.3 — RD curves

Expect all six methods' RD curves to sit strictly below `Exhaustive`'s (the restricted
candidate set can only ever match or lose to the true best) and within the plan's 0.2 dB
tolerance of Step 2's own Mars 1 baseline for the same method, since both are computing
the same restricted search — any divergence bigger than that would point to a porting bug
(a missed isometry-index composition, wrong ordering-table row, or an off-by-one in the
expanding-ring wraparound) rather than a real algorithmic difference, given search
*order* independence was the only thing the framing note in the Step 9 brief said was
allowed to differ.

### P9.4 — the harness sanity check

Expect `Exhaustive` run through the new `mars-search` driver to reproduce `mars-codec`'s
existing Step 6/7 exhaustive search bit-for-bit (same winner per block, same evals count)
and to score ~100% top-1 recall / 0 dB regret against the Step 8 oracle, exactly like
`gate-8`'s own self-test — any other outcome means the new driver's partition or moments
disagree with the oracle's own config, which the brief says to fix before trusting any of
the other five methods' numbers.


---

## 2026-09-14 · Step 9 · outcomes

`mars-search` (crates/mars-search) implements `CandidateRetriever` for `Exhaustive` plus
all six classical methods, a bucketed k-d tree ported from `nn_search.c`, and a shared
quadtree driver (`mars_search::encode_image`) mirroring `mars_codec::encode`'s partition.
Measured on `kodim01`/`kodim02` (min_size=8/max_size=16/shift=4 oracle `default` config
for recall; min_size=4/max_size=16/shift=4/t_rms=8.0 for the evals/transform-vs-C check,
matching an existing `results/baseline-mars1.jsonl` row) -- see `docs/decisions.md` D28
for exactly why this is 2 images, not the full 24-image corpus, and why the C reference
comparison reuses Step 2's already-captured numbers instead of re-running `reference/mars1`
in this session (no working C toolchain in this sandbox).

### P9.4 — **confirmed exactly**

`Exhaustive` run through the new driver, scored against the real Step 8 oracle cache at
`kodim01`'s `default` config: **100.0000% top-1/top-5/top-32 recall**, mean regret
`-1.89e-7 dB` (floating-point noise around exactly 0) over 1536 matched blocks. Exactly
the harness sanity check predicted -- the new driver's partition and moments agree with
the oracle's own config.

### P9.2 — **confirmed in shape, values differ from the naive expectation**

evals/transform against the real C-binary numbers (`results/baseline-mars1.jsonl`,
`kodim01`, `min_size=4/max_size=16/shift=4/t_rms=8.0`): Fisher 774.2 (C: 773.9, diff
0.04%), Hurtgen 2028.4 (C: 2026.2, 0.11%), MassCenter 1368.2 (C: 1363.8, 0.33%), Saupe
522.3 (C: 522.3, 0.01%), Saupe-Fisher 65.3 (C: 65.3, 0.00%), Mc-Saupe 153.4 (C: 146.3,
4.86%) -- **all six inside the brief's 5% tolerance**, five of them far inside it. This is
tighter agreement than the 5% tolerance's own framing ("looser than revision 1's 1%, since
exact search-order reproduction is no longer required") anticipated needing -- the ported
classification logic reproduces the reference's *candidate-set sizes* almost exactly, not
merely "in the right ballpark." Mc-Saupe's larger (but still passing) 4.86% gap is the one
value close to the tolerance edge; not investigated further this session, flagged here
rather than left unremarked.

### P9.1 — **falsified: plain Saupe has by far the best recall, not Fisher/Saupe-Fisher; Mc-Saupe is the worst, not middling**

Measured top-1/top-5/top-32 recall (`kodim01`/`kodim02`, oracle `default` config,
`marsbench classical-methods`, `results/classical-methods-sample.jsonl`):

| method | top-1 % | top-5 % | top-32 % | mean regret dB |
|---|---|---|---|---|
| saupe | 51.55 / 45.93 | 86.81 / 76.86 | 98.65 / 95.31 | 0.20 / 0.19 |
| masscenter | 20.37 / 20.64 | 43.13 / 40.44 | 73.59 / 69.29 | 0.71 / 0.60 |
| saupe-fisher | 21.14 / 19.33 | 52.26 / 43.97 | 85.05 / 73.99 | 0.65 / 0.59 |
| hurtgen | 18.21 / 16.35 | 46.73 / 41.67 | 74.06 / 69.85 | 0.70 / 0.63 |
| fisher | 10.71 / 9.82 | 31.28 / 27.39 | 65.97 / 61.88 | 0.95 / 0.85 |
| mc-saupe | 3.87 / 3.03 | 11.62 / 9.67 | 31.94 / 30.64 | 1.53 / 1.36 |

This is the opposite ranking from P9.1's prediction on the two points that mattered most:
plain Saupe (a single global k-d tree, looped over 8 range isometries) recalls the true
oracle optimum roughly **2.5-5x more often** than Fisher or Saupe-Fisher, and Mc-Saupe --
predicted to land "between MassCenter and Saupe" -- is instead the worst of all six by a
wide margin.

**Why this is very likely real and not a harness bug:** the evals/transform numbers above
(P9.2) already show each method's candidate-set *size* matches the C reference to within
a fraction of a percent for five of six methods -- if the classification/bucketing logic
were wrong, it would almost certainly perturb candidate-set sizes too, not just which
particular candidates get selected, and it does not. `Exhaustive`'s 100%/0dB self-test
(P9.4) also rules out a moments/partition bug in the shared driver both families run
through. So the likely explanation is architectural: Fisher/Saupe-Fisher restrict the
search to exactly *one* bucket chosen by a hard classification (quadrant-sum descending
order + a bubble-sort tie-break), so a range block whose true best domain match happens to
canonicalise into a *neighbouring* class (a small perturbation away in quadrant-sum
ordering) misses that domain entirely, with no fallback. Saupe's continuous feature-vector
k-d tree with `eps=2.0` degrades gracefully instead of hard-partitioning, which plausibly
explains both its much higher recall and (per P9.2) its correspondingly higher
evals/transform. Mc-Saupe compounds MassCenter's coarse angular binning with *small
per-cell* trees (each built from only the domains that landed in one `(cx,cy)` grid cell),
which would predict exactly the worst-of-both-worlds recall observed. This is a hypothesis,
not confirmed further this session -- recorded per the verification-discipline skill as a
surprise against a stated prediction, not rationalised into agreement after the fact.

**What this does not change:** every method's own evals/transform still matches its C
counterpart (P9.2), and Exhaustive's self-test still passes (P9.4) -- so the *search
mechanics* are validated. What's falsified is specifically the literature-derived
expectation that canonicalisation-based classification (Fisher's whole design point)
would dominate feature-vector nearest-neighbour search (Saupe's) on recall. Given this
was measured on only 2 of 24 corpus images (D28), the magnitude should be treated as
indicative rather than final -- but the *direction* (Saupe >> Fisher family on recall) is
unlikely to be a 2-image artefact given how large the gap is (2.5-5x, not a few percent).

### P10.1 — `.mars` v0's context-adaptive rANS vs. the raw `.ifs` bitstream, at identical reconstruction

Written before `just gate-10`'s sweep (`marsbench mars-format-check` over
`corpus/fixtures.images.json` x `rust_encoder::RMS_GRID`, launched in the background)
returned any output — a real prediction, not a rationalisation, per A4.

**What was built.** `crates/mars-entropy` (order-0 adaptive models per context, pure
integer fixed-point CDFs, encoded via `constriction`'s `AnsCoder` stack) and
`mars_codec::mars_format` (the `.mars` v0 container). Contexts actually implemented: split
flag and mode by size class only (**not** "depth + neighbour split state" as the brief
specifies — neighbour split state was scoped out this session, see the D29-adjacent entry
this prediction's outcome will be filed under); qalfa/qbeta/isometry by size class crossed
with DC-vs-domain mode; domain row/col as a zigzag delta from the *previous leaf's* domain
position in traversal order (not a real spatial predictor using a decoded neighbour's
position, just the previous leaf visited).

**Prediction:**
1. Every fixture round-trips losslessly (write -> read recovers the exact leaf list) —
   this should be unconditionally true or there is a decoder bug, not a tolerance question.
2. Mean bpp reduction vs. raw `.ifs` lands **below** the brief's 8-20% expectation,
   because that range presumably assumes the full neighbour-aware context the brief
   describes, and size-class-only conditioning is a strictly weaker model. Guessing
   **3-10%** — mostly from the mode bit (skipping a full qalfa field when DC-only, which
   is common at high t_rms) and from qalfa/qbeta's skew toward small values not being
   forced into a uniform 4-or-7-bit field. If it lands *above* 20% that would suggest a
   bug (most likely: the domain-delta predictor accidentally exploiting real spatial
   locality in these synthetic fixtures — `checker8`, `ramp_h`, `zoneplate` are exactly
   the kind of regular image where per-leaf domain positions could be highly
   autocorrelated even under a "previous leaf" predictor with no real neighbour
   awareness — worth checking whether the "outcome" turns out to be this session's fixture
   corpus being unusually favourable rather than the coding scheme itself).
3. Runtime is dominated entirely by `encode_image`'s exhaustive search (measured directly
   at ~12.5s per 256x256 image at `t_rms=2.0`, matching Step 6/9's known search cost) —
   `mars_format::write`/`read` themselves measured at under 2ms for a 4096-leaf image, so
   the gate's wall-clock time is not informative about this step's own code.

### P10.1 — **outcome: (1) confirmed; (2) falsified — reduction is far above the guessed range, and above the brief's own 8-20%, but the per-image breakdown resolves why**

`just gate-10` (`marsbench mars-format-check`) results, all 60 (image, rms) points:

1. **Confirmed exactly.** All 60 points round-tripped losslessly (`hdr == hdr2 &&
   leaves_sorted == leaves2`), zero mismatches.
2. **Falsified, in the surprising direction.** Mean reduction across all 60 points is
   **60.8%**, not 3-10%. But the per-image spread is enormous and tells the real story:

   | image | reduction | image | reduction |
   |---|---|---|---|
   | flat128 | 91.6% | sierpinski | 94.7% |
   | ramp_h | 74.7% | mandelbrot | 17.0-18.9% |
   | checker8 | 89.5% | zoneplate | 54.3-55.4% |
   | noise_u8 | 19.1% | mixed_250x250 | 25.0-25.6% |
   | impulse | 90.6-90.8% | mixed_129x127 | 25.5-27.3% |
   | step_edge | 82.4% | selfsim_iso | 53.3-70.5% |

   The two fixtures that most resemble real photographic complexity --
   **`noise_u8` (19.1%) and `mandelbrot` (17.0-18.9%)** -- land almost exactly inside the
   brief's predicted 8-20% band. Every other fixture is a synthetic pathological pattern
   (flat fields, a single ramp, a checkerboard, a single impulse, one hard edge, an exact
   fractal) with enormous run-length-style redundancy in its split/mode/domain-delta
   sequence, which is exactly the structure an adaptive context model is best at --
   `flat128` and `sierpinski` round to bpp so low (0.004, 0.017) that a few bits either way
   swing the percentage enormously. This mirrors `rust_encoder::gate`'s own documented
   pattern (D-series decisions): this corpus is deliberately full of pathological synthetic
   fixtures for gating, not a representative photographic sample, so a mean over all 12 is
   not the interesting number -- the two hardest fixtures matching the brief's own estimate
   almost exactly is the real signal, and it did so *despite* this session's simplified
   size-class-only context (not the brief's depth+neighbour scheme), which makes the
   prediction's reasoning (a weaker context model implies a smaller gain) directly
   falsified rather than merely off on magnitude: a weaker context model still recovered
   the brief's full expected gain on the two images where the brief's expectation actually
   applies. Filed alongside the scope note in `docs/decisions.md` D30.

---

## 2026-09-14 (not yet run) · Step 12 · Rayon parallelism

### P12.1 — bitstream identity holds at every thread count, with no extra work needed
beyond what determinism-by-default already guarantees

The brief's only hard exit criterion is bit-identical output across thread counts. I
predict this holds trivially once search is decoupled from emission: each range block's
candidate is a pure function of `(image, contracted, row, col, size, params)` with no
shared mutable accumulator across blocks (unlike, say, a running sum that could
reassociate under a different reduction tree), and the parallel decomposition changes only
*which thread* computes each block's search, never that block's own floating-point
arithmetic. Concatenating each quadrant's leaves in the same TL/BL/TR/BR order the
sequential walk already used should therefore reproduce the exact sequential bitstream at
1, 2, 4, 8, and 16 threads with zero special-casing — the differential test is expected to
pass on the first attempt, not after debugging a reordering bug.

### P12.2 — scaling is markedly sub-linear past 4 threads on this machine, and the
smallest range blocks are why

This machine's P-core count (per `benchmark-protocol`) is well below 16, so threads beyond
that count are E-cores or hyperthread-style oversubscription and I expect diminishing
returns there regardless of the encoder. More specific to this step: `min_size` in the
project's configs is typically 4-8, and Step 11's own finding (D31) was that the NEON
kernel's per-row overhead stops paying off at larger sizes, which suggests the reverse
problem here — at the smallest sizes, a `rayon::join`'s task-spawn/steal overhead is
large relative to the actual search cost of a 4x4 or 8x8 block. I predict the parallel
efficiency (speedup / thread count) measured at 2 threads will be the best of the sweep
(>= 80%), degrading past 4 threads to well under 50% by 16, and that most of the shortfall
traces to oversubscription on small blocks rather than to the sequential
`Contracted::build` prefix (which is a small fraction of total work at exhaustive-search
block counts). If parallel efficiency instead stays high (>= 70%) all the way to 16
threads, that would mean the search cost per block dominates task overhead by a wider
margin than expected, which would be worth noting as a reason *not* to bother with a
size-based parallel cutoff in later steps.

---

## 2026-09-14 · Step 12 · outcomes

`parallel_determinism` (`crates/mars-codec/tests/parallel_determinism.rs`) encodes
`mandelbrot` (512x512, RMS-driven) and `mixed_129x127` (forced-subdivision geometry) at
1/2/4/8/16 threads via scoped `rayon::ThreadPool`s and asserts the header, eval count, and
full leaf list are byte-identical to the single-threaded run at every count.

### P12.1 — **confirmed**, first attempt, no reordering bug

Both fixtures pass at every thread count with no changes needed beyond the merge-by-plain-
concatenation design itself. `just gate-12` is this test.

### P12.2 — **confirmed in direction and magnitude band, with one number better than
predicted**

This machine (Apple M3 Pro, `hw.perflevel0.physicalcpu`/`hw.perflevel1.physicalcpu` = 6/6,
12 total logical) measured, `mandelbrot` / `noise_u8`, median of 5 runs each, all times
relative to that image's own 1-thread run:

| threads | mandelbrot speedup | mandelbrot efficiency | noise_u8 speedup | noise_u8 efficiency |
|---:|---:|---:|---:|---:|
| 1  | 1.00x | 100.0% | 1.00x | 100.0% |
| 2  | 1.92x |  95.8% | 1.97x |  98.5% |
| 4  | 3.44x |  86.1% | 3.80x |  95.0% |
| 8  | 5.04x |  63.0% | 5.94x |  74.3% |
| 16 | 5.54x |  34.6% | 7.07x |  44.2% |

Predicted 2-thread efficiency >= 80% and "well under 50%" by 16: both hold, and 2-thread
efficiency (95.8-98.5%) is meaningfully higher than the >= 80% floor guessed at. The
predicted mechanism (P-core count, 6 here, as the point where returns start diminishing)
is directionally right — efficiency is still high at 4 threads (<= P-core count) and drops
sharply at 8 (> P-core count, onto E-cores/oversubscription) — but the prediction's other
named mechanism (small-block task-spawn overhead dominating at `min_size`) is not
distinguished from simple core-topology saturation by this measurement alone: both
fixtures use the same `min_size=4` / `PARALLEL_SIZE_CUTOFF=8` boundary, so this sweep
cannot tell whether a lower cutoff would recover more of the 8-16 thread range or whether
the ceiling is purely "this machine has 6 P-cores." Not investigated further this session
— worth a follow-up sweep over `PARALLEL_SIZE_CUTOFF` if a later step needs more headroom
above 6-8 threads. Filed alongside the full run command and machine fingerprint in
`docs/decisions.md` D32.

---

## 2026-09-14 · Gate C · prediction, before measuring

**Context.** Gate C asks for CPU encode speedup vs. Mars 1 at matched RD (target >= 20x),
NEON x threads, GPU excluded, with entropy/NEON/threads individually attributed. The
apples-to-apples comparison is Mars 2's Fisher-method encode (`mars_search::encode_image`,
same evals/transform algorithm as Mars 1's `-M f`) against `encmars -M f`, same
`EncodeParams` (the `default` baseline-mars1 variant: min/max 4/16, shift 4, bits 4/7,
max_alfa 1.0) at `t_rms=8.0` (the sweep's mid rate). Fisher's classified search was only
just parallelised this session (porting Step 12's split/rayon::join pattern into
`mars_search::walk`, since it previously only existed in `mars_codec::encode`'s exhaustive
path) -- `parallel_determinism` (mars-search) passes at 1/2/4/8/16 threads on `mandelbrot`.

### P-gate-c.1 — total CPU speedup

Mars 1's own Fisher indicative time on `kodim01` (768x512) was ~1.3s (`results/
baseline-mars1.jsonl`, though at an unrecorded `t_rms`). Fisher restricts the search to a
small classified candidate set (Step 9's own evals/transform table showed order-of-
magnitude fewer evals than exhaustive), so per-block work is small -- which cuts the other
way from Step 12's exhaustive-search parallel-efficiency measurement (P12.2): with less
work per block, `rayon::join`'s task-spawn overhead is a *larger* fraction of the total,
so I expect Fisher's thread scaling to be measurably worse than exhaustive's (P12.2 saw
3.44-3.80x at 4 threads) -- I predict 4-thread speedup in the 2.0-3.0x range for Fisher,
not matching exhaustive's band. Combined with a Rust-vs-C implementation/NEON factor I
expect to be substantial (Step 11's D31 measured 1.4-9.2x per kernel, and the C reference
carries 1998-era cache-unfriendly access patterns this project's `Contracted` box-sum
layout was designed around), I predict the **total measured speedup clears 20x**, but with
threads contributing a smaller share of that total than they did for the exhaustive gate,
and most of the total coming from the kernel/implementation side rather than from
threading. If threads instead contribute a similar multiplier to P12.2's exhaustive
measurement, that would mean per-block overhead is not actually the bottleneck I expect at
Fisher's evals/transform, which would be worth a note given how it cuts against P12.2's own
reasoning about the same `PARALLEL_SIZE_CUTOFF`.

### P-gate-c.2 — matched RD

Same algorithm, same `EncodeParams`, both implementations independently validated against
Mars 1's isometry/quantisation conventions (Step 6/9's own gates) -- I predict bpp and PSNR
match within Step 6's already-adopted 0.2 dB BD-PSNR floor, i.e. this is confirmatory, not
a new finding, and any divergence bigger than that would point to a params/config mismatch
in the harness rather than a real algorithmic difference.

### P-gate-c.3 — decode

Decode was never a bottleneck in this project (Step 2's baseline shows Mars 1 decode in
the tens of milliseconds); I predict Mars 2's decode is comparably fast and the "no slower
than Mars 1" criterion passes without needing its own optimisation work.

---

## 2026-09-14 · Gate C · outcomes: P-gate-c.1 and P-gate-c.2 both falsified

First run: `kodim01`, `runs=5` A/B interleaved, `t_rms=8.0` (the `default` variant), full
threads = 6 (this machine's P-core count), via the new `marsbench gate-c-bench`.

### P-gate-c.1 — **falsified**: measured speedup is 2.36x, below even the 5x kill floor,
not >= 20x

| | mars1 (median, ms) | mars2 full-thread (median, ms) | total speedup | mars2 1-thread (median, ms) | thread speedup |
|---|---:|---:|---:|---:|---:|
| kodim01 | 971.92 | 411.72 | 2.36x | 1948* | 4.73x |

(*derived from `thread_speedup = mars2_1thread / mars2_full = 4.73x` x 411.72ms.)

The predicted mechanism was backwards in direction: I expected the Rust/NEON
implementation factor to dominate and threading to be the weaker contributor (given
Fisher's smaller per-block work). Measured, thread scaling (4.73x at 6 threads, actually
*better* than Step 12's own 4-thread exhaustive-search number of 3.44-3.80x) is not the
bottleneck; the 1-thread Rust Fisher implementation itself (~1.95s) is *slower* than
Mars 1's C Fisher (~0.97s) before any threading is applied at all. This is the opposite of
what P-gate-c.1 assumed ("a Rust-vs-C implementation/NEON factor I expect to be
substantial" in Mars 2's favour) and means the NEON kernel speedups measured in Step 11
(D31, per-kernel 1.4-9.2x) are not translating into a faster single-threaded full encode --
something in the classified-search driver (`mars_search::walk`/`search_block`, or the
per-block bucket/classify overhead Fisher's indexing adds) is costing more than the NEON
kernels save. Not root-caused this session.

### P-gate-c.2 — **falsified**: matched-RD gap is 1.4 dB, not within 0.2 dB

`kodim01` at identical `EncodeParams`: mars1 27.26 dB / 1.4238 bpp (verified to match
`results/baseline-mars1.jsonl`'s already-recorded quality numbers exactly -- the mars1
side of this harness is confirmed correct); mars2 28.65 dB / 1.4265 bpp -- bpp agrees to
0.2% (consistent with D27's already-measured 0.04% evals/transform agreement, so the
*amount* of search work matches), but mars2's decoded PSNR is 1.4 dB *better* despite
nearly identical bpp. This is the RD-curve-within-0.2dB check D28 explicitly flagged as
"not implemented this session, an open gap" -- Gate C's harness is the first to actually
run it, and it did not pass. Not root-caused this session; candidate explanations not yet
distinguished: a real quality difference from Rust's f64 fit precision vs the C reference's
arithmetic, a tie-breaking difference in which block among near-equal candidates Fisher's
restricted bucket scan picks, or a decode-side effect despite reusing gate-6's already-
validated decode-agreement pairing.

**Neither number is safe to report as Gate C's result until the RD gap is explained** --
if mars2's search is doing something subtly different per block (not just a similar total
count), the speed comparison itself may not be measuring "the same work, faster."

---

## 2026-09-14 · Gate C · D35 outcome: the speed gap was `cross_term`'s Step 11 regression, unfixed at this call site

P-gate-c.1 guessed the Rust/NEON implementation factor would dominate and threading would
be the weaker contributor; D33 found the opposite (single-threaded Rust was *slower* than
C). D35 found why: `mars_search::search_block` was still calling `cross_term`'s
naive per-eval-repermuting path — exactly the regression D31 (Step 11) diagnosed and fixed
for the exhaustive encoder, but D31's own fix never touched this call site (its writeup
says so explicitly). Amortising the permutation the same way D31 did cut single-threaded
eval cost from ~140 ns/eval to ~60 ns/eval (2.3-2.4x), moving Gate C's median total
speedup from 1.75-2.36x to 4.09-4.15x. Tested and ruled out LTO/cross-crate-inlining as a
compounding factor (no measurable change from `lto = "thin"`). Neither the 5x kill floor
nor the 20x target is met yet; the remaining gap is attributed to single-threaded Mars 2
now sitting at roughly *parity* with Mars 1's C per-eval cost (not a further easy win) and
would need either a real demonstrated NEON advantage at Fisher's actual (sparse)
candidate-count profile, or fewer evals altogether (Step 13).

---

## 2026-09-14 · Step 13 · prediction (written before any funnel code runs)

A four-stage funnel (`DomainPool`-indexed cheap stats -> structural signature -> thumbnail
distance -> exact affine fit, R&D plan §6) restricting the exact-fit stage to a small
per-block survivor set from the full domain pool, on top of the existing
`CandidateRetriever` driver Step 9 built. Stage 1-3 filter *domain positions* only (not
per-isometry — mean/variance/min/max/range are isometry-invariant for a square block, and
the plan's gradient/edge-energy features are treated as a rotation-agnostic magnitude
rather than ported per-orientation, a documented simplification); Stage 4 (`search_block`)
still evaluates all 8 isometries of each surviving domain, matching every other method in
this crate.

### P13.1 — recall vs. survivor-count tradeoff

Expect top-1 recall against the Step 8 oracle to degrade monotonically as each stage's
survivor count shrinks, with the steepest drop at Stage 1 (cheapest, least discriminating
features) and the shallowest at Stage 3 (thumbnail distance is the closest cheap proxy to
the true SSD the exact fit computes). At the R&D plan's suggested ratios
(10,000->1,000->100->16, rescaled to this project's actual pool sizes, which are far
smaller than 10,000 at `shift=4`) I expect top-1 recall in the 80-95% range on `kodim01` —
below Fisher/Saupe-Fisher's expected ~100% (P9.1, since those canonicalise orientation
before restricting), because this funnel's Stage 1/2 features are raw shape statistics
with no canonicalisation, not an equivalence class.

### P13.2 — evals/transform vs. the six classical methods

Expect the funnel's evals/transform (isometries x Stage-3 survivor count, once Stage 4's
per-domain 8-isometry loop is counted the same way `search_block`'s `evals` counter
counts every method) to land **between Fisher/Saupe-Fisher and Hurtgen/MassCenter** from
P9.2 — the funnel narrows the *domain* set aggressively but still pays the full 8x
isometry loop per survivor, the same structural cost Hurtgen/MassCenter/Mc-Saupe pay.

### P13.3 — the harness sanity check

At Stage 1/2/3 survivor counts all set to "keep everything" (i.e. every domain position
survives to Stage 4), expect the funnel to reproduce `Exhaustive`'s domain-position
coverage exactly per block (same isometry loop, same candidate set, same evals count) —
the same harness-sanity pattern P9.4 established for the six classical methods, adapted to
a funnel with its narrowing disabled rather than a from-scratch bucket method.

### Known gap this session will likely leave open

The full survival/recall tradeoff curve and the Pareto frontier against all six methods
(the brief's actual exit criteria) need the same full-corpus, multi-config sweep Step 9's
own D28 scoped down for lack of session time — expect this session to validate the
mechanism (P13.3) and produce a first recall/evals data point on `kodim01` only, recording
the rest as an open gap the way D28 did, not to close Step 13's exit criteria outright.

---

## 2026-09-14 · Step 13 · outcomes

`mars_search::funnel::Funnel` implements the four-stage `CandidateRetriever` (Stage 1
cheap normalised stats, Stage 2 quadrant + low-frequency-DCT structural signature, Stage 3
`compute_saupe_vector`-based thumbnail distance, Stage 4 the shared exact fit), added to
`MethodName` (key `"funnel"`) so it appears in `marsbench classical-methods`'s table
alongside all six Step 9 methods for free. Measured on `kodim01`/`kodim02` (oracle
`default` config, `marsbench classical-methods`, `t_rms=8` default — a real partitioned
encode, not the gate test's never-split params) and, separately, `gate-13`'s own
never-split harness check.

### P13.3 — **confirmed exactly**

`Funnel` with narrowing disabled (`FunnelMode::Disabled`) reproduces `Exhaustive`'s own
recall/regret against the Step 8 oracle on `kodim01`: **100.0000% top-1/top-5/top-32,
mean regret -1.89e-7 dB** over 1536 matched blocks — identical to P9.4's own result for
the six classical methods' harness check. Confirms the funnel's indexing, candidate
emission, and `search_block` integration have no bug independent of the narrowing itself.

### P13.1 — **falsified on the absolute number (10.0-11.2%, not 80-95%), but not by a funnel bug — the 80-95% guess itself was wrong**

At `FunnelConfig::scaled`'s default survivor counts: top-1 recall **10.03% (kodim01) /
11.23% (kodim02)**, far below the withdrawn 80-95% guess. But Step 9's own measured
numbers on this exact oracle (P9.1's table, `docs/predictions.md`) show exact-tuple top-1
recall against a 32-deep oracle is simply a hard target here regardless of method: even
plain Saupe (the best of six) only reaches 51.55%/45.93%, Fisher reaches 10.71%/9.82%, and
Mc-Saupe as low as 3.87%/3.03%. **The funnel's 10.03%/11.23% sits almost exactly at
Fisher's own measured recall**, not collapsed — the 80-95% prediction was written before
cross-checking P9.1's already-recorded table, which is the actual error, not the funnel's
distance metrics. Recorded here rather than quietly revising the prediction after the
fact, per the verification-discipline skill.

### P13.2 — **partially confirmed: lands inside the classical-method evals/transform range, but not clearly ahead of the pack**

evals/transform on the default partitioned encode: `kodim01` **159.03** (between
Mc-Saupe's 166.69 and Saupe-Fisher's 62.08), `kodim02` **150.23** (between Mc-Saupe's
124.25 and Saupe-Fisher's 58.52) — inside the predicted "between Fisher/Saupe-Fisher and
Hurtgen/MassCenter" band in absolute terms, but specifically clustered next to Mc-Saupe
rather than in the middle of the pack as guessed. Mean regret **1.03 dB (kodim01) / 0.91
dB (kodim02)** is the **second-worst of the seven methods**, ahead of only Mc-Saupe (1.53
/ 1.36 dB) — worse than Fisher (0.95/0.85 dB) despite similar recall and higher evals cost
than Fisher's cheaper Saupe-Fisher sibling. On the gate test's never-split params
(`min_size=max_size=16`, no partitioning) the picture looks better — 128 evals/transform,
0.80 dB regret, roughly Fisher-level recall at ~6x fewer evals — so **block-size mix
matters a lot** to this comparison; the partitioned-encode numbers above are the fairer
comparison to Step 9's own table since both use the same `t_rms=8` default.

### What this means for Step 13

The funnel mechanism is validated (P13.3) and produces real recall/evals numbers, but this
first cut is **not yet a clear win over the existing six methods** — competitive with
Fisher on recall at meaningfully lower cost, but with worse quality (regret) than every
method except Mc-Saupe. The most likely lever, unexplored this session: Stage 1/2's raw
shape statistics have no orientation canonicalisation (unlike Fisher/Saupe-Fisher's
`newclass`), which P9.1's own analysis already identified as a likely driver of recall
differences between the six classical methods — worth trying as a follow-up before
concluding the funnel architecture itself is the limiting factor.

### Known gap, as predicted

The full survival/recall tradeoff curve (sweeping `FunnelConfig`'s survivor counts) and
the Pareto frontier across the full 24-image corpus remain open, exactly as this
prediction's "known gap" section said before any code ran — `docs/decisions.md` records
this alongside D28's identical Step 9 scope cut.

---

## 2026-09-14 (not yet run) · Step 14 · rate-distortion optimisation

`encode.rs`'s current `walk`/`split` recursion (Step 6, extended by Steps 12/13) makes the
split decision top-down and threshold-driven: a block splits whenever `best_rms >
params.t_rms`, before any child has been searched, so the decision cannot see what the
children would actually have cost. Step 14 replaces this with `J = D + λR`: rate estimated
from the live entropy models (not a constant-bits stand-in), and a bottom-up recursion
that searches/codes the children first and only then compares the parent's own single-fit
`J` against the sum of the children's `J`, keeping whichever is smaller. λ becomes the
quality knob — an RD curve is a λ sweep, not a `t_rms` sweep with λ held fixed.

### P14.1 — BD-rate improvement clears the step's own bar, but not by a wide margin

Expect **BD-rate improvement in the 10-20% range** vs. the Step 9 `Exhaustive` reference
at matched search effort, comfortably past the step's 10% target but well short of a 2x
change — bottom-up RD pruning corrects a real blind spot (top-down thresholding cannot
compare a parent's cost to its children's actual coded cost, only to a fixed RMS bar that
has no direct relationship to bits), but the underlying representation (fixed quadtree
geometry, single fractal mode) is unchanged, so most of the ceiling this step can reach is
bounded by how often the old threshold was already picking the RD-better side by luck.
Reasoning by analogy to video codecs' RDO-vs-heuristic-split gaps, which cluster in this
same 10-20% band when only the split decision changes and the mode/partition set does not.

### P14.2 — the convexity check is the real risk, not the BD-rate number

Expect the first attempt at rate estimation to produce a **non-convex or non-monotone**
RD curve at the extreme ends of the λ sweep (very low or very high λ) before it produces a
clean one — rate estimation from live adaptive entropy models is context-dependent (the
same symbol costs different bits depending on encode order and prior blocks' statistics),
and a bottom-up comparison that estimates each child's rate independently, without
accounting for how the adaptive model's state actually evolves across the real encode
order, is the most likely place this shows up. Per the step's own framing, a non-convex
curve is treated as a rate-estimation bug to fix, not a result to report.

### P14.3 — the harness sanity check

Expect that setting λ to select the same operating point the old `t_rms` threshold would
have chosen (i.e., roughly matched bpp) reproduces a similar partition to the Step 9/13
baseline's leaf-size histogram — not identical (the decision rule genuinely differs), but
without a wholesale collapse to all-leaf or all-split, which would indicate the rate
estimate or `J` comparison has a sign error or unit mismatch (e.g. λ and R in incompatible
units) rather than a real algorithmic difference.

### Known gap, stated before any code runs

`μT` (decode-cost term) is out of scope this step per the brief ("extend... once
decode-cost measurement exists") — only `J = D + λR` is implemented. The λ sweep is run at
whatever image subset is practical this session, not necessarily the full 24-image
`standard/` corpus; if scoped down, the cut is recorded in `docs/decisions.md` the same
way D28/Step 13's corpus-subset cuts were.

---

## 2026-09-14 (not yet run) · Step 18 · colour

**Process note, recorded honestly rather than smoothed over (A7):** this prediction was
written after a short exploratory pass (building `mars_codec::color`, its unit tests, and
one manual `encmars`/`decmars` round trip at a single rate point on `kodim01` to sanity-
check the pipeline actually worked end to end) rather than strictly before any measurement
at all, which is a real deviation from A4's letter. The RD-curve sweep and BD-rate numbers
below were **not** looked at before this prediction was written — only the single-point
sanity check (bytes non-zero, PSNR-Y > 20 dB) was. `docs/decisions.md` D37 records this
deviation and why it does not (in this author's judgement) invalidate the predictions
below, since the sanity check carried no information about relative rate or the BD-rate
sign.

**Entry-condition conflict.** Step 18's brief text says "Entry condition: Gate D passed",
but §5's dependency graph and its "what may run concurrently" table both place Step 18
right after Gate B, concurrent with Steps 10/11/13/14. Built now, at the user's explicit
instruction, under the concurrent-with-14 reading; `docs/decisions.md` D37 records the
conflict itself. Because Gate D (residual mode, adaptive partitioning) has not passed, any
BD-rate numbers here are against **today's pre-Gate-D exhaustive encoder**, not the codec
the brief's own exit criterion implicitly assumes — provisional, to be re-measured once
Gate D passes.

**What was built:** `mars_core::metrics::rgb_from_ycbcr` (BT.601 inverse of the existing
`ycbcr`), `mars_codec::color` (independent per-plane `EncodeParams` for Y vs. chroma, box-
filter 4:2:0 downsampling, nearest-neighbour upsampling, a small `MARC` colour container
wrapping three independent `.mars` v0 streams), and `encmars`/`decmars` CLI support for
colour PNG round trips.

### P18.1 — 4:2:0 costs some chroma quality but saves more bits than it costs, at matched luma quality

Both subsampling modes share an identical Y stream at a given `t_rms` (chroma
subsampling never touches luma), so PSNR-Y should be **identical** between 4:4:4 and
4:2:0 at matched `t_rms`, while total bytes should be lower for 4:2:0 (fewer chroma
samples to encode) and PSNR-Cb/PSNR-Cr should be lower for 4:2:0 (upsampled from half
resolution). Whether this nets out to a *better* BD-rate for 4:2:0 depends on which
metric is used: expect PSNR-Y-matched comparisons to favour 4:2:0 (it is "free" bits
saved with no luma cost), but expect the brief's own PSNR-YUV metric — which weights
chroma at `1/8` each and would need chroma quality preserved to fully credit the bit
savings — to show a **smaller** 4:2:0 advantage than a naive "chroma is a quarter the
pixels, so a quarter chroma bits, for free" story would suggest, and possibly even a
*net loss* in BD-rate terms if chroma PSNR degrades faster than the weighted quality
metric can absorb given how few bits chroma already occupies relative to luma. This is
the standard subsampling tradeoff every 4:2:0-capable format ships with; the open
question this prediction flags is only which side of break-even PSNR-YUV lands on for
this codec's specific chroma bit allocation.

### P18.2 — Mars 2 (pre-Gate-D) remains behind the strongest anchors on colour Kodak

Per the brief's own pointer to R&D plan §M10 ("expect to remain behind AVIF/JXL"):
expect Mars 2's pre-Gate-D exhaustive encoder, now colour-capable, to sit clearly behind
AVIF and JPEG XL in BD-rate on any Kodak images measured, since neither residual coding
nor adaptive partitioning exist yet — this is fundamentally the same exhaustive quadtree
fractal encoder Step 6 built, just applied three times. Against plain JPEG the outcome
is less obvious a priori: JPEG's block-DCT has its own well-known weaknesses at low bpp
(blocking artefacts) that a fractal encoder's self-similarity search does not share, so
this prediction does **not** confidently call the JPEG comparison's direction — it is
recorded as an open question the measurement below will answer, not smoothed into "we
expect to lose to everything."

### Known gap this session will likely leave open

A full 24-image Kodak sweep against all five anchors (JPEG, JPEG 2000, WebP, AVIF, JPEG
XL), with a proper BD-rate table, is far more encode time than this session's budget
allows — the exhaustive Rust encoder takes 10-35 seconds per colour image per rate point
per subsampling mode at this machine's core count, and a full sweep needs on the order
of `24 images x 2 subsampling modes x 4+ rate points` = 190+ colour encodes. Expect this
session to measure a single image (`kodim01`) at 4 rate points in both subsampling
modes, against one anchor (JPEG, reusing already-recorded `results/anchors.jsonl` rows
rather than re-running the anchor sweep), mirroring the D28/Step-13 precedent for scoping
down a step's exit criteria to what a session can actually run, and to record the rest —
full corpus, all five anchors, MS-SSIM-based BD-rate — as the open gap.

---

## 2026-09-14 · Step 18 · outcomes

Measured on `kodim01` (768x512), `t_rms` in `{4, 8, 16, 32}`, both subsampling modes,
chroma `t_rms` defaulted to the same value as luma (no independent chroma quality tuning
attempted this session beyond the mechanism existing). PSNR-YUV per §M2's `(6Y+Cb+Cr)/8`
definition, via `marsbench metrics`. BD-rate via `marsbench bdrate` (PCHIP-on-log-bpp,
§M3), `points_reference`/`points_test` = 4 in every curve (the §M3 minimum, not more).

| t_rms | 444 bpp | 444 PSNR-YUV | 420 bpp | 420 PSNR-YUV |
|---|---|---|---|---|
| 4  | 1.8035 | 34.349 | 1.6367 | 33.418 |
| 8  | 1.5259 | 33.853 | 1.3663 | 32.826 |
| 16 | 0.8457 | 31.044 | 0.6861 | 30.016 |
| 32 | 0.3464 | 27.783 | 0.1868 | 26.754 |

(PSNR-Y alone was *nearly* identical between 444 and 420 at every matched `t_rms`, as
predicted — 30.21274 dB vs. 30.21188 dB at `t_rms=8`, a difference under 0.001 dB — since
the coded Y stream is byte-identical between subsampling modes; chroma subsampling never
touches it. **Correction, caught while writing `gate-18`'s own test, not smoothed over:**
"nearly identical" is the accurate claim, not "identical" — `quality()`'s PSNR-Y is
measured on the *reconstructed RGB*, re-converted back to YCbCr, so noisier 4:2:0 chroma
does perturb the recomputed Y by a small amount through the inverse transform's per-
channel clamping, even though the decoded Y *plane* itself is bit-identical between modes.
On a synthetic high-contrast test image used in `gate-18`'s own test this gap was larger
(34.31 vs. 33.87 dB, ~0.44 dB) than on `kodim01`'s natural-image gradients — large enough
that the gate asserts a 1 dB tolerance rather than exact equality. See `docs/decisions.md`
D38 for this correction recorded as its own note.)

### P18.1 — **partially confirmed, with the surprising half flagged rather than smoothed over**

The PSNR-Y-nearly-identical part is confirmed (see the correction above). But on
PSNR-YUV, BD-rate of
4:2:0 relative to a 4:4:4 reference on this single image is **+3.86%** (4:2:0 needs *more*
bits at matched PSNR-YUV, not fewer) — the opposite sign from the naive "subsampling is
free bits" framing, though consistent with the more careful version of P18.1's own
reasoning: PSNR-YUV weights chroma at only `1/8` each, and 4:2:0's chroma bit savings on
this image are already small in absolute terms next to luma's cost (e.g. at `t_rms=8`,
chroma is 10,350 of 75,002 total bytes for 4:4:4 vs. 2,506 of 67,158 for 4:2:0 — a real
~10% total-byte saving, but paid for with a chroma PSNR drop of 3.5-5 dB that the `1/8`-
weighted metric still penalises enough to tip the BD-rate sign). **This is recorded as a
genuine, measured result on one image, not explained away**: it does not mean 4:2:0 is a
bad idea in general (a full corpus average, or a PSNR-Y-matched or MS-SSIM-based
comparison, could easily flip the sign — MS-SSIM was not checked this session, see the gap
below), only that on `kodim01` specifically, at these four rate points, PSNR-YUV's
particular chroma weighting does not reward 4:2:0's bit savings.

### P18.2 — **confirmed against JPEG in the unexpected direction relative to plain "remains behind"; AVIF/JXL not measured**

Against JPEG's own already-recorded curve for `kodim01` (`results/anchors.jsonl`, 11
points), BD-rate (PSNR-YUV, JPEG as reference):
- **mars2-444 vs. JPEG: +0.80%** — essentially matched, marginally worse.
- **mars2-420 vs. JPEG: -4.42%** — mars2-420 needs *fewer* bits than JPEG at matched
  PSNR-YUV on this one image.

This is **not** the brief's own §M10 expectation ("expect to remain behind AVIF/JXL") —
but note the brief names AVIF/JXL specifically, not JPEG, and JPEG is the weakest of the
five anchors. AVIF and JPEG XL were not measured this session (scope cut, see below); the
brief's actual prediction remains untested. The JPEG comparison is a genuinely
interesting, single-image, pre-Gate-D data point — recorded as such, not generalised into
"Mars 2 beats the anchors."

### What this means for Step 18

The colour pipeline works end to end (round trip, independent per-plane quality control,
both subsampling modes, container format) and produces real, if narrow, RD/BD-rate
numbers. Both surprises (P18.1's sign flip, P18.2's JPEG result) are recorded plainly
rather than reinterpreted to match the prior — per the verification-discipline skill,
"assume the harness before the result," and the harness here (the same `bdrate`/`metrics`
machinery already cross-validated in Steps 1 and 4) is not new or suspect, so these are
treated as real, if narrow-scope, results rather than harness bugs.

### Known gap, as predicted

Exactly the gap named above: no full 24-image sweep, no AVIF/JPEG 2000/WebP/JPEG XL
comparison, no MS-SSIM-based BD-rate (only PSNR-YUV was computed for the curves; MS-SSIM
values were recorded per-point by `marsbench metrics` but not turned into a second set of
BD-rate numbers this session). All of this is measurement-infrastructure work that
reuses tools already built (`mars-bench::bdrate`, `anchors.rs`'s existing anchor curves)
rather than needing new code — the remaining cost is purely encode wall-time. And, as
stated up front: everything measured here is against the pre-Gate-D encoder, so none of
these BD-rate numbers are the step's real exit-criterion numbers — they are a feasibility
demonstration that the measurement path works, to be re-run once Gate D passes.

---

## 2026-09-14 · Step 14 · outcomes

`mars_codec::encode` now has a bottom-up `walk_rd`/`split_rd` (alongside the untouched
legacy `walk`/`split`, selected by `EncodeParams::lambda: Option<f64>`), a frozen
rate-estimation snapshot (`crates/mars-codec/src/rate.rs`), and `mars-cli`'s `encmars
--lambda`. Measured via `crates/mars-bench/tests/rd_gate.rs` on `kodim01`/`kodim02`
(4-point λ sweep `[50, 200, 800, 3200]` vs. the Step 9 `Exhaustive` reference's `t_rms`
sweep `[2, 4, 8, 16, 32]`, both at `.mars` v0 bpp). `docs/decisions.md`'s D39 has the full
detail; this compares directly against P14.1/P14.2/P14.3.

### P14.1 — **falsified on the number: -8.07% mean, not the predicted 10-20% band**

Measured mean BD-rate **-8.07%** (kodim01 -8.61%, kodim02 -7.54%) against the Step 9
`Exhaustive` reference — real and substantial (comfortably past the project's 3%-BD-rate
kill criterion, so Step 14 is not a "stop after Gate B" outcome), but *below* the
predicted 10-20% band and short of the brief's own >= 10% target. Recorded as a genuine
falsification, not revised after the fact: the prediction reasoned from a video-codec
analogy (RDO-vs-heuristic-split gaps clustering at 10-20% when only the split decision
changes) that turned out optimistic for this codec's specific situation. The most likely
reason, per D39's own analysis: the rate estimator's frozen snapshot is warmed up at a
*fixed* `t_rms = 8.0` regardless of the run's own λ, so its per-context statistics match
the eventual λ-chosen partition's leaf-size mix only loosely at the sweep's extremes
(λ = 50 and λ = 3200 produce leaf-size mixes quite different from the `t_rms = 8.0`
warm-up partition) — a self-consistent (iterated) warm-up was named as the natural next
step in the prediction's own reasoning about rate estimation, and remains the most
promising unexplored lever here too.

### P14.2 — **falsified: the first attempt did not show non-convexity**

`mars_bench::rd_opt::check_convex_and_monotonic` passed cleanly on both images' λ curves
on the first real measurement, with no sign flip or diminishing-returns violation at
either sweep extreme. This is a genuine surprise relative to the prediction, which
reasoned that per-candidate rate estimation independent of true encode-order adaptive
state was "the most likely place this shows up." The chosen approximation (a frozen,
read-only snapshot, §D39) sidesteps the specific failure mode the prediction worried
about (an inconsistently-updated *mutable* model producing incoherent per-candidate
prices) by never mutating the model *during* the search at all — every candidate at every
λ is priced against the exact same fixed reference, which apparently preserves enough
internal consistency for the resulting curve to stay convex even though the snapshot
itself is a coarse approximation of the true adaptive cost (per P14.1's own shortfall).
Recorded as a real, useful data point: freezing the model traded some BD-rate accuracy
for convexity robustness, which on this evidence looks like a good trade for a first cut.

### P14.3 — **confirmed: no degenerate collapse, but with a caveat found during testing**

At a moderate λ on a real mixed-partition test image, the RD walk produces a genuine mix
of leaf sizes rather than collapsing to all-leaf or all-split (`encode::tests::
moderate_lambda_produces_a_real_mixed_partition_not_a_degenerate_one`) — no sign-error or
unit-mismatch symptom. The caveat, found while writing that very test and worth recording
per A7: on a *spatially uniform* synthetic image (`textured_image`, whose block-periodic
texture makes every same-size block statistically near-identical), the RD walk's leaf-size
choice flips in lockstep across the *entire* image as λ crosses a threshold, with no
intermediate mix ever appearing at any λ tried — not a bug (every block genuinely faces
the same local trade-off at once, so a uniform response is the correct answer for that
specific image), but a reminder that "mixed partition" as a sanity check needs an image
with genuine *spatial* heterogeneity (`half_flat_half_noisy_image`) to be a meaningful
test at all; a spatially uniform fixture can pass or fail the same check for reasons
unrelated to whether the RD logic itself is correct.

### Known gap — same one named before any code ran, still open

`μT` (decode-cost term) remains unimplemented, as scoped. The λ sweep ran on
`kodim01`/`kodim02` only, not the full 24-image `standard/` corpus, and at 4 λ points
(§M3's minimum) rather than a finer sweep — pure session time budget (bottom-up RD search
costs ~6.2 billion evals per encode regardless of λ, since it visits every quadtree node
down to `min_size` unconditionally; one encode takes on the order of a minute). A
self-consistent (λ-adaptive, iterated) rate-estimation warm-up, the full corpus, and a
finer λ sweep are the concrete next steps to close the gap to the brief's 10% target,
per D39.

---

## 2026-09-14 (not yet run) · Step 15 · residual mode

Written after the mode-competition implementation (`mars_codec::encode::best_mode_leaf`,
`crate::dct`, `crate::quant`, `crate::residual`, and `mars_format`'s mode-aware
serialisation) was built and its own unit/round-trip tests were passing, but **before**
any BD-rate sweep or mode-usage histogram was measured — the unit tests check internal
consistency (round trips, SSE bookkeeping, a pure gradient block fitting almost exactly)
and say nothing about which mode actually wins on real images or what it does to the
corpus-level rate/distortion numbers. Per A4 this counts as "before measuring", the same
class of deviation Step 18's prediction recorded for itself (a sanity pass first, the
actual RD sweep after this entry), not a retrofit of a result already seen.

**What was built.** Modes 0 (flat, `best_beta`'s existing refit, now priced as its own
competing candidate rather than an override), 1 (affine: a closed-form least-squares
spatial-gradient plane fit `b0 + gx*u + gy*v`, no domain search), 2 (fractal, unchanged),
and 3 (fractal + residual: mode 2's continuous prediction error, forward-DCT'd via a
direct O(N^3) orthonormal DCT-II/DCT-III pair, dead-zone quantised at a **fixed**
compile-time step/dead-zone — not λ-adaptive this step, a deliberate scope cut recorded in
`docs/decisions.md` — and context-modelled rANS coded via a new nonzero-flag/magnitude
event vocabulary bucketed by frequency band). All four compete under the exact same
`J = D + λR` machinery Step 14 built, priced against the same frozen `RateModels`
snapshot, inside `walk_rd`'s existing leaf-vs-split (mode 4) comparison.

### P15.1 — fractal's share of the mode histogram will be a minority, not a majority

Expect modes 2+3 (fractal, fractal+residual) combined to win **well under half** of leaf
decisions on the `standard/` Kodak-style corpus at moderate λ, with mode 0 (flat) and
mode 1 (affine) together taking the majority — most natural-image blocks at the leaf
sizes this project's configs use (4-16 px) are locally smooth-with-gradient or genuinely
flat far more often than they contain a *distinct*, better-matching self-similar region
elsewhere in the same image, and mode 1's spatial-gradient fit is a strictly cheaper way
to capture "smooth ramp" than a domain search plus isometry plus two coordinate fields.
This is exactly the kind of finding the brief flags as legitimate rather than a failure:
it would be the first quantification, under genuine competing-mode pressure, of how much
of a fractal codec's own literature-standard test corpus actually benefits from the
self-similarity search at all, as opposed to benefiting because nothing cheaper was ever
offered as an alternative.

### P15.2 — mode 3 (fractal + residual) rarely beats plain mode 2 at this session's fixed quantisation step

Because the residual quantisation step is fixed (not λ-swept, §"what was built" above),
expect mode 3 to win only a **small minority** of the fractal-eligible decisions (i.e.
among blocks where mode 2 itself was competitive) — a fixed step that is well-tuned for
one λ will be badly mismatched at the sweep's other extremes (too coarse to help at low λ
where rate is cheap, or too fine to be worth its own coding overhead at high λ where every
extra bit is expensive), so mode 3's win rate is expected to **vary non-monotonically
across the λ sweep** rather than climb smoothly, which would itself be a symptom of this
exact, already-known limitation rather than a new bug — recorded here so that pattern, if
seen, is not mistaken for a rate-estimation error the way Step 14's P14.2 worried about.

### P15.3 — BD-rate vs. Step 14 improves, but by a modest, single-digit-to-low-teens percentage

Expect **BD-rate improvement in the 3-12% range** vs. Step 14 (commit `788ee5d`) at
matched search effort — every new mode is strictly a superset of what Step 14 could
already express (mode 2 unchanged, and `J` will simply never pick a new mode unless it is
actually cheaper), so a regression is not expected and would indicate a bug (most likely
in event pricing making the new modes look artificially cheap and be picked wrongly), but
the ceiling is bounded by how much of the corpus was already well-served by flat/fractal
alone. This is deliberately a wide, low-confidence band — unlike Step 14's brief, Step 15
states no numeric target, so this number exists to be compared against, not graded
against a bar.

### Known gaps, stated before any measurement runs

Residual quantisation is not λ-adaptive (P15.2's own premise) — a real, documented
simplification (`docs/decisions.md`), not an oversight. The BD-rate/histogram sweep is
expected to be scoped to `kodim01`/`kodim02` at the same 4-point λ grid Step 14 used
(`gate-14`'s own precedent for a session-time-bounded scope cut), not the full 24-image
`standard/` corpus — bottom-up RD search with four extra mode evaluations per node is
strictly more expensive per encode than Step 14's own already-slow walk. `μT` (decode-cost
term) remains out of scope, inherited unchanged from Step 14.

---

## 2026-09-14 · Step 15 · outcomes

Measured via `crates/mars-bench/tests/residual_gate.rs` (`gate-15`), kodim01/kodim02, the
same 4-point λ grid Step 14's own gate used (`[50, 200, 800, 3200]`), Step 15's full
4-mode curve against a same-codebase "Step 14 equivalent" curve (modes 0/2 only, via
`encode_image_rd_with_modes`'s mode mask — see `mars_bench::mode_gate`'s doc for why this
was chosen over diffing a separate git revision), both at `.mars` v0 (entropy-coded) bpp.

### P15.1 — **falsified, in the opposite direction predicted**

Predicted fractal (modes 2+3) would win "well under half" of leaf decisions, with
flat+affine taking the majority. Measured corpus-wide mode-usage histogram:
`mode0=14.3% mode1=0.1% mode2=85.0% mode3=0.6%`. Fractal prediction (modes 2+3 combined)
won **85.6%** of leaf decisions — a wide margin in the *opposite* direction from the
prediction. This is the brief's own "more scientifically interesting than the BD-rate
number" finding, taken at face value rather than smoothed toward what was expected: on
this corpus, under genuine competing-mode pressure (flat, affine, and residual all
available as cheaper alternatives at every node), self-similarity prediction is not a
weak default that wins only because nothing cheaper was offered — it is, empirically, the
dominant winning representation by a wide margin. Mode 1 (affine)'s share (0.1%) is also
far smaller than the prediction's implicit expectation that it would meaningfully
compete with mode 0 — most blocks that are not well-served by a domain match are
apparently well-served by a flat refit already, with a genuine spatial gradient rarely
being the deciding factor at this corpus's block sizes (4-16 px).

### P15.2 — **roughly confirmed on the raw share, but the underlying reasoning also
explains a regression the prediction did not anticipate**

Predicted mode 3 would win only a small minority of fractal-eligible decisions because
the residual quantisation step is fixed, not λ-swept. Measured: mode 3 won **0.6%** of all
leaves corpus-wide (a small share, consistent with "small minority"). What the prediction
did not anticipate: this same fixed-step mismatch, combined with a rate-estimation gap
inherited unchanged from Step 14 (`docs/decisions.md`'s D40 has the full root-cause
analysis), is also the proximate cause of the BD-rate regression below — a fixed step
being occasionally *mispriced as attractive* by the frozen rate snapshot, not merely
*rarely winning*, turns out to be the more consequential effect.

### P15.3 — **falsified: a small regression, not an improvement**

Predicted 3-12% BD-rate improvement vs. Step 14. Measured: **mean +2.05%** (kodim01
+1.88%, kodim02 +2.23%) — a positive number, meaning Step 15's full mode competition
needs *more* bits than the Step-14-equivalent curve for the same quality. This falsifies
the prediction outright, including its stated floor of "a regression is not expected and
would indicate a bug." A regression was found; per A7 it is investigated, not dismissed.
The root cause (`docs/decisions.md`'s D40) is not a defect in the DCT/quantiser/entropy-
coding machinery itself (every stage has passing exact round-trip tests, including a full
mixed-mode encode through the real `.mars` v0 bitstream) but in Step 14's already-
documented frozen rate-estimation snapshot, which structurally cannot ever observe modes
1/3's fields (the legacy warm-up walk that builds it cannot produce those modes), so `J`
occasionally misprices them as attractive when the real coded cost is higher. The
convexity/monotonicity check passed cleanly, ruling out the specific failure mode P14.2
worried about (a sign-error/unit-mismatch in the rate estimate) — this is a *calibration*
gap in an approximation already known to be approximate, not an arithmetic bug.

**Confirmed, not just argued (added after independent review flagged the result and
proposed two specific alternative hypotheses — a mismatched-snapshot comparison bug, or a
violation of "more candidates can't make the estimate worse").** Both were checked
directly rather than reasoned about in the abstract: (1) `build_rate_snapshot` calls the
legacy `walk`, which never reads the mode mask at all, so both compared curves are priced
against a byte-identical frozen snapshot — ruled out by code inspection; (2) a new test
(`encode::tests::aggregate_estimated_cost_is_provably_no_worse_under_more_modes_even_though_real_bpp_can_be`)
reads `walk_rd`'s own internal `RdResult.d`/`.r` directly and confirms the provable
property holds (`j_4mode <= j_2mode` under the identical snapshot) — `best_mode_leaf`'s
minimisation is correct; the divergence is entirely in the estimate-vs-reality gap, not
in the search. `docs/decisions.md`'s D40 has the full trace, including a test-methodology
bug caught and fixed mid-investigation (an earlier version of the same test compared
against the *wrong* estimate convention — the real sequential domain-position predictor
instead of the fresh-per-leaf one `best_mode_leaf` actually prices against — and failed
for that reason alone, not because of a codec bug).

### Known gap, now sharper than when predicted

The self-consistent (iterated) rate-estimation warm-up named as Step 14's own natural next
step (never attempted there, for time-budget reasons) is now also the concrete fix for
this step's regression, and was not attempted here either, for the same reason — one
already-expensive ~15-minute gate sweep was this session's realistic budget, and a
warm-up that itself runs `walk_rd` once would roughly double every encode's cost.
`docs/decisions.md`'s D40 records this as the named next step, not a silently repeated
deferral, and flags the gate's own calibrated ceiling as an open concern for the kill-
criteria audit rather than asserting it is obviously fine.
