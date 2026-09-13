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
