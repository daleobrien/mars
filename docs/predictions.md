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
