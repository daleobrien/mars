# Decisions

**Append-only (§A7).** Anomalies halt; they are never averaged away. Widening a tolerance
requires a recorded reason here. So does resolving a cross-validation disagreement by any
means other than fixing the code.

Each entry: what was decided, what it rules out, and what would reverse it.

---

## D1 · 2026-09-13 · ffmpeg is the PSNR reference; scikit-image is the SSIM reference

**Context.** Step 1's exit criterion suggests `ffmpeg -lavfi ssim/psnr` as the independent
implementation to cross-validate against.

**Finding.** ffmpeg's `ssim` filter computes SSIM over **8×8 uniform** windows (the
x264-derived variant), not the 11×11 Gaussian §M2 pins. The disagreement is structural and
is not small; it cannot be brought inside 0.001 by any correct implementation of either
definition.

**Decision.** Split the reference by metric:

- **PSNR** → `ffmpeg -lavfi psnr`. No definitional ambiguity; agrees to ~4e-7 dB (the
  limit of ffmpeg's printed precision).
- **SSIM** → `skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5,
  use_sample_covariance=False, data_range=255)`. This *is* Wang's original definition:
  with `truncate=3.5` the Gaussian is 11×11, and skimage crops the 5-pixel border, making
  it a `valid` convolution with population covariance. Agrees to ~3e-14.
- **MS-SSIM** → `sewar.full_ref.msssim`, phase-matched (see D2).

**What this rules out.** Using ffmpeg's SSIM number anywhere, including as a sanity check
— a table containing both would invite a comparison between two different metrics.

**What would reverse it.** ffmpeg gaining a Gaussian-window SSIM, or the project re-pinning
§M2's SSIM to the 8×8 uniform variant. The latter is not advisable: the Gaussian variant is
what the literature the project will be compared against reports.

**Tolerance impact.** None. No tolerance was widened; the reference was corrected.

---

## D2 · 2026-09-13 · MS-SSIM decimation is pinned to a non-overlapping 2×2 box average

**Context.** §M2 pins MS-SSIM's scale count and weights but not the inter-scale decimation,
and published implementations disagree about it.

**Finding.** Wang's reference `msssim.m` applies `imfilter(ones(2)/4, 'symmetric', 'same')`
then takes `1:2:end`, which is a **non-overlapping** 2×2 box average — identical to the
`avg_pool2d(2)` used by the PyTorch and TensorFlow implementations. `sewar` instead uses
`scipy.ndimage.uniform_filter(im, 2)` followed by `[::2, ::2]`, a backward half-pixel-shifted
window with reflected edges. The two differ by ~1.2e-3 on real images: over the 0.001
cross-validation tolerance, and far too small to look like a bug.

**Decision.** Pin `Downsample::Box2x2` (Wang / avg_pool2d) for every reported number.
Implement `Downsample::ScipyUniform2` **solely** so the MS-SSIM scale chain and weighting
can be cross-validated against `sewar` with the phase difference removed — at which point
agreement is 1.1e-16.

**What this rules out.** Quoting a `ScipyUniform2` MS-SSIM anywhere. One test
(`the_two_downsample_phases_really_do_differ`) asserts both halves of the relationship: the
two phases must stay distinct, so the cross-validation cannot silently become vacuous, and
their difference must stay under 0.05, so a genuine bug in either cannot hide behind the
phrase "phase difference". The cross-validation itself re-asserts the first half on every
pair it checks.

**Tolerance impact.** None; `MSSSIM_CROSSVAL_ABS` stays at 0.001 and is met with 13 orders
of magnitude to spare.

---

## D3 · 2026-09-13 · Mars 1 must be driven from a short working directory

**Finding, the hard way.** `reference/mars1/globals.h` declares `char filein[50]` and
`char fileout[50]`, and `getopt_enc` copies `argv` into them with `strcpy`. An absolute
path longer than 49 bytes overflows the buffer. Observed symptom: `encmars` reads the
input correctly, reports progress normally, and then fails with `Can't open output file` —
a plausible, misleading error that looks like a permissions problem.

**Decision.** The Step 2 subprocess driver must `chdir` into a short scratch directory and
pass **relative** filenames. This is recorded here rather than only in the driver because
the failure mode is misleading enough to cost a debugging session to anyone who meets it
cold, and agents start cold (§A5).

**What this rules out.** Driving the 1998 binaries with absolute paths, including in CI —
`.github/workflows/ci.yml` uses a `mktemp -d /tmp/m1.XXXX` working directory for exactly
this reason.

**What would reverse it.** Patching the reference C, which §3 forbids: `reference/mars1/`
is the unmodified 1998 source, and a patched baseline is not a baseline.

---

## D4 · 2026-09-13 · MS-SSIM's scale count is `weights.len()`, not a separate field

**Context.** A config with both `scales: usize` and `weights: [f64; 5]` can hold two values
that disagree.

**Decision.** `MsSsimConfig.weights` is a `Vec<f64>` and the scale count *is* its length.
This makes a useful identity expressible: a single-element weight vector reduces MS-SSIM to
plain SSIM exactly, which is the test that ties MS-SSIM to the separately cross-validated
SSIM. A two-element `[1.0, 0.0]` vector likewise isolates the claim that luminance enters
only at the final scale.

**What this rules out.** Nothing in the reported configuration; `WANG_WEIGHTS` remains the
default and has five entries.

---

## D5 · 2026-09-13 · Mars 1 has no exhaustive search mode; the unflagged default is MassCenter

**Context.** `implementation-plan.md` §4 Step 2 lists the sweep's method axis as
`-F -X -C -S -Z -Y`, and the `mars1-reference` skill's CLI table annotated the method flag
"(none = exhaustive)". The natural reading is that omitting the flag gives an exhaustive
search, which would make the 1998 codec its own RD upper bound.

**Finding.** It does not. `reference/mars1/globals.h:201` is
`EXTERN int method INIT(= MassCenter)`, and `miscell.c:getopt_enc` only ever *overwrites*
that. Running `encmars` with no method flag prints `Speed-up method: MassCenter` and runs
MassCenter. Confirmed empirically: `encmars -W 512 -H 512 -r 8 lena.raw` reports
MassCenter, 4423 transforms, 3 471 616 comparisons. **There is no exhaustive mode in the
1998 codec at all** — `def.h` defines exactly six methods and every code path goes through
one of them.

**Decision.**

1. Every invocation the Step 2 driver makes passes an explicit method flag, and
   `mars1::encode` **fails** unless the label the encoder echoes equals the method that was
   requested. A dropped or mistyped flag would otherwise not error — it would silently
   measure MassCenter and label the row with something else, which is precisely the
   "silent plausible wrongness" §2.1 names as the characteristic failure here.
2. The `mars1-reference` skill's CLI table is corrected to say "(none = MassCenter, the
   compiled-in default — there is no exhaustive mode)".

**What this rules out.** Treating any Mars 1 run as the exhaustive RD upper bound. The
oracle of §M6 has no Mars 1 equivalent and must be built in Rust at Step 6/7 as the plan
already schedules. It also rules out reading the six methods' RD spread as "distance from
optimal"; without an exhaustive baseline it is only distance from each other, and Step 8's
recall/regret table is the first point at which that question becomes answerable.

**What would reverse it.** Nothing short of a different reference codec.

**Tolerance impact.** None. The new check is an exact string equality.

---

## D6 · 2026-09-13 · The structural grid is one-at-a-time around the 1998 default

**Context.** Step 2's sweep grid lists six parameters. Read as a cross product it is
6 methods × 5 rates × 3 (min,max) × 2 steps × 2 alfa × 2 beta = 720 configurations per
image, 17 280 encodes over Kodak, on the order of a day of compute.

**Finding.** The table does not read as a cross product. Four of its six rows are written
as "*default*; also X" — alternatives to a default, not axes to multiply. The question
those rows ask is what each structural parameter does to the RD curve, which a
one-at-a-time sweep answers and a full cross product answers no better.

**Decision.** The grid is a **base point plus five one-parameter deviations**:
`default` (the 1998 settings, carrying the headline curves), `min2` (`-m 2`), `max32`
(`-M 32`), `step8` (`-d 8`), `alfa5` (`-A 5`), `beta6` (`-B 6`). Each is crossed with all
six methods and all five rates: 6 variants × 6 methods × 5 rates × 24 images = 4320
encodes on Kodak, plus 360 on `fixtures/` at the default point only. Every encode is
decoded in **both** modes, giving 9360 rows.

**What this rules out.** Any claim about *interactions* between structural parameters —
for example whether `-d 8`'s search saving is larger when `-m 2` deepens the partition.
Nothing in Group A or B depends on such a claim; if one later does, the config file is the
only thing that has to change (§2.3).

**What would reverse it.** A step that needs an interaction effect. `configs/baseline-mars1.json`
expresses variants as independent overrides, so a crossed variant is a config edit, not a
code change.

**Tolerance impact.** None.

---

## D7 · 2026-09-13 · BD-rate is computed per image and then summarised; averaged curves are for plotting

**Context.** A 24-image corpus can be reduced to one RD curve per method by averaging bpp
and PSNR at each rate, and BD-rate taken between those averaged curves. It is the cheaper
and more commonly published option.

**Finding.** BD-rate is a nonlinear functional of the curve pair, so the BD-rate of two
averaged curves is not the average of the per-image BD-rates. On a corpus with a wide
difficulty spread — Kodak's kodim03 and kodim13 differ by several dB at matched rate — the
two can disagree by several percent, and the averaged number is dominated by whichever
images sit at the ends of the bpp range.

**Decision.** Every BD-rate quoted in a Step 2 table is computed **per image** and then
summarised as mean, median, min and max across images, with `n` stated. The
corpus-averaged curve is computed too, plotted, and reported in its own column labelled
`averaged-curve %` so the two can be compared — but it is never the quoted number.

An image whose curve is unusable (non-monotonic, or no PSNR overlap with the reference
method) is **named in the exclusions column with its reason**, never dropped. A silently
excluded hard image makes a method look better than it is, and `just gate-2` fails if any
exclusion occurs at all, so an exclusion is something a human reads rather than something
the harness absorbs (§A7, §A8).

**What this rules out.** Quoting the averaged-curve BD-rate as the headline, and any table
whose `n` is smaller than the corpus without saying which images are missing and why.

**Tolerance impact.** None.

---

## D8 · 2026-09-13 · Wall-clock seconds in the baseline rows are indicative, not §M4 results

**Context.** The sweep runs 4680 encodes across six worker threads and records how long
each subprocess took. Those seconds are useful — they are how the sweep was sized — and
they are also exactly the shape of number §M4 governs.

**Decision.** They are recorded, and every row carries a `timing_protocol` field reading
*"indicative-single-run; NOT a §M4 timing result (no interleaving, no median, concurrent
jobs)"*. The sweep is a single run per configuration, is not A/B interleaved, reports no
median or MAD, and runs six jobs concurrently, so it fails four of §M4's requirements at
once.

**What this rules out.** Quoting any speedup from `results/baseline-mars1.jsonl`. Mars 2
timing comparisons come from the Step 11/12 protocol on a quiet machine; the headline
efficiency number for Step 2 is `evals/transform`, which is machine-independent,
deterministic, and unaffected by how many jobs were in flight (§M5).

**Tolerance impact.** None.

---

## D9 · 2026-09-13 · Decode iterations are recorded per row; the decode *mode* is nearly free

**Context.** §4 Step 2 and the `mars1-reference` skill both warn that comparing a pyramidal
decode against an iterative one is "a silent 0.2–1 dB error", and require the mode to be
stated on every reported number. P2.2 predicted the same 0.2–1 dB.

**Finding.** Measured over 720 matched settings on Kodak, the two modes differ by a mean of
**+0.0022 dB** (range −0.0146 to +0.0476) — two orders of magnitude below the warning. The
warning is not wrong so much as attributed to the wrong variable. Sweeping iterations on
one bitstream gives −6.11 dB at 1 iteration, −0.26 dB at 5, **−0.0039 dB at the default
10**, +0.0033 dB at 20.

An IFS has a unique attracting fixed point, so both decoders converge to the same image;
pyramidal reaches it sooner by iterating at reduced resolution first. It is a **convergence
accelerator, not an approximation**, which also explains why it is *better* than iterative
wherever the two differ — the opposite of the predicted direction.

**Decision.**

1. `decode_iterations` and `decode_postprocess` are recorded on **every** result row, and
   the sweep was re-run to produce rows carrying them. The mode was already recorded and
   stays recorded — it costs nothing — but the count is the variable that can actually move
   a PSNR by 6 dB, and a row that omits it is not reproducible.
2. The claim is pinned by a test asserting **both** halves: the modes must be
   distinguishable at 1 iteration (or `-i` is not taking effect and recording the mode
   measures nothing) and converged by 10 (or the mode is a genuine confound and every
   comparison must be mode-matched).

**What this rules out.** Treating a mode mismatch as an explanation for a ~0.5 dB
discrepancy at default settings. At 10 iterations it cannot produce one, and reaching for
it would mask the real cause.

**Scope — this holds at `min_size = 4`, and fails badly at `min_size = 2`.** See D11, which
was found while writing up this entry and which corrects an overgeneralisation in it: the
0.002 dB figure is the `default` variant's. On the `min2` variant the two modes differ by a
mean of **3.89 dB** and a maximum of **19.77 dB**.

**What would reverse it.** A `MAX_ALFA` above 1.0, which weakens contractivity and could
leave 10 iterations short of convergence. The sweep holds `-y 1.0` throughout, so this is
untested outside that setting and the test would catch it.

**Tolerance impact.** None. The new assertions are additions.

---

## D10 · 2026-09-13 · `evals/transform` does not count index-structure work, and Gate D should say so

**Context.** §M5 names `evals/transform` the project's primary efficiency metric and calls
it "machine-independent, deterministic, and **non-gameable**" — "the most trustworthy
number the project will produce". Gate D requires it to be "≥ 10× reduced".

**Finding.** All six `comparisons++` sites in `coding_func.c` (lines 98, 266, 445, 634,
811, 972) are at the *exactly* analogous point — the top of the innermost loop, immediately
before loading `s1`/`s2` for the affine fit — so the six methods' counts **are** mutually
comparable, and §M5's definition is met. But the metric counts only the fits, and the
kd-tree methods do substantial *uncounted* work in `kdtree_search` before reaching them:

| method | evals/transform | mean encode s | µs per eval |
|---|---:|---:|---:|
| saupe-fisher | 64.3 | 0.77 | 1.128 |
| mc-saupe | 127.9 | 0.28 | 0.192 |
| saupe | 513.7 | 4.29 | 0.800 |
| fisher | 626.4 | 0.58 | 0.085 |
| masscenter | 1162.4 | 0.95 | 0.076 |
| hurtgen | 1506.8 | 1.19 | 0.073 |

The three classification methods cost 0.073–0.085 µs per eval — near-identical, as they
should be, since an eval is the same affine fit in each. The kd-tree methods cost 0.19–1.13
µs per eval. **Saupe performs fewer evals per transform than Fisher (513.7 vs 626.4) and
takes 7.4× longer in wall-clock** (4.29 s vs 0.58 s). On this pair the metric and the clock
disagree about which method is cheaper.

**Decision.** Record the limitation now, at the point where it was first measured, and
report `µs per eval` alongside `evals/transform` in the Step 2 tables so the gap is visible
rather than latent. `evals/transform` remains the headline metric: it is exact, machine-
independent, and the right measure of *search* work.

**What this rules out.** Reading `evals/transform` as a proxy for encode time across
methods that differ in index structure. It is a valid cost model only within a family
sharing the same per-eval cost and the same (or no) index.

**What this flags for later, and does not decide.** Gate D's "`evals/transform` ≥ 10×
reduced" is satisfiable by moving work into an uncounted index — which is precisely the
shape of Step 13's hierarchical funnel and Step 17's learned pruning, the headline
experiment. As written, Mars 2 could pass Gate D while encoding more slowly than it does
today, and the one method here that is 10× cheaper in evals than Fisher (Saupe-Fisher, at
64.3) is 1.3× *slower* in wall-clock. Whether Gate D should therefore carry a companion
wall-clock condition is a change to the plan's success criteria and belongs to the human at
a gate (§A8), not to this step. It is raised here so that it is raised *before* the steps
that would exploit it, rather than discovered at Gate D.

**Tolerance impact.** None.

---

## D11 · 2026-09-13 · Pyramidal decode breaks on sub-pixel range blocks; the rule is `min_size / 2^levels ≥ 1`

**Context.** D9 concluded from the `default` variant that the decode mode is worth ~0.002 dB
and that the iteration count is the only variable that matters. Checking the `min2`
variant's BD-rate column — which came back undefined for five of six methods — showed that
conclusion was drawn too narrowly.

**Finding.** Per-variant decode-mode deltas over 720 matched settings each:

| variant | mean iterative − pyramidal | min | max |
|---|---:|---:|---:|
| default | +0.0022 dB | −0.0146 | +0.0476 |
| **min2** | **+3.8912 dB** | −0.0046 | **+19.7733** |
| max32 | +0.0018 dB | −0.0247 | +0.0676 |
| step8 | +0.0019 dB | −0.0225 | +0.0366 |
| alfa5 | +0.0028 dB | −0.0182 | +0.0547 |
| beta6 | +0.0026 dB | −0.0162 | +0.0479 |

`decmars` decodes at `1/2^levels` scale before raising resolution (`mars_dec.c:108-123`), so
a range block of `min_size` occupies `min_size / 2^levels` pixels there. Varying image size
and `min_size` independently isolates it:

| image | min_size | levels | block at that level | pyramidal | iterative | gap |
|---|---:|---:|---|---:|---:|---:|
| zoneplate 256² | 2 | 1 | 1.00 px | 22.615 | 22.576 | −0.039 |
| zoneplate 256² | 4 | 1 | 2.00 px | 12.125 | 12.120 | −0.005 |
| mandelbrot 512² | 2 | 2 | **0.50 px** | 32.992 | 38.738 | **+5.746** |
| mandelbrot 512² | 4 | 2 | 1.00 px | 28.764 | 28.810 | +0.046 |
| kodim01 768×512 | 2 | 2 | **0.50 px** | 23.870 | 29.238 | **+5.368** |
| kodim01 768×512 | 4 | 2 | 1.00 px | 27.421 | 27.418 | −0.003 |

The 256² row at `min_size = 2` is the control that makes this a mechanism rather than a
correlation: there the pyramid is one level deep, the block is a full pixel, and the penalty
does not appear. **Small blocks are harmless; sub-pixel blocks at the pyramid's depth are
not.** The threshold is exact in every case measured — ≥ 1 px agrees to < 0.05 dB, 0.5 px
costs 5+ dB.

The consequence for the RD curves is severe enough to be worth stating separately: under
pyramidal decode the `min2` curves run **backwards**. kodim01 reaches 25.38 dB at 1.05 bpp
and only 23.71 dB at 6.30 bpp. That is why `bd_metrics` refuses them, and refusing is
correct — interpolating through that shape would manufacture a number.

**Decision.**

1. Report decode-mode deltas **per variant**, never pooled. A single corpus-wide mean would
   have averaged a 19.8 dB effect into a 0.002 dB one.
2. The report states why `min2`'s BD-rate is undefined rather than printing an em-dash, so
   an unusable cell reads as a result and not as an omission (§A7).
3. A test pins the rule in both directions: sub-pixel must cost > 1 dB, whole-pixel must
   cost < 0.05 dB.

**What this rules out.** Quoting any `min_size = 2` result from a pyramidal decode, and the
general claim in D9 that the decode mode is cheap — it is cheap only while range blocks stay
at or above one pixel at the pyramid's reduced resolution.

**What this means for Mars 2.** Mars 2 has no pyramidal decoder (the plan specifies a
~80-line iterative one), so it does not inherit the defect. But any future multi-resolution
or progressive decoder — **Step 19 is exactly that** — reintroduces the hazard, and its
minimum block size and level count are not independent parameters. Worth knowing before
Step 19 rather than during it.

**Tolerance impact.** None; the new assertions are additions.
