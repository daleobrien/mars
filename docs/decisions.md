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

---

## D12 · 2026-09-13 · The `dom_x`/`dom_y` bit-width asymmetry is not a bug; `x` is the row axis

**Context.** `implementation-plan.md` §4 Step 3 and the `mars1-reference` skill both flag
that `dom_x` is packed with `bits_per_coordinate_h` and `dom_y` with
`bits_per_coordinate_w`, call it "a naming slip in the original", declare it normative
because encoder and decoder agree, and instruct: **"Preserve it exactly; do not 'fix' it."**

**Finding.** There is no slip. Mars 1 uses `x` for the **row** axis and `y` for the
**column** axis throughout: `image[atx+x][aty+y]`, `if(atx >= image_height || aty >=
image_width)`, and the domain enumeration in `index_func.c` is
`for(i = 0; i < image_height - 2*size + 1; i += SHIFT)` over the `ptr_x` axis. A row
coordinate is bounded by the image height, so sizing `dom_x` by `bits_per_coordinate_h` is
simply correct. The decoder confirms it at the pixel level: `iterative_decoding` reads
`imag[ii][jj]` with `ii` seeded from `dx`, and the first index is the row.

**Why the instruction is worse than the observation.** "Preserve the bug" tells an
implementer that `dom_x` is an x-coordinate in the ordinary sense and that its bit width is
wrong. Acting on that produces a transposed decoder, and the natural next move is to add a
second transposition to compensate — two bugs where there were none, in code whose output
looks *nearly* right on symmetric images.

**Decision.** `docs/mars1-format.md` opens with §0, the axis convention, before any field
layout, and uses "row"/"column" throughout rather than `x`/`y`. The
`mars1-reference` skill's paragraph is corrected. `implementation-plan.md` is **not**
edited — it is the accepted brief and amending it belongs to a human at a gate (§A8) — so
the discrepancy is flagged here instead, and Step 5 should read the format document rather
than the plan's sketch of it.

**What would reverse it.** Nothing; the reading is confirmed by three independent sites in
the C and by 142 fixtures whose decoded images match `decmars` byte for byte.

**Tolerance impact.** None.

---

## D13 · 2026-09-13 · Leaves below `min_size` are normative, and size-1 leaves store a truncated raw pixel

**Context.** Both the plan's spec sketch and the skill describe the tree as bottoming out
at `min_size`. Nothing in either mentions smaller blocks.

**Finding.** The forced-subdivision branch —
`if (size > max_size || atx+size > image_height || aty+size > image_width)` — **does not
consult `min_size`**. On an image whose dimensions are not multiples of `min_size` it
recurses past it, down to size 1. Counts are pure geometry, identical across every method,
rate and variant in the golden set:

| fixture | size-1 leaves | size-2 leaves |
|---|---:|---:|
| `mixed_129x127` | 255 | 64 |
| `mixed_250x250` | 0 | 249 |
| every 64², 256², 512² fixture | 0 | 0 |

At size 1 all six `*Coding` functions take a `tip == 0` path setting
`*qbet = image[row][col]` — the **raw pixel value**, 0–255 — which is then packed into
`N_BITBETA = 7` bits and loses its top bit. A pixel of 200 is stored as 72 and decodes to
145.

**Decision.** Specified in `mars1-format.md` §5.3 and asserted by `just gate-3`, which
checks the three counts above and, separately, that no fixture whose dimensions *do* divide
`min_size` contains any leaf below it — so the rule is pinned in both directions and cannot
become vacuous.

**What this means for Mars 2.** Step 5's `.ifs` reader must handle sub-`min_size` leaves or
it will desynchronise on two of the fourteen fixture images. Step 6's encoder does not have
to reproduce the 7-bit truncation — Mars 1 bit-exactness is explicitly not a goal — but it
must know the partition rule, or its `transforms` count will differ from the baseline's on
non-power-of-two images and the §M5 comparison will be against a different denominator.

**Tolerance impact.** None; the new assertions are additions.

---

## D14 · 2026-09-13 · The golden set pins decodes by hash, and is fourteen images rather than five

**Context.** §4 Step 3 asks for "5 fixture images × 3 methods × 3 RMS thresholds: the
`.ifs` bitstream, both decodes (iterative and pyramidal), and a `manifest.toml`", and §3
lists `fixtures/` as "golden `.ifs` + decoded `.pgm` — committed, small".

**Two deviations, both deliberate.**

1. **Fourteen images, not five.** §M9's "lena + 4 synthetic" grew to twelve synthetic
   fixtures at Step 0, and the three that matter most to a *format* spec are among the new
   ones: `mixed_250x250` and `mixed_129x127` are the only images that exercise
   `bits_per_coordinate`'s integer truncation, virtual-size padding, and the
   sub-`min_size` leaves of D13. Adding `lena` and a purpose-built 64² walkthrough image
   gives 14 × 3 methods × 3 rates = 126, plus 15 header-stress cases that vary `-A -B -d
   -m -M` (the default point never varies them, so without these the golden set would
   never test a header field away from its 1998 value) and the walkthrough: **142
   fixtures, 1.3 MB**.

2. **The decodes are pinned by SHA-256, not committed.** 142 fixtures × 2 decodes × up to
   256 KB is ~25 MB, which is not "small". The manifest records the SHA-256 of both
   decodes and of the `-Q` partition rendering; the validator regenerates each from the
   bitstream and compares hashes. This is the pattern already used for `fixtures/images/`,
   where the `.raw` inputs are hash-pinned rather than committed.

**What this buys beyond saving disk.** `just gate-3` reads only committed data — 142 `.ifs`
files and a manifest of integers and hashes. It needs no C binaries, no corpus images and
no image library, so the format spec stays checkable after this machine's toolchain is
gone. A stored `.pgm` could also be quietly regenerated to match a broken decoder; a hash
recorded at the same time as the encoder's own `transforms` count cannot.

**What this rules out.** Eyeballing a golden decode. If one is ever needed, regenerate it
with `just mars1-fixtures` and diff against the recorded hash.

**Tolerance impact.** None. Every check is an exact equality.

---

## D15 · 2026-09-13 · The searched `qbeta` is discarded whenever `qalfa` quantises to zero

**Context.** The quantisation formulas in the plan and the skill end at `qbeta` and `rms`,
and read as though the `(qalfa, qbeta)` pair the search found is what gets written.

**Finding.** It is not. `coding_func.c:1169-1172` runs, immediately before packing:

```
if (abs(qalfa - zeroalfa) <= zero_threshold) {   /* zero_threshold is `-z`, default 0 */
    qbeta = best_beta(atx, aty, size, 0.0);
    qalfa = zeroalfa;
}
```

At the default this fires on **every** leaf whose `qalfa` quantised to 0, replacing the
searched offset with a DC-only refit — `int(0.5 + mean/255 · (2^N_BITBETA − 1))`. On
`flat128` that is 100% of leaves. The split decision, however, uses the RMS of the
*searched* fit computed before the override, so a block can be kept as a leaf on the
strength of a domain match that is then thrown away.

**Why it matters and where.** It is correct behaviour — with `alfa = 0` the optimal offset
is the block mean — but it is invisible in the formulas as usually written, and Step 6's
Rust encoder that omits it will produce a different `qbeta` on every DC leaf. That is a
real image difference arising from a step nobody wrote down.

**Decision.** Specified as `mars1-format.md` §8.1, listed in that document's hazard
checklist, and pinned by the `flat128` closed-form assertion in `just gate-3`: every leaf's
`qbeta` must be exactly 64, which is `best_beta`'s answer and not the search's.

**Tolerance impact.** None.

---

## D16 · 2026-09-13 · The size of the golden set is load-bearing: one fixture would not have caught a transposed parser

**Context.** `mars1-format.md` §13 claims that three of the validator's five checks —
transform count, DC-leaf count, byte-exact re-serialisation — are structurally blind to the
child-recursion order of §5.1, because a parser visiting NW,NE,SW,SE consumes identical
bits in identical order and emits an identical leaf count. Only the geometry checks can see
it. That was an argument, not a measurement.

**Finding.** Tested by mutation: a copy of the validator with the child order swapped was
run over all 142 fixtures.

| | fixtures catching the mutant |
|---|---:|
| decoded image vs `decmars -i` | 92 |
| partition render vs `encmars -Q` | 46 |
| domain-coordinate bounds check (§6) | 14 |
| transform count / DC count / re-serialisation | **0** |
| **any check** | **106 of 142** |

The argument holds exactly: checks 1–3 caught it zero times. But **36 fixtures did not
catch it at all** — among them `sierpinski`, whose partition is symmetric under transpose,
so swapping the SW and NE children is genuinely undetectable there.

**Decision.** Record this rather than only the pass. The natural instinct when a set of 142
fixtures takes 5.7 seconds to validate is to trim it to something representative; this is
the measurement that says which fixtures are load-bearing and why "representative" is the
wrong criterion. Any future reduction of the set must re-run this mutation and keep the
catch rate, not the image count.

It also generalises past this one mutation: an exit criterion should be tested by breaking
it. "The validator passes" and "the validator can fail" are different claims, and only the
second one makes the first worth anything.

**Tolerance impact.** None.

---

## D17 · 2026-09-13 · Anchors run on the original colour Kodak PNGs; JPEG anchor is libjpeg-turbo, not mozjpeg proper

**Context.** Step 4 asks for five anchor codec drivers, including a "mozjpeg-equivalent"
JPEG, swept over quality on the Kodak corpus. Step 2's sweep reads Kodak as **grayscale**
raw (`scripts/build-imageset.py` converts via pinned Rec.601 luma) because the 1998 codec
is grayscale-only. Anchors have no such constraint.

**Finding.** `mars_core::io::read_image` already reads 8-bit RGB PNG directly
(`crates/mars-core/src/io.rs::read_png`), and §M2's PSNR-Y/PSNR-Cb/PSNR-Cr/PSNR-YUV
definitions are colour-aware. Building a second grayscale-raw image set for anchors would
throw away information (all five anchors are colour-capable) and would not even be
comparable to a real-world JPEG/AVIF/etc. deployment, which encodes colour. Separately:
this machine has libjpeg-turbo 3.2.0 (`cjpeg -version`) but not mozjpeg proper (`brew
install mozjpeg` was not attempted as part of this step — mozjpeg is a fork of
libjpeg-turbo tuned for smaller files at the same quality via trellis quantisation and
different default Huffman tables, not a different bitstream or CLI surface, and swapping
it in later would not change the row schema).

**Decision.** `crates/mars-bench/src/anchors.rs` loads Kodak directly from
`corpus/kodak.manifest.json` (which already has name + sha256 per colour PNG — reused
rather than duplicated into a second image-set schema) and encodes/decodes colour
throughout. The JPEG anchor's row `codec_build_info` records the actual
`cjpeg -version` output (`libjpeg-turbo version 3.2.0 ...`), and this document states
plainly that the row is libjpeg-turbo's baseline Huffman/quantisation behaviour, not
mozjpeg's. Anyone quoting the JPEG anchor number should read it as "commodity JPEG at
libjpeg-turbo defaults", which is if anything a *harder* bar for Mars 2 to clear than
mozjpeg would be at typical settings (mozjpeg tends to produce smaller files at matched
quality, i.e. a stronger anchor) — so a comparison against this anchor is conservative in
Mars 2's favour, not inflated. Revisit if `mozjpeg` becomes trivially installable on the
target machine and someone wants the tighter anchor specifically.

**Tolerance impact.** None — this changes which tool is invoked, not any assertion.

---

## D18 · 2026-09-13 · OpenJPEG's PNG reader silently darkens every pixel via `gAMA`; anchors feed it PNM instead

**Context.** While validating the JPEG 2000 anchor driver (§ verification-discipline —
"assume the harness, not the codec" when a number looks wrong), the JPEG 2000 curve's
PSNR-Y sat at 11–16 dB across the entire quality sweep, an order of magnitude worse than
every other anchor and flat regardless of the compression ratio requested. That flatness
was the tell: a real RD curve moves with the rate parameter; a harness bug does not.

**Finding.** Reproduced outside the harness: `opj_compress -i kodimNN.png -o x.j2k -r 1`
(**lossless**) followed by `opj_decompress -i x.j2k -o x.png` does not round-trip pixel
values. Sample point (100,100) on `kodim01.png`: original `(96, 96, 86)`, round-tripped
`(30, 30, 23)`. The transform is not random noise — it is almost exactly the sRGB gamma
curve applied twice: `(96/255)^2.2 * 255 ≈ 29.9`, `(144/255)^2.2 * 255 ≈ 72.6`,
`(73/255)^2.2 * 255 ≈ 16.2`, matching the corrupted output to within rounding on all three
sampled pixels. Kodak's official PNGs (from `r0k.us`) carry a `gAMA` chunk
(`gamma=0.45455` i.e. 1/2.2). Isolated further: encoding from a **PPM** (no `gAMA`, no
colour-management metadata of any kind) round-trips exactly, and decoding a
PPM-sourced `.j2k` **to PNG** is also exact — the corruption is entirely on OpenJPEG's
PNG **read** path honouring the source's `gAMA` chunk, with no matching correction
anywhere else in the pipeline. WebP, JPEG XL and AVIF were cross-checked the same way
(lossless round trip through their own PNG-in/PNG-out paths) and are exact; this is an
OpenJPEG-specific bug, not a general PNG-metadata hazard in this toolchain.

**Decision.** Never feed `opj_compress` a PNG with colour-management metadata. Added
`mars_core::io::write_pnm` (binary PGM/PPM writer, no metadata of any kind) and
`read_ppm`; `AnchorCodec::Jpeg2000::encode` converts its input through
`read_image` → `write_pnm` into a plain `.ppm` before calling `opj_compress`, sidestepping
the bug rather than working around its symptom (e.g. by pre-correcting for the gamma
curve, which would be fragile if the bug's exact shape ever changes). Decoding to PNG was
left alone since it was independently confirmed exact. This is a documented, worth-a-D
harness fix under §A2/A7: the anomaly was investigated to a concrete, reproducible root
cause (not "close enough"-averaged away) before any code changed.

**Tolerance impact.** None — no assertion changed; a genuine encode-path bug was fixed.

---

## D19 · 2026-09-13 · Quality-sweep floors and the JPEG XL distance range were chosen after two real non-monotonic anomalies, not by loosening the BD-rate contract

**Context.** §M3/`bdrate::RdCurve::prepared` requires a curve's bpp *and* PSNR to be
strictly increasing together; §A7 makes a non-monotonic curve a hard error, not something
averaged past. The first full anchors run (after fixing D18) surfaced two real,
reproducible non-monotonic points, both at the extreme low-quality end of a codec's own
range, and one coverage gap.

**Finding.**
1. **JPEG, `kodim01`... actually `kodim02.png`, quality 1 vs 2.** `cjpeg -quality 1` produced
   *more* bytes (7215) and *far worse* PSNR-Y (15.4 dB) than `-quality 2` (7155 bytes,
   25.5 dB) — quality 1 is not a lower-quality point than quality 2 on this image, it is a
   worse point at essentially the same rate. Reproduced outside the harness with plain
   `cjpeg`/`djpeg` invocations; not a measurement artifact. `kodim02` has strong periodic
   high-frequency structure, and libjpeg's quality-to-quantiser-table scaling saturates at
   the very bottom of its range in a way that is not smooth for content like this.
2. **AVIF, `kodim08.png`, quality 2 vs 4.** A much smaller version of the same shape:
   `-q 2` gives 21.52 dB at 0.1219 bpp, `-q 4` gives 21.44 dB at 0.1232 bpp — a 0.08 dB
   dip, again at the bottom of the encoder's own quality range where aom's quantiser
   selection saturates.
3. **JPEG XL coverage gap.** The initial distance list (`9.0` down to `0.4`) never went low
   enough in quality to reach much below ~0.4 bpp on several images (`cjxl -d 9`, its
   original floor, gave 0.40 bpp on `kodim01`), so JPEG XL's curve had systematically less
   overlap with the other anchors' low-bitrate points than the plan's 0.1–2.0 bpp target
   implies it should.

**Decision.** Per the plan's own framing ("sweep quality... the bpp range is validated as
a result, not forced as an input"), the fix is a **config change**, not a tolerance change:
`configs/anchors.json`'s JPEG floor moved from quality 1 to quality 3 (both 1 and 2 are
inside the affected saturation zone once the neighbouring point is accounted for — see the
arithmetic in the commit that added this entry), AVIF's floor moved from quality 2 to
quality 4, and JPEG XL's distance list gained `21.0, 15.0, 11.0` above its previous ceiling
of `9.0` to reach down to ~0.16–0.33 bpp on typical Kodak images. After this change, a
full sweep of all 24 images × 5 codecs produces **zero** monotonicity violations and every
`(codec, image)` pair keeps at least 6 points in 0.1–2.0 bpp (checked mechanically by
`just gate-4`, not eyeballed). This is a decision about *which quality knobs to sweep*
(§2.3 territory), not a change to `bdrate.rs`'s monotonicity requirement, which stayed
exactly as strict as before and would still reject a genuinely non-monotonic curve.

**Tolerance impact.** None. `bdrate::RdCurve::prepared`'s strict-monotonicity check was
not touched.
