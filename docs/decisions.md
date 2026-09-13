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
