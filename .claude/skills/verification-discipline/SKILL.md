---
name: verification-discipline
description: What to do when a check fails or a tolerance looks tight in Mars 2 — the exact-equality oracles (transform counts, integer moments, CPU/GPU bit-identity, unchanged bitstream), the differential test every optimisation must ship with, and the rule that assertions are never weakened without a recorded CONTRACT-CHANGE and a decisions.md entry. Use when a test, gate, or cross-validation fails, when adding a SIMD, GPU, or parallel path, or when tempted to widen a tolerance or adjust an assertion.
---

# Verification discipline

Source of record: `implementation-plan.md` §2.1 (A2, A3, A7) and §6.

In this project typing is nearly free and **verification is the binding constraint**. The
characteristic failure is not unfinished work — it is a number that looks entirely
reasonable, is wrong, and is inherited by everything downstream.

## A2 — Tests are the contract; they may not be weakened

The natural failure mode of an agent that cannot make code pass is to adjust the
assertion. That is the single thing this project most needs not to happen.

- Golden files are committed. A diff that changes a golden bitstream requires an explicit
  `FORMAT-CHANGE:` trailer in the commit message.
- A CI check fails on any diff that deletes or loosens an assertion without a
  `CONTRACT-CHANGE:` trailer explaining why.
- **Tolerances are declared once in a central constants module, never inline**, so
  widening one is a visible diff rather than a character change buried in a test.

## A7 — Anomalies halt; they are never averaged away

If a cross-validation disagrees, the step **fails**. Widening a tolerance requires a
recorded reason in `docs/decisions.md`. "Close enough" is how a 0.4 dB error becomes
permanent. A disagreement between two metric implementations almost always means a
colour-conversion or window-normalisation difference that would have silently biased
every later result — find it.

**Project-level kill criterion:** if two consecutive gates pass only after a tolerance
was widened, **stop and audit**. Each widening is individually defensible; the sequence
is what quietly empties the numbers of meaning.

## When a check fails — order of investigation

1. **Assume the harness, not the codec.** If a plot disagrees with the pre-registered
   expectation, the harness is more likely wrong than the result is surprising
   (Step 4 states this explicitly for the anchor plot).
2. Reproduce deterministically: fixed seeds, single thread, `fixtures/` before corpus.
3. Bisect against the nearest exact oracle below — they are binary and cheap.
4. If the failure is real and the tolerance is genuinely mis-specified, change it in the
   constants module, in its own commit, with `CONTRACT-CHANGE:` and a `docs/decisions.md`
   entry saying what was measured and why the old bound was wrong.
5. Never silently rescale, crop, re-window, or exclude data to make a check pass.
   Exclusions (e.g. MS-SSIM below 176 px) are **flagged**, never implicit.

## The exact-equality oracles — prefer these to any tolerance

Bit-exact Mars 1 reproduction was dropped as a goal, which removed a harsh
machine-checkable check. These replacements are deliberate, binary, and cheap:

| Where | Oracle |
|---|---|
| Step 5 — `.ifs` parser | Recovered transform count **exactly equals** the C encoder's `transforms` |
| Step 6 — moments | Integer moments **equal** an f64 reference over 10⁶ random blocks — equality, not closeness |
| Step 7 — GPU | Results **bit-identical** to the CPU exhaustive path on every fixture |
| Step 11 — NEON | Differential tests assert **exact equality**; **bitstream unchanged** |
| Step 12 — Rayon | **Bitstream identical at every thread count**; `evals` invariant under thread count |
| Step 10 — format | Lossless round-trip on all fixtures; decoder fuzzed (`cargo-fuzz`): no panics, no OOM, no unbounded allocation |
| Step 8 — oracle | Exhaustive self-test: 100% top-1 recall, 0 dB regret |

Any divergence in these is a **bug, never a tolerance**. A GPU/CPU divergence is a
project-level kill criterion, because it would mean the integer-exact numerics were wrong
and the oracle untrustworthy.

## Tolerances that do exist, and why they are loose

`0.2 dB` against the Mars 1 baseline (Steps 6, 9) and `5%` on `evals/transform` (Step 9)
are deliberately looser than a bit-exact project would use: without bit-exactness,
unreachably tight bounds only invite the tolerance-widening A2 exists to prevent. They
are not invitations to loosen further.

Also fixed: CI regression gate — golden bitstream changes need `FORMAT-CHANGE:`, PSNR
may not drop > 0.05 dB at matched settings, encode time may not regress > 15% vs. the
rolling median of the last 10 `main` runs (loose on purpose; CI machines are noisy —
real performance work is measured on a pinned box).

## A3 — Differential testing carries the load code review would

Every optimisation ships **alongside the reference implementation it replaces**, plus a
randomised differential test: scalar vs NEON, CPU vs GPU, f64 vs integer-exact.

Prefer **properties that hold regardless of implementation**:

- round-trip identity,
- RD monotonicity,
- λ-sweep convexity (non-convexity is the fastest available diagnostic for a
  rate-estimation bug),
- `evals` counts invariant under thread count,
- bitstream invariance under any change claimed to be performance-only.

`unsafe` is confined to `mars-simd` and `mars-gpu`, and only behind a safe wrapper with a
scalar reference and a differential test.

## Before claiming a step passed

- [ ] The exact oracle for this step passes as *equality*, where one exists.
- [ ] No assertion was deleted or loosened; if one was, it has `CONTRACT-CHANGE:` and a
      `docs/decisions.md` entry.
- [ ] Every new optimisation has a scalar reference and a randomised differential test.
- [ ] Performance-only changes left the bitstream byte-identical.
- [ ] Any surprise versus `docs/predictions.md` is written down, not explained away.
