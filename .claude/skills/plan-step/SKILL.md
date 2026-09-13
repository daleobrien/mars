---
name: plan-step
description: Procedure for executing one numbered step of implementation-plan.md — locating the step's self-contained brief, writing the prediction before measuring, worktree and branch rules, the `just gate-N` exit command, what belongs in docs/decisions.md, and the gate and kill criteria that decide whether to continue. Use when starting, resuming, or closing out a Step N, when asked what to work on next, or when a gate is being checked.
---

# Executing a plan step

Source of record: `implementation-plan.md` §2.1 (working agreement A1–A8), §4 (the
steps), §5 (dependency graph), §6 (kill criteria).

Agents start cold every session. Each step in the plan is written as a **self-contained
brief**: goal, deliverable, exit criteria, size, verification burden. Read the whole step
section before writing code, plus §1 and §2 once per session.

## Order of work

1. **Locate the brief.** `grep -n "^#### Step N" implementation-plan.md`, then read that
   section through to the next `---`. Check §5's table for what the step depends on and
   what may run concurrently with it. Do not start a group before its gate has passed.
2. **Write the prediction first (A4).** Before any sweep or benchmark, append the
   expected outcome to `docs/predictions.md` (append-only). An agent that measures first
   and explains afterwards will rationalise almost any result; the prediction is what
   makes a surprise legible as a surprise. It costs a paragraph and catches harness bugs
   nothing else will.
3. **One step per worktree (A6).** Parallel work happens on separate branches in
   separate git worktrees, one agent each. Fan out on *development*; serialise on
   *measurement*.
4. **Build the deliverable exactly as listed.** Traits are introduced when there are two
   implementations, not in anticipation of one. Every experiment is a config file, not a
   code edit. No `unsafe` outside `mars-simd` and `mars-gpu`, and there only behind a
   safe wrapper with a scalar reference and a differential test.
5. **Exit is a command, not a judgement (A1).** `just gate-N` exits 0 or 1. "The RD curve
   looks reasonable" is not an exit criterion. If a criterion cannot be expressed as a
   check, make it checkable or delete it.
6. **Record decisions and anomalies.** Any tolerance change, any surprise, any
   deviation from the brief goes in `docs/decisions.md` with its reason. Anomalies halt
   the step; they are never averaged away (A7).

## Determinism and code rules (§2.3)

- No wall-clock seeds, no unordered iteration affecting output, no floating-point
  reassociation. Nondeterminism is opt-in and recorded.
- Bitstream output must be identical across thread counts and across SIMD/scalar paths.
- Deterministic-by-default is what makes the cheap exactness oracles possible — see the
  `verification-discipline` skill.

## Gates

Gates are where a **human** reads output (A8); between gates the harness is the reviewer.
Do not mark a gate passed on your own inspection — present the gate command's output.

- **Gate A** — metrics cross-validated, BD-rate tested, Mars 1 baselines recorded for
  6 methods × 5 rates × 24 images, `evals/transform` per method, golden fixtures frozen
  and the format spec independently validated, anchor curves recorded. Nothing of Mars 2
  exists yet, and that is the point.
- **Gate B** (`just gate-b`) — `.ifs` parser validated by exact transform-count
  equality; exhaustive encoder within 0.2 dB of baseline with integer moments proven
  exact; GPU search bit-identical to CPU and ≥ 50× faster with the Kodak oracle built;
  six classical methods ported with the recall/regret table published. **This is a
  defensible stopping point** — everything after it is optional and gate-driven.
- **Gate C** — the performance gate; see `benchmark-protocol`.
- **Gate D** — BD-rate ≥ 35% better than the Mars 1 baseline, `evals/transform` ≥ 10×
  reduced, every step's contribution separately attributed in one table.

## Kill criteria — check these rather than pushing on

- GPU search cannot be made bit-identical to CPU → **stop the project**; the oracle would
  be untrustworthy and it is the measurement foundation.
- After Gate C, CPU encode speedup < 5× → **stop the project**.
- Step 14's RD optimisation yields < 3% BD-rate → **stop after Gate B and write it up**;
  the representation, not the optimiser, is the binding constraint.
- Step 13's funnel already achieves > 99% eval reduction at full recall → **drop Group D**.
- **Two consecutive gates passed only after a tolerance was widened → stop and audit.**
  Each widening is individually defensible; the *sequence* is what quietly empties the
  numbers of meaning, and no per-step check can see it.

Group D steps (17, 19, 20) are hypotheses with entry conditions and abort rules, not
commitments. A well-measured negative result is a deliverable: nobody has published the
recall/regret frontier for fractal search.

## Closing a step

- [ ] `just gate-N` exits 0, with output captured.
- [ ] Prediction was recorded before the measurement, and the outcome compared to it.
- [ ] Results appended to `results/` with provenance; nothing overwritten.
- [ ] Any tolerance change or anomaly recorded in `docs/decisions.md`.
- [ ] Deliverables in the brief all present — including docs, fixtures, and manifests,
      not only code.
