# Mars 2 — Post-Gate-D Improvement Plan

Companion to [implementation-plan.md](implementation-plan.md), which remains the source of
record for Steps 0–21, §2.1's working agreement, and §6's kill criteria — this document
does not restate or override any of that. It exists because Gate D's own attribution table
(`docs/decisions.md` D44) and the Step 15/19 outcomes (D40/D42, D47) named **specific,
already-root-caused regressions and left-on-the-table levers** that Steps 0–19 did not have
budget to act on. Those findings are real work, not speculation: each step below traces to
a numbered decision entry with measured numbers, not a guess about what might help.

**Version:** 1.0 · **Status:** proposed · **Last updated:** 2026-09-15

Steps here are numbered **22 onward**, continuing `implementation-plan.md`'s sequence.
Steps 20 (INR fallback) and 21 (Release) from that plan are still open and are not
duplicated here — see that document. Every step below follows the same brief contract as
`implementation-plan.md` (§2.1 A1–A8) and the `plan-step` skill: entry condition,
hypothesis, exit criteria (a command, not a judgement), abort rule, size. `just gate-N`
must exist and exit 0/1 before a step is marked closed.

---

## Why these three, in this order

Each targets a distinct, already-measured shortfall, cheapest-verification-first:

1. **Step 22** fixes a *regression* (Step 15's residual mode makes BD-rate worse, not
   better, at some sweep points) — a correctness-adjacent gap, and the cheapest to check
   (same gate, same corpus, same metric already wired up).
2. **Step 23** fixes a *known-inefficient design* (Step 19's progressive penalty, 64.22%
   vs. a 10–20% prediction) using three specific mechanisms D47 already named and
   attributed numerically — this is closing a gap whose cause is understood, not
   open-ended exploration.
3. **Step 24** revisits Step 17's learned pruning on the cross-corpus and full-corpus
   scope D45 explicitly flagged as untested — lowest confidence of the three, correctly
   sequenced last.

None of these are new hypotheses about the representation; all three are follow-through on
work already done. That is deliberate — per A7 (`implementation-plan.md` §2.1), an anomaly
already found and explained is a debt, not a data point to re-litigate.

---

#### Step 22 — Residual mode: close the +2.05% regression

**Entry condition.** Step 15 (`gate-15`) passed on record. `crate::dct`, `crate::quant`,
`crate::residual` exist and are unchanged in interface.

**Context, from `docs/decisions.md` D40/D42.** Mode 3 (fractal + residual) competes under
`J = D + λR` against modes 0/1/2, but its DCT-domain quantisation step is a **compile-time
constant** (`RESIDUAL_QSTEP_DEFAULT = 8.0`, `RESIDUAL_DEAD_ZONE = 0.5`), not λ-adaptive —
a deliberate first-cut simplification named at the time, with its cost stated plainly: mode
3 is quantisation-mismatched at the sweep's extremes, producing a measured **+2.05% mean
BD-rate regression** relative to Step 14 rather than the improvement the mode exists to
provide. D42 records a two-pass warm-up attempt at fixing a related Step-15-era anomaly
that made the regression *worse*, reverted for time-budget reasons — that attempt is not
to be re-run unmodified; read D41/D42 before touching warm-up behaviour.

**Hypothesis.** Making `RESIDUAL_QSTEP` a function of λ (coarser quantisation at high λ /
low bpp, finer at low λ / high bpp) — the header/format-growth cost D40 named as the reason
it wasn't done originally — removes most or all of the +2.05% regression, because the
mismatch D40 attributes the regression to is specifically a fixed-step-vs-varying-target-
rate problem.

**Deliverable**
- `RESIDUAL_QSTEP` becomes a per-encode parameter derived from λ (closed-form or small
  lookup, not learned), threaded through `mars_codec::encode` the same way Step 14's other
  λ-dependent decisions already are.
- Format/header change, if any, documented in `docs/mars-format.md` alongside the existing
  Step 10 fields, with a version bump — do not silently reinterpret an existing field.
- Mode-usage histogram re-run at the same rate points as Step 15's original measurement,
  so mode-share shifts (if any) are visible, not just the BD-rate delta.

**Exit criteria**
- `just gate-22`: BD-rate vs. Step 14 baseline, same corpus and rate points D40 used,
  reported as a signed percentage exactly like D40's — a regression that shrinks but does
  not flip sign must be reported as such, not rounded to "fixed."
- Explicit before/after comparison against D40's +2.05% number in the gate's own output.

**Abort rule.** If a λ-adaptive quantisation step reduces the regression by less than half,
stop and record why in `docs/decisions.md` rather than iterating on quantisation further —
that would indicate the mismatch D40 named is not the dominant cause, and the residual
coding scheme itself (dead-zone width, band bucketing) needs to be questioned before
quantisation is tuned further.

**Size.** M · **Verification burden:** medium (same metrics pipeline as Step 15, no new
oracle).

---

#### Step 23 — Progressive bitstream: rate-optimise the three named levers

**Entry condition.** Step 19 (`gate-19`) passed on record. `progressive_gate.rs` and its
`MARS_RUN_PROGRESSIVE_GATE=1` harness exist and are unchanged.

**Context, from `docs/decisions.md` D47.** Step 19's progressive penalty measured 64.22%
BD-rate on kodim01 at λ=200 — over 3× the 10–20% predicted in `docs/predictions.md` P19.2
— and D47 root-caused it to three **named, separately attributed** mechanisms, not one:
1. Layer 2's flat-DC bits for fractal leaves (mode 2/3) are spent and then wholly
   superseded by layer 3, with zero reuse.
2. `qbeta` is coded twice — once as layer 2's flat guess, once as layer 3's real fractal
   value — with no delta between them, hitting 7,097 of 7,737 leaves in the measured case.
3. Each of the 4 layers is an independent `mars_entropy` stream with cold-started adaptive
   context models, rather than one continuously-adapting model — a cost D47 flags as *not*
   accounted for in the original P19.2 prediction at all.

D47 explicitly names the three fixes and explicitly declines to attempt them under time
pressure: delta-code `qbeta` between layers 2 and 3; warm-start each layer's entropy models
from a frozen snapshot (D47 points at Step 14's `crate::rate` module, which already does
snapshot-based warm-starting for a different purpose, as prior art in this codebase); make
layer 3 code a true residual correction against layer 2's guess instead of an independent
value.

**Hypothesis.** The three mechanisms are independently additive and separately
measurable — implementing and gating them **one at a time**, in the order D47 lists them
(highest attributed leaf-count impact first), will show which of the three actually moves
the number, rather than bundling them into one change whose attribution would be lost.
This mirrors D40's own caution: an unattributed combined fix is exactly the kind of change
this project's convention (A7) treats as suspect.

**Deliverable**
- 23a — delta-code `qbeta` (layer 3 value − layer 2 value) instead of two independent
  codes, for mode-2/3 leaves only (mode 0/1 leaves are already fully resolved by layer 2
  per D46's design and are out of scope).
- 23b — entropy model warm-start: each layer after the first initialises its adaptive
  context models from the previous layer's final state (a frozen snapshot, per D47's
  pointer to `crate::rate`), instead of cold-starting.
- 23c — layer 3 codes a residual against layer 2's rendered value for mode-2/3 leaves,
  rather than an independent qalfa/qbeta/isometry/domain tuple. This is the deepest change
  of the three — check it does not reintroduce D11's hazard (D46's own warning: any change
  that makes a leaf's *rendered value* depend on more than its own full-resolution `row`/
  `col`/`size` at every layer needs the same scrutiny D46 gave the original design) and
  does not break P19.1's bit-exact "every prefix decodes" property.
- Each sub-step gated and recorded separately; do not land 23a–23c as one commit or one
  gate run — the attribution is the point.

**Exit criteria**
- `just gate-23`: re-measures kodim01 at λ=200 with the same `LAMBDA_GRID` and prefix-curve
  methodology D47 used, reporting the progressive-penalty percentage after each of 23a,
  23b, 23c cumulatively, so the marginal contribution of each lever is visible in one
  table — the attribution table D47 itself did not have time to build forward-looking.
- P19.1 (bit-exact 4-layer equality) and "every prefix decodes" re-verified unchanged after
  each sub-step — these are exact-equality oracles per the `verification-discipline`
  skill, not something a rate-optimisation change is allowed to loosen.

**Abort rule.** If after all three levers the penalty remains above 30% (still outside
P19.2's original band even loosened by 50%), stop, record the residual gap in
`docs/decisions.md` as D47 did, and treat "independent per-layer streams" as a structural
property of this design rather than a bug to keep chasing — per-layer streams were chosen
for prefix-decodability, and D47 already flagged that as the least name-checked but
possibly largest of the three mechanisms; if it dominates, further gains would require
revisiting D46's layering design itself, which is out of this step's scope.

**Size.** L (three sub-steps) · **Verification burden:** high — touches the bitstream
format and must not regress P19.1's exact-equality oracle.

---

#### Step 24 — Learned pruning: cross-corpus and full-corpus re-measurement

**Entry condition.** Step 17 (`gate-17`) passed on record, closed as a documented negative
result (D45: `Learned` does not beat `Funnel` same-corpus, in-sample).

**Context, from `docs/decisions.md` D45.** Step 17's abort rule was checked honestly and
correctly triggered — this is not being reopened because the result was wrong. D45 names
two specific, explicit scope reductions, not general uncertainty: the recall/regret table
covered 2 of 24 `standard/`-corpus images (`kodim01`, `kodim02`), and cross-corpus
generalisation (train Kodak, eval CLIC/USC-SIPI) "was not attempted" because no CLIC or
USC-SIPI images existed on disk that session. D45 states plainly that neither gap changes
the step's already-negative outcome given the in-sample loss to `Funnel`, but both remain
named, real limitations of that entry's evidence, not resolved ones.

**Hypothesis.** This step's own prior is that the result will not change — D45's
same-corpus, in-sample loss to `Funnel` is unlikely to flip on more data — and Step 24
exists to either confirm that at proper scope (closing D45's named gap for good) or
surface a narrower regime (e.g. a specific corpus or recall band) where `Learned` wins,
which D45's 2-image measurement was too small to detect either way.

**Deliverable**
- Full 24-image `standard/` Kodak corpus recall/regret table for `Learned` vs. `Funnel`
  vs. `Exhaustive`, matching D45's original methodology exactly (same oracle config, same
  held-out split discipline) so the two results are directly comparable.
- CLIC and/or USC-SIPI images fetched into `corpus/` (respecting whatever corpus-fetch
  policy is current — D45 records that the prior session was explicitly told not to fetch
  a new large corpus; confirm that instruction no longer applies before spending the
  bandwidth) and the same table computed cross-corpus (train Kodak, eval CLIC/USC-SIPI).

**Exit criteria**
- `just gate-24`: reports recall/regret/evals-per-transform for `Learned` vs. `Funnel` at
  full 24-image same-corpus scope and at cross-corpus scope, both compared explicitly
  against D45's 2-image numbers (kodim01 recall/regret/evals, kodim02 same).
- A plain statement of whether the full-scope result confirms, narrows, or overturns D45's
  conclusion — this step's deliverable is the honest comparison, not a specific verdict.

**Abort rule.** If the full same-corpus result already confirms D45 (Learned still loses to
Funnel, no narrower winning regime visible), skip the cross-corpus leg — D45's own
reasoning ("this gap does not change the step's outcome") applies with more force once the
easier check is confirmed at full scope, and cross-corpus work should not be spent
confirming an already-confirmed negative result.

**Size.** M (same-corpus only) to L (with cross-corpus) · **Verification burden:** low —
reuses Step 17's harness and oracle, no new format or code path.

---

## What this document deliberately does not include

- Step 20 (INR fallback) and Step 21 (Release) — open items in `implementation-plan.md`,
  unrelated to the regressions this document tracks.
- Any new representation or search idea not already measured and attributed in
  `docs/decisions.md`. Per A7, a fix follows a named cause; it does not precede one.
- A retry of D41/D42's reverted two-pass warm-up as-is — Step 22 must read those entries
  before proposing anything that touches warm-up behaviour, so the same anomaly is not
  reintroduced blind.
