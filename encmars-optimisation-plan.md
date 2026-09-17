# Mars 2 — `encmars` Optimisation Implementation Plan

**Status:** proposed  
**Purpose:** implementation order for the Mars 2 encoder after the research already completed in the repository  
**Companion documents:**  
- [`implementation-plan.md`](implementation-plan.md) — authoritative implementation steps and gates
- [`technical-research-and-development-plan.md`](technical-research-and-development-plan.md) — research rationale
- [`improvement-plan.md`](improvement-plan.md) — post-Gate-D measured follow-up
- [`encmars-decmars-cli-plan.md`](encmars-decmars-cli-plan.md) — CLI integration

---

## 1. Purpose

Mars 2 has already done the expensive part of the codec design:

- the measurement contract exists;
- the Mars 1 baseline exists;
- the `.mars` representation direction is defined;
- the exhaustive oracle is part of the implementation plan;
- exact integer moment arithmetic has been designed;
- classical search methods have been identified for comparison;
- the hierarchical search funnel is specified;
- NEON/SIMD and Rayon execution models are specified;
- `J = D + λR` is established as the rate-distortion decision rule;
- residual coding and modern entropy coding are specified;
- adaptive partitioning and adaptive domain density are established research directions;
- post-Gate-D work has already identified concrete regressions and their causes.

The goal of this document is therefore **not more codec research**.

The goal is:

> **Build the first Mars 2 encoder in the order most likely to produce a fast, small codec, while preserving the measurement discipline that makes every optimisation attributable.**

This plan deliberately avoids re-running research that the repository has already settled.

---

# 2. Guiding strategy

The implementation should proceed in four broad stages:

```text
1. Build a trustworthy encoder around the exhaustive/oracle machinery
                  ↓
2. Make the CPU implementation fast
                  ↓
3. Spend that search budget on better rate-distortion decisions
                  ↓
4. Reduce the resulting representation overhead
```

That gives the following priority:

```text
oracle + reference encoder
        ↓
exact integer search machinery
        ↓
modern .mars representation
        ↓
NEON
        ↓
multicore
        ↓
hierarchical search funnel
        ↓
rate-distortion optimisation
        ↓
mode competition + residuals
        ↓
adaptive partitioning
        ↓
entropy/reference coding improvements
        ↓
optional learned pruning
```

The reason for this ordering is important:

- The oracle tells us whether a fast search is losing useful candidates.
- Exact integer arithmetic makes CPU/GPU comparison trustworthy.
- SIMD and threads make the remaining search cheap enough to experiment with.
- RD optimisation determines whether extra search actually buys compression.
- Residuals and alternative modes improve representation efficiency after the basic fractal path is working.
- Entropy coding removes bits after the semantic representation is stable.
- Learned pruning is only worthwhile after the classical funnel gives us a strong measured baseline.

---

# 3. Existing research that is considered DONE

Do **not** open new research tasks for the following unless new measurements contradict the existing findings.

## 3.1 Measurement methodology

Already settled:

- one metrics implementation;
- BD-rate with its integration interval;
- reproducible timing protocol;
- `evals/transform` as the machine-independent search-cost metric;
- oracle cache;
- provenance on every result;
- CI regression gates.

These are part of the project contract, not optimisation work.

## 3.2 Target platform

Already settled:

- Apple Silicon / aarch64;
- NEON is the baseline CPU SIMD path;
- Metal is primarily a research/oracle instrument initially;
- do not add portable SIMD abstraction until portability is a real requirement.

## 3.3 Exact integer moments

Already settled:

- domain pixels use the fixed-point `D = 4d` representation;
- moments `s1`, `s2`, `t0`, `t1`, `t2` are accumulated exactly in integers;
- only final affine fitting needs floating point.

Do not redesign this unless a differential test proves the current formulas wrong.

## 3.4 Search funnel

The intended architecture is already known:

```text
large domain pool
    ↓
cheap statistics
    ↓
structural signature
    ↓
thumbnail comparison
    ↓
exact affine fitting
```

The open implementation problem is not *whether* to use a funnel. It is to implement it and determine the best measured operating points.

## 3.5 Rate-distortion model

Already settled:

```text
J = D + λR
```

and λ is the quality-control parameter.

Do not introduce another unrelated quality knob just to get a convenient benchmark curve.

## 3.6 SIMD and multicore architecture

Already settled:

```text
NEON inside the hot kernels
Rayon/thread-level parallelism across independent work
```

The open question is implementation quality and measured scaling, not whether SIMD or threading is theoretically useful.

---

# 4. Implementation sequence

## Step O1 — Build the first useful Mars 2 encoder

### Goal

Get a complete, correct, independently decodable Mars 2 image path working before attempting optimisation.

### Implement

Use the existing architecture to implement:

```text
image
  ↓
fixed quadtree
  ↓
domain pool
  ↓
exhaustive domain/isometry search
  ↓
affine fit
  ↓
quantise
  ↓
Mars 2 bitstream
  ↓
decoder
```

Initially keep the search intentionally simple.

### Important constraint

Do not combine:

- SIMD,
- multithreading,
- adaptive partitioning,
- residuals,
- learned pruning

into the first implementation.

The first encoder must be the trusted scalar reference.

### Exit

`just gate-6` / the repository's current Step-6 encoder gate must pass.

Required properties:

- decoder round-trip;
- integer moments exact against the floating-point reference;
- RD quality close to the measured baseline;
- `evals` reported.

### Result

This becomes the **scalar Mars 2 reference encoder**.

### Size

L

---

# 5. Build the oracle before making the search clever

## Step O2 — Complete the exhaustive oracle and cache

### Goal

Make exhaustive search a reusable measurement instrument.

### Implement

For each relevant configuration cache:

```text
best candidate
top-32 candidates
qalfa
qbeta
rms
```

per range block.

### Also record

```text
candidate rank
domain coordinates
isometry
fit parameters
```

### Why now

The oracle is needed for almost every subsequent optimisation:

```text
fast search
    vs
oracle
```

gives:

- top-1 recall;
- top-k recall;
- RMS regret;
- exact evaluation reduction.

Do not substitute subjective image quality for this comparison.

### Exit

`just gate-7` and `just gate-8` / the repository's corresponding oracle gates pass.

### Size

L

---

# 6. Establish the classical search baseline

## Step O3 — Port the measured classical search methods

### Goal

Create a hard baseline for search efficiency before designing a new search.

### Implement

Port the six planned classical methods plus exhaustive search.

Retain the KD-tree where already specified.

### Measure

For each method:

```text
BD-rate
evals/transform
top-1 recall
top-k recall
RMS regret
encode time
```

### Critical output

Produce the first:

```text
recall ↔ search-cost ↔ BD-rate
```

frontier.

### Why this matters

The new hierarchical funnel must beat something concrete.

### Exit

Repository Gate B passes.

### Size

L

---

# 7. Replace brute-force candidate evaluation with the funnel

## Step O4 — Implement the hierarchical candidate funnel

### Goal

This is the first major encoder optimisation.

### Initial structure

```text
Stage 0
all candidate domains

Stage 1
cheap statistics
    ↓
~1,000

Stage 2
structural signature
    ↓
~100

Stage 3
4×4 / 8×8 thumbnail
    ↓
~16–32

Stage 4
exact affine fit
    ↓
best candidate
```

These counts are starting points only.

### Stage 1

Use:

```text
mean
variance
min
max
range
gradient / edge energy
```

Reject only with a safe lower bound initially.

### Stage 2

Use:

```text
quadrant means
quadrant variances
low-frequency structure
gradient orientation
```

### Stage 3

Use small thumbnail distance.

### Stage 4

Run the exact existing affine fit.

### Instrument every stage

For every range block measure:

```text
input candidates
survivors
oracle recall
exact evals
```

### Acceptance target

The important result is not "16 candidates".

It is:

```text
maximum evaluation reduction
while preserving the useful oracle frontier
```

The repository's existing target is a very high-recall funnel; do not trade away recall just to produce a pretty candidate-count number.

### Exit

`just gate-13` / the repository's Step-13 gate passes.

### Size

L

---

# 8. Make exact search fast

## Step O5 — Implement NEON kernels

### Goal

Accelerate the work that remains after candidate pruning.

### Kernel order

Follow the already-measured priority:

1. moment accumulation:
   ```text
   s1 s2 t0 t1 t2
   ```
2. SAD / SSD
3. 2:1 downsampling
4. gradient calculations
5. DCT

### Specific investigation

Measure the `UDOT` implementation for:

```text
t1 = Σ r·D
```

using the already identified `u8 × u8` expansion.

Do not assume `UDOT` wins; benchmark it.

### Design

Use:

```text
scalar reference
        ↓
NEON implementation
        ↓
exact differential test
```

### Constraint

SIMD must not alter the bitstream.

### Exit

Repository `gate-11` passes:

- exact differential tests;
- microbenchmark improvement;
- end-to-end improvement;
- unchanged output.

### Size

L

---

# 9. Add multicore execution

## Step O6 — Parallelise the encoder

### Goal

Turn the fast scalar/SIMD encoder into a scalable CPU encoder.

### Architecture

Separate:

```text
search
```

from:

```text
bitstream emission
```

so range-block searches can execute independently.

Then:

```text
parallel search
        ↓
ordered result collection
        ↓
canonical serial emission
```

### Constraints

- deterministic output;
- explicit Rayon thread count for benchmark runs;
- no shared mutable search state;
- no thread-dependent candidate ordering.

### Measure

```text
1
2
4
8
...
```

threads appropriate to the reference machine.

### Exit

Repository `gate-12` passes.

### Important gate

Do not hide a poor CPU algorithm behind thread scaling.

The existing plan explicitly requires the CPU path to demonstrate its speedup before later RD work is layered onto it.

### Size

M

---

# 10. Close known post-Gate-D regressions before adding new features

## Step O7 — Fix residual quantisation

### Goal

Remove the already-measured residual regression.

The repository has already attributed the +2.05% mean BD-rate regression to the mismatch between:

```text
fixed residual quantisation
```

and:

```text
varying λ / target bitrate
```

### Implement

Make residual quantisation a per-encode function of λ.

Use either:

```text
closed-form mapping
```

or:

```text
small validated lookup
```

Do not introduce learning here.

### Verify

Run the existing Step-22 gate against the original +2.05% result.

### Abort

If the regression improves by less than half, stop tuning the qstep itself and revisit the residual representation.

### Exit

`just gate-22`

### Size

M

---

# 11. Implement real rate-distortion optimisation

## Step O8 — Replace threshold decisions with RD decisions

### Goal

Make the encoder spend bits where they buy distortion reduction.

### Implement

For each candidate representation:

```text
J = D + λR
```

Use the live rate model rather than a fixed bits-per-mode approximation.

For quadtree decisions:

```text
encode children
    ↓
compare:
    parent J
    vs
    sum(child J)
```

Choose the lower cost.

### Important

This is not just a quality-feature step.

It is the mechanism that allows all later features to compete fairly:

```text
constant
affine
fractal
fractal + residual
split
```

### Exit

Repository `gate-14` passes.

Target from the existing plan:

```text
≥ 10% BD-rate improvement
```

from RD optimisation alone.

Also require:

```text
monotone λ sweep
roughly convex RD curve
```

### Size

L

---

# 12. Add explicit mode competition

## Step O9 — Implement the modern block-mode architecture

### Modes

Start with:

```text
0 constant
1 affine
2 fractal
3 fractal + residual
4 split
```

### Decision

Every eligible mode competes under the same:

```text
J = D + λR
```

objective.

### Fast path

Use block statistics to order likely winners:

```text
flat       → constant
smooth     → affine
self-similar → fractal
poor match → residual / split
```

This is search ordering, not a correctness shortcut.

### Measure

Per image and rate point:

```text
mode share
bits/mode
distortion/mode
time/mode
```

### Scientific purpose

The project should explicitly record how often fractal prediction actually wins when competing fairly against simpler predictors.

### Exit

Repository `gate-15` passes, followed by the improved Step-22 residual gate.

### Size

L

---

# 13. Make partitioning content-adaptive

## Step O10 — Adaptive quadtree / HV splitting

### Goal

Stop spending the same partition budget everywhere.

### Implement

Use the RD machinery to choose:

```text
leaf
quad split
```

and then add:

```text
horizontal split
vertical split
```

only where the existing `hv-split-plan.md` says they are justified.

### Also vary domain density

Use local complexity to choose:

```text
sparse domain pool
medium domain pool
dense domain pool
```

The repository already has measured evidence that adaptive domain density is valuable; expose and integrate it rather than treating it as a new research idea.

### Measure

```text
BD-rate
encode time
evals/transform
leaf count
mean block size
block size vs. local variance
```

### Exit

Repository `gate-16` plus the existing adaptive-density measurement path passes.

### Size

M/L

---

# 14. Reduce bitstream overhead after the representation is stable

## Step O11 — Optimise parameter representation

### Goal

Reduce bits without changing reconstruction.

### First targets

Encode predictable parameters using deltas/predictors:

```text
domain x/y
qbeta
similar spatial parameters
```

Prefer:

```text
neighbour predictor
parent predictor
scan-order delta
```

based on measured entropy.

### Rule

Do not add a predictor just because the sequence "looks correlated".

Measure:

```text
raw bits
entropy estimate
actual coded bits
```

### Relation to progressive coding

For the existing progressive stream, the repository has already identified three specific overheads:

1. duplicated `qbeta`;
2. cold-started entropy contexts between layers;
3. independent layer-3 parameter coding.

These belong to the post-Gate-D progressive follow-up, not to the initial static encoder.

Keep them as separate, attributable changes.

### Size

M

---

# 15. Optimise entropy coding

## Step O12 — Make the bitstream statistically efficient

### Goal

Exploit the non-uniform distributions produced by the final codec.

### Symbol classes

At minimum consider:

```text
split flags
mode
domain index delta
isometry
qalfa
qbeta
residual coefficients
```

### Contexts

Start with the simplest useful contexts:

```text
depth
parent state
neighbour mode
previous domain
```

Add more only when measured.

### Crucial rule

Entropy coding must be benchmarked with distortion held constant.

That makes this a pure representation-efficiency experiment.

### Exit

The output must decode to exactly the same reconstruction as the uncoded parameter representation, while reducing total file size.

Run the format fuzz target at the same time.

### Size

L

---

# 16. Reinvest CPU savings into compression

## Step O13 — Search harder with the recovered compute budget

At this stage the encoder should be dramatically cheaper per exact candidate.

Do not simply declare victory on speed.

Use part of the recovered CPU budget to test:

```text
more final candidates
more mode alternatives
larger thumbnail survivor sets
better partition alternatives
better residual evaluation
```

The important experiment is:

```text
same approximate encode-time budget
        vs.
lower/higher search budget
```

### Output

Construct a measured Pareto frontier:

```text
fast
balanced
compression-first
```

### Exit

Retain only search increases that produce a useful BD-rate gain for their CPU cost.

### Size

M

---

# 17. Validate and then consider learned pruning

## Step O14 — Learned candidate pruning

### Entry condition

Only start after:

- oracle exists;
- funnel exists;
- full-corpus recall/eval data exists;
- the classical Pareto frontier is established.

### Goal

Predict whether a candidate belongs in the exact-search top-K.

### Inputs

Use already generated features:

```text
range signature
domain signature
thumbnail distance
relative position
block size
local complexity
```

### Training labels

Use oracle results.

### Evaluation

Measure:

```text
top-K recall
exact evaluations saved
encode time
BD-rate
generalisation across corpora
```

Model inference time counts against the encode budget.

### Exit

The existing target from `implementation-plan.md` is:

```text
≥ 90% reduction in exact affine evaluations
vs. best classical method
at equal BD-rate within ±0.5%
```

with cross-corpus generalisation reported.

### Abort

If the classical funnel already reaches the project's target evaluation reduction at full recall, do not add ML simply for novelty.

### Size

L

---

# 18. Do not make Metal part of the product encoder yet

## Step O15 — Keep GPU work focused on the oracle

The repository's current architecture deliberately moved GPU exhaustive search early because it makes the oracle affordable on unified-memory Apple Silicon.

That is valuable.

However:

```text
GPU oracle ≠ GPU product encoder
```

The product encoder should remain CPU-first until the CPU implementation is complete and measured.

Only revisit a Metal encoder after the CPU path has a clear performance ceiling.

If a GPU encoder is eventually attempted, benchmark:

```text
kernel time
orchestration
synchronisation
end-to-end encode time
```

not kernel throughput alone.

### Size

L/XL, optional

---

# 19. Final optimisation pass: remove anything that did not pay

## Step O16 — Complexity audit

Once the encoder is competitive, inspect every optimisation layer:

```text
adaptive density
feature funnel
thumbnail search
SIMD
threads
RD
modes
residuals
adaptive partitioning
delta coding
entropy contexts
ML
```

For each retain/remove decision ask:

```text
What measurable result justified this complexity?
```

Delete machinery that:

- saves negligible CPU;
- saves negligible bits;
- complicates the format without meaningful gain;
- only wins on one image;
- duplicates another search stage;
- exists solely because the historical Mars implementation did it.

The fact that a feature came from the research plan is not by itself a reason to keep it.

---

# 20. Recommended dependency graph

The implementation dependencies should look approximately like:

```text
O1 reference encoder
 │
 ├── O2 oracle
 │    │
 │    └── O3 classical search baseline
 │
 ├── O5 NEON
 │
 └── O6 multicore
 │
 └───────────────┐
                 ▼
             O4 funnel
                 │
                 ▼
              O8 RD
                 │
          ┌──────┴──────┐
          ▼             ▼
        O9 modes      O10 partition
          │             │
          └──────┬──────┘
                 ▼
               O7 residual fix
                 │
                 ▼
               O11
       parameter/reference coding
                 │
                 ▼
               O12
        entropy optimisation
                 │
                 ▼
               O13
        search-budget reinvestment
                 │
                 ▼
               O14
        learned pruning (optional)
```

O2 and O3 must precede O4.

O5 and O6 can proceed independently once O1's scalar implementation is stable.

O8 should happen before aggressive mode/partition experimentation because those decisions need a trustworthy rate model.

O11/O12 should follow representation stabilisation rather than precede it.

---

# 21. What should be implemented first

For an AI coding agent working on the repository, the immediate queue should be:

```text
1. O1 — first scalar Mars 2 encoder
2. O2 — exhaustive oracle/cache
3. O3 — classical search baselines
4. O5 — NEON kernels
5. O6 — multicore
6. O4 — hierarchical search funnel
7. O8 — RD optimisation
8. O9 — mode competition
9. O10 — adaptive partitioning
10. O7 — λ-aware residual quantisation
11. O11 — compact references/parameters
12. O12 — entropy optimisation
13. O13 — spend saved CPU on compression
14. O14 — learned pruning, only if justified
15. O15 — Metal product encoder, only if justified
16. O16 — final complexity audit
```

There is one deliberate exception:

> **If the current branch already contains the post-Gate-D residual/progressive implementation, execute the measured Step-22/23 fixes immediately rather than waiting for the new encoder sequence.**

Those fixes are already-root-caused work, not new experiments.

---

# 22. Gates and evidence

Every implementation stage should leave behind three things:

## Code

The smallest production-quality implementation needed for the step.

## Test

An automated gate proving it works.

## Evidence

A committed result row and a short prediction/decision record.

Never merge an optimisation with only:

```text
"It looks faster."
```

or:

```text
"The images look better."
```

Use the repository's existing measurement contract.

---

# 23. Definition of success

Mars 2 should ultimately demonstrate a **Pareto improvement**, not one magic benchmark number.

The desired result is:

```text
                 Smaller files
                      ↑
                      │
             compression ●
                      │
                ● balanced
                      │
          ● fast      │
                      └──────────────────→
                         faster encode
```

The project should be able to explain:

1. how many expensive candidate evaluations remain;
2. how much search was removed by the funnel;
3. how much speed SIMD contributes;
4. how much speed multicore contributes;
5. how much BD-rate RD optimisation contributes;
6. which modes actually win;
7. how much residual coding contributes;
8. how much partition adaptation contributes;
9. how many bits are removed by parameter prediction;
10. how many bits are removed by entropy coding;
11. whether learned pruning adds anything beyond the classical funnel.

---

# 24. What not to spend time on

Do not spend implementation time on:

- re-proving the measurement methodology;
- redesigning the fixed-point moment representation without evidence of a bug;
- preserving Mars 1 bitstream compatibility;
- adding portable SIMD dispatch for non-aarch64 targets;
- building a second search framework alongside the existing oracle/funnel design;
- adding ML before the classical funnel is measured;
- moving the encoder to Metal merely because the GPU is available;
- tuning residuals further while λ adaptation is still missing;
- adding more entropy contexts before measuring symbol costs;
- preserving historical Mars behaviour when a better Mars 2 representation is supported by measurements.

---

# 25. The architectural end state

The intended encoder is:

```text
                  source image
                       │
                       ▼
              content statistics
                       │
                       ▼
                adaptive tree
                       │
                       ▼
              candidate domain pool
                       │
                       ▼
              cheap feature funnel
                       │
                       ▼
              thumbnail filtering
                       │
                       ▼
             NEON exact affine fit
                       │
                       ▼
               candidate modes
          ┌────────────┼────────────┐
          │            │            │
       constant      affine       fractal
          │            │            │
          └────────────┼────────────┘
                       │
                + residual
                       │
                       ▼
                RD optimisation
                  D + λR
                       │
                       ▼
             compact parameters
                       │
                       ▼
                 context models
                       │
                       ▼
                     rANS
                       │
                       ▼
                   `.mars`
```

The important design principle is:

> **Fractal prediction is a strength of Mars 2, not a prison.**

The encoder should use fractal prediction when it is the best rate-distortion choice and use simpler or more conventional coding when those representations are better.

That is the point of breaking Mars 1 compatibility.

---

# 26. Relationship to the existing plans

This document is an **encoder-focused execution roadmap**, not a replacement for the project's existing implementation plan.

Use:

- `implementation-plan.md` for the authoritative numbered steps and gates;
- `technical-research-and-development-plan.md` for the research rationale;
- `improvement-plan.md` for already-measured post-Gate-D regressions;
- this document for the question:

> **"What should the encoder implementation work on next to maximise our chance of ending up with a fast, small Mars 2 codec?"**

When this document and a measured gate disagree, the measured gate wins and the discrepancy should be recorded in `docs/decisions.md`.
