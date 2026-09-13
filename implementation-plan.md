# Mars 2 — Step-by-Step Implementation Plan

Companion to [technical-research-and-development-plan.md](technical-research-and-development-plan.md).
That document is the *why* and *what*. This one is the *how*, *in what order*, and
*how we know it worked*.

**Version:** 2.0 · **Status:** accepted · **Last updated:** 2026-09-13

### Decisions of record

| Question | Answer | Consequence |
|---|---|---|
| Who executes? | **AI agents, no human team** | Verification, not typing, is the bottleneck — see §2.1 |
| Target hardware? | **Apple M3 (aarch64) + its GPU** | One SIMD target (NEON), Metal, unified memory — see §2.2 |
| Mars 1 bit-exactness? | **Not required** — reference only | Group B shrinks ~40%; a whole class of FP hazards deleted |
| File extension? | **`.mars`** | The 1998 codec wrote `.ifs`, so `.mars` is free |
| Publication? | **GitHub only** | Corpus rigour relaxed; the write-up step collapses into the README |

---

## 0. The one structural change from the R&D plan

The R&D plan puts benchmarking at Phase 13 ("Final research: Benchmark + papers").
That is backwards and it is the single most important thing to fix.

**Measurement is Step 1. The first subject of measurement is Mars 1, not Mars 2.**

Concretely: before a line of Rust codec code is written, we build a harness, point it
at the 1998 C binaries, and produce the project's first rate-distortion curves and
search-cost numbers. Every subsequent claim is then a *delta against a recorded
baseline* rather than an assertion.

This costs roughly 3–5 days up front and pays for itself the first time someone asks
"is the SIMD version actually faster, or did the laptop just boost differently?"

---

## 1. The measurement contract

These rules are binding for the life of the project. They exist because most
compression "results" are irreproducible, and the ones that are reproducible usually
became so by accident.

### M1 — One metrics implementation, never self-reported

All quality metrics are computed by `mars-bench` from two files on disk: the original
image and the decoded image. No codec ever reports its own PSNR. This is non-negotiable
— it is the single most common source of bogus comparisons in the compression
literature, because every codec rounds, crops, and colour-converts slightly differently.

### M2 — Metric definitions are pinned

Ambiguity here silently invalidates everything downstream.

| Metric | Definition |
|---|---|
| MSE | Mean over the **original W×H region only**. Mars 1 pads to `virtual_size` (next power of two); the padding is never measured. |
| PSNR | `10 · log10(255² / MSE)`, 8-bit. Grayscale → Y only. Colour → per-channel, plus PSNR-Y, plus PSNR-YUV = `(6·Y + Cb + Cr)/8`. |
| SSIM | 11×11 Gaussian window, σ=1.5, K1=0.01, K2=0.03, computed on Y, mean of the map. Implementation pinned and version-recorded. |
| MS-SSIM | 5 scales, Wang weights `[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`. Requires min dimension ≥ 176 — smaller images are **flagged and excluded**, never silently rescaled. |
| bpp | `8 · total_file_size_bytes / (W · H)`. Whole file, header included. "Payload-only" bitrates are forbidden. |

### M3 — RD comparisons are BD-rate, never single points

A single `(bpp, PSNR)` pair is **not** a valid comparison between two codecs. Any RD
claim requires:

- ≥ 4 quality points per curve,
- an overlapping bpp range between the two curves,
- piecewise-cubic interpolation over (log bpp, PSNR),
- reported BD-rate % **and** the bpp interval the number was computed over.

A BD-rate quoted without its overlap interval is meaningless and will be rejected in
review. The harness computes this; nobody computes it by hand.

### M4 — Timing protocol

Timing is the easiest number to get wrong. Rules:

- Median of N ≥ 5 runs; report **median + median absolute deviation**, never a single run.
- **A/B interleaved** execution (`A B A B A B…`), not `AAAAA BBBBB`, so thermal drift
  cancels instead of accumulating into the second group.
- Record the machine fingerprint with every row: exact chip variant, core counts,
  compiler + version + flags, OS build, and whether the machine was otherwise idle.
- Encode and decode reported separately, plus normalised `s/megapixel`.
- Default to warm file cache; state it.
- **A speedup figure without absolute times and a machine fingerprint is not a result.**

**Apple Silicon specifics — these will silently corrupt results if ignored.**

- **Record the chip variant, not just "M3".** M3 / M3 Pro / M3 Max span roughly 8→16 CPU
  cores and 10→40 GPU cores. A number without the variant is not comparable to the next one.
- **P-cores vs E-cores.** macOS schedules by QoS class. A benchmark that lands on
  efficiency cores reads 2–3× slow for reasons unrelated to the code. Run timing at
  user-initiated QoS or higher, never under `nice`, and **never from a background agent
  session** — timing runs are foreground work.
- **Set Rayon's thread count explicitly to the P-core count** for compute benchmarks.
  Defaulting to `num_cpus` includes E-cores and bends the scaling curve for scheduling
  reasons rather than algorithmic ones — exactly the kind of plausible wrong answer §2.1
  exists to catch.
- **Thermal throttling is real on laptops.** A/B interleaving (above) is the mitigation;
  report MAD, and treat MAD above ~5% as a failed measurement rather than noise to average.
- **Gone with x86:** no x87 excess precision, no AVX-vs-SSE divergence, no runtime feature
  detection on the CPU path. aarch64 gives IEEE-754 binary64 scalar plus NEON
  unconditionally. This deletes an entire hazard class the previous revision spent a step
  defending against.
- **Metal has no fp64**, and 64-bit integer support in MSL is limited. Both constrain
  kernel design — see the integer-exact moments decision in Step 6.

### M5 — Search work is the primary efficiency metric

Mars 1 already counts this (`comparisons` in `coding_func.c`, incremented at six sites
— one per speed-up method). Mars 2 must increment at the *exactly analogous* point or
the numbers are not comparable.

**Definition:** `evals` = the number of `(domain, isometry)` pairs for which a full
affine fit + RMS evaluation was performed. Report:

```
evals                  — total
transforms             — leaf blocks emitted
evals / transform      — the headline number
```

This metric is **machine-independent, deterministic, and non-gameable**, which makes it
the most trustworthy number the project will produce. Wall-clock speedups will be
argued about; `evals/transform` will not.

### M6 — The oracle (built early, used everywhere)

For each `(image, min_size, max_size, SHIFT, N_BITALFA, N_BITBETA, MAX_ALFA)` config,
run a full exhaustive search **once** and persist to a cache:

- the true best `(domain, isometry, qalfa, qbeta, rms)` per range block,
- the **top-32** candidate list per range block.

This is expensive (hours per config on a 512² image) but done once and cached. It buys
three things that are otherwise separate projects:

1. **The RD upper bound** for fractal-only mode — the ceiling any fast search is chasing.
2. **Top-1 / top-k recall** for every search method: what fraction of blocks does the
   fast method find the exhaustive optimum for? This turns "is the search good?" into a
   measured number rather than an inference from PSNR.
3. **Free training labels** for the learned-pruning experiment (R&D plan §20–21). The
   corpus-generation step disappears — it is a byproduct of Step 7.

### M7 — Provenance on every row

Every result row carries: git SHA + dirty flag, build profile, corpus manifest hash,
parameter-set hash, ISO-8601 timestamp, run index, harness version. Results are
**append-only JSONL** under `results/`. Nothing is ever overwritten or edited. A result
you cannot regenerate from its own row is not a result.

### M8 — CI regression gate

Runs on every PR against the `quick` corpus, budget < 2 minutes. Fails the build if:

- any golden bitstream fixture changes without an explicit `FORMAT-CHANGE:` trailer in
  the commit message,
- PSNR drops > 0.05 dB at matched settings,
- encode time regresses > 15% vs. the rolling median of the last 10 `main` runs.

The third check is deliberately loose — CI machines are noisy. It catches 2× regressions,
not 5% ones. Real performance work is measured on a pinned box, not in CI.

### M9 — Corpus

| Set | Contents | Purpose |
|---|---|---|
| `fixtures/` | `lena.raw` + 4 synthetic (flat, linear gradient, checkerboard, white noise) | Mars 1 bit-exactness and edge cases |
| `quick/` | 4 images ≤ 256² | CI gate |
| `standard/` | Kodak, 24 images, 768×512 | **All headline RD numbers** |
| `extended/` | CLIC professional validation subset; USC-SIPI textures | Large photographic; self-similarity stress |

**On Lena:** with publication off the table the venue-policy objection no longer applies,
so use it freely as a fixture and smoke-test image. Kodak still carries the headline
numbers, for the statistical reason rather than the political one: 24 images beats 1, and
single-image RD curves are how projects talk themselves into conclusions that do not
survive contact with a second image.

**On texture images:** USC-SIPI textures matter more here than for a normal codec —
fractal coding's entire premise is block self-similarity, so a corpus of only
photographs will systematically understate and misattribute where the method works.

### M10 — Anchor codecs are context, not targets

`mozjpeg`, OpenJPEG, `cwebp`, `libavif`/`aom`, `libjxl`. Pinned versions, exact CLI
invocations recorded in the manifest, re-run whenever versions change.

**Be honest about what these are for.** Fractal coding is not going to beat AVIF or
JPEG XL on rate-distortion. It was not competitive with JPEG 2000 in 2001 and nothing
in this plan changes that; the transform-coding + learned-entropy line has had twenty-five
more years of investment. Including AVIF and JXL in the table without saying this invites
the reader — and worse, the project itself — to adopt a success criterion it cannot meet,
and then to read a predictable loss as failure.

The anchors answer "where does this sit?" They do not define success. Success is defined
in §7 below, and it is about **search complexity and the structure of the representation**,
which is where the genuinely open questions are.

---

## 2. Working agreement

### 2.1 — Because the executor is AI agents

This inverts the usual risk profile and is the most important adjustment in this revision.

For a human team the binding constraint is time and the characteristic failure is *not
finishing*. For agents, typing is nearly free and the binding constraint is
**verification**. The characteristic failure is **silent plausible wrongness**: a PSNR
number that looks entirely reasonable, is wrong by 0.4 dB, and is inherited by every
result downstream. Nobody notices, because agents start each session cold and never
accumulate the practitioner's intuition that would flag the anomaly.

The measurement contract in §1 is therefore not merely good practice here. **It is the
primary defence mechanism**, and it is worth considerably more than it would be to a
human team.

**A1 — Every exit criterion is a command that exits 0 or 1.**
`just gate-a` either passes or it does not. "The RD curve looks reasonable" is not an exit
criterion. If a criterion cannot be expressed as a check, either make it checkable or
delete it.

**A2 — Tests are the contract; agents may not weaken them.**
The single most important guard in this document. The natural failure mode of an agent
that cannot make code pass is to adjust the assertion. Mitigations: golden files
committed; a CI check that fails on any diff deleting or loosening an assertion without a
`CONTRACT-CHANGE:` trailer explaining why; tolerances declared once in a central
constants module and never inline, so widening one is a visible diff rather than a
character change buried in a test.

**A3 — Differential testing carries the load that code review would.**
Every optimisation ships alongside the reference implementation it replaces, plus a
randomised differential test: scalar vs NEON, CPU vs GPU, f64 vs integer-exact. Prefer
properties that hold regardless of implementation — round-trip identity, RD monotonicity,
λ-sweep convexity, `evals` counts invariant under thread count.

**A4 — Predict before measuring.**
Before any sweep, the step appends its expected outcome to `docs/predictions.md`
(append-only). An agent that measures first and explains afterwards will rationalise
almost any result; the prediction is what makes a surprise legible as a surprise. Costs a
paragraph, catches harness bugs nothing else will.

**A5 — Each step is a self-contained brief.**
Agents start cold. A step must be executable from its section of this document alone:
goal, inputs, deliverable, exit command, files involved. Where a step needs context from
elsewhere, inline it. This document is the brief.

**A6 — One step per worktree.**
Parallel agents work in separate git worktrees on separate branches. §5's dependency
graph states what may run concurrently.

**A7 — Anomalies halt; they are never averaged away.**
If a cross-validation disagrees, the step fails. Widening a tolerance requires a recorded
reason in `docs/decisions.md`. "Close enough" is how a 0.4 dB error becomes permanent.

**A8 — Humans look at gates, not diffs.**
Gates A–D are where a person actually reads output. Between gates the harness is the
reviewer — which is only safe if A1 and A2 hold, which is why they come first.

### 2.2 — Because the target is one machine

Targeting a single architecture is a real simplification; take the win.

- **No runtime SIMD dispatch.** NEON is unconditional on aarch64. Use
  `std::arch::aarch64` intrinsics or `core::simd` directly. `pulp` (R&D plan §7) was the
  right call for a portable x86/ARM codec and is unnecessary overhead for this one.
  Revisit only if portability becomes a goal.
- **Do not assume SVE or SME.** Availability varies across M-series generations. Detect
  and fall back; do not hard-code beyond baseline NEON + DOTPROD without checking the
  actual chip.
- **Unified memory changes the GPU calculus entirely.** There is no host↔device transfer
  penalty to amortise — the main reason the R&D plan deferred GPU to Phase 9. On this
  target the GPU becomes useful *early*; see Step 7.
- **Metal has no fp64.** Step 6 resolves this by making the numerics exact in integers
  rather than negotiating a float tolerance across two backends.

### 2.3 — Code rules

- **Deterministic by default.** No wall-clock seeds, no unordered iteration affecting
  output, no floating-point reassociation. Nondeterminism is opt-in and recorded.
- **Every experiment is a config file**, not a code edit.
- **Traits are introduced when there are two implementations**, not in anticipation of one.
- **No `unsafe` outside `mars-simd` and `mars-gpu`**, and there only behind a safe wrapper
  with a scalar reference and a differential test.

## 3. Repository layout (start small)

The R&D plan proposes 13 crates on day one. That is premature: `mars-gpu`,
`mars-residual`, and `mars-entropy` would be empty for months, and workspace churn has a
real cost in build times, import noise, and merge conflicts.

**Start with four:**

```
mars/
├── Cargo.toml              # workspace
├── crates/
│   ├── mars-core/          # image types, blocks, transforms, quantisers, metrics
│   ├── mars-codec/         # partition, search, fit, encode, decode, bitstream
│   ├── mars-bench/         # THE HARNESS — built first
│   └── mars-cli/           # encmars / decmars / marsbench
├── reference/mars1/        # the 1998 C source, unmodified, + a build wrapper
├── corpus/                 # manifests; images fetched by script, not committed
├── results/                # append-only JSONL — committed
├── fixtures/               # golden .ifs + decoded .pgm — committed, small
├── docs/
│   ├── mars1-format.md     # Step 3 output
│   ├── mars-format.md     # Step 9 output
│   ├── measurement.md      # §1 of this document, expanded
│   └── claims.md           # §8 — the claims register
└── scripts/
```

Split out `mars-search`, `mars-simd`, `mars-entropy`, `mars-gpu` at the steps that
introduce them (10, 12, 9, 19 respectively) — by then the boundary is known rather than
guessed.

---

## 4. The steps

Each step states: **Goal · Deliverable · Exit criteria · Size**. Sizes are focused-days
for one competent Rust developer, and are estimates, not commitments.

Steps are grouped by gate. **Do not start a group until the previous gate passes.**

---

### Group A — Measure before building (Steps 0–4)

> Exit: we can measure everything, and we have recorded baselines, with zero Mars 2
> codec code written.

---

#### Step 0 — Repo, CI, result store

**Goal.** A workspace that builds, tests, and records, with nothing in it yet.

**Deliverable**
- Cargo workspace with the four crates from §3, each compiling with one `hello` test.
- CI: `fmt`, `clippy -D warnings`, `test`, `cargo-deny` (licenses + advisories).
- `reference/mars1/` containing the unmodified 1998 C, plus `scripts/build-mars1.sh`
  that compiles it with **pinned flags**: `-O2 -fno-fast-math -fno-unsafe-math-optimizations`,
  targeting x86-64 or aarch64 (**not** 32-bit x86 — see the x87 note in Step 5).
- `results/` with a schema doc and a JSONL appender.

**Exit criteria**
- `cargo test` green; `scripts/build-mars1.sh` produces working `encmars`/`decmars`.
- `cargo deny check licenses` passes — **and see the licence catch in §9, which should be
  resolved before dependencies accumulate, not after.**

**Size.** S · **Verification burden:** low.

---

#### Step 1 — `mars-bench` v0: the metrics engine

**Goal.** Correct, pinned, tested implementations of every quality metric.

**Deliverable**
- `mars-bench` computes MSE, PSNR, SSIM, MS-SSIM, bpp per §M2.
- PGM (P2/P5) and raw reader; PNG via `image`.
- BD-rate / BD-PSNR with the overlap interval reported (§M3).
- JSONL emitter with the full provenance block (§M7).
- `marsbench report` → a Markdown/HTML table plus RD curves.

**Exit criteria — this step lives or dies on its tests**
- **Known-answer vectors**: identical images → PSNR = ∞ (emitted as `null`, not a large
  float); a pair with hand-computed MSE matches to 1e-9; SSIM of an image with itself = 1.0.
- **Cross-validation**: PSNR and SSIM agree with an independent implementation
  (e.g. `ffmpeg -lavfi ssim/psnr`) to < 0.01 dB / < 0.001 on 10 Kodak images. Any
  disagreement is investigated, not averaged away — it usually means a colour-conversion
  or window-normalisation difference that would have silently biased every later result.
- **BD-rate sanity**: a curve against itself → 0.0%; a curve shifted by a known factor
  → the analytically expected value.

**Size.** M · **Verification burden:** **high** — do not let an agent rush this step; everything downstream inherits its errors, and they will be invisible.

---

#### Step 2 — Baseline capture: Mars 1 under the harness

**Goal.** The project's first real data. On Mars 1, before Mars 2 exists.

**Deliverable**
- A subprocess driver wrapping `encmars` / `decmars`, parsing their stdout for
  `transforms`, `comparisons`, `zero_alfa_transform`, and bytes written.
- **A pinned decode configuration.** `decmars` defaults to **pyramidal** decoding
  (`piramidal INIT(=1)`) with `iterations = 10`, *not* iterative. Golden fixtures and
  all reported numbers must state which decode mode was used; comparing a pyramidal
  decode against an iterative one is a silent 0.2–1 dB error.
- A parameter sweep over the grid below, across `fixtures/` + `standard/`.
- `results/baseline-mars1.jsonl` and a committed report.

**The sweep grid**

| Parameter | Flag | Values |
|---|---|---|
| Speed-up method | `-F -X -C -S -Z -Y` | all six: Fisher, Hurtgen, MassCenter, Saupe, Saupe-Fisher, Mc-Saupe |
| RMS threshold | `-r` | 2, 4, 8, 16, 32 (→ the RD curve) |
| Min / max range | `-m` / `-M` | (4,16) default; also (2,16), (4,32) |
| Domain step | `-d` | 4 (default), 8 |
| Scale bits | `-A` | 4 (default), 5 |
| Offset bits | `-B` | 7 (default), 6 |
| Decode | `-i` vs pyramidal | both, recorded |

**Exit criteria**
- RD curves for all six methods on Kodak, with BD-rate between them.
- A table of `evals/transform` per method — the first row of the project's headline metric.
- The whole sweep reproducible from one command.

**Expect surprises here, and record them.** Some 1998 defaults are near-inert: with
`T_ENT = 8.0`, the entropy pre-split can never fire (an 8-bit block's entropy is bounded
by 8.0, and by 4.0 for a 4×4 block), and with `T_VAR = 1e6` the variance pre-split cannot
fire either (8-bit variance is bounded by ~16256). **At default settings the Mars 1
partition is purely RMS-driven.** Knowing this before porting saves a week of confusion.

**Size.** M + unattended compute · **Verification burden:** medium.

---

#### Step 3 — Mars 1 bitstream specification + golden fixtures

**Goal.** A written format spec good enough that someone could implement a decoder
without reading the C. Most of the work is already done below.

**The format, as read from `mars_enc.c`, `coding_func.c`, `image_io.c`, `mars_dec.c`:**

Bit packing is LSB-first within each value, MSB-first into bytes (`pack()` in
`image_io.c:180` shifts `sum` left and tests `value & 1`). One continuous bitstream; the
final byte is left-padded by `pack(-1, …)`.

```
Header (60 bits, in order):
  N_BITALFA      : 4    bits   (default 4)
  N_BITBETA      : 4    bits   (default 7)
  min_size       : 7    bits   (default 4)
  max_size       : 7    bits   (default 16)
  SHIFT          : 6    bits   (domain step, default 4)
  image_width    : 12   bits
  image_height   : 12   bits
  int_max_alfa   : 8    bits   (MAX_ALFA encoded as round(MAX_ALFA/8 · 256))

Derived by BOTH sides (must match exactly):
  virtual_size          = 1 << ceil(log2(max(width, height)))
  bits_per_coordinate_w = ceil(log2(image_width  / SHIFT))    # integer division first
  bits_per_coordinate_h = ceil(log2(image_height / SHIFT))
  MAX_ALFA              = int_max_alfa / 256 · 8.0

Tree, walked as quadtree(0, 0, virtual_size):
  if atx >= height or aty >= width:                 emit nothing, return
  if size > max_size or block crosses the edge:     emit nothing, recurse into 4 children
  else:
      if size > min_size:  1 bit  split_flag
      if split_flag:       recurse into 4 children
      else:
          qalfa : N_BITALFA bits
          qbeta : N_BITBETA bits
          if qalfa != zeroalfa (== 0):
              isom  : 3 bits
              dom_x : bits_per_coordinate_h bits   (= domx / SHIFT)
              dom_y : bits_per_coordinate_w bits   (= domy / SHIFT)
```

Note the asymmetry: `dom_x` is packed with `bits_per_coordinate_h` and `dom_y` with
`bits_per_coordinate_w`. That looks like a naming slip in the original, but encoder and
decoder agree, so it is normative. Preserve it exactly; do not "fix" it.

**Quantisation (normative, from `coding_func.c`):**

```
alfa   = clamp((s0·t1 − s1·t0) / det, ≥ 0)
qalfa  = clamp(int(0.5 + alfa/MAX_ALFA · 2^N_BITALFA), 0, 2^N_BITALFA − 1)
alfa'  = qalfa / 2^N_BITALFA · MAX_ALFA

beta   = (t0 − alfa'·s1) / s0
if alfa' > 0:  beta += alfa' · 255
qbeta  = clamp(int(0.5 + beta / ((1+|alfa'|)·255) · (2^N_BITBETA − 1)), 0, 2^N_BITBETA − 1)
beta'  = qbeta / (2^N_BITBETA − 1) · (1+|alfa'|) · 255
if alfa' > 0:  beta' −= alfa' · 255

rms    = sqrt((t2 − 2·alfa'·t1 − 2·beta'·t0 + alfa'²·s2 + 2·alfa'·beta'·s1 + s0·beta'²) / s0)
```

where `s0 = pixel count`, `s1/s2` are domain sum/sum-of-squares, `t0/t1/t2` are the
range sum, cross term, and sum-of-squares.

**Deliverable**
- `docs/mars1-format.md` — the above, expanded, with worked examples.
- `fixtures/` — for each of 5 fixture images × 3 methods × 3 RMS thresholds: the `.ifs`
  bitstream, both decodes (iterative and pyramidal), and a `manifest.toml` recording the
  exact command line, binary SHA-256, and compiler version.
- A hexdump-annotated walkthrough of one tiny (64×64) bitstream, byte by byte. This is
  the artefact that makes Step 5 debuggable.

**Exit criteria**
- A throwaway Python script parses every golden `.ifs` using only `docs/mars1-format.md`
  and recovers a transform list identical to the C decoder's. **If the spec cannot be
  validated independently of the C, it is not a spec.**

**Size.** S · **Verification burden:** medium — the independent validator is the check.

---

#### Step 4 — Anchor codecs

**Goal.** Context for every future RD plot.

**Deliverable**
- Drivers for mozjpeg, OpenJPEG, cwebp, avifenc, cjxl with pinned versions.
- Quality sweeps producing ≥ 6 points per codec per image over 0.1–2.0 bpp.
- `results/anchors.jsonl`, regenerated only on version bumps.

**Exit criteria**
- A Kodak RD plot with all anchors plus the six Mars 1 methods, and a BD-rate table
  against JPEG as the anchor of record.
- **Write down the expected verdict before looking** (a pre-registration, cheap and
  worth it): Mars 1 is expected to land below JPEG over most of the range. If the plot
  disagrees with that expectation, the harness is more likely wrong than the codec is
  surprising — investigate before celebrating.

**Size.** S · **Verification burden:** medium — anchor CLI flags are easy to get subtly wrong.

---

### 🚦 Gate A

---

### 🚦 Gate A

- [ ] Every metric in §M2 implemented and cross-validated.
- [ ] BD-rate machinery tested against known cases.
- [ ] Mars 1 baselines recorded for 6 methods × 5 rates × 24 images.
- [ ] `evals/transform` recorded per method.
- [ ] Golden fixtures frozen, format spec independently validated.
- [ ] Anchor curves recorded.

**Nothing of Mars 2 exists yet, and that is the point.** Sizes: S+M+M+S+S — see §5.

---

### Group B — Reference, oracle, and the GPU that makes it affordable (Steps 5–9)

> Exit: we can compute the *true optimum* for any block cheaply, we know the recall and
> regret frontier of every classical method, and the baseline is recorded.

**Three changes from revision 1, all following from "Mars 1 bit-exactness not required":**

1. **No Mars 1 encoder in Rust, and no bit-exact decoder.** The C binaries *are* the
   baseline; they run under the harness at Step 2 and that is sufficient. What Rust
   actually needs is a read-only `.ifs` parser to pull transform lists out for analysis.
   This deletes the entire libm / `ilog2` / x87 hazard investigation — which the M3 target
   had already halved — along with the pyramidal-decoder port.
2. **GPU moves from Step 19 to Step 7.** The justification is no longer "GPU is fast".
3. **Numerics become integer-exact**, which is what makes CPU/GPU agreement possible at
   all given Metal's lack of fp64.

Decoder-first still holds, but for a weaker reason than before: the parser validates the
format spec against artefacts that already exist, so a format misunderstanding surfaces
before any encoder work depends on it.

---

#### Step 5 — `.ifs` reader + baseline bridge

**Goal.** Read 1998 bitstreams into Rust structures. Nothing more.

**Deliverable**
- `.ifs` parser per the Step 3 spec → transform list + partition tree.
- A minimal iterative decoder (~80 lines) for sanity checking only.
- **No** pyramidal decoder, **no** zoom, **no** writer. Shell out to `decmars` for those.

**Exit — `just gate-5`**
- Every golden `.ifs` parses, and the recovered transform count equals the C encoder's
  reported `transforms` **exactly**. This integer equality is the real check: it validates
  the format spec without requiring any floating-point agreement.
- Rust iterative decode within 0.1 dB of `decmars -i` on all fixtures. A larger gap means
  a spec error — investigate, do not widen (A7).

**Size.** S · **Verification burden:** low.

---

#### Step 6 — Rust encoder, exhaustive, with integer-exact moments

**Goal.** A correct exhaustive encoder whose numerics are chosen for the M3 + GPU target
rather than inherited from 1998.

**The numerics decision — this is the load-bearing part of the step.**

Mars 1 accumulates six block moments in `double`:

```
s0 = N      s1 = Σd     s2 = Σd²
t0 = Σr     t1 = Σr·d   t2 = Σr²
```

Range pixels `r` are `u8`. Domain pixels `d` are 2:1 contractions — the mean of four
`u8` — so `D = 4d` is always an integer in `0..=1020`. Working in that fixed-point domain,
**every moment becomes an exact integer**:

| Moment | Exact form | Max (32×32 block) | Type |
|---|---|---|---|
| `s1` | `(Σ D) / 4` | ~1.04 M | i32 |
| `s2` | `(Σ D²) / 16` | ~1.07 G | i64 |
| `t0` | `Σ r` | ~261 k | i32 |
| `t1` | `(Σ r·D) / 4` | ~267 M | i64 |
| `t2` | `Σ r²` | ~66.6 M | i64 |

Three consequences, which together make this a design decision rather than a
micro-optimisation:

1. **CPU and GPU can agree exactly.** Metal has no fp64. A float pipeline would force
   either an f32 CPU path or a permanent tolerance between backends — and f32 is not
   merely imprecise here, it is *wrong*: `Σ r²` reaches 66.6 M, well past f32's 16.7 M
   exact-integer limit, so naive f32 accumulation silently loses low bits on larger
   blocks. Integer accumulation removes the choice entirely.
2. **It is faster**, via integer MACs and NEON widening ops. Structuring `t1` to expose
   `u8 × u8` operands would additionally make it a candidate for `UDOT` (four MACs per
   lane per instruction) — flagged for Step 11, not assumed here.
3. **It is more accurate than the original**, not a compromise for it.

Only the final fit (`alfa`, `beta`, `rms`) needs floating point, and there f32 vs f64
matters only insofar as it changes a 4-bit `qalfa` or 7-bit `qbeta`. That is an empirical
question, so it gets measured rather than argued.

**Deliverable.** Partition; domain pool; 8 isometries; integer-exact moments; affine fit
and quantisation per the Step 3 formulas; an `.ifs` writer used only for cross-checking
against `decmars`; `evals` counter per §M5.

**Exit — `just gate-6`**
- Rust-encode → `decmars` decodes within 0.1 dB of Rust's own decode of the same stream.
- RD curve within **0.2 dB** of the Step 2 baseline at exhaustive-equivalent settings.
  (0.2, not 0.05: bit-exactness is not a goal, and an unreachably tight bound just invites
  tolerance-widening, which A2 exists to prevent.)
- **Integer moments proven exact** — differential test against an f64 reference over 10⁶
  random blocks asserting *equality*, not closeness.
- **f32-vs-f64 fit divergence measured and recorded**: the percentage of blocks where
  `qalfa` or `qbeta` differ. Below 0.1%, adopt f32 and note it; above, keep f64 on CPU and
  document an explicit CPU/GPU tie-break rule.

**Size.** L · **Verification burden:** high.

---

#### Step 7 — GPU exhaustive search ★ moved forward from Step 19

**Goal.** Make the oracle affordable. This is the step that unlocks the project's entire
methodology.

**Why it moved.** The R&D plan put GPU last, reasoning from discrete GPUs — where
host↔device transfer has to be amortised, and where a GPU only pays once everything else
is already fast. Neither premise holds on this target.

On unified memory there is no transfer to amortise. And the workload most in need of
acceleration is not the product encoder — it is the **oracle**: a full
`N_ranges × M_domains × 8 isometries` sweep that is embarrassingly parallel, needs no
algorithmic cleverness, is exact and therefore carries no rate-distortion risk, and is
otherwise simply too expensive to run across a corpus.

The GPU here does not make the codec faster. **It makes the measurement methodology
possible**, which is worth considerably more.

**Deliverable**
- `mars-gpu` crate. **wgpu/WGSL first** for tooling and portability; drop to direct Metal
  only if profiling demands it, and record the decision either way.
- Kernel: one workgroup per range block, integer moment accumulation in registers per
  Step 6, tiled over the domain pool.
- Top-k reduction: two-pass (compute RMS into a tiled buffer, then reduce) is simpler and
  likely sufficient. Single-pass in-register top-32 is the optimisation, not the starting
  point.
- **Verify 64-bit integer support in the target MSL version before writing the kernel.**
  If unavailable, split the `s2`/`t1`/`t2` accumulators across two 32-bit lanes. Plan for
  this rather than discovering it halfway through a shader.

**Exit — `just gate-7`**
- **Bit-identical results to the CPU exhaustive path** on every fixture — achievable
  precisely because Step 6 made the numerics integral. Any divergence is a bug, never a
  tolerance.
- ≥ 50× faster than the Rayon CPU exhaustive on the same machine, transfer included
  (nominal on UMA — but measure it rather than assuming it).
- **A full Kodak oracle build completes in hours, not weeks.** This is the real exit
  criterion; the speedup number matters only insofar as it achieves this.

**Size.** L · **Verification burden:** medium — the bit-identical check does most of the work.

---

#### Step 8 — Oracle cache + recall harness

**Goal.** Turn Step 7's cheap exhaustive search into the project's most reusable asset (§M6).

**Deliverable**
- Per range block, persisted: features, the true best match, and the top-32 candidates
  with their RMS values.
- Compact cache format, one file per image×config, with a manifest hash so a stale cache
  can never be silently reused. (Silent stale-cache reuse is a textbook agent failure
  mode — the hash is not optional.)
- `marsbench recall` — given a method's chosen domains plus the oracle, reports top-1 /
  top-5 / top-32 recall and the **RMS regret** distribution in dB.

**Exit — `just gate-8`**
- Oracle cache built for `standard/` at three configurations.
- Exhaustive self-test: 100% top-1 recall, 0 dB regret. Anything else is a harness bug.

**Size.** M · **Verification burden:** medium.

---

#### Step 9 — Classical speed-up methods

**Goal.** Fisher, Hurtgen, MassCenter, Saupe, Saupe-Fisher and Mc-Saupe in Rust, behind
one trait — the point at which `mars-search` becomes its own crate.

```rust
pub trait CandidateRetriever {
    fn index(&mut self, pool: &DomainPool);
    fn candidates(&self, range: &RangeBlock) -> CandidateIter<'_>;
}
```

Mars 1's README notes that a new method needs only two functions — an indexing function
and a coding function. This trait is that observation made type-safe, and it is the
extension point that makes the repository useful to anyone else (R&D plan §29).

**Framing note:** without a bit-exactness requirement these are no longer compatibility
targets. They are **baselines for the recall comparison** — which is what they were
always scientifically for.

**Deliverable.** Six implementations plus `Exhaustive`; a KD-tree for the Saupe family
(port the `nn_search.c` structure, and replicate its tie-breaking deliberately since the
behaviour is observable in output).

**Exit — `just gate-9`**
- Each method's RD curve within 0.2 dB of its Step 2 Mars 1 baseline.
- Each method's `evals/transform` within 5% of the C implementation. (Looser than
  revision 1's 1%, since exact search-order reproduction is no longer required.)
- **The new result: top-k recall and RMS regret for all six methods against the oracle.**
  This measurement did not exist before this project and is the first genuinely novel
  output.

**Size.** L · **Verification burden:** medium.

---

### 🚦 Gate B — `just gate-b`

- [ ] `.ifs` parser validated by exact transform-count equality.
- [ ] Exhaustive encoder within 0.2 dB of baseline; integer moments proven exact.
- [ ] GPU search bit-identical to CPU, ≥ 50× faster; Kodak oracle built.
- [ ] Six classical methods ported; recall/regret table published.

**This is a defensible stopping point.** If the project ended here it would have produced
a clean-room Rust reimplementation of a historically significant codec, a written format
specification, a GPU-backed exhaustive-search oracle, and the first quantitative
recall/regret comparison of the classical fractal speed-up methods. Everything after this
is optional and gate-driven.

---

### Group C — Modern machinery (Steps 10–16)

> Exit: Mars 2 is a modern codec — new format, entropy coding, SIMD, parallelism,
> RD optimisation, residuals, adaptive partitioning — with every gain individually
> attributed.

**Rule for this group: one change per measurement.** Landing SIMD and a new search
together makes the combined speedup unattributable, and unattributable speedups are how
projects end up carrying complexity that never actually helped.

---

#### Step 10 — `.mars` format v0 + rANS

**Goal.** Break Mars 1 compatibility deliberately and for stated reasons.

**Deliverable**
- A versioned container per R&D plan §12: magic, version, flags, dimensions, channels,
  bit depth, colour space, then sectioned payloads.
- **Colour in the format from day one**, even though the encoder initially fills only Y.
  The R&D plan is right that colour must not be retrofitted (§16); the resolution is that
  the *format* carries it at Step 10 while the *encoder* gains it at Step 18. Retrofitting
  a container is expensive; leaving a field unused is free.
- rANS via `constriction`, replacing raw `pack()`. `mars-entropy` becomes a crate.
- Adaptive context models for: split flags (context = depth + neighbour split state),
  mode, qalfa, qbeta, domain index (coded as a delta from a spatial predictor).
- **The decoder must not depend on encoder internals** — the format describes a
  representation, never "run algorithm X with settings Y" (R&D plan §12).

**Exit criteria**
- Lossless round-trip on all fixtures.
- **Attribution required:** report bpp reduction from entropy coding alone at identical
  reconstruction. The reconstruction must be byte-identical to the Step 6 output, which
  makes this a pure rate measurement with distortion held exactly constant. Expect
  8–20% from the split flags and domain deltas alone.
- A fuzz target (`cargo-fuzz`) on the decoder: no panics, no OOM, no unbounded allocation
  on arbitrary input. **Do this now, not later** — a format parser without fuzzing is a
  liability, and it is far cheaper to add before the format grows sections.

**Size.** L · **Verification burden:** high — format decisions are expensive to reverse.

---

#### Step 11 — SIMD kernels

**Goal.** Data-level parallelism as architecture, not as an afterthought (R&D plan §7).

**Deliverable**
- `mars-simd` targeting **NEON directly** — no `pulp`, no runtime dispatch, since NEON is
  unconditional on aarch64 (§2.2).
- Kernels in priority order by measured profile: the moment accumulation
  `(s1, s2, t0, t1, t2)` first, since Step 6 made it the inner loop and Step 7 proved it
  dominates; then SAD, 2:1 downsample, gradient, DCT.
- **Investigate `UDOT`** for `t1 = Σ r·D`. `D` is the sum of four `u8`, so expanding
  `Σ r·D = Σ r·a + Σ r·b + Σ r·c + Σ r·e` exposes `u8 × u8` operands that `UDOT`
  accumulates four-at-a-time into `u32`. Potentially the largest single win in the step —
  and if the expansion does not pay, record why, because the next agent will have the same
  idea.
- **SoA block batches** (R&D plan §8); AoS blocks defeat vectorisation.
- Every kernel ships with a scalar reference and a randomised differential test.

**Exit — `just gate-11`**
- Differential tests pass, asserting **exact equality** — the integer moments from Step 6
  mean there is no float reassociation to tolerate, so this is a stronger check than the
  usual SIMD epsilon comparison.
- Speedup reported **per kernel** (microbenchmark) **and** end-to-end, per §M4 including
  the Apple Silicon rules.
- **Bitstream unchanged.** SIMD is a pure performance change; any output difference is a
  bug. Hard gate — if the bitstream moves, the kernel is wrong, however good the speedup
  looks.

**Size.** L · **Verification burden:** low — exact differential tests do the work.

---

#### Step 12 — Rayon parallelism

**Goal.** Thread-level over data-level (R&D plan §9).

**Deliverable**
- Parallel domain-pool indexing and feature extraction.
- Parallel range-block coding. Note the dependency: Mars 1 writes the bitstream during
  the quadtree walk, so parallel coding requires **decoupling search from emission** —
  collect per-block results, then serialise in canonical tree order. Do this explicitly;
  it is the only real design change in this step.

**Exit criteria**
- **Bitstream identical to single-threaded, at every thread count.** Any variation is a
  determinism bug (§2), not an acceptable tolerance.
- Scaling curve at 1, 2, 4, 8, 16 threads, with parallel efficiency reported. Sub-linear
  scaling is expected and fine; report it rather than quoting only the best number.

**Size.** M · **Verification burden:** low — determinism check does the work.

---

### 🚦 Gate C — the performance gate

- [ ] Encode speedup vs. Mars 1 at matched RD, per §M4: **target ≥ 20×** on the CPU path
      (NEON × threads), measured *excluding* the GPU so that Step 13's search work has an
      honest CPU baseline to improve on.
- [ ] Decode no slower than Mars 1.
- [ ] Every gain individually attributed — entropy / NEON / threads — in one table.

**If ≥ 20× is not met, stop and diagnose before proceeding.** Layering RD optimisation on
an unexplained performance shortfall compounds the problem. Note that the GPU path from
Step 7 is available but deliberately excluded here: a GPU speedup would mask a broken CPU
kernel, and the CPU path is what Steps 13–16 build on.

---

#### Step 13 — Hierarchical funnel search

**Goal.** The R&D plan's §6 funnel: `10,000 → 1,000 → 100 → 16 → 1`.

**Deliverable**
- Stage 1 — cheap statistics (mean, variance, min, max, range, gradient/edge energies).
- Stage 2 — structural signature (quadrant means/variances, low-frequency DCT,
  gradient orientation).
- Stage 3 — thumbnail distance at 4×4 and 8×8.
- Stage 4 — exact affine fit on the survivors.
- Each stage's survivor count is configurable and **logged per block**.

**Exit criteria**
- **Recall against the oracle at each stage** — this is what Step 8 was for. Report the
  survival/recall tradeoff curve, not just the endpoint. A stage that drops top-1 recall
  below ~95% is discarding matches the RD numbers will eventually miss.
- `evals/transform` vs. all six classical methods at matched RD.
- Pareto frontier: `evals/transform` against BD-rate, with every method plotted.

**Size.** L · **Verification burden:** medium.

---

#### Step 14 — Rate-distortion optimisation

**Goal.** Replace threshold-driven splitting with `J = D + λR` (R&D plan §19).

**This step needs a caveat the R&D plan omits.** λ is not a free parameter you pick once:
it *is* the quality control. The encoder exposes λ; quality presets are λ values; and an
RD curve is a λ sweep. Getting this wrong — e.g. sweeping the RMS threshold while λ stays
fixed — produces curves that look fine and mean nothing.

**Deliverable**
- Rate estimation from the live entropy models (not a constant-bits approximation —
  that would bias every decision toward the modes with cheap headers).
- `J = D + λR` for the mode decision; extend to `J = D + λR + μT` once decode-cost
  measurement exists.
- Bottom-up quadtree pruning: code children first, then decide whether the parent's
  single transform beats the sum of the children's `J`. This is strictly better than
  Mars 1's top-down threshold test, which cannot see what the children would have cost.
- λ sweep as the quality interface.

**Exit criteria**
- BD-rate vs. the Step 9 reference at matched search effort. **Target: ≥ 10% BD-rate
  improvement** from RD optimisation alone.
- Convexity check on the λ sweep: the RD curve must be monotone and roughly convex.
  Non-convexity indicates a rate-estimation bug and is the fastest available diagnostic.

**Size.** L · **Verification burden:** **high** — λ errors produce plausible curves.

---

#### Step 15 — Residual mode

**Goal.** R&D plan §4's Mode 3 — `a · transform(domain) + b + residual` — plus the
flat and affine modes, all competing under `J`.

**Deliverable**
- Modes 0 (constant), 1 (affine), 2 (fractal), 3 (fractal + residual), 4 (split).
- Residual coding: DCT + dead-zone quantiser + context-modelled rANS. Start here rather
  than with a wavelet — it is simpler, and Step 14's machinery will report honestly
  whether residuals earn their bits.
- Mode statistics per image, per rate.

**Exit criteria**
- BD-rate vs. Step 14.
- **Mode-usage histogram across the corpus.** This is the interesting scientific output,
  not the BD-rate: it measures how often fractal prediction is actually the winning mode
  under fair competition. A low fractal share is a *finding*, not a failure — it would be
  the first honest quantification of where self-similarity genuinely pays, which is a
  question the 1990s literature never posed this way because it had no competing modes to
  compare against.

**Size.** L · **Verification burden:** medium.

---

#### Step 16 — Adaptive partitioning

**Goal.** R&D plan §5 — replace fixed quadtree geometry with content-adaptive structure.

**Deliverable**
- Split decisions driven by predicted RD gain rather than a threshold (Step 14 already
  supplies the mechanism; this step supplies the geometry).
- Non-uniform block sizes; optionally HV/binary splits alongside quad splits.
- Domain pool density varying with local complexity.

**Exit criteria**
- BD-rate vs. Step 15.
- Partition statistics: leaves per depth, mean block size vs. local variance.
- **Cost accounting:** report the encode-time increase alongside the BD-rate gain. An
  adaptive scheme that wins 3% BD-rate for 4× encode time is a different proposition
  from one that wins 3% for free, and the plan should force that comparison rather than
  quoting the gain alone.

**Size.** M · **Verification burden:** medium.

---

### 🚦 Gate D — the research gate

- [ ] BD-rate vs. Mars 1 baseline: **target ≥ 35% better** (cumulative over Steps 9–15).
- [ ] `evals/transform` reduced ≥ 10× vs. the best classical method at matched RD.
- [ ] Every step's contribution separately attributed in one table.

**Sizes: L+L+M+L+L+L+M.** Serialise every benchmark run in this group (§5).

---

### Group D — Research experiments (Steps 17–21, each independently gated)

> These are hypotheses, not commitments. Each has an entry condition, a falsifiable
> prediction, and an abort rule. Run them in the order given; skip any whose entry
> condition is unmet.

---

#### Step 17 — Learned candidate pruning ★ the headline experiment

**This is the most likely novel contribution in the project (R&D plan §21), and Steps 7–8
already paid its main cost.**

**Entry condition.** Step 13's funnel has a measured recall/eval frontier to beat.

**Hypothesis (pre-registered, stated before training).**

> A lightweight learned model, scoring `P(domain ∈ top-k | range features, domain
> features, relative position)`, reduces exact affine evaluations by ≥ 90% versus the
> best classical method at equal BD-rate (within 0.5%).

**Deliverable**
- Training corpus: **already exists** — the Step 8 oracle cache is exactly the labelled
  dataset (range features, domain features, true top-32).
- A model small enough to be worth it: gradient-boosted trees or a ≤ 3-layer MLP.
  **Inference cost must be counted in the eval budget**, or the comparison is rigged —
  a model that costs more than the evaluations it saves is not an acceleration.
- Comparison across: exhaustive, KD-tree, Fisher/Hurtgen/MassCenter/Saupe, DCT class,
  funnel, learned, funnel+learned hybrid.

**Exit criteria**
- The full comparison on `standard/` and `extended/`, reporting per §M5 and §M6:
  BD-rate, `evals/transform`, top-k recall, RMS regret, wall-clock **including inference**.
- **Generalisation test:** train on Kodak, evaluate on CLIC and USC-SIPI. A model that
  only works on its training distribution is not a result. Report the cross-corpus gap
  explicitly — it is the number a reviewer will ask for first.

**Abort rule.** If the hybrid cannot beat the funnel on the eval/BD-rate Pareto frontier
after two weeks, publish the negative result and move on. A well-measured negative here
is genuinely useful: nobody has published the recall/regret frontier for fractal search,
and "learning does not beat a good funnel" is a finding.

**Size.** XL · **Verification burden:** **high** — see A4; predict before training.

---

#### Step 18 — Colour

**Entry condition.** Gate D passed.

**Deliverable.** RGB → YCbCr; independent quality control per plane; chroma subsampling
options; eventually λ allocation across planes decided by the optimiser rather than fixed.

**Exit criteria.** BD-rate on Kodak RGB against the anchors, using PSNR-YUV and MS-SSIM.
Note that **this is the first point at which the anchor comparison is apples-to-apples** —
every earlier comparison was grayscale-vs-colour-capable and should be labelled as such
in the reports. Re-check §M10's framing here: expect to remain behind AVIF/JXL.

**Size.** M · **Verification burden:** medium.

---

#### Step 19 — Progressive decoding

**Entry condition.** Format v0 stable; Step 15's residual layering in place.

**Deliverable.** Layered bitstream — base → partition refinement → fractal refinement →
residual refinement — such that any prefix decodes to a valid image (R&D plan §13).

**Exit criteria.** RD curve *of the prefixes*, versus separately-encoded single-layer
streams at the same rates. **The honest metric is the progressive penalty:** how much
BD-rate is lost for the privilege of truncatability. Report it; it is typically 5–15%
and quoting the progressive curve without it overstates the result.

**Size.** L · **Verification burden:** medium.

---

#### Step 20 — INR fallback (experimental)

**Entry condition.** Steps 14 and 16 complete. **Framed as an experiment, never as
foundation** (R&D plan §14).

**Hypothesis.** For the worst-performing ~5% of blocks under mode competition, a small
implicit neural representation wins on `J` against fractal+residual.

**Exit criteria.** Mode-usage share for INR; BD-rate delta; and — decisively — the
**encode-time cost**, since per-image network fitting is orders of magnitude slower.
The existing literature identifies this as the central obstacle. The result to report is
the tradeoff curve, not a single operating point.

**Abort rule.** If INR wins < 1% of blocks or costs > 10× encode time for < 2% BD-rate,
document and stop.

**Size.** XL, genuinely uncertain · **Verification burden:** medium.

---

#### Step 21 — Release

With publication off the table this collapses from a paper-writing step into a
documentation one.

- **Reproducibility package:** one command regenerates every figure in the README from
  `results/`. This matters more without peer review, not less — it is the only remaining
  external check on the numbers.
- `docs/mars-format.md` complete enough for an independent decoder.
- README carrying the three results worth having: the classical-method recall/regret
  frontier (Gate B), learned pruning (Step 17), and the mode-competition study (Step 15),
  which is probably the most interesting of the three.

**Size.** M · **Verification burden:** low.

---

## 5. Sizing and the dependency graph

Day estimates were dropped in revision 2. With agents executing, focused-days is the
wrong unit — typing is not the constraint, and an estimate in days mostly encodes how
long a human would have taken to type it.

**Sizes** are rough agent-session counts: **S** ≈ one session · **M** ≈ 2–4 ·
**L** ≈ 5–10 · **XL** ≈ 10+, and expect to re-scope an XL rather than finish it as written.

**Verification burden** is the number that actually matters: how much human attention the
step needs at its gate. It is not correlated with size. Step 1 is M-sized with *high*
burden — a wrong SSIM implementation poisons every result in the project and nothing
downstream will surface it. Step 12 is M-sized with *low* burden, because a determinism
check either passes or it does not.

**Compute-bound steps** (2, 8) are wall-clock bound, not session bound; start them early
and let them run.

### Dependency graph

```
Step 0 ─┬─→ Step 1 ─┬─→ Step 2 ──┐
        │           └─→ Step 4 ──┤
        └─→ Step 3 ──────────────┤
                          Gate A ┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
                 Step 5                    Step 6
              (.ifs reader)          (encoder, int moments)
                    │                         │
                    └───────────┬─────────────┘
                                ▼
                             Step 7  (GPU exhaustive)
                                ▼
                             Step 8  (oracle cache) ──┐
                                ▼                     │
                             Step 9  (classical) ─────┤
                                               Gate B ┘
                                                      │
        ┌──────────────┬──────────────┬───────────────┤
        ▼              ▼              ▼               ▼
     Step 10        Step 11        Step 13        (Step 18
    (format)        (NEON)         (funnel)        colour,
        │              │              │            after 10)
        │              ▼              │
        │           Step 12           │
        │           (Rayon)           │
        │              │              │
        └──────┬───────┴──────────────┘
               ▼                                Gate C
            Step 14  (RD optimisation)
               ├──────────────┐
               ▼              ▼
            Step 15        Step 16
           (residual)     (adaptive)
               │              │
               └──────┬───────┘               Gate D
                      ▼
            Step 17 (learned pruning)  ← also needs Steps 8 + 13
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     Step 19       Step 20       Step 21
  (progressive)     (INR)       (release)
```

### What may run concurrently

| Step | Depends on | Concurrent with |
|---|---|---|
| 1, 3 | 0 | each other |
| 2, 4 | 1 | each other |
| 5, 6 | Gate A | each other |
| 7 | 6 | — |
| 8 | 7 | 9 (9's *exit* needs 8) |
| 10, 11, 13 | Gate B | each other |
| 12 | 11 | 10, 13 |
| 14 | 10, 13 | 18 |
| 15, 16 | 14 | each other |
| 17 | 8, 13, 15 | 18 |
| 19, 20 | 15, 17 | each other |

Five points of genuine fan-out: `{1,3}`, `{2,4}`, `{5,6}`, `{10,11,13}`, `{15,16}`. One
agent per worktree per A6.

**A caution on parallelism.** Cheap parallelism is the main temptation this execution
model creates, and the main way it goes wrong. Steps landing concurrently make gains
unattributable — which is exactly what Group C's "one change per measurement" rule exists
to prevent. Fan out on *development*; serialise on *measurement*. Two agents may build
Steps 11 and 13 simultaneously, but their benchmark runs are taken one at a time, on an
idle machine, per §M4.

---

## 6. Kill criteria

Stated in advance, because research projects without them run on inertia.

**Stop the whole project if:**
- **Step 7's GPU search cannot be made bit-identical to the CPU path.** That would mean
  the integer-exact numerics of Step 6 were wrong, and the alternative — a permanent
  tolerance between backends — would make the oracle untrustworthy, which removes the
  project's entire measurement foundation.
- After Gate C, CPU encode speedup is < 5× (an order of magnitude below target; the
  premise that modern machinery transforms this workload would be wrong).

**Stop after Gate B and write it up if:**
- Step 14's RD optimisation yields < 3% BD-rate. That would say the *representation*,
  not the optimiser, is the binding constraint, and Steps 15–16 would be pushing on the
  wrong end.

**Drop Group D entirely if:**
- Step 13's funnel already achieves > 99% eval reduction at full recall. There would be
  nothing left for learning to win.

**A kill criterion specific to agent execution:**
- **If two consecutive gates pass only after a tolerance was widened, stop and audit.**
  Under A7 each widening is individually recorded and individually defensible; the
  failure mode is the *sequence*, where no single relaxation looks unreasonable and the
  numbers have quietly stopped meaning anything. Nothing in the per-step checks can catch
  this, which is why it belongs here.

---

## 7. Success criteria — restated to be falsifiable

The R&D plan's §27 Goal B ("beat the original Mars by an absurd margin") is not testable.
Replacements, each with a defined measurement protocol:

### Goal A — Reference and oracle (Gate B)
- [ ] `.ifs` parser validated by exact transform-count equality on all fixtures.
- [ ] RD curves matching Mars 1 within 0.2 dB for all six methods.
- [ ] A format specification validated by an independent implementation.
- [ ] **GPU exhaustive search bit-identical to CPU**, with a Kodak oracle built in hours.

*(Bit-exact Mars 1 reproduction was dropped as a goal — the C binaries serve as the
baseline directly. The GPU-backed oracle replaced it as the thing Group B must deliver,
and it is worth considerably more: reproduction proves we understood 1998, whereas the
oracle is what every later measurement is graded against.)*

### Goal B — Engineering (Gate C/D)
- [ ] CPU encode ≥ 20× faster at matched BD-rate, per §M4 including the Apple Silicon
      rules, measured excluding the GPU path.
- [ ] BD-rate ≥ 35% better than Mars 1 on Kodak.
- [ ] Colour, progressive decoding, and a fuzzed format parser.

### Goal C — Science (Step 17)
- [ ] Exact affine evaluations reduced ≥ 90% vs. the best classical method at equal
      BD-rate (± 0.5%), **with model inference counted in the budget**.
- [ ] Generalisation demonstrated across corpora with the cross-corpus gap reported.
      Without peer review this matters *more*, not less — it is the check that would
      otherwise have come from a reviewer.
- [ ] The recall/regret frontier published for all methods — the measurement that did
      not exist before this project.

### Explicitly **not** a success criterion
- Beating AVIF or JPEG XL on rate-distortion. See §M10. Setting this target would
  guarantee a failed project and misrepresent what the research question is.

---

## 8. Claims register

The R&D plan cites recent papers and patents — a 2026 non-uniform partitioning paper, a
September 2026 tree-structured VQ paper, CN117241042A/B, CN120259451A, WO2025256970A1.
Some of these are very recent, and none were verified while writing this plan.

**Rule: no design decision may depend on an unverified claim.**

Maintain `docs/claims.md` with one row per external claim:

| Field | Meaning |
|---|---|
| Claim | What is asserted |
| Source | Full citation + DOI/URL + access date |
| Status | `unverified` / `verified` / `contradicted` / `withdrawn` |
| Load-bearing? | Does any design decision depend on it? |
| Our measurement | What we independently observed |

Anything marked both `unverified` and load-bearing is a project risk and must be either
verified or designed around. In particular: reported speedups in the fractal-compression
literature are frequently measured against unoptimised exhaustive-search baselines on
unstated hardware, and are not comparable to ours. **Our own baseline (Step 2) is the
only baseline our numbers are ever quoted against.**

The patent notes (R&D plan §22) stay useful as prior-art and design-space mapping, with
the existing framing preserved: not legal advice, not a freedom-to-operate opinion, and
revisited if the project ever stops being non-commercial research.

---

## 9. Licensing — one catch to resolve at Step 0

Mars 1 is GPL-2.0-or-later, and the R&D plan proposes keeping that for continuity.
That is reasonable, but there is a specific interaction worth settling *before*
dependencies accumulate:

**Apache-2.0 code cannot be combined into a GPL-2.0-*only* work** (the patent-termination
and indemnification clauses are additional restrictions under GPLv2). It *is* compatible
with GPL-3.0. Since most of the Rust ecosystem — including likely dependencies — is
dual-licensed MIT/Apache-2.0, and MIT alone is GPLv2-compatible, this is navigable, but
the consequence should be stated deliberately rather than discovered later:

- Keeping **GPL-2.0-or-later** means a recipient may choose v3, so the combination is
  lawful, but **the effective licence of the distributed binary becomes GPL-3.0**.
- Any Apache-2.0-only dependency (no MIT option) forces that outcome.

**Action at Step 0:** enable `cargo-deny` license checking, record the decision in
`docs/licensing.md`, and prefer MIT-or-Apache dual-licensed crates where a choice exists.
Also note that Mars 2 is a **clean-room Rust implementation** — the C source is a
historical reference, not something to mechanically translate — which keeps the
copyright story simple regardless.

---

## 10. First 20 issues (ready to file)

| # | Step | Issue |
|---|---|---|
| 1 | 0 | Cargo workspace + CI (fmt, clippy -D warnings, test, cargo-deny) |
| 2 | 0 | `scripts/build-mars1.sh` — pinned flags, aarch64 |
| 2b | 0 | `justfile` with `gate-N` targets that exit 0/1 (A1) |
| 2c | 0 | CI check: fail on loosened assertions without `CONTRACT-CHANGE:` (A2) |
| 2d | 0 | `docs/predictions.md` + `docs/decisions.md`, append-only (A4, A7) |
| 3 | 0 | JSONL result store + provenance block + schema doc |
| 4 | 0 | `docs/licensing.md` — resolve the GPLv2/Apache question (§9) |
| 5 | 1 | PGM (P2/P5), raw, and PNG readers |
| 6 | 1 | MSE / PSNR with known-answer tests |
| 7 | 1 | SSIM + MS-SSIM, cross-validated against ffmpeg |
| 8 | 1 | BD-rate / BD-PSNR with overlap-interval reporting |
| 9 | 1 | `marsbench report` — tables + RD plots |
| 10 | 2 | Mars 1 subprocess driver + stdout metric parser |
| 11 | 2 | Parameter-sweep runner (declarative config) |
| 12 | 2 | Capture baselines: 6 methods × 5 rates × 24 images |
| 13 | 3 | Write `docs/mars1-format.md` from the spec in §4/Step 3 |
| 14 | 3 | Freeze golden fixtures + manifests |
| 15 | 3 | Independent Python validator for the format spec |
| 16 | 4 | Anchor codec drivers, pinned versions |
| 17 | 4 | Anchor sweeps + the first combined RD plot |
| 18 | 5 | `.ifs` parser — exit criterion is exact transform-count equality |
| 19 | 5 | Minimal iterative decoder for sanity checking (~80 lines) |
| 20 | 6 | Integer-exact moment kernel + 10⁶-block equality differential test |

---

## 11. What changed from the R&D plan

| # | R&D plan | This plan | Why |
|---|---|---|---|
| 1 | Benchmarking is Phase 13 | Measurement is Step 1; Mars 1 is its first subject | The user requirement; also prevents unattributable results |
| 2 | Encoder first (§25) | **Decoder first** | 500 vs 3500 LOC; validates the format independently; enables three-way cross-validation |
| 3 | 13 crates on day one (§17) | 4 crates, split when boundaries are proven | Empty crates cost build time and import noise |
| 4 | "Beat Mars by an absurd margin" (§27) | Numeric targets with measurement protocols | Unfalsifiable criteria cannot be failed, so they cannot be passed |
| 5 | AVIF/JXL in the benchmark table (§26) | Same table, but explicitly context rather than targets | Fractal coding will lose; framing it as a target guarantees perceived failure |
| 6 | Waterfall of 13 phases (§24) | Four gated groups with kill criteria | Steps 19/20 are expensive and may become pointless — gates make that decidable |
| 7 | Oracle/training corpus implied at Phase 10 (§20) | Oracle built at **Step 7** | Turns one expensive run into RD bound + recall metric + training labels |
| 8 | λ mentioned but unspecified (§19) | λ **is** the quality control; BD-rate mandatory | Sweeping a threshold with λ fixed produces meaningless curves |
| 9 | "bit-exact or mathematically equivalent" (§25) | Exact hazard list; bit-exact at defaults, tolerance stated otherwise | libm `log` and x87 precision would otherwise be discovered at day 30 |
| 10 | Colour "from the beginning" (§16) | Format at Step 10, encoder at Step 18 | Mars 1 compatibility is grayscale by definition; the container must still carry colour |
| 11 | Lena as the working image | Lena as a fixture only; Kodak for reporting | Lena is discouraged at major imaging venues |
| 12 | Citations stated inline | Claims register with load-bearing flags | Several cited works are very recent and unverified |
| 13 | GPL-2.0-or-later, unexamined | The Apache-2.0/GPLv2 interaction resolved at Step 0 | Cheap now, expensive after dependencies land |
| 14 | — | Fuzzing the format parser at Step 10 | A format parser without fuzzing is a liability |
| 15 | — | Mode-usage histogram as a primary output (Step 15) | The most interesting question the project can answer: *when does self-similarity actually pay?* |

### Revision 2 — following the four decisions of record

| # | Revision 1 | Revision 2 | Why |
|---|---|---|---|
| 16 | GPU at Step 19, deferred and optional | **GPU at Step 7**, and load-bearing | Unified memory removes the transfer penalty the deferral assumed; and the workload that most needs it is the *oracle*, not the encoder. GPU here buys methodology, not product speed |
| 17 | Bit-exact Mars 1 decoder + encoder in Rust | **Read-only `.ifs` parser**; C binaries are the baseline | Bit-exactness not required. Deletes the pyramidal port, the writer, and the whole libm/`ilog2` hazard investigation |
| 18 | `double` moments inherited from 1998 | **Integer-exact fixed-point moments** | Metal has no fp64, and f32 is *wrong* here (`Σ r²` exceeds f32's exact-integer range). Integers make CPU/GPU agreement exact rather than tolerated — and are faster and more accurate than the original |
| 19 | `pulp` runtime SIMD dispatch | **NEON directly**, no dispatch | One target. `pulp` was right for a portable codec, is overhead for this one |
| 20 | Estimates in focused-days | **Agent-session sizes + verification burden + a dependency DAG** | Days encode human typing speed. Verification burden is the number that actually predicts risk, and it does not correlate with size |
| 21 | Code ground rules | **§2.1 agent working agreement** (A1–A8) | The failure mode changes from *not finishing* to *silent plausible wrongness*. A2 (agents may not weaken tests) is the single most important line in the document |
| 22 | Lena excluded on venue grounds | Lena used freely; Kodak still headline | GitHub-only. The statistical reason for Kodak survives; the political one does not |
| 23 | Papers as Step 21 | **README + reproducibility package** | No publication goal — but the one-command figure regeneration matters *more* without peer review, being the only remaining external check |
| 24 | — | Tolerances loosened (0.05→0.2 dB, 1%→5% on `evals`) | Without bit-exactness, unreachably tight bounds only invite the tolerance-widening that A2 exists to prevent |
| 25 | — | Kill criterion on *sequences* of tolerance widenings (§6) | Each widening is individually defensible; the sequence is what kills the numbers, and no per-step check can see it |

---

## 12. Questions — answered, and what is newly open

### Answered (2026-09-13)

| Question | Answer | Where it lands |
|---|---|---|
| Team size and cadence | AI agents, no team | §2.1, §5 |
| Publication a goal? | GitHub only | §M9, Step 21, Goal C |
| Bit-exactness hard or nice? | Not required | Group B, §11 rows 17–18, 24 |
| Target platforms | Apple M3 (aarch64) + GPU | §2.2, §M4, Steps 7, 11 |

Two of these changed the plan more than they might appear to.

**"Not required" removed a constraint that was doing useful work.** Bit-exactness is a
harsh, unambiguous, machine-checkable oracle — exactly the kind of check §2.1 says agent
execution most needs, and dropping it removes the one place where the answer could not be
argued with. The replacement is deliberate rather than incidental: **exact transform-count
equality** at Step 5, **exact integer moment equality** at Step 6, and **bit-identical
CPU/GPU results** at Step 7. Each is binary, each is cheap, and together they cover the
same ground for less work. That substitution is the main reason Group B still holds
together after losing its original organising goal.

**"M3 + GPU" was the larger change.** It moved the GPU from a deferred optional backend
to the keystone of the measurement methodology, and it forced the numerics decision in
Step 6 that makes CPU/GPU agreement exact rather than negotiated.

### Newly open

1. **Which M3 variant is the reference machine?** Every timing number in `results/` is
   tied to it (§M4), and 8-core/10-GPU versus 16-core/40-GPU is not a detail — it changes
   whether the Step 7 exit criterion ("Kodak oracle in hours") is comfortable or tight.
   **Blocks Step 2.**
2. **Is the GPU budget for the oracle, or for the product encoder too?** This plan assumes
   oracle-only: the shipped encoder stays CPU, and the GPU is a research instrument.
   Extending it to the product path is defensible but would add a whole backend-parity
   burden to every later step. *Recommendation: keep it oracle-only until Gate D; revisit
   with data.*
3. **How much unattended compute is available?** Steps 2 and 8 are wall-clock bound, not
   session bound. If the machine is also the daily driver, the corpus sizes in §M9 should
   be cut before Step 2 rather than after.
4. **Who reads the gates?** A8 assumes a human checkpoint at Gates A–D. If nobody reads
   them, A2 and A7 are all that stand between the project and a set of well-formatted
   numbers that mean nothing — and the honest response would be to cut scope to Gate B
   rather than to loosen the gates.
