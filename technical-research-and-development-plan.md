# Mars — Technical Research & Development Plan

> **Status:** research direction document (the *why* and *what*).
> The executable counterpart — ordered steps, gates, measurement contract, kill
> criteria — lives in [implementation-plan.md](implementation-plan.md).
> Where the two disagree, the implementation plan wins; §11 of that document lists
> every deliberate divergence and its reason.
>
> **Revision note (2026-09-13):** this document has been revised for measurement-first
> sequencing, falsifiable success criteria, and honest positioning against modern
> codecs. Revised passages are marked **[rev]**.

## 1. The central idea

The original Mars (in the folder fratal-mars) is actually a remarkably good starting point because it already separates several ideas that we can now recombine much more intelligently.

Its 1998 design is essentially:

```
quadtree partition → classify blocks → search candidate domains → affine fractal transform → quantize parameters → iterate decoder
```

It already contains Fisher, Hurtgen, MassCenter and Saupe-style acceleration, adaptive split thresholds, multiple decoding strategies, and arbitrary zooming.

The problem isn't the underlying idea.

The problem is that the 1998 implementation is constrained by:

* exhaustive-ish candidate searches
* CPU scalar processing
* hand-built KD-tree infrastructure
* relatively primitive feature vectors
* fixed block/classification strategies
* no rate-distortion optimisation
* no residual layer
* no modern entropy model
* grayscale-only assumptions
* an old binary format
* C/global-state architecture
* no SIMD/GPU architecture
* no learned candidate selection

So I would make the project Mars 2: a modern experimental fractal image codec, while preserving the original Mars implementation as the reference baseline.

---

## 2. The design philosophy

I recommend five principles.

### A. Preserve the mathematics, replace the machinery

Don't throw away fractal coding.

Instead:

Fractal transforms become one of several prediction mechanisms selected by a modern rate-distortion controller.

That is much more interesting than simply making the old algorithm 20× faster.

### B. Separate research from the codec core

Every major algorithm should be a replaceable component:

* Partitioner
* FeatureExtractor
* CandidateRetriever
* TransformEstimator
* ResidualCoder
* EntropyModel
* RateDistortionModel
* Decoder

That lets us publish experiments without continually rewriting the codec.

### C. Make encoding expensive if necessary — decoding must stay cheap

This is one place where fractal coding has an interesting advantage.

We can allow the encoder to perform:

* large searches
* multiple candidate models
* neural inference
* optimisation
* beam searches

provided the resulting bitstream remains extremely cheap to decode.

### D. Treat the partition tree as information

The tree itself is not merely an implementation detail.

It tells us:

* where the image is complex
* where self-similarity exists
* where prediction works
* where residual coding is required
* where additional bits are valuable

That makes it a natural backbone for the whole codec.

### E. Keep the research reproducible

Every experiment should produce:

* image
* parameters
* bitstream
* decoded image
* PSNR
* SSIM
* MS-SSIM
* bits/pixel
* encode time
* decode time
* memory
* partition statistics
* candidate-search statistics

That will make Mars 2 considerably more valuable as an open research project than simply being "another image codec."

---

## 3. What I would actually build

The architecture I recommend is:

```
                    ┌─────────────────────┐
                    │       Image         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Colour conversion   │
                    │ / pyramid           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Adaptive partition  │
                    │      controller     │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Range features        Range thumbnail
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Candidate retrieval │
                    │  coarse → precise   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Fractal transform   │
                    │ fitting             │
                    └──────────┬──────────┘
                               │
                     error too high?
                       /           \
                     yes            no
                      │              │
                      ▼              ▼
               ┌────────────┐   ┌─────────────┐
               │ residual / │   │ accept      │
               │ subdivision│   │ transform   │
               └─────┬──────┘   └──────┬──────┘
                     │                 │
                     └────────┬────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Rate-distortion     │
                    │ optimisation        │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Quantisation        │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Context modelling   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ ANS entropy coding  │
                    └──────────┬──────────┘
                               ▼
                       .mars2 bitstream
```

This is the core architecture I would commit to.

---

## 4. The most important new idea: fractal prediction + residual

This is where I think Mars 2 could become substantially more interesting than the 1998 system.

Traditional fractal coding effectively says:

> Find a domain block whose transformed version resembles this range block.

Instead, Mars 2 should say:

> Find the cheapest predictive representation of this range block.

Candidate representations become:

**Mode 0 — flat**

```
constant
```

**Mode 1 — affine**

```
a*x + b
```

**Mode 2 — fractal**

```
a * transform(domain) + b
```

**Mode 3 — fractal + residual**

```
a * transform(domain) + b + residual
```

**Mode 4 — subdivide**

```
split into children
```

**Future Mode 5 — INR**

```
small implicit neural representation
```

Then select:

```
J = D + λR
```

where:

* D = reconstruction distortion
* R = number of bits
* λ = quality/compression tradeoff

Later we can extend this to:

```
J = D + λR + μT
```

where T is decoding cost.

This gives us a genuinely modern codec architecture.

---

## 5. Adaptive partitioning should become the heart of Mars 2

The original Mars already has adaptive thresholding.

Recent research strongly reinforces this direction.

A 2026 paper on non-uniform fractal partitioning proposes adapting range/domain block sizes according to local texture and edge characteristics, while using task-serial/data-parallel processing and SIMD acceleration. It reports very large encoding/decoding improvements, although those numbers should be regarded as the authors' experimental results rather than assumptions for Mars 2.

A 2025 patent application similarly explores adaptive quadtree depth using local standard deviation, area-weighted error and complexity prioritisation.

So rather than:

```rust
if error > threshold {
    split();
}
```

Mars 2 should estimate:

```
expected_gain_from_split
--------------------------------
additional_bits
+ additional_decode_cost
```

and split only when worthwhile.

For example:

```rust
struct SplitDecision {
    split: bool,
    predicted_gain: f32,
    estimated_rate_cost: f32,
    estimated_decode_cost: f32,
}
```

That turns the partition tree into an optimisation problem.

---

## 6. Candidate search: don't build another KD-tree first

The old Mars uses KD-tree nearest-neighbour methods, following the 1990s feature-vector approach. Its manual exposes parameters for neighbour count and search tolerance.

I would not immediately reproduce that architecture.

Instead, use a funnel:

**Stage 1 — very cheap features**

For every block:

* mean
* variance
* min
* max
* range
* gradient energy
* horizontal energy
* vertical energy
* edge energy

**Stage 2 — structural signature**

Add:

* 4 quadrant means
* 4 quadrant variances
* DCT low-frequency coefficients
* gradient orientation

This is particularly well motivated by recent work using DCT-derived block classes to reduce fractal search time.

**Stage 3 — thumbnail distance**

Compare, say:

* 4×4
* 8×8

downsampled representations.

**Stage 4 — exact affine fit**

Only now calculate:

* scale
* offset
* MSE

for the best candidates.

So instead of:

```
10,000 candidates × expensive comparison
```

we aim for:

```
10,000
  ↓ cheap statistics
1,000
  ↓ DCT/feature filtering
100
  ↓ thumbnail similarity
16
  ↓ affine fitting
1
```

That is likely to produce a much larger practical speedup than simply throwing more threads at the old search.

---

## 7. SIMD is not an optimisation phase — it is part of the architecture

Rust's current SIMD ecosystem is considerably more capable than what was available to Mars in 1998.

`pulp` provides runtime-dispatched SIMD implementations and supports both automatic and manually vectorised operations.

Rust's own portable SIMD API has also evolved substantially, although the standard `Simd` API remains experimental/nightly as of the current documentation.

I'd therefore initially use `pulp` for the performance-critical abstraction rather than coupling Mars 2 directly to x86 intrinsics.

**[rev] Superseded by the platform decision.** With Apple M3 (aarch64) as the sole target,
NEON is unconditional and there is nothing to dispatch between. Use `std::arch::aarch64`
or `core::simd` directly; `pulp` was the right call for a portable x86/ARM codec and is
pure overhead for a single-target one. Revisit if portability ever becomes a goal.

**[rev] And one kernel that belongs on this list ahead of `SSD`:** the six block moments
`(s1, s2, t0, t1, t2)`. Mars 1 accumulates them in `double`, but range pixels are `u8` and
domain pixels are 2:1 contractions, so `4d` is always an integer — which makes every
moment **exactly representable in fixed-point integer arithmetic**. That is faster than
the float version, more accurate, and it is what allows a GPU backend to agree with the
CPU bit-for-bit. See implementation-plan.md Step 6.

The important SIMD kernels will be:

* block mean
* block variance
* SAD
* SSD
* dot products
* gradient calculation
* downsampling
* DCT
* affine transform
* residual reconstruction
* quantisation

The most important one is probably:

```
SSD(range, transformed_domain)
```

because that is executed potentially millions of times.

---

## 8. Memory layout matters enormously

The old implementation thinks in terms of individual blocks.

Mars 2 should think in terms of arrays of blocks.

For example:

```rust
struct BlockBatch {
    positions: Vec<BlockPosition>,
    means: Vec<f32>,
    variances: Vec<f32>,
    gradients: Vec<f32>,
    features: Vec<FeatureVector>,
}
```

Rather than:

```rust
struct Block {
    pixels: Vec<u8>,
    mean: f32,
    variance: f32,
    features: FeatureVector,
}
```

we favour SoA-style storage:

```
means      = [ ... ]
variances  = [ ... ]
gradients  = [ ... ]
features   = [ ... ]
```

because that maps naturally to SIMD and parallel processing.

---

## 9. Rayon for coarse parallelism

Then put SIMD inside Rayon tasks.

Conceptually:

```
image
 ├── partition jobs
 │     ├── SIMD
 │     ├── SIMD
 │     └── SIMD
 │
 ├── feature jobs
 │     ├── SIMD
 │     └── SIMD
 │
 └── search jobs
       ├── SIMD
       ├── SIMD
       └── SIMD
```

This gives us two levels of parallelism:

* thread-level parallelism

plus

* data-level parallelism

which matches the direction of the 2026 fractal-compression work.

---

## 10. GPU comes later — but the architecture should permit it

**[rev] Reversed by the platform decision — the GPU should come early, and for a
different reason than speed.**

The reasoning below is sound for *discrete* GPUs, where host↔device transfer has to be
amortised and a GPU only pays once everything else is already fast. Neither premise holds
on Apple Silicon: unified memory means there is no transfer to amortise.

More importantly, the workload most worth accelerating is not the product encoder — it is
the **oracle**, the full exhaustive `N × M × 8` sweep that yields the RD upper bound, the
top-k recall metric, and the training labels for §20–21. It is embarrassingly parallel,
needs no cleverness, is exact and so carries no rate-distortion risk, and is otherwise too
expensive to run across a corpus at all.

So the GPU does not make the codec faster here. **It makes the measurement methodology
affordable**, which is worth more. It moves from Phase 9 to implementation-plan.md Step 7,
where it sits inside the reference group rather than after it.

One constraint that follows: **Metal has no fp64**, which is the other reason the moment
arithmetic in §7 needs to be integer-exact rather than floating-point.

I would not begin with CUDA — and on this target, not with CUDA at all; Metal via `wgpu`.

First get:

```
scalar Rust
→ SIMD Rust
→ multicore Rust
```

working.

Only then move the embarrassingly parallel kernels to GPU.

The candidate-search pipeline is a particularly attractive GPU workload:

```
N range blocks
×
M candidate domains
×
P pixels
```

But the GPU should be an optional backend:

```rust
trait CandidateSearch {
    fn search(&self, ranges: &[RangeBlock]) -> Vec<Match>;
}
```

with:

* `CpuSearch`
* `SimdSearch`
* `GpuSearch`

behind it.

---

## 11. Entropy coding

I would not write an entropy coder ourselves initially.

The Rust `constriction` library provides ANS and range coding, with composable entropy models and Rust/Python APIs. It explicitly targets both research and production use.

I'd start with rANS.

Why?

Mars 2 will eventually have symbols such as:

* partition decisions
* mode
* domain index delta
* transform ID
* scale
* offset
* residual coefficients
* block size

These have highly non-uniform distributions.

ANS lets us exploit those distributions efficiently. `constriction` also supports entropy models and random-access-related facilities useful to a structured codec.

---

## 12. The .mars2 format

I'd make the format explicitly versioned from day one.

Something like:

```
MARS
2
flags
width
height
channels
bit_depth
colour_space
partition section
feature/model metadata
transform section
residual section
entropy-model section
payloads
indexes
```

But importantly, the decoder must never depend on encoder implementation details.

The format describes a representation.

Not:

> "run algorithm X with these settings."

That makes future decoder implementations possible.

---

## 13. Progressive decoding

There's another opportunity here.

Instead of one monolithic bitstream:

```
image
```

make the representation naturally progressive:

```
base layer
   ↓
coarse image
   ↓
partition refinement
   ↓
fractal refinement
   ↓
residual refinement
   ↓
high-quality image
```

This is an especially interesting research direction because recent compression research continues to investigate progressive and hierarchical representations. A September 2026 paper, for example, explores tree-structured vector quantisation specifically so that prefixes of a representation correspond to progressively improved reconstructions.

The connection to Mars 2 is conceptually strong:

```
quadtree + progressive refinement + entropy coding
```

could become a distinctive feature of the codec.

---

## 14. INR should be an experiment, not the foundation

I still strongly recommend experimenting with implicit neural representations.

But not in v1.

The 2021 INR compression work demonstrated a complete pipeline involving:

* fitting an INR to the image
* quantisation
* quantisation-aware retraining
* entropy coding

but also explicitly identifies the major problem: encoding can be orders of magnitude slower because the network is fitted to each image. Meta-learned initialisation can reduce that cost.

That's exactly why I would make INR a fallback representation for difficult blocks.

Imagine:

```
Range block
    │
    ├── fractal prediction excellent → fractal
    │
    ├── fractal prediction mediocre  → fractal + residual
    │
    ├── self-similarity poor         → local transform
    │
    └── highly irregular             → INR
```

That's much more interesting than building yet another "INR image compressor."

---

## 15. And there is an intriguing future hybrid

Recent patent activity is moving toward exactly the territory where this becomes interesting.

For example, InterDigital's 2025 PCT application WO2025256970 describes partitioning an image into coding units, representing those units using INR networks, then entropy coding their parameters.

That doesn't mean we should copy the approach.

It does suggest the broader design direction:

```
image
 ↓
adaptive regions
 ↓
different representation per region
 ↓
entropy coding
```

Mars 2 can explore a different point in that design space:

```
fractal + transform + residual + optional INR
```

That would be a genuinely worthwhile research question.

---

## 16. Colour should be supported from the beginning

Mars itself is grayscale.

Mars 2 shouldn't be.

I'd use:

```
RGB
 ↓
YCbCr / opponent colour space
 ↓
Y:      high-quality fractal coding
Cb/Cr:  independently controlled coding
```

Initially:

* Y quality = high
* chroma quality = lower

Eventually the optimiser decides the allocation.

This also makes the codec useful for modern datasets rather than restricting it to historical grayscale benchmarks.

---

## 17. Rust workspace

I'd structure the repository approximately like this:

```
mars2/
│
├── crates/
│   ├── mars-core/
│   ├── mars-image/
│   ├── mars-partition/
│   ├── mars-features/
│   ├── mars-search/
│   ├── mars-transform/
│   ├── mars-residual/
│   ├── mars-entropy/
│   ├── mars-format/
│   ├── mars-codec/
│   ├── mars-simd/
│   ├── mars-gpu/
│   └── mars-cli/
│
├── python/
│   ├── analysis/
│   ├── training/
│   └── experiments/
│
├── benches/
├── tests/
├── datasets/
├── papers/
├── patent-notes/
├── experiments/
└── docs/
```

**[rev] Do not create all thirteen crates on day one.** `mars-gpu`, `mars-residual`
and `mars-entropy` would sit empty for months, and workspace churn costs real build
time, import noise and merge conflicts. Start with four — `mars-core`, `mars-codec`,
`mars-bench`, `mars-cli` — and split each remaining crate out at the step that
introduces it, when the boundary is known rather than guessed. The layout above is the
*destination*, not the starting point.

And the key public API should remain tiny:

```rust
pub fn encode(
    image: &Image,
    options: &EncodeOptions,
) -> Result<EncodedImage>;

pub fn decode(
    data: &[u8],
) -> Result<Image>;
```

Everything underneath can evolve.

---

## 18. Core data model

Something along these lines:

```rust
pub struct RangeBlock {
    pub x: u32,
    pub y: u32,
    pub size: u16,
    pub depth: u8,
}

pub struct DomainBlock {
    pub x: u32,
    pub y: u32,
    pub size: u16,
}

pub struct FractalTransform {
    pub domain: DomainRef,
    pub orientation: Orientation,
    pub scale: QuantizedScale,
    pub offset: QuantizedOffset,
}

pub enum CodingMode {
    Constant,
    Fractal(FractalTransform),
    FractalResidual(FractalTransform),
    Transform,
    Split,
    Inr,
}

pub struct CodeBlock {
    pub range: RangeBlock,
    pub mode: CodingMode,
}
```

This is deliberately boring.

That's good.

---

## 19. The encoder should become an optimiser

The really interesting part is:

```rust
fn optimise_block(block: RangeBlock) -> Candidate {
    let candidates = generate_candidates(block);
    candidates
        .into_iter()
        .map(evaluate_rate_distortion)
        .min_by(|a, b| a.cost.total_cmp(&b.cost))
        .unwrap()
}
```

Each candidate has:

```rust
struct Candidate {
    mode: CodingMode,
    distortion: f32,
    estimated_bits: f32,
    decode_cost: f32,
}
```

Then:

```
cost = distortion + lambda * estimated_bits + mu * decode_cost;
```

This gives us a single mathematical framework into which almost every future technique can plug.

**[rev] λ is not a constant to be tuned once — it *is* the quality control.**

This needs saying explicitly, because getting it wrong invalidates every rate-distortion
comparison in the project:

* the encoder exposes λ as its primary knob;
* a "quality preset" is a λ value;
* **an RD curve is a λ sweep**, not a sweep of some other threshold with λ held fixed.

Two further consequences:

* `estimated_bits` must come from the *live* entropy models, not a constant-bits
  approximation — a fixed estimate biases every decision toward whichever modes have
  cheap headers.
* The resulting RD curve must be monotone and roughly convex. Non-convexity is the
  cheapest available diagnostic for a rate-estimation bug; check it on every sweep.

---

## 20. Learned models come after the deterministic system

This is important.

Don't start by training a neural network.

First generate a giant corpus of decisions from the deterministic encoder:

* range block
* features
* candidate list
* candidate errors
* candidate bit costs
* chosen mode
* chosen split

Then train models to predict:

**Model A**

Should this block split?

**Model B**

Which candidate domains are worth evaluating?

**Model C**

Which coding mode is likely to win?

This gives us learned acceleration, rather than a neural codec replacing the entire project.

That's a much cleaner scientific experiment.

---

## 21. The experiment that I think could be genuinely novel

I'd particularly investigate:

**Learned candidate pruning for fractal coding**

Train a lightweight model:

* range features
* domain features
* relative position
* scale
* texture descriptors

to estimate:

```
P(domain is among top-k)
```

Then only evaluate domains whose probability exceeds a threshold.

Compare:

* exhaustive
* KD-tree
* feature classification
* ANN
* learned pruning
* hybrid ANN + learned pruning

Measure:

* PSNR
* SSIM
* bits/pixel
* encode time
* candidate evaluations
* false-negative rate

This gives us an excellent research paper experiment even if the final codec doesn't use the neural model.

---

## 22. Patent landscape: how I'd handle it

Because this is explicitly open source and non-commercial, I would treat patents as:

1. prior-art research
2. algorithm-history documentation
3. design-space mapping
4. future-risk awareness

rather than allowing patent concerns to dominate the project.

There is nevertheless recent activity worth documenting.

The 2023 Chinese patent CN117241042A/B covers DCT-based block classification for fractal compression and parallel affine search. Google Patents lists it as granted/active, although its legal-status display carries the usual disclaimer.

CN120259451A, filed in 2025, covers adaptive quadtree segmentation using standard deviation, area weighting and complexity-based prioritisation; Google lists it as pending.

WO2025256970A1, filed by InterDigital, covers coding-unit-based INR representation and entropy coding of INR parameters and is listed as pending.

For the open-source project, I'd create:

```
patent-notes/
    README.md
    fractal-classification.md
    adaptive-quadtree.md
    neural-representation.md
    entropy-coding.md
```

with:

* Patent
* Priority date
* Jurisdiction
* Status
* Relevant concept
* What Mars 2 does differently
* Relevant claims
* Open-source implementation notes

Not legal advice, and not a freedom-to-operate opinion. If the project ever changes from non-commercial research to commercial distribution, we'd do a proper patent review.

**[rev] The same discipline should extend to the papers cited in this document**, not
just the patents. Several of the works referenced above are very recent and none were
verified while drafting. Keep `docs/claims.md` with one row per external claim recording
the citation, an access date, a status (`unverified` / `verified` / `contradicted` /
`withdrawn`), whether any design decision depends on it, and what we independently
measured.

The rule that matters: **no design decision may depend on an unverified claim.** This is
not pedantry in this particular field — reported speedups in the fractal-compression
literature are frequently measured against unoptimised exhaustive-search baselines on
unstated hardware, which makes them incomparable to ours by construction. Our own
recorded baseline is the only thing our numbers are ever quoted against.

---

## 23. Licensing

The original Mars is GPL-2-or-later.

I'd initially keep Mars 2 GPL-2.0-or-later, assuming we want the strongest continuity with the original project.

However, there's an important distinction:

* ideas/algorithms → can be independently reimplemented
* original source code → copyright/licensing matters
* third-party dependencies → each license must be tracked
* patents → separate question from copyright

So I would make Mars 2 a clean Rust implementation, using the original Mars source as a historical/reference implementation rather than mechanically translating its C.

That also gives us a much cleaner codebase.

**[rev] One concrete interaction to settle before dependencies accumulate.**
Apache-2.0 code cannot be combined into a GPL-2.0-**only** work — its patent-termination
and indemnification clauses count as additional restrictions under GPLv2. It *is*
compatible with GPL-3.0. Most of the Rust ecosystem is dual-licensed MIT/Apache-2.0, and
MIT alone is GPLv2-compatible, so this is navigable — but the consequence should be a
decision rather than a discovery:

* keeping **GPL-2.0-or-later** means a recipient may choose v3, so the combination is
  lawful, but **the effective licence of the distributed binary becomes GPL-3.0**;
* any Apache-2.0-only dependency (with no MIT option) forces that outcome.

Enable `cargo-deny` license checking in the very first commit, record the decision in
`docs/licensing.md`, and prefer dual-licensed crates where there is a choice. Cheap now;
expensive once a dependency tree exists.

---

## 24. Development phases

Here's the roadmap I'd actually follow.

**[rev] The phase table below had benchmarking at Phase 13. That was backwards** — it
made every intermediate claim an assertion rather than a measured delta. Measurement is
now Phase 0, and its first subject is Mars 1, not Mars 2. Phases are also grouped into
four gated blocks with explicit kill criteria; see
[implementation-plan.md](implementation-plan.md) §4–§6.

| Phase | Goal | Result |
|---|---|---|
| **0a** | **Measurement harness** | **Metrics, BD-rate, result store, CI gate — before any codec code** |
| **0b** | **Baseline capture** | **Mars 1 RD curves + `evals/transform`, recorded** |
| 0 | Archaeology | Exact Mars 1 behavioural specification |
| 1 | Rust reference | Correct, boring Mars-compatible codec |
| 2 | Modern representation | New .mars2 format |
| 3 | SIMD | Major CPU acceleration |
| 4 | Adaptive partition | Non-uniform quadtree |
| 5 | Hierarchical search | Multi-stage candidate retrieval |
| 6 | R-D optimisation | Proper rate/distortion decisions |
| 7 | Residual mode | Fractal + residual |
| 8 | Parallel encoder | Rayon + SIMD |
| 9 | GPU backend | Optional accelerated search |
| 10 | Learned pruning | ML-assisted candidate search |
| 11 | Progressive codec | Quality refinement layers |
| 12 | INR experiment | Hybrid representation |
| 13 | Final research | Papers + reproducibility package + release |

**[rev]** Phases 9 (GPU), 10 (learned pruning), 11 (progressive) and 12 (INR) are
*hypotheses with entry conditions*, not commitments. Each carries an abort rule. In
particular, the GPU backend should not be started if the hierarchical search and learned
pruning have already removed the workload it was meant to accelerate.

---

## 25. What I would build first

Not the fancy stuff.

The first milestone should be:

**Mars 2 Reference Encoder**

It should implement:

```
PGM/PNG input
       ↓
grayscale
       ↓
fixed quadtree
       ↓
domain pool
       ↓
all candidate searches
       ↓
affine fitting
       ↓
quantisation
       ↓
simple entropy coding
       ↓
Mars2 file
```

And the decoder:

```
Mars2 file
 ↓
partition
 ↓
fractal transforms
 ↓
iteration
 ↓
image
```

**[rev] Correction: build the decoder first, not the encoder.**

The decoder is roughly 500 lines against the encoder's 3,500, it needs no search
infrastructure at all, and it can be validated against artefacts that already exist —
`.ifs` files produced by the 1998 binary. That ordering keeps format bugs and encoder
bugs from masking each other, and it means the encoder later gets validated three ways
(Rust→Rust, Rust-encode→C-decode, C-encode→Rust-decode) instead of one.

**[rev] "Bit-exact or mathematically equivalent" needs to be made precise**, because the
hazards are specific and would otherwise be discovered a month in:

1. **x87 excess precision** — on 32-bit x86, C `double` intermediates may be evaluated at
   80-bit precision, which Rust will not reproduce. Pin the reference build to x86-64 or
   aarch64.
2. **`-ffast-math` must stay off** in the reference build, or the C itself becomes
   compiler-dependent.
3. **libm `log` is not correctly rounded** and differs across platforms — and Mars 1 calls
   it in three places that affect the *bitstream*: `virtual_size`,
   `bits_per_coordinate_*`, and the adaptive-threshold update. The first two take small
   integer arguments and should be replaced by exact integer `ilog2`, with equivalence
   proven by exhaustive check over the parameter domain. The third depends on libm for
   arbitrary `adapt` — but at the default `adapt = 1.0`, `log(1.0)` is exactly `0.0` on
   every conforming implementation, so the term vanishes and the default path is
   bit-exact.

So the honest claim is: **bit-exact at default settings; numerically equivalent within a
stated tolerance otherwise.** Say that in the documentation rather than claiming more.

Only after that do we optimise.

---

## 26. The first benchmark

We should make this brutally measurable.

For every image:

| Codec | bpp | PSNR | SSIM | encode | decode |
|---|---|---|---|---|---|
| Mars 1 | | | | | |
| Mars 2 reference | | | | | |
| Mars 2 SIMD | | | | | |
| Mars 2 adaptive | | | | | |
| Mars 2 search | | | | | |
| Mars 2 residual | | | | | |
| Mars 2 learned | | | | | |
| JPEG | | | | | |
| JPEG 2000 | | | | | |
| WebP | | | | | |
| AVIF | | | | | |
| JPEG XL | | | | | |

And importantly:

```
candidate evaluations / range block
```

because that's the metric that will tell us whether our search innovations are actually working.

**[rev] Three things this table needs before it means anything.**

**First — the modern codecs are context, not targets.** Fractal coding is not going to
beat AVIF or JPEG XL on rate-distortion. It was not competitive with JPEG 2000 in 2001,
and nothing in this plan changes that; transform coding plus learned entropy modelling
has had twenty-five more years of investment. Listing AVIF and JXL without saying so
invites the project to adopt a success criterion it cannot meet and then to read a
predictable loss as failure. They answer "where does this sit?" — they do not define
success. See §27.

**Second — a single `(bpp, PSNR)` pair is not a comparison.** Every RD claim needs
BD-rate: at least four quality points per curve, an overlapping bpp range, interpolation
over (log bpp, PSNR), and the overlap interval reported alongside the percentage. Rows in
the table above are curves, not cells.

**Third — timing needs a protocol.** Median of at least five runs with the median
absolute deviation, A/B interleaved rather than grouped so thermal drift cancels, and a
recorded machine fingerprint. A speedup figure without absolute times and a fingerprint
is not a result.

**[rev] Corpus.** Lena should be kept only as a Mars 1 compatibility fixture — the
original ships `lena.raw`, so golden outputs require it — and never used for a reported
number; major imaging venues have discouraged it. Kodak is the reporting corpus. Add
USC-SIPI textures deliberately: fractal coding's whole premise is block self-similarity,
so a corpus of photographs alone will systematically misattribute where the method works.

**[rev] One metric to add now, because it is nearly free and unlocks several later
steps: the oracle.** Run exhaustive search once per configuration and cache, per range
block, the true best match and the top-32 candidates. That single artefact yields the
RD upper bound for fractal-only mode, a **top-k recall and RMS-regret** measurement for
every fast search method, and the labelled training corpus that §20 and §21 would
otherwise have to generate as a separate effort.

---

## 27. Success criteria

I would define three separate goals rather than one.

**Goal A — research success**

Demonstrate that modern search reduces fractal encoding complexity dramatically while preserving RD performance.

**Goal B — codec success**

**[rev]** "Beat the original Mars by an absurd margin" is not testable — a criterion that
cannot be failed cannot be passed either. Replaced with numbers, each with a measurement
protocol:

* **encode ≥ 20× faster at matched BD-rate**, on a pinned machine, timed per the protocol
  in §26;
* **BD-rate ≥ 35% better than Mars 1** on Kodak;
* colour support, progressive decoding, and a fuzzed format parser.

The original list of dimensions was right; it just needed thresholds and a way to check
them.

**Goal C — experimental success**

Demonstrate at least one interesting new result, such as:

> A learned candidate-pruning system reduces exact fractal evaluations by 90–99% while maintaining essentially the same rate-distortion curve.

That would be a very respectable result even before INR/GPU work.

**[rev]** To make Goal C falsifiable, it needs two additions. Model inference cost must be
counted inside the evaluation budget — a model that costs more than the evaluations it
saves is not an acceleration. And generalisation must be demonstrated across corpora
(train on Kodak, evaluate on CLIC and USC-SIPI), with the cross-corpus gap reported
explicitly; that is the first thing a reviewer will ask for.

**[rev] And one criterion that is explicitly *not* on the list:** beating AVIF or
JPEG XL on rate-distortion. See §26.

---

## 28. My proposed "killer" architecture

If we eventually combine everything, I see Mars 2 becoming:

```
                   MARS 2
                     │
          ┌──────────┴──────────┐
          │ Adaptive quadtree   │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │ Feature extraction  │
          └──────────┬──────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Fractal       Local        INR
     search       transform    fallback
        │            │            │
        └────────────┼────────────┘
                     ▼
             Rate/distortion
                optimiser
                     │
             ┌───────┴───────┐
             ▼               ▼
        reconstruction     residual
             │               │
             └───────┬───────┘
                     ▼
              Context models
                     │
                     ▼
                   rANS
                     │
                     ▼
                 .mars2
```

And that is substantially more ambitious than "Mars rewritten in Rust."

It is essentially a research platform for self-similarity-based image compression.

---

## 29. One particularly nice aspect of this project

The original author explicitly designed Mars so that new acceleration methods could be plugged in independently; the README says that a new speed-up method could be integrated by implementing only two functions.

We should preserve that spirit—but take it much further.

Imagine being able to write:

```rust
impl CandidateRetriever for MyResearchIdea {
    // ...
}
```

and immediately benchmark it against:

* Exhaustive
* Fisher
* Hurtgen
* Saupe
* KDTree
* DCTClass
* SIMD
* ANN
* Learned
* MyResearchIdea

That turns Mars 2 into something researchers can actually experiment with, rather than merely another codec.

---

## My recommended next move

I think the next step should be concrete rather than theoretical:

I would now design the actual Mars 2 repository and Rust API in detail — crate by crate, core structs, traits, data layouts, .mars2 bitstream, encoder pipeline, decoder pipeline, SIMD kernels, benchmark harness, test strategy, and the first ~30 implementation issues.

That would give us something close to a buildable v0.1 engineering specification, rather than just a research roadmap.

And I'd make the first implementation deliberately small enough that we could start writing Rust immediately, while keeping all the extension points for the more exotic research later.

---

**[rev] That next step now exists: [implementation-plan.md](implementation-plan.md).**

It contains the ordered steps, the binding measurement contract, four gates with kill
criteria, the reverse-engineered Mars 1 bitstream specification (header layout, tree
encoding, and the exact quantisation formulas), the first twenty issues ready to file,
and a table of every point at which it deliberately diverges from this document.

Its first instruction is the one that matters most: **build the measurement harness
before the codec, and point it at Mars 1 first.**
