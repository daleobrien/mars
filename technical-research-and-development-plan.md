# Mars — Technical Research & Development Plan

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

I would not begin with CUDA.

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

---

## 24. Development phases

Here's the roadmap I'd actually follow.

| Phase | Goal | Result |
|---|---|---|
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
| 13 | Final research | Benchmark + papers + release |

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

Then establish bit-exact or mathematically equivalent reference behaviour where possible.

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

---

## 27. Success criteria

I would define three separate goals rather than one.

**Goal A — research success**

Demonstrate that modern search reduces fractal encoding complexity dramatically while preserving RD performance.

**Goal B — codec success**

Beat the original Mars by an absurd margin in:

* speed
* image size
* image quality
* supported image types

**Goal C — experimental success**

Demonstrate at least one interesting new result, such as:

> A learned candidate-pruning system reduces exact fractal evaluations by 90–99% while maintaining essentially the same rate-distortion curve.

That would be a very respectable result even before INR/GPU work.

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
