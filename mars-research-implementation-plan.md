# Mars: research implementation and benchmarking plan

## 1. Scope and recommendation

**Source inspected:** [daleobrien/mars](https://github.com/daleobrien/mars), commit [`5fc80de5e28388728915fbcd24ca4843eaf94284`](https://github.com/daleobrien/mars/tree/5fc80de5e28388728915fbcd24ca4843eaf94284), 17 September 2026. Repository paths below are relative to that checkout. This is a proposed implementation plan, not a claim that the proposed experiments have run.

**Objective:** determine whether bounded random search, Pearson/APCC indexing, improved partition optimization, and optional reconstruction enhancements improve Mars's measured rate–quality–time tradeoff enough to maintain them.

**Recommended sequence:**

1. Establish a trustworthy serialized-output benchmark and resolve correctness blockers.
2. Unify candidate retrieval with the existing production encoder, retaining exhaustive search as the reference.
3. Add bounded seeded random search as the inexpensive challenger.
4. Add Pearson/APCC retrieval as the main deterministic challenger.
5. Compare binary-feature and hierarchical-classification alternatives only after the runner works.
6. Reuse and improve the existing quadtree RD optimization; do not build it again.
7. Add optional boundary postprocessing.
8. Attempt sparse multi-domain coding only if earlier stages justify it.

Do not start with metaheuristics, another learned model, GPU kernels, or rectangular partitions. Mars already has several sophisticated search implementations; a literature speedup against exhaustive search is not sufficient reason to add another one.

## 2. What Mars already implements

The README's statement that no Mars 2 codec exists is stale. Actual source is substantially ahead of it.

| Area | Existing implementation | Consequence for this plan |
|---|---|---|
| Codec | `mars-codec/src/encode.rs`, `ifs.rs`, `mars_format.rs`: quantized fitting, iterative decoding, threshold quadtree, entropy-coded streams | Extend; do not rebuild the baseline. |
| RD partitioning | `encode.rs::walk_rd`, `split_rd`, `best_mode_leaf`; `rate.rs::RateModels` | Bottom-up `D + lambda R` selection already exists. Audit the objective and integrate retrieval first. |
| Modes | Flat, spatial affine, single-domain fractal, fractal plus DCT residual | Keep mode sets controlled in search experiments. Sparse DCT residuals are not multi-domain fractal coding. |
| Search | `mars-search`: Exhaustive, Fisher, Hurtgen, MassCenter, Saupe, SaupeFisher, McSaupe, Funnel, Learned | Use existing methods as serious challengers, not just historical context. Random and 2013 APCC are missing. |
| Search interface | `CandidateRetriever`, `DomainPool`, `RangeBlock`, `search_block`, `SizedRetrievers` | Good starting point, but a separate encoder currently consumes them. |
| Acceleration | `mars-simd` integer moments/NEON; `mars-gpu` search; Rayon paths | Preserve existing exact kernels and deterministic threading; profile before more hardware work. |
| Measurement | `mars-core` metrics; `mars-bench` BD-rate, provenance, JSONL store, oracle/recall, anchors | Reuse these components rather than introduce a second metrics implementation. |
| Additional coding | Color, progressive output, adaptive domain density | Existing compatibility restrictions must become explicit capability checks. |
| HV partitions | `hv-split-plan.md` | Proposal only; square leaf geometry and implicit quadtree syntax still dominate the implementation. |

### Critical integration gap

`mars-search` depends on `mars-codec`, but `mars_search::encode_image` is a separate RMS-threshold encoder. Production `encode.rs::walk` and `walk_rd` call exhaustive search directly. The CLI rejects `--method` with `--lambda` or color. Some flags, including mode selection outside RD, have no effect.

Therefore, adding APCC to `MethodName` alone would **not** make it available to the main RD encoder.

### How much to trust existing results

The audit found 10,906 enveloped historical result rows, all marked dirty with run index zero, plus 16 classical-method rows without the standard provenance envelope. This does not prove the measurements wrong, but they cannot serve as a clean, reproducible baseline for this revision. Preserve them; produce a new versioned result series.

Existing anchor support already covers JPEG, OpenJPEG, WebP, AVIF, and JPEG XL. The JPEG implementation recorded by the audit is libjpeg-turbo, not automatically mozjpeg. Existing anchor results use color Kodak; many Mars research gates use grayscale. Do not compare these as equivalent workloads.

## 3. Phase P0 — correctness and measurement baseline

**Priority: required before scientific comparisons.**

### P0.1 Make every quality measurement cross the serialization boundary

Inspect and update `mars-bench/src/rd_opt.rs::sample`, `mode_gate.rs`, `density_gate.rs`, and new experiment code so the authoritative path is:

```text
input file → encode → serialize → actual coded file
                                  ↓
                         read/parse coded file → decode → decoded file
                                                               ↓
                                     mars-core metrics from input + decoded files
```

Some existing gates serialize for byte count but decode the original in-memory leaves. A previous adaptive-density result was withdrawn after this pattern hid a coordinate-serialization problem. The corrected historical density result was about −0.14% BD-rate, not the withdrawn −6.82%.

- Measure whole-container bytes, not estimated payload or summed leaf costs.
- Keep in-memory reconstruction as a differential diagnostic, never the headline quality source.
- Record bitstream hash, decoded-image hash, format version, dimensions, and crop rules.
- Preserve existing golden streams. A change to decoded semantics requires a compatibility decision and regression fixture, even if syntax stays the same.

### P0.2 Reproduce suspected source-level defects before fixing them

| Concern | Evidence at inspected revision | Required regression / decision |
|---|---|---|
| Residual orientation | `residual_for_candidate` constructs residuals at mapped range coordinates `[i,j]`; `ifs::decode_leaf` adds `res[u,v]` while writing `[i,j]` | Nonzero asymmetric residual, all eight isometries, serialize/parse/decode. Establish correct coordinate convention and compatibility impact. |
| Contrast header precision | Fitting uses `params.max_alfa`; header stores a 1/32-quantized value; decoder uses header value | Test nonrepresentable parameters. Normalize before fitting or reject them explicitly. |
| Contractivity | Default maximum is contractive, but configurable/header values can allow coefficients ≥1 | Validate dequantized coefficients. Separate contractive research profile from explicitly noncontractive legacy settings. |
| Domain-grid phase | Contracted samples use even-origin 2×2 averages; candidate lookup divides full coordinates by two | Test odd stride/origins against direct decoder sampling. Reject unsupported grids or implement their phase correctly. |
| Flat/tied features | Zero-energy normalization and non-total sort comparison patterns exist in current search code | Constant blocks, duplicate keys, finite stable sorting, deterministic DC policy. |
| Empty/tiny pools | Search pool enumeration can produce an invalid origin when domains do not fit; forced border splits can fall below indexed sizes | Checked geometry, guaranteed termination, explicit no-candidate handling, odd/non-power-of-two image tests. |

These are source-audit findings, not all reproduced defects. Fix only after a failing focused test identifies the behavior. Keep fixes in isolated changes, then establish a corrected baseline before comparing algorithms.

**Validation actually performed during planning:** `cargo test --locked --offline -p mars-codec residual` passed four tests. This covers residual entropy round-trips and a zero-residual fit, not the nonidentity/asymmetric residual regression above. No headline benchmark or full test suite was run.

### P0.3 Build a strict unified experiment runner

Suggested new modules: `crates/mars-bench/src/experiment.rs`, `experiment_config.rs`, `experiment_report.rs`; thin CLI wiring in `mars-cli/src/bin/marsbench.rs`. Names are proposals, not existing APIs.

Reuse `store::Row`, `ResultStore`, `measure`, `bdrate`, `provenance`, and reports. Keep legacy row readers working; add a versioned experiment payload.

Each planned case must have a stable identity and persist:

- Source SHA, dirty status, dirty-source patch hash if applicable, lockfile/build identity, compiler/flags, harness version.
- Experiment/config/split/image/oracle/model hashes and artifact paths.
- All codec parameters, enabled modes, partition/search strategy, domain stride, quantizers, seed/RNG version.
- Decoder initialization, iteration/stopping policy, filter settings, color/subsampling, whole-container bytes.
- Phase work counters, timing boundaries, raw repetition samples, threads, machine/OS/QoS/operator conditions.
- PSNR, SSIM, MS-SSIM availability, bpp, convergence status, candidate coverage/regret, leaf/mode histograms.
- Explicit success/failure/timeout/unsupported/missing-prerequisite status and denominator counts.

Requirements:

- Append each completed case immediately; support resume only on matching case/build hashes.
- Never silently skip missing oracle caches or opt-in tests in a requested release gate.
- Distinguish unavailable optional metrics from missing experiments.
- Reject unsupported flag combinations rather than accept no-op options.
- Validate corpus completeness before expensive work.
- Do not overwrite historical JSONL or reinterpret its timing as the new protocol.

**Exit P0:** codec round-trip tests pass; defects affecting the selected profile are resolved or excluded explicitly; runner produces reproducible smoke artifacts; missing inputs cause an explicit failure; documentation status is corrected.

## 4. Phase P1 — one search pipeline for legacy and RD encoding

### Architecture

Introduce a codec-owned search-provider interface, implemented by an adapter in `mars-search`. Do not add a reverse `mars-codec → mars-search` dependency.

Suggested ownership:

- `mars-codec`: quantized fit, fitted candidate type, search request/outcome contracts, tree/mode decisions, exhaustive reference implementation.
- `mars-search`: immutable indexes, candidate retrieval strategies, adapters to the codec contract.
- `mars-bench`: comparisons, exhaustive/oracle diagnostics, policy sweeps; no fitting reimplementation.

Route `walk`, `walk_rd`, and eventually RD warm-up through the shared provider. First preserve today's **RMS-best candidate followed by mode competition** behavior. Considering multiple domains under the RD objective is a later separate experiment.

### API requirements

- Immutable query indexes usable through `Sync`; separate index construction from querying.
- Explicit block/plane identity, geometry, legal domain stride, candidate-budget units.
- Separate candidate retrieval from exact fitting; permit a fit-aware driver for early stopping and widening.
- Return fitted parameters and raw moments needed by residual modes, not just coordinates.
- Deterministic candidate order/ties; for a changed method define canonical tie IDs `(row,col,isometry)`.
- Bounded, explicitly counted fallback; distinguish no match from a legitimate flat-block result.
- Cache domain moments per position/size and reuse cross terms where valid. Avoid recalculating domain sums for every orientation.
- Cover forced boundary blocks and sizes below the nominal minimum.

### Instrumentation

Report separately: contracted-plane construction, index build, feature queries, exact fits, fallback fits, RD warm-up fits, tree/mode work, serialization, total encode, decoder setup/iterations/postprocessing.

`evals` continues to mean complete domain/isometry fit evaluations. Also count descriptors scored, pixels/dots processed, nodes visited, cache memory, and candidate allocations. Reducing fits does not prove a speedup if feature scanning still traverses the entire pool. Thread-summed internal timers are not wall time.

For controlled search comparisons initially retain the same exhaustive RD warm-up and charge its full cost. Then test selected-provider warm-up as a separate factor, since it changes both time and frozen rate models.

**Exit P1:** default exhaustive path preserves pinned bytes on the supported fixtures; each existing method can exercise the production grayscale threshold and RD paths; flags behave explicitly; work includes warm-up; thread-count tests pass.

## 5. Phase P2 — bounded seeded random search

Suggested addition: `mars-search/src/random.rs`, configuration/CLI registration, focused search and integration tests.

1. Enumerate legal domain positions deterministically once per block size.
2. Sample without replacement with a pinned PRNG/algorithm; avoid new dependencies if existing deterministic machinery suffices.
3. Define `K` as **domain positions**, giving at most `8K` full fits with eight isometries. Expose actual pair counts too.
4. Derive each block's random stream from master seed, image hash, plane, coordinates, and size—not scheduling order or a process-randomized hash.
5. Keep nested candidate sets across budget sweeps where practical; sort selected positions into reference order for stable ties.
6. Initially evaluate the full bounded sample. Add first-acceptable-RMS stopping as an independently labeled variant with clearly defined RMS units.
7. Budget ≥ pool size must enumerate the exhaustive set in the exhaustive order. Disable early stopping for this equivalence test.
8. Do not silently add exhaustive fallback. If enabled, label it unbounded and count its work.

**Development grid:** K = 16, 32, 64, 128, 256, 512 domain positions; master seeds 0–4. Hold geometry, quantizers, mode set, and stride fixed first. These budgets are experimental choices, not claimed paper reproductions.

Tests: reproducibility across threads, unique legal samples, small pools, empty pools, all-budget equivalence, no dependence on query order, correct early-stop cap, complete stream round-trip.

**Exit P2:** boundedness and determinism proven; fixed-block regret and full-stream RD/time curves recorded; retain a small set of nondominated budgets as future controls.

## 6. Phase P3 — Pearson/APCC indexing

Source: [2013 APCC paper summary](summaries/A_Novel_Fractal_Image_Compression_Scheme_With_Block_Classification_and_Sorting_Based_on_Pearsons_Correlation_Coefficient.md). Suggested new `mars-search/src/apcc.rs` plus an offline training/export utility under `mars-bench`.

### Important limitation: APCC is not Mars's exact objective

For unconstrained real affine contrast and offset, minimizing least-squares error corresponds to maximizing absolute correlation. Mars clamps negative contrast to zero and quantizes contrast/offset; its optimum need not maximize absolute correlation.

Therefore:

- Use correlation only to retrieve/order a shortlist; select the winner with the pinned Mars quantized fitter.
- Do not claim exact equivalence, lossless pruning, or the paper's measured speedup.
- Separate a **Mars-compatible APCC-derived method** from a later faithful signed-contrast reproduction. The latter needs quantization, decoder, contractivity, and format design; it is not necessary for the first challenger.

### Implementation

1. Verify the original PDF's reference-block training, class canonicalization, sign handling, and window construction before labeling a method paper-faithful.
2. Reuse Fisher classification/orientation tables only after tests confirm composition and tie rules.
3. Cache sum/centered energy and stable IDs for domains. Handle constant blocks separately with no NaNs.
4. Train/export a nonconstant reference block per class and supported block size from the training split only. Store configuration, seed, data hash, and weight hash.
5. At index build, sort each class by the verified reference-correlation key with a deterministic tie-breaker.
6. At query time, binary-search and inspect a bounded key-neighborhood. Count both query branches if implementing the paper's positive/negative-range handling.
7. Refit legal candidates under Mars's nonnegative quantized model. Verify signs and isometries with synthetic negative-correlation cases; do not invent a contrast sign through geometric rotation.
8. Define empty-class behavior and bounded widening/fallback explicitly; count all candidate fits and deduplicate deterministically.
9. Provide a train-free fixed-reference control to measure whether offline training is actually beneficial.

**Development grid:** k = 20, 44, 76, 128, 256, measured as total unique candidates as well as per-window allocation; stride 4 first, stride 8 as a separate axis. Compare at equal fit counts and at equal measured time. Only extend to stride 2 or new sizes once coverage and serialization tests support them.

Tests: flat blocks, tied keys, negative correlation, contrast-cap ranking reversal, all isometries, training reproducibility/no split leakage, small/empty classes, serial/parallel identity. An all-window query is exhaustive only over the indexed class/canonical candidates; expose explicit full-search mode for whole-pool equivalence.

**Exit P3:** APCC beats or complements the best existing deterministic method on the validation Pareto frontier; frozen final settings are evaluated without retuning. Otherwise record the negative result and avoid making it default.

## 7. Phase P4 — optional low-cost retrieval challengers

Run these as alternatives before stacking filters; combinations can discard good candidates twice.

### Binary local features

Source: [local-feature paper](summaries/Enhancing_fractal_image_compression_spee.md).

- Implement the paper's 12-bit perimeter-versus-central-mean descriptor first for 4×4 ranges and matching contracted domains.
- Use XOR/popcount as a separately documented implementation choice instead of a 4096² lookup table.
- Compare fixed Hamming thresholds and the paper's contrast-adaptive policy; define no-candidate fallback.
- Existing Hurtgen provides a useful four-bit classification control, not the same algorithm.
- Treat extensions to 8×8/16×16 as new descriptors and revalidate. Negative contrast and orientation handling must follow Mars's actual model.

### Hierarchical intensity-sum classes

Source: [hierarchical-classification paper](summaries/Fractal_Image_Compression_using_Hierarch.md).

- Implement P-I first: quadrant and subquadrant rank codes, sparse occupied buckets, deterministic tie policy.
- Do not allocate a dense 24^5 class table per block size.
- Define parent/sibling backoff with a global candidate budget; a changed fallback is a labeled adaptation.
- Defer P-II usage heaps unless profiling predicts a benefit; their updates complicate deterministic parallel queries and the paper showed inconsistent gains.

**Exit P4:** add a method to the supported set only if it offers a distinct validated quality/time/memory tradeoff over P2/P3 and existing Mars search.

## 8. Phase P5 — improve existing rate–distortion optimization

Source: [optimal hierarchical partition paper](summaries/Optimal_hierarchical_partitions_for_frac.md).

### P5a: verify the current objective

Mars already searches a full square quadtree and prunes bottom-up. Its costs are frozen entropy-model estimates; domain-coordinate events are priced differently from the real sequential predictor, and actual entropy models adapt while writing.

Consequently, the current solution is optimal only over evaluated choices under its additive surrogate—not all domains, partitions, stream lengths, or final decoded distortion.

- Log estimated versus actual total bits and category costs (partition, modes, coordinates, coefficients, residuals).
- Count the fixed `t_rms=8` warm-up explicitly; do not claim user `t_rms` controls it.
- Compare threshold partitions versus existing RD partitions with the **same retrieval provider** and mode set.
- Begin with modes 0 and 2 (flat/fractal). Reintroduce affine and residual modes as separate ablations after P0.
- Sweep lambda on validation to obtain overlapping curves; retain all raw points.

### P5b: reduce search work without changing the result

Add optional branch-and-bound only when the lower bound is proven valid for the frozen objective, including split cost and all enabled modes. Variance/flatness alone does not bound what a fractal or residual mode can achieve.

Gate: identical objective, leaves, bytes, and decoded pixels versus the unpruned full walk; test lambda zero, ties, borders, mode sets, and threads. If no useful admissible bound exists, keep full traversal and improve caching instead.

### P5c: deliberate quality extensions

Independently test allowing several retrieved domain candidates into RD mode competition rather than passing only the minimum-RMS candidate. Charge extra evaluations and bits. This is not an output-preserving optimization.

Do not call this an exact reproduction of the 1998 HV/BFOS paper. Rectangular HV geometry, explicit split topology, compatible isometries, and versioned serialization remain a separate project. Square-tree pruning is the lower-risk first step; pursue HV only if partition analysis shows a compelling limitation.

**Exit P5:** state the achieved outcome precisely: rate gain at a time cost, or identical output with time savings. No claim of exact byte-optimal coding from additive approximate rate costs.

## 9. Phase P6 — adaptive boundary postprocessing

Source: [adaptive post-processing paper](summaries/Adaptive_post_processing_for_fractal_ima.md). Suggested new `mars-codec/src/postprocess.rs`; explicit `decmars` option and benchmark payload fields.

1. Start with an out-of-loop edge-aware boundary filter using decoded pixels and actual leaf geometry.
2. Compare off, a clearly labeled reference-style smoother, and the adaptive filter. Original C `smooth_image` is a useful control, not automatically equivalent Rust behavior.
3. Specify mixed-size boundaries, deduplication, corners, processing order, clipping, and constant-image preservation.
4. Keep the encoded stream identical across filter arms; charge filter time separately and within total decode time.
5. Select filter parameters on validation only. Never choose per-image parameters using the original unless those choices are transmitted and counted.
6. Add paper-style in-loop smoothing only as a second experiment. It changes the decoder mapping/fixed point; bound iterations and assess convergence from multiple initializations.
7. Start native-resolution grayscale. Color placement relative to chroma upsampling, progressive prefixes, and zoom each need an explicit later contract.

**Exit P6:** decoded quality/artifact gains at unchanged bytes, acceptable edge/detail regressions, measured decode overhead. Keep off as the default until results support promotion. Verify exact PDF constants before claiming faithful reproduction.

## 10. Phase P7 — conditional sparse multi-domain coding

Source: [fast sparse fractal paper](summaries/Fast_sparse_fractal_image_compression.md). This is a new coding model, not another retrieval switch or the existing DCT residual mode.

- Start with at most two domain terms: `prediction = b + sum(a_i * transformed_domain_i)`.
- Reuse APCC retrieval for residual searches; add joint least-squares/OMP fitting with domain–domain cross terms.
- Handle correlated/singular candidates, deterministic atom ordering, duplicates, zero coefficients, and stopping criteria.
- Quantize jointly, reconstruct the quantized prediction, and count every domain ID, orientation, coefficient, support count, and offset.
- Enforce a conservative post-quantization condition such as `sum(abs(a_i)) <= rho < 1`. Individual coefficients below one do not guarantee contraction of their sum. Integer rounding still requires cycle/stopping diagnostics.
- Define a new versioned leaf mode and stream syntax; preserve old decoding. Update writer, parser, rate model, decoder, fixtures, and color wrapper handling.
- Progressive serialization is separate: implement/version it explicitly or reject sparse streams in that path.
- Compare against single-domain coding and corrected DCT-residual coding at matched full-stream bitrate. More atoms at the same block size is not a fair quality-only win.
- Only after two-domain results justify it, test up to four atoms and paper-derived retrieval budgets.

**Exit P7:** useful BD-rate improvement survives storage, encoding, decoding, memory, and convergence costs. Otherwise retain results as a negative experiment rather than carrying an unhelpful production format.

## 11. Benchmark design

### 11.1 Datasets and leakage

- **Correctness:** committed Tiny64/golden streams; generated flat, gradient, checkerboard, noise, edge/rotation fixtures; odd dimensions and invalid geometry cases. Pin generator seeds and hashes.
- **Development:** existing kodim01/kodim02 are historically used for learned training/evaluation and must not be described as fresh blind tests.
- **Training/validation:** create explicit image-level manifests before APCC or threshold tuning. Prefer separately licensed training data and reserve Kodak for final reporting. If initially limited to Kodak, freeze a documented split and label whole-Kodak results development-influenced.
- **Headline standard set:** all 24 Kodak images with pinned grayscale conversion. Existing raw images are not present in the planning checkout and must be fetched/generated.
- **Generalization:** separately hashed textures and larger photographs with documented licenses, no overlap with tuning data. Existing CLIC/USC-SIPI mentions are plans, not downloaded benchmark sets.
- All block crops from an image belong to that image's split. Pin APCC reference artifacts per class/size. Account for training/model storage separately; encoder-only reference data need not inflate decoder stream size.

### 11.2 Compare in three layers

| Layer | Fixed factors | Measured outcome |
|---|---|---|
| A: retrieval | Identical sampled blocks, sizes, domain pool, quantizers, orientations | Shortlist coverage, winner quality, excess SSE/RMS, exact fits, retrieval/index cost, memory |
| B: whole codec | Same threshold or lambda grid, mode set, stride, decoder policy | Actual bpp/PSNR/SSIM curves, total encode/decode time, leaf distribution |
| C: production profile | Frozen selected settings per method | Pareto comparison against best existing Mars configuration and external anchors |

Layer A must use a common block population, not method-dependent output leaves. Existing recall reports membership of the selected tuple in oracle top-k; add candidate-set coverage separately. Report sample count, zero-error cases, tied optima, and unavailable coverage.

Current oracle is GPU f32 top-32 and starts at block size 8. Use CPU f64 exhaustive fitting as the ranking reference on bounded common samples, cross-check GPU near ties, and avoid describing GPU output as unconditionally exact. Full-grid size-4 oracle builds can be prohibitively expensive; use a frozen stratified size-4 sample with disclosed coverage first.

### 11.3 Limit experiment growth

Do not run every method × budget × stride × mode × lambda × seed × thread count immediately.

1. Smoke correctness on tiny fixtures.
2. Screen all nine existing search methods and new methods on bounded common block samples.
3. Tune budgets on validation; retain at most two Pareto configurations per method.
4. Whole-image development comparisons: exhaustive, best existing deterministic method(s), random, APCC; P4 alternatives only if promising.
5. Freeze finalists and run full standard/extended sets.
6. Repeat final timing configurations, not every discarded tuning point.

Initial controlled profile: grayscale; min/max 8/16; shift 4; 4-bit contrast, 7-bit offset, max contrast 1.0; eight isometries; modes 0/2 for RD; adaptive density off; filter off. Threshold starting grid: 4, 6, 8, 12, 16. Lambda starting grid: 25, 50, 100, 200, 400, 800, 1600, 3200. These are proposed sweeps, not equal-bitrate settings.

Reproduce current default behavior first, then vary size 4, max size 32, and stride 8 separately. Add color, more modes, and density only after search effects are understood.

Record a manifest-derived case count and wall-time estimate from a small pilot before a large run. Set per-case and per-stage timeouts; preserve incomplete evidence and do not rank methods on incompatible completed subsets.

### 11.4 RD analysis

- Compute per-image curves and per-image BD results, then aggregate with coverage and dispersion; do not form an average-image curve first.
- Reuse the existing minimum-four-point, strict-monotonicity, PCHIP/no-extrapolation rules. Combined BD-rate/BD-PSNR currently needs both rate and PSNR overlap.
- Preserve nonmonotonic raw curves. Do not silently discard bad points to manufacture a score; report invalid comparisons or apply a separately specified, preregistered operating-point selection policy to every arm.
- Record PSNR and bpp integration intervals and excluded-image reasons. Use a common interval across arms when making a combined leaderboard; pairwise different intervals are not directly interchangeable.
- Report per-image results, mean/median BD-rate, worst cases, and image-level bootstrap confidence intervals. Aggregate stochastic results hierarchically by image and seed, not by treating all blocks as independent images.
- Report quality regret at matched bitrate and speed at disclosed operating points. Same lambda or RMS threshold is not matched rate.

### 11.5 Timing and resource protocol

Follow the repo's M4 intent, not historical indicative sweep times:

- Release build, locked dependencies, explicit compiler/flags; record Apple chip variant, OS, physical/P/E cores, and GPU details if relevant.
- Foreground, controlled machine, warm file cache, one untimed warm-up, at least five measured repetitions.
- Interleave arms in balanced order; fix algorithm seed within timing repeats. Random-seed variability is a separate axis.
- Run single-thread and explicitly chosen P-core-count thread profiles. Thread count does not guarantee core affinity; record QoS/scheduling assumptions honestly.
- Do not run competing benchmark jobs, training, or oracle generation concurrently during timing.
- Report median/MAD, absolute milliseconds, seconds/megapixel, encode/decode separately. Flag MAD/median >5% and investigate instead of claiming a precise speedup.
- Primary encode timing includes input preparation, contraction, indexing, warm-up, search, partitioning, serialization, and file output under declared boundaries. Also expose in-memory kernel timing as a secondary diagnostic.
- Primary decode includes parse/setup/iterations/filter/output. Metrics calculation is outside encode/decode timing.
- Peak memory must include indexes and caches; separate model training and oracle build time from per-image runtime.
- Turn tracing/diagnostic logs off for timing. Preserve raw samples and process failures.

Use fixed 10-iteration decoding initially for controlled comparisons, then a separate convergence study at 5/10/20/40 iterations and multiple initializations on representative cases. Label the headline iteration policy and any unconverged results; do not imply ten steps guarantee convergence.

### 11.6 External anchors

Reuse `anchors.rs` and `configs/anchors.json`; pin actual encoder/decoder versions and effort/thread settings. Verify CLI compatibility with installed tools.

- First run grayscale anchors on the identical grayscale inputs. Later run color comparisons with disclosed 4:4:4/4:2:0 handling and identical metric definitions.
- Include JPEG and JPEG 2000 as the initial context; retain WebP/AVIF/JPEG XL where installed and reproducible.
- Compare complete files and equivalent input precision/color. Never compare grayscale Mars bits against RGB-source ratios.
- Publish anchor curves even when fractal coding loses. The objective is a useful Mars implementation, not a presumption it beats modern codecs.

## 12. Proposed acceptance criteria

These are **prospective engineering targets**, not predictions or paper results. Freeze them before final evaluation; tune algorithms on validation, not thresholds on final results. Correctness failure blocks merging; missing a performance target is a valid scientific result.

| Change | Mandatory correctness gate | Proposed promotion target |
|---|---|---|
| Shared search integration | Exhaustive default byte/pixel identity; deterministic threads; full counters | No >5% median end-to-end slowdown after accounting for noise; otherwise profile before promotion |
| Random search | Legal unique bounded samples; seed determinism; full-budget equivalence | ≥5× encode speedup over corrected exhaustive with ≤0.5 dB mean matched-rate PSNR loss; retain as control even if target missed |
| APCC / P4 alternative | Finite stable keys, quantized refit, complete fallback accounting | Versus best existing deterministic arm: ≥20% encode-time reduction at ≤+2% mean BD-rate, or ≥3% BD-rate improvement at ≤20% encode-time cost |
| RD output-preserving optimization | Same objective, leaves, bytes, and decoded pixels | ≥15% total encode reduction on representative full-image RD workloads |
| Expanded RD candidate competition | Round-trip and correct complete rate accounting | ≥3% mean BD-rate improvement at ≤25% encode-time cost |
| Out-of-loop filter | Identical coded stream; constants/edges tests; deterministic | ≥0.2 dB mean PSNR gain at unchanged bitrate, no mean SSIM loss, ≤15% total decode-time increase |
| Sparse multi-domain | Versioned round-trip, legal support, post-quantization contraction, bounded decoding | ≥5% mean BD-rate improvement versus strongest single-domain/DCT-residual arm at ≤2× encode/decode time |

For each promotion show worst-image behavior. Proposed guardrails: no unexplained >1 dB matched-rate loss for approximate search, >0.2 dB loss for the filter, or >10% per-image rate regression for the sparse extension. Report exceptions rather than hiding them in averages. Apply targets only where valid curve overlap exists; no overlap is inconclusive, not a pass.

## 13. Delivery breakdown and dependencies

| Increment | Main write scope | Deliverable |
|---|---|---|
| 0a | `mars-codec` focused tests and isolated fixes | Validated baseline semantics; unchanged legacy fixtures or explicit compatibility decision |
| 0b | `mars-bench`, benchmark CLI, experiment configs | Strict serialized-output runner, schema, completeness checks, smoke report |
| 1 | Codec search interface, `mars-search` adapter, `encmars` | Shared production search for threshold/RD paths |
| 2 | `mars-search/random.rs` and tests | Budgeted random baseline and reproducibility tests |
| 3 | APCC retrieval/training utility and tests | Versioned reference blocks; APCC validation report |
| 4 | Binary/hierarchical modules | Optional challenger reports; no automatic default changes |
| 5 | `encode.rs`, `rate.rs`, RD benchmark tests | Audited objective, retrieval-driven RD, optionally exact-result pruning |
| 6 | Postprocessor, decoder option, metrics experiments | Fixed-stream quality/time ablation |
| 7 | Sparse model/format/decoder/rate integration | Conditional versioned experimental codec and RD report |
| 8 | Result store/reporting and docs | Reproducible final leaderboard, limitations, promote/defer decisions |

0a and 0b can be delegated independently only with disjoint file ownership. Random, APCC, and P4 work can proceed in parallel after the shared interface is frozen; one integration owner controls common enums/CLI files. P5 and P6 can proceed independently after P0/P1. Sparse coding follows successful APCC and corrected RD/mode accounting.

Use the repository's contract/format-change review conventions; do not loosen existing tolerances to make new gates pass. New Rust APIs should be documented, use safe code by default, and preserve target-specific SIMD assumptions explicitly. Do not bundle an unrelated edition migration with algorithm benchmarks; the inspected workspace currently declares edition 2021.

## 14. Execution commands and artifacts

### Existing commands: source-verified, not all executed

Run from a working Mars checkout on supported Apple Silicon hardware:

```sh
cargo build --locked --release -p mars-cli
cargo test --locked -p mars-codec
cargo test --locked -p mars-search
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
```

Corpus/reference preparation uses existing recipes; network downloads and Python/external codec prerequisites must be available:

```sh
just corpus
just corpus-gray
just corpus-gray-check
just mars1
```

Small existing round-trip (creates output under `target/`; build first):

```sh
./target/release/encmars fixtures/mars1/tiny64.raw target/random-plan-smoke.mars --raw-width 64 --raw-height 64 --method fisher --threads 1 --t-rms 8
./target/release/decmars target/random-plan-smoke.mars target/random-plan-smoke.pgm --iterations 10
./target/release/marsbench metrics fixtures/mars1/tiny64.raw target/random-plan-smoke.pgm --raw-dims 64x64 --coded target/random-plan-smoke.mars
```

The filename is arbitrary; this smoke command uses **existing Fisher**, not the proposed random method. For the existing RD path replace `--method fisher --t-rms 8` with `--lambda 200 --modes 0,2`; do not combine method and lambda before P1. Tiny64 cannot supply the pinned five-scale MS-SSIM and is not scientific corpus evidence.

### Proposed commands: implement in P0/P1 before use

```text
marsbench experiment --config configs/research-smoke.json --stage smoke
marsbench experiment --config configs/research-search.json --stage retrieval
marsbench experiment --config configs/research-rd.json --stage rd
marsbench experiment --config configs/research-final.json --stage timing
marsbench experiment-report --experiment-id <recorded-id>
```

These command names/config files do not exist at the inspected revision. Define resumability, stage selection, artifact locations, preflight, and failure exits as part of their CLI contract. Avoid a misleading `cargo bench` label: the deliverable is a full codec experiment runner, not only microbenchmarks.

### Required final artifacts

- Frozen configs and train/validation/test manifests; model/reference hashes and reproduction instructions.
- Versioned append-only JSONL with every attempted case and raw timing repetition.
- Coded/decoded artifact hashes and retained representative streams/images; keep large generated files outside Git according to existing policy.
- Per-image RD curves, interval-qualified BD tables, matched-operating-point timing, memory and convergence tables.
- Search diagnostic report: sample coverage, candidate recall/regret, fits, index/query/warm-up breakdown.
- Ablation report separating search, partition, modes, density, and filtering.
- Final decision per method: promote, optional niche, inconclusive, or reject, with measured evidence and limitations.

## 15. Bottom line

The implementation goal is not to reproduce the earlier shortlist in isolation. Mars already has stronger infrastructure and more alternatives than that shortlist assumed.

**First deliver a trustworthy shared-path benchmark, then random search, then APCC.** Reuse the existing quadtree RD optimizer, add filtering as an independent low-risk experiment, and defer new sparse syntax until a measured benefit warrants it. A negative result against Saupe/Fisher/Funnel or existing modes is useful progress; an impressive number from mismatched bitrates or an unparsed stream is not.
