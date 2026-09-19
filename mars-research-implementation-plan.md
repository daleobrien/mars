# Mars: research implementation and benchmarking plan

## 1. Scope and recommendation

**Source reviewed:** local checkout at commit `250af5bc184ac4998297904ae9b32567431f432a`, 18 September 2026. The original planning audit targeted parent `5fc80de5e28388728915fbcd24ca4843eaf94284` on 17 September. Paths are repository-relative; crate names in write-scope tables refer to `crates/<name>/`. Historical measurements retain their recorded build provenance; reviewing them at HEAD does not make them clean-HEAD runs.

This revision incorporates the residual-quantisation experiment and changed `encmars` defaults. It distinguishes implemented infrastructure from uncompleted research gates. No new corpus benchmark was run for this review; focused validation is listed in §3.

**Research sources:** PDFs are now available under `Research/`, with summaries under `Research/summaries/`. The seven phase-relevant summaries were read for this update; paper-specific formulas and reproduction claims still require verification against the linked PDFs. The imported research is not part of the cited HEAD commit. This updated root-level plan is the codebase review; `Research/mars-research-implementation-plan.md` remains the older imported planning copy.

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

## Execution log — implementation started 18 September 2026

The source audit below describes `250af5b`; this log records subsequent implementation steps from `37e5e9c`. Historical benchmark numbers are not reattributed to the corrected encoder.

### Step 1 — residual orientation and distortion (P0.2)

- Reproduced the asymmetric residual regression (isometry 4 failed before correction).
- Corrected **encoding**, not decoding: DCT residuals are domain-local `[u,v]`, matching the existing decoder. Existing streams retain their decoded semantics; newly encoded mode-3 coefficients and RD decisions may change. No format/version change.
- Mode-3 distortion now includes decoder-equivalent clipping and truncation, evaluated at the source-domain prediction state. This remains collage distortion, not converged-image distortion.
- Added an independent inverse-isometry oracle covering all eight orientations, serialization/parsing, two fixed and two adaptive qsteps; added unit coverage comparing scored distortion with reconstructed pixels.
- Validation: residual orientation (1), qstep integration (6), residual unit tests (5) passed. A broader codec run passed 49 unit tests and two encoder tests, but timed out during the 512×512 parallel-determinism test; no full-suite or corpus performance pass is claimed.
- Next: serialized measurement, CLI guards, tiny/odd geometry, and file-to-file experiments. Mode-3 orientation is no longer an untested blocker; corrected residual RD benefit still needs new measurements.

### Step 2 — serialized benchmark reconstruction (P0.1)

- `rd_opt`, `mode_gate`, and `density_gate` now share a parser-backed measurement path: reported quality uses the full parsed header and leaves; density partition statistics use parsed leaves too.
- Wire-step/payload differential tests prove serialized metadata affects quality; parse errors cannot fall back to encoder leaves.
- Validation: three integration tests and seven targeted benchmark unit tests passed.
- This repairs the in-memory inner-stream bypass, not whole-container file-to-file measurement. Historical results remain historical; do not silently replace them.

### Step 3 — explicit encoder CLI capabilities (P0)

- `encmars` validates finite nonnegative lambda/RMS, legal power-of-two block sizes, even representable stride, coefficient bit widths, and header-representable contrast before image I/O.
- Explicit mode masks and adaptive density are rejected on legacy/method paths; adaptive residual requires mode 3. Defaults and explicit lambda overriding thresholds are preserved.
- Corrected CLI help and README: RD warm-up is fixed at RMS 8. README now describes the implemented codec rather than an empty crate.
- Validation: encoder unit tests plus research-options, residual-qstep and classical CLI integration tests passed (17 tests). No corpus performance claim.

### Step 4 — tiny/odd geometry and indexed search (P0.2)

- Reproduced and fixed singleton contracted-plane chunking, DC decoder reads from nonexistent domains, and incorrectly masked size-1 border brightness. Zero-alpha residuals still retain their orientation and coefficients.
- Oversized indexed domain pools are empty; smaller boundary sizes are indexed; empty mass-center queries terminate; flat Saupe descriptors are finite and KD-tree equal-key sorting is stable.
- Validation: seven tiny codec regressions and nine all-method search geometry tests passed, including singletons, odd dimensions, color 8×8 4:2:0, serialization and one/two-thread identity. Existing codec golden encoder/decoder checks also passed.
- Ordinary decoded stream semantics are unchanged where decoding previously succeeded; newly encoded size-1 border DC values intentionally change. Candidate scan order remains row-major. No format change.

### Step 5 — usable decoder controls and output failures

- Added `progressive::decode_with_iterations`; existing `decode` retains ten iterations. `decmars --iterations` now controls every progressive layer rather than being ignored.
- Decoder CLI rejects zero iterations, nonfinite/nonpositive zoom, and explicit threshold without auto. Requests beyond an ordinary available progressive prefix return a clear error.
- Progression frame write failures now fail the command; regular `--debug-rects` actually writes the existing quadtree visualization.
- Validation: six decoder CLI tests, three progressive iteration tests, three existing progressive CLI tests and five residual CLI tests passed. Manual Tiny64 progressive encode → decode at 1/10 iterations → file metrics succeeded (75 bytes, 48.7107 dB at 10 iterations; smoke only, not corpus evidence).
- `cargo test --locked --offline -p mars-cli` timed out at 120 seconds in `cli_a_gate::omitting_adaptive_density_matches_explicit_false_byte_for_byte`, which runs two full-Kodak exhaustive four-mode encodes. It is not a full-suite pass; no assertions or corpus fixture were weakened to bypass it.

### Step 6 — real file-to-file smoke runner (partial P0.3)

- Added `marsbench experiment-smoke`: runs the actual encoder/decoder, writes complete MARC and PNG files, then uses the existing file-based metrics implementation.
- New artifact directory only; persisted report, command arguments, binary/input/stream/decoded SHA256, process wall times, provenance, stdout/stderr logs and explicit failures/timeouts. Timeout kills/reaps the direct codec process; no arbitrary descendant-process or metrics timeout guarantee.
- Restricted initial profile: lambda 200, modes0/2, fixed8, native resolution, one case. Not the proposed resumable sweep/repeated-timing runner and not a promotion benchmark.
- Validation: four harness tests and two CLI integration tests passed (raw/PGM/RGB PNG, missing prerequisites, no overwrite, failed/timed-out cases). Manual Tiny64 smoke completed with report under `target/research-file-smoke-01/` (local generated artifact).

### Step 7 — shared production search (P1 implementation)

- Added codec-owned `SearchProvider`, full fitted candidates/moments, `EncodeOutcome`, and separately labeled search/warm-up counters. Default exhaustive bytes match four pre-refactor pinned stream hashes; old tuple APIs retain historical search-only eval counts.
- `mars-search::IndexedSearchProvider` owns immutable per-size indexes, including doubled-stride density indexes. All nine methods can now use the production grayscale threshold/RD walk. RD warm-up stays exhaustive at RMS 8 and is counted, not silently omitted.
- `encmars --method fisher --lambda 200 --modes 0,2` is now supported; method alone retains threshold 8. RD masks, density and adaptive residual compose with method selection. Color/method and progressive/method combinations remain unsupported.
- Explicit mode masks are checked against final leaves before output: impossible masks (no usable fractal candidate, DC border) fail with an actionable request to enable mode 0. No hidden exhaustive fallback or out-of-mask success. Default codec/DC fallback behavior remains unchanged.
- Validation: pinned provider tests, all-method fitted-moment/stream/thread comparisons, tiny/odd geometry, CLI method matrix, strict-mask rejection, default byte identity and residual round-trips passed. Fisher adaptive residual support uses explicit modes0/3, retaining a nonzero-residual assertion.
- Targeted production-library/binary Clippy found the existing `decmars::image_writer` type-complexity lint; no codec/search compile errors. Full end-to-end speed/memory/repeated timing acceptance remains unmeasured; P1 performance promotion is not claimed.

### Step 8 — bounded seeded random search (P2 implementation)

- Added opt-in `encmars --method random --budget K --seed N` through the same production threshold/RD provider. Budget counts unique domain positions, each fitted under all eight isometries; no early stop or exhaustive fallback. Explicit positive budget required; seed defaults to zero.
- Query-local, versioned grayscale FNV-1a identity + SplitMix64 + unbiased sparse partial Fisher–Yates; candidate sets are nested across budgets and sorted into reference order. Sampling expected O(K), sorting O(K log K), temporary memory O(K); owned legal-position indexes are built once per image/size/stride.
- Full-budget output matches exhaustive bytes, including RD, density, residual modes and thread counts. Random remains grayscale-only and nondefault. RD warm-up remains exhaustive and **outside the per-query random budget**, included in separate counters and wall time.
- Validation: seven library integration tests and four CLI tests passed; pinned RNG/hash/sample vectors, no duplicates, query-order independence, thread identity, different-seed outputs, empty/tiny pools and full-budget equivalence. Targeted Clippy and release CLI build passed.
- P2 scientific exit is still open: no frozen validation Pareto frontier, full-stream corpus RD curves, or promotion result yet. The general experiment runner still needs method/budget/seed sweep support.
- **Bounded smoke (not a promotion benchmark):** kodim01 top-left 256×256 crop, lambda 200, modes0/2, threads 4, 3 repetitions, release build, decoded-file PSNR. Exhaustive 2.749 s / 8080 B / 22.41 dB; random K=64 1.425 s / 7872 B / 20.46 dB; random K=256 1.532 s / 7996 B / 21.32 dB. Not matched-rate, one image, no interval statistics. Fit counters show the fixed exhaustive RD warm-up (161,358,144 evals) is ~98% of random64's total fits, so warm-up — not retrieval — now bounds further end-to-end speedup; selected-provider warm-up remains the labeled follow-up experiment. At these budgets the PSNR loss exceeds the plan's proposed ≤0.5 dB matched-rate guardrail; random stays a control, not a promoted default.

### Step 9 — smoke runner method passthrough

- `marsbench experiment-smoke` accepts optional `--method`/`--budget`/`--seed`, validated against the ten encoder keys; omitted options preserve the original default bytes and report shape. Random requires explicit positive budget; budget/seed without random are rejected with a persisted report.
- Validation: five harness and three CLI smoke tests passed, including unchanged default argv/bytes, a real random file-to-file case, and invalid-option persistence. Targeted Clippy passed after replacing the pre-existing `decmars::image_writer` type-complexity lint with a `ImageWriter` alias (decoder CLI tests unchanged).

### Step 10 — APCC bucket-indexed retrieval (P3 implementation)

- Added opt-in `encmars --method apcc --budget K`: Fisher-style class canonicalization, per-bucket domains sorted by |Pearson correlation| against a fixed train-free reference, lower-bound binary search with a wrapped bounded neighborhood, isometry composition via the existing mapping table, and refitting through the shared quantized fitter (real moments, 8 evals per inspected domain).
- Deterministic: documented key formula, ascending-key then `(row,col)` tie-break, no NaN (constant blocks excluded; constant ranges yield no candidates and the codec's DC fallback applies). No exhaustive fallback; full budget is exhaustive **within the matching bucket**, not the whole pool.
- Validation: six library tests (pinned ordering, ties, nesting, flat/negative-correlation, threads, production threshold/RD/density/mode3/tiny/odd) and five CLI tests (option validation, thread determinism, budget sensitivity, bucket-exhaustive byte identity on a single-domain fixture, density+mode3+adaptive-residual round-trip) passed; targeted clippy clean.
- Not paper-faithful: no offline-trained reference blocks, no signed-contrast reproduction, no paper speedup claims. P3 scientific exit (Pareto comparison vs best existing deterministic method) remains open.

### Step 11 — out-of-loop boundary smoothing (P6 first cut)

- Added `mars-codec::postprocess::smooth_boundaries(decoded, leaf geometry)`: edge-aware cross-boundary blending with fixed weight `0.5·max(0, 1−g/64)²`; constant images and strong edges preserved exactly, corners deduplicated, borders untouched, deterministic, input immutable.
- `decmars --smooth` applies it after a regular grayscale decode; progressive/color/unknown containers are refused clearly. Encoded streams are untouched (hash-verified in tests).
- Synthetic evidence: 8×8 blocking fixture 22.067→22.408 dB; natural ramp unchanged (42.110 dB). Fifteen codec/CLI tests passed; targeted clippy clean.
- Limitations: grayscale only, out-of-loop only, fixed constants (not the paper's), synthetic evidence only — no corpus BD/PSNR promotion claim yet.

### Step 12 — strict unified experiment runner (P0.3 completion)

- Added the config-driven, resumable experiment runner: `crates/mars-bench/src/experiment_config.rs` (schema, strict validation, corpus/input resolution, deterministic planning), `experiment_report.rs` (versioned `experiment` JSONL payload, status tally, Markdown report), and `experiment.rs::run_experiment`.
- CLI: `marsbench experiment --config <path> [--stage <name>] [--out-dir <dir>] [--root <dir>] [--encmars/--decmars] [--limit N]` and `marsbench experiment-report --store <path>|--experiment-id <id> [--reference-group <label>] [--markdown-out <path>]`. Shipped configs: `configs/research-{smoke,search,rd,final}.json`. Arms carry an optional `group` so an RD curve can be built across a swept axis.
- Strictness: a corpus index's `count` must equal its image list, and every referenced input/recorded hash and codec binary is checked before any work. Statically unsupported arm combinations (method+progressive, budget/seed misuse, modes on the legacy path, adaptive-residual without mode 3, smooth+progressive, `decoder.layer` without progressive arms) are rejected together, never accepted as no-ops. Data-dependent refusals (`--method`/`--progressive`/`--smooth` on colour input) are recorded as explicit `unsupported` cases, so they still count in the denominator.
- Accounting: one append-only row per planned case, appended as each completes, carrying case/config identity, the exact encode/decode argv, codec/arm/decoder parameters, container type, whole-container bytes and SHA-256 for input/stream/decoded, raw per-repetition timing samples, the encoder's own summary line plus parsed eval/transform counters, and either metrics or an explicit status (`succeeded`/`failed`/`timed_out`/`missing_prerequisite`/`unsupported`). Any non-success makes the command exit non-zero.
- Resume: skips only an already-completed case whose config hash and build identity (git sha, dirty-patch hash, `Cargo.lock` hash, rustc version) match; a completed case recorded under a different build is a hard error naming the store, and failed/timed-out cases are re-attempted into a fresh `attempt-N` directory so nothing is overwritten.
- Not exposed by the subprocess runner, and recorded as an unavailable diagnostic rather than a zero: per-block candidate coverage/regret and leaf/mode histograms (`diagnostics_unavailable` points at `marsbench recall` for oracle-based coverage).
- Validation: 90 `mars-bench` unit tests (config/plan/report/tally/store round-trip), 6 runner integration tests with native stub codecs (resume, failed-case retry, unsupported, timeout, preflight failures, repeated-repetition identity), the existing 5+3 smoke tests, and 2 CLI end-to-end tests against the shipped smoke config driving the real encmars/decmars (run → resume → report). Manual `research-smoke` run on `tiny64`: 48.711 dB, 0.117 bpp, one row; the second invocation resumed the case (`resumed: 1`, still one row). Targeted Clippy clean for the changed crates; the pre-existing `encmars::items_after_test_module` lint blocks a `mars-cli --bins --tests` sweep and is unrelated to this step.
- Limitations: one stage per invocation; subprocess-based, so no per-block search internals; no oracle-cache requirement or coverage computation yet; the `research-search`/`research-rd`/`research-final` corpus sweeps are opt-in definitions and were not run here. The runner P0.3 asked for now exists; the corpus/science exits that depend on it remain open.

### Step 13 — P5a: measure the RD objective's own error

- Added `mars-entropy::encode_with_bits`: the same recording pass `encode` already ran, additionally reporting each event's real information content (`-log2 p`) under the **live, evolving** models. Byte-identical output to `encode` on the same events (asserted); the extra vector is a query, never a mutation.
- Added `mars-codec::audit` + `encode::audit_rd`: one partition priced twice — by the frozen `t_rms = 8` warm-up snapshot the RD search decides with (`estimated`), and by the live models the stream is really coded with (`actual`) — with both attributed to P5a's five categories by field id (`partition`/`modes`/`coordinates`/`coefficients`/`residuals`). Unrecognised field ids are counted, never dropped. Container overhead (header, section table, alignment, coder final state) is reported separately rather than folded in.
- Added `mars-bench::rate_audit` + `marsbench rate-audit`: a partition grid over an image set, one append-only row per case (full category breakdown, both bit totals, quality, provenance), and a deterministic Markdown report. Both partition families are priced against the *identical* snapshot, which is what makes P5a's "same retrieval provider and mode set" comparison meaningful; a threshold row's estimate is labelled `reference_only` because that walk never consulted it.
- No encode behaviour changed. The only edit to the encoder was extracting the existing header construction into `build_header` so the audit cannot disagree with it; the pinned byte-identity, all-method and thread-determinism tests still pass.
- **Measurement** (`target/p5a/results.jsonl`, report `target/p5a/report.md`; 2 Kodak grayscale images, modes 0/2, fixed residual step 8, no search-method override, ~5m37s foreground):

| image | partition | parameter | leaves | bpp | PSNR-Y dB | estimated bits | actual bits | error |
|---|---|---|---:|---:|---:|---:|---:|---:|
| kodim01 | RD | λ=200 | 7737 | 0.53809 | 26.5104 | 225083.4 | 211240.7 | +6.553% |
| kodim01 | RD | λ=800 | 2442 | 0.16781 | 23.1354 | 79307.9 | 65645.3 | +20.813% |
| kodim01 | RD | λ=3200 | 1548 | 0.08610 | 21.5172 | 48153.9 | 33512.3 | +43.690% |
| kodim02 | RD | λ=200 | 2310 | 0.14119 | 30.6731 | 61695.6 | 55162.6 | +11.843% |
| kodim02 | RD | λ=800 | 1602 | 0.06706 | 29.0229 | 40456.3 | 26002.0 | +55.589% |
| kodim02 | RD | λ=3200 | 1542 | 0.04093 | 28.1497 | 36775.7 | 15743.0 | +133.599% |
| kodim01 | threshold | t_rms=8 | 18186 | 1.31527 | 30.2127 | 511946.4 | 516847.9 | −0.948% |
| kodim02 | threshold | t_rms=8 | 5835 | 0.43172 | 33.3018 | 160514.8 | 169396.9 | −5.243% |

- **The error is category-localized, not diffuse.** Across every RD row, `coordinates` is priced within −10%..+5% and `coefficients` within +5%..+47%, while `partition` (split flags) ranges +51%..+9507% and `modes` +203%..+1318%. On kodim01 λ=3200 those two fields are 24% of the estimated rate but only 4.5% of the actual payload (11645 of 48154 estimated bits vs 1521 of 33512 actual).
- **The bias has a direction.** The snapshot's statistics come from a *fine* `t_rms = 8` partition, where a size class's split flag is mostly `1`; a coarse partition emits mostly `0` at the few nodes it still considers, and the snapshot prices that rare symbol at roughly `log2` of a small probability while the live model rapidly makes it nearly free. The surrogate therefore over-penalizes *not* splitting, i.e. it is biased toward more splitting than a live rate model would justify. `threshold` partitions, whose statistics match the warm-up by construction, are priced almost exactly (−0.948%, −5.243%) — a clean internal control that the accounting itself is sound.
- **Container overhead is negligible at these rates**: 248–278 bits per stream (header, section table, word alignment, coder final state), under 0.5% of the payload.
- Validation: `mars-entropy` 5 unit tests (including byte-identity and a slack bound tying the summed live costs to the byte-aligned payload); `mars-codec` 54 lib tests including a new `audit_rd` test asserting every event is attributed, the category totals reproduce the totals, exactly two coordinate events per domain-referencing leaf, zero residual fields with mode 3 excluded, and that the threshold branch really responds to `t_rms` (an accidentally shared RD path would ignore it); `mars-bench` 5 `rate_audit` unit tests plus 95 lib tests; `mars-cli` 2 end-to-end audit tests against the pinned `tiny64` fixture (real encode, decode, measure; append-only across two runs; invalid `--modes` and `--dims`-without-`--input` refused before anything is written). Targeted Clippy clean for every changed crate and target; changed files rustfmt-clean.
- Limits, stated rather than papered over: λ=50 is **not** measured — the provided `--lambdas 3200,800,200` grid was chosen to bound foreground time (the trend is already monotone in λ, and Step22's own sweep set the precedent of reporting partial grids plainly); two images, one profile, modes 0/2 (so `residuals` is identically zero and says nothing about mode 3); error percentages are meaningless wherever the actual magnitude is near zero (the threshold arm's `modes` field: 1.8 estimated vs 10.3 actual bits out of ~512000). Nothing here is a promotion claim: P5a asked for the objective's error to be measured, and it now is, per category.

## 2. What Mars already implements

README status/layout and warm-up wording were corrected in execution step 3. Use the execution log, source, and [the optimisation status](docs/encmars-optimisation-status.md) to distinguish implemented behavior from historical measurements.

**Research status:** P0 is partially implemented, not complete. Residual-step metadata, focused round-trip tests, and a three-arm serialized-stream experiment exist. A restricted file-to-file smoke and shared production retrieval interface are implemented (steps 6/7); random (step 8) and APCC (step 10) are opt-in providers; the out-of-loop boundary filter is implemented (step 11); the strict config-driven experiment runner is implemented (step 12); P5a's objective audit is implemented and measured (step 13), and it found the surrogate's error to be localized in the partition and mode fields rather than diffuse. The sparse multi-domain format remains proposed. P5 extends an existing RD optimizer; it is not a new optimizer implementation.

| Area | Existing implementation | Consequence for this plan |
|---|---|---|
| Codec | `crates/mars-codec/src/encode.rs`, `ifs.rs`, `mars_format.rs`: quantized fitting, iterative decoding, threshold quadtree, entropy-coded streams | Extend; do not rebuild the baseline. |
| RD partitioning | `encode.rs::walk_rd`, `split_rd`, `best_mode_leaf`; `rate.rs::RateModels` | Bottom-up `D + lambda R` selection already exists. Audit the objective and integrate retrieval first. |
| Modes | Flat, spatial affine, single-domain fractal, fractal plus DCT residual | Keep mode sets controlled in search experiments. Sparse DCT residuals are not multi-domain fractal coding. |
| Residual quantisation | `quant.rs::ResidualQstep`, `encode.rs::ResidualQuantisation`; per-stream step propagated through grayscale, color planes, and progressive output | Fixed8 remains default; lambda-adaptive quantisation is implemented but opt-in and not promoted by the incomplete experiment. Preserve metadata through decoding. |
| Search | `mars-search`: Exhaustive, Fisher, Hurtgen, MassCenter, Saupe, SaupeFisher, McSaupe, Funnel, Learned | Use existing methods as serious challengers, not just historical context. Random (step 8) and APCC (step 10) are now implemented. |
| Search interface | `CandidateRetriever`, `DomainPool`, `RangeBlock`, `search_block`, `SizedRetrievers` | Good starting point, but a separate encoder currently consumes them. |
| Acceleration | `mars-simd` integer moments/NEON; `mars-gpu` search; Rayon paths | Preserve existing exact kernels and deterministic threading; profile before more hardware work. |
| Measurement | `mars-core` metrics; `mars-bench` BD-rate, provenance, JSONL store, oracle/recall, anchors | Reuse these components rather than introduce a second metrics implementation. |
| Additional coding | Color, progressive output, adaptive domain density | Existing compatibility restrictions must become explicit capability checks. |
| HV partitions | `hv-split-plan.md` | Proposal only; square leaf geometry and implicit quadtree syntax still dominate the implementation. |

### Production search integration

**Implemented in execution step 7:** `mars-search → mars-codec` remains the dependency direction. Production `walk`/`walk_rd` consume a codec-owned provider; `IndexedSearchProvider` adapts the existing retrieval methods. The separate `mars_search::encode_image` remains available for diagnostic/legacy callers, but no longer owns CLI method partitioning. Warm-up stays exhaustive and is counted separately.

The CLI now accepts method plus explicit lambda for grayscale RD. Color/method and progressive/method remain rejected; new methods must register a provider and pass production integration tests, not only extend an enum.

### Current defaults and compatibility

- Plain `encmars` now selects RD at lambda 200, modes 0/2, fixed residual step 8, sizes 4–16, stride 4, color 4:4:4, and automatic threads. Density and progressive output remain off. Library/benchmark defaults were not changed to this CLI profile; `EncodeOptions::default()` still allows all four modes.
- Explicit `--t-rms`, `--chroma-t-rms`, or `--method` selects the legacy path unless explicit lambda overrides the thresholds. **RD warm-up remains hard-coded to RMS 8** in `build_rate_snapshot`; user thresholds do not seed that pass. Execution step 3 corrected the CLI help/README without changing this baseline algorithm.
- `--adaptive-residual` requires explicit `--lambda` and now supports `--method` on grayscale RD. It does not enable mode 3: request e.g. `--lambda 200 --modes 0,2,3 --adaptive-residual`. Progressive containers are grayscale-only and incompatible with `--method`; color iteration callbacks are not color progressive-container support.
- `ResidualQstep` validates [1, 65535] and stores binary32 bits. Quantization, scoring, and reconstruction use the wire-rounded step. The adaptive policy is `clamp(sqrt(6 * lambda / ln(2)), 1, 65535)`; fixed8 remains default. Color planes can carry separate steps. Keep the full `mars_format::Header`/`DecodeHeader` when decoding: passing only `.geometry` loses step metadata and implies legacy step 8.
- O7 expanded `MARS` headers from 20 to 24 bytes and `MPRG` from 31 to 35 bytes **without changing version byte 0 or adding a legacy-layout fallback**. Old Mars 2 streams require re-encoding with this decoder; `MARC`'s unchanged outer layout does not make its embedded old streams compatible. Pin layout revision and decoder build as well as version byte. Mars 1 `.ifs` syntax is separate and unchanged; it cannot carry affine/residual modes or explicit residual steps.

### How much to trust existing results

The local inventory contains 10,906 enveloped historical rows (all dirty, run index zero), 16 classical-method rows without that envelope, and 23 Step22 records in `results/step22-o7-1789648127131947000-41238.jsonl` (metadata, 19 samples, two comparisons, one operator-added timeout record). These are not all independent measurements or one schema. Preserve them; establish a new clean, versioned result series.

The Step22/O7 experiment compares modes0/2 fixed8, all-modes fixed8, and all-modes adaptive at lambda 50/200/800/3200 on kodim01/02. It timed out after **19/24 points**. Only kodim01 has complete comparisons against modes0/2:

| Arm | Reported BD-rate | PSNR integration interval |
|---|---:|---|
| All-modes fixed8 | +1.880847% | 21.521209–29.279225 dB |
| All-modes adaptive | +3.342325% | 21.521209–29.010198 dB |

The new two-image mean is unknown; the +1.025% adaptive mean acceptance ceiling remains unresolved. Different intervals prevent subtracting these numbers as an adaptive-versus-fixed8 BD-rate result. Fixed8 and CLI modes0/2 remain the baseline, not evidence of universal optimality.

Step22 records dirty parent `5fc80de`, a tracked patch, embedded harness source, and build metadata—not a clean HEAD run. Its embedded adaptive arm also differs from HEAD's explicit `LambdaAdaptive` selection. Reproduce the current harness before attributing those results to HEAD. Single observations per point do not establish timing speedups. See [O7 status and limitations](docs/encmars-optimisation-status.md).

Existing anchor support already covers JPEG, OpenJPEG, WebP, AVIF, and JPEG XL. The JPEG implementation recorded by the audit is libjpeg-turbo, not automatically mozjpeg. Existing anchor results use color Kodak; many Mars research gates use grayscale. Do not compare these as equivalent workloads.

## 3. Phase P0 — correctness and measurement baseline

**Priority: required before scientific comparisons.**

### P0.1 Make every quality measurement cross the serialization boundary

Update `crates/mars-bench/src/rd_opt.rs::sample`, `mode_gate.rs::sample_with_modes`, `density_gate.rs::sample_with_density`, and new experiment code so the authoritative path is:

```text
input file → encode → serialize → actual coded file
                                  ↓
                         read/parse coded file → decode → decoded file
                                                               ↓
                                     mars-core metrics from input + decoded files
```

**Updated in execution step 2:** all three named helpers now decode the serialized/parsed full header and leaves through `rd_opt::sample_from_bytes`; `mode_gate` explicitly pins fixed8. They remain inner-stream measurements, not CLI whole-file measurements. A previous adaptive-density result was withdrawn after this pattern hid a coordinate-serialization problem. The historically reported corrected density result was about −0.14% BD-rate, not the withdrawn −6.82%; this is not proof that today's gate enforces serialized reconstruction.

`crates/mars-bench/tests/residual_qstep_gate.rs::measure` already serializes, parses, checks headers/leaves, qstep and mode counts, then decodes the **parsed** stream for PSNR. Reuse that pattern. It is still an in-memory stream experiment, not the proposed file-to-file benchmark: no coded/decoded file reload, decoded-image hash, or CLI container/output coverage. Its `encode_elapsed_s` times the encoder call only; `elapsed_s` also includes serialization/parsing, decode, and PSNR, not primary end-to-end encode/decode timing.

- Measure whole-container bytes, not estimated payload or summed leaf costs.
- Keep in-memory reconstruction as a differential diagnostic, never the headline quality source.
- Record bitstream hash, decoded-image hash, format version **and layout/build identity**, container type (`MARS`/`MARC`/`MPRG`), dimensions, and crop rules. Compare like containers; an inner-plane stream omits CLI wrapper bytes.
- Preserve existing golden streams with their decoder provenance. Account explicitly for the already-breaking O7 layout change. Further syntax or decoded-semantics changes require a compatibility decision and regression fixture; do not repeat an indistinguishable version-0 layout change.

### P0.2 Reproduce suspected source-level defects before fixing them

| Concern | Evidence at HEAD | Required regression / decision |
|---|---|---|
| Residual orientation | **Resolved in execution step 1:** encoder stores domain-local residuals to match the unchanged decoder; independent all-eight-isometry serialized regression passes. | Preserve old decode semantics. Re-measure new mode-3 streams; their coefficients and scored distortion can change. |
| Contrast header precision | Fitting still uses `params.max_alfa`; header stores a 1/32-quantized value; decoder uses header value. Residual-step wire rounding is handled, contrast normalization is not. | Test nonrepresentable parameters. Normalize before all fitting/scoring or reject them explicitly. |
| Contractivity | Actual contrast is `qalfa / 2^bits_alfa * (int_max_alfa / 32)`. Default maximum coefficient is 15/16, but other accepted settings can allow ≥1. | Validate dequantized coefficients for the selected profile; distinguish noncontractive legacy settings. Fixed residuals do not increase the continuous map's Lipschitz constant, but integer iterations still need stopping/cycle diagnostics. |
| Domain-grid phase | Density's finer-than-header-grid branch is already removed: only base/doubled stride is used. Independently, contracted samples use even-origin 2×2 averages and lookup divides coordinates by two. | Retain the density serialization regression. Test odd stride/origins against direct decoder sampling; use positive even stride in the baseline until phase-aware sampling is validated. |
| Flat/tied features | Execution step 4 handles zero-energy Saupe features, stable KD-tree equal keys, and bounded empty mass-center search. | All-method constant/tiny deterministic tests pass; extend coverage when introducing new feature/index methods. |
| Empty/tiny pools and decoding | **Resolved for tested profiles in execution step 4:** legal empty pools, smaller boundary indexes, singleton contraction, DC decoding and border quantisation have focused round-trip regressions. | Keep geometry tests across all methods and threads. This is not a claim that arbitrary library parameters or every color/progressive geometry have been validated. |

These are source-audit findings, not all reproduced defects. Fix only after a failing focused test identifies the behavior. Keep fixes in isolated changes, then establish a corrected baseline before comparing algorithms.

**Validation actually performed:** the original parent-revision planning run `cargo test --locked --offline -p mars-codec residual` passed four tests. During this HEAD review:

| Command | Result |
|---|---|
| `cargo test --locked --offline -p mars-codec --test residual_qstep` | 6 passed |
| `cargo test --locked --offline -p mars-codec --lib residual` | 4 passed |
| `cargo test --locked --offline -p mars-bench --release --test residual_qstep_gate arms_and_real_stream_measurement_are_wired -- --exact --nocapture --test-threads=1` | 1 passed |

These establish focused qstep bounds/metadata, malformed-step rejection, round-trips, color/progressive propagation, residual unit behavior, and three-arm synthetic 32×32 measurement wiring. They do **not** establish the all-isometry invariant above, Kodak acceptance, speed, or convergence. No full suite or new headline benchmark was run. Broader historical validation and timeouts are separately reported in [the optimisation status](docs/encmars-optimisation-status.md); routine ignored/opt-in test results are not corpus-gate evidence.

### P0.3 Build a strict unified experiment runner

Suggested new modules: `crates/mars-bench/src/experiment.rs`, `experiment_config.rs`, `experiment_report.rs`; thin CLI wiring in `mars-cli/src/bin/marsbench.rs`. Names are proposals, not existing APIs.

**Implemented in execution step 12:** those three modules and the `marsbench experiment` / `experiment-report` wiring exist, with four shipped `configs/research-*.json`. The runner is config-driven, validates corpus completeness and prerequisites before any work, appends one row per planned case (including `unsupported` ones), resumes only on matching case/config/build hashes, and exits non-zero on any non-succeeded case. The corpus gates built on it — the P2 full-stream RD/time curves, the P3 validation Pareto, and P5a's objective ablation — were not run for this step.

Reuse `store::Row`, `ResultStore`, `measure`, `bdrate`, `provenance`, and reports. Keep legacy row readers working; add a versioned experiment payload. Reuse Step22's parsed-stream checks and per-record flushing, but do not mistake its separate JSONL schema for this runner. Its timeout completion was operator-added after termination; implement runner-owned timeout/failure accounting and explicit interrupted-run recovery.

Each planned case must have a stable identity and persist:

- Source SHA, dirty status, dirty-source patch hash if applicable, lockfile/build identity, compiler/flags, harness version.
- Experiment/config/split/image/oracle/model hashes and artifact paths.
- All codec parameters, enabled modes, partition/search strategy, domain stride, quantizers (including residual policy and actual wire step per plane), seed/RNG version.
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

**Exit P0:** codec round-trip tests pass; defects affecting the selected profile are resolved or excluded explicitly; runner produces reproducible smoke artifacts; missing inputs cause an explicit failure; README status and warm-up documentation are corrected. Existing qstep tests and the incomplete Step22 run satisfy only parts of this gate. Start with modes0/2, fixed8, representable max contrast, and even stride; require the independent residual regression (now passing in execution step 1) before mode3 experiments.

## 4. Phase P1 — one search pipeline for legacy and RD encoding

### Architecture

The codec-owned `SearchProvider` and `mars-search::IndexedSearchProvider` adapter are implemented in execution step 7. Keep the remaining instrumentation/performance gates below; do not add a reverse `mars-codec → mars-search` dependency.

Suggested ownership:

- `mars-codec`: quantized fit, fitted candidate type, search request/outcome contracts, tree/mode decisions, exhaustive reference implementation.
- `mars-search`: immutable indexes, candidate retrieval strategies, adapters to the codec contract.
- `mars-bench`: comparisons, exhaustive/oracle diagnostics, policy sweeps; no fitting reimplementation.

Route `walk`, `walk_rd`, and eventually RD warm-up through the shared provider. First preserve today's **RMS-best candidate followed by mode competition** behavior. Considering multiple domains under the RD objective is a later separate experiment.

### API requirements

- Immutable query indexes usable through `Sync`; separate index construction from querying.
- Explicit block/plane identity, geometry, legal domain stride, candidate-budget units.
- Separate candidate retrieval from exact fitting; permit a fit-aware driver for early stopping and widening.
- Return fitted parameters and raw moments needed by residual modes, not just coordinates. Preserve the codec's full header/quantisation context across adapters, including residual step; do not narrow it to `ifs::Header` geometry.
- Deterministic candidate order/ties; for a changed method define canonical tie IDs `(row,col,isometry)`.
- Bounded, explicitly counted fallback; distinguish no match from a legitimate flat-block result.
- Cache domain moments per position/size and reuse cross terms where valid. Avoid recalculating domain sums for every orientation.
- Cover forced boundary blocks and sizes below the nominal minimum.

### Instrumentation

Report separately: contracted-plane construction, index build, feature queries, exact fits, fallback fits, RD warm-up fits, tree/mode work, serialization, total encode, decoder setup/iterations/postprocessing.

`evals` continues to mean complete domain/isometry fit evaluations. Also count descriptors scored, pixels/dots processed, nodes visited, cache memory, and candidate allocations. Reducing fits does not prove a speedup if feature scanning still traverses the entire pool. Thread-summed internal timers are not wall time.

For controlled search comparisons initially retain the same exhaustive RD warm-up and charge its full cost. Then test selected-provider warm-up as a separate factor, since it changes both time and frozen rate models.

**Exit P1:** default exhaustive path preserves pinned **post-P0/current-layout** bytes on the supported fixtures; each existing method can exercise the production grayscale threshold and RD paths; flags behave explicitly; work includes warm-up; thread-count tests pass. Do not use pre-O7 container bytes as an unqualified identity target.

## 5. Phase P2 — bounded seeded random search

**Implementation delivered in execution step 8.** `crates/mars-search/src/random.rs` implements the full bounded sample (no early-stop variant). CLI registration is intentionally separate from legacy benchmark `MethodName`; general benchmark sweeps still require wiring. Correctness gates below are covered by focused tests; measured RD/time promotion remains open.

Sources: [randomized approach summary](Research/summaries/Fractal_image_compression_a_randomized_a.md) and [PDF](Research/Fractal_image_compression_a_randomized_a.pdf), Ghosh, Mukherjee and Das (2004). The paper uses first-acceptable stopping, fixed 4×4 ranges, one-pixel stride, and empirically calibrated trial caps. The bounded full-sample, even-stride quadtree control below is a deliberate Mars adaptation, not a paper-faithful reproduction. Do not import its published speedup or calibration thresholds as Mars expectations.

Suggested addition: `crates/mars-search/src/random.rs`, configuration/CLI registration, focused search and integration tests.

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

Sources: [APCC summary](Research/summaries/A_Novel_Fractal_Image_Compression_Scheme_With_Block_Classification_and_Sorting_Based_on_Pearsons_Correlation_Coefficient.md) and [PDF](Research/A_Novel_Fractal_Image_Compression_Scheme_With_Block_Classification_and_Sorting_Based_on_Pearsons_Correlation_Coefficient.pdf), Wang and Zheng (2013), DOI 10.1109/TIP.2013.2268977. The summary describes class canonicalization, offline-trained reference blocks, and approximately 2k comparisons from R/−R queries. Count both branches and training cost separately; do not equate one paper window with the total Mars fit budget.

The train-free APCC provider is implemented in execution step 10 (`crates/mars-search/src/apcc.rs`); offline-trained reference blocks and the signed-contrast reproduction below remain future work.

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

Sources: [local-feature summary](Research/summaries/Enhancing_fractal_image_compression_spee.md) and [PDF](Research/Enhancing_fractal_image_compression_spee.pdf), Jaferzadeh, Moon and Gholami (2016). Its fixed grayscale scale factor and unresolved reported bit-accounting discrepancy differ from Mars's quantized variable-contrast model; use it as retrieval inspiration, not a matched-rate baseline.

- Implement the paper's 12-bit perimeter-versus-central-mean descriptor first for 4×4 ranges and matching contracted domains.
- Use XOR/popcount as a separately documented implementation choice instead of a 4096² lookup table.
- Compare fixed Hamming thresholds and the paper's contrast-adaptive policy; define no-candidate fallback.
- Existing Hurtgen provides a useful four-bit classification control, not the same algorithm.
- Treat extensions to 8×8/16×16 as new descriptors and revalidate. Negative contrast and orientation handling must follow Mars's actual model.

### Hierarchical intensity-sum classes

Sources: [hierarchical-classification summary](Research/summaries/Fractal_Image_Compression_using_Hierarch.md) and [PDF](Research/Fractal_Image_Compression_using_Hierarch.pdf), Bhattacharya et al. (2015). The timing table's comparator is FISHER24, despite the abstract's BFIC wording; do not interpret its reported speedup as a measured comparison against Mars exhaustive search.

- Implement P-I first: quadrant and subquadrant rank codes, sparse occupied buckets, deterministic tie policy.
- Do not allocate a dense 24^5 class table per block size.
- Define parent/sibling backoff with a global candidate budget; a changed fallback is a labeled adaptation.
- Defer P-II usage heaps unless profiling predicts a benefit; their updates complicate deterministic parallel queries and the paper showed inconsistent gains.

**Exit P4:** add a method to the supported set only if it offers a distinct validated quality/time/memory tradeoff over P2/P3 and existing Mars search.

## 8. Phase P5 — improve existing rate–distortion optimization

Sources: [optimal hierarchical partition summary](Research/summaries/Optimal_hierarchical_partitions_for_frac.md) and [PDF](Research/Optimal_hierarchical_partitions_for_frac.pdf), Saupe et al. (1998). Its BFOS result is over prunings of a chosen HV tree under collage error, not globally optimal partitions or decoded distortion; the reported implementation omits isometries.

### P5a: verify the current objective

**Implemented and measured in execution step 13.** `marsbench rate-audit` prices one partition twice — against the frozen warm-up snapshot the RD search decides with, and against the live models the stream is really coded with — and attributes both to the five categories below. On two Kodak images the total error was +6.6%..+133.6% and rose with λ, but the error was concentrated in `partition` (split flags) and `modes`; `coordinates` stayed within −10%..+5%. Because the snapshot comes from a fine `t_rms = 8` partition, the surrogate over-penalizes *not* splitting, which biases the search toward finer partitions than a live rate model would justify. λ=50, mode 3 and more images remain unmeasured.

Mars already searches a full square quadtree and prunes bottom-up. Its costs are frozen entropy-model estimates; domain-coordinate events are priced differently from the real sequential predictor, and actual entropy models adapt while writing.

Consequently, the current solution is optimal only over evaluated choices under its additive surrogate—not all domains, partitions, stream lengths, or final decoded distortion.

- Log estimated versus actual total bits and category costs (partition, modes, coordinates, coefficients, residuals).
- Preserve the fixed `t_rms=8` exhaustive warm-up; execution step 7 exposes its evaluations in `EncodeCounters` and the method CLI. Legacy tuple APIs still return search-only evals for compatibility. Expand explicit counters to remaining benchmark/color paths before total-work comparisons.
- Audit residual distortion against quantized decoded predictions, clipping/truncation, and iterative reconstruction. Current residual scoring uses continuous source-domain prediction error; wire-rounded qstep consistency does not make that final-image distortion.
- Keep fixed8 versus lambda-adaptive quantisation as a separate factor with identical allowed modes. Step22's unfinished three-arm result is a starting diagnostic, not a passed residual improvement gate.
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

Sources: [adaptive post-processing summary](Research/summaries/Adaptive_post_processing_for_fractal_ima.md) and [PDF](Research/Adaptive_post_processing_for_fractal_ima.pdf), Giang and Saupe (year/venue unresolved in the summary). Reported gains combine in-loop smoothing and a final edge-adaptive filter; they are not predictions for the out-of-loop-only first stage below. Verify poorly extracted mathematical constants visually in the PDF.

The out-of-loop filter is implemented in execution step 11 (`crates/mars-codec/src/postprocess.rs`, `decmars --smooth`, grayscale regular streams). In-loop smoothing, color/progressive composition, and paper-constant verification remain future work.

1. Start with an out-of-loop edge-aware boundary filter using decoded pixels and actual leaf geometry.
2. Compare off, a clearly labeled reference-style smoother, and the adaptive filter. Original C `smooth_image` is a useful control, not automatically equivalent Rust behavior.
3. Specify mixed-size boundaries, deduplication, corners, processing order, clipping, and constant-image preservation.
4. Keep the encoded stream identical across filter arms; charge filter time separately and within total decode time.
5. Select filter parameters on validation only. Never choose per-image parameters using the original unless those choices are transmitted and counted.
6. Add paper-style in-loop smoothing only as a second experiment. It changes the decoder mapping/fixed point; bound iterations and assess convergence from multiple initializations.
7. Start native-resolution grayscale. Color placement relative to chroma upsampling, progressive prefixes, and zoom each need an explicit later contract.

**Exit P6:** decoded quality/artifact gains at unchanged bytes, acceptable edge/detail regressions, measured decode overhead. Keep off as the default until results support promotion. Verify exact PDF constants before claiming faithful reproduction.

## 10. Phase P7 — conditional sparse multi-domain coding

Sources: [fast sparse fractal summary](Research/summaries/Fast_sparse_fractal_image_compression.md) and [PDF](Research/Fast_sparse_fractal_image_compression.pdf), Wang et al. (2017). This is a new coding model, not another retrieval switch or the existing DCT residual mode. The paper's full-candidate rate–quality results and restricted-search timing settings are different experiments; do not combine them into one claimed operating point.

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
- **Headline standard set:** all 24 Kodak images with pinned grayscale conversion. This local checkout contains 24 PNGs and 24 grayscale raw images; their hashes match `corpus/kodak.manifest.json` and `corpus/standard.images.json`, and raw byte lengths were checked during review. They are ignored/generated artifacts, not supplied by the commit; fresh checkouts still require preparation. The conversion itself was not regenerated in this review.
- **Generalization:** separately hashed textures and larger photographs with documented licenses, no overlap with tuning data. Existing CLIC/USC-SIPI mentions are plans, not downloaded benchmark sets.
- All block crops from an image belong to that image's split. Pin APCC reference artifacts per class/size. Account for training/model storage separately; encoder-only reference data need not inflate decoder stream size.

### 11.2 Compare in three layers

| Layer | Fixed factors | Measured outcome |
|---|---|---|
| A: retrieval | Identical sampled blocks, sizes, domain pool, quantizers, orientations | Shortlist coverage, winner quality, excess SSE/RMS, exact fits, retrieval/index cost, memory |
| B: whole codec | Same threshold or lambda grid, mode set, stride, decoder policy | Actual bpp/PSNR/SSIM curves, total encode/decode time, leaf distribution |
| C: production profile | Frozen selected settings per method | Pareto comparison against best existing Mars configuration and external anchors |

Layer A must use a common block population, not method-dependent output leaves. Existing recall reports membership of the selected tuple in oracle top-k; add candidate-set coverage separately. Report sample count, zero-error cases, tied optima, and unavailable coverage.

There are 72 oracle cache files locally; their presence does not establish freshness or validity, and their contents were not validated during this review. Current oracle is GPU f32 top-32 and starts at block size 8. Use CPU f64 exhaustive fitting as the ranking reference on bounded common samples, cross-check GPU near ties, and avoid describing GPU output as unconditionally exact. Full-grid size-4 oracle builds can be prohibitively expensive; use a frozen stratified size-4 sample with disclosed coverage first.

### 11.3 Limit experiment growth

Do not run every method × budget × stride × mode × lambda × seed × thread count immediately.

1. Smoke correctness on tiny fixtures.
2. Screen all nine existing search methods and new methods on bounded common block samples.
3. Tune budgets on validation; retain at most two Pareto configurations per method.
4. Whole-image development comparisons: exhaustive, best existing deterministic method(s), random, APCC; P4 alternatives only if promising.
5. Freeze finalists and run full standard/extended sets.
6. Repeat final timing configurations, not every discarded tuning point.

Initial controlled profile: grayscale; min/max 8/16; shift 4; 4-bit contrast, 7-bit offset, max contrast 1.0 (maximum dequantized coefficient 15/16); fixed residual step 8; eight isometries; modes 0/2 for RD; adaptive density off; filter off. Threshold starting grid: 4, 6, 8, 12, 16. Lambda starting grid: 25, 50, 100, 200, 400, 800, 1600, 3200. These are proposed sweeps, not equal-bitrate settings.

Pin the current CLI default profile (§2, minimum size 4) separately from this controlled minimum-size-8 research profile. Reproduce both before changing retrieval; vary size 4, max size 32, and stride 8 as labeled factors from the controlled profile. Add color, more modes, adaptive residual quantisation, and density only after search effects and P0 restrictions are understood.

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
| 0a | `mars-codec` focused tests and isolated fixes; status/help docs | Build on existing qstep tests; resolve independent residual/geometry regressions and layout compatibility; correct stale default/warm-up claims |
| 0b | `mars-bench`, benchmark CLI, experiment configs | Build on Step22 parsed-stream checks; repair three legacy gate bypasses; add strict file-to-file runner, schema, completeness/timeout handling, smoke report |
| 1 | Codec search interface, `mars-search` adapter, `encmars` | Shared production search for threshold/RD paths |
| 2 | `mars-search/random.rs` and tests | Budgeted random baseline and reproducibility tests |
| 3 | APCC retrieval/training utility and tests | Versioned reference blocks; APCC validation report |
| 4 | Binary/hierarchical modules | Optional challenger reports; no automatic default changes |
| 5 | `encode.rs`, `rate.rs`, RD benchmark tests | Audited objective and warm-up counters, controlled fixed8/adaptive residual ablation, retrieval-driven RD, optionally exact-result pruning |
| 6 | Postprocessor, decoder option, metrics experiments | Fixed-stream quality/time ablation |
| 7 | Sparse model/format/decoder/rate integration | Conditional versioned experimental codec and RD report |
| 8 | Result store/reporting and docs | Reproducible final leaderboard, limitations, promote/defer decisions |

**Next deliverable:** P5a's audit is delivered and measured; its finding (split-flag and mode symbols over-priced against a fine warm-up snapshot) is the input to P5b/P5c, not P5b itself. Next is P5c's deliberate quality extension — let several retrieved domain candidates into RD mode competition, without claiming it is output-preserving — or, if the audit's bias is judged the binding constraint, a re-snapshotting objective; either way, do not register a new search enum before those measured curves exist.

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

The filename is arbitrary; this smoke command uses **existing Fisher**, not the proposed random method. For production RD use `--method fisher --lambda 200 --modes 0,2`, now supported by P1. Omit `--method` or use `--method exhaustive` for the exhaustive control. Rebuild before mixing binaries/streams from different header layouts. Tiny64 cannot supply the pinned five-scale MS-SSIM and is not scientific corpus evidence.

A new file-to-file smoke is available (output directory must not already exist; parent must exist):

```sh
./target/debug/marsbench experiment-smoke fixtures/mars1/tiny64.raw --raw-dims 64x64 --out-dir target/research-smoke --lambda 200 --modes 0,2 --iterations 10 --threads 1 --timeout-secs 30
```

Build the debug CLI first with `cargo build --locked -p mars-cli`, or use release binaries consistently. The runner defaults to sibling `encmars`/`decmars`; explicit `--encmars`/`--decmars` paths are supported. Inspect `report.json` and logs even on failure. This command is implemented; the general commands below remain proposed.

The existing opt-in residual acceptance sweep is `just gate-22`. It is expensive, is not run by routine tests, and its previous recorded attempt timed out; do not describe it as passed. The three-arm synthetic smoke actually executed in this review is listed in §3. The CLI adaptive example is `encmars input.png output.mars --lambda 200 --modes 0,2,3 --adaptive-residual`; this exercises an experimental policy, not a recommended default.

### Implemented commands (P0.3, execution step 12)

```text
marsbench experiment --config configs/research-smoke.json --stage smoke
marsbench experiment --config configs/research-search.json --stage retrieval
marsbench experiment --config configs/research-rd.json --stage rd
marsbench experiment --config configs/research-final.json --stage timing
marsbench experiment-report --experiment-id <recorded-id> --reference-group modes0-2
```

CLI contract: `--config` is required; `--stage` selects one stage and is required when the config defines more than one. Artifacts default to `<root>/target/experiments/<experiment-id>` and hold an append-only `results.jsonl` plus a derived `summary.json`; `--root` resolves config-relative index/input paths; `--encmars`/`--decmars` override sibling discovery; `--limit N` caps planned cases for a cheap smoke. Resume is automatic and requires matching case/config/build hashes (a completed case under a different build is an error, not a silent re-run). `experiment-report` reads a store given directly or derived from `--experiment-id` under `--root`, prints Markdown, and computes per-image BD-rate between arm `group`s when a reference group is named. `research-search`, `research-rd` and `research-final` are opt-in corpus sweeps and are not run by routine tests. Avoid a misleading `cargo bench` label: the deliverable is a full codec experiment runner, not only microbenchmarks.

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

**Finish the restricted-profile serialized-output baseline and shared search path, then random search, then APCC.** Build on the delivered qstep metadata/tests without treating the incomplete residual experiment as a success. Reuse the existing quadtree RD optimizer, resolve residual semantics before mode-3 comparisons, add filtering as an independent low-risk experiment, and defer new sparse syntax until a measured benefit warrants it. A negative result against Saupe/Fisher/Funnel or existing modes is useful progress; an impressive number from mismatched bitrates or an unparsed stream is not.
