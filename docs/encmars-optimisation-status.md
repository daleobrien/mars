# encmars optimisation execution status

2026-09-17. Target: Rust `mars-cli`'s `encmars` and `decmars`, not the C reference.
This is a checkpoint, **not completion of O1–O16**. Existing scoped gates are not
substitutes for the roadmap's full acceptance requirements.

## Current implementation: O7 / Step 22 priority exception

The branch already had residual/progressive coding, so §21's exception was followed.

- Residual quantisation is now a per-stream parameter, shared by quantisation,
  distortion scoring, and reconstruction. `ResidualQstep` validates finite values
  in [1, 65535] and canonicalises them to binary32 before encoding.
- Experimental mapping: `sqrt(6 * lambda / ln(2))`, clamped to those bounds.
  Prediction and rationale were recorded before measuring. No corpus tuning.
- **Fixed step 8 remains the default.** `encmars --adaptive-residual --lambda N`
  explicitly selects the experiment for gray, RGB, and grayscale progressive input.
  `decmars` reads the step; no matching decoder option is needed.
- Invalid/nonfinite lambda is rejected by the CLI. The adaptive option requires
  lambda and conflicts with the legacy `--method` path.
- Single experimental format layout, revised in place by user direction. No old-layout
  fallback/version migration machinery: `MARS` header is 24 bytes, qstep binary32 LE
  at bytes 20–23; `MPRG` header is 35 bytes, qstep at bytes 31–34. Existing version
  marker remains zero. `MARC` wraps the revised plane streams unchanged.
  Previously generated `.mars` files must be re-encoded. `.ifs` is unchanged.
- `just gate-22` measures modes0/2 baseline, fixed8/all modes, adaptive/all modes at
  lambda [50, 200, 800, 3200] on kodim01/02. It checks real serialized reconstruction,
  reports overlaps and mode shares, and retains the +1.025% mean acceptance ceiling.
  Existing Step15 comparison explicitly retains fixed8 for attribution.

### Evidence and disposition

The 1,200-second foreground measurement timed out after **19/24 points**. Completed
rows, provenance and explicit timeout status are retained locally in
`results/step22-o7-1789648127131947000-41238.jsonl` (generated results are gitignored;
this file is not included in the source commit).

Completed kodim01 BD-rate versus the modes0/2 baseline:

| Policy | BD-rate | PSNR integration interval |
|---|---:|---|
| Fixed8 | +1.880847% | 21.521209–29.279225 dB |
| Adaptive | +3.342325% | 21.521209–29.010198 dB |

Historical two-image mean was +2.05%; **the new two-image mean is unknown**.
Do not compare the completed single-image figure as if it were a new corpus mean.
No further quantisation tuning was attempted. Adaptive has not earned default status;
O7 acceptance is unresolved and its completed-image result is negative. Revisiting
residual representation/prediction and rate estimation is preferable to tuning this
mapping blindly. The gate remains strict and does not report the timeout as success.

### Validation actually run

- Codec release library tests: 48 passed.
- Codec residual-qstep integration tests: 6 passed.
- New CLI integration tests: 5 passed, including nonzero residuals, fixed default,
  adaptive reconstruction, RGB420, progressive prefixes, and argument validation.
- Step22 harness tests: 3 passed; expensive acceptance test ignored in routine runs.
- After the CLI default-policy adjustment, `cargo test -p mars-cli --release --all-targets`
  passed all reported suites (21 tests). CLI-A's opt-in full RD comparison was not
  enabled; its real-image byte-equality test passed with matching explicit mode masks
  and thresholds. CLI-B verifies plain invocation equals lambda 200 / modes 0,2.
- `git diff --check` passed. `cargo fmt --all -- --check` failed with formatting
  differences in changed and untouched files; no workspace-wide reformat was applied.
- Focused Clippy for codec/all-targets and encmars/new CLI test passed.
- Workspace Clippy is blocked by pre-existing `unusual_byte_groupings` in
  `mars-bench/examples/train_learned.rs:32` and `type_complexity` in
  `mars-cli/src/bin/decmars.rs:75`. Those unrelated files were not changed.
- Full codec integration run timed out after ten minutes in the real-Kodak color test;
  it is not claimed as a full-suite pass.
- Real RGB CLI smoke test: `encmars corpus/images/kodak/kodim01.png
  target/o7-kodim01.mars --lambda 200 --adaptive-residual --subsampling 420 --threads 8`,
  then `decmars` and `marsbench metrics`: 768×512, 28,138 bytes, 0.572469 bpp,
  PSNR-Y 26.450003 dB, SSIM-Y 0.740129. Matching fixed-default CLI run:
  28,106 bytes, 0.571818 bpp, PSNR-Y 26.512978 dB. Adaptive uses 32 more bytes
  and loses 0.062975 dB at this point; it is not an improvement.
- `cargo test --workspace --release` compiled the workspace but timed out at 120 seconds
  during `cli_a_gate`'s real-image test. No full-suite pass is claimed. Opt-in measurement
  gates returning immediately in this run did not execute their corpus sweeps.

## Roadmap checkpoint

The following audit describes existing code/evidence, not freshly passed full gates.

| Step | Status / remaining work |
|---|---|
| O1 | Exhaustive encoder and independent decoder exist. No isolated selectable scalar product backend; gate6 primarily validates `.ifs`, gate10 covers `.mars`. |
| O2 | Top32 cache and recall harness exist; 72 cache files were confirmed locally. Configurations start at size8, excluding size4 production leaves. CPU/GPU contract differs from original ideal; see D25/D26. |
| O3 | Six classical methods plus exhaustive/KD-tree exist. Full-corpus recall/cost/BD-rate frontier and formal Gate B remain open. |
| O4 | Funnel exists in legacy search path. Stage1 truncates feature distance, not safe-bound-only pruning. Stage-wise recall and high-recall full frontier remain open. |
| O5 | Exact moment SIMD tests exist. Isolated scalar/SIMD end-to-end benchmark, UDOT investigation, other kernel priorities remain open. |
| O6 | Ordered Rayon collection/thread controls and determinism tests exist. Full CPU speed acceptance remains unmet in historical evidence. |
| O7 | Per-stream qstep and experimental CLI path implemented here; negative partial result above, gate22 not passed. |
| O8 | Bottom-up RD exists, but frozen rate snapshot rather than live emission state; historical −8.07% two-image gain is below −10% target. |
| O9 | Four leaf modes and split competition exist. Per-point mode bits/distortion/time attribution and improved residual acceptance remain open. |
| O10 | RD quadtree and sparsify-only density exist. Earlier density quality gain was withdrawn (D48). HV/denser domains remain deferred under their own plan. |
| O11 | Scan-order domain deltas exist. Measured predictor alternatives and qbeta deltas are not implemented. Progressive Step23 remains separate and open. |
| O12 | rANS and size contexts exist. Full mixed-mode fixed-reconstruction attribution/context optimisation and fuzz execution remain open. |
| O13 | No measured fast/balanced/compression-first frontier. Candidate retrieval is not integrated into RD. |
| O14 | Existing learned experiment is negative (D45); entry conditions for new work remain unmet. No ML added. |
| O15 | CPU product / GPU oracle separation retained. No product GPU work justified or added. |
| O16 | Final competitive-codec complexity audit cannot be closed without the preceding attribution evidence. |

## Next dependency

`encmars --method` still uses the legacy `mars_search::encode_image` walk and refuses
`--lambda` and RGB. Connecting a shared candidate-search interface to the existing RD,
mode, and color machinery is the principal integration gap. Do not create another
encoder traversal or advertise performance presets before a measured frontier exists.
The progressive Step23 fixes must still be implemented and measured individually;
they were not bundled with this residual change.
