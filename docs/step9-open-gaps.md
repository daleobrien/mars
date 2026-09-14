# Step 9 — open gaps

`gate-9` passes, but on a scope reduced from the brief in `implementation-plan.md`.
Full reasoning is in `docs/decisions.md` D27/D28; this file is just the punch list of
what's left before Step 9 can be considered fully closed against the original brief.

1. **RD-curve-within-0.2dB check is not implemented.** `mars_search::encode_image`'s
   output isn't wired through `mars_codec::ifs::write`/`decode_iterative` and
   `mars_core::metrics` to produce real PSNR/bpp points, so there is no BD-rate
   comparison against `results/baseline-mars1.jsonl`'s Step 2 baseline yet. `gate-9`'s
   own output says "RD-curve check not yet implemented" rather than silently skipping it.

2. **Recall/regret table covers 2 of 24 `standard/` corpus images** (`kodim01`,
   `kodim02`), not the full Kodak set — session time budget; `Exhaustive` alone costs
   ~90-110s per image since it's the true `O(domains × isometries)` baseline everything
   else is scored against.

3. **evals/transform comparison reused Step 2's already-captured C-binary baseline**
   rather than a fresh run — this session's sandbox couldn't compile
   `reference/mars1` (`xcrun` has no working arm64 `cc`). The comparison itself is real
   (not fabricated), just not freshly regenerated.

None of `gate-9`'s asserted checks had a tolerance loosened to pass — the gap is in what
was measured, not in how strictly it was judged.
