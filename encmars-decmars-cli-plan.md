# Mars 2 — `encmars`/`decmars` CLI Completeness Plan

Standalone companion to `implementation-plan.md` and `improvement-plan.md`. Those two plans
are about the *codec* — new representation work and fixing measured regressions. This one
is about the two **binaries** in `crates/mars-cli/src/bin/{encmars,decmars}.rs` — the
Rust programs that stand in for the 1998 `encmars`/`decmars` (see the `mars1-reference`
skill) as this project's own command-line front ends.

**Finding that motivates this document.** Reading `crates/mars-cli/Cargo.toml` and the two
binaries against what `mars-search`, `mars-gpu`, and `mars-codec::progressive` already
provide shows the binaries are several steps behind the codec they wrap: capability that is
built, tested, and in one case already measured as a real BD-rate win is **unreachable from
the command line**. This is not a codec problem — every module below already exists and
has its own tests/gates. It is a plumbing gap between the library and the two programs
users (and this project's own scripts) actually invoke.

**Version:** 1.0 · **Status:** proposed · **Last updated:** 2026-09-15

---

## What's already wired vs. what's dependency-only

`mars-cli/Cargo.toml` depends on `mars-search`, `mars-gpu`, and `rayon` — but grep shows
`encmars.rs`/`decmars.rs` never reference any of them; only `marsbench.rs` does. And
`mars-codec::encode::encode_image` (the function `encode_color_image` — and therefore
`encmars` — actually calls) hardcodes two parameters that a caller lower in the same crate
can already vary:

```rust
// crates/mars-codec/src/encode.rs:662-664
pub fn encode_image(image: &Plane, params: &EncodeParams) -> (Header, Vec<Leaf>, u64) {
    encode_image_rd_with_modes_and_density(image, params, [true; 4], false)
    //                                                     ^^^^^^^^^  ^^^^^
    //                                            allowed_modes   adaptive_density
}
```

`adaptive_density: false` is Step 16's flag (`docs/decisions.md` D43) for content-adaptive
domain-pool density — **measured at -6.82% mean BD-rate**, a genuine, clean improvement,
and it is permanently off for every user of `encmars`. `allowed_modes: [true; 4]` means
mode masking (useful for isolating Step 22's residual-mode work, or for reproducing D40's
mode-competition measurements from the command line) is compiled in but not exposed.

---

## CLI-A — Expose Step 16's adaptive domain-pool density

**Why first.** This is the highest-value, lowest-risk item here: the feature is built,
tested (`encode.rs`'s `adaptive_density_*` tests, including a bit-identical-across-thread-
counts check and a byte-identical-when-off regression guard), and already has a measured
-6.82% BD-rate number on record. The only work is a CLI flag and threading a `bool` through
`EncodeParams` (or a new `ColorEncodeParams` field, since it's a codec-wide encode choice,
not a luma/chroma-specific one like `t_rms`).

**Deliverable**
- `encmars --adaptive-density` (default off, matching today's byte-identical behaviour —
  this must stay an opt-in flag, not a default flip, until re-measured at CLI-exercised
  scope; see exit criteria).
- Threads to `encode_image_rd_with_modes_and_density` via a new `ColorEncodeParams`/
  `EncodeParams` field, not a second code path — `encode_color_image` must gain exactly one
  new bool parameter, not a parallel implementation.
- `--help` text states the measured D43 number and points at `docs/decisions.md` D43, the
  way `--lambda`'s help text already narrates Step 14's framing.

**Exit criteria**
- `just gate-cli-a` (new): runs `encmars` with and without `--adaptive-density` on the
  existing `results/`-tracked corpus at the same rate points D43 used, reports BD-rate, and
  asserts it lands within a stated tolerance of D43's -6.82% (this is a **re-measurement at
  CLI scope**, not a re-use of D43's number — the CLI path threads through `color.rs`'s
  YCbCr/subsampling wrapping that D43's own measurement may or may not have exercised;
  confirm before assuming they're identical).
- The existing `adaptive_density_false_is_byte_identical_to_the_pre_step16_path` unit test
  continues to pass unmodified — the flag's default-off arm must remain provably
  regression-free for every existing caller.

**Abort rule.** None expected — this is wiring a already-measured, already-tested win, not
new research. If the CLI-scope re-measurement disagrees materially with D43's number,
that's itself a finding worth a `docs/decisions.md` entry (a `color.rs`-layer effect D43's
original measurement didn't account for), not a reason to drop the flag.

**Size.** S · **Verification burden:** low (reuses existing unit tests; new gate is a
thin re-measurement).

---

## CLI-B — Expose mode masking for diagnostics

**Deliverable**
- `encmars --modes <list>`, e.g. `--modes 0,1,2` to disable mode 3 (residual) or
  `--modes 2` to force fractal-only — a diagnostic/comparison knob, not a quality control a
  typical user reaches for.
- Threads `allowed_modes: [bool; 4]` through the same new params field as CLI-A.
- Primary use: makes Step 22's improvement-plan work (re-measuring the residual-mode
  regression before/after a fix) drivable from the CLI instead of a one-off test harness,
  and lets anyone reproduce D40's mode-usage histograms without writing Rust.

**Exit criteria**
- `just gate-cli-b`: `--modes 0,1,2,3` (all enabled) is byte-identical to omitting the flag;
  `--modes 2` on a known image matches a hand-computed "fractal-only" leaf count.

**Abort rule.** None — small, mechanical, low-risk.

**Size.** S · **Verification burden:** low.

---

## CLI-C — Classical/funnel/learned method selection (mirror the C `-F -X -C -S -Z -Y`)

**Why this matters.** `mars-search`'s `MethodName` enum (`Exhaustive`, `Fisher`,
`Hurtgen`, `MassCenter`, `Saupe`, `SaupeFisher`, `McSaupe`, `Funnel`, `Learned`) is a full,
gated, tested port of Mars 1's six classical speed-ups plus Step 13's novel Funnel method —
and it is **completely unreachable from `encmars`**, which only ever runs `mars-codec`'s
own exhaustive/bottom-up walk. Anyone who wants to run the classical methods today has to
write a Rust test or use `marsbench`'s recall/regret harness, which is built for
measurement sweeps, not for producing a `.mars` file to look at.

**Design question to resolve before implementing.** `mars-search`'s methods restrict the
*candidate set* the domain/isometry search considers per range block (a speed-up over
Step 6's exhaustive search); `mars-codec`'s own `EncodeParams::lambda` controls the
*partition* (bottom-up RD-pruned split vs. legacy top-down `t_rms`). These are orthogonal
axes today (`docs/decisions.md`'s Step 14 entry: `t_rms` "is a separate axis: candidate
restriction for speed, not the split decision itself"). Decide and document whether
`encmars --method funnel --lambda 200` is a supported combination (candidate-restricted
search inside an RD-optimal partition) or whether `--method` and `--lambda` are mutually
exclusive in the CLI — **do not let this be an implicit default nobody chose**, per this
project's own convention (A7) of never leaving a silent interaction between two flags for
a user to discover the hard way, the same failure class D5 named for Mars 1's own silent
method fallback.

**Deliverable**
- `encmars --method {exhaustive,fisher,hurtgen,masscenter,saupe,saupe-fisher,mc-saupe,
  funnel,learned}` (default: today's behaviour — `mars-codec`'s own walk, not
  `mars-search::Exhaustive`, which is a distinct implementation kept for cross-validation;
  the CLI's `--help` must say which is which, since they are two different code paths that
  happen to share a name).
- The `--method`/`--lambda` interaction resolved per the design question above and stated
  in `--help`.
- stdout reports the method actually run and its `evals`/`transforms` counts, mirroring the
  `mars1-reference` skill's stdout-fields table for the C binary, so a script diffing Mars 1
  vs. Mars 2 output can parse both with the same regex.

**Exit criteria**
- `just gate-cli-c`: for each of the 9 `MethodName` variants, `encmars --method X` produces
  a `.mars` file that `decmars` decodes without error, and the reported `evals` count
  matches what `marsbench`'s own harness reports for the same method/image/params —
  cross-validating the new CLI path against the existing measurement path rather than
  trusting the new wiring on inspection alone.

**Abort rule.** If `Learned`'s CLI wiring would require shipping model weights or a training
step as part of a plain `encmars` invocation, exclude `--method learned` from this item and
open it as a separate, explicitly-scoped follow-up — per D45, `Learned` is already a
documented negative result (loses to `Funnel`) and does not justify CLI-path complexity
disproportionate to Step 22-24's [`improvement-plan.md`] Step 24 finding.

**Size.** M · **Verification burden:** medium (new gate cross-checks two independent code
paths against each other).

---

## CLI-D — GPU search and explicit thread count

**Context.** `mars_gpu::GpuSearcher`/`GpuSearchParams` and `rayon::ThreadPoolBuilder` are
both already used by `marsbench` (Step 7/12's own benchmark harness) but neither is reachable
from `encmars`. Today, encoding a large image or corpus through the CLI always uses Rayon's
default global thread pool and never the GPU path Step 7 built and gated on ≥ 50× speedup
and bit-identical-to-CPU output.

**Deliverable**
- `encmars --threads <n>` (default: Rayon's default — do not change existing behaviour
  when omitted), matching `marsbench`'s existing `ThreadPoolBuilder` usage exactly so the
  two binaries' thread-count semantics don't silently diverge.
- `encmars --gpu` to run the search on `mars_gpu::GpuSearcher` instead of the CPU path,
  for methods/modes where a GPU path exists — gate this to whatever subset Step 7 actually
  covers; do not claim GPU support for a combination (e.g. `--method funnel --gpu`) that
  Step 7 never built or gated.

**Exit criteria**
- `just gate-cli-d`: `--gpu` output is bit-identical to the CPU path on the same input and
  params — reusing Step 7's own bit-identity oracle, not a new tolerance — and `--threads`
  produces byte-identical output across at least 2 different thread counts (determinism,
  per `implementation-plan.md` §2.3, is not optional).

**Abort rule.** If wiring `--gpu` into `encmars` would require duplicating
`GpuSearchParams` construction logic currently private to `marsbench` in a way that risks
the two binaries' GPU invocations drifting apart, factor that construction into
`mars-cli::lib.rs` (already the shared-code home for the three binaries) first, rather
than copy-pasting it — a second copy is exactly the kind of unattributed divergence this
project's conventions (A7) warn against.

**Size.** M · **Verification burden:** medium (bit-identity oracle already exists in
Step 7; reused, not built new).

---

## CLI-E — Progressive bitstream support (Step 19)

**Context.** `mars_codec::progressive::{encode, decode, layer_end_offsets}` exist, are
gated (`gate-19`, P19.1's bit-exact 4-layer property), and are the subject of
`improvement-plan.md`'s Step 23 rate-optimisation work — but neither `encmars` nor
`decmars` can produce or consume a progressive stream at all today. `progressive::encode`
operates on a single `Plane`, so this item also has to resolve whether/how a progressive
`.mars` container composes with `color.rs`'s YCbCr wrapping (Step 18), which
`progressive.rs` was not designed against — **check this rather than assume it composes
cleanly**; if it doesn't, scope this item to grayscale first and record the colour gap
explicitly rather than silently shipping a colour progressive path nobody validated.

**Deliverable**
- `encmars --progressive` writes the 4-layer progressive container instead of the single-
  layer `.mars` v0 format (grayscale first; colour only if the composition check above
  passes cleanly, else a separate follow-up item).
- `decmars --layer {1,2,3,4}` decodes only the requested prefix — the whole point of a
  progressive stream being that a prefix is independently decodable (P19.1's own property).
  Default (no `--layer`): decode the full 4-layer stream, equivalent to today's output.
- stdout reports which layer was decoded and the file-offset each layer boundary falls at
  (from `layer_end_offsets`), so a script can slice a partial download and know what it has.

**Exit criteria**
- `just gate-cli-e`: `decmars --layer N` on a full progressive file, for each `N`, is
  byte-identical to `decmars` run on a file truncated to that layer's own end offset —
  this is P19.1's "every prefix decodes" property, exercised through the CLI rather than
  only through `progressive_gate.rs`'s internal test.

**Abort rule.** If colour composition is unresolved by the time this item is scheduled,
ship grayscale-only and record the colour gap in `docs/decisions.md` rather than blocking
the whole item on it — grayscale progressive is independently useful and already fully
specified by Step 19.

**Size.** L (grayscale) to XL (if colour composition needs new design work) ·
**Verification burden:** high — reuses P19.1's exact-equality oracle, must not weaken it.

---

## Suggested order

CLI-A and CLI-B first (small, zero-risk, and CLI-A unlocks a real BD-rate win that today's
`encmars` simply cannot produce). CLI-C next (moderate size, resolves a real design gap).
CLI-D and CLI-E can run in parallel with each other once C is done, since GPU/thread
plumbing and progressive-container plumbing don't share code paths.

## What this document deliberately does not include

- Any new codec capability — everything referenced here (`adaptive_density`, `MethodName`,
  `GpuSearcher`, `progressive`) already exists, is already tested, and in one case already
  has a measured BD-rate number on record. This is exclusively about making the two
  binaries reflect what the library crates already do.
- Mars 1's `-Q` (quadtree visualisation output) or other purely-diagnostic C flags with no
  Mars 2 equivalent yet — out of scope unless a specific need for one surfaces.
