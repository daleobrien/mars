# Mars 2 — HV/Binary Split Plan

Companion to [implementation-plan.md](implementation-plan.md) and
[improvement-plan.md](improvement-plan.md), which remain the source of record for their own
steps' scope and conventions — this document does not restate or override either. It exists
because Step 16 (`docs/decisions.md` D43, later corrected by D48) explicitly scoped **out**
non-quad split shapes ("HV/binary splits"), recording that omission as a deliberate,
brief-sanctioned cut (`implementation-plan.md`'s own brief text: "optionally") rather than
an oversight — see `docs/predictions.md`'s Step 16 prediction and `docs/decisions.md` D43
for the original reasoning, and `docs/decisions.md` D48 for the correction (Step 16's own
measured -6.82% BD-rate number is withdrawn; not relevant to this plan's own scope, but
cited here so a reader following D43's trail does not stop at a superseded number).

Steps here are numbered **25 onward**, continuing `implementation-plan.md`/
`improvement-plan.md`'s sequence (which ends at Step 24). Every step below follows the same
brief contract as those documents: entry condition, hypothesis, deliverable, exit criteria
(a command, not a judgement), abort rule, size. `just gate-N` must exist and exit 0/1 before
a step is marked closed.

**Version:** 1.0 · **Status:** proposed · **Last updated:** 2026-09-15

---

## Why this is a bigger lift than the brief's "optionally" suggests

Before any step below, this is the single most important finding this plan is built on,
established by reading the actual code rather than assuming from the brief's one-line
mention: **Mars 2's entire fractal/DCT machinery assumes square blocks at every layer**,
not just at the bitstream's `FIELD_SPLIT` flag Step 16 already named. Concretely, from the
code as it stands today:

1. **`Leaf::size` is one `u32`, not a `(width, height)` pair**
   (`crates/mars-codec/src/ifs.rs`). Every leaf — range and (implicitly, via `dom_row`/
   `dom_col` plus `size`) domain — is square by construction. A leaf produced by an HV
   split is rectangular; there is no field to hold its width and height independently
   today.
2. **The 8-way isometry group is defined only for squares**
   (`crates/mars-codec/src/isometry.rs::map`): four of the eight codes (90°-rotate and the
   two diagonal flips) *swap* width and height, which only produces a shape-preserving map
   when width == height. Applied to a genuinely rectangular block, they would need a domain
   of the *transposed* aspect ratio — a real complication most fractal codecs sidestep by
   restricting non-square blocks to the four isometries that actually preserve shape
   (identity, 180° rotation, horizontal flip, vertical flip).
3. **The DCT residual (mode 3) is a square-only 2D transform**
   (`crates/mars-codec/src/dct.rs::forward_dct2d`/`inverse_dct2d`: `assert_eq!(block.len(),
   size * size)`). A non-square 2D-DCT (separate row/column transform lengths) is not hard
   to write, but it does not exist yet, and mode 3's residual field (`Vec<i32>`, row-major,
   implicitly `size*size` long) would need the same width/height generalisation as `Leaf`
   itself.
4. **The domain pool, contracted-image cache, and exhaustive/candidate-restricted search**
   (`crates/mars-codec/src/encode.rs`, `mars-search`'s `DomainPool`/`Block`/
   `CandidateRetriever`) all key off one `size`, both for range blocks and for the domain
   positions searched against them. A rectangular range needs domain candidates of the
   *same* rectangular shape (or a defined resampling rule if not — not attempted here), at
   every size the partition can produce.
5. **Mode 1 (affine)'s plane fit** (`crates/mars-codec/src/encode.rs::affine_fit`) is
   written against a square `size x size` sample grid (closed-form normal-equation
   coefficients derived from `0..size` on both axes). Generalising to `width != height` is
   mechanical (separate row/column index ranges) but is real, unwritten code, not a
   parameter rename.

None of this is a reason not to do the work — it is the reason the brief marked it
"optionally" and Step 16 declined it under that step's own time budget. It is stated here,
before any step is written, so this plan's size estimates are honest: **this is a format
change plus a partial rewrite of the fractal/DCT core, not a new split-type enum value.**

---

## Design decision, made here rather than left implicit (per this project's own convention, A7)

**Scope this to axis-aligned 2:1 binary splits only, with domain search restricted to the
same aspect ratio, and mode 1/3 support deferred.** Reasoning:

- **2:1 splits only** (a size×size block splits into two size×(size/2) or (size/2)×size
  rectangles), not arbitrary split ratios — matches every other coding-tree design this
  project has looked at (HEVC/VP9-style recursive partitioning), keeps `min_size`/
  `max_size` reasoning nearly unchanged (a rectangle's short side still bottoms out at the
  existing `min_size` floor), and avoids inventing a new geometry-description field beyond
  a small split-type symbol.
- **Domains restricted to the same aspect ratio as the range being matched** — the
  alternative (matching a rectangular range against a square-only domain pool via
  anisotropic resampling) is a materially harder, unmeasured research question this plan
  does not attempt. This does mean the domain pool must be built per-shape as well as
  per-size once rectangles exist (Step 25a/25c below).
- **Isometry restricted to the four shape-preserving codes** (identity, 180° rotation,
  horizontal flip, vertical flip) for any non-square leaf — per the finding above. Square
  leaves (every leaf under today's quad-only partition) are completely unaffected and keep
  all eight.
- **Mode 1 (affine) and mode 3 (fractal + residual) are deferred for non-square leaves to a
  named follow-up, not built now.** Every step-15/16/48 measurement on record shows mode 2
  (fractal) and mode 0 (flat) accounting for the overwhelming majority of leaves on real
  photographs (D43/D48: fractal 68-92% depending on image, flat most of the remainder,
  affine consistently ~0.1%, residual under 1%) — shipping HV split for modes 0/2 only
  covers the leaf population that actually matters for a first measurement, at a fraction
  of the implementation risk of also generalising the DCT and the affine fit. This is
  recorded as an explicit, scoped cut here — not a silent gap — mirroring CLI-E's own
  "grayscale first, colour composition deferred" precedent in
  `encmars-decmars-cli-plan.md`.
- **Grayscale only, `.mars` v0's colour/progressive containers untouched.** Composing a
  rectangular-leaf partition with `mars_codec::color`'s per-plane wrapping or
  `mars_codec::progressive`'s 4-layer container is out of scope here — both are already
  built against a `Vec<Leaf>` of whatever shape `mars_format` can serialise, so once Step
  25b lands they may compose for free, but that composition is not itself gated by this
  plan.

If any step below finds this scope still too large for its own size estimate, the fallback
recorded up front (so it is a documented option, not an improvised one under pressure) is
to cut mode 2 down further still: ship HV split for **mode 0 (flat) only** first — no
domain search generalisation at all, since flat leaves carry no domain reference — as a
strictly smaller Step 25a-only deliverable that still exercises the format/decoder change
end to end, with mode 2 support following as its own step.

---

## Suggested order

25a and 25b are pure plumbing (data model, then bitstream format) and must land in that
order — 25b's new split symbol is meaningless without 25a's `(width, height)` `Leaf`. 25c
(isometry/domain-pool restriction) and 25d (mode 0/2 support for rectangles) can proceed
together once 25a/25b are gated. 25e (the actual RD split-decision integration) depends on
all four and is where the real "does this help" measurement happens — it is deliberately
last and is the only step with a BD-rate exit criterion; everything before it is
plumbing verified by exact-equality oracles, per this project's own
`verification-discipline` convention (an optimisation or new representational choice earns
a tolerance-based gate; a refactor earns a bit-identical one).

---

#### Step 25a — `Leaf`/format plumbing: `size: u32` → `width: u32, height: u32`

**Entry condition.** `gate-16`/`gate-19` pass on record (unaffected by this step — see
exit criteria). No other step in this plan has started.

**Context.** Finding 1/3/4/5 above all trace back to one root cause: `Leaf::size` is a
single field. This step generalises it without changing behaviour for any *existing*
(square) leaf — a pure, behaviour-preserving refactor, not new capability.

**Deliverable**
- `Leaf::size: u32` replaced by `Leaf::width: u32, Leaf::height: u32` throughout
  `mars-codec` (`ifs.rs`, `mars_format.rs`, `encode.rs`, `dct.rs`, `residual.rs`,
  `progressive.rs`, `color.rs`) and `mars-search` (`Block`, `DomainPool`, `RangeBlock`,
  every method's own `index()`/`candidates()`).
- Every existing call site that constructs a square leaf passes `width == height` — no
  behavioural change for the square-only path.
- `mars_format`'s bitstream is unchanged for a purely-square partition (this step does
  **not** yet add the new split symbol — that is 25b) — a square-only round trip must
  still produce byte-identical output to before this refactor.
- `dct.rs::forward_dct2d`/`inverse_dct2d` gain non-square-capable signatures
  (`forward_dct2d(block, width, height)`), even though nothing calls them non-square yet
  (mode 3 for rectangles is 25d+'s own follow-up, deferred per the design decision above) —
  written and unit-tested now while the square-only behaviour is being touched anyway,
  rather than as a second pass later.

**Exit criteria**
- `just gate-25a`: `cargo test -p mars-codec -p mars-search --release` passes unmodified in
  assertion strength (no test's tolerance loosened to accommodate the refactor).
- A new byte-identical-output regression test: every existing gate corpus/params
  combination this plan can cheaply re-run (`gate-14`/`gate-15`/`gate-16`'s own λ grids on
  kodim01/kodim02) produces **byte-identical** `.mars` bytes before and after this step —
  the refactor's own self-check, run once and recorded, not part of routine CI given its
  cost.

**Abort rule.** If the refactor cannot be made byte-identical for the square-only path
without touching more than `mars-codec`/`mars-search` (e.g. if `mars-bench`'s own harnesses
need behavioural, not just type-signature, changes), stop and re-scope: that would indicate
the `size` assumption is more deeply load-bearing than this step's own estimate, and the
remaining steps' size estimates need revisiting before continuing.

**Size.** L · **Verification burden:** high (touches every leaf-constructing call site in
two crates) but low-risk (mechanical, exact-equality oracle throughout).

---

#### Step 25b — Bitstream format: a real split-type symbol, format version bump

**Entry condition.** Step 25a closed (`gate-25a` passing).

**Context.** `FIELD_SPLIT` is coded today as a 2-way symbol (`walk_read`/`walk_write` in
`mars_format.rs`): `dec.next((FIELD_SPLIT, size_class), 2) == 1`. This step widens it to a
4-way symbol — leaf / quad-split / h-split / v-split — which is a genuine `.mars` format
change (existing `.mars` v0 files remain readable **only** if a version byte gates the new
symbol width; do not silently reinterpret the existing 2-way field for old files).

**Deliverable**
- `.mars` format version bump (`MARS_VERSION`, `mars_format.rs`) — a v0 file still parses
  as 2-way `FIELD_SPLIT`; a new-version file parses as 4-way. Document the new value in
  `docs/mars-format.md` alongside the existing Step 10 fields, per `improvement-plan.md`
  Step 22's own precedent for a format change.
- `walk_write`/`walk_read` updated: an h-split emits/reads two `width x (height/2)`
  children (top/bottom); a v-split emits/reads two `(width/2) x height` children
  (left/right); a quad-split is unchanged (still four `(width/2) x (height/2)` children).
  Recursion order for h/v-split children follows the same top-left-first convention
  quad-split already uses, stated explicitly in `mars_format.rs`'s own doc comment.
- `min_size` applies to the **short side**: a block whose short side is already `min_size`
  may not h/v-split further in that direction (it may still quad-split if both sides
  exceed `min_size`, or must leaf). This is a real new rule, not implied by the existing
  square-only `min_size` check — write it down and test it explicitly.

**Exit criteria**
- `just gate-25b`: a synthetic partition exercising all four split outcomes (leaf, quad,
  h-split, v-split) at multiple depths round-trips through `mars_format::write`/`read`
  exactly. A v0-version file with the old 2-way encoding still parses correctly (backward
  compatibility, not merely forward capability). An oversized/malformed split-type symbol
  errors rather than panics (mirrors `mars_format`'s existing malformed-input discipline).

**Abort rule.** None expected — this is a mechanical format extension once 25a's data
model exists. If `min_size`-on-short-side interacts badly with `max_size`/virtual-size
padding logic (`Header::virtual_size`, `ceil_log2`) in a way that produces a block the
decoder cannot address, stop and fix the geometry rule before proceeding to 25c — a
partition the decoder cannot losslessly address is a correctness bug, not a design
tradeoff.

**Size.** M · **Verification burden:** high (bitstream format changes always are) but
narrowly scoped (one field, one module).

---

#### Step 25c — Isometry and domain-pool restriction for rectangular blocks

**Entry condition.** Step 25b closed.

**Context.** Per the design decision above: a rectangular block (`width != height`) may
only use the four shape-preserving isometries (identity, 180° rotation, horizontal flip,
vertical flip — codes 0, 3, 4, 5 in `crate::isometry::map`'s existing numbering), and its
domain candidates must be drawn from positions of the *same* `width x height` shape, not
resampled from a square pool.

**Deliverable**
- `crate::isometry::map` generalised to `(width, height)` instead of one `size`, with the
  four shape-changing codes (1, 2, 6, 7) returning an explicit "not applicable to a
  non-square block" result the caller must handle (a `Result`/`Option`, not a silent wrong
  answer) — square callers (`width == height`) are unaffected and keep using all eight.
- The domain pool (`mars-codec::encode::Contracted`/`search_with_shift`,
  `mars-search::DomainPool`) indexed per `(width, height, shift)` rather than per `size`
  alone, so a rectangular range block's search only ever considers same-shape domains.
- `mars-search`'s `CandidateRetriever` trait and its nine implementations: confirm (or, if
  needed, extend) the trait's own shape assumptions — this step does **not** need every
  method working for rectangles (that is CLI-C-adjacent follow-up work, out of scope here),
  but `Exhaustive`'s own rectangular-capable path must exist, since it is the reference
  implementation this plan's own BD-rate measurement (25e) needs.

**Exit criteria**
- `just gate-25c`: unit tests confirming (a) `isometry::map` on a non-square block rejects
  the four shape-changing codes rather than producing a silently-wrong index, (b) an
  exhaustive rectangular domain search only ever returns same-shape candidates, checked by
  construction against a synthetic image with a known best rectangular match.

**Abort rule.** None expected — this is applying the design decision's own restriction, not
new research. If restricting to same-shape domains empirically starves the search (very
few or no candidates at some sizes on a real image), that is itself a finding for Step
25e's own measurement to report, not a reason to abort this step.

**Size.** M · **Verification burden:** medium.

---

#### Step 25d — Mode 0/2 support for rectangular leaves (mode 1/3 explicitly deferred)

**Entry condition.** Step 25c closed.

**Context.** Per the design decision: ship the two leaf modes that dominate real leaf
populations (flat, fractal) for rectangular blocks; mode 1 (affine)/mode 3 (residual) stay
square-only for now, refused (not silently produced incorrectly) when a candidate leaf is
non-square.

**Deliverable**
- `best_mode_leaf` (`crates/mars-codec/src/encode.rs`) accepts non-square candidates for
  modes 0/2 only; modes 1/3 are excluded from the competition entirely when `width !=
  height` (not merely never selected — never evaluated, so there is no silently-wrong
  affine/DCT computation to worry about).
- `crate::ifs::decode_iterative`'s reconstruction loop (currently `size x size` pixel
  copy per leaf) generalised to `width x height` for modes 0/2 (mode 1/3's non-square path
  is simply unreachable given the encoder-side restriction above, so the decoder does not
  need to handle a non-square mode-1/3 leaf it will never receive — but should still error
  cleanly, not panic, if one is ever seen, e.g. from a hand-crafted malformed file).

**Exit criteria**
- `just gate-25d`: a synthetic image with genuinely non-uniform local complexity produces
  a real mix of rectangular mode-0 and mode-2 leaves under an RD-driven encode (harness-
  sanity: the mechanism is not inert), and every one decodes back to the exact pixel values
  `decode_iterative`'s own algorithm computes for each mode (an exact-equality check
  against the same computation, not a tolerance).
- A malformed file claiming a non-square mode-1/3 leaf is rejected with an error, not a
  panic (`mars_format`'s existing malformed-input discipline, extended).

**Abort rule.** None expected. If real-image encodes show mode-0/2-only rectangular leaves
losing badly to square leaves at matched rate almost everywhere (i.e., the mechanism looks
structurally unable to help even before considering mode 1/3), that is itself the answer
Step 25e's own measurement should report plainly — not a reason to add mode 1/3 support
under pressure to "make HV split competitive."

**Size.** M · **Verification burden:** medium.

---

#### Step 25e — RD split-decision integration and the real BD-rate measurement

**Entry condition.** Steps 25a-25d closed.

**Context.** Everything before this step is plumbing, verified by exact-equality oracles.
This step is the actual research question the brief's "optionally" was pointing at: does
adding h-split/v-split as competing options inside Step 14's `J = D + λR` bottom-up walk
(`walk_rd`) improve the RD curve over quad-split-only, on real images? Mirrors Step 15's
own mode-competition precedent (`best_mode_leaf`) for how a new discrete option is added to
an existing `J`-minimisation without disturbing the modes/splits already there.

**Deliverable**
- `walk_rd` evaluates, at each node with both sides above `min_size`: leaf, quad-split
  (existing), h-split, v-split (new) — same `J = D + λR` comparison, four candidates
  instead of two. `crate::rate`'s frozen `RateModels` warm-up snapshot needs the new
  `FIELD_SPLIT` symbol's context observed at least once during warm-up construction, per
  D39/D40's own already-documented "new field, empty snapshot bucket" hazard — check this
  explicitly before assuming the warm-up generalises for free, the same way D40 had to be
  checked for Step 15's own new fields.
- Mode-usage *and* split-usage histogram (leaf / quad / h-split / v-split share, alongside
  the existing per-leaf-mode histogram) — the brief's own "more scientifically interesting
  than the BD-rate number" framing (D43's own header finding for Step 16) applies here too.

**Exit criteria**
- `just gate-25`: BD-rate of "HV split enabled" vs. "quad-only" (Step 14/15's own unchanged
  behaviour), same-codebase A/B per D40's precedent, on kodim01/kodim02 at the same 4-point
  λ grid every prior RD gate in this project uses. Convexity/monotonicity check.
  Encode-time cost accounting (the brief's own "report the cost alongside the gain"
  instruction, per D43/D48's own precedent) — four candidates per RD-pruned node instead of
  two is real extra search cost. Split-usage histogram printed.
- P19.1-style exact-equality regression: HV-split-disabled behaves byte-identically to
  before this step (the additive-superset guarantee every prior step in this project's own
  history has held itself to).

**Abort rule.** If the measured BD-rate is a regression, or an improvement smaller than
roughly 1% (the low end of Step 16's own original 1-5% prediction band for a materially
*smaller* single-lever change, and this lever costs a real format version and four-way
split-decision cost against Step 16's two-value density lever), stop and record the result
plainly in `docs/decisions.md` — per this plan's own design-decision section, the deferred
mode 1/3 support and the same-aspect-only domain restriction are both named, standing
hypotheses for *why* a disappointing result occurred, worth stating explicitly rather than
re-opening scope under pressure to "make it work."

**Size.** L · **Verification burden:** high — this is the step with a real BD-rate claim
and a live format-version interaction with the RD warm-up snapshot.

---

## What this document deliberately does not include

- Mode 1 (affine)/mode 3 (fractal + residual) support for rectangular leaves — named as a
  standing follow-up in the design decision above, not attempted here. If Step 25e's
  measurement shows mode-0/2-only HV split is a real win, generalising mode 1's affine fit
  and mode 3's DCT to non-square blocks (both mechanical, per the "why this is bigger"
  section's own findings) is the natural next step.
- Split ratios other than 2:1 (no 1:3, no arbitrary ratios) — deliberately out of scope,
  per the design decision.
- Domain candidates of a *different* aspect ratio than the range being matched (anisotropic
  resampling) — an unmeasured research question, not attempted.
- Colour (`mars_codec::color`) and progressive (`mars_codec::progressive`) composition —
  both operate on a `Vec<Leaf>` of whatever shape `mars_format` can serialise, so they may
  compose for free once Step 25b lands, but that composition is not gated by this plan (the
  same "check, don't assume" caution `encmars-decmars-cli-plan.md`'s CLI-E item applied to
  progressive's own colour composition applies here).
- Exposing HV split as an `encmars` CLI flag — once Step 25e closes with a real result
  (positive or negative), wiring it through the CLI is exactly the kind of small, mechanical
  plumbing `encmars-decmars-cli-plan.md`'s CLI-A/B items were; not written here to keep this
  plan's own scope to the codec change itself.
- `mars-search`'s nine candidate-restriction methods gaining rectangular support beyond
  `Exhaustive` (Step 25c's own reference implementation) — each method's own feature
  extraction (Saupe vectors, Fisher/Hurtgen classification, Funnel's stages) would need
  its own per-method review for square-only assumptions, out of scope for this plan.
