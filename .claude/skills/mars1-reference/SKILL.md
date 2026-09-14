---
name: mars1-reference
description: Operating and reading the 1998 C reference codec in reference/mars1 — pinned build flags, the encmars and decmars CLI, the stdout fields the harness parses, the .ifs bitstream layout, the normative quantisation formulas, and the defaults that behave surprisingly. Use when running or scripting the Mars 1 baseline, generating or reading golden fixtures, writing or debugging the .ifs parser, or porting any Mars 1 behaviour into Rust.
---

# Mars 1 (1998 C) reference

`reference/mars1/` holds the original C, unmodified. It is a **historical reference and
the project's baseline, not something to mechanically translate** — Mars 2 is a
clean-room Rust implementation, which keeps the copyright story simple. Bit-exact
reproduction is explicitly **not** a goal (`implementation-plan.md`, decisions of
record); the C binaries run under the harness and *are* the baseline.

## Build

`scripts/build-mars1.sh` compiles with **pinned flags**:

```
-O2 -fno-fast-math -fno-unsafe-math-optimizations
```

Target aarch64 or x86-64 — **never 32-bit x86** (x87 excess precision). Record the
compiler and version in every fixture manifest and result row.

## CLI

**`encmars [options] <infile> <outfile>`** (defaults `lena.raw` → `lena.ifs`):

| Flag | Meaning | Default |
|---|---|---|
| `-F -X -C -S -Z -Y` | method: Fisher, Hurtgen, MassCenter, Saupe, Saupe-Fisher, Mc-Saupe | **none = MassCenter** |
| `-r <f>` | RMS threshold `T_RMS` | 8.0 |
| `-e <f>` / `-v <f>` | entropy / variance pre-split thresholds | 8.0 / 1e6 |
| `-m <n>` / `-M <n>` | min / max range size (powers of two, 2–64) | 4 / 16 |
| `-d <n>` | domain step `SHIFT` (even, 2–32) | 4 |
| `-A <n>` / `-B <n>` | `N_BITALFA` / `N_BITBETA` (1–15) | 4 / 7 |
| `-y <f>` | `MAX_ALFA` (0.0–8.0) | 1.0 |
| `-W <n>` / `-H <n>` | image width / height (raw input) | — |
| `-z <n>` | zero-alfa threshold | 0 |
| `-l -p -f -s -k -c -n -a` | method-specific search tuning (matches, eps, full classes, shrink factors, features, classes, adapt) | see `miscell.c:getopt_enc` |
| `-Q` | quadtree output | off |

**`decmars [options] <infile> <outfile>`** (defaults `lena.ifs` → `lena.dec.pgm`):

| Flag | Meaning |
|---|---|
| *(none)* | **pyramidal decode**, `piramidal = 1`, `iterations = 10` — the default |
| `-i` | iterative decode instead |
| `-n <n>` | iteration count |
| `-r` | raw output instead of PGM |
| `-p` / `-q` / `-d` / `-z <f>` | postprocess / quality / display / zoom |

## Encoder stdout — the fields the driver parses

```
 Image Entropy      : %f
 Image Variance     : %f
 Entropy threshold  : %f
 Variance threshold : %f
 Rms threshold      : %f
 Zero_alfa_transformations   : %d
 Number of transformations   : %d      <- transforms
 Number of comparisons       : %ld     <- the Mars 1 `evals` analogue
 Comparisons/Transformations : %f
 %d bytes written in %s
```

`comparisons` is incremented at six sites in `coding_func.c`, one per speed-up method.
Mars 2's `evals` counter must increment at the *exactly analogous* point or the two are
not comparable.

## Defaults that surprise people — know these before porting

- **At default settings the Mars 1 partition is purely RMS-driven.** With `T_ENT = 8.0`
  the entropy pre-split can never fire (an 8-bit block's entropy is bounded by 8.0, and
  by 4.0 for a 4×4 block), and with `T_VAR = 1e6` the variance pre-split cannot fire
  either (8-bit variance is bounded by ~16256). Knowing this saves a week of confusion.
- **There is no exhaustive mode, and no flag does not mean one.** `globals.h:201` is
  `EXTERN int method INIT(= MassCenter)`, so an omitted or mistyped method flag silently
  runs MassCenter instead of failing. Always pass a method explicitly and check the
  `Speed-up method:` line the encoder echoes. The exhaustive RD upper bound (M6's oracle)
  has no Mars 1 equivalent and is built in Rust at Step 6/7. See `docs/decisions.md` D5.
- **Absolute paths silently corrupt the invocation.** `globals.h` declares `char
  filein[50]`/`fileout[50]` and `getopt_enc` copies `argv` in with `strcpy` — no bounds
  check. A path over 49 bytes overflows the buffer, and the failure mode is misleading:
  the encoder reads the input fine, reports progress normally, then fails with `Can't open
  output file`, which looks like a permissions problem rather than a buffer overflow.
  Always `chdir` into a short scratch directory and pass **relative** filenames; never
  drive `encmars`/`decmars` with absolute paths, including in CI. See `docs/decisions.md`
  D3.
- **Decode mode must be pinned and recorded — and pyramidal decode is not just "slightly
  different," it can be catastrophic.** Pyramidal (`decmars` with no flags) renders at
  `1/2^levels` scale before upsampling, so a range block of `min_size` occupies `min_size /
  2^levels` pixels at that stage. The rule is exact: `min_size / 2^levels ≥ 1` costs under
  0.05 dB versus iterative decode; once the effective footprint drops to 0.5 px, the gap is
  **5–20 dB**, and RD curves measured under pyramidal decode can run *backwards* (kodim01
  at `min_size=2`: 25.38 dB @ 1.05 bpp but only 23.71 dB @ 6.30 bpp). This bites hardest at
  `min_size` below the default 4 — not just a rounding caveat, a load-bearing threshold.
  See `docs/decisions.md` D11.
- **Padding is not measured.** Mars 1 pads to `virtual_size`; MSE is over the original
  W×H region only.

## `.ifs` bitstream (normative, from `mars_enc.c`, `coding_func.c`, `image_io.c`)

Bit packing is LSB-first within each value, MSB-first into bytes (`pack()` in
`image_io.c:180` shifts `sum` left and tests `value & 1`). One continuous bitstream; the
final byte is left-padded by `pack(-1, …)`.

```
Header (60 bits, in order):
  N_BITALFA      : 4    (default 4)
  N_BITBETA      : 4    (default 7)
  min_size       : 7    (default 4)
  max_size       : 7    (default 16)
  SHIFT          : 6    (domain step, default 4)
  image_width    : 12
  image_height   : 12
  int_max_alfa   : 8    (MAX_ALFA encoded as round(MAX_ALFA/8 · 256); default 1.0 -> 32)

Derived by BOTH sides (must match exactly):
  virtual_size          = 1 << ceil(log2(max(width, height)))
  bits_per_coordinate_w = ceil(log2(image_width  / SHIFT))   # integer division first
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

**The `dom_x`/`dom_y` width asymmetry is not a naming slip.** Mars 1 uses `x` for the
**row** axis and `y` for the **column** axis everywhere, so `dom_x` is a row coordinate and
is correctly sized by the image *height*. Reading it as a bug leads to a transposed decoder
plus a second bug to compensate. See `docs/decisions.md` D12.

**Leaves smaller than `min_size` exist.** The forced-subdivision branch never consults
`min_size`, so on dimensions that are not multiples of it the walk goes down to size 1, and
a size-1 leaf stores the raw pixel truncated to `N_BITBETA` bits. `docs/decisions.md` D13.

**`docs/mars1-format.md` is the full specification** — this table is the summary. That
document is independently validated against 142 golden fixtures by `just gate-3`; this
table is not.

## Quantisation (normative, from `coding_func.c`)

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

`s0` = pixel count; `s1`, `s2` = domain sum and sum of squares; `t0`, `t1`, `t2` = range
sum, cross term, and range sum of squares.

**These formulas are not the last word before packing.** If `|qalfa − zeroalfa| <=
zero_threshold` (true whenever `qalfa == 0` at the default `-z 0`), the encoder *discards*
the searched `qbeta` and refits it as `best_beta` = the block mean quantised. The split
decision still uses the RMS of the searched fit. See `docs/decisions.md` D15 and
`docs/mars1-format.md` §8.1.

**Mars 2 accumulates these moments as exact integers** (Step 6): domain pixels are 2:1
contractions, so `D = 4d` is an integer in `0..=1020` and every moment is exact in
i32/i64. f32 is not merely imprecise here — `Σr²` reaches 66.6 M, past f32's 16.7 M
exact-integer limit. Only the final `alfa`/`beta`/`rms` fit needs floating point.

## What Rust actually needs

A **read-only `.ifs` parser** plus a ~80-line iterative decoder for sanity checking. No
pyramidal decoder, no zoom, no writer beyond cross-checking against `decmars` — shell out
to the C binaries for those. The parser's exit criterion is **exact transform-count
equality** with the C encoder's reported `transforms`; that integer check validates the
spec without requiring any floating-point agreement.

The format spec in `docs/mars1-format.md` is only a spec if an independent implementation
(a throwaway Python validator) can parse every golden `.ifs` from the document alone.
