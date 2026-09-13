# The Mars 1 `.ifs` bitstream format

**Status:** normative for reading 1998 `.ifs` files · **Step 3 deliverable** ·
**Last updated:** 2026-09-13

This document specifies the file format written by the 1998 C encoder in
`reference/mars1/` (`encmars`) and read by `decmars`. It is written so that a decoder can
be implemented **without reading the C**, which is not a stylistic preference: it is the
step's exit criterion. `scripts/validate-ifs.py` is an independent parser written from this
document alone, and `just gate-3` fails if it disagrees with the reference on any golden
fixture. If the spec is not sufficient, the gate says so.

Everything below was read out of `image_io.c` (`pack`/`unpack`), `mars_enc.c` (header,
derived parameters), `coding_func.c` (`quadtree`, the six `*Coding` functions, `best_beta`),
`mars_dec.c` (`read_transformations`, `iterative_decoding`, `piramidal_decoding`),
`index_func.c` (the domain grid) and `def.h` (isometry numbering).

This is **not** a description of Mars 2. Mars 2 writes `.mars` (Step 10) and is not
bit-compatible with anything here. This format matters because it is the baseline's
bitstream, and Step 5 builds a reader for it.

---

## 0. The axis convention — read this first

**Mars 1 calls the row axis `x` and the column axis `y`.** Every array in the codec is
indexed `[row][column]`, and the variables named `atx`, `rx`, `dx`, `ptr_x`,
`bits_per_coordinate_h` all belong to the **row** axis; `aty`, `ry`, `dy`, `ptr_y`,
`bits_per_coordinate_w` all belong to the **column** axis. The bounds checks read
`atx >= image_height` and `aty >= image_width`, and the domain enumeration in
`index_func.c` is `for(i = 0; i < image_height - 2*size + 1; i += SHIFT)` over `i`, the
`x`/`ptr_x` axis.

This one sentence dissolves the format's most-reported oddity. `implementation-plan.md`
§4 Step 3 and the `mars1-reference` skill both say that packing `dom_x` with
`bits_per_coordinate_h` "looks like a naming slip in the original … Preserve it exactly;
do not fix it." **There is nothing to preserve.** `dom_x` is a row coordinate, a row
coordinate ranges over the image height, and sizing it by the height is correct. Reading it
as a bug invites an implementer to write a transposed decoder and then add a second,
compensating bug — see `docs/decisions.md` D12.

Throughout this document: **row** and **column** are used in place of `x` and `y`, except
where a C identifier is quoted.

---

## 1. Container

One continuous bitstream, no framing, no magic number, no checksum, no trailer. A `.ifs`
file is the 60-bit header immediately followed by the quadtree, padded to a byte boundary.

A consequence worth stating: **the format is not self-delimiting and has no integrity
check.** A truncated file decodes to a partial image rather than an error, and a corrupt
bit shifts every subsequent field. Nothing in this document can detect that; the golden
fixtures' SHA-256s in `fixtures/mars1/manifest.toml` are the only integrity mechanism the
project has for these files.

---

## 2. Bit packing

`pack(size, value, f)` writes the low `size` bits of `value`, **least-significant bit
first**. Bits are accumulated into a byte **most-significant bit first**: the first bit
written lands in bit 7 of the first byte, the second in bit 6, and so on.

So for a value `v` of width `n`, the bit at stream position `p + i` (0-based from the start
of the value) is `(v >> i) & 1`, and the bit at stream position `q` lives in byte `q / 8`
at bit position `7 - (q % 8)`.

Reading is the exact inverse (`unpack`): accumulate `n` bits from the stream in order,
placing the `i`-th into bit `i` of the result.

### The final byte

`pack(-1, …)` flushes a partial byte. Reading `pack` closely: after `m` bits (`m` not a
multiple of 8) the accumulator holds them at bit positions `1..m` and the flush is
`fputc(sum << (8 - ptr))` with `ptr == m + 1`, which places the first-written bit at bit 7.

**The data is left-aligned and the `8 - (m mod 8)` unused low-order bits are zero.**
`implementation-plan.md` describes this as "the final byte is left-padded", which is the
wrong way round; the padding is on the right. If the total bit count is an exact multiple
of 8 no extra byte is written.

Total file size is therefore exactly `ceil(total_bits / 8)` bytes, and this is checkable:
`encmars` prints `%d bytes written`, and the two must agree.

### Worked example

The first eight bytes of every `flat128` fixture are `2e 20 10 80 08 00 80 40`. As a bit
string, most-significant bit of each byte first:

```
00101110 00100000 00010000 10000000 00001000 00000000 10000000 01000000
|<-4->|<-4->|<---7--->|<---7--->|<--6->|<-----12----->|<-----12----->|<--8-->|
```

Reading each field low-bit-first out of that stream gives `N_BITALFA = 4`,
`N_BITBETA = 7`, `min_size = 4`, `max_size = 16`, `SHIFT = 4`, `image_width = 256`,
`image_height = 256`, `int_max_alfa = 32`. §12 walks a whole file byte by byte.

---

## 3. Header — 60 bits

Written by `mars_enc.c:199-206`, read by `mars_dec.c:65-72`. In order:

| # | Field | Bits | 1998 default | Notes |
|---:|---|---:|---:|---|
| 1 | `N_BITALFA` | 4 | 4 | quantiser width for the scale factor; `-A`, range 1–15 |
| 2 | `N_BITBETA` | 4 | 7 | quantiser width for the offset; `-B`, range 1–15 |
| 3 | `min_size` | 7 | 4 | smallest range block the *split decision* may produce; `-m` |
| 4 | `max_size` | 7 | 16 | largest range block; `-M` |
| 5 | `SHIFT` | 6 | 4 | domain grid step; `-d` |
| 6 | `image_width` | 12 | — | columns |
| 7 | `image_height` | 12 | — | rows |
| 8 | `int_max_alfa` | 8 | 32 | `MAX_ALFA` in 1/256ths of 8.0; `-y` |

Field 3 says *the split decision*: leaves smaller than `min_size` do occur, from a
different mechanism — see §5.3.

`N_BITALFA` and `N_BITBETA` are 4-bit fields, so a value of 15 is representable and 16 is
not; the CLI enforces 1–15. `min_size` and `max_size` are 7-bit fields holding a power of
two in 2–64. `image_width` and `image_height` are 12-bit, so **the format cannot express a
dimension above 4095**, and there is no check: encoding a 5000-pixel-wide image writes
`5000 & 4095 = 904` and produces a file that decodes to the wrong size, silently.

---

## 4. Derived parameters

These are computed **identically on both sides** and never transmitted. An implementation
that derives any of them differently from the rules here will desynchronise from the
bitstream, usually several hundred bits in.

### 4.1 `MAX_ALFA`

```
MAX_ALFA = int_max_alfa / 256 · 8.0
```

Exact in binary64 for every `int_max_alfa` (it is a dyadic rational). The default 32 gives
`MAX_ALFA = 1.0`.

The encoder derives `int_max_alfa` from the requested value the other way first, and
**then overwrites its own `MAX_ALFA` with the round-tripped value** (`mars_enc.c:188-192`),
so encoder and decoder always agree even when `-y` asks for something unrepresentable:

```
int_max_alfa = clamp(int(0.5 + MAX_ALFA_requested / 8.0 · 256), 0, 255)
MAX_ALFA     = int_max_alfa / 256 · 8.0
```

### 4.2 `virtual_size`

```
virtual_size = 1 << ceil(log2(max(image_width, image_height)))
```

The quadtree is walked over a `virtual_size × virtual_size` square, so a non-square or
non-power-of-two image is conceptually padded up. **The padding is never coded and never
measured** — blocks entirely outside the image emit nothing (§5.1), and §M2 defines MSE
over the original W×H region only.

| image | `virtual_size` |
|---|---:|
| 256×256 | 256 |
| 512×512 | 512 |
| 250×250 | 256 |
| 129×127 | 256 |
| 768×512 (Kodak) | 1024 |

### 4.3 `bits_per_coordinate_w` / `bits_per_coordinate_h`

```
bits_per_coordinate_w = ceil(log2(image_width  / SHIFT))     # integer division first
bits_per_coordinate_h = ceil(log2(image_height / SHIFT))     # integer division first
```

The division is C integer division and truncates **before** the logarithm: at
`image_width = 250`, `SHIFT = 4` the argument is 62, not 62.5, giving 6 bits rather than 6
— here they agree, but at `image_width = 129`, `SHIFT = 4` the argument is 32, not 32.25,
giving **5** bits where 32.25 would give 6. Getting this wrong costs one bit per domain
coordinate and desynchronises the stream.

Implement it as exact integer arithmetic:

```
q    = dim // SHIFT
bits = 0 if q <= 1 else ceil_log2(q)          # ceil_log2(q) == (q-1).bit_length()
```

The C computes it in binary64 as `ceil(log(q) / log(2.0))`, which for `q` an exact power of
two is one ULP away from returning a value one too large. **It does not, for any dimension
this project uses** — verified over `dim ∈ {64, 127, 129, 250, 256, 512, 768}` ×
`SHIFT ∈ {2, 4, 8, 16, 32}`, all 35 combinations agreeing with the exact integer rule. The
hazard is real and unarmed; use the integer rule and the question does not arise.

`q == 0` (an image narrower than `SHIFT`) evaluates `log(0)` and is undefined behaviour in
the C. Treat it as unrepresentable.

---

## 5. The quadtree

The walk is `quadtree(0, 0, virtual_size)` on the encoder (`coding_func.c:1096`) and
`read_transformations(0, 0, virtual_size)` on the decoder (`mars_dec.c:184`). The two are
structurally identical, which is what makes the format parseable: **every branch that emits
bits is mirrored by a branch that reads them, and every branch that emits nothing is
mirrored by a branch that reads nothing.**

A parser needs only `read_transformations`. Given `(row, col, size)`:

```
parse(row, col, size):

    # 5.1 — wholly outside the image
    if row >= image_height or col >= image_width:
        return                                     # reads nothing

    # 5.2 — too big, or crossing the right/bottom edge: forced subdivision
    if size > max_size or row + size > image_height or col + size > image_width:
        parse(row,          col,          size/2)  # reads nothing itself
        parse(row + size/2, col,          size/2)
        parse(row,          col + size/2, size/2)
        parse(row + size/2, col + size/2, size/2)
        return

    # 5.3 — an ordinary block: split flag, then children or a leaf
    if size > min_size and read_bits(1) == 1:
        parse(row,          col,          size/2)
        parse(row + size/2, col,          size/2)
        parse(row,          col + size/2, size/2)
        parse(row + size/2, col + size/2, size/2)
    else:
        emit_leaf(row, col, size)                  # §6
```

### 5.1 Child order is row-major-by-column, and it is not Z-order

The four recursive calls are, in order:

```
(row, col)   (row + size/2, col)   (row, col + size/2)   (row + size/2, col + size/2)
   top-left       bottom-left            top-right            bottom-right
```

That is **NW, SW, NE, SE** — the second child is the one *below* the first, not the one to
its right. Both `quadtree` and `read_transformations` use this order, so it is normative.

This is the detail most likely to be got wrong and least likely to be caught: a parser that
uses NW, NE, SW, SE consumes exactly the same bits in exactly the same order and produces
exactly the same **number** of leaves. Only their **positions** differ. Transform-count
equality cannot see it, and neither can re-serialising the parsed tree. It is caught by
reconstructing the partition geometry (`quadtree.pgm`, §13) or by decoding the image.

### 5.2 The split flag is conditional

The 1-bit split flag exists **only when `size > min_size`**. At `size == min_size` the block
is a leaf with no flag, and at `size < min_size` (§5.3) likewise. Note also the C's
short-circuit: `if (size > min_size && unpack(1, input))` — when `size <= min_size` no bit
is consumed at all.

### 5.3 Leaves below `min_size` exist

**The forced-subdivision branch of §5.2 does not consult `min_size`.** On an image whose
dimensions are not multiples of `min_size`, the edge-crossing test keeps firing all the way
down, and produces leaves of size `min_size/2`, `min_size/4`, … and ultimately **size 1**.

These are ordinary leaves in every syntactic respect (no split flag, since
`size > min_size` is false), and they are counted in the encoder's reported `transforms`.
Their positions are **pure geometry**: they do not depend on the method, on the RMS
threshold, or on the image content. Measured over the golden fixtures at `-m 4 -M 16`:

| fixture | leaves of size 1 | of size 2 |
|---|---:|---:|
| `mixed_129x127` (129×127) | 255 | 64 |
| `mixed_250x250` (250×250) | 0 | 249 |
| all 64², 256² and 512² fixtures | 0 | 0 |

For 129×127 the 255 is row 126 in full (129 leaves, since every column's size-2 block
crosses the bottom edge) plus column 128 for rows 0–125 (126 leaves). Both counts are
identical across all three methods and all three rates in the golden set, which is the
observable form of "pure geometry".

The size-1 count survives a change of `min_size` — `mixed_129x127` at `-m 2` still has
exactly 255 — because the forced branch never consulted `min_size` in the first place. The
size-2 count does not, because at `-m 2` size-2 blocks become ordinary leaves as well.

**Size-1 leaves carry a raw pixel, truncated.** All six `*Coding` functions special-case
`size == 1` (`tip == 0`) with `*qbet = image[row][col]`, `*qalf = zeroalfa`, rms 0 — the
"quantised offset" is the pixel value itself, 0–255, which is then packed into
`N_BITBETA = 7` bits and **loses its top bit**. A pixel of 200 is stored as 72 and decodes
to `72/127 · 255 = 144.6 → 145`. This is a defect in the 1998 codec, it is reachable only
on images whose dimensions are not multiples of `min_size`, and it is normative for anyone
parsing these files.

---

## 6. Leaf payload

```
emit_leaf(row, col, size):
    qalfa = read_bits(N_BITALFA)
    qbeta = read_bits(N_BITBETA)
    if qalfa != 0:                                  # `zeroalfa`, always 0
        isometry = read_bits(3)
        dom_row  = SHIFT · read_bits(bits_per_coordinate_h)
        dom_col  = SHIFT · read_bits(bits_per_coordinate_w)
    else:
        isometry, dom_row, dom_col = 0, 0, 0        # not present in the stream
```

`zeroalfa` is a global set to 0 in both `mars_enc.c:193` and `mars_dec.c:77` and never
changed. It exists as a variable because the 1998 code contemplated a nonzero "no
transform" codeword; nothing sets one. **A leaf with `qalfa == 0` is a DC-only block** — a
constant fill, with no domain reference at all — and costs `N_BITALFA + N_BITBETA` bits
instead of `N_BITALFA + N_BITBETA + 3 + bits_h + bits_w`.

The encoder counts these and prints them as `Zero_alfa_transformations`, which gives a
parser a second exact integer to check itself against, for free.

A leaf therefore costs, at the 1998 defaults on a 256² image
(`bits_h = bits_w = 6`):

| leaf kind | bits |
|---|---:|
| DC-only (`qalfa == 0`), `size > min_size` | 1 + 4 + 7 = **12** |
| DC-only, `size <= min_size` | 4 + 7 = **11** |
| with a domain, `size > min_size` | 1 + 4 + 7 + 3 + 6 + 6 = **27** |
| with a domain, `size <= min_size` | 4 + 7 + 3 + 6 + 6 = **26** |

### Legal ranges for the domain coordinates

`index_func.c` enumerates domains as
`for(i = 0; i < image_height - 2·size + 1; i += SHIFT)` over rows and the analogous loop
over columns, so a well-formed file has

```
0 <= dom_row <= image_height - 2·size,   dom_row  ≡ 0 (mod SHIFT)
0 <= dom_col <= image_width  - 2·size,   dom_col  ≡ 0 (mod SHIFT)
```

The coordinate always fits its field: `dom_row / SHIFT < image_height / SHIFT <=
2^bits_per_coordinate_h`. A parser should check these bounds — not because the encoder
violates them, but because a corrupt or desynchronised stream announces itself here first,
several hundred bits before it becomes visible as garbage in the image.

---

## 7. Dequantisation — what a decoder computes

From `mars_dec.c:216-221`, in binary64:

```
alfa = qalfa / 2^N_BITALFA · MAX_ALFA
beta = qbeta / (2^N_BITBETA − 1) · (1 + |alfa|) · 255
if alfa > 0:  beta −= alfa · 255
```

Note the asymmetry between the two divisors: `alfa` divides by `2^N_BITALFA` (so `qalfa`
can never reach `MAX_ALFA`; the top code is `15/16 · MAX_ALFA` at the default width), while
`beta` divides by `2^N_BITBETA − 1` (so `qbeta` *can* reach the top of its range). This is
not a typo in this document; it is what both binaries do.

`alfa > 0` is a strict comparison and `alfa` is exactly 0 when `qalfa` is 0, so the `−255α`
term applies to every non-DC leaf and to no DC leaf.

---

## 8. Quantisation — what the encoder computed

Not needed to *read* a file. Recorded here because Step 6 must reproduce it, and because
one part of it is missing from every other description of this format.

With `s0` = pixel count (`size²`), `s1`/`s2` the contracted domain block's sum and sum of
squares, and `t0`/`t1`/`t2` the range block's sum, its cross term with the domain, and its
sum of squares:

```
det   = s0·s2 − s1²
alfa  = 0                        if det == 0
        (s0·t1 − s1·t0) / det    otherwise
alfa  = max(alfa, 0)                                  # negative scalings are not coded

qalfa = clamp(int(0.5 + alfa / MAX_ALFA · 2^N_BITALFA), 0, 2^N_BITALFA − 1)
alfa' = qalfa / 2^N_BITALFA · MAX_ALFA

beta  = (t0 − alfa'·s1) / s0
if alfa' > 0:  beta += alfa' · 255
qbeta = clamp(int(0.5 + beta / ((1 + |alfa'|)·255) · (2^N_BITBETA − 1)), 0, 2^N_BITBETA − 1)
beta' = qbeta / (2^N_BITBETA − 1) · (1 + |alfa'|) · 255
if alfa' > 0:  beta' −= alfa' · 255

rms   = sqrt((t2 − 2·alfa'·t1 − 2·beta'·t0 + alfa'²·s2 + 2·alfa'·beta'·s1 + s0·beta'²) / s0)
```

`int(...)` is C truncation toward zero, so `int(0.5 + v)` is round-half-up for `v >= 0`;
the clamps make the negative side unreachable. The RMS is the quantity compared against
`T_RMS` to decide whether to split.

### 8.1 The zero-alfa override — the part everyone misses

Before packing, `coding_func.c:1169-1172` runs:

```
if |qalfa − zeroalfa| <= zero_threshold:            # zero_threshold is `-z`, default 0
    qbeta = best_beta(row, col, size, 0.0)
    qalfa = zeroalfa
```

At the default `zero_threshold = 0` this fires whenever `qalfa == 0`, which is often — 100%
of leaves on a flat image. **The `qbeta` that was found by the search is discarded** and
replaced by a DC-only refit:

```
best_beta(row, col, size, 0.0):
    mean  = (sum of the range block's pixels) / size²
    return clamp(int(0.5 + mean / 255 · (2^N_BITBETA − 1)), 0, 2^N_BITBETA − 1)
```

which is the right thing to do — with `alfa = 0` the optimal offset is the block mean, and
the searched `qbeta` was optimal for a domain that is no longer being referenced — but a
Rust encoder that omits it will produce different `qbeta` values on every DC leaf and a
measurably different image, for a reason that is nowhere in the plan's spec sketch.

Note that `best_beta` is called with `alfa = 0.0` literal, so the `+alfa·255` pre-bias in
its body is skipped and the divisor is exactly 255.

Note also that the split decision uses the RMS of the *searched* fit, computed **before**
this override. A block can therefore be kept as a leaf on the strength of a domain match
that is then thrown away.

---

## 9. Isometries

`def.h:37-44`. The 3-bit field takes these values:

| value | name | maps range `(i, j)` from domain sample `(u, v)` |
|---:|---|---|
| 0 | `IDENTITY` | `(u, v) = (i, j)` |
| 1 | `L_ROTATE90` | rotate left 90° |
| 2 | `R_ROTATE90` | rotate right 90° |
| 3 | `ROTATE180` | rotate 180° |
| 4 | `R_VERTICAL` | reflect about the vertical axis (columns reversed) |
| 5 | `R_HORIZONTAL` | reflect about the horizontal axis (rows reversed) |
| 6 | `F_DIAGONAL` | transpose (reflect about the main diagonal) |
| 7 | `S_DIAGONAL` | reflect about the anti-diagonal |

**Note that 1 is *left* and 2 is *right*.** The natural ordering is the other way round and
the two are otherwise indistinguishable in a round-trip test that applies the same table on
both sides.

The authoritative definition is the decoder's traversal order, because that is what the
pixels do. For a leaf at `(row, col, size)` with domain at `(dom_row, dom_col)`, define the
contracted domain sample

```
D(u, v) = ( I[dom_row + 2u][dom_col + 2v] + I[dom_row + 2u + 1][dom_col + 2v]
          + I[dom_row + 2u][dom_col + 2v + 1] + I[dom_row + 2u + 1][dom_col + 2v + 1] ) / 4
```

for `0 <= u, v < size` — an unweighted 2×2 box average, in binary64, of a `2size × 2size`
region. Then, writing `i` for the row offset and `j` for the column offset within the range
block, `mars_dec.c:285-351` walks:

| value | destination `(i, j)` for source `(u, v)` |
|---:|---|
| 0 `IDENTITY` | `i = u`, `j = v` |
| 1 `L_ROTATE90` | `j = u`, `i = size−1−v` |
| 2 `R_ROTATE90` | `j = size−1−u`, `i = v` |
| 3 `ROTATE180` | `i = size−1−u`, `j = size−1−v` |
| 4 `R_VERTICAL` | `i = u`, `j = size−1−v` |
| 5 `R_HORIZONTAL` | `i = size−1−u`, `j = v` |
| 6 `F_DIAGONAL` | `i = v`, `j = u` |
| 7 `S_DIAGONAL` | `i = size−1−v`, `j = size−1−u` |

---

## 10. Decoding

### 10.1 Iterative (`decmars -i`)

The bitstream is an iterated function system; decoding is fixed-point iteration and needs
no information beyond the transform list.

```
I := 128 everywhere                    # image_height × image_width, one byte per pixel
repeat `iterations` times (default 10):
    J := new buffer
    for each leaf, in the order parsed:
        for u, v in 0..size:
            (i, j) := isometry_map(u, v)          # §9
            J[row + i][col + j] := bound(0.5 + D(u, v) · alfa + beta)
    I := J
```

with

```
bound(a) = 0    if a <  0.0
           255  if a > 255.0
           a    otherwise                          # then truncated to unsigned char
```

Three details that change the last grey level and are easy to miss:

- **The `0.5` is added to the product, not to the sum.** The C expression is
  `0.5 + pixel * alfa + beta`, and `+` associates left, so it evaluates as
  `(0.5 + pixel·alfa) + beta`. Computing `(pixel·alfa + beta) + 0.5` instead is a
  different binary64 result in the last bit, and the truncation below turns some of those
  into a whole grey level. Written out: `v = (0.5 + D·alfa) + beta`.
- `bound` is a macro on a `double` whose result is **assigned to an `unsigned char`**, so
  the truncation happens *after* the clamp, and `bound(0.5 + v)` is round-half-up rather
  than a rounded-then-clamped value. The clamp compares against `255.0`, not `255`, so a
  value of exactly 255.4 clamps to 255 by truncation, not by the bound.
- The buffers are **double-buffered and swapped** each iteration: every leaf in one
  iteration reads the *previous* iteration's image. An in-place implementation converges to
  a different (and slightly better-looking) image.

The domain read `D(u, v)` samples `I` at rows `dom_row .. dom_row + 2·size − 1`. Since
`dom_row + 2·size <= image_height` (§6) that is always in bounds — but note the decoder
allocates `2 + image_height` rows and relies on it.

### 10.2 Pyramidal (the `decmars` default) — and its defect

The 1998 default is *not* §10.1. `decmars` with no flags runs `piramidal = 1`: it decodes
at `1/2^lev` resolution, then doubles resolution `lev` times, replacing the 2×2 average with
a single pixel read at `(dom_row >> 1, dom_col >> 1)`.

```
lev = 0;  m = min(image_width, image_height);  step = SHIFT
while m >= 200 and step is even:  m /= 2;  step /= 2;  lev += 1
```

giving `lev = 1` for 256², `lev = 2` for 512² and 768×512.

**Both modes converge to the same IFS fixed point** and at the default 10 iterations agree
to ~0.002 dB — pyramidal is a convergence accelerator, not an approximation (`decisions.md`
D9). **Except** when a range block falls below one pixel at the pyramid's reduced
resolution: the rule is `min_size / 2^lev >= 1`, and at half a pixel pyramidal loses 5+ dB
and the RD curve runs backwards (`decisions.md` D11). At `min_size = 2` on a 512² image the
two modes differ by up to 19.8 dB.

Mars 2 implements §10.1 only. §10.2 is specified here because the golden fixtures record
both decodes and because Step 19 (progressive decoding) reintroduces exactly this hazard.

---

## 11. Worked example A — `flat128`, in closed form

`fixtures/images/flat128.raw` is 256×256, constant 128. At the 1998 defaults
(`-m 4 -M 16 -d 4 -A 4 -B 7 -y 1.0 -r 8`) every number in the file can be derived by hand,
which makes it the first thing to run a new parser against.

- A constant block has `det = s0·s2 − s1² = 0`, so `alfa = 0` and `qalfa = 0`: **every leaf
  is DC-only**, no isometry, no domain coordinates.
- §8.1 fires on every leaf, so `qbeta = int(0.5 + 128/255 · 127) = int(64.249…) = **64**`
  everywhere.
- The reconstruction is `alfa = 0`, `beta = 64/127 · 255 = 128.50393…`, so every pixel is
  `bound(0.5 + 128.50393…) = 129.00393… → **129**`. Not 128 — the file is one grey level
  brighter than the original, in both decode modes.
- MSE is exactly 1, so **PSNR = 10·log10(255²) = 48.1308 dB**.
- The RMS of each leaf is `sqrt(t2 − 2βt0 + s0β²)/√s0 = 0.5039…`, well under `T_RMS = 8`, so
  nothing splits: the partition is exactly the `16×16` grid, **256 leaves**.
- Bits: 60 header + 256 × (1 split flag + 4 + 7) = 3132 → **392 bytes**, with the last byte
  holding 4 data bits and a zero low nibble.

All eight of those figures are asserted by `just gate-3`.

---

## 12. Worked example B — a 24-byte file, byte by byte

`fixtures/mars1/tiny64.raw` is a 64×64 image built for this walkthrough: three flat
quadrants (100 top-left, 170 top-right, 60 bottom-right) and a vertical ramp
`40 + 4·(row−32)` in the bottom-left. Encoded with

```
encmars -F -W 64 -H 64 -r 8.0 -m 16 -M 32 -d 4 -A 4 -B 7 -y 1.0 tiny64.raw tiny64.ifs
```

it produces 7 transforms, 3 of them DC-only, in 24 bytes. `virtual_size = 64`,
`bits_per_coordinate_w = bits_per_coordinate_h = ceil(log2(64/4)) = 4`.

Regenerate this section with
`.venv-crossval/bin/python scripts/validate-ifs.py --walkthrough`; it is reproduced
verbatim in `fixtures/mars1/tiny64.walkthrough.txt`, and `just gate-3` fails if either this
block or that file has gone stale.

Read the leaf order against §5.1: after the first quadrant, the walk visits `(32,0)` —
*below* — before `(0,32)`. That is the NW, SW, NE, SE child order, visible in a real file.

<!-- WALKTHROUGH:BEGIN -->
```
24 bytes, 185 bits of payload + 7 of padding, 7 leaves (3 DC-only)

        bits      bytes  field                        value
------------  ---------  ---------------------------- ----------------------------------
    0..4              0  header N_BITALFA             4
    4..8              0  header N_BITBETA             7
    8..15             1  header min_size              16
   15..22           1-2  header max_size              32
   22..28           2-3  header SHIFT                 4
   28..40           3-4  header image_width           64
   40..52           5-6  header image_height          64
   52..60           6-7  header int_max_alfa          32  MAX_ALFA = 1
                         (0,0) 64x64                  forced subdivision (§5.2), no bits
   60..61             7  (0,0) 32x32 split            0 = leaf
   61..65           7-8  (0,0) 32x32 qalfa            0  -> alfa 0.000000
   65..72             8  (0,0) 32x32 qbeta            50  -> beta 100.393701
                         (0,0) 32x32 DC-only          qalfa == 0: no isometry, no domain (§6)
   72..73             9  (32,0) 32x32 split           1 = subdivide
   73..77             9  (32,0) 16x16 qalfa           8  -> alfa 0.500000
   77..84          9-10  (32,0) 16x16 qbeta           49  -> beta 20.078740
   84..87            10  (32,0) 16x16 isometry        0
   87..91         10-11  (32,0) 16x16 dom_row         8 -> row 32
   91..95            11  (32,0) 16x16 dom_col         0 -> col 0
   95..99         11-12  (48,0) 16x16 qalfa           8  -> alfa 0.500000
   99..106        12-13  (48,0) 16x16 qbeta           70  -> beta 83.326772
  106..109           13  (48,0) 16x16 isometry        0
  109..113        13-14  (48,0) 16x16 dom_row         8 -> row 32
  113..117           14  (48,0) 16x16 dom_col         0 -> col 0
  117..121        14-15  (32,16) 16x16 qalfa          8  -> alfa 0.500000
  121..128           15  (32,16) 16x16 qbeta          49  -> beta 20.078740
  128..131           16  (32,16) 16x16 isometry       0
  131..135           16  (32,16) 16x16 dom_row        8 -> row 32
  135..139        16-17  (32,16) 16x16 dom_col        0 -> col 0
  139..143           17  (48,16) 16x16 qalfa          8  -> alfa 0.500000
  143..150        17-18  (48,16) 16x16 qbeta          70  -> beta 83.326772
  150..153        18-19  (48,16) 16x16 isometry       0
  153..157           19  (48,16) 16x16 dom_row        8 -> row 32
  157..161        19-20  (48,16) 16x16 dom_col        0 -> col 0
  161..162           20  (0,32) 32x32 split           0 = leaf
  162..166           20  (0,32) 32x32 qalfa           0  -> alfa 0.000000
  166..173        20-21  (0,32) 32x32 qbeta           85  -> beta 170.669291
                         (0,32) 32x32 DC-only         qalfa == 0: no isometry, no domain (§6)
  173..174           21  (32,32) 32x32 split          0 = leaf
  174..178        21-22  (32,32) 32x32 qalfa          0  -> alfa 0.000000
  178..185        22-23  (32,32) 32x32 qbeta          30  -> beta 60.236220
                         (32,32) 32x32 DC-only        qalfa == 0: no isometry, no domain (§6)
  185..192           23  padding                      zero, low bits of the final byte (§2)

hexdump
  0000  2e 08 08 80 20 02 00 40  00101110 00001000 00001000 10000000 00100000 00000010 00000000 01000000
  0008  26 8c 60 20 2c 40 80 c6  00100110 10001100 01100000 00100000 00101100 01000000 10000000 11000110
  0010  02 02 c4 08 02 a8 1e 00  00000010 00000010 11000100 00001000 00000010 10101000 00011110 00000000
```
<!-- WALKTHROUGH:END -->

---

## 13. How this document is validated

`just gate-3` runs `scripts/validate-ifs.py` over every golden fixture in
`fixtures/mars1/`. The validator was written from this document, does not import anything
from the project, and never reads `reference/mars1/`. It applies four checks, all exact
equalities — no tolerances anywhere:

1. **Transform count.** The number of leaves parsed equals the `Number of transformations`
   the C encoder printed, recorded in `manifest.toml`.
2. **DC-leaf count.** The number of leaves with `qalfa == 0` equals the C's
   `Zero_alfa_transformations`. Free, and it independently exercises the branch in §6.
3. **Byte-exact re-serialisation.** The parsed tree, packed again by §2–§6, reproduces the
   original `.ifs` byte for byte, including the padding of §2. This proves every field
   width and every field order.
4. **Byte-exact partition geometry.** The validator reconstructs `quadtree.pgm` — the
   partition rendering `encmars -Q` writes — from the parsed tree alone, and compares it to
   the C's. **This is the check that pins §5.1.** Checks 1–3 are all blind to child order:
   a NW/NE/SW/SE parser consumes identical bits and emits an identical count. Only the
   geometry distinguishes them.

and one end-to-end check:

5. **Decoded-image equality.** An iterative decoder written from §7–§10.1 reproduces
   `decmars -i` output byte for byte. This is the literal form of the step's exit criterion
   — the transform list is recovered well enough to regenerate the reference's own image.

Check 5 is the one that could legitimately fail for a reason that is not a spec error: the
pinned build flags do not disable FP contraction, so clang may emit `fmadd` for
`pixel·alfa + beta`. It does not, on this toolchain — agreement is exact on every fixture —
and the gate asserts exact equality rather than a pixel tolerance, so a future toolchain
that does contract will fail loudly instead of drifting.

---

## 14. Hazard checklist

Everything in this document that has bitten someone, in one list.

| # | Hazard | Where |
|---:|---|---|
| 1 | `x` is the **row** axis; `dom_x` is correctly sized by the image *height* | §0 |
| 2 | The final byte is **right**-padded, not left-padded | §2 |
| 3 | `bits_per_coordinate` truncates `dim / SHIFT` **before** the log | §4.3 |
| 4 | Child order is NW, **SW**, NE, SE — and no bit-level check can see a mistake | §5.1 |
| 5 | Leaves **below `min_size`** occur on non-multiple dimensions, down to size 1 | §5.3 |
| 6 | A size-1 leaf stores a raw pixel truncated to `N_BITBETA` bits | §5.3 |
| 7 | `qalfa == 0` means **no isometry and no domain in the stream** | §6 |
| 8 | `alfa` divides by `2^N`, `beta` by `2^N − 1` | §7 |
| 9 | The searched `qbeta` is **discarded and refit** whenever `qalfa == 0` | §8.1 |
| 10 | Isometry 1 is *left* rotate, 2 is *right* | §9 |
| 11 | `bound()` clamps a double, then **truncates** on assignment | §10.1 |
| 12 | Decoding is **double-buffered**; in-place iteration converges elsewhere | §10.1 |
| 13 | Pyramidal decode breaks when `min_size / 2^lev < 1` | §10.2 |
| 14 | 12-bit dimension fields wrap silently above 4095 | §3 |
| 15 | Paths over 49 bytes overflow `filein[50]` in the 1998 CLI | `decisions.md` D3 |
