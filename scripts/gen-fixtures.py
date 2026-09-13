#!/usr/bin/env python3
"""Generate the Mars 2 synthetic fixture images (corpus role `fixtures/`, M9).

Standard library only, fully deterministic. The images are NOT committed: they
are regenerated from this script, and `manifest.toml` records the SHA-256 of
every output so a regeneration can be *proved* identical -- M7, "a result you
cannot regenerate from its own row is not a result". `--check` is the CI form.

Output format: 8-bit grayscale, headerless, row-major, width-fast. That is the
layout `readimage_raw()` expects: `reference/mars1/image_io.c:158` allocates
`matrix_allocate(image, image_width, image_height, PIXEL)`, i.e. `image_height`
rows of `image_width` pixels, indexed `image[row][col]`. Pass each image to the
reference codec as `-W <width> -H <height>`.

Each fixture exists to catch one named failure mode; see `purpose` in the
manifest. Deliberate properties worth knowing before you interpret a run:

  * `flat128` can round-trip exactly at a loose RMS threshold, which is how the
    "identical images -> PSNR is null, not a large float" path gets exercised.
  * `mixed_250x250` and `mixed_129x127` have dimensions that are not multiples
    of SHIFT (default 4), so they exercise the integer-division truncation in
    `bits_per_coordinate_* = ceil(log2(W / SHIFT))`, plus heavy padding to
    `virtual_size` and the edge force-split path.
  * `mixed_129x127` is below the 176-pixel MS-SSIM floor on both axes. It must
    be *flagged and excluded* by the harness, never silently rescaled.

Usage:
  gen-fixtures.py                  write images + manifest.toml
  gen-fixtures.py --check          regenerate in memory, verify against manifest
  gen-fixtures.py --pgm            also write .pgm copies for eyeballing
  gen-fixtures.py --out DIR        default: <repo>/fixtures/images
"""

import argparse
import hashlib
import math
import pathlib
import random
import sys

SEED = 20260913  # pinned; changing it invalidates every golden fixture downstream

# ---------------------------------------------------------------------------
# Isometries, transcribed from `flips()` in reference/mars1/coding_func.c:1200.
# Indices are the #defines in reference/mars1/def.h:35-42. These are normative.
# ---------------------------------------------------------------------------
ISOMETRY_NAMES = {
    0: "IDENTITY",
    1: "L_ROTATE90",
    2: "R_ROTATE90",
    3: "ROTATE180",
    4: "R_VERTICAL",
    5: "R_HORIZONTAL",
    6: "F_DIAGONAL",
    7: "S_DIAGONAL",
}


def isometry(block, size, iso):
    """Apply Mars 1 isometry `iso` to a size*size list-of-rows. Mirrors flips()."""
    n = size
    if iso == 0:    # IDENTITY        flip[i][j] = block[i][j]
        return [[block[i][j] for j in range(n)] for i in range(n)]
    if iso == 1:    # L_ROTATE90      flip[i][j] = block[j][n-i-1]
        return [[block[j][n - i - 1] for j in range(n)] for i in range(n)]
    if iso == 2:    # R_ROTATE90      flip[i][j] = block[n-j-1][i]
        return [[block[n - j - 1][i] for j in range(n)] for i in range(n)]
    if iso == 3:    # ROTATE180       flip[i][j] = block[n-i-1][n-j-1]
        return [[block[n - i - 1][n - j - 1] for j in range(n)] for i in range(n)]
    if iso == 4:    # R_VERTICAL      flip[i][j] = block[i][n-j-1]
        return [[block[i][n - j - 1] for j in range(n)] for i in range(n)]
    if iso == 5:    # R_HORIZONTAL    flip[i][j] = block[n-i-1][j]
        return [[block[n - i - 1][j] for j in range(n)] for i in range(n)]
    if iso == 6:    # F_DIAGONAL      flip[i][j] = block[j][i]
        return [[block[j][i] for j in range(n)] for i in range(n)]
    if iso == 7:    # S_DIAGONAL      flip[i][j] = block[n-j-1][n-i-1]
        return [[block[n - j - 1][n - i - 1] for j in range(n)] for i in range(n)]
    raise ValueError(f"isometry index out of range: {iso}")


# ---------------------------------------------------------------------------
# Small image helper: bytearray, row-major, width-fast.
# ---------------------------------------------------------------------------
class Img:
    def __init__(self, width, height, fill=0):
        self.w = width
        self.h = height
        self.d = bytearray([fill]) * (width * height)

    def put(self, row, col, value):
        self.d[row * self.w + col] = value

    def blit(self, row, col, block):
        """Place a list-of-rows block with its top-left at (row, col)."""
        for i, blockrow in enumerate(block):
            base = (row + i) * self.w + col
            self.d[base:base + len(blockrow)] = bytes(blockrow)


def q8(x):
    """Round to nearest, half away from zero, clamped to 8-bit."""
    v = math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)
    return 0 if v < 0 else (255 if v > 255 else int(v))


def rng(name):
    """Per-image stream. Seeding from a str goes through SHA-512 in CPython's
    `Random.seed(version=2)`, so it is stable across runs and unaffected by
    PYTHONHASHSEED -- unlike seeding from a tuple, which would use hash()."""
    return random.Random(f"{SEED}:{name}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def f_flat128():
    return Img(256, 256, 128)


def f_ramp_h():
    im = Img(256, 256)
    for col in range(im.w):
        v = q8(255.0 * col / (im.w - 1))
        for row in range(im.h):
            im.put(row, col, v)
    return im


def f_checker8():
    im = Img(256, 256)
    for row in range(im.h):
        for col in range(im.w):
            im.put(row, col, 255 if ((row // 8) + (col // 8)) % 2 == 0 else 0)
    return im


def f_noise_u8():
    im = Img(256, 256)
    r = rng("noise_u8")
    im.d = bytearray(r.getrandbits(8) for _ in range(im.w * im.h))
    return im


def f_impulse():
    im = Img(256, 256, 0)
    im.put(128, 128, 255)
    return im


def f_step_edge():
    im = Img(256, 256, 0)
    for row in range(im.h):
        for col in range(im.w // 2, im.w):
            im.put(row, col, 255)
    return im


def f_selfsim_iso():
    """Exact-match oracle for the domain->range map and the isometry table.

    A 16x16 base patch is placed pixel-replicated 2x as a 32x32 block at (0,0):
    averaging that block 2:1 returns the patch *exactly*, with no rounding. The
    eight isometries of the same patch are then placed as 16x16 blocks on the
    quadtree grid (row 64, columns 0,16,...,112), so for every one of them a
    domain exists whose 2:1 downsample equals an isometry of the range.

    The assertion this supports is "RMS is quantisation error only" for those
    eight blocks. It does NOT license asserting `isom == k` for block k: the
    encoder flips the *range* and then composes through `mapping[][]`
    (`coding_func.c:100`), so the recorded index must be read off that table
    rather than assumed. Nor is the matching domain position necessarily
    unique -- overlapping search windows at SHIFT granularity may tie.
    """
    im = Img(256, 256, 128)
    patch = [[(3 * x * x + 7 * y) % 256 for x in range(16)] for y in range(16)]

    # Domain: patch pixel-replicated 2x -> 32x32. 2x2 mean == patch, exactly.
    domain = [[patch[i // 2][j // 2] for j in range(32)] for i in range(32)]
    im.blit(0, 0, domain)

    # Ranges: the eight isometries, 16x16, aligned to the quadtree grid.
    for iso in range(8):
        im.blit(64, iso * 16, isometry(patch, 16, iso))
    return im


def f_sierpinski():
    """(x & y) == 0 -- the Pascal-mod-2 triangle. Exactly 2:1 self-similar,
    which is precisely the scale the domain->range map uses."""
    im = Img(512, 512)
    for row in range(im.h):
        for col in range(im.w):
            im.put(row, col, 255 if (row & col) == 0 else 0)
    return im


MANDELBROT = {"re": (-2.2, 0.8), "im": (-1.5, 1.5), "max_iter": 64}


def f_mandelbrot():
    im = Img(512, 512)
    re0, re1 = MANDELBROT["re"]
    im0, im1 = MANDELBROT["im"]
    maxit = MANDELBROT["max_iter"]
    for row in range(im.h):
        ci = im0 + (im1 - im0) * row / (im.h - 1)
        for col in range(im.w):
            cr = re0 + (re1 - re0) * col / (im.w - 1)
            zr = zi = 0.0
            it = 0
            while it < maxit and zr * zr + zi * zi <= 4.0:
                zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
                it += 1
            im.put(row, col, 0 if it >= maxit else q8(255.0 * it / maxit))
    return im


def f_zoneplate():
    """Radial chirp reaching Nyquist (0.5 cyc/px) at r = W/2: with phase a*r^2
    the instantaneous radial frequency is a*r/pi, so a = pi/(2R)."""
    im = Img(256, 256)
    cx = cy = 127.5
    a = math.pi / (2.0 * (im.w / 2.0))
    for row in range(im.h):
        dy = row - cy
        for col in range(im.w):
            dx = col - cx
            im.put(row, col, q8(127.5 * (1.0 + math.cos(a * (dx * dx + dy * dy)))))
    return im


def f_mixed(width, height, name):
    """Quadrants: flat | ramp / checker | noise. Non-dyadic dimensions."""
    im = Img(width, height)
    r = rng(name)
    midr, midc = height // 2, width // 2
    for row in range(height):
        for col in range(width):
            if row < midr and col < midc:
                v = 128
            elif row < midr:
                span = max(1, width - midc - 1)
                v = q8(255.0 * (col - midc) / span)
            elif col < midc:
                v = 255 if ((row // 8) + (col // 8)) % 2 == 0 else 0
            else:
                v = r.getrandbits(8)
            im.put(row, col, v)
    return im


FIXTURES = [
    ("flat128", f_flat128,
     "256x256 constant 128.",
     "Domain variance zero -> det == 0 exactly, and the zero-alfa branch. May "
     "round-trip exactly at a loose RMS threshold, exercising the null-PSNR path."),
    ("ramp_h", f_ramp_h,
     "256x256 exact horizontal linear ramp, 0..255.",
     "Perfectly affine, so alfa/beta must land on quantiser-exact values. "
     "Residual RMS above quantisation error is a formula bug, not coding loss."),
    ("checker8", f_checker8,
     "256x256 checkerboard, 8-pixel squares, 0/255.",
     "Worst-case partition: forces the quadtree down to min_size everywhere."),
    ("noise_u8", f_noise_u8,
     "256x256 uniform white noise, seeded.",
     "Anti-self-similar. Every domain match fails, alfa -> 0; upper bound on "
     "both bitrate and search time."),
    ("impulse", f_impulse,
     "256x256 zeros with a single 255 pixel at (128,128).",
     "Drives det = s0*s2 - s1^2 toward zero -- the division-by-near-zero path "
     "in alfa = (s0*t1 - s1*t0)/det, where a NaN can enter and propagate silently."),
    ("step_edge", f_step_edge,
     "256x256, left half 0, right half 255.",
     "Full-range beta' clamping at both extremes plus the alfa' > 0 +/-255 "
     "offset branch, alongside det == 0 flat blocks in the same image."),
    ("selfsim_iso", f_selfsim_iso,
     "256x256 oracle: a 16x16 patch pixel-replicated 2x as a 32x32 domain, "
     "plus its eight isometries as 16x16 range blocks on the quadtree grid.",
     "Exact-equality test for the 2:1 domain map and the isometry table: an "
     "RMS-equals-quantisation-error match must exist for all eight blocks."),
    ("sierpinski", f_sierpinski,
     "512x512 (row & col) == 0 triangle.",
     "Exactly 2:1 self-similar -- the positive case for the method's premise."),
    ("mandelbrot", f_mandelbrot,
     "512x512 escape-time render, max_iter 64.",
     "Scale-invariance at arbitrary non-dyadic scales; structure at every "
     "block size without the synthetic regularity of the dyadic fixtures."),
    ("zoneplate", f_zoneplate,
     "256x256 radial chirp reaching Nyquist at r = W/2.",
     "Aliasing probe for the 2:1 domain decimation; decimation bugs show up "
     "as moire rings here and essentially nowhere else."),
    ("mixed_250x250", lambda: f_mixed(250, 250, "mixed_250x250"),
     "250x250 quadrants: flat, ramp, checker, noise.",
     "Dimensions not a multiple of SHIFT=4: exercises the integer-division "
     "truncation in bits_per_coordinate, plus padding to virtual_size 256."),
    ("mixed_129x127", lambda: f_mixed(129, 127, "mixed_129x127"),
     "129x127 quadrants: flat, ramp, checker, noise.",
     "Non-square, odd, non-multiple of SHIFT, maximal padding to virtual_size "
     "256, and below the 176-pixel MS-SSIM floor: must be flagged and excluded."),
]


def toml_escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main():
    here = pathlib.Path(__file__).resolve()
    repo = here.parent.parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path, default=repo / "fixtures" / "images")
    ap.add_argument("--pgm", action="store_true", help="also write .pgm copies")
    ap.add_argument("--check", action="store_true",
                    help="verify against manifest.toml instead of writing")
    args = ap.parse_args()

    script_sha = hashlib.sha256(here.read_bytes()).hexdigest()
    built = []
    for name, fn, desc, purpose in FIXTURES:
        im = fn()
        assert len(im.d) == im.w * im.h, name
        built.append((name, im, hashlib.sha256(im.d).hexdigest(), desc, purpose))

    manifest = args.out / "manifest.toml"

    if args.check:
        if not manifest.exists():
            print(f"FAIL no manifest at {manifest}", file=sys.stderr)
            return 2
        text = manifest.read_text()
        bad = 0
        for name, im, sha, _desc, _purpose in built:
            if f'sha256 = "{sha}"' not in text:
                print(f"FAIL {name}: regenerated sha256 {sha} not in manifest",
                      file=sys.stderr)
                bad += 1
            raw = args.out / f"{name}.raw"
            if not raw.exists():
                print(f"FAIL {name}: {raw} missing", file=sys.stderr)
                bad += 1
            elif hashlib.sha256(raw.read_bytes()).hexdigest() != sha:
                print(f"FAIL {name}: {raw} on disk differs from regeneration",
                      file=sys.stderr)
                bad += 1
        print(f"{'FAIL' if bad else 'OK'} {len(built)} fixtures, {bad} problem(s)")
        return 1 if bad else 0

    args.out.mkdir(parents=True, exist_ok=True)
    # Self-ignoring: generated images stay out of git without touching the
    # repo-root .gitignore. manifest.toml is committed.
    (args.out / ".gitignore").write_text("*.raw\n*.pgm\n")

    lines = [
        "# Synthetic fixture images for the `fixtures/` corpus role (M9).",
        "# GENERATED by scripts/gen-fixtures.py -- do not edit by hand.",
        "# The .raw files are not committed; regenerate them with that script and",
        "# `--check` proves the regeneration is byte-identical to these hashes.",
        "",
        "[generator]",
        'script = "scripts/gen-fixtures.py"',
        f'script_sha256 = "{script_sha}"',
        f"seed = {SEED}",
        'rng = "CPython random.Random, MT19937, seeded from a string via SHA-512"',
        'format = "8-bit grayscale, headerless, row-major, width-fast"',
        'note = "pass to the reference codec as -W <width> -H <height>"',
        "",
    ]
    for name, im, sha, desc, purpose in built:
        raw = args.out / f"{name}.raw"
        raw.write_bytes(im.d)
        if args.pgm:
            (args.out / f"{name}.pgm").write_bytes(
                f"P5\n{im.w} {im.h}\n255\n".encode("ascii") + im.d)
        lines += [
            "[[image]]",
            f'name = "{name}"',
            f'file = "{name}.raw"',
            f"width = {im.w}",
            f"height = {im.h}",
            f"bytes = {len(im.d)}",
            f'sha256 = "{sha}"',
            f'description = "{toml_escape(desc)}"',
            f'purpose = "{toml_escape(purpose)}"',
            "",
        ]
    manifest.write_text("\n".join(lines))

    corpus_hash = hashlib.sha256(
        "\n".join(f"{n}\t{sha}" for n, _, sha, _, _ in sorted(built)).encode()
    ).hexdigest()
    print(f"wrote {len(built)} fixtures to {args.out}")
    print(f"manifest: {manifest}")
    print(f"fixtures corpus manifest hash: {corpus_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
