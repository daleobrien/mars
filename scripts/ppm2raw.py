#!/usr/bin/env python3
"""PPM -> 8-bit grayscale raw, with the grayscale conversion PINNED here.

Why this exists: ImageMagick and ffmpeg both have version-dependent defaults for
"convert to gray" (Rec.601 vs Rec.709, luma vs linear-light luminance). A corpus
whose grayscale depends on the installed tool version is not a pinned corpus. So
the external tool only decodes the container to PPM -- which involves no colour
transform -- and the luma arithmetic happens here, in code we version.

Pinned definition:
    Y = 0.299*R + 0.587*G + 0.114*B     (Rec.601 luma, on sRGB-encoded values)
    rounded half away from zero, clamped to [0, 255]

Output layout is row-major, width-fast, headerless -- what `readimage_raw()`
expects (reference/mars1/image_io.c:158). Prints "<width> <height> <sha256>" on
stdout so callers can record provenance without re-reading the file.

Usage:
  ppm2raw.py IN.ppm -o OUT.raw [--crop center:256x256 | --crop 256x256+32+16]
"""

import argparse
import hashlib
import math
import pathlib
import sys


def read_ppm(path):
    """Parse a binary PPM (P6) or PGM (P5). Returns (width, height, channels, data)."""
    blob = path.read_bytes()
    fields, pos = [], 0
    while len(fields) < 4:
        while pos < len(blob) and blob[pos:pos + 1].isspace():
            pos += 1
        if blob[pos:pos + 1] == b"#":                     # comment to end of line
            while pos < len(blob) and blob[pos:pos + 1] not in (b"\n", b"\r"):
                pos += 1
            continue
        start = pos
        while pos < len(blob) and not blob[pos:pos + 1].isspace():
            pos += 1
        fields.append(blob[start:pos])
    pos += 1                                              # exactly one whitespace byte
    magic, width, height, maxval = fields
    if magic not in (b"P6", b"P5"):
        raise SystemExit(f"{path}: not a binary PPM/PGM (magic {magic!r})")
    if int(maxval) != 255:
        raise SystemExit(f"{path}: maxval {int(maxval)}, expected 255 "
                         "(decode with -depth 8)")
    channels = 3 if magic == b"P6" else 1
    w, h = int(width), int(height)
    data = blob[pos:pos + w * h * channels]
    if len(data) != w * h * channels:
        raise SystemExit(f"{path}: truncated pixel data")
    return w, h, channels, data


def to_gray(w, h, channels, data):
    if channels == 1:
        return bytearray(data)
    out = bytearray(w * h)
    for i in range(w * h):
        r, g, b = data[3 * i], data[3 * i + 1], data[3 * i + 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        v = math.floor(y + 0.5) if y >= 0 else math.ceil(y - 0.5)
        out[i] = 0 if v < 0 else (255 if v > 255 else v)
    return out


def parse_crop(spec, w, h):
    """'center:WxH' or 'WxH+X+Y' -> (cw, ch, x, y)."""
    if spec.startswith("center:"):
        geom = spec.split(":", 1)[1]
        cw, ch = (int(v) for v in geom.lower().split("x"))
        return cw, ch, (w - cw) // 2, (h - ch) // 2
    size, _, offs = spec.partition("+")
    cw, ch = (int(v) for v in size.lower().split("x"))
    x, _, y = offs.partition("+")
    return cw, ch, int(x or 0), int(y or 0)


def crop(gray, w, h, cw, ch, x, y):
    if x < 0 or y < 0 or x + cw > w or y + ch > h:
        raise SystemExit(f"crop {cw}x{ch}+{x}+{y} does not fit in {w}x{h}")
    out = bytearray(cw * ch)
    for row in range(ch):
        src = (y + row) * w + x
        out[row * cw:(row + 1) * cw] = gray[src:src + cw]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=pathlib.Path)
    ap.add_argument("-o", "--output", type=pathlib.Path, required=True)
    ap.add_argument("--crop", default=None,
                    help="'center:WxH' or 'WxH+X+Y', applied after conversion")
    args = ap.parse_args()

    w, h, channels, data = read_ppm(args.input)
    gray = to_gray(w, h, channels, data)
    if args.crop:
        cw, ch, x, y = parse_crop(args.crop, w, h)
        gray = crop(gray, w, h, cw, ch, x, y)
        w, h = cw, ch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gray)
    print(f"{w} {h} {hashlib.sha256(gray).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
