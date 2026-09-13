#!/usr/bin/env python3
"""Generate `fixtures/mars1/tiny64.raw`, the image behind the byte-by-byte walkthrough
in `docs/mars1-format.md` §12.

It is designed backwards from the bitstream we want to be able to print in full. At
`-m 16 -M 32` on a 64x64 image the walk is: virtual_size 64 is above `max_size`, so it
subdivides with no bits at all; each 32x32 quadrant then gets one split flag. Three flat
quadrants stay whole and code as DC-only leaves (no isometry, no domain coordinates); the
ramp quadrant splits into four 16x16 leaves that do carry domains. That is every branch of
the tree walk and both leaf kinds, in 7 transforms and 24 bytes.

Deliberately *not* part of `scripts/gen-fixtures.py`: adding an image there would change
`corpus/fixtures.images.json`, and the corpus manifest hash is recorded in every Step 2
result row. A walkthrough prop is not a corpus image.

Usage:  python3 scripts/gen-tiny64.py [--check]
"""

import argparse
import hashlib
import pathlib
import sys

W = H = 64
OUT = pathlib.Path("fixtures/mars1/tiny64.raw")


def build() -> bytes:
    px = bytearray(W * H)
    for r in range(H):
        for c in range(W):
            if r < 32 and c < 32:
                v = 100  # flat        -> DC-only leaf at size 32
            elif r < 32:
                v = 170  # flat        -> DC-only leaf at size 32
            elif c < 32:
                v = 40 + (r - 32) * 4  # vertical ramp -> splits to four 16x16 leaves
            else:
                v = 60  # flat        -> DC-only leaf at size 32
            px[r * W + c] = v
    return bytes(px)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the committed file instead of writing")
    args = ap.parse_args()

    data = build()
    digest = hashlib.sha256(data).hexdigest()

    if args.check:
        if not OUT.exists():
            print(f"{OUT} is missing", file=sys.stderr)
            return 1
        have = OUT.read_bytes()
        if have != data:
            print(
                f"{OUT} does not match this script:\n"
                f"  committed sha256 {hashlib.sha256(have).hexdigest()}\n"
                f"  regenerated      {digest}",
                file=sys.stderr,
            )
            return 1
        print(f"tiny64.raw matches ({len(data)} bytes, sha256 {digest})")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"wrote {OUT} ({len(data)} bytes, sha256 {digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
