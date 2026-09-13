#!/usr/bin/env python3
"""Independent validator for `docs/mars1-format.md` (Step 3's exit criterion).

This script is deliberately isolated. It implements the Mars 1 `.ifs` format **from the
specification document alone** -- it does not import anything from this project, does not
read `reference/mars1/`, and does not shell out to `encmars` or `decmars`. Its only inputs
are the committed bitstreams in `fixtures/mars1/` and the integers and hashes that
`fixtures/mars1/manifest.toml` records from the 1998 binaries.

If the spec is wrong or incomplete, this script cannot be written correctly from it, and
that is the whole point: "if the spec cannot be validated independently of the C, it is
not a spec."

Five checks per fixture, all exact equalities -- no tolerances anywhere:

  1. transform count      == what `encmars` printed
  2. DC-leaf count        == `Zero_alfa_transformations`
  3. re-serialised bytes  == the committed `.ifs`, byte for byte
  4. rendered partition   == the `-Q` `quadtree.pgm`, by SHA-256
  5. iterative decode     == `decmars -i` output, by SHA-256

Checks 1-3 are all blind to the child-recursion order of §5.1: a parser that visits
NW,NE,SW,SE consumes the identical bits and emits the identical count. Checks 4 and 5 are
what pin the geometry.

Usage:
    python3 scripts/validate-ifs.py                     # validate every fixture
    python3 scripts/validate-ifs.py --only tiny64       # substring filter
    python3 scripts/validate-ifs.py --walkthrough       # regenerate the §12 walkthrough
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import tomllib
from dataclasses import dataclass

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment problem, not a spec problem
    sys.exit(
        "numpy is required (the decode check is a 10-iteration fixed point over every\n"
        "fixture; pure Python takes minutes). Use the pinned venv:\n"
        "    just crossval-setup && .venv-crossval/bin/python scripts/validate-ifs.py"
    )

FIXTURE_DIR = pathlib.Path("fixtures/mars1")
WALKTHROUGH_STEM = "tiny64__walkthrough__fisher__r8"
WALKTHROUGH_OUT = FIXTURE_DIR / "tiny64.walkthrough.txt"
SPEC_DOC = pathlib.Path("docs/mars1-format.md")
MARK_BEGIN = "<!-- WALKTHROUGH:BEGIN -->"
MARK_END = "<!-- WALKTHROUGH:END -->"
DECODE_ITERATIONS = 10  # globals.h: iterations INIT(= 10)


# --------------------------------------------------------------------------- §2 packing


class BitReader:
    """§2. Bits leave each byte most-significant first; each value is assembled
    least-significant bit first."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def bit(self) -> int:
        byte = self.data[self.pos >> 3]
        b = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return b

    def read(self, n: int) -> int:
        v = 0
        for i in range(n):
            if self.bit():
                v |= 1 << i
        return v


class BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def write(self, n: int, value: int) -> None:
        for i in range(n):
            self.bits.append((value >> i) & 1)

    def finish(self) -> bytes:
        """§2: the last partial byte is left-aligned and its low bits are zero."""
        out = bytearray()
        for i in range(0, len(self.bits), 8):
            chunk = self.bits[i : i + 8]
            byte = 0
            for b in chunk:
                byte = (byte << 1) | b
            byte <<= 8 - len(chunk)
            out.append(byte)
        return bytes(out)


# ------------------------------------------------------------------------ §4 derived


def ceil_log2(q: int) -> int:
    """§4.3, as exact integer arithmetic."""
    return 0 if q <= 1 else (q - 1).bit_length()


@dataclass
class Header:
    bits_alfa: int
    bits_beta: int
    min_size: int
    max_size: int
    shift: int
    width: int
    height: int
    int_max_alfa: int

    @property
    def max_alfa(self) -> float:
        return self.int_max_alfa / 256 * 8.0

    @property
    def virtual_size(self) -> int:
        return 1 << ceil_log2(max(self.width, self.height))

    @property
    def bits_coord_row(self) -> int:  # bits_per_coordinate_h
        return ceil_log2(self.height // self.shift)

    @property
    def bits_coord_col(self) -> int:  # bits_per_coordinate_w
        return ceil_log2(self.width // self.shift)


HEADER_FIELDS = [
    ("N_BITALFA", 4),
    ("N_BITBETA", 4),
    ("min_size", 7),
    ("max_size", 7),
    ("SHIFT", 6),
    ("image_width", 12),
    ("image_height", 12),
    ("int_max_alfa", 8),
]


@dataclass
class Leaf:
    row: int
    col: int
    size: int
    qalfa: int
    qbeta: int
    isometry: int
    dom_row: int
    dom_col: int

    @property
    def dc_only(self) -> bool:
        return self.qalfa == 0


# ------------------------------------------------------------------------- §5-6 parse


class SpecViolation(Exception):
    pass


def parse(data: bytes) -> tuple[Header, list[Leaf]]:
    r = BitReader(data)
    hdr = Header(*(r.read(n) for _, n in HEADER_FIELDS))
    if hdr.width == 0 or hdr.height == 0 or hdr.shift == 0:
        raise SpecViolation(f"degenerate header: {hdr}")
    leaves: list[Leaf] = []

    def walk(row: int, col: int, size: int) -> None:
        if row >= hdr.height or col >= hdr.width:  # §5.1
            return
        if (  # §5.2 forced subdivision, no bit consumed
            size > hdr.max_size
            or row + size > hdr.height
            or col + size > hdr.width
        ):
            half = size // 2
            walk(row, col, half)
            walk(row + half, col, half)  # §5.1: NW, SW, NE, SE
            walk(row, col + half, half)
            walk(row + half, col + half, half)
            return
        if size > hdr.min_size and r.read(1):
            half = size // 2
            walk(row, col, half)
            walk(row + half, col, half)
            walk(row, col + half, half)
            walk(row + half, col + half, half)
            return
        qalfa = r.read(hdr.bits_alfa)
        qbeta = r.read(hdr.bits_beta)
        if qalfa != 0:  # §6, `zeroalfa` is always 0
            isom = r.read(3)
            dom_row = hdr.shift * r.read(hdr.bits_coord_row)
            dom_col = hdr.shift * r.read(hdr.bits_coord_col)
            if dom_row + 2 * size > hdr.height or dom_col + 2 * size > hdr.width:
                raise SpecViolation(
                    f"leaf at ({row},{col},{size}) references a domain at "
                    f"({dom_row},{dom_col}) that runs past the image"
                )
        else:
            isom = dom_row = dom_col = 0
        leaves.append(Leaf(row, col, size, qalfa, qbeta, isom, dom_row, dom_col))

    walk(0, 0, hdr.virtual_size)

    # The stream must end here, with only §2 padding left over.
    left = len(data) * 8 - r.pos
    if not 0 <= left < 8:
        raise SpecViolation(f"{left} bits left after the tree; expected 0-7 of padding")
    if int.from_bytes(data, "big") & ((1 << left) - 1):
        raise SpecViolation("padding bits are not zero")
    return hdr, leaves


def serialise(hdr: Header, leaves: list[Leaf]) -> bytes:
    """Re-emit the tree. Driven by the *parsed leaf set* rather than by a recorded list of
    decisions, so it independently checks that the leaves tile the image."""
    by_pos = {(lf.row, lf.col, lf.size): lf for lf in leaves}
    if len(by_pos) != len(leaves):
        raise SpecViolation("two leaves share a position")
    w = BitWriter()
    for name, n in HEADER_FIELDS:
        w.write(
            n,
            {
                "N_BITALFA": hdr.bits_alfa,
                "N_BITBETA": hdr.bits_beta,
                "min_size": hdr.min_size,
                "max_size": hdr.max_size,
                "SHIFT": hdr.shift,
                "image_width": hdr.width,
                "image_height": hdr.height,
                "int_max_alfa": hdr.int_max_alfa,
            }[name],
        )

    def walk(row: int, col: int, size: int) -> None:
        if row >= hdr.height or col >= hdr.width:
            return
        half = size // 2
        if size > hdr.max_size or row + size > hdr.height or col + size > hdr.width:
            walk(row, col, half)
            walk(row + half, col, half)
            walk(row, col + half, half)
            walk(row + half, col + half, half)
            return
        lf = by_pos.get((row, col, size))
        if lf is None:
            if size <= hdr.min_size:
                raise SpecViolation(f"no leaf at ({row},{col},{size}) and none can follow")
            w.write(1, 1)
            walk(row, col, half)
            walk(row + half, col, half)
            walk(row, col + half, half)
            walk(row + half, col + half, half)
            return
        if size > hdr.min_size:
            w.write(1, 0)
        w.write(hdr.bits_alfa, lf.qalfa)
        w.write(hdr.bits_beta, lf.qbeta)
        if lf.qalfa != 0:
            w.write(3, lf.isometry)
            w.write(hdr.bits_coord_row, lf.dom_row // hdr.shift)
            w.write(hdr.bits_coord_col, lf.dom_col // hdr.shift)

    walk(0, 0, hdr.virtual_size)
    return w.finish()


def render_partition(hdr: Header, leaves: list[Leaf]) -> bytes:
    """Reproduce `encmars -Q`'s `quadtree.pgm`: 255 everywhere, 0 on the virtual image's
    border, and 0 along the cross drawn at every subdivision, forced or flagged."""
    vs = hdr.virtual_size
    qtt = np.full((vs, vs), 255, dtype=np.uint8)
    qtt[:, 0] = 0
    qtt[:, vs - 1] = 0
    qtt[0, :] = 0
    qtt[vs - 1, :] = 0
    by_pos = {(lf.row, lf.col, lf.size) for lf in leaves}

    def cross(row: int, col: int, size: int) -> None:
        half = size // 2
        qtt[row + half, col : col + size] = 0
        qtt[row : row + size, col + half] = 0

    def walk(row: int, col: int, size: int) -> None:
        if row >= hdr.height or col >= hdr.width:
            return
        half = size // 2
        forced = (
            size > hdr.max_size
            or row + size > hdr.height
            or col + size > hdr.width
        )
        if forced or (row, col, size) not in by_pos:
            cross(row, col, size)
            walk(row, col, half)
            walk(row + half, col, half)
            walk(row, col + half, half)
            walk(row + half, col + half, half)

    walk(0, 0, vs)
    body = qtt[: hdr.height, : hdr.width].tobytes()
    return b"P5\n%d %d\n255\n" % (hdr.width, hdr.height) + body


# ---------------------------------------------------------------------- §7-10 decoding


def _isometry(block: np.ndarray, k: int) -> np.ndarray:
    """§9, applied to the trailing two axes of a batch."""
    if k == 0:
        return block
    if k == 1:  # L_ROTATE90 -- counterclockwise
        return np.rot90(block, 1, axes=(1, 2))
    if k == 2:  # R_ROTATE90 -- clockwise
        return np.rot90(block, -1, axes=(1, 2))
    if k == 3:
        return np.rot90(block, 2, axes=(1, 2))
    if k == 4:  # R_VERTICAL -- columns reversed
        return block[:, :, ::-1]
    if k == 5:  # R_HORIZONTAL -- rows reversed
        return block[:, ::-1, :]
    if k == 6:  # F_DIAGONAL -- transpose
        return np.swapaxes(block, 1, 2)
    if k == 7:  # S_DIAGONAL -- anti-transpose
        return np.swapaxes(block[:, ::-1, ::-1], 1, 2)
    raise SpecViolation(f"isometry {k} is outside 0-7")


def decode_iterative(hdr: Header, leaves: list[Leaf], iterations: int) -> bytes:
    """§10.1. Batched by (size, isometry); the batching changes nothing arithmetically
    because no two leaves overlap and every operation is elementwise."""
    h, w = hdr.height, hdr.width
    img = np.full((h, w), 128, dtype=np.uint8)

    groups: dict[tuple[int, int], list[Leaf]] = {}
    for lf in leaves:
        groups.setdefault((lf.size, lf.isometry), []).append(lf)

    packed = []
    for (size, isom), members in sorted(groups.items()):
        off = np.arange(size)
        dom_r = np.array([m.dom_row for m in members])[:, None] + 2 * off
        dom_c = np.array([m.dom_col for m in members])[:, None] + 2 * off
        dst_r = np.array([m.row for m in members])[:, None] + off
        dst_c = np.array([m.col for m in members])[:, None] + off
        alfa = np.array(
            [m.qalfa / (1 << hdr.bits_alfa) * hdr.max_alfa for m in members]
        )
        beta = np.empty(len(members))
        for i, m in enumerate(members):
            b = m.qbeta / ((1 << hdr.bits_beta) - 1) * ((1.0 + abs(alfa[i])) * 255)
            if alfa[i] > 0.0:
                b -= alfa[i] * 255
            beta[i] = b
        packed.append(
            (
                isom,
                dom_r[:, :, None],
                dom_c[:, None, :],
                dst_r[:, :, None],
                dst_c[:, None, :],
                alfa[:, None, None],
                beta[:, None, None],
            )
        )

    for _ in range(iterations):
        nxt = np.empty_like(img)
        for isom, dr, dc, rr, cc, alfa, beta in packed:
            # The 2x2 box average of §9. Every term is an integer below 1021, so the sum
            # and the division by 4 are exact in binary64 whatever the order.
            d = (
                img[dr, dc].astype(np.float64)
                + img[dr + 1, dc]
                + img[dr, dc + 1]
                + img[dr + 1, dc + 1]
            ) / 4.0
            # The C is `0.5 + pixel * alfa + beta`, and `+` is left-associative, so the
            # 0.5 is added to the product *before* beta. Reassociating this changes the
            # last bit, and bound()'s truncation can turn that into a whole grey level.
            v = (0.5 + _isometry(d, isom) * alfa) + beta
            np.clip(v, 0.0, 255.0, out=v)
            nxt[rr, cc] = v.astype(np.uint8)
        img = nxt

    return b"P5\n%d %d\n255\n" % (w, h) + img.tobytes()


# ------------------------------------------------------------------------- validation


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def validate(entry: dict, data: bytes) -> list[str]:
    """Return a list of failures; empty means the fixture passed."""
    fails: list[str] = []
    hdr, leaves = parse(data)

    if hdr.width != entry["width"] or hdr.height != entry["height"]:
        fails.append(
            f"header says {hdr.width}x{hdr.height}, manifest says "
            f"{entry['width']}x{entry['height']}"
        )
    for name, got, want in [
        ("min_size", hdr.min_size, entry["min_size"]),
        ("max_size", hdr.max_size, entry["max_size"]),
        ("SHIFT", hdr.shift, entry["shift"]),
        ("N_BITALFA", hdr.bits_alfa, entry["bits_alfa"]),
        ("N_BITBETA", hdr.bits_beta, entry["bits_beta"]),
    ]:
        if got != want:
            fails.append(f"header {name} = {got}, manifest asked for {want}")

    if len(leaves) != entry["transforms"]:
        fails.append(f"parsed {len(leaves)} leaves, encmars reported {entry['transforms']}")
    dc = sum(1 for lf in leaves if lf.dc_only)
    if dc != entry["zero_alfa_transforms"]:
        fails.append(
            f"parsed {dc} DC-only leaves, encmars reported {entry['zero_alfa_transforms']}"
        )

    again = serialise(hdr, leaves)
    if again != data:
        where = next(
            (i for i, (a, b) in enumerate(zip(again, data)) if a != b),
            min(len(again), len(data)),
        )
        fails.append(
            f"re-serialised stream differs at byte {where} "
            f"({len(again)} bytes vs {len(data)})"
        )

    got = sha256(render_partition(hdr, leaves))
    if got != entry["quadtree_pgm_sha256"]:
        fails.append("rendered partition does not match encmars -Q (child order? §5.1)")

    got = sha256(decode_iterative(hdr, leaves, DECODE_ITERATIONS))
    if got != entry["decode_iterative_sha256"]:
        fails.append("decoded image does not match decmars -i")

    return fails


# ------------------------------------------------------------------------ walkthrough


def walkthrough(data: bytes, hdr: Header, leaves: list[Leaf]) -> str:
    """The byte-by-byte dissection of §12: every field with the bit range it occupies,
    then a hexdump annotated with which field each byte belongs to."""
    spans: list[tuple[int, int, str, str]] = []  # start, length, field, meaning
    pos = 0
    values = [
        hdr.bits_alfa,
        hdr.bits_beta,
        hdr.min_size,
        hdr.max_size,
        hdr.shift,
        hdr.width,
        hdr.height,
        hdr.int_max_alfa,
    ]
    for (name, n), v in zip(HEADER_FIELDS, values):
        note = f"MAX_ALFA = {hdr.max_alfa:g}" if name == "int_max_alfa" else ""
        spans.append((pos, n, f"header {name}", f"{v}  {note}".rstrip()))
        pos += n

    by_pos = {(lf.row, lf.col, lf.size): lf for lf in leaves}

    def walk(row: int, col: int, size: int) -> None:
        nonlocal pos
        if row >= hdr.height or col >= hdr.width:
            return
        half = size // 2
        if size > hdr.max_size or row + size > hdr.height or col + size > hdr.width:
            spans.append((pos, 0, f"({row},{col}) {size}x{size}", "forced subdivision (§5.2), no bits"))
            for r, c in ((row, col), (row + half, col), (row, col + half), (row + half, col + half)):
                walk(r, c, half)
            return
        lf = by_pos.get((row, col, size))
        if lf is None:
            spans.append((pos, 1, f"({row},{col}) {size}x{size} split", "1 = subdivide"))
            pos += 1
            for r, c in ((row, col), (row + half, col), (row, col + half), (row + half, col + half)):
                walk(r, c, half)
            return
        tag = f"({row},{col}) {size}x{size}"
        if size > hdr.min_size:
            spans.append((pos, 1, f"{tag} split", "0 = leaf"))
            pos += 1
        alfa = lf.qalfa / (1 << hdr.bits_alfa) * hdr.max_alfa
        beta = lf.qbeta / ((1 << hdr.bits_beta) - 1) * ((1.0 + abs(alfa)) * 255)
        if alfa > 0.0:
            beta -= alfa * 255
        spans.append((pos, hdr.bits_alfa, f"{tag} qalfa", f"{lf.qalfa}  -> alfa {alfa:.6f}"))
        pos += hdr.bits_alfa
        spans.append((pos, hdr.bits_beta, f"{tag} qbeta", f"{lf.qbeta}  -> beta {beta:.6f}"))
        pos += hdr.bits_beta
        if lf.qalfa != 0:
            spans.append((pos, 3, f"{tag} isometry", f"{lf.isometry}"))
            pos += 3
            spans.append(
                (pos, hdr.bits_coord_row, f"{tag} dom_row", f"{lf.dom_row // hdr.shift} -> row {lf.dom_row}")
            )
            pos += hdr.bits_coord_row
            spans.append(
                (pos, hdr.bits_coord_col, f"{tag} dom_col", f"{lf.dom_col // hdr.shift} -> col {lf.dom_col}")
            )
            pos += hdr.bits_coord_col
        else:
            spans.append((pos, 0, f"{tag} DC-only", "qalfa == 0: no isometry, no domain (§6)"))

    walk(0, 0, hdr.virtual_size)
    pad = len(data) * 8 - pos
    if pad:
        spans.append((pos, pad, "padding", "zero, low bits of the final byte (§2)"))

    out: list[str] = []
    out.append(f"{len(data)} bytes, {pos} bits of payload + {pad} of padding, "
               f"{len(leaves)} leaves ({sum(1 for l in leaves if l.dc_only)} DC-only)")
    out.append("")
    out.append(f"{'bits':>12}  {'bytes':>9}  {'field':<28} value")
    out.append(f"{'-'*12}  {'-'*9}  {'-'*28} {'-'*34}")
    for start, n, field, meaning in spans:
        if n == 0:
            out.append(f"{'':>12}  {'':>9}  {field:<28} {meaning}")
            continue
        end = start + n
        byte_range = (
            f"{start//8}" if (end - 1) // 8 == start // 8 else f"{start//8}-{(end-1)//8}"
        )
        out.append(f"{start:>5}..{end:<5}  {byte_range:>9}  {field:<28} {meaning}")

    out.append("")
    out.append("hexdump")
    for i in range(0, len(data), 8):
        chunk = data[i : i + 8]
        hexs = " ".join(f"{b:02x}" for b in chunk)
        bits = " ".join(f"{b:08b}" for b in chunk)
        out.append(f"  {i:04x}  {hexs:<23}  {bits}")
    return "\n".join(out)


# ------------------------------------------------------------------------------- main


# ------------------------------------------------------------------------- §11 contract


def check_contract(entries: list[dict], directory: pathlib.Path) -> list[str]:
    """The closed-form claims the spec makes, asserted against the committed fixtures.

    These are not measurements read back out of the data. §11 derives every one of them
    from the format rules by hand, before any file existed; if the fixtures ever stop
    satisfying them, either the reference changed or the spec is wrong, and both are
    things a human has to look at.
    """
    fails: list[str] = []
    by_stem = {e["stem"]: e for e in entries}

    def leaves_of(stem: str) -> tuple[dict, Header, list[Leaf]]:
        e = by_stem[stem]
        hdr, lv = parse((directory / e["file"]).read_bytes())
        return e, hdr, lv

    def histogram(lv: list[Leaf]) -> dict[int, int]:
        h: dict[int, int] = {}
        for x in lv:
            h[x.size] = h.get(x.size, 0) + 1
        return h

    # §11 -- flat128 at the 1998 defaults, in closed form.
    stem = "flat128__default__fisher__r8"
    if stem in by_stem:
        e, hdr, lv = leaves_of(stem)
        if e["transforms"] != 256:
            fails.append(f"§11: {stem} has {e['transforms']} transforms, expected 256")
        if e["zero_alfa_transforms"] != 256:
            fails.append(f"§11: {stem} has {e['zero_alfa_transforms']} DC leaves, expected 256")
        if e["ifs_bytes"] != 392:
            fails.append(f"§11: {stem} is {e['ifs_bytes']} bytes, expected 392")
        if histogram(lv) != {16: 256}:
            fails.append(f"§11: {stem} partition is {histogram(lv)}, expected 256 leaves of 16")
        if {x.qbeta for x in lv} != {64}:
            fails.append(f"§11: {stem} qbeta values are {sorted({x.qbeta for x in lv})}, expected 64")
        pgm = decode_iterative(hdr, lv, DECODE_ITERATIONS)
        body = set(pgm.split(b"255\n", 1)[1])
        if body != {129}:
            fails.append(f"§11: {stem} decodes to {sorted(body)}, expected a constant 129")

    # §5.3 -- the forced-subdivision geometry, at -m 4 -M 16.
    for stem, want in (
        ("mixed_129x127__default__{m}__r{r}", {1: 255, 2: 64}),
        ("mixed_250x250__default__{m}__r{r}", {2: 249}),
    ):
        for method in ("fisher", "masscenter", "saupe-fisher"):
            for rate in (2, 8, 32):
                key = stem.format(m=method, r=rate)
                if key not in by_stem:
                    continue
                _, _, lv = leaves_of(key)
                h = histogram(lv)
                got = {size: h.get(size, 0) for size in want}
                if got != want:
                    fails.append(f"§5.3: {key} sub-min_size leaves {got}, expected {want}")

    # §5.3 -- and none at all when the dimensions divide.
    for e in entries:
        if e["width"] % e["min_size"] or e["height"] % e["min_size"]:
            continue
        _, _, lv = leaves_of(e["stem"])
        small = sorted({x.size for x in lv if x.size < e["min_size"]})
        if small:
            fails.append(
                f"§5.3: {e['stem']} has leaves of size {small} below min_size "
                f"{e['min_size']} on dimensions that divide it"
            )

    return fails


def embedded_block(text: str) -> str:
    """The §12 walkthrough as it should appear in the spec, fenced."""
    return f"{MARK_BEGIN}\n```\n{text.rstrip()}\n```\n{MARK_END}"


def splice_doc(text: str) -> bool:
    """Rewrite the spec's §12 block. Returns True if the file changed."""
    doc = SPEC_DOC.read_text()
    start = doc.index(MARK_BEGIN)
    end = doc.index(MARK_END) + len(MARK_END)
    new = doc[:start] + embedded_block(text) + doc[end:]
    if new == doc:
        return False
    SPEC_DOC.write_text(new)
    return True


def doc_is_in_sync(text: str) -> bool:
    doc = SPEC_DOC.read_text()
    return embedded_block(text) in doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=pathlib.Path, default=FIXTURE_DIR)
    ap.add_argument("--only", default=None, help="substring filter on the fixture stem")
    ap.add_argument("--walkthrough", action="store_true", help="regenerate the §12 dissection")
    ap.add_argument("--stats", action="store_true", help="print leaf-size histograms")
    args = ap.parse_args()

    manifest_path = args.dir / "manifest.toml"
    if not manifest_path.exists():
        print(f"{manifest_path} is missing; run `just mars1-fixtures`", file=sys.stderr)
        return 1
    manifest = tomllib.loads(manifest_path.read_text())
    entries = manifest["fixture"]

    if args.walkthrough:
        entry = next(e for e in entries if e["stem"] == WALKTHROUGH_STEM)
        data = (args.dir / entry["file"]).read_bytes()
        hdr, leaves = parse(data)
        text = walkthrough(data, hdr, leaves) + "\n"
        WALKTHROUGH_OUT.write_text(text)
        changed = splice_doc(text)
        print(text, end="")
        print(
            f"\nwrote {WALKTHROUGH_OUT}"
            f" and {'updated' if changed else 'left unchanged'} {SPEC_DOC} §12",
            file=sys.stderr,
        )
        return 0

    selected = [e for e in entries if args.only is None or args.only in e["stem"]]
    if not selected:
        print(f"no fixture matches {args.only!r}", file=sys.stderr)
        return 1

    failed = 0
    for entry in selected:
        path = args.dir / entry["file"]
        data = path.read_bytes()
        if sha256(data) != entry["ifs_sha256"]:
            print(f"FAIL {entry['stem']}: {path} does not match its recorded sha256")
            failed += 1
            continue
        try:
            fails = validate(entry, data)
        except (SpecViolation, IndexError, KeyError) as exc:
            fails = [f"{type(exc).__name__}: {exc}"]
        if fails:
            failed += 1
            print(f"FAIL {entry['stem']}")
            for f in fails:
                print(f"       {f}")
        elif args.stats:
            _, leaves = parse(data)
            hist: dict[int, int] = {}
            for lf in leaves:
                hist[lf.size] = hist.get(lf.size, 0) + 1
            sizes = " ".join(f"{s}:{n}" for s, n in sorted(hist.items()))
            print(f"ok   {entry['stem']:<48} {sizes}")

    if args.only is None:
        try:
            contract_fails = check_contract(entries, args.dir)
        except (SpecViolation, IndexError, KeyError) as exc:
            contract_fails = [f"contract check could not run: {type(exc).__name__}: {exc}"]
        for f in contract_fails:
            print(f"FAIL {f}")
            failed += 1

    # The walkthrough in the spec is generated from a fixture, so it can go stale in
    # exactly the way a hand-written hexdump does. Checked here rather than trusted.
    if args.only is None:
        entry = next(e for e in entries if e["stem"] == WALKTHROUGH_STEM)
        hdr, leaves = parse((args.dir / entry["file"]).read_bytes())
        text = walkthrough((args.dir / entry["file"]).read_bytes(), hdr, leaves) + "\n"
        if not WALKTHROUGH_OUT.exists() or WALKTHROUGH_OUT.read_text() != text:
            print(f"FAIL {WALKTHROUGH_OUT} is stale; run --walkthrough")
            failed += 1
        elif not doc_is_in_sync(text):
            print(f"FAIL {SPEC_DOC} §12 is stale; run --walkthrough")
            failed += 1

    total = len(selected)
    if failed:
        print(f"\n{failed} of {total} fixtures FAILED", file=sys.stderr)
        return 1
    print(f"\nvalidate-ifs: {total} fixtures pass all five checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
