#!/usr/bin/env python3
"""Build the image-set index the Mars 1 baseline sweep consumes.

Mars 1 reads **headerless 8-bit grayscale** (`readimage_raw()`,
reference/mars1/image_io.c:158) and takes the dimensions on the command line. The
corpus, meanwhile, is colour PNG (Kodak) or already-raw synthetic fixtures. This
script bridges the two and emits one JSON schema for both, so the Rust sweep
driver has a single thing to read:

    {"set": ..., "provenance": {...},
     "images": [{"name","file","width","height","bytes","sha256"}, ...]}

Two rules it exists to enforce:

1. **The grayscale conversion stays pinned in `ppm2raw.py`.** ffmpeg only decodes
   the PNG container to PPM, which involves no colour transform; the Rec.601 luma
   arithmetic happens in code we version. ImageMagick's and ffmpeg's "to gray"
   defaults are version-dependent (Rec.601 vs Rec.709, luma vs linear-light), and
   a corpus whose grayscale depends on the installed tool version is not pinned.

2. **Nothing is regenerated silently.** `--check` recomputes every hash and exits
   non-zero on any difference, so a changed corpus cannot slip into a result row
   whose provenance block still cites the old manifest (M7).

Usage:
  build-imageset.py --set standard   [--check]
  build-imageset.py --set fixtures   [--check]
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

try:
    import tomllib
except ModuleNotFoundError:                                # Python < 3.11
    sys.exit("python 3.11+ required (tomllib) to read the fixtures manifest")

ROOT = pathlib.Path(__file__).resolve().parent.parent
PPM2RAW = ROOT / "scripts" / "ppm2raw.py"

SETS = {
    "standard": {
        "source": "corpus/kodak.manifest.json",
        "gray_dir": "corpus/images/kodak-gray",
        "out": "corpus/standard.images.json",
        "role": "M9 `standard`: Kodak, 24 images. All headline RD numbers.",
    },
    "fixtures": {
        "source": "fixtures/images/manifest.toml",
        "gray_dir": None,                                  # already raw grayscale
        "out": "corpus/fixtures.images.json",
        "role": "M9 `fixtures`: synthetic edge cases. Not a headline corpus.",
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_version(argv):
    try:
        out = subprocess.run(argv, capture_output=True, text=True, check=True)
        return out.stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def png_to_gray_raw(png, dest, workdir):
    """PNG -> PPM (container decode only) -> pinned Rec.601 raw. Returns (w, h, sha)."""
    ppm = workdir / (png.stem + ".ppm")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(png),
         "-pix_fmt", "rgb24", "-f", "image2", "-c:v", "ppm", str(ppm)],
        check=True,
    )
    out = subprocess.run(
        [sys.executable, str(PPM2RAW), str(ppm), "-o", str(dest)],
        capture_output=True, text=True, check=True,
    )
    w, h, sha = out.stdout.split()
    ppm.unlink(missing_ok=True)
    return int(w), int(h), sha


def build_standard(check):
    spec = SETS["standard"]
    manifest_path = ROOT / spec["source"]
    manifest = json.loads(manifest_path.read_text())
    src_dir = ROOT / manifest["dir"]
    gray_dir = ROOT / spec["gray_dir"]
    gray_dir.mkdir(parents=True, exist_ok=True)

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg required to decode the PNG corpus (brew install ffmpeg)")

    images, failed = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        for entry in manifest["images"]:
            png = src_dir / entry["name"]
            if not png.is_file():
                sys.exit(f"{png} missing; run `just corpus` first")
            stem = png.stem
            dest = gray_dir / f"{stem}.raw"
            if check:
                if not dest.is_file():
                    print(f"MISSING  {dest.relative_to(ROOT)}", file=sys.stderr)
                    failed += 1
                    continue
                w, h, sha = png_to_gray_raw(png, workdir / f"{stem}.raw", workdir)
                got = sha256_file(dest)
                if got != sha:
                    print(f"MISMATCH {stem}: on disk {got}, regenerated {sha}",
                          file=sys.stderr)
                    failed += 1
                    continue
                print(f"ok       {stem} ({w}x{h})")
            else:
                w, h, sha = png_to_gray_raw(png, dest, workdir)
                print(f"gray     {stem} ({w}x{h})")
            images.append({
                "name": stem,
                "file": str(dest.relative_to(ROOT)),
                "width": w,
                "height": h,
                "bytes": w * h,
                "sha256": sha,
                "source": str(png.relative_to(ROOT)),
            })

    if failed:
        sys.exit(f"{failed} image(s) failed the check")

    provenance = {
        "source_manifest": spec["source"],
        "source_manifest_sha256": sha256_file(manifest_path),
        "container_decoder": tool_version(["ffmpeg", "-version"]),
        "container_decode_cmd": "ffmpeg -i IN.png -pix_fmt rgb24 -f image2 -c:v ppm OUT.ppm",
        "gray_script": "scripts/ppm2raw.py",
        "gray_script_sha256": sha256_file(PPM2RAW),
        "gray_definition": "Y = 0.299R + 0.587G + 0.114B (Rec.601 luma on sRGB-encoded "
                           "values), rounded half away from zero, clamped to [0,255]",
        "layout": "8-bit grayscale, headerless, row-major, width-fast",
    }
    return images, provenance


def build_fixtures(check):
    spec = SETS["fixtures"]
    manifest_path = ROOT / spec["source"]
    manifest = tomllib.loads(manifest_path.read_text())
    base = manifest_path.parent

    images, failed = [], 0
    for entry in manifest["image"]:
        path = base / entry["file"]
        if not path.is_file():
            print(f"MISSING  {path.relative_to(ROOT)}; run scripts/gen-fixtures.py",
                  file=sys.stderr)
            failed += 1
            continue
        got = sha256_file(path)
        if got != entry["sha256"]:
            print(f"MISMATCH {entry['name']}: on disk {got}, manifest {entry['sha256']}",
                  file=sys.stderr)
            failed += 1
            continue
        print(f"ok       {entry['name']} ({entry['width']}x{entry['height']})")
        images.append({
            "name": entry["name"],
            "file": str(path.relative_to(ROOT)),
            "width": entry["width"],
            "height": entry["height"],
            "bytes": entry["bytes"],
            "sha256": got,
            "source": spec["source"],
        })

    if failed:
        sys.exit(f"{failed} fixture(s) failed the check")

    provenance = {
        "source_manifest": spec["source"],
        "source_manifest_sha256": sha256_file(manifest_path),
        "generator": manifest["generator"]["script"],
        "generator_sha256": manifest["generator"]["script_sha256"],
        "gray_definition": "synthetic, generated directly as 8-bit grayscale",
        "layout": manifest["generator"]["format"],
    }
    return images, provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--check", action="store_true",
                    help="verify hashes instead of writing; non-zero exit on any change")
    args = ap.parse_args()

    spec = SETS[args.set]
    builder = build_standard if args.set == "standard" else build_fixtures
    images, provenance = builder(args.check)
    if not images:
        sys.exit(f"image set {args.set!r} is empty")

    index = {
        "set": args.set,
        "role": spec["role"],
        "count": len(images),
        "provenance": provenance,
        "images": images,
    }
    out = ROOT / spec["out"]
    text = json.dumps(index, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not out.is_file():
            sys.exit(f"{out.relative_to(ROOT)} missing")
        if out.read_text() != text:
            sys.exit(f"{out.relative_to(ROOT)} is stale; re-run without --check")
        print(f"checked  {len(images)} image(s) against {out.relative_to(ROOT)}")
        return 0

    out.write_text(text)
    print(f"wrote    {out.relative_to(ROOT)} ({len(images)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
