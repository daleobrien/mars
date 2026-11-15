# Encoding and decoding

[Back to the README](../README.md)

```bash
cargo build --release -p mars-cli
./target/release/encmars lena.jpg lena.mars
./target/release/decmars lena.mars decoded.png
```

A plain encode uses RD partitioning at `--lambda 200`, modes `0,2` (flat and fractal),
4:4:4 colour, and automatic thread selection. The mode choice follows the existing
kodim01/02 comparisons, not a claim of universal optimality. RD can be slower than
legacy threshold encoding. Library/benchmark defaults are unchanged.

Inputs may be `.png`, `.jpg`/`.jpeg`, `.tga`, `.tif`/`.tiff`, `.webp`, `.pgm`, `.ppm` or
headerless raw. PNG, JPEG, TIFF and WebP are detected from the file's contents rather than
its name; TGA (which has no signature), PNM and raw are selected by extension. `decmars`
picks its output format from the output filename's extension: `.png`, `.tga`, `.tif`/`.tiff`
and `.webp` (all lossless, though WebP has no grayscale mode), `.jpg`/`.jpeg` (lossy, for
visual inspection only; `decmars --quality 1-100` sets its quality, default 75) or
`.pgm`/`.ppm` (raw PNM).

- `--lambda 50`: prioritise quality; `--lambda 800`: prioritise smaller files.
- `--subsampling 420`: optionally trade chroma detail for smaller colour files.
- `--modes 0,1,2,3`: explicitly enable affine and residual modes too.
- `--t-rms 8`: restore legacy threshold partitioning. `--chroma-t-rms` also selects
  that path unless `--lambda` is supplied; explicit lambda takes precedence and ignores
  the thresholds. The rate-estimation warm-up uses fixed RMS 8.
- `--method fisher --lambda 200 --modes 0,2`: use indexed search with production
  grayscale RD partitioning. All nine existing methods support this path; method alone
  retains legacy threshold 8. Color/method and progressive/method remain unsupported.
  Method output separates search fits, exhaustive warm-up fits, and total fits.
- Explicit masks are strict: if a block needs mandatory DC fallback but mode 0 was
  excluded, encoding fails before writing output. Prefer `0,2` or `0,2,3` over a
  fractal-only mask when using restricted search or tiny/border-heavy images.
- Explicit `--modes` and `--adaptive-density` require RD, rather than silently doing
  nothing on the legacy/method path. Numeric options are checked before reading input;
  domain stride must be even and contrast limits must be representable in the header.
- Adaptive density, progressive output and experimental adaptive residual quantisation
  remain opt-in. To try the latter, include mode 3:
  `--lambda 200 --modes 0,2,3 --adaptive-residual`.

Progressive streams also honor decoder iteration counts:

```bash
encmars input.pgm output.mars --progressive
decmars output.mars decoded.png --layer 4 --iterations 20
```

For regular streams, `decmars --auto --iterations 40` stops when the pixel-change
threshold is reached or the iteration limit is exhausted. `--progression frames/`
writes iteration images; failures writing frames are reported as command failures.
`--debug-rects tree.png` writes native-resolution leaf boundaries for regular streams.
Progressive streams still reject auto/progression/zoom/debug-rects combinations.

## More examples

```bash
encmars input.png output.mars --lambda 200 --modes 0,2 --subsampling 420

encmars input.png output.mars --lambda 50       # Higher quality
encmars input.png output.mars --subsampling 420 # Smaller colour files
encmars input.png output.mars --t-rms 8         # Legacy encoding
```
