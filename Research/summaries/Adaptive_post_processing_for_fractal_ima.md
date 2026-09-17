# Adaptive Post-Processing for Fractal Image Compression
- **Authors:** Nguyen Ky Giang and Dietmar Saupe.
- **Year / venue:** Not identifiable in this PDF; references include a 2000 publication.

## Core method
- Smooth range-block boundaries during iterative decoding to suppress artifacts copied into block interiors; apply a separate edge-adaptive boundary filter after decoding.
- This addresses both recursive interior artifacts and ordinary blocking without changing the encoded bitstream.

## Key techniques
- Partition-aware weighted averaging before the last 3–4 decoder iterations changes the reconstructed fixed point.
- Final filter uses Sobel gradients and piecewise smoothing strengths to preserve edges and texture.
- Special low-pass treatment at block-boundary intersections; tested with quadtree and adaptive split-and-merge partitions.

## Reported results
- Split-and-merge, Lena/Peppers 512×512: baseline 27.36–32.55 dB at 0.084–0.302 bpp; combined gains 0.26–0.53 dB (Tables 1–2).
- Lena at 0.105 bpp (76:1): 28.65 → 29.08 dB; Peppers at the same rate gains 0.53 dB.
- Quadtree at 0.18–0.23 bpp: gains 0.55–0.59 dB (Tables 3–4); in-loop smoothing alone contributes approximately 0.1 dB.
- No SSIM or runtime measurements reported; post-processing adds no coded data.

## Implementation assessment
- Practical, inexpensive decoder enhancement requiring range-boundary information and access to the iteration loop, not an encoder search speedup.
- Heuristic weights/thresholds need validation on other images; modest PSNR gains but useful artifact reduction. Mathematical glyphs extract poorly, so verify exact filter constants visually before implementation.
