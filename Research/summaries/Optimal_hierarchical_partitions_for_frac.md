# Optimal Hierarchical Partitions for Fractal Image Compression
- **Authors:** Dietmar Saupe, Matthias Ruhl, Raouf Hamzaoui, Luigi Grandi, Daniele Marini.
- **Year / venue:** 1998; IEEE International Conference on Image Processing (ICIP), Chicago, October.
## Core method
- Build a fine horizontal/vertical (HV) binary partition using image statistics, encode every node, then prune bottom-up with generalized BFOS to minimize distortion plus a weighted bit rate.
- Optimality is over prunings of the chosen initial tree using **collage error**, not all possible partitions or actual decoded-image distortion.
## Key techniques
- Choose horizontal/vertical splits minimizing constant-block squared error; bias against thin rectangles; terminal blocks have sides of 2 or 3 pixels.
- Domains come from the image downsampled by two; affine least-squares matching with quantized, contractive scaling; no isometries in this implementation.
- Cache/reuse hierarchical inner products; recursive scheduling limits temporary arrays to a root-to-leaf path.
- Count partition side information in the rate objective; **5-bit scale + 6-bit offset + 16-bit domain address**, or **11 bits** for constant blocks.
- BFOS requires additive, monotonic rate/error costs; constant-block cases may require a monotonicity correction.
## Reported results
- **512×512 Lenna only:** ratio / decoded PSNR: **6.47:1 / 39.10 dB**, **12.65:1 / 36.07 dB**, **20.76:1 / 33.89 dB**.
- Higher compression: **31.00:1 / 32.13 dB**, **61.43:1 / 29.43 dB**, **123.78:1 / 27.05 dB**.
- Beats tested greedy collage-error and variance partitions; matches the cited fully enabled Fisher–Menlove HV coder without entropy coding and with a smaller domain pool.
- Encoding the entire rate–distortion curve took **several hours**; no precise per-image timing or SSIM.
- Preliminary adaptive arithmetic coding reduced partition side information to **about 80%** of the tabulated sizes; not included in main results.
## Implementation assessment
- Strong candidate for rate control in an existing fractal encoder; requires node-level encodings, exact bit accounting, and a BFOS pruning implementation.
- Main cost is searching domains for all tree nodes; acceleration and memory planning matter more than pruning itself.
- Evidence is proof-of-concept on one image; benchmark decoded distortion separately from the optimized collage-error surrogate.
