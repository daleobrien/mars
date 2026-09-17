# Proposed Pyramid Fractal Image Compression
- **Author / year / venue:** Jamila Harbi; 2014; *International Journal of Development Research*, 4(5), 929–932.
## Core method
- Builds a mean-image pyramid and performs coarse-to-fine domain matching, refining only promising coarse-level fractal codes.
- Combines this search with quadtree range partitioning and level-dependent encoding-error thresholds.
## Key techniques
- Low-pass filtering / 2×2 averaging and subsampling create progressively coarser images.
- Starts with 16×16 ranges; excessive error triggers splitting into four 8×8 blocks (gain factor 5).
- Uses local-contrast-based error criteria and optimizes affine intensity parameters during matching.
- Contrast and brightness coefficients use 5-bit and 7-bit uniform quantizers.
- Omits the eight-way symmetry search used by the traditional comparison encoder.
## Reported results
- On 256×256 grayscale images, claims approximately **5.7× faster encoding**; cited search comparison is under **144 s** versus under **25 s** at step h=16.
- Figure 1, k=1: **34.97 dB PSNR**, **CR 4.68**, **19 s**, **3,025 blocks**.
- Figure 1, k=2: **32.33 dB PSNR**, **CR 10.12**, **13 s**, **1,399 blocks**.
- Reconstruction reaches the stated attractor in **3 iterations**, versus **8** for traditional FIC.
- Iteration PSNR rises **22.20 → 33.23 → 34.97 dB**; no SSIM reported.
## Implementation assessment
- Attractive deterministic speedup: reduced-resolution comparisons and candidate refinement are reusable in existing fractal encoders.
- Requires pyramid storage, quadtree bookkeeping, and carefully tuned thresholds; dropping symmetries may sacrifice matches.
- Reproduction needs clarification: the paper mixes pyramid search with residual-recompression language, and pseudocode/equations are partly missing or garbled in text extraction.
- Treat reported timings as illustrative; hardware and baseline controls are insufficiently documented.
