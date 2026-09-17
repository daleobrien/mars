# Stochastic Image Compression Using Fractals
- **Authors / year / venue:** Aditya Kapoor, Kush Arora, Ajai Jain, G. P. Kapoor; 2003; IEEE ITCC (Information Technology: Computers and Communications).
## Core method
- Extends Jacquin-style fractal coding by learning the most frequent best isometry from an initial subset of edge blocks, then reusing it for remaining edge blocks.
- Avoids repeated eight-isometry searches and omits redundant isometry identifiers from the output.
## Key techniques
- Classifies domain/range blocks as shade, midrange, or edge; builds domains using a sliding window.
- Shade blocks store only their mean; edge detection uses Sobel-like gradients with threshold 30.
- Midrange classification uses a variance-derived statistic; contrast is quantized to {0.5, 0.6, 0.7, 0.8, 0.9, 1.0}.
- Initial isometry-learning subset is exemplified as 100 blocks; decoding iteratively applies stored transforms.
## Reported results
- Pentium III 700 MHz; Lena 4×4 ranges: **3 min 32 s → 1 min 23 s**, **39,795 → 35,280 bytes**.
- Baboon 4×4: **8 min 29 s → 2 min 42 s**, **56,433 → 48,988 bytes**.
- Baboon 2×2: **54 min 7 s → 19 min 23 s**, **209,574 → 182,703 bytes**.
- Reported stochastic/non-stochastic **SNR**: Peppers **43.20/42.70**, Face **46.94/43.73**, Cat **46.64/46.86**; not identified as PSNR. No SSIM.
- Example 256×256, 4×4-range compression ratios are stated as **1:8 (Lena)** and **1:6 (Baboon)**.
## Implementation assessment
- Simple heuristic to prototype, but a globally preferred isometry may fail on images with diverse orientations; retain a quality-based fallback.
- Smaller 2×2 ranges reduce visible blockiness but sharply increase encoding cost and file size.
- Claimed **55–80% time savings** are broadly plausible, but several percentage entries disagree with their raw timings.
- The abstract's **60–80% size reduction versus non-stochastic coding** is not supported by the paired sizes above (roughly 11–13% savings).
- Ratios use unusually large PGM source-file sizes (~266 KB for 256×256 grayscale), so compare against raw 8-bit pixels before adopting claims.
