# Optimization of Fractal Image Compression
- **Author / year / venue:** Rafik Menassel; 2020; IntechOpen, *Fractal Analysis – Selected Examples*.
- **DOI:** 10.5772/intechopen.93051.
## Core method
- Applies wolf pack (WPA) and bat-inspired (BIA) metaheuristics to reduce fractal encoding cost while retaining reconstruction quality.
- WPA searches for matching domains; BIA describes homogeneity-driven block formation and selection, followed by Huffman coding.
## Key techniques
- Nonoverlapping range blocks, larger domains, spatial isometries, and intensity transformations.
- WPA uses scouting and best-candidate updates, terminating after a fixed period without improvement.
- BIA stores block locations, sizes, and mean values in a sparse representation.
- Selected BIA parameters: 8 bats, loudness 8, frequency 30; 10–100 iterations.
## Reported results
- Lena 256×256, WPA: **33.305 dB PSNR**, **668.810 s** encoding, **0.850 s** decoding, **CR 1.596**.
- Lena 256×256, BIA: **33.115 dB**, **732.345 s** encoding, **0.763 s** decoding, **CR 1.604**.
- Listed exhaustive-search comparison: **32.69 dB**, **8,400 s**, **CR 1.3** for Lena 256×256.
- WPA Cameraman 256×256: **31.605 dB**, **774.438 s**, **CR 1.741**. No SSIM reported.
## Implementation assessment
- Potential research prototypes, but hundreds of seconds at 256×256 and modest reported ratios limit practicality.
- Faster comparison methods exist in the chapter, albeit with lower PSNR; these are quality–speed tradeoffs, not universal wins.
- BIA's description is not a complete conventional PIFS specification; consult the cited 2018 WPA and 2019 BAT papers before reproducing it.
- Validate metrics independently: tabulated MSE/PSNR pairs do not consistently match standard 8-bit PSNR, and comparisons lack a clearly controlled common benchmark.
