# Using Adaptive Contraction for Fractal Image Coding Based on Local Fractal Dimension
- **Authors:** Aura Conci and Felipe R. Aquino, Universidade Federal Fluminense.
- **Year / venue:** Not identifiable in the supplied PDF text; references include a 1999 publication.
## Core method
- Estimates local fractal dimension (LFD) to choose a suitable spatial contraction/domain pool for each group of range blocks.
- Smooth regions use stronger reductions and smaller candidate pools; complex regions retain finer search detail.
## Key techniques
- Modified differential box-counting estimates LFD over groups of 2×2 range blocks.
- Three complexity bands: **[2, 2.33)**, **[2.33, 2.66)**, and **[2.66, 3]**.
- Domain pools use spatial contractions **1/4, 1/5, 1/6**, averaging 4×4, 5×5, or 6×6 pixels per reduced pixel.
- Searches the selected pool for each range; retains affine brightness/contrast fitting and eight spatial symmetries.
- LFD-controlled range-block sizes are future work, not part of the tested algorithm.
## Reported results
- Same-platform comparison on four **128×128, 8-bit grayscale** images: Milk, Lena, Goldhill, Peppers.
- Proposed averages: **44.45 s**, **RMS error 9.88**, **SNRrms 9.35**, **1.5625 bpp**.
- Exhaustive averages: **469.39 s**, **RMS error 7.90**, **SNRrms 11.80**, **3.5020 bpp**; proposed encoding is approximately **10.6× faster** (derived).
- Restricted-area search: **106.64 s**, **RMS error 9.11**, **1.4375 bpp**; local search: **9.48 s**, **RMS error 11.71**, **1.4375 bpp**.
- Proposed bitrate corresponds to **5.12:1** versus raw 8-bit pixels (derived). SNRrms is a linear root ratio, not PSNR/dB; no SSIM.
## Implementation assessment
- Promising deterministic pool-selection strategy, with a measurable quality–speed tradeoff rather than lossless acceleration.
- Requires multiscale domain buffers, LFD preprocessing, and storing/decoding the selected scale; benchmark overhead on larger images.
- Exact LFD estimator details depend on cited Aquino/Conci work; naive estimators may saturate and misclassify blocks.
- Small evaluation set and differing output bitrates limit conclusions; compare all methods at matched rate and reconstruction quality.
