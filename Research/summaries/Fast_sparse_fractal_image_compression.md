# Fast sparse fractal image compression
- **Year / venue:** 2017; PLOS ONE 12(9), e0184408; DOI: 10.1371/journal.pone.0184408.
- **Authors:** Jianji Wang, Pei Chen, Bao Xi, Jianyi Liu, Yi Zhang, Shujian Yu.
## Core method
- Approximates each fixed-size range with a sparse linear combination of multiple domain blocks plus an offset, rather than one domain.
- FSFIC combines adaptive sparse coding with absolute Pearson correlation (APCC) classification/sorting to accelerate each residual search.
## Key techniques
- Matching pursuit or orthogonal matching pursuit (OMP; used experimentally); stop at an error threshold or maximum domain count k1.
- Fisher three-class canonicalization, eight geometric isometries, and correlation sorting against an offline-trained reference block per class.
- Search only k2 nearby entries in the sorted class; reuse one fixed-size domain pool throughout coding.
- Adapt domain count instead of quadtree block size; store count, domain indices/transforms, coefficients, and offset.
## Reported results
- VC++6.0 on Intel i3-2100 / Windows XP; predominantly 512×512 images with 8×8 ranges.
- Two-domain exhaustive BSFIC: mean PSNR 30.80 vs BFIC 28.42 dB across 13 images (+2.38 dB); Lena 35.43 vs 32.20 dB.
- FSFIC Lena: approximately 35.5 dB in 0.3 s versus approximately 0.75 s for sorting-only and polar-angle/NRMS sparse-search alternatives.
- With k1=4: k2=100 takes under 0.2 s; k2=300 approximately 0.2–0.4 s on the three parameter-study images.
- Full-candidate rate-quality optimization: Lena 33.97 dB at CR 10:1 and 37.03 dB at 5:1; these are not the restricted-search timing settings.
- At CR 7.6418:1, FSFIC Lena/Baboon/Pepper PSNR is 35.40/23.59/34.18 dB; exhaustive BSFIC takes over 3 minutes per image.
- No SSIM or numerical decoding-time results reported; convergence improvements are empirical.
## Implementation assessment
- Promising quality/speed extension if APCC indexing already exists; depends on the authors’ 2013 APCC method and trained reference blocks.
- OMP adds least-squares updates and multi-domain decoding; more atoms increase storage and encoding work, with diminishing quality gains.
- Start with k1=4 and k2=100–300 as paper-tested settings; validate quantization, error-threshold normalization, and iterative convergence independently.
