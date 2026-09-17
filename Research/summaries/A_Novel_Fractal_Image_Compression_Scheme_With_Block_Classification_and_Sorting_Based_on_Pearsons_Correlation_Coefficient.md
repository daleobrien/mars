# A Novel Fractal Image Compression Scheme with Block Classification and Sorting Based on Pearson’s Correlation Coefficient
- **Authors / year / venue:** Jianji Wang and Nanning Zheng; 2013; IEEE Transactions on Image Processing 22(9), 3690–3702.
- **DOI:** 10.1109/TIP.2013.2268977.
- **Core method:** Exploits the equivalence between least-squares affine block matching and maximizing absolute Pearson correlation (APCC). Classify and sort domains, then search only a small APCC-neighbor set per range block.
## Key techniques
- Fisher's three-class quadrant-luminance ordering canonicalizes geometric orientations, retaining one eighth of the baseline transformed domain pool.
- Sort each class by APCC to an offline-trained preset block; binary search selects an interval of k candidate domains.
- Search canonicalized R and −R, giving approximately 2k comparisons per range block.
- Standardize domains so matching becomes an absolute inner product; precompute classification and transform-composition lookup tables.
- Tune candidate count k and domain stride together; a denser domain pool is not necessarily better with small k.
## Reported results
- C++ on Intel i3-2100 / Windows XP; includes 16 standard 512×512 images plus larger/smaller examples.
- Lena 512×512, 4×4 ranges, domain stride 8: k=44 gives 37.32 dB PSNR in 41.8 ms; k=76 gives 37.82 dB in 61.8 ms.
- Fisher72 comparison: 37.27 dB in 62.2 ms under the same partition.
- Around 0.3 s encoding approaches baseline quality within 0.3 dB, versus approximately 8.7 s baseline (~29× faster).
- Lena candidate comparisons fall from 32,768 per range to 40 at k=20; this is an operation-count reduction, not measured end-to-end speedup.
- Evaluation emphasizes PSNR/time; no numerical compression-ratio or SSIM results reported.
## Implementation assessment
- **Strong candidate:** Practical deterministic search acceleration for a conventional PIFS encoder; no online clustering required.
- Requires offline preset-block training for each class/block size, sorting, and careful handling of ties, flat blocks, negative contrast, and orientation composition.
- Candidate pruning is approximate: validate quality across images and k/stride settings; reported timings exclude offline training.
