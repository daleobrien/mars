# Fractal Image Compression using Hierarchical Classification of Sub-images
- **Year / venue:** 2015; VISAPP, pp. 46–53; DOI: 10.5220/0005265900460053.
- **Authors:** Nilavra Bhattacharya, Swalpa Kumar Roy, Utpal Nandi, Soumitro Banerjee.
## Core method
- P-I hierarchically classifies domain/range blocks by relative pixel-intensity sums, restricting matching to the same fine-grained class.
- P-II prioritizes frequently selected domains and terminates search when a candidate meets an RMS-error threshold.
## Key techniques
- Level I: rank four quadrant sums into 24 permutations; Level II: rank sub-quadrants within each quadrant into 24^4 combinations.
- Together these yield 24^5 = 7,962,624 possible classes per block size; only sums are required, unlike Fisher mean/variance classification.
- Separate domain pools for 4×4, 8×8, and 16×16 blocks; class buckets hold candidate lists.
- P-II maintains a timesUsed counter and max-heap per class to prioritize reusable domains.
## Reported results
- Java; Core i7-2630QM 2 GHz, 8 GB RAM, Windows 8 x64; five 512×512 8-bit images.
- Lenna: P-I 1.371 s, P-II 1.374 s, versus FISHER24 193.066 s; all give 30.60 dB and 89.58% space savings (derived CR ≈9.60:1).
- P-I across images: 1.082–2.035 s versus FISHER24 147.441–193.066 s; P-II: 1.102–1.995 s.
- PSNR spans 22.22–30.60 dB; space savings 86.88–89.58% (derived CR ≈7.62–9.60:1); quality/storage essentially match FISHER24.
- P-II is faster than P-I on only three of five images; no SSIM or decoding times reported.
## Implementation assessment
- Simple classification arithmetic makes P-I attractive; sparse bucket storage would avoid a huge mostly empty dense class table.
- P-II adds bookkeeping for modest, inconsistent gains; implement and benchmark P-I first.
- Abstract calls the approximately 140× Lenna speedup relative to BFIC, but the actual timing table labels its comparator FISHER24.
- Uniform class occupancy is only an analytical assumption; the “exponential” claim is not a general complexity guarantee.
- Tie handling, empty-class fallback, and thresholds need clarification/testing; preserve quality checks before adopting aggressive candidate exclusion.
