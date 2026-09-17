# Graph-Directed Fractal Image Compression
- **Authors:** Derya Çelik, Ali Deniz, Yunus Özdemir.
- **Year / venue:** 2022; Eskişehir Technical University Journal of Science and Technology B – Theoretical Sciences, 10(1), 1–10.
- **DOI:** 10.20290/estubtdb.988203.
## Core method
- Generalizes partitioned fractal coding to jointly compress multiple images: each range may reference a domain in any image in the group.
- Proves contraction of the joint operator on a product function space, guaranteeing a unique jointly decoded fixed point.
## Key techniques
- Fixed square ranges; domains twice the linear size; eight rotation/reflection orientations.
- Affine contrast/brightness mappings; exhaustive cross-image domain search plus an image identifier per range.
- Joint iterative reconstruction; no classification, adaptive partitioning, or search acceleration demonstrated.
## Reported results
- Two 256×256 images, 4×4 ranges, 8×8 domains: PSNR **19.5 and 24.33 dB after 10 iterations**.
- Joint code: **32 bits/range**, 8,192 ranges, **32,768 bytes** total; implies **4:1** versus two raw 8-bit grayscale images (derived, excluding headers).
- Classical illustrative code uses 31 bits/range and 15,872 bytes/image; joint coding adds one bit/range for two images.
- **992,016 comparisons/range** for two images versus 496,008 for one; no measured encoding time or SSIM.
## Implementation assessment
- Primarily a theoretical extension; plausible for related image collections, but larger domain pools do not establish better decoded PSNR experimentally against a matched baseline.
- For N images, search work per range grows roughly N-fold and identifiers cost ceil(log2 N) bits; cross-image references couple decoding and storage dependencies.
- Worth prototyping only with search acceleration and matched rate–distortion tests; not a demonstrated compression-ratio or speed improvement.
