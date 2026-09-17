# An Improved Fractal Image Compression Using Wolf Pack Algorithm
- **Authors:** R. Menassel, B. Nini, T. Mekhaznia.
- **Year / venue:** 2017, Journal of Experimental & Theoretical Artificial Intelligence; DOI: 10.1080/0952813X.2017.1409281.

## Core method
- Replace exhaustive fractal domain-block search with Wolf Pack Algorithm (WPA), a population-based stochastic optimizer seeking similar 2b×2b domains for b×b range blocks.

## Key techniques
- Candidate wolves move relative to the best leader using `x_i ← x_i + λ|x_g − x_i|`; weak candidates are replaced randomly.
- Standard fractal contraction/block matching supplies fitness; stopping uses iteration limits or lack of leader improvement.
- Experiments use 20 scouting wolves, 10–100 iterations, five-run averages; experimental λ range is [0.5,1], versus [-1,1] in the general algorithm.

## Reported results
- 64×64 images (Tables 1–2): exhaustive → WPA time is 3.11 → 2.04 s (Peppers), 2.28 → 1.98 s (Building), 3.28 → 2.83 s (Boat): roughly 1.15–1.52× speedup.
- Table 2 WPA ratios: 1.109–1.111 at 64×64, 1.199–1.231 at 128×128, 1.294–1.355 at 256×256; largest images take 20.143–27.867 s.
- Lena ratio: WPA 1.655 vs PSO 1.89 and single-level GA 1.277; comparisons use separate tables/settings.
- Quadtree beats WPA on 64×64 Cameraman: ratio 2.212 vs 1.121, time 0.983 vs 1.99 s.
- No WPA PSNR/SSIM reported; PSNR numbers elsewhere describe other methods. Table 1's “quality ratio (%)” is ambiguously defined.

## Implementation assessment
- Low-priority experimental baseline: modest demonstrated speedup, low reported ratios, stochastic tuning, and no quantitative reconstruction-quality evidence.
- Requires an existing fractal codec plus WPA search; unclear parameter/ratio definitions and preliminary comparisons limit reproducibility and claims of superiority.
