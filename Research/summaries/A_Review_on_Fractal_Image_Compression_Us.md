# A Review on Fractal Image Compression Using Optimization Techniques
- **Authors / year / venue:** Shaimaa S. AL-Bundi and Mustafa S. Abd; 2020; Journal of Al-Qadisiyah for Computer Science and Mathematics 12(1), Comp 38–48.
- **DOI:** 10.29304/jqcm.2020.12.1.674; use first-page metadata despite inconsistent running headers.
- **Core method:** Surveys metaheuristic solutions to the expensive domain-search / inverse-IFS problem. This is a review and comparison, not a complete specification of a single new codec.
## Key techniques
- Genetic algorithms (GA): selection, crossover, mutation, variable mappings, and quadtree-based variants.
- Crowding optimization (COM): preserve population diversity and avoid premature convergence through pairing/replacement rules.
- Particle swarm optimization (PSO): personal/global-best search, stopping criteria, and cited wavelet-classification or edge-guided variants.
- Harmony search (HSA): harmony memory, pitch adjustment, random exploration, and replacement of poor solutions.
## Reported results
- Table 1 compares four pictured images with range-block size 4; coding-time units and hardware are not specified there.
- GA: coding time 1.94–3.08; MSE 0.109–0.262; CR 7.33–12.6.
- COM: coding time 0.87–0.99; MSE 0.026–0.39; CR 4.33–7.04.
- PSO: coding time 0.98–2.98; MSE 0.025–0.094; CR 5.01–6.45.
- HSA: coding time 0.107–0.110; MSE 0.026–0.088; CR 5.5–8.5.
- Cites a 2008 visual-based PSO study reporting 125× acceleration with 0.89 dB PSNR loss versus full search; not a new benchmark of this review.
- No SSIM results; Table 1 uses MSE rather than PSNR and does not explain its normalization.
## Implementation assessment
- **Useful as a literature map:** Follow the cited original algorithms/thesis for executable detail and parameter settings.
- HSA has the lowest tabulated coding times, while GA has higher CR; the table does not establish a universal rate–distortion winner.
- Reproducibility is limited by unclear benchmark conditions and stochastic tuning; compare original methods under a shared codec, bitrate, hardware, and repeated seeds.
