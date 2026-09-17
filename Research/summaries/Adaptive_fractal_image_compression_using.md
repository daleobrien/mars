# Adaptive Fractal Image Compression using PSO
- **Authors / year / venue:** A. Muruganandham and R. S. D. Wahida Banu; 2010; ICEBT 2010, Procedia Computer Science 2, 338–344.
- **DOI:** 10.1016/j.procs.2010.11.044.
- **Core method:** Replace exhaustive domain-block search with particle swarm optimization (PSO) minimizing range/domain MSE. Adapt search duration using stagnation of the swarm's best solution rather than adaptive image partitioning.
## Key techniques
- Fixed nonoverlapping L×L ranges and overlapping 2L×2L domains; downsample domains and evaluate eight dihedral orientations.
- Compute least-squares contrast/brightness for each candidate; quantize scale to 5 bits and offset to 7 bits.
- Particles represent candidate domain positions; update velocity using personal-best and global-best solutions.
- Stop when global best remains unchanged for 10% of the maximum PSO iteration budget; relate swarm size to domain count / maximum iterations.
## Reported results
- Table I reports all cases at 0.5 bpp; encoding times explicitly use hh:mm:ss.
- Lena: full search 09:07:20 / 35.80 dB → PSO 00:15:34 / 35.03 dB.
- Goldhill: 09:02:12 / 33.64 dB → 00:17:37 / 32.77 dB.
- Cameraman: 09:02:49 / 35.11 dB → 00:15:24 / 34.23 dB.
- These entries imply approximately 31–35× acceleration with 0.77–0.88 dB PSNR loss; the conclusion separately claims 1.2 dB loss.
- 0.5 bpp implies nominal 16:1 compression for 8-bit input, excluding overhead (derived, not tabulated); no SSIM reported.
## Implementation assessment
- **Reasonable experimental baseline:** Can reuse a conventional PIFS decoder and affine-fitting implementation; no training data required.
- Requires swarm tuning, random-seed control, and discrete coordinate/boundary handling; exact reproducibility details are incomplete.
- Image examples are labeled 256×256, but stopping-criterion plots use 512×512; benchmark hardware is unspecified, so absolute times need caution.
- Medical imaging is suggested, not clinically evaluated; classification is proposed as future improvement rather than part of the demonstrated encoder.
