# Evaluation of Pure-Fractal and Wavelet-Fractal Compression Techniques
- **Year / venue:** 2009; ICGST-GVIP Journal, 9(4), August, pp. 41–47.
- **Authors:** Mohammad R. N. Avanaki, Hamid Ahmadinejad, Reza Ebrahimpour.
## Core method
- Compares spatial-domain fractal coding with wavelet-tree fractal prediction on repetitive cDNA microarray images.
- Spatial coding fits downsampled domains to ranges using scale/offset; wavelet coding predicts coefficients across scales without an additive offset.
## Key techniques
- Sliding domain pool; domains twice the range width; least-squares matching and iterative spatial decoding (typically eight iterations).
- Wavelet-tree matching with either shared or per-level scale factors; experiments include quantized scales and least-squares scales.
- Wavelet implementation omits rotations and flips; block size trades reconstruction quality against storage and search cost.
## Reported results
- MATLAB 2007, 2 GHz Pentium IV, 1 GB RAM; spatial experiment uses a 64×64 crop of a 16-bit microarray image.
- Table 1 spatial PSNR / CR: 50.6 dB / 5.33:1; 40.5 / 25.6:1; 33.32 / 128:1; 29.55 / 682:1.
- Table 2 wavelet PSNR / CR: 28.81 dB / 328:1; 35.51 / 23.67:1; 30.9 / 99.9:1; 31.19 / 63:1; 36.15 / 16.7:1.
- Summary table differs from detailed tables (e.g., spatial 51 dB / 5.12:1); figures above preserve detailed-table values.
- No numerical encoding/decoding timings or SSIM reported; wavelet coding is described as less computationally expensive.
## Implementation assessment
- Straightforward spatial baseline; wavelet variant adds transform/tree bookkeeping and scale-quantization choices.
- Speedups such as classification, k-d trees, and parallelism are cited possibilities, not demonstrated components here.
- Extreme compression ratios come with substantial quality loss and tiny, specialized test data; benchmark on representative full-size images.
- “Semi-lossless” is the authors’ label, not losslessness; expert visual judgments do not establish preservation of downstream measurements.
