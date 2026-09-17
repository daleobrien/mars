# Large Texture Storage Using Fractal Image Compression
- **Authors:** Jerzy Stachera, Sławomir Nikiel.
- **Year / venue:** Not identifiable in the supplied PDF; references include a 2004 conference paper, but this does not establish its publication date.
## Core method
- Restricts fractal domain searches to independent square texture regions, enabling region-of-interest access without decoding the entire texture.
- Uses quadtree-adaptive ranges and coarse-to-fine hierarchical decoding for local, multiresolution texture reconstruction.
## Key techniques
- Domains twice range size; 2×2 pixel averaging, eight isometries, affine intensity scaling/offset.
- Search-region size trades compression efficiency against minimum independently decodable region size.
- YUV coding with half-resolution chrominance; fractal upsampling restores output dimensions, not guaranteed original detail.
- Fisher-style allocation: scale **5 bits**, offset **7 bits**, isometry **3 bits**, variable-length domain address.
- Extends prior hierarchical/quadtree decoding (Baharav/Malah); finite resolution-doubling stages rather than repeated full-resolution iteration.
## Reported results
- Six **2048×2048 RGB** textures; **32×32** search regions: **47.41–74.75:1**, **27.25–35.68 dB PSNR**.
- **64×64** search regions: **65.25–182.6:1**, **28.9–33.74 dB**; Sky: **182.6:1 / 33.74 dB**, Earth: **100:1 / 30.36 dB**.
- Sky comparison: JPEG **161.1:1 / 31.4 dB**, JPEG2000 **542.9:1 / 36.7 dB**; JPEG2000 has higher ratios and PSNR on most listed textures.
- Claims linear-time local decoding and near-real-time operation, but reports **no measured encoding/decoding times or SSIM**.
## Implementation assessment
- Worth considering for independent texture tiles and multiresolution access, rather than best rate–distortion performance.
- Requires a custom codec/decoder; encoding remains expensive and local search sacrifices cross-region matches.
- Validate actual GPU/mobile throughput against modern texture formats: real-time and super-resolution claims are not quantitatively demonstrated here.
