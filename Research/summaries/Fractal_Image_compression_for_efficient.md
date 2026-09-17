# Fractal Image compression for efficient texture mapping
- **Year / venue:** 2004; WSCG Posters proceedings, February 2–6, Plzen, Czech Republic.
- **Authors:** Jerzy Stachera and Sławomir Nikiel.
## Core method
- Compresses natural textures using quadtree partitioned iterated function systems (PIFS), then reconstructs a resolution pyramid for texture mapping and level of detail.
- Uses coarse-to-fine hierarchical decoding rather than repeatedly iterating over the full-resolution texture.
## Key techniques
- RGB-to-YUV conversion; halve chrominance resolution and reconstruct U/V at twice their encoded resolution.
- Domains twice range width, four-pixel averaging, eight isometries, and affine contrast/brightness adjustment.
- Fisher domain classification and nearest-neighbor search; quantized transforms and variable-length domain-location codes.
- Hierarchical decoding follows Baharav’s method with Malah’s quadtree extension; initialize the coarsest level from gray with one transformation application.
- Experiments use quadtree depths 4–6, 5-bit scale, 7-bit offset, and 3-bit isometry parameters.
## Reported results
- Five 512×512 textures/images; fractal CR / PSNR: Brick 89.4:1 / 28.9 dB; Leaf 187.7:1 / 33.8 dB; Sky 139.8:1 / 32.4 dB.
- Marble: 146.7:1 / 32.3 dB; Lenna: 74.1:1 / 26.7 dB.
- Leaf comparison: JPEG 39.7:1 / 34.3 dB; JPEG2000 191.3:1 / 34.4 dB.
- Brick comparison: JPEG 81.7:1 / 29.26 dB; JPEG2000 253.1:1 / 29.5 dB.
- No measured encoding/decoding times, frame rates, or SSIM; real-time feasibility is asserted rather than numerically demonstrated.
## Implementation assessment
- Useful reference for multiresolution texture reconstruction; requires a quadtree codec, color handling, search index, and hierarchical decoder.
- Tables show higher compression than JPEG at somewhat lower PSNR, but JPEG2000 has higher compression and PSNR on every listed image.
- Random texel access is a stated requirement, not a demonstrated implementation; GPU throughput and dependency handling need independent validation.
- Coarsest-level one-step initialization is approximate; resolution scalability does not guarantee recovery of lost texture detail.
- Worth prototyping for LOD research, not treating as a proven replacement for current hardware texture codecs.
