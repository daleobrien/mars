# The JPEG Algorithm for Image Compression: A Software Implementation and some Test Results
- **Authors / year:** David Wright and fred harris; 1990.
- **Venue:** 24th Asilomar Conference on Signals, Systems and Computers (24ACSSC footer), pp. 870–875.
## Core method
- Implements baseline sequential JPEG and compares per-image optimized, ensemble-trained, and draft-default Huffman tables on nine ISO color images.
- This is a JPEG implementation/statistics study, not a fractal compression method.
## Key techniques
- YUYV color representation with horizontal chroma subsampling; 8×8 DCT, perceptual quantization, and zigzag ordering.
- Separate DC and AC coding; AC zero-run/significance Huffman symbols followed by coefficient-value bits.
- Four Huffman distributions distinguish luminance/chrominance and DC/AC; image-specific histograms support optimized tables.
- Fixed-point DCT with eight 2,048-entry multiplication lookup tables; assembly replaces slow QuickBasic transform routines.
## Reported results
- Nine 720×576 images: optimized coding **0.526–1.055 bpp**, average **0.789 bpp**; ensemble **0.796**, default **0.823 bpp**.
- Against 24-bit RGB, 0.789 bpp implies approximately **30.4:1** compression (derived, before excluded overhead).
- Average optimized/default payload sizes: **40,903 / 42,648 bytes**; headers/restart markers excluded, and bpp figures omit byte stuffing/markers.
- Initial implementation: approximately **600 pixels/s**; assembly DCT approximately **25× faster**; mixed-language decoder approximately **20,000 pixels/s** on a 12 MHz PC-AT.
- No PSNR/SSIM or full encoding-time table; reconstruction quality is assessed subjectively as high.
## Implementation assessment
- Useful conventional-codec baseline and evidence that default Huffman tables lose little efficiency versus image-specific tables.
- Custom tables typically require under **600 bytes**, but add substantial encoding work; optimize only when small size gains matter.
- Historical lookup-table/assembly tricks are hardware-specific; use a maintained JPEG library for production rather than this draft-Revision-5 description.
