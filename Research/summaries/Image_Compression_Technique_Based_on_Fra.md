# Image Compression Technique Based on Fractal Image Compression Using Neural Network – A Review
- **Author:** Diyar Waysi Naaman.
- **Year / venue:** 2021; Asian Journal of Research in Computer Science, 10(4), 47–57.
- **DOI:** 10.9734/AJRCOS/2021/v10i430249; mini-review, not a new controlled benchmark.
## Core method
- Reviews neural-network-assisted fractal compression, especially MRI coding: train a back-propagation expert system from range/domain matches, then restrict encoding searches to predicted candidate domains.
- Store fractal transformation parameters and reconstruct iteratively, with approximately 10 decoding iterations described.
## Key techniques
- Block descriptors: mean, skewness, standard deviation; eight domain orientations and within-class minimum-distance matching.
- Training on similar images; affine scaling/offset coding and Huffman coding.
- Surveys quadtree partitioning, hybrid genetic–neural methods (HGANN), and FPGA implementations; these are separate cited approaches.
## Reported results (reproduced from earlier studies)
- **Barbara, 256×256:** FIC / NN / HGANN: PSNR **32.674 / 29.788 / 30.05 dB**; ratios **1.2:1 / 6.73:1 / 6.73:1**; encoding **8,400 / 2,800 / 2,978 s**.
- **Butterfly, color:** same ordering: **28.534 / 24.632 / 24.978 dB**, **1.1:1 / 6.73:1 / 6.73:1**, **25,000 / 7,500 / 7,590 s**.
- Medical-image table: fixed FIC **109.16 s**, quadtree **6.7 s**; GUI quadtree without/with NN **295.6 / 222.406 s**.
- FPGA value **66.45 ns** is printed but lacks enough timing context for a credible whole-image comparison; no SSIM reported.
## Implementation assessment
- Useful as a bibliography and candidate-pruning concept, not a reproducible implementation specification; consult the cited Lakshmi/Rao and Chakrapani/Rajan papers.
- Training cost, candidate recall, model inputs/targets, hardware, and generalization require independent validation.
- Claims of preserved/improved quality conflict with Table 1: NN variants reduce PSNR while increasing compression; comparisons are not at matched rates.
