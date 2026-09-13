# Mars 2 -- Kodak rate-distortion: five anchor codecs + six Mars 1 methods

> 1536 anchor rows from `results/anchors.jsonl` (+ 9360 Mars 1 rows from `results/baseline-mars1.jsonl`) · report at be7f5ee0b422-dirty · harness 0.1.0 · Apple M3 Pro (6 P / 6 E cores) · 2026-09-13T09:10:01Z

<details><summary>Anchor codec versions</summary>

```
avif: Version: 1.4.2 (dav1d [dec]:1.5.4, aom [enc/dec]:3.15.0)
jpeg2000: It compresses various image formats with the JPEG 2000 algorithm.
jpeg: libjpeg-turbo version 3.2.0 (build 20260630)
jpegxl: cjxl v0.8.2 0.8.2 [SSE4,SSSE3]
webp: 1.6.0
```

</details>

## Operating points

| curve | bpp | PSNR (dB) |
|---|---:|---:|
| avif | 0.0800 | 26.398 |
| avif | 0.0895 | 26.709 |
| avif | 0.1004 | 27.093 |
| avif | 0.1192 | 27.619 |
| avif | 0.1559 | 28.479 |
| avif | 0.1952 | 29.247 |
| avif | 0.2544 | 30.301 |
| avif | 0.3649 | 31.790 |
| avif | 0.5407 | 33.781 |
| avif | 0.8231 | 36.155 |
| avif | 1.2822 | 39.472 |
| avif | 1.7756 | 42.173 |
| jpeg | 0.1796 | 23.458 |
| jpeg | 0.2223 | 25.540 |
| jpeg | 0.2867 | 27.315 |
| jpeg | 0.3687 | 28.671 |
| jpeg | 0.4618 | 29.825 |
| jpeg | 0.5591 | 30.820 |
| jpeg | 0.6598 | 31.687 |
| jpeg | 0.7856 | 32.629 |
| jpeg | 0.9055 | 33.427 |
| jpeg | 1.1265 | 34.740 |
| jpeg | 1.5702 | 37.038 |
| jpeg2000 | 0.1196 | 27.112 |
| jpeg2000 | 0.1709 | 28.194 |
| jpeg2000 | 0.2396 | 29.369 |
| jpeg2000 | 0.3417 | 30.768 |
| jpeg2000 | 0.4790 | 32.286 |
| jpeg2000 | 0.6841 | 34.122 |
| jpeg2000 | 0.9578 | 36.071 |
| jpeg2000 | 1.3313 | 38.151 |
| jpeg2000 | 1.8441 | 40.398 |
| jpeg2000 | 2.6640 | 43.058 |
| jpeg2000 | 3.9958 | 46.277 |
| jpeg2000 | 5.9954 | 51.290 |
| jpegxl | 0.1541 | 26.840 |
| jpegxl | 0.2011 | 27.881 |
| jpegxl | 0.2648 | 29.012 |
| jpegxl | 0.3114 | 29.772 |
| jpegxl | 0.3629 | 30.500 |
| jpegxl | 0.4256 | 31.295 |
| jpegxl | 0.5047 | 32.273 |
| jpegxl | 0.6080 | 33.361 |
| jpegxl | 0.7303 | 34.520 |
| jpegxl | 0.8627 | 35.663 |
| jpegxl | 1.0199 | 36.867 |
| jpegxl | 1.1924 | 38.067 |
| jpegxl | 1.4495 | 39.628 |
| jpegxl | 1.7387 | 41.189 |
| jpegxl | 2.2127 | 43.426 |
| jpegxl | 3.1434 | 47.005 |
| mars1-fisher | 0.1728 | 25.229 |
| mars1-fisher | 0.4174 | 27.787 |
| mars1-fisher | 0.8374 | 30.016 |
| mars1-fisher | 1.1853 | 30.838 |
| mars1-fisher | 1.4762 | 31.019 |
| mars1-hurtgen | 0.1687 | 25.336 |
| mars1-hurtgen | 0.4037 | 27.930 |
| mars1-hurtgen | 0.8215 | 30.247 |
| mars1-hurtgen | 1.1756 | 31.150 |
| mars1-hurtgen | 1.4698 | 31.351 |
| mars1-masscenter | 0.1692 | 25.406 |
| mars1-masscenter | 0.4008 | 27.994 |
| mars1-masscenter | 0.8157 | 30.325 |
| mars1-masscenter | 1.1701 | 31.256 |
| mars1-masscenter | 1.4622 | 31.461 |
| mars1-mc-saupe | 0.1883 | 25.028 |
| mars1-mc-saupe | 0.4524 | 27.409 |
| mars1-mc-saupe | 0.8703 | 29.248 |
| mars1-mc-saupe | 1.2080 | 29.889 |
| mars1-mc-saupe | 1.4954 | 30.024 |
| mars1-saupe | 0.1581 | 25.584 |
| mars1-saupe | 0.3743 | 28.270 |
| mars1-saupe | 0.7835 | 30.945 |
| mars1-saupe | 1.1475 | 32.097 |
| mars1-saupe | 1.4438 | 32.375 |
| mars1-saupe-fisher | 0.1646 | 25.429 |
| mars1-saupe-fisher | 0.3950 | 28.114 |
| mars1-saupe-fisher | 0.8054 | 30.674 |
| mars1-saupe-fisher | 1.1606 | 31.710 |
| mars1-saupe-fisher | 1.4566 | 31.957 |
| webp | 0.1086 | 26.509 |
| webp | 0.1653 | 27.919 |
| webp | 0.1869 | 28.328 |
| webp | 0.2209 | 28.921 |
| webp | 0.2600 | 29.523 |
| webp | 0.2963 | 30.033 |
| webp | 0.3529 | 30.761 |
| webp | 0.4070 | 31.399 |
| webp | 0.4918 | 32.306 |
| webp | 0.5987 | 33.408 |
| webp | 0.7218 | 34.557 |
| webp | 0.8740 | 35.795 |
| webp | 1.2163 | 38.201 |

## BD-rate vs `jpeg`

BD-rate is negative when the test curve needs fewer bits. §M3: the interval is part of the number.

| curve | BD-rate % | BD-PSNR dB | PSNR interval (dB) | bpp interval | pts |
|---|---:|---:|---|---|---:|
| avif | -49.50 | +3.568 | 26.40 – 37.04 | 0.0800 – 1.5702 | 12 |
| jpeg2000 | -38.35 | +2.600 | 27.11 – 37.04 | 0.1196 – 1.5702 | 12 |
| jpegxl | -33.19 | +2.454 | 26.84 – 37.04 | 0.1541 – 1.5702 | 16 |
| mars1-fisher | +35.48 | -1.926 | 25.23 – 31.02 | 0.1728 – 1.4762 | 5 |
| mars1-hurtgen | +28.86 | -1.658 | 25.34 – 31.35 | 0.1687 – 1.4698 | 5 |
| mars1-masscenter | +26.90 | -1.559 | 25.41 – 31.46 | 0.1692 – 1.4622 | 5 |
| mars1-mc-saupe | +54.61 | -2.739 | 25.03 – 30.02 | 0.1883 – 1.4954 | 5 |
| mars1-saupe | +13.39 | -0.923 | 25.58 – 32.37 | 0.1581 – 1.4438 | 5 |
| mars1-saupe-fisher | +21.56 | -1.287 | 25.43 – 31.96 | 0.1646 – 1.4566 | 5 |
| webp | -38.90 | +2.663 | 26.51 – 37.04 | 0.1086 – 1.5702 | 13 |

## Per-image BD-rate vs `jpeg` (anchor codecs only)

Computed per image over the PSNR overlap, then summarised across images -- §M3/D7-style: a BD-rate between two *averaged* curves is a different, less honest number. Nothing is dropped silently; exclusions are named (§A7).

| codec | n | mean % | median % | min % | max % | PSNR interval (dB) | excluded |
|---|---:|---:|---:|---:|---:|---|---|
| avif | 24 | -49.59 | -51.11 | -57.20 | -39.36 | 20.78-40.78 | none |
| jpeg2000 | 24 | -35.58 | -36.05 | -40.97 | -28.96 | 20.55-40.78 | none |
| jpegxl | 24 | -33.47 | -33.44 | -39.79 | -26.93 | 21.76-40.78 | none |
| webp | 24 | -41.85 | -41.82 | -50.72 | -31.54 | 21.87-40.18 | none |
