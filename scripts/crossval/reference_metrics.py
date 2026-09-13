#!/usr/bin/env python3
"""Independent reference metrics for Step 1's cross-validation.

Nothing in this file shares code with `mars-core`. That is the whole point: §M1 puts one
metric implementation in the project, and the only way to know it is right is to compare
it against implementations we did not write.

  PSNR     computed here with numpy, and separately with ffmpeg (see crossval.sh).
  SSIM     scikit-image `structural_similarity`, configured to Wang's original:
           gaussian_weights=True, sigma=1.5, use_sample_covariance=False. With
           truncate=3.5 that is an 11x11 window, and skimage crops the 5-pixel border,
           so it is exactly a 'valid' convolution with population covariance.
  MS-SSIM  `sewar.full_ref.msssim`. Note sewar decimates with
           scipy.ndimage.uniform_filter(im, 2) followed by [::2, ::2], which is a
           half-pixel-shifted window, NOT Wang's non-overlapping 2x2 box. mars-bench
           must be run with --ms-ssim-downsample scipy-uniform2 to compare against it.

Usage: reference_metrics.py ORIGINAL DECODED
Both must be 8-bit grayscale PNG or PGM. Prints JSON.
"""
import json
import sys

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from sewar.full_ref import msssim


def load_gray(path):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if a.ndim != 2:
        raise SystemExit(f"{path}: expected a 2-D grayscale image, got shape {a.shape}")
    return a


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a = load_gray(sys.argv[1])
    b = load_gray(sys.argv[2])
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")

    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    mse = float(np.mean((af - bf) ** 2))
    psnr = None if mse == 0.0 else float(10.0 * np.log10(255.0**2 / mse))

    ssim = float(
        structural_similarity(
            a,
            b,
            data_range=255,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )

    ms = msssim(a, b, MAX=255)
    # sewar returns a complex type from its power helper; the imaginary part is zero for
    # well-behaved input and a non-zero one means the cs terms went negative.
    ms = complex(ms)
    if abs(ms.imag) > 1e-12:
        raise SystemExit(f"sewar msssim returned a complex value: {ms}")

    print(
        json.dumps(
            {
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim,
                "ms_ssim_scipy_phase": float(ms.real),
                "impl": {
                    "psnr": "numpy",
                    "ssim": "skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5, use_sample_covariance=False)",
                    "ms_ssim": "sewar.full_ref.msssim (scipy uniform_filter(2) decimation)",
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
