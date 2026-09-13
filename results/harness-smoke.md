# Harness smoke test — Mars 1 on Lena

> release @ b94d62441385-dirty · harness 0.1.0 · Apple M3 Pro (6 P-cores / 6 E-cores) · 2026-09-13T02:44:22Z

## Operating points

| curve | bpp | PSNR (dB) |
|---|---:|---:|
| Mars 1 · MassCenter | 0.1205 | 27.049 |
| Mars 1 · MassCenter | 0.2333 | 29.932 |
| Mars 1 · MassCenter | 0.4826 | 33.459 |
| Mars 1 · MassCenter | 0.8761 | 35.558 |
| Mars 1 · MassCenter | 1.5788 | 36.405 |
| Mars 1 · Fisher | 0.1235 | 26.882 |
| Mars 1 · Fisher | 0.2477 | 29.817 |
| Mars 1 · Fisher | 0.5037 | 33.276 |
| Mars 1 · Fisher | 0.8946 | 35.176 |
| Mars 1 · Fisher | 1.5968 | 35.913 |

## BD-rate vs `Mars 1 · MassCenter`

BD-rate is negative when the test curve needs fewer bits. §M3: the interval is part of the number.

| curve | BD-rate % | BD-PSNR dB | PSNR interval (dB) | bpp interval | pts |
|---|---:|---:|---|---|---:|
| Mars 1 · Fisher | +10.75 | -0.395 | 27.05 – 35.91 | 0.1205 – 1.5968 | 5 |
