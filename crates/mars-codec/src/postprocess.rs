//! Conservative grayscale boundary smoothing, applied only after iterative decoding.
//! These fixed research constants are not claimed to reproduce any paper's filter.

use mars_core::Plane;

const GRADIENT_CUTOFF: i32 = 64;
const WEIGHT_DENOMINATOR: i32 = 2 * GRADIENT_CUTOFF * GRADIENT_CUTOFF;
const LEFT: u8 = 1;
const RIGHT: u8 = 2;
const UP: u8 = 4;
const DOWN: u8 = 8;

/// Return a new plane with edge-aware smoothing on square leaf boundaries.
///
/// `leaves` contains `(row, col, size)` in decoded pixel coordinates. Normally these
/// form a non-overlapping partition; mixed sizes and clipped border leaves are allowed.
/// Empty/outside leaves are ignored, and oversized extents are clipped without overflow.
/// Duplicate leaves/directions have no effect. No boundary means no change.
///
/// Each boundary pixel is processed once, blending toward the mean of its distinct
/// orthogonal cross-boundary neighbors. All samples come from `decoded`, never from
/// partially filtered output. Missing image-border neighbors are omitted; a pixel with
/// no cross-boundary neighbor is unchanged. Interior pixels are untouched.
///
/// The gradient `g` is the maximum absolute unnormalised central difference in x/y
/// (replicated borders), guarded by the largest adjacent difference to retain thin
/// edges where central differences cancel. The weight is
/// `0.5 * max(0, 1 - g/64)^2`: at most one half avoids overshooting the neighbor mean,
/// the quadratic taper conservatively reduces blur, and a 64-level contrast cutoff
/// protects strong edges exactly. These constants are fixed, not tuned per image.
/// Integer rational arithmetic rounds to nearest (ties upward), then clamps to 0..=255.
/// Constant planes are byte-identical, including at corners and image borders.
///
/// This is an out-of-loop, grayscale-only operation: it changes neither the input,
/// encoded bytes, nor the iterative decoder's state. Color/progressive composition
/// belongs to callers and is not implemented here.
pub fn smooth_boundaries(decoded: &Plane, leaves: &[(usize, usize, usize)]) -> Plane {
    let (width, height) = (decoded.width(), decoded.height());
    let mut output = decoded.clone();
    if width == 0 || height == 0 {
        return output;
    }
    let mut directions = vec![0u8; decoded.as_slice().len()];
    for &(row, col, size) in leaves {
        if size == 0 || row >= height || col >= width {
            continue;
        }
        let bottom = row.saturating_add(size).min(height);
        let right = col.saturating_add(size).min(width);
        for y in row..bottom {
            for x in [col, right] {
                if x > 0 && x < width {
                    directions[y * width + x - 1] |= RIGHT;
                    directions[y * width + x] |= LEFT;
                }
            }
        }
        for x in col..right {
            for y in [row, bottom] {
                if y > 0 && y < height {
                    directions[(y - 1) * width + x] |= DOWN;
                    directions[y * width + x] |= UP;
                }
            }
        }
    }

    for (i, mask) in directions.into_iter().enumerate() {
        if mask == 0 {
            continue;
        }
        let (x, y) = (i % width, i / width);
        let p = i32::from(decoded.as_slice()[i]);
        let left = i32::from(decoded.get(x.saturating_sub(1), y));
        let right = i32::from(decoded.get((x + 1).min(width - 1), y));
        let up = i32::from(decoded.get(x, y.saturating_sub(1)));
        let down = i32::from(decoded.get(x, (y + 1).min(height - 1)));
        let mut gradient = (right - left).abs().max((down - up).abs());
        let mut sum = 0;
        let mut count = 0;
        for (direction, sample) in [(LEFT, left), (RIGHT, right), (UP, up), (DOWN, down)] {
            gradient = gradient.max((sample - p).abs());
            if mask & direction != 0 {
                sum += sample;
                count += 1;
            }
        }
        let taper = (GRADIENT_CUTOFF - gradient).max(0);
        let denominator = WEIGHT_DENOMINATOR * count;
        let numerator = p * denominator + (sum - p * count) * taper * taper;
        output.as_mut_slice()[i] =
            ((numerator + denominator / 2) / denominator).clamp(0, 255) as u8;
    }
    output
}
