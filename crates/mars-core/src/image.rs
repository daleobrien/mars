//! 8-bit image types.
//!
//! Mars 1 is a grayscale codec; Mars 2's container must carry colour (Step 10). The
//! types here therefore model "one to three 8-bit planes of identical size", which
//! covers both without committing to a colour pipeline yet.

/// How to interpret a multi-plane image's channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// One plane, luma.
    Gray,
    /// Three planes, non-subsampled R, G, B.
    Rgb,
}

/// A single 8-bit plane in row-major order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plane {
    width: usize,
    height: usize,
    data: Vec<u8>,
}

impl Plane {
    /// # Panics
    /// If `data.len() != width * height`.
    pub fn from_vec(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(
            data.len(),
            width * height,
            "plane data length {} does not match {width}x{height}",
            data.len()
        );
        Self {
            width,
            height,
            data,
        }
    }

    pub fn filled(width: usize, height: usize, value: u8) -> Self {
        Self {
            width,
            height,
            data: vec![value; width * height],
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    #[inline]
    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.width..(y + 1) * self.width]
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.width + x]
    }

    /// The top-left `w x h` sub-plane.
    ///
    /// This is the operation that keeps §M2's "original W×H region only" rule honest:
    /// Mars 1 pads to the next power of two, and the padding must never be measured.
    ///
    /// # Panics
    /// If the requested region does not fit.
    pub fn crop_top_left(&self, w: usize, h: usize) -> Plane {
        assert!(
            w <= self.width && h <= self.height,
            "crop {w}x{h} does not fit in {}x{}",
            self.width,
            self.height
        );
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            out.extend_from_slice(&self.row(y)[..w]);
        }
        Plane::from_vec(w, h, out)
    }

    /// Plane values as `f64`, which is what every metric consumes.
    pub fn to_f64(&self) -> Vec<f64> {
        self.data.iter().map(|&v| f64::from(v)).collect()
    }
}

/// One to three planes of identical dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    color: ColorSpace,
    planes: Vec<Plane>,
}

impl Image {
    pub fn gray(plane: Plane) -> Self {
        Self {
            color: ColorSpace::Gray,
            planes: vec![plane],
        }
    }

    /// # Panics
    /// If the three planes do not share dimensions.
    pub fn rgb(r: Plane, g: Plane, b: Plane) -> Self {
        assert!(
            r.width() == g.width()
                && g.width() == b.width()
                && r.height() == g.height()
                && g.height() == b.height(),
            "RGB planes must share dimensions"
        );
        Self {
            color: ColorSpace::Rgb,
            planes: vec![r, g, b],
        }
    }

    #[inline]
    pub fn color(&self) -> ColorSpace {
        self.color
    }

    #[inline]
    pub fn planes(&self) -> &[Plane] {
        &self.planes
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.planes[0].width()
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.planes[0].height()
    }

    #[inline]
    pub fn pixels(&self) -> usize {
        self.width() * self.height()
    }

    /// The luma plane.
    ///
    /// For `Gray` this is the plane itself, with no conversion applied at all — the
    /// common case must not pick up rounding error from a colour transform it does not
    /// need. For `Rgb` this is BT.601 Y, the conversion JPEG and every 8-bit still
    /// codec in the anchor set uses, rounded half-away-from-zero and clamped.
    pub fn luma(&self) -> Plane {
        match self.color {
            ColorSpace::Gray => self.planes[0].clone(),
            ColorSpace::Rgb => {
                let (r, g, b) = (&self.planes[0], &self.planes[1], &self.planes[2]);
                let data = r
                    .as_slice()
                    .iter()
                    .zip(g.as_slice())
                    .zip(b.as_slice())
                    .map(|((&r, &g), &b)| {
                        let y = 0.299 * f64::from(r) + 0.587 * f64::from(g) + 0.114 * f64::from(b);
                        y.round().clamp(0.0, 255.0) as u8
                    })
                    .collect();
                Plane::from_vec(r.width(), r.height(), data)
            }
        }
    }

    pub fn crop_top_left(&self, w: usize, h: usize) -> Image {
        Image {
            color: self.color,
            planes: self.planes.iter().map(|p| p.crop_top_left(w, h)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_takes_the_top_left_region() {
        let p = Plane::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(p.crop_top_left(2, 2).as_slice(), &[1, 2, 4, 5]);
        assert_eq!(p.crop_top_left(3, 2), p);
    }

    #[test]
    fn gray_luma_is_the_identity() {
        // A colour conversion applied to an already-grayscale plane is a silent source
        // of +-1 LSB error; assert it does not happen.
        let p = Plane::from_vec(2, 2, vec![0, 1, 254, 255]);
        assert_eq!(Image::gray(p.clone()).luma(), p);
    }

    #[test]
    fn rgb_luma_is_bt601() {
        let img = Image::rgb(
            Plane::filled(1, 1, 255),
            Plane::filled(1, 1, 0),
            Plane::filled(1, 1, 0),
        );
        // 0.299 * 255 = 76.245 -> 76
        assert_eq!(img.luma().as_slice(), &[76]);
    }
}
