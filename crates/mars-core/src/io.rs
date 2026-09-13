//! Image readers: PGM (P2 and P5), headerless raw, and PNG.
//!
//! Raw has no header, so its dimensions must be supplied by the caller. Mars 1 takes
//! them on the command line (`-w`/`-h`); we require them explicitly rather than guessing,
//! because a silently transposed or mis-sized raw read produces a plausible-looking
//! image and a completely wrong PSNR.

use std::fs;
use std::path::Path;

use crate::image::{Image, Plane};

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("io error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: {reason}")]
    Malformed { path: String, reason: String },
    #[error("{path}: unsupported format; expected .pgm, .png or .raw/.y/.gray")]
    UnknownFormat { path: String },
    #[error("{path}: raw images have no header; dimensions must be given explicitly")]
    RawDimensionsRequired { path: String },
    #[error("decoding {path}: {source}")]
    Png {
        path: String,
        #[source]
        source: image::ImageError,
    },
}

fn malformed(path: &Path, reason: impl Into<String>) -> ImageError {
    ImageError::Malformed {
        path: path.display().to_string(),
        reason: reason.into(),
    }
}

/// Read an image, dispatching on extension.
///
/// `raw_dims` is required for headerless raw input and ignored otherwise.
pub fn read_image(path: &Path, raw_dims: Option<(usize, usize)>) -> Result<Image, ImageError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "pgm" => read_pgm(path).map(Image::gray),
        "png" => read_png(path),
        "raw" | "y" | "gray" => {
            let (w, h) = raw_dims.ok_or_else(|| ImageError::RawDimensionsRequired {
                path: path.display().to_string(),
            })?;
            read_raw(path, w, h).map(Image::gray)
        }
        _ => Err(ImageError::UnknownFormat {
            path: path.display().to_string(),
        }),
    }
}

/// Read a headerless 8-bit grayscale raw plane of known dimensions.
pub fn read_raw(path: &Path, width: usize, height: usize) -> Result<Plane, ImageError> {
    let bytes = fs::read(path).map_err(|source| ImageError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let want = width * height;
    if bytes.len() != want {
        return Err(malformed(
            path,
            format!(
                "raw file is {} bytes but {width}x{height} needs exactly {want}",
                bytes.len()
            ),
        ));
    }
    Ok(Plane::from_vec(width, height, bytes))
}

/// Read a PNG (8-bit gray or RGB; 16-bit and palette inputs are rejected rather than
/// silently reduced, because a quiet bit-depth change is a quiet metric change).
pub fn read_png(path: &Path) -> Result<Image, ImageError> {
    let decoded = image::open(path).map_err(|source| ImageError::Png {
        path: path.display().to_string(),
        source,
    })?;
    let (w, h) = (decoded.width() as usize, decoded.height() as usize);
    match decoded {
        image::DynamicImage::ImageLuma8(buf) => {
            Ok(Image::gray(Plane::from_vec(w, h, buf.into_raw())))
        }
        image::DynamicImage::ImageRgb8(buf) => {
            let raw = buf.into_raw();
            let mut r = Vec::with_capacity(w * h);
            let mut g = Vec::with_capacity(w * h);
            let mut b = Vec::with_capacity(w * h);
            for px in raw.chunks_exact(3) {
                r.push(px[0]);
                g.push(px[1]);
                b.push(px[2]);
            }
            Ok(Image::rgb(
                Plane::from_vec(w, h, r),
                Plane::from_vec(w, h, g),
                Plane::from_vec(w, h, b),
            ))
        }
        other => Err(malformed(
            path,
            format!(
                "expected 8-bit gray or 8-bit RGB PNG, got {:?}",
                other.color()
            ),
        )),
    }
}

/// Read a binary (P5) or ASCII (P2) PGM.
///
/// Mars 1's own reader handles P5 only; P2 is here because hand-written known-answer
/// fixtures are far easier to review as text.
pub fn read_pgm(path: &Path) -> Result<Plane, ImageError> {
    let bytes = fs::read(path).map_err(|source| ImageError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut cur = Cursor::new(&bytes);

    let magic = cur.token().ok_or_else(|| malformed(path, "empty file"))?;
    let binary = match magic.as_slice() {
        b"P5" => true,
        b"P2" => false,
        other => {
            return Err(malformed(
                path,
                format!(
                    "expected magic P2 or P5, got {:?}",
                    String::from_utf8_lossy(other)
                ),
            ))
        }
    };

    let width = cur.uint().ok_or_else(|| malformed(path, "missing width"))?;
    let height = cur
        .uint()
        .ok_or_else(|| malformed(path, "missing height"))?;
    let maxval = cur
        .uint()
        .ok_or_else(|| malformed(path, "missing maxval"))?;
    if width == 0 || height == 0 {
        return Err(malformed(path, "zero-sized image"));
    }
    if maxval != 255 {
        // Rescaling here would silently change every metric downstream, so refuse.
        return Err(malformed(
            path,
            format!("maxval must be 255 for 8-bit metrics, got {maxval}"),
        ));
    }

    let want = width * height;
    let data = if binary {
        // Exactly one whitespace byte separates the header from the raster.
        cur.skip_single_whitespace()
            .ok_or_else(|| malformed(path, "missing whitespace after maxval"))?;
        let rest = cur.rest();
        if rest.len() < want {
            return Err(malformed(
                path,
                format!("P5 raster is {} bytes, need {want}", rest.len()),
            ));
        }
        rest[..want].to_vec()
    } else {
        let mut data = Vec::with_capacity(want);
        for i in 0..want {
            let v = cur.uint().ok_or_else(|| {
                malformed(path, format!("P2 raster ended after {i} of {want} samples"))
            })?;
            if v > 255 {
                return Err(malformed(path, format!("P2 sample {v} exceeds maxval 255")));
            }
            data.push(v as u8);
        }
        data
    };

    Ok(Plane::from_vec(width, height, data))
}

/// A byte cursor with just enough PNM lexing: `#` starts a comment that runs to
/// end-of-line, and any whitespace run separates tokens.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn skip_space_and_comments(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b'#' => {
                    while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                }
                c if c.is_ascii_whitespace() => self.pos += 1,
                _ => break,
            }
        }
    }

    fn token(&mut self) -> Option<Vec<u8>> {
        self.skip_space_and_comments();
        let start = self.pos;
        while self.pos < self.bytes.len()
            && !self.bytes[self.pos].is_ascii_whitespace()
            && self.bytes[self.pos] != b'#'
        {
            self.pos += 1;
        }
        (self.pos > start).then(|| self.bytes[start..self.pos].to_vec())
    }

    fn uint(&mut self) -> Option<usize> {
        let tok = self.token()?;
        if tok.is_empty() || !tok.iter().all(|c| c.is_ascii_digit()) {
            return None;
        }
        std::str::from_utf8(&tok).ok()?.parse().ok()
    }

    fn skip_single_whitespace(&mut self) -> Option<()> {
        let c = *self.bytes.get(self.pos)?;
        c.is_ascii_whitespace().then(|| self.pos += 1)
    }

    fn rest(&self) -> &'a [u8] {
        &self.bytes[self.pos..]
    }
}
