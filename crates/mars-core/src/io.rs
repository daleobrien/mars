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
    #[error("{path}: unsupported format; expected .pgm, .ppm, .png or .raw/.y/.gray")]
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
        "ppm" => read_ppm(path),
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

/// Write an image as a binary PNM: P5 (grayscale) for a one-plane image, P6 (colour) for
/// a three-plane RGB image, interleaved.
///
/// This exists for feeding image data to external tools that must see raw sample values
/// with **no colour-management metadata** — no `gAMA`, no ICC profile, nothing a tool's
/// PNG reader might "correct" against. `mars-bench`'s JPEG 2000 anchor driver uses this
/// because OpenJPEG's PNG reader honours an input's `gAMA` chunk on read but its PNG
/// writer does not restore it on write, so a plain PNG round trip through
/// `opj_compress`/`opj_decompress` silently darkens every pixel (roughly squares the
/// normalised sample value) even losslessly. PNM carries no such chunk, so there is
/// nothing for the reader to misinterpret. See `docs/decisions.md` D18.
pub fn write_pnm(path: &Path, image: &crate::image::Image) -> Result<(), ImageError> {
    let (w, h) = (image.width(), image.height());
    let planes = image.planes();
    let mut out = Vec::with_capacity(planes.len() * w * h + 32);
    match planes.len() {
        1 => {
            out.extend_from_slice(format!("P5\n{w} {h}\n255\n").as_bytes());
            out.extend_from_slice(planes[0].as_slice());
        }
        3 => {
            out.extend_from_slice(format!("P6\n{w} {h}\n255\n").as_bytes());
            let (r, g, b) = (
                planes[0].as_slice(),
                planes[1].as_slice(),
                planes[2].as_slice(),
            );
            for i in 0..w * h {
                out.push(r[i]);
                out.push(g[i]);
                out.push(b[i]);
            }
        }
        n => {
            return Err(malformed(
                path,
                format!("cannot write a PNM with {n} planes; expected 1 or 3"),
            ))
        }
    }
    fs::write(path, out).map_err(|source| ImageError::Io {
        path: path.display().to_string(),
        source,
    })
}

/// Write an image as PNG: 8-bit gray for a one-plane image, 8-bit RGB for a three-plane
/// one. For side-by-side visual inspection (e.g. `decmars` output), where a viewable
/// format matters more than the colour-management neutrality `write_pnm` is for.
pub fn write_png(path: &Path, image: &crate::image::Image) -> Result<(), ImageError> {
    let (w, h) = (image.width() as u32, image.height() as u32);
    let planes = image.planes();
    let color = match planes.len() {
        1 => image::ColorType::L8,
        3 => image::ColorType::Rgb8,
        n => {
            return Err(malformed(
                path,
                format!("cannot write a PNG with {n} planes; expected 1 or 3"),
            ))
        }
    };
    let buf = match planes.len() {
        1 => planes[0].as_slice().to_vec(),
        _ => {
            let (r, g, b) = (
                planes[0].as_slice(),
                planes[1].as_slice(),
                planes[2].as_slice(),
            );
            let mut out = Vec::with_capacity(3 * w as usize * h as usize);
            for i in 0..w as usize * h as usize {
                out.push(r[i]);
                out.push(g[i]);
                out.push(b[i]);
            }
            out
        }
    };
    image::save_buffer(path, &buf, w, h, color).map_err(|source| ImageError::Png {
        path: path.display().to_string(),
        source,
    })
}

/// Read a binary (P6) colour PPM.
///
/// Written for the same reason `write_pnm` exists: reading back what an external tool
/// wrote as PNM/PPM, with no gamma or colour-management reinterpretation possible.
pub fn read_ppm(path: &Path) -> Result<Image, ImageError> {
    let bytes = fs::read(path).map_err(|source| ImageError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut cur = Cursor::new(&bytes);
    let magic = cur.token().ok_or_else(|| malformed(path, "empty file"))?;
    if magic.as_slice() != b"P6" {
        return Err(malformed(
            path,
            format!(
                "expected magic P6, got {:?}",
                String::from_utf8_lossy(&magic)
            ),
        ));
    }
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
        return Err(malformed(
            path,
            format!("maxval must be 255 for 8-bit metrics, got {maxval}"),
        ));
    }
    cur.skip_single_whitespace()
        .ok_or_else(|| malformed(path, "missing whitespace after maxval"))?;
    let want = width * height * 3;
    let rest = cur.rest();
    if rest.len() < want {
        return Err(malformed(
            path,
            format!("P6 raster is {} bytes, need {want}", rest.len()),
        ));
    }
    let raster = &rest[..want];
    let mut r = Vec::with_capacity(width * height);
    let mut g = Vec::with_capacity(width * height);
    let mut b = Vec::with_capacity(width * height);
    for px in raster.chunks_exact(3) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
    }
    Ok(Image::rgb(
        Plane::from_vec(width, height, r),
        Plane::from_vec(width, height, g),
        Plane::from_vec(width, height, b),
    ))
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
