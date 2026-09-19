//! Image readers: PGM (P2 and P5), headerless raw, PNG and JPEG; writers for PNG, JPEG
//! and PNM.
//!
//! Raw has no header, so its dimensions must be supplied by the caller. Mars 1 takes
//! them on the command line (`-w`/`-h`); we require them explicitly rather than guessing,
//! because a silently transposed or mis-sized raw read produces a plausible-looking
//! image and a completely wrong PSNR.

use std::fs;
use std::io::{Read, Write};
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
    #[error("{path}: unsupported format; expected .pgm, .ppm, .png, .jpg/.jpeg or .raw/.y/.gray")]
    UnknownFormat { path: String },
    #[error("{path}: raw images have no header; dimensions must be given explicitly")]
    RawDimensionsRequired { path: String },
    #[error("decoding {path}: {source}")]
    Decode {
        path: String,
        #[source]
        source: image::ImageError,
    },
    #[error("encoding {path}: {source}")]
    Encode {
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

/// Read an image, dispatching on the file's contents and, where there is no signature
/// to read, its extension.
///
/// `raw_dims` is required for headerless raw input and ignored otherwise.
pub fn read_image(path: &Path, raw_dims: Option<(usize, usize)>) -> Result<Image, ImageError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    // Headerless raw has no signature at all and needs its name to be read; PNM's magic
    // is plain text that arbitrary bytes could imitate, so neither is autodetected.
    match ext.as_str() {
        "pgm" => return read_pgm(path).map(Image::gray),
        "ppm" => return read_ppm(path),
        "raw" | "y" | "gray" => {
            let (w, h) = raw_dims.ok_or_else(|| ImageError::RawDimensionsRequired {
                path: path.display().to_string(),
            })?;
            return read_raw(path, w, h).map(Image::gray);
        }
        _ => {}
    }
    // PNG and JPEG carry unambiguous binary signatures, so those are detected from the
    // file's own leading bytes: a mis-named or extension-less file still reads as what
    // it is, and an unrecognised blob is refused without guessing.
    if sniff_encoded(path)? {
        read_encoded(path)
    } else {
        Err(ImageError::UnknownFormat {
            path: path.display().to_string(),
        })
    }
}

/// Does `path` begin with a PNG or JPEG signature?
///
/// `read_image` uses this to autodetect those two formats from content rather than name.
/// The check is deliberately limited to their unambiguous binary signatures: PNM's magic
/// is text that arbitrary bytes could imitate, and headerless raw has none at all.
fn sniff_encoded(path: &Path) -> Result<bool, ImageError> {
    const PNG: &[u8] = b"\x89PNG\r\n\x1a\n";
    const JPEG: &[u8] = &[0xFF, 0xD8, 0xFF];
    let mut file = fs::File::open(path).map_err(|source| ImageError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut head = [0u8; 8];
    let mut filled = 0;
    while filled < head.len() {
        match file.read(&mut head[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            // A signal can interrupt a read before any bytes arrive; retry rather than
            // treating it as end-of-file.
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(source) => {
                return Err(ImageError::Io {
                    path: path.display().to_string(),
                    source,
                })
            }
        }
    }
    let head = &head[..filled];
    Ok(head.starts_with(PNG) || head.starts_with(JPEG))
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

/// Read a PNG or JPEG with the `image` crate, guessing the decoder from the file's own
/// signature and only falling back to the extension. 8-bit gray and 8-bit RGB are
/// accepted; 16-bit and palette inputs are rejected rather than silently reduced, because
/// a quiet bit-depth or colour change is a quiet metric change.
pub fn read_encoded(path: &Path) -> Result<Image, ImageError> {
    let io_error = |source: std::io::Error| ImageError::Io {
        path: path.display().to_string(),
        source,
    };
    let decoded = image::ImageReader::open(path)
        .map_err(io_error)?
        .with_guessed_format()
        .map_err(io_error)?
        .decode()
        .map_err(|source| ImageError::Decode {
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
                "expected 8-bit gray or 8-bit RGB image, got {:?}",
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

/// Flatten `image` into an interleaved 8-bit buffer, with its geometry and colour type:
/// L8 for a one-plane image, Rgb8 for a three-plane one. Any other plane count is refused
/// rather than guessed at, since every writer here expects 1 or 3 planes.
fn image_payload(
    path: &Path,
    image: &crate::image::Image,
) -> Result<(Vec<u8>, u32, u32, image::ColorType), ImageError> {
    let (w, h) = (image.width() as u32, image.height() as u32);
    let planes = image.planes();
    let color = match planes.len() {
        1 => image::ColorType::L8,
        3 => image::ColorType::Rgb8,
        n => {
            return Err(malformed(
                path,
                format!("cannot write an image with {n} planes; expected 1 or 3"),
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
    Ok((buf, w, h, color))
}

/// Write an image with the `image` crate's encoder for `codec` (PNG, or JPEG at the
/// crate's default quality). The format comes from the argument, not the path's
/// extension, so the chosen function is what the file actually is whatever it is named.
fn write_encoded(
    path: &Path,
    image: &crate::image::Image,
    codec: image::ImageFormat,
) -> Result<(), ImageError> {
    let (buf, w, h, color) = image_payload(path, image)?;
    image::save_buffer_with_format(path, &buf, w, h, color, codec).map_err(|source| {
        ImageError::Encode {
            path: path.display().to_string(),
            source,
        }
    })
}

/// Write an image as PNG: 8-bit gray for a one-plane image, 8-bit RGB for a three-plane
/// one. For side-by-side visual inspection (e.g. `decmars` output), where a viewable
/// format matters more than the colour-management neutrality `write_pnm` is for.
pub fn write_png(path: &Path, image: &crate::image::Image) -> Result<(), ImageError> {
    write_encoded(path, image, image::ImageFormat::Png)
}

/// Write an image as JPEG: 8-bit gray for a one-plane image, 8-bit RGB for a three-plane
/// one, at the `image` crate's default quality (75).
///
/// Unlike PNG this is **lossy**, so it is for visual inspection only and must not be fed
/// back into a metric: the stored pixels differ from the decoder's output by the
/// encoder's quantisation. The format comes from this function, not `path`'s extension,
/// so it writes a JPEG whatever the file is named.
pub fn write_jpeg(path: &Path, image: &crate::image::Image) -> Result<(), ImageError> {
    write_encoded(path, image, image::ImageFormat::Jpeg)
}

/// Write an image as JPEG at an explicit `quality` in 1-100, higher being better (the
/// `image` crate's default is 75). Same lossiness caveat as `write_jpeg`: this is for
/// visual inspection, not a metric-preserving round trip.
pub fn write_jpeg_with_quality(
    path: &Path,
    image: &crate::image::Image,
    quality: u8,
) -> Result<(), ImageError> {
    let (buf, w, h, color) = image_payload(path, image)?;
    let file = fs::File::create(path).map_err(|source| ImageError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut writer = std::io::BufWriter::new(file);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut writer, quality)
        .encode(&buf, w, h, color.into())
        .map_err(|source| ImageError::Encode {
            path: path.display().to_string(),
            source,
        })?;
    writer.flush().map_err(|source| ImageError::Io {
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
