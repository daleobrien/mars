//! The Mars 1 `.ifs` reader — Step 5.
//!
//! A read-only port of `docs/mars1-format.md` §2–§10.1, written from that document alone
//! (it does not consult `reference/mars1/` or `scripts/validate-ifs.py`). It recovers the
//! transform list and includes a minimal iterative decoder for sanity-checking the parse
//! against `decmars -i`. There is **no writer, no pyramidal decode, and no zoom** — Step 6
//! adds a writer for its own cross-checking, and anything else shells out to `decmars`.

use mars_core::Plane;

/// The 60-bit header (§3), plus the parameters derived from it (§4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub bits_alfa: u32,
    pub bits_beta: u32,
    pub min_size: u32,
    pub max_size: u32,
    pub shift: u32,
    pub width: u32,
    pub height: u32,
    pub int_max_alfa: u32,
}

impl Header {
    /// §4.1. Exact in binary64 for every `int_max_alfa` (it is a dyadic rational).
    pub fn max_alfa(&self) -> f64 {
        f64::from(self.int_max_alfa) / 256.0 * 8.0
    }

    /// §4.2. The quadtree is walked over this square, not the image's own dimensions.
    pub fn virtual_size(&self) -> u32 {
        1 << ceil_log2(self.width.max(self.height))
    }

    /// §4.3 `bits_per_coordinate_h`. Integer division before the log, per the hazard
    /// documented there — this is not the same value the C's binary64 formula gives for
    /// `dim == 0`, but that case is unrepresentable and rejected in `parse`.
    pub fn bits_coord_row(&self) -> u32 {
        ceil_log2(self.height / self.shift)
    }

    /// §4.3 `bits_per_coordinate_w`.
    pub fn bits_coord_col(&self) -> u32 {
        ceil_log2(self.width / self.shift)
    }
}

/// §4.3: exact integer `ceil(log2(q))`, i.e. `(q - 1).bit_length()`, with `q <= 1` giving 0.
fn ceil_log2(q: u32) -> u32 {
    if q <= 1 {
        0
    } else {
        32 - (q - 1).leading_zeros()
    }
}

/// One leaf of the quadtree (§5–§6): a range block, its quantised fit, and — unless it is
/// DC-only — the domain block and isometry it references.
///
/// Step 15 (R&D plan §4): `mode` distinguishes the five candidate representations `J`
/// competes among. Modes 0/2 are exactly this project's original Mars 1-derived
/// constant/domain-reference leaves (`qalfa == 0` / `qalfa != 0`, unchanged); modes 1
/// (`qgx`/`qgy`, a spatial-gradient plane fit needing no domain search at all) and 3
/// (mode 2's fields plus `residual`, the quantised DCT coefficients of the fractal
/// prediction's error) are new. `mode` is only ever non-0/2 on leaves `mars_codec::encode`
/// produces with `EncodeParams::lambda` set (Step 14's bottom-up RD walk) -- the legacy
/// `t_rms`-threshold `walk` and the raw `.ifs` writer/reader below never emit or expect
/// modes 1/3, since Mars 1's `.ifs` format has no representation for either (`crate::ifs`
/// stays a read/write port of the 1998 format; `crate::mars_format` is where Step 15's new
/// modes actually round-trip -- see that module's doc). `residual` is empty on every leaf
/// except mode 3, where it holds exactly `size*size` levels in [`crate::dct`]'s row-major
/// order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Leaf {
    pub row: u32,
    pub col: u32,
    pub size: u32,
    pub mode: u8,
    pub qalfa: u32,
    pub qbeta: u32,
    /// Meaningless when `dc_only()` — always 0 in that case, per §6.
    pub isometry: u8,
    pub dom_row: u32,
    pub dom_col: u32,
    /// Mode 1 only: quantised spatial-gradient coefficients (§ the struct doc).
    pub qgx: i32,
    pub qgy: i32,
    /// Mode 3 only: `size*size` quantised DCT coefficient levels, row-major, DC first.
    pub residual: Vec<i32>,
}

impl Leaf {
    /// §6: `qalfa == 0` means a constant-fill block with no domain reference at all.
    /// Step 15: true for mode 0 (flat) and mode 1 (affine) leaves, both of which carry no
    /// domain reference — kept as a `qalfa`-based check rather than `mode == 0` so every
    /// pre-Step-15 caller (which only ever produces mode 0/2 leaves) is unaffected.
    pub fn dc_only(&self) -> bool {
        self.qalfa == 0
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IfsError {
    #[error("degenerate header: width, height and shift must all be nonzero")]
    DegenerateHeader,
    #[error("bitstream ended while reading the tree")]
    Truncated,
    #[error(
        "leaf at ({row},{col},{size}) references a domain at ({dom_row},{dom_col}) \
         that runs past the image"
    )]
    DomainOutOfBounds {
        row: u32,
        col: u32,
        size: u32,
        dom_row: u32,
        dom_col: u32,
    },
    #[error("{0} bits left after the tree; expected 0-7 of padding (§2)")]
    TrailingBits(usize),
    #[error("padding bits are not zero (§2)")]
    NonZeroPadding,
    #[error("no leaf at ({row},{col},{size}) and none can follow (size <= min_size)")]
    MissingLeaf { row: u32, col: u32, size: u32 },
    #[error("two leaves claim the same position ({row},{col},{size})")]
    DuplicateLeaf { row: u32, col: u32, size: u32 },
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// §2: bits leave each byte most-significant-bit first.
    fn bit(&mut self) -> Result<u32, IfsError> {
        let byte = *self.data.get(self.pos >> 3).ok_or(IfsError::Truncated)?;
        let b = (byte >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(u32::from(b))
    }

    /// §2: each value is assembled least-significant-bit first.
    fn read(&mut self, n: u32) -> Result<u32, IfsError> {
        let mut v = 0u32;
        for i in 0..n {
            v |= self.bit()? << i;
        }
        Ok(v)
    }
}

/// Parse a `.ifs` bitstream per §2–§6, returning the header and the leaves in the order
/// `read_transformations` produces them (§5.1: NW, SW, NE, SE — not Z-order).
pub fn parse(data: &[u8]) -> Result<(Header, Vec<Leaf>), IfsError> {
    let mut r = BitReader::new(data);
    let hdr = Header {
        bits_alfa: r.read(4)?,
        bits_beta: r.read(4)?,
        min_size: r.read(7)?,
        max_size: r.read(7)?,
        shift: r.read(6)?,
        width: r.read(12)?,
        height: r.read(12)?,
        int_max_alfa: r.read(8)?,
    };
    if hdr.width == 0 || hdr.height == 0 || hdr.shift == 0 {
        return Err(IfsError::DegenerateHeader);
    }

    let mut leaves = Vec::new();
    walk(&mut r, &hdr, 0, 0, hdr.virtual_size(), &mut leaves)?;

    // §2: 0-7 bits of zero padding, never more (a full padding byte would mean the last
    // real byte was never written).
    let left = (data.len() * 8)
        .checked_sub(r.pos)
        .ok_or(IfsError::Truncated)?;
    if left >= 8 {
        return Err(IfsError::TrailingBits(left));
    }
    if left > 0 && data[data.len() - 1] & ((1 << left) - 1) != 0 {
        return Err(IfsError::NonZeroPadding);
    }
    Ok((hdr, leaves))
}

/// §5: `read_transformations(row, col, size)`.
fn walk(
    r: &mut BitReader,
    hdr: &Header,
    row: u32,
    col: u32,
    size: u32,
    leaves: &mut Vec<Leaf>,
) -> Result<(), IfsError> {
    if row >= hdr.height || col >= hdr.width {
        return Ok(()); // §5.1: wholly outside the image, reads nothing
    }

    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    if forced || (size > hdr.min_size && r.read(1)? == 1) {
        let half = size / 2;
        walk(r, hdr, row, col, half, leaves)?; // §5.1: NW, SW, NE, SE
        walk(r, hdr, row + half, col, half, leaves)?;
        walk(r, hdr, row, col + half, half, leaves)?;
        walk(r, hdr, row + half, col + half, half, leaves)?;
        return Ok(());
    }

    let qalfa = r.read(hdr.bits_alfa)?;
    let qbeta = r.read(hdr.bits_beta)?;
    let (isometry, dom_row, dom_col) = if qalfa != 0 {
        let isometry = r.read(3)? as u8;
        let dom_row = hdr.shift * r.read(hdr.bits_coord_row())?;
        let dom_col = hdr.shift * r.read(hdr.bits_coord_col())?;
        if dom_row + 2 * size > hdr.height || dom_col + 2 * size > hdr.width {
            return Err(IfsError::DomainOutOfBounds {
                row,
                col,
                size,
                dom_row,
                dom_col,
            });
        }
        (isometry, dom_row, dom_col)
    } else {
        (0, 0, 0) // §6: zeroalfa — not present in the stream
    };
    leaves.push(Leaf {
        row,
        col,
        size,
        mode: if qalfa == 0 { 0 } else { 2 },
        qalfa,
        qbeta,
        isometry,
        dom_row,
        dom_col,
        qgx: 0,
        qgy: 0,
        residual: Vec::new(),
    });
    Ok(())
}

/// §10.1: fixed-point iteration over the transform list, starting from a flat grey image.
/// This exists to sanity-check `parse`, not to replace `decmars` — no pyramidal mode, no
/// postprocessing, no attempt at speed.
pub fn decode_iterative(hdr: &Header, leaves: &[Leaf], iterations: u32) -> Plane {
    let (w, h) = (hdr.width as usize, hdr.height as usize);
    let mut img = vec![128u8; w * h];
    for _ in 0..iterations {
        let mut next = vec![0u8; w * h];
        for leaf in leaves {
            decode_leaf(hdr, leaf, &img, w, &mut next);
        }
        img = next;
    }
    Plane::from_vec(w, h, img)
}

/// Same fixed-point iteration as [`decode_iterative`], but stops as soon as the image has
/// stabilised instead of running a caller-chosen fixed count. The IFS map is contractive
/// (that is what guarantees the fixed point exists at all — §10.1), so the per-pixel delta
/// between successive iterations decreases monotonically in the limit; stopping once the
/// worst-case pixel moves by at most `threshold` gives a decode that is visually converged
/// without spending iterations past the point of diminishing returns.
///
/// Returns the decoded plane and the number of iterations actually run, so callers can
/// report how quickly convergence was reached. Runs at most `max_iterations` — a required
/// cap, since a `threshold` of 0 combined with 8-bit rounding can cycle between two states
/// forever rather than settling on one.
pub fn decode_until_stable(
    hdr: &Header,
    leaves: &[Leaf],
    threshold: u8,
    max_iterations: u32,
) -> (Plane, u32) {
    let (w, h) = (hdr.width as usize, hdr.height as usize);
    let mut img = vec![128u8; w * h];
    let mut used = 0;
    for i in 0..max_iterations.max(1) {
        let mut next = vec![0u8; w * h];
        for leaf in leaves {
            decode_leaf(hdr, leaf, &img, w, &mut next);
        }
        let max_delta = img
            .iter()
            .zip(next.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        img = next;
        used = i + 1;
        if max_delta <= threshold {
            break;
        }
    }
    (Plane::from_vec(w, h, img), used)
}

/// §7 dequantisation and §9/§10.1 reconstruction for one leaf, reading `img` (the
/// previous iteration) and writing `next` — the double-buffering of §10.1 is what makes
/// leaves independent of each other and safe to iterate in any order.
///
/// Step 15: dispatches on `leaf.mode`. Modes 0/2 are exactly the original §9/§10.1
/// formula (mode 0 is simply mode 2 with `alfa == 0`, so both fall through the same
/// arithmetic below unchanged); mode 1 reconstructs the spatial-gradient plane instead of
/// reading a domain at all; mode 3 adds mode 2's fractal prediction to the residual's
/// inverse DCT. See [`crate::mars_format`] for where modes 1/3 are actually produced and
/// consumed -- this function exists so [`decode_iterative`] (and hence any PSNR
/// measurement built on it, e.g. `mars-bench`'s RD sampler) reconstructs every mode
/// correctly, not only the two the legacy `.ifs` bitstream itself can express.
fn decode_leaf(hdr: &Header, leaf: &Leaf, img: &[u8], stride: usize, next: &mut [u8]) {
    let size = leaf.size as usize;

    if leaf.mode == 1 {
        // Mode 1 -- affine: `b0 + gx*u + gy*v`, no domain reference. `qgx`/`qgy` are
        // already real-valued fixed-point gradients scaled by `crate::encode`'s
        // `AFFINE_GRAD_SCALE` -- see that constant's doc for the quantisation.
        let b0 = f64::from(leaf.qbeta) / f64::from((1u32 << hdr.bits_beta) - 1) * 255.0;
        let gx = f64::from(leaf.qgx) / crate::encode::AFFINE_GRAD_SCALE;
        let gy = f64::from(leaf.qgy) / crate::encode::AFFINE_GRAD_SCALE;
        for u in 0..size {
            for v in 0..size {
                let value = (b0 + gx * u as f64 + gy * v as f64).round().clamp(0.0, 255.0) as u8;
                next[(leaf.row as usize + u) * stride + leaf.col as usize + v] = value;
            }
        }
        return;
    }

    let alfa = f64::from(leaf.qalfa) / f64::from(1u32 << hdr.bits_alfa) * hdr.max_alfa();
    let mut beta = f64::from(leaf.qbeta) / f64::from((1u32 << hdr.bits_beta) - 1)
        * ((1.0 + alfa.abs()) * 255.0);
    if alfa > 0.0 {
        beta -= alfa * 255.0;
    }

    let residual = if leaf.mode == 3 && !leaf.residual.is_empty() {
        let levels: Vec<f64> = leaf
            .residual
            .iter()
            .map(|&l| {
                crate::quant::dead_zone_dequantize(
                    l,
                    crate::encode::RESIDUAL_QSTEP_DEFAULT,
                    crate::encode::RESIDUAL_DEAD_ZONE,
                )
            })
            .collect();
        Some(crate::dct::inverse_dct2d(&levels, size))
    } else {
        None
    };

    for u in 0..size {
        for v in 0..size {
            let dr = leaf.dom_row as usize + 2 * u;
            let dc = leaf.dom_col as usize + 2 * v;
            let d = (f64::from(img[dr * stride + dc])
                + f64::from(img[(dr + 1) * stride + dc])
                + f64::from(img[dr * stride + dc + 1])
                + f64::from(img[(dr + 1) * stride + dc + 1]))
                / 4.0;
            let (i, j) = crate::isometry::map(leaf.isometry, u, v, size);
            // §10.1: the 0.5 is added to the product, not to the sum — `+` is
            // left-associative in the C, and reassociating changes the truncated result.
            let mut value = (0.5 + d * alfa) + beta;
            if let Some(res) = &residual {
                value += res[u * size + v];
            }
            next[(leaf.row as usize + i) * stride + leaf.col as usize + j] =
                value.clamp(0.0, 255.0) as u8;
        }
    }
}

// ---------------------------------------------------------------------------- the writer

struct BitWriter {
    bits: Vec<bool>,
}

impl BitWriter {
    fn new() -> Self {
        Self { bits: Vec::new() }
    }

    /// §2: each value is written least-significant-bit first.
    fn write(&mut self, n: u32, value: u32) {
        for i in 0..n {
            self.bits.push((value >> i) & 1 != 0);
        }
    }

    /// §2: bits pack into each byte most-significant-bit first; the final partial byte is
    /// right-padded with zero.
    fn finish(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bits.len().div_ceil(8));
        for chunk in self.bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                if bit {
                    byte |= 1 << (7 - i);
                }
            }
            out.push(byte);
        }
        out
    }
}

/// Serialise a header and its leaves back into a `.ifs` bitstream — the inverse of
/// [`parse`], used only for cross-checking the encoder against `decmars` (Step 6). Driven
/// by the leaf *positions* rather than a recorded list of split decisions, so it
/// independently checks that the leaves tile the image exactly as `parse`'s walk expects.
pub fn write(hdr: &Header, leaves: &[Leaf]) -> Result<Vec<u8>, IfsError> {
    let mut by_pos = std::collections::HashMap::with_capacity(leaves.len());
    for leaf in leaves {
        if by_pos
            .insert((leaf.row, leaf.col, leaf.size), leaf)
            .is_some()
        {
            return Err(IfsError::DuplicateLeaf {
                row: leaf.row,
                col: leaf.col,
                size: leaf.size,
            });
        }
    }

    let mut w = BitWriter::new();
    w.write(4, hdr.bits_alfa);
    w.write(4, hdr.bits_beta);
    w.write(7, hdr.min_size);
    w.write(7, hdr.max_size);
    w.write(6, hdr.shift);
    w.write(12, hdr.width);
    w.write(12, hdr.height);
    w.write(8, hdr.int_max_alfa);

    write_walk(&mut w, hdr, 0, 0, hdr.virtual_size(), &by_pos)?;
    Ok(w.finish())
}

#[allow(clippy::type_complexity)]
fn write_walk(
    w: &mut BitWriter,
    hdr: &Header,
    row: u32,
    col: u32,
    size: u32,
    by_pos: &std::collections::HashMap<(u32, u32, u32), &Leaf>,
) -> Result<(), IfsError> {
    if row >= hdr.height || col >= hdr.width {
        return Ok(());
    }
    let half = size / 2;
    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    if forced {
        write_walk(w, hdr, row, col, half, by_pos)?;
        write_walk(w, hdr, row + half, col, half, by_pos)?;
        write_walk(w, hdr, row, col + half, half, by_pos)?;
        write_walk(w, hdr, row + half, col + half, half, by_pos)?;
        return Ok(());
    }

    let Some(&leaf) = by_pos.get(&(row, col, size)) else {
        if size <= hdr.min_size {
            return Err(IfsError::MissingLeaf { row, col, size });
        }
        w.write(1, 1);
        write_walk(w, hdr, row, col, half, by_pos)?;
        write_walk(w, hdr, row + half, col, half, by_pos)?;
        write_walk(w, hdr, row, col + half, half, by_pos)?;
        write_walk(w, hdr, row + half, col + half, half, by_pos)?;
        return Ok(());
    };

    if size > hdr.min_size {
        w.write(1, 0);
    }
    w.write(hdr.bits_alfa, leaf.qalfa);
    w.write(hdr.bits_beta, leaf.qbeta);
    if leaf.qalfa != 0 {
        w.write(3, u32::from(leaf.isometry));
        w.write(hdr.bits_coord_row(), leaf.dom_row / hdr.shift);
        w.write(hdr.bits_coord_col(), leaf.dom_col / hdr.shift);
    }
    Ok(())
}
