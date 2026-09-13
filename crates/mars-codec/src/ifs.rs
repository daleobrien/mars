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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Leaf {
    pub row: u32,
    pub col: u32,
    pub size: u32,
    pub qalfa: u32,
    pub qbeta: u32,
    /// Meaningless when `dc_only()` — always 0 in that case, per §6.
    pub isometry: u8,
    pub dom_row: u32,
    pub dom_col: u32,
}

impl Leaf {
    /// §6: `qalfa == 0` means a constant-fill block with no domain reference at all.
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
        qalfa,
        qbeta,
        isometry,
        dom_row,
        dom_col,
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

/// §7 dequantisation and §9/§10.1 reconstruction for one leaf, reading `img` (the
/// previous iteration) and writing `next` — the double-buffering of §10.1 is what makes
/// leaves independent of each other and safe to iterate in any order.
fn decode_leaf(hdr: &Header, leaf: &Leaf, img: &[u8], stride: usize, next: &mut [u8]) {
    let alfa = f64::from(leaf.qalfa) / f64::from(1u32 << hdr.bits_alfa) * hdr.max_alfa();
    let mut beta = f64::from(leaf.qbeta) / f64::from((1u32 << hdr.bits_beta) - 1)
        * ((1.0 + alfa.abs()) * 255.0);
    if alfa > 0.0 {
        beta -= alfa * 255.0;
    }

    let size = leaf.size as usize;
    for u in 0..size {
        for v in 0..size {
            let dr = leaf.dom_row as usize + 2 * u;
            let dc = leaf.dom_col as usize + 2 * v;
            let d = (f64::from(img[dr * stride + dc])
                + f64::from(img[(dr + 1) * stride + dc])
                + f64::from(img[dr * stride + dc + 1])
                + f64::from(img[(dr + 1) * stride + dc + 1]))
                / 4.0;
            let (i, j) = isometry_map(leaf.isometry, u, v, size);
            // §10.1: the 0.5 is added to the product, not to the sum — `+` is
            // left-associative in the C, and reassociating changes the truncated result.
            let value = ((0.5 + d * alfa) + beta).clamp(0.0, 255.0) as u8;
            next[(leaf.row as usize + i) * stride + leaf.col as usize + j] = value;
        }
    }
}

/// §9: destination `(i, j)` for source `(u, v)`, by isometry code.
fn isometry_map(k: u8, u: usize, v: usize, size: usize) -> (usize, usize) {
    match k {
        0 => (u, v),                       // IDENTITY
        1 => (size - 1 - v, u),            // L_ROTATE90
        2 => (v, size - 1 - u),            // R_ROTATE90
        3 => (size - 1 - u, size - 1 - v), // ROTATE180
        4 => (u, size - 1 - v),            // R_VERTICAL
        5 => (size - 1 - u, v),            // R_HORIZONTAL
        6 => (v, u),                       // F_DIAGONAL
        7 => (size - 1 - v, size - 1 - u), // S_DIAGONAL
        _ => unreachable!("isometry is a 3-bit field parsed as 0..=7"),
    }
}
