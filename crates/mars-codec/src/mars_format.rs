//! The `.mars` container format v0 -- Step 10.
//!
//! Deliberately not Mars 1 compatible (`implementation-plan.md` Step 10: "break Mars 1
//! compatibility deliberately and for stated reasons"). The raw fixed-width `.ifs`
//! bitstream ([`crate::ifs`]) leaves an easy 8-20%+ on the table that context-adaptive
//! rANS recovers for free; format decisions are expensive to reverse, so that break
//! happens once, deliberately, here. This module reuses [`Header`]/[`Leaf`] as the
//! in-memory representation Step 6's encoder already produces -- only the *serialisation*
//! changes.
//!
//! The container is a fixed byte-aligned header (magic, version, colour fields --
//! including channel/colour-space fields the encoder does not populate yet, so the format
//! never has to grow a colour section later, R&D plan §16 via the resolution in Step 10's
//! brief) followed by a sequence of `(id, length, payload)` sections. v0 defines exactly
//! one: `TREE`, the quadtree entropy-coded with [`mars_entropy`].
//!
//! **The decoder does not run any part of the encoder's search.** It reads a
//! representation -- mode, qalfa, qbeta, isometry, domain position, per leaf -- never "run
//! algorithm X with settings Y" (R&D plan §12). The contexts below are a defensible first
//! cut, not the brief's full "depth + neighbour split state": split flags and mode are
//! conditioned on size class (a monotonic proxy for quadtree depth), qalfa/qbeta/isometry
//! on size class crossed with DC-vs-domain mode, and the domain position on a zigzag delta
//! from the *previous leaf's* domain position in traversal order. Neighbour split state is
//! not modelled -- see `docs/decisions.md` for why this was scoped down and what would
//! close the gap.

use std::collections::HashMap;

use mars_entropy::{Decoder as EntropyDecoder, Event};

use crate::ifs::{Header, Leaf};

const MAGIC: [u8; 4] = *b"MARS";
const VERSION: u8 = 0;
const HEADER_LEN: usize = 20;
const SECTION_TABLE_ENTRY_LEN: usize = 5;
const SECTION_TREE: u8 = 1;

// `pub(crate)` (Step 14): the rate estimator (`crate::rate`) prices split-flag and leaf
// events against these exact context keys, so its cost estimate is keyed identically to
// what the real entropy coder charges -- see that module's doc for why a constant-bits
// approximation is exactly the bug this step exists to avoid.
pub(crate) const FIELD_SPLIT: u8 = 0;
pub(crate) const FIELD_MODE: u8 = 1;
pub(crate) const FIELD_QALFA: u8 = 2;
pub(crate) const FIELD_QBETA: u8 = 3;
pub(crate) const FIELD_QBETA_DC: u8 = 4;
pub(crate) const FIELD_ISOMETRY: u8 = 5;
pub(crate) const FIELD_DOM_ROW: u8 = 6;
pub(crate) const FIELD_DOM_COL: u8 = 7;

/// `bits_alfa`/`bits_beta` are stored as a whole byte (unlike Mars 1's packed 4-bit
/// fields) but are still bounded well short of 32, so that `1 << bits` and the zigzag
/// domain-delta alphabet (`2 * 2^bits_coord`) can never overflow a `u32` regardless of
/// what a corrupted file claims.
const MAX_BITS: u32 = 24;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MarsFormatError {
    #[error("not a .mars file: bad magic")]
    BadMagic,
    #[error("unsupported .mars version {0}")]
    UnsupportedVersion(u8),
    #[error("header truncated")]
    Truncated,
    #[error(
        "degenerate header: width, height and shift must be nonzero, bits_alfa must be \
         2..={MAX_BITS} and bits_beta 1..={MAX_BITS}"
    )]
    DegenerateHeader,
    #[error("width/height do not fit in this format's 16-bit fields")]
    DimensionsOverflow,
    #[error("min_size/max_size/shift do not fit in this format's 8-bit fields")]
    GeometryOverflow,
    #[error("section table claims {claimed} bytes but only {available} remain")]
    SectionOutOfBounds { claimed: usize, available: usize },
    #[error("the TREE section is missing")]
    MissingTreeSection,
    #[error("compressed tree payload is invalid: {0}")]
    InvalidTreePayload(#[from] mars_entropy::InvalidCompressedData),
    #[error("no leaf at ({row},{col},{size}) and none can follow (size <= min_size)")]
    MissingLeaf { row: u32, col: u32, size: u32 },
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
    #[error("two leaves claim the same position ({row},{col},{size})")]
    DuplicateLeaf { row: u32, col: u32, size: u32 },
}

fn zigzag(x: i64) -> u32 {
    ((x << 1) ^ (x >> 63)) as u32
}

fn unzigzag(z: u32) -> i64 {
    (i64::from(z >> 1)) ^ -i64::from(z & 1)
}

/// The largest total leaf count [`read`] will walk a tree looking for, derived purely
/// from header fields before any symbol is decoded. Without this, a `min_size` of 1 on a
/// header claiming the format's maximum 65535x65535 dimensions lets a handful of header
/// bytes demand a walk of up to `(65536/1)^2` quadtree nodes -- an "unbounded allocation
/// on arbitrary input" (the exact failure mode Step 10's brief names) even though every
/// individual step is well-defined. 16M leaf positions comfortably covers any realistic
/// photographic image (e.g. 4096x4096 at `min_size=1`, or far larger at the `min_size=4`
/// every fixture in this project actually uses) while rejecting the pathological ratios.
const MAX_LEAF_POSITIONS: u64 = 1 << 24;

fn valid_header_fields(
    width: u32,
    height: u32,
    shift: u32,
    bits_alfa: u32,
    bits_beta: u32,
    min_size: u32,
    max_size: u32,
) -> bool {
    width != 0
        && height != 0
        && shift != 0
        && (2..=MAX_BITS).contains(&bits_alfa)
        && (1..=MAX_BITS).contains(&bits_beta)
        // `min_size == 0` (or `max_size < min_size`, which forces splitting forever since
        // `forced` never clears) would make `walk_read`/`walk_write` recurse on a `size`
        // that never reaches 0 -- unbounded recursion, not merely a wrong answer.
        && min_size != 0
        && max_size >= min_size
}

/// Separate from [`valid_header_fields`] because it needs [`Header::virtual_size`], which
/// needs a constructed `Header` -- see that constant's doc for what this guards against.
fn leaf_count_within_bound(hdr: &Header) -> bool {
    let virtual_size = u64::from(hdr.virtual_size());
    let min_size = u64::from(hdr.min_size);
    (virtual_size / min_size).pow(2) <= MAX_LEAF_POSITIONS
}

/// Encode `(hdr, leaves)` -- Step 6's own output -- as a `.mars` v0 file.
pub fn write(hdr: &Header, leaves: &[Leaf]) -> Result<Vec<u8>, MarsFormatError> {
    if !valid_header_fields(
        hdr.width,
        hdr.height,
        hdr.shift,
        hdr.bits_alfa,
        hdr.bits_beta,
        hdr.min_size,
        hdr.max_size,
    ) || !leaf_count_within_bound(hdr)
    {
        return Err(MarsFormatError::DegenerateHeader);
    }
    if hdr.width > u32::from(u16::MAX) || hdr.height > u32::from(u16::MAX) {
        return Err(MarsFormatError::DimensionsOverflow);
    }
    if hdr.min_size > 255 || hdr.max_size > 255 || hdr.shift > 255 {
        return Err(MarsFormatError::GeometryOverflow);
    }

    let mut by_pos = HashMap::with_capacity(leaves.len());
    for leaf in leaves {
        if by_pos
            .insert((leaf.row, leaf.col, leaf.size), leaf)
            .is_some()
        {
            return Err(MarsFormatError::DuplicateLeaf {
                row: leaf.row,
                col: leaf.col,
                size: leaf.size,
            });
        }
    }

    let events = build_events(hdr, &by_pos)?;
    let tree = mars_entropy::encode(&events);

    let mut out = Vec::with_capacity(HEADER_LEN + SECTION_TABLE_ENTRY_LEN + tree.len());
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(0); // flags: none defined yet
    out.push(1); // channels: Y only -- Step 18 fills the rest this field already allows for
    out.push(8); // bit_depth
    out.push(0); // colour_space: unspecified/greyscale
    out.push(hdr.min_size as u8);
    out.push(hdr.max_size as u8);
    out.push(hdr.shift as u8);
    out.push(hdr.bits_alfa as u8);
    out.push(hdr.bits_beta as u8);
    out.push(hdr.int_max_alfa as u8);
    out.extend_from_slice(&(hdr.width as u16).to_le_bytes());
    out.extend_from_slice(&(hdr.height as u16).to_le_bytes());
    out.push(1); // section_count
    out.push(SECTION_TREE);
    out.extend_from_slice(&(u32::try_from(tree.len()).expect("tree fits in u32") ).to_le_bytes());
    out.extend_from_slice(&tree);
    Ok(out)
}

/// Decode a `.mars` v0 file back into `(hdr, leaves)`. Rejects any structurally invalid
/// input with a [`MarsFormatError`] rather than panicking, allocating unboundedly, or
/// indexing out of bounds -- this is the function `mars-codec`'s fuzz target drives.
pub fn read(data: &[u8]) -> Result<(Header, Vec<Leaf>), MarsFormatError> {
    if data.len() < HEADER_LEN {
        return Err(MarsFormatError::Truncated);
    }
    if data[0..4] != MAGIC {
        return Err(MarsFormatError::BadMagic);
    }
    let version = data[4];
    if version != VERSION {
        return Err(MarsFormatError::UnsupportedVersion(version));
    }
    let min_size = u32::from(data[9]);
    let max_size = u32::from(data[10]);
    let shift = u32::from(data[11]);
    let bits_alfa = u32::from(data[12]);
    let bits_beta = u32::from(data[13]);
    let int_max_alfa = u32::from(data[14]);
    let width = u32::from(u16::from_le_bytes([data[15], data[16]]));
    let height = u32::from(u16::from_le_bytes([data[17], data[18]]));
    let section_count = data[19];

    if !valid_header_fields(width, height, shift, bits_alfa, bits_beta, min_size, max_size) {
        return Err(MarsFormatError::DegenerateHeader);
    }
    let hdr = Header {
        bits_alfa,
        bits_beta,
        min_size,
        max_size,
        shift,
        width,
        height,
        int_max_alfa,
    };
    if !leaf_count_within_bound(&hdr) {
        return Err(MarsFormatError::DegenerateHeader);
    }

    let mut offset = HEADER_LEN;
    let mut tree: Option<&[u8]> = None;
    for _ in 0..section_count {
        if offset + SECTION_TABLE_ENTRY_LEN > data.len() {
            return Err(MarsFormatError::Truncated);
        }
        let id = data[offset];
        let len = u32::from_le_bytes([
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
        ]) as usize;
        offset += SECTION_TABLE_ENTRY_LEN;
        if offset + len > data.len() {
            return Err(MarsFormatError::SectionOutOfBounds {
                claimed: len,
                available: data.len() - offset,
            });
        }
        let payload = &data[offset..offset + len];
        offset += len;
        if id == SECTION_TREE {
            tree = Some(payload);
        }
    }
    let tree = tree.ok_or(MarsFormatError::MissingTreeSection)?;

    let mut dec = EntropyDecoder::new(tree)?;
    let mut pred = Predictor::default();
    let mut leaves = Vec::new();
    walk_read(&hdr, 0, 0, hdr.virtual_size(), &mut dec, &mut pred, &mut leaves)?;
    Ok((hdr, leaves))
}

/// The domain-position predictor threaded through traversal order (NW, SW, NE, SE, same
/// as [`crate::ifs`]): each domain-referencing leaf's row/col (in `shift` units) is coded
/// as a zigzag delta from the previous one, rather than its absolute value.
#[derive(Default)]
struct Predictor {
    prev_row_units: i64,
    prev_col_units: i64,
}

/// The forward pass shared by [`write`] and, indirectly, Step 14's rate-estimation
/// warm-up (`crate::rate`): walk the tree once in canonical order and record every event
/// [`mars_entropy::encode`] will need, without touching the entropy coder itself.
fn build_events(
    hdr: &Header,
    by_pos: &HashMap<(u32, u32, u32), &Leaf>,
) -> Result<Vec<Event>, MarsFormatError> {
    let mut events = Vec::new();
    let mut pred = Predictor::default();
    walk_write(hdr, 0, 0, hdr.virtual_size(), by_pos, &mut pred, &mut events)?;
    Ok(events)
}

/// `pub(crate)`: the same event stream [`write`] would encode for `(hdr, leaves)`, built
/// from a plain leaf slice rather than the caller's own `by_pos` map. Step 14's rate
/// estimator uses this to turn a representative (not necessarily RD-optimal) partition
/// into the real, observed per-context symbol frequencies its "warm-up" snapshot is built
/// from -- see `crate::rate`'s module doc for why a snapshot beats a constant-bits guess.
pub(crate) fn events_for_leaves(hdr: &Header, leaves: &[Leaf]) -> Result<Vec<Event>, MarsFormatError> {
    let mut by_pos = HashMap::with_capacity(leaves.len());
    for leaf in leaves {
        if by_pos
            .insert((leaf.row, leaf.col, leaf.size), leaf)
            .is_some()
        {
            return Err(MarsFormatError::DuplicateLeaf {
                row: leaf.row,
                col: leaf.col,
                size: leaf.size,
            });
        }
    }
    build_events(hdr, &by_pos)
}

/// `pub(crate)`: the events one leaf alone would contribute, keyed by `size_class`, with a
/// *fresh* domain-position predictor (`Predictor::default()`) rather than the real
/// whole-image sequential one. Used only by Step 14's rate estimator, which prices leaf
/// candidates independently and out of final traversal order during the bottom-up search
/// (the true predecessor leaf is not yet known at decision time) -- see `crate::rate`'s
/// module doc for this specific, documented approximation and why it is confined to the
/// two domain-position fields.
pub(crate) fn leaf_events(hdr: &Header, leaf: &Leaf, size_class: u32) -> Vec<Event> {
    let mut events = Vec::new();
    let mut pred = Predictor::default();
    emit_leaf(hdr, leaf, size_class, &mut pred, &mut events);
    events
}

#[allow(clippy::too_many_arguments)]
fn walk_write(
    hdr: &Header,
    row: u32,
    col: u32,
    size: u32,
    by_pos: &HashMap<(u32, u32, u32), &Leaf>,
    pred: &mut Predictor,
    events: &mut Vec<Event>,
) -> Result<(), MarsFormatError> {
    if row >= hdr.height || col >= hdr.width {
        return Ok(());
    }
    let half = size / 2;
    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    let size_class = size.trailing_zeros();

    if let Some(&leaf) = by_pos.get(&(row, col, size)) {
        if !forced && size > hdr.min_size {
            events.push(Event {
                ctx: (FIELD_SPLIT, size_class),
                alphabet: 2,
                symbol: 0,
            });
        }
        emit_leaf(hdr, leaf, size_class, pred, events);
        return Ok(());
    }

    if !forced {
        if size <= hdr.min_size {
            return Err(MarsFormatError::MissingLeaf { row, col, size });
        }
        events.push(Event {
            ctx: (FIELD_SPLIT, size_class),
            alphabet: 2,
            symbol: 1,
        });
    }
    walk_write(hdr, row, col, half, by_pos, pred, events)?;
    walk_write(hdr, row + half, col, half, by_pos, pred, events)?;
    walk_write(hdr, row, col + half, half, by_pos, pred, events)?;
    walk_write(hdr, row + half, col + half, half, by_pos, pred, events)?;
    Ok(())
}

fn emit_leaf(hdr: &Header, leaf: &Leaf, size_class: u32, pred: &mut Predictor, events: &mut Vec<Event>) {
    if leaf.qalfa == 0 {
        events.push(Event {
            ctx: (FIELD_MODE, size_class),
            alphabet: 2,
            symbol: 0,
        });
        events.push(Event {
            ctx: (FIELD_QBETA_DC, size_class),
            alphabet: 1 << hdr.bits_beta,
            symbol: leaf.qbeta,
        });
        return;
    }

    events.push(Event {
        ctx: (FIELD_MODE, size_class),
        alphabet: 2,
        symbol: 1,
    });
    events.push(Event {
        ctx: (FIELD_QALFA, size_class),
        alphabet: (1 << hdr.bits_alfa) - 1,
        symbol: leaf.qalfa - 1,
    });
    events.push(Event {
        ctx: (FIELD_QBETA, size_class),
        alphabet: 1 << hdr.bits_beta,
        symbol: leaf.qbeta,
    });
    events.push(Event {
        ctx: (FIELD_ISOMETRY, size_class),
        alphabet: 8,
        symbol: u32::from(leaf.isometry),
    });

    let row_units = i64::from(leaf.dom_row / hdr.shift);
    let col_units = i64::from(leaf.dom_col / hdr.shift);
    let row_range = 1i64 << hdr.bits_coord_row();
    let col_range = 1i64 << hdr.bits_coord_col();
    events.push(Event {
        ctx: (FIELD_DOM_ROW, size_class),
        alphabet: (2 * row_range) as u32,
        symbol: zigzag(row_units - pred.prev_row_units),
    });
    events.push(Event {
        ctx: (FIELD_DOM_COL, size_class),
        alphabet: (2 * col_range) as u32,
        symbol: zigzag(col_units - pred.prev_col_units),
    });
    pred.prev_row_units = row_units;
    pred.prev_col_units = col_units;
}

#[allow(clippy::too_many_arguments)]
fn walk_read(
    hdr: &Header,
    row: u32,
    col: u32,
    size: u32,
    dec: &mut EntropyDecoder,
    pred: &mut Predictor,
    leaves: &mut Vec<Leaf>,
) -> Result<(), MarsFormatError> {
    if row >= hdr.height || col >= hdr.width {
        return Ok(());
    }
    let half = size / 2;
    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    let size_class = size.trailing_zeros();

    let split = if forced {
        true
    } else if size > hdr.min_size {
        dec.next((FIELD_SPLIT, size_class), 2) == 1
    } else {
        false
    };

    if split {
        walk_read(hdr, row, col, half, dec, pred, leaves)?;
        walk_read(hdr, row + half, col, half, dec, pred, leaves)?;
        walk_read(hdr, row, col + half, half, dec, pred, leaves)?;
        walk_read(hdr, row + half, col + half, half, dec, pred, leaves)?;
        return Ok(());
    }

    let mode = dec.next((FIELD_MODE, size_class), 2);
    if mode == 0 {
        let qbeta = dec.next((FIELD_QBETA_DC, size_class), 1 << hdr.bits_beta);
        leaves.push(Leaf {
            row,
            col,
            size,
            qalfa: 0,
            qbeta,
            isometry: 0,
            dom_row: 0,
            dom_col: 0,
        });
        return Ok(());
    }

    let qalfa = dec.next((FIELD_QALFA, size_class), (1 << hdr.bits_alfa) - 1) + 1;
    let qbeta = dec.next((FIELD_QBETA, size_class), 1 << hdr.bits_beta);
    let isometry = dec.next((FIELD_ISOMETRY, size_class), 8) as u8;

    let row_range = 1i64 << hdr.bits_coord_row();
    let col_range = 1i64 << hdr.bits_coord_col();
    let row_delta = unzigzag(dec.next((FIELD_DOM_ROW, size_class), (2 * row_range) as u32));
    let col_delta = unzigzag(dec.next((FIELD_DOM_COL, size_class), (2 * col_range) as u32));
    let row_units = pred.prev_row_units + row_delta;
    let col_units = pred.prev_col_units + col_delta;
    pred.prev_row_units = row_units;
    pred.prev_col_units = col_units;

    if row_units < 0 || col_units < 0 || row_units > i64::from(u32::MAX) || col_units > i64::from(u32::MAX) {
        return Err(MarsFormatError::DomainOutOfBounds {
            row,
            col,
            size,
            dom_row: 0,
            dom_col: 0,
        });
    }
    let dom_row = hdr.shift * row_units as u32;
    let dom_col = hdr.shift * col_units as u32;
    if dom_row + 2 * size > hdr.height || dom_col + 2 * size > hdr.width {
        return Err(MarsFormatError::DomainOutOfBounds {
            row,
            col,
            size,
            dom_row,
            dom_col,
        });
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode_image, EncodeParams};
    use mars_core::Plane;

    fn synthetic_image(w: usize, h: usize) -> Plane {
        let mut data = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                data[r * w + c] = ((r * 7 + c * 13) % 256) as u8;
            }
        }
        Plane::from_vec(w, h, data)
    }

    fn params() -> EncodeParams {
        EncodeParams {
            min_size: 4,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            t_rms: 8.0,
            zero_threshold: 0,
            lambda: None,
        }
    }

    #[test]
    fn round_trips_a_real_encode() {
        let image = synthetic_image(64, 48);
        let (hdr, leaves, _evals) = encode_image(&image, &params());

        let bytes = write(&hdr, &leaves).unwrap();
        let (hdr2, leaves2) = read(&bytes).unwrap();

        assert_eq!(hdr, hdr2);
        assert_eq!(leaves.len(), leaves2.len());
        let mut a: Vec<_> = leaves.iter().map(|l| (l.row, l.col, l.size, l.qalfa, l.qbeta, l.isometry, l.dom_row, l.dom_col)).collect();
        let mut b: Vec<_> = leaves2.iter().map(|l| (l.row, l.col, l.size, l.qalfa, l.qbeta, l.isometry, l.dom_row, l.dom_col)).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn compresses_smaller_than_the_raw_ifs_bitstream() {
        let image = synthetic_image(128, 96);
        let (hdr, leaves, _evals) = encode_image(&image, &params());
        let raw = crate::ifs::write(&hdr, &leaves).unwrap();
        let mars = write(&hdr, &leaves).unwrap();
        assert!(
            mars.len() < raw.len(),
            "entropy-coded {} bytes should beat raw {} bytes",
            mars.len(),
            raw.len()
        );
    }

    #[test]
    fn rejects_bad_magic_without_panicking() {
        let mut bytes = vec![0u8; HEADER_LEN + SECTION_TABLE_ENTRY_LEN];
        bytes[0..4].copy_from_slice(b"NOPE");
        assert_eq!(read(&bytes), Err(MarsFormatError::BadMagic));
    }

    #[test]
    fn rejects_truncated_input_without_panicking() {
        assert_eq!(read(&[]), Err(MarsFormatError::Truncated));
        assert_eq!(read(&MAGIC), Err(MarsFormatError::Truncated));
    }

    #[test]
    fn rejects_oversized_section_claims_without_panicking() {
        let image = synthetic_image(32, 32);
        let (hdr, leaves, _evals) = encode_image(&image, &params());
        let mut bytes = write(&hdr, &leaves).unwrap();
        // Corrupt the TREE section's length to claim far more than remains.
        let len_off = HEADER_LEN + 1;
        bytes[len_off..len_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(read(&bytes), Err(MarsFormatError::SectionOutOfBounds { .. })));
    }

    #[test]
    fn rejects_degenerate_header_fields_without_panicking() {
        let mut bytes = vec![0u8; HEADER_LEN + SECTION_TABLE_ENTRY_LEN];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = VERSION;
        bytes[12] = 0; // bits_alfa = 0, invalid
        bytes[13] = 4; // bits_beta
        bytes[15..17].copy_from_slice(&16u16.to_le_bytes());
        bytes[17..19].copy_from_slice(&16u16.to_le_bytes());
        bytes[11] = 4; // shift
        assert_eq!(read(&bytes), Err(MarsFormatError::DegenerateHeader));
    }
}
