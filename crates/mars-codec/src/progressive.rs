//! Layered ("progressive") bitstream — Step 19, with O7/Step22 qstep metadata.
//!
//! # Revised version-0 wire format
//!
//! The 35-byte header contains: `MPRG` (0..4), version = 0 (4), min/max size
//! (5/6), shift (7), alfa/beta bits (8/9), integer max alfa (10), LE u16
//! width/height (11..15), four LE u32 payload lengths (15..31), and the same
//! validated LE binary32 residual qstep as `MARS` at 31..35. The four entropy
//! payloads follow; length entries count only payload bytes. Reconstruction
//! uses the header step when layer 4 is complete; prefix offsets start at 35.
//! This replaces the old layout without a version bump or fallback: pre-O7
//! files must be re-encoded. Layer coding and the encoder warm-up are unchanged.
//!
//! `docs/decisions.md` D46 has the full design reasoning; the short version: this is
//! **not** a spatial multi-resolution pyramid. D11 (this file's sibling in `docs/`)
//! measured that Mars 1's pyramidal decoder loses 5+ dB whenever a range block's rendered
//! footprint (`size / 2^level`) drops below 1 px, and named Step 19 as the step that would
//! reintroduce the hazard if it reached for that design. Instead, every layer here is
//! rendered by building a full, real-resolution `Vec<`[`Leaf`]`>` — real `row`/`col`/`size`
//! for every leaf, never divided by a level — and calling the existing, unmodified
//! [`crate::ifs::decode_iterative`] once. What differs between layers is which *fields* of
//! each leaf are known accurately yet, never a leaf's spatial footprint:
//!
//! 1. **Base** — a fixed grid of `max_size` cells (positions derived from the header
//!    alone, no bits spent on them), each a quantised mean of the source pixels in that
//!    cell, rendered as a flat (mode 0) leaf.
//! 2. **Partition** — the real quadtree split flags, plus each leaf's final mode and a
//!    flat/affine approximation: mode-0/1 leaves get their *real* qbeta (and qgx/qgy for
//!    mode 1) and are fully resolved from here on; mode-2/3 leaves get a quantised mean of
//!    the source pixels in their own region, standing in for the not-yet-revealed fractal
//!    prediction, rendered as mode 0.
//! 3. **Fractal** — mode-2/3 leaves get their real qalfa/qbeta/isometry/domain position,
//!    rendered as mode 2 (no residual yet, even for leaves whose final mode is 3).
//! 4. **Residual** — mode-3 leaves get their real DCT residual coefficients (reusing
//!    [`crate::residual`]'s event vocabulary unchanged), rendered as mode 3.
//!
//! After layer 4, every leaf's rendered fields are bit-identical to the original
//! `(Header, Vec<Leaf>)`, so the full decode is expected to reproduce
//! `decode_iterative(hdr, leaves, N)` on the original leaves exactly for the same
//! iteration count `N` — the hard equality this module's tests pin (`P19.1`,
//! `docs/predictions.md`'s Step 19 entry).
//!
//! Each layer is its own independent [`mars_entropy`] byte stream (its own event
//! vocabulary, its own adaptive models, no cross-layer prediction — a documented, deliberate
//! scope cut, see `docs/predictions.md`). The container is a small fixed header followed by
//! four `(length, payload)` layers; any byte prefix landing on a layer boundary decodes,
//! since [`decode`] only ever uses however many *complete* layers are present in `data`.

use std::collections::HashMap;

use mars_core::Plane;
use mars_entropy::{Decoder as EntropyDecoder, Event};

use crate::ifs::Leaf;
use crate::mars_format::Header;
use crate::quant::ResidualQstep;

const MAGIC: [u8; 4] = *b"MPRG";
const VERSION: u8 = 0;
/// Magic, version, geometry, dimensions, four layer lengths and binary32 qstep.
const HEADER_LEN: usize = 4 + 1 + 6 + 4 + 16 + 4;
/// Default fixed-point iteration count used by [`decode`].
pub const DECODE_ITERATIONS: u32 = 10;

/// Whether `data` starts with this module's own magic bytes -- for a caller (CLI-E's
/// `decmars`) that needs to dispatch between a progressive stream and `mars_codec::
/// color`'s single-layer `MARC` container before knowing which one it has. Only checks
/// the magic, not the full header (`decode`/`layer_end_offsets` still validate the rest
/// and return their own error if it's malformed) -- this is a format *sniff*, not a
/// validity check.
pub fn is_progressive(data: &[u8]) -> bool {
    data.len() >= MAGIC.len() && data[..MAGIC.len()] == MAGIC
}

/// Mirrors `mars_format::MAX_LEAF_POSITIONS`'s reasoning exactly: a header claiming
/// pathological dimensions/`min_size` ratios must not make [`decode`] walk or allocate an
/// unbounded number of positions before any symbol is even decoded.
const MAX_LEAF_POSITIONS: u64 = 1 << 24;

// ---------------------------------------------------------------------------- event fields
//
// Each layer is encoded as its own, independent `mars_entropy::encode` call (own
// `HashMap<ContextKey, AdaptiveModel>`), so these ids only need to be unique *within* one
// layer's own event vocabulary, not globally across layers or against `mars_format`'s own
// `FIELD_*` constants.

const F_DC: u8 = 0;
const F_SPLIT: u8 = 1;
const F_FINAL_MODE: u8 = 2;
const F_QGX: u8 = 3;
const F_QGY: u8 = 4;
const F_QALFA: u8 = 5;
const F_QBETA_FRAC: u8 = 6;
const F_ISOMETRY: u8 = 7;
const F_DOM_ROW: u8 = 8;
const F_DOM_COL: u8 = 9;

const MODE_ALPHABET: u32 = 4;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProgressiveError {
    #[error("iterations must be positive")]
    InvalidIterations,
    #[error("not a progressive Mars stream: bad magic")]
    BadMagic,
    #[error("residual qstep must be finite and in [1, 65535]")]
    InvalidResidualQstep,
    #[error("unsupported progressive stream version {0}")]
    UnsupportedVersion(u8),
    #[error("stream truncated before the fixed header could be read")]
    Truncated,
    #[error(
        "degenerate header: width, height and shift must be nonzero, bits_alfa must be \
         1..={MAX_BITS} and bits_beta 1..={MAX_BITS}, min_size nonzero, max_size >= min_size"
    )]
    DegenerateHeader,
    #[error("min_size/max_size/shift do not fit in this format's 8-bit fields")]
    GeometryOverflow,
    #[error("width/height do not fit in this format's 16-bit fields")]
    DimensionsOverflow,
    #[error("no layer is fully present in the given bytes (need at least the base layer)")]
    InsufficientData,
    #[error("layer payload is invalid: {0}")]
    InvalidLayerPayload(#[from] mars_entropy::InvalidCompressedData),
    #[error("no leaf at ({row},{col},{size}) and none can follow (size <= min_size)")]
    MissingLeaf { row: u32, col: u32, size: u32 },
    #[error("two leaves claim the same position ({row},{col},{size})")]
    DuplicateLeaf { row: u32, col: u32, size: u32 },
}

const MAX_BITS: u32 = 24;

fn valid_header_fields(hdr: &Header) -> bool {
    hdr.width != 0
        && hdr.height != 0
        && hdr.shift != 0
        && (1..=MAX_BITS).contains(&hdr.bits_alfa)
        && (1..=MAX_BITS).contains(&hdr.bits_beta)
        && hdr.min_size != 0
        && hdr.max_size >= hdr.min_size
}

fn leaf_count_within_bound(hdr: &Header) -> bool {
    let virtual_size = u64::from(hdr.virtual_size());
    let min_size = u64::from(hdr.min_size);
    (virtual_size / min_size).pow(2) <= MAX_LEAF_POSITIONS
}

fn zigzag(x: i64) -> u32 {
    ((x << 1) ^ (x >> 63)) as u32
}

fn unzigzag(z: u32) -> i64 {
    (i64::from(z >> 1)) ^ -i64::from(z & 1)
}

/// Mirrors `mars_format`'s own (private) `grad_alphabet` exactly — duplicated rather than
/// shared, matching this workspace's existing precedent of each bitstream module owning
/// its own small walk/codec helpers (`ifs.rs` and `mars_format.rs` each have their own
/// bit reader/writer rather than a shared trait, per §2.3 "traits when there are two
/// implementations, not in anticipation of one").
fn grad_alphabet() -> u32 {
    (2 * i64::from(crate::encode::AFFINE_GRAD_CLAMP) + 1) as u32
}

/// The domain-position predictor for layer 3, identical in spirit to `mars_format`'s own:
/// each fractal leaf's domain row/col (in `shift` units) is coded as a zigzag delta from
/// the *previous fractal leaf's* domain position, in canonical traversal order.
#[derive(Default)]
struct DomPredictor {
    prev_row_units: i64,
    prev_col_units: i64,
}

/// The quantised mean of `image`'s pixels in the `size x size` region at `(row, col)`,
/// scaled to the same `[0, max_qbeta]` range `decode_leaf`'s mode-0 dequantisation expects
/// (`beta = qbeta / max_qbeta * 255`, `alfa == 0`) — so a synthetic mode-0 [`Leaf`] built
/// from this value renders as the flat approximation this module's layers 1-2 intend.
fn quantized_mean(image: &Plane, row: u32, col: u32, size: u32, max_qbeta: u32) -> u32 {
    let (row, col, size) = (row as usize, col as usize, size as usize);
    let mut sum = 0u64;
    let mut n = 0u64;
    for r in row..row + size {
        for c in col..col + size {
            sum += u64::from(image.get(c, r));
            n += 1;
        }
    }
    let mean = sum as f64 / n as f64;
    (mean / 255.0 * f64::from(max_qbeta))
        .round()
        .clamp(0.0, f64::from(max_qbeta)) as u32
}

/// The `max_size`-grid layer-1 tiling: positions only, derived from `hdr` alone (no bits
/// are ever spent transmitting it) — the same "force-split at the image edge" rule every
/// other quadtree walk in this crate uses, just stopped unconditionally at `max_size`
/// rather than continuing down to `min_size`.
fn base_grid(hdr: &Header) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    fn walk(hdr: &Header, row: u32, col: u32, size: u32, out: &mut Vec<(u32, u32, u32)>) {
        if row >= hdr.height || col >= hdr.width || size == 0 {
            return;
        }
        let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
        if forced {
            let half = size / 2;
            walk(hdr, row, col, half, out);
            walk(hdr, row + half, col, half, out);
            walk(hdr, row, col + half, half, out);
            walk(hdr, row + half, col + half, half, out);
        } else {
            out.push((row, col, size));
        }
    }
    walk(hdr, 0, 0, hdr.virtual_size(), &mut out);
    out
}

// -------------------------------------------------------------------------------- encoding

/// Encode an already-encoded `(hdr, leaves)` (e.g. `mars_codec::encode::encode_image_rd`'s
/// output) as a 4-layer progressive stream. `image` is the *source* image — needed for
/// layer 1's grid means and layer 2's flat approximation of mode-2/3 leaves, neither of
/// which exists anywhere in `leaves` itself.
pub fn encode(image: &Plane, hdr: &impl crate::ifs::DecodeHeader, leaves: &[Leaf]) -> Result<Vec<u8>, ProgressiveError> {
    let hdr = &Header { geometry: *hdr.geometry(), residual_qstep: hdr.residual_qstep() };
    if !valid_header_fields(hdr) || !leaf_count_within_bound(hdr) {
        return Err(ProgressiveError::DegenerateHeader);
    }
    if hdr.width > u32::from(u16::MAX) || hdr.height > u32::from(u16::MAX) {
        return Err(ProgressiveError::DimensionsOverflow);
    }
    if hdr.min_size > 255 || hdr.max_size > 255 || hdr.shift > 255 {
        return Err(ProgressiveError::GeometryOverflow);
    }

    let mut by_pos = HashMap::with_capacity(leaves.len());
    for leaf in leaves {
        if by_pos
            .insert((leaf.row, leaf.col, leaf.size), leaf)
            .is_some()
        {
            return Err(ProgressiveError::DuplicateLeaf {
                row: leaf.row,
                col: leaf.col,
                size: leaf.size,
            });
        }
    }

    let max_qbeta = (1u32 << hdr.bits_beta) - 1;

    // ---- Layer 1: base grid.
    let grid = base_grid(hdr);
    let mut l1_events = Vec::with_capacity(grid.len());
    for &(row, col, size) in &grid {
        let dc = quantized_mean(image, row, col, size, max_qbeta);
        l1_events.push(Event {
            ctx: (F_DC, 0),
            alphabet: max_qbeta + 1,
            symbol: dc,
        });
    }
    let layer1 = mars_entropy::encode(&l1_events);

    // ---- Layer 2: real partition + per-leaf base fields. Also records canonical
    // traversal order, reused by layers 3/4 exactly like `mars_format::walk_write` reuses
    // its own traversal for the domain-position predictor.
    let mut l2_events = Vec::new();
    let mut order: Vec<&Leaf> = Vec::new();
    walk_write_l2(
        hdr,
        0,
        0,
        hdr.virtual_size(),
        &by_pos,
        image,
        max_qbeta,
        &mut l2_events,
        &mut order,
    )?;
    let layer2 = mars_entropy::encode(&l2_events);

    // ---- Layer 3: fractal refinement for mode 2/3 leaves, in canonical order.
    let mut l3_events = Vec::new();
    let mut pred = DomPredictor::default();
    for &leaf in &order {
        if leaf.mode == 2 || leaf.mode == 3 {
            emit_fractal(hdr, leaf, &mut pred, &mut l3_events);
        }
    }
    let layer3 = mars_entropy::encode(&l3_events);

    // ---- Layer 4: residual for mode 3 leaves, in canonical order.
    let mut l4_events = Vec::new();
    for &leaf in &order {
        if leaf.mode == 3 {
            crate::residual::encode_events(
                &leaf.residual,
                leaf.size.trailing_zeros(),
                &mut l4_events,
            );
        }
    }
    let layer4 = mars_entropy::encode(&l4_events);

    let mut out =
        Vec::with_capacity(HEADER_LEN + layer1.len() + layer2.len() + layer3.len() + layer4.len());
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(hdr.min_size as u8);
    out.push(hdr.max_size as u8);
    out.push(hdr.shift as u8);
    out.push(hdr.bits_alfa as u8);
    out.push(hdr.bits_beta as u8);
    out.push(hdr.int_max_alfa as u8);
    out.extend_from_slice(&(hdr.width as u16).to_le_bytes());
    out.extend_from_slice(&(hdr.height as u16).to_le_bytes());
    for layer in [&layer1, &layer2, &layer3, &layer4] {
        out.extend_from_slice(
            &(u32::try_from(layer.len()).expect("layer fits in u32")).to_le_bytes(),
        );
    }
    out.extend_from_slice(&hdr.residual_qstep.to_le_bytes());
    out.extend_from_slice(&layer1);
    out.extend_from_slice(&layer2);
    out.extend_from_slice(&layer3);
    out.extend_from_slice(&layer4);
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn walk_write_l2<'a>(
    hdr: &Header,
    row: u32,
    col: u32,
    size: u32,
    by_pos: &HashMap<(u32, u32, u32), &'a Leaf>,
    image: &Plane,
    max_qbeta: u32,
    events: &mut Vec<Event>,
    order: &mut Vec<&'a Leaf>,
) -> Result<(), ProgressiveError> {
    if row >= hdr.height || col >= hdr.width {
        return Ok(());
    }
    let half = size / 2;
    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    let size_class = size.trailing_zeros();

    if let Some(&leaf) = by_pos.get(&(row, col, size)) {
        if !forced && size > hdr.min_size {
            events.push(Event {
                ctx: (F_SPLIT, size_class),
                alphabet: 2,
                symbol: 0,
            });
        }
        events.push(Event {
            ctx: (F_FINAL_MODE, size_class),
            alphabet: MODE_ALPHABET,
            symbol: u32::from(leaf.mode),
        });
        match leaf.mode {
            0 => {
                events.push(Event {
                    ctx: (F_DC, size_class),
                    alphabet: max_qbeta + 1,
                    symbol: leaf.qbeta,
                });
            }
            1 => {
                events.push(Event {
                    ctx: (F_DC, size_class),
                    alphabet: max_qbeta + 1,
                    symbol: leaf.qbeta,
                });
                let alphabet = grad_alphabet();
                events.push(Event {
                    ctx: (F_QGX, size_class),
                    alphabet,
                    symbol: zigzag(i64::from(leaf.qgx)),
                });
                events.push(Event {
                    ctx: (F_QGY, size_class),
                    alphabet,
                    symbol: zigzag(i64::from(leaf.qgy)),
                });
            }
            2 | 3 => {
                let dc = quantized_mean(image, row, col, size, max_qbeta);
                events.push(Event {
                    ctx: (F_DC, size_class),
                    alphabet: max_qbeta + 1,
                    symbol: dc,
                });
            }
            _ => unreachable!("Leaf::mode is only ever constructed as 0..=3"),
        }
        order.push(leaf);
        return Ok(());
    }

    if !forced {
        if size <= hdr.min_size {
            return Err(ProgressiveError::MissingLeaf { row, col, size });
        }
        events.push(Event {
            ctx: (F_SPLIT, size_class),
            alphabet: 2,
            symbol: 1,
        });
    }
    walk_write_l2(hdr, row, col, half, by_pos, image, max_qbeta, events, order)?;
    walk_write_l2(
        hdr,
        row + half,
        col,
        half,
        by_pos,
        image,
        max_qbeta,
        events,
        order,
    )?;
    walk_write_l2(
        hdr,
        row,
        col + half,
        half,
        by_pos,
        image,
        max_qbeta,
        events,
        order,
    )?;
    walk_write_l2(
        hdr,
        row + half,
        col + half,
        half,
        by_pos,
        image,
        max_qbeta,
        events,
        order,
    )?;
    Ok(())
}

fn emit_fractal(hdr: &Header, leaf: &Leaf, pred: &mut DomPredictor, events: &mut Vec<Event>) {
    let size_class = leaf.size.trailing_zeros();
    events.push(Event {
        ctx: (F_QALFA, size_class),
        alphabet: (1 << hdr.bits_alfa) - 1,
        symbol: leaf.qalfa - 1,
    });
    events.push(Event {
        ctx: (F_QBETA_FRAC, size_class),
        alphabet: 1 << hdr.bits_beta,
        symbol: leaf.qbeta,
    });
    events.push(Event {
        ctx: (F_ISOMETRY, size_class),
        alphabet: 8,
        symbol: u32::from(leaf.isometry),
    });

    let row_units = i64::from(leaf.dom_row / hdr.shift);
    let col_units = i64::from(leaf.dom_col / hdr.shift);
    let row_range = 1i64 << hdr.bits_coord_row();
    let col_range = 1i64 << hdr.bits_coord_col();
    events.push(Event {
        ctx: (F_DOM_ROW, size_class),
        alphabet: (2 * row_range) as u32,
        symbol: zigzag(row_units - pred.prev_row_units),
    });
    events.push(Event {
        ctx: (F_DOM_COL, size_class),
        alphabet: (2 * col_range) as u32,
        symbol: zigzag(col_units - pred.prev_col_units),
    });
    pred.prev_row_units = row_units;
    pred.prev_col_units = col_units;
}

// -------------------------------------------------------------------------------- decoding

/// One leaf's progressively-accumulated knowledge, decode-side only. `final_mode` and
/// `qbeta` (the flat/base value) are known from layer 2 onward; `qalfa`/`frac_qbeta`/
/// `isometry`/`dom_row`/`dom_col` from layer 3 onward (mode 2/3 leaves only); `residual`
/// from layer 4 onward (mode 3 leaves only).
#[derive(Default, Clone)]
struct LeafState {
    row: u32,
    col: u32,
    size: u32,
    final_mode: u8,
    qbeta: u32,
    qgx: i32,
    qgy: i32,
    qalfa: u32,
    frac_qbeta: u32,
    isometry: u8,
    dom_row: u32,
    dom_col: u32,
    residual: Vec<i32>,
}

/// The result of decoding a (possibly truncated) progressive stream.
#[derive(Debug)]
pub struct Decoded {
    pub image: Plane,
    /// How many of the 4 layers were fully present and used — `1..=4`.
    pub layers: u8,
}

struct ParsedHeader {
    hdr: Header,
    layer_lens: [usize; 4],
}

fn parse_header(data: &[u8]) -> Result<ParsedHeader, ProgressiveError> {
    if data.len() < HEADER_LEN {
        return Err(ProgressiveError::Truncated);
    }
    if data[0..4] != MAGIC {
        return Err(ProgressiveError::BadMagic);
    }
    let version = data[4];
    if version != VERSION {
        return Err(ProgressiveError::UnsupportedVersion(version));
    }
    let residual_qstep = ResidualQstep::from_le_bytes(data[31..35].try_into().unwrap())
        .ok_or(ProgressiveError::InvalidResidualQstep)?;
    let min_size = u32::from(data[5]);
    let max_size = u32::from(data[6]);
    let shift = u32::from(data[7]);
    let bits_alfa = u32::from(data[8]);
    let bits_beta = u32::from(data[9]);
    let int_max_alfa = u32::from(data[10]);
    let width = u32::from(u16::from_le_bytes([data[11], data[12]]));
    let height = u32::from(u16::from_le_bytes([data[13], data[14]]));

    let hdr = Header { residual_qstep, geometry: crate::ifs::Header {
        bits_alfa,
        bits_beta,
        min_size,
        max_size,
        shift,
        width,
        height,
        int_max_alfa,
    }};
    if !valid_header_fields(&hdr) || !leaf_count_within_bound(&hdr) {
        return Err(ProgressiveError::DegenerateHeader);
    }

    let mut layer_lens = [0usize; 4];
    for (i, len) in layer_lens.iter_mut().enumerate() {
        let off = 15 + i * 4;
        *len =
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
    }
    Ok(ParsedHeader { hdr, layer_lens })
}

/// Cumulative end-offset of each of the 4 layers, from `layer_lens` (each an independent
/// payload length straight from the header) — shared by [`layer_end_offsets`] and
/// [`decode`] so the two can never compute this differently.
fn cumulative_offsets(layer_lens: &[usize; 4]) -> Result<[usize; 4], ProgressiveError> {
    let mut offsets = [0usize; 4];
    let mut acc = HEADER_LEN;
    for (offset, &len) in offsets.iter_mut().zip(layer_lens.iter()) {
        acc = acc.checked_add(len).ok_or(ProgressiveError::Truncated)?;
        *offset = acc;
    }
    Ok(offsets)
}

/// The byte offset after each of the 4 layers (cumulative, starting from `HEADER_LEN`) —
/// exposed so tests and the RD/gate harness can truncate a real stream to an exact
/// N-layer prefix rather than guessing at byte counts.
pub fn layer_end_offsets(data: &[u8]) -> Result<[usize; 4], ProgressiveError> {
    let parsed = parse_header(data)?;
    cumulative_offsets(&parsed.layer_lens)
}

/// Decode however many of the 4 layers are *fully* present in `data` — i.e. `data` may be
/// any prefix of a stream [`encode`] produced, truncated at (or past) a layer boundary.
/// Bytes belonging to a partially-present layer are ignored, not treated as an error: the
/// caller gets back the image built from the highest layer count fully available.
pub fn decode(data: &[u8]) -> Result<Decoded, ProgressiveError> {
    decode_with_iterations(data, DECODE_ITERATIONS)
}

/// Decode all fully present layers using a positive fixed-point iteration count.
/// Like [`decode`], incomplete trailing layers are ignored. Iteration starts from
/// the flat grey seed; base and partition-only layers settle after one iteration.
pub fn decode_with_iterations(
    data: &[u8],
    iterations: u32,
) -> Result<Decoded, ProgressiveError> {
    if iterations == 0 {
        return Err(ProgressiveError::InvalidIterations);
    }
    let parsed = parse_header(data)?;
    let hdr = parsed.hdr;
    let offsets = cumulative_offsets(&parsed.layer_lens)?;

    let avail = data.len();
    if avail < offsets[0] {
        return Err(ProgressiveError::InsufficientData);
    }
    let max_qbeta = (1u32 << hdr.bits_beta) - 1;

    // ---- Layer 1 (always fully decoded if reachable at all).
    let grid = base_grid(&hdr);
    let mut dec1 = EntropyDecoder::new(&data[HEADER_LEN..offsets[0]])?;
    let mut states: Vec<LeafState> = grid
        .iter()
        .map(|&(row, col, size)| LeafState {
            row,
            col,
            size,
            final_mode: 0,
            qbeta: dec1.next((F_DC, 0), max_qbeta + 1),
            ..Default::default()
        })
        .collect();

    if avail < offsets[1] {
        let image = render(&hdr, &states, 1, iterations);
        return Ok(Decoded { image, layers: 1 });
    }

    // ---- Layer 2: real partition + per-leaf base fields, replacing `states`.
    let mut dec2 = EntropyDecoder::new(&data[offsets[0]..offsets[1]])?;
    let mut l2_states = Vec::new();
    walk_read_l2(
        &hdr,
        0,
        0,
        hdr.virtual_size(),
        &mut dec2,
        max_qbeta,
        &mut l2_states,
    )?;
    states = l2_states;

    if avail < offsets[2] {
        let image = render(&hdr, &states, 2, iterations);
        return Ok(Decoded { image, layers: 2 });
    }

    // ---- Layer 3: fractal refinement for mode 2/3 leaves, canonical (= `states`) order.
    let mut dec3 = EntropyDecoder::new(&data[offsets[1]..offsets[2]])?;
    let mut pred = DomPredictor::default();
    for state in states.iter_mut() {
        if state.final_mode == 2 || state.final_mode == 3 {
            read_fractal(&hdr, &mut dec3, state, &mut pred);
        }
    }

    if avail < offsets[3] {
        let image = render(&hdr, &states, 3, iterations);
        return Ok(Decoded { image, layers: 3 });
    }

    // ---- Layer 4: residual for mode 3 leaves.
    let mut dec4 = EntropyDecoder::new(&data[offsets[2]..offsets[3]])?;
    for state in states.iter_mut() {
        if state.final_mode == 3 {
            let n = (state.size * state.size) as usize;
            state.residual =
                crate::residual::decode_values(&mut dec4, state.size.trailing_zeros(), n);
        }
    }

    let image = render(&hdr, &states, 4, iterations);
    Ok(Decoded { image, layers: 4 })
}

#[allow(clippy::too_many_arguments)]
fn walk_read_l2(
    hdr: &Header,
    row: u32,
    col: u32,
    size: u32,
    dec: &mut EntropyDecoder,
    max_qbeta: u32,
    states: &mut Vec<LeafState>,
) -> Result<(), ProgressiveError> {
    if row >= hdr.height || col >= hdr.width {
        return Ok(());
    }
    let half = size / 2;
    let forced = size > hdr.max_size || row + size > hdr.height || col + size > hdr.width;
    let size_class = size.trailing_zeros();

    let split = if forced {
        true
    } else if size > hdr.min_size {
        dec.next((F_SPLIT, size_class), 2) == 1
    } else {
        false
    };

    if split {
        walk_read_l2(hdr, row, col, half, dec, max_qbeta, states)?;
        walk_read_l2(hdr, row + half, col, half, dec, max_qbeta, states)?;
        walk_read_l2(hdr, row, col + half, half, dec, max_qbeta, states)?;
        walk_read_l2(hdr, row + half, col + half, half, dec, max_qbeta, states)?;
        return Ok(());
    }

    let final_mode = dec.next((F_FINAL_MODE, size_class), MODE_ALPHABET) as u8;
    let mut state = LeafState {
        row,
        col,
        size,
        final_mode,
        ..Default::default()
    };
    match final_mode {
        0 | 2 | 3 => {
            state.qbeta = dec.next((F_DC, size_class), max_qbeta + 1);
        }
        1 => {
            state.qbeta = dec.next((F_DC, size_class), max_qbeta + 1);
            let alphabet = grad_alphabet();
            state.qgx = unzigzag(dec.next((F_QGX, size_class), alphabet)) as i32;
            state.qgy = unzigzag(dec.next((F_QGY, size_class), alphabet)) as i32;
        }
        _ => unreachable!("MODE_ALPHABET == 4 rules out anything else"),
    }
    states.push(state);
    Ok(())
}

fn read_fractal(
    hdr: &Header,
    dec: &mut EntropyDecoder,
    state: &mut LeafState,
    pred: &mut DomPredictor,
) {
    let size_class = state.size.trailing_zeros();
    state.qalfa = dec.next((F_QALFA, size_class), (1 << hdr.bits_alfa) - 1) + 1;
    state.frac_qbeta = dec.next((F_QBETA_FRAC, size_class), 1 << hdr.bits_beta);
    state.isometry = dec.next((F_ISOMETRY, size_class), 8) as u8;

    let row_range = 1i64 << hdr.bits_coord_row();
    let col_range = 1i64 << hdr.bits_coord_col();
    let row_delta = unzigzag(dec.next((F_DOM_ROW, size_class), (2 * row_range) as u32));
    let col_delta = unzigzag(dec.next((F_DOM_COL, size_class), (2 * col_range) as u32));
    let row_units = pred.prev_row_units + row_delta;
    let col_units = pred.prev_col_units + col_delta;
    pred.prev_row_units = row_units;
    pred.prev_col_units = col_units;
    state.dom_row = hdr.shift * row_units.max(0) as u32;
    state.dom_col = hdr.shift * col_units.max(0) as u32;
}

/// Render `states` at `layer_reached` (1..=4) into a full-resolution [`Plane`] by building
/// a real [`Leaf`] list and calling [`crate::ifs::decode_iterative`] — the one and only
/// place any progressive layer touches pixels, and the reason this design never divides a
/// leaf's footprint by a pyramid level (D46).
fn render(hdr: &Header, states: &[LeafState], layer_reached: u8, iterations: u32) -> Plane {
    let leaves: Vec<Leaf> = states
        .iter()
        .map(|s| match s.final_mode {
            0 => Leaf {
                row: s.row,
                col: s.col,
                size: s.size,
                mode: 0,
                qalfa: 0,
                qbeta: s.qbeta,
                isometry: 0,
                dom_row: 0,
                dom_col: 0,
                qgx: 0,
                qgy: 0,
                residual: Vec::new(),
            },
            1 => Leaf {
                row: s.row,
                col: s.col,
                size: s.size,
                mode: 1,
                qalfa: 0,
                qbeta: s.qbeta,
                isometry: 0,
                dom_row: 0,
                dom_col: 0,
                qgx: s.qgx,
                qgy: s.qgy,
                residual: Vec::new(),
            },
            2 | 3 if layer_reached >= 3 => {
                let use_residual = s.final_mode == 3 && layer_reached >= 4;
                Leaf {
                    row: s.row,
                    col: s.col,
                    size: s.size,
                    mode: if use_residual { 3 } else { 2 },
                    qalfa: s.qalfa,
                    qbeta: s.frac_qbeta,
                    isometry: s.isometry,
                    dom_row: s.dom_row,
                    dom_col: s.dom_col,
                    qgx: 0,
                    qgy: 0,
                    residual: if use_residual {
                        s.residual.clone()
                    } else {
                        Vec::new()
                    },
                }
            }
            2 | 3 => Leaf {
                // Not yet revealed (layer < 3): render the flat approximation.
                row: s.row,
                col: s.col,
                size: s.size,
                mode: 0,
                qalfa: 0,
                qbeta: s.qbeta,
                isometry: 0,
                dom_row: 0,
                dom_col: 0,
                qgx: 0,
                qgy: 0,
                residual: Vec::new(),
            },
            _ => unreachable!("LeafState::final_mode is only ever constructed as 0..=3"),
        })
        .collect();
    crate::ifs::decode_iterative(hdr, &leaves, iterations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{encode_image_rd, EncodeParams};
    use mars_core::metrics::psnr;

    fn textured_gradient_image(w: usize, h: usize) -> Plane {
        let mut data = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                let base = (r * 255 / h.max(1)) as i32;
                let texture = (((r / 4) * 13 + (c / 4) * 7) % 32) as i32 - 16;
                data[r * w + c] = (base + texture).clamp(0, 255) as u8;
            }
        }
        Plane::from_vec(w, h, data)
    }

    fn rd_params(lambda: f64) -> EncodeParams {
        EncodeParams {
            min_size: 4,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
            t_rms: 8.0,
            zero_threshold: 0,
            lambda: Some(lambda),
        }
    }

    /// P19.1: once all four layers are present, the progressive decode must reproduce
    /// `decode_iterative` on the *original* leaves exactly, pixel for pixel — a hard
    /// equality oracle (verification-discipline), not a tolerance. Uses a lambda low
    /// enough to realistically exercise all four modes (mirrors `mars_format`'s own
    /// `every_mode_the_rd_search_picks_round_trips_through_mars_v0` test image/lambda).
    #[test]
    fn four_layer_decode_is_bit_exact_with_decode_iterative_on_the_original_leaves() {
        let image = textured_gradient_image(96, 96);
        let (hdr, leaves, _evals, stats) = encode_image_rd(&image, &rd_params(40.0));
        assert!(!leaves.is_empty());
        // Sanity: this test image/lambda combination should exercise more than one mode,
        // otherwise the mode-2/3-specific layers 3/4 would never be tested at all.
        let modes_used: std::collections::HashSet<u8> = leaves.iter().map(|l| l.mode).collect();
        assert!(
            modes_used.len() > 1,
            "expected a genuine mode mix, got only {modes_used:?} (stats: {stats:?})"
        );

        let stream = encode(&image, &hdr, &leaves).unwrap();
        let decoded = decode(&stream).unwrap();
        assert_eq!(decoded.layers, 4);

        let reference = crate::ifs::decode_iterative(&hdr, &leaves, DECODE_ITERATIONS);
        assert_eq!(
            decoded.image.as_slice(),
            reference.as_slice(),
            "layer-4 progressive decode must be bit-exact with decode_iterative on the \
             original leaves"
        );
    }

    /// Every one of the 4 prefix lengths must decode without error, and produce an image
    /// of the right dimensions — the brief's own "any prefix decodes to a valid image".
    #[test]
    fn every_prefix_length_decodes_without_error() {
        let image = textured_gradient_image(64, 64);
        let (hdr, leaves, _evals, _stats) = encode_image_rd(&image, &rd_params(20.0));
        let stream = encode(&image, &hdr, &leaves).unwrap();
        let offsets = layer_end_offsets(&stream).unwrap();

        for (i, &end) in offsets.iter().enumerate() {
            let prefix = &stream[..end];
            let decoded =
                decode(prefix).unwrap_or_else(|e| panic!("layer {} prefix failed: {e}", i + 1));
            // `>=`, not `==`: if a later layer's payload happens to be empty (e.g. no
            // mode-3 leaves at all, so layer 4 has zero residual events), a prefix that
            // ends exactly at an earlier boundary already contains that trailing empty
            // layer too -- correctly reported as the higher layer count, not a bug.
            assert!(
                decoded.layers as usize > i,
                "layer {} prefix reported only {} layers",
                i + 1,
                decoded.layers
            );
            assert_eq!(decoded.image.width(), image.width());
            assert_eq!(decoded.image.height(), image.height());
        }
    }

    /// A prefix that ends strictly *inside* a layer (not exactly on a boundary) must still
    /// decode, using whatever full layers came before it — never panic or silently corrupt.
    #[test]
    fn a_mid_layer_prefix_falls_back_to_the_last_complete_layer() {
        let image = textured_gradient_image(64, 64);
        let (hdr, leaves, _evals, _stats) = encode_image_rd(&image, &rd_params(20.0));
        let stream = encode(&image, &hdr, &leaves).unwrap();
        let offsets = layer_end_offsets(&stream).unwrap();

        // One byte short of the layer-3 boundary but past layer 2's: must fall back to 2.
        assert!(
            offsets[2] > offsets[1] + 1,
            "test needs a non-trivial layer 3"
        );
        let prefix = &stream[..offsets[2] - 1];
        let decoded = decode(prefix).unwrap();
        assert_eq!(decoded.layers, 2);
    }

    /// Sanity check on quality direction: PSNR should not get *worse* as more layers are
    /// revealed by more than a small numerical slack — not asserted as strict monotonicity
    /// (the brief explicitly warns strict per-layer monotonicity is an empirical question,
    /// not an a-priori requirement), but a large regression would indicate a real bug (e.g.
    /// a layer overwriting a leaf with garbage) rather than a genuine RD tradeoff.
    #[test]
    fn quality_does_not_regress_sharply_across_layers() {
        let image = textured_gradient_image(96, 96);
        let (hdr, leaves, _evals, _stats) = encode_image_rd(&image, &rd_params(20.0));
        let stream = encode(&image, &hdr, &leaves).unwrap();
        let offsets = layer_end_offsets(&stream).unwrap();

        let mut psnrs = Vec::new();
        for &end in &offsets {
            let decoded = decode(&stream[..end]).unwrap();
            psnrs.push(psnr(&image, &decoded.image).unwrap_or(0.0));
        }
        for w in psnrs.windows(2) {
            assert!(
                w[1] > w[0] - 0.5,
                "layer PSNRs regressed sharply: {psnrs:?}"
            );
        }
    }

    #[test]
    fn rejects_bad_magic_without_panicking() {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(b"NOPE");
        assert_eq!(decode(&bytes).unwrap_err(), ProgressiveError::BadMagic);
    }

    #[test]
    fn rejects_truncated_input_without_panicking() {
        assert_eq!(decode(&[]).unwrap_err(), ProgressiveError::Truncated);
        assert_eq!(decode(&MAGIC).unwrap_err(), ProgressiveError::Truncated);
    }

    #[test]
    fn rejects_oversized_layer_length_claims_without_panicking() {
        let image = textured_gradient_image(48, 48);
        let (hdr, leaves, _evals, _stats) = encode_image_rd(&image, &rd_params(20.0));
        let mut stream = encode(&image, &hdr, &leaves).unwrap();
        // Corrupt layer 1's declared length to claim far more than remains -- must be
        // treated as "layer 1 not fully present", never panic on an out-of-bounds slice.
        stream[15..19].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            decode(&stream).unwrap_err(),
            ProgressiveError::InsufficientData
        );
    }
}
