//! The two fixed lookup tables `reference/mars1/globals.h` compiles in verbatim:
//! `ordering` (the 24 permutations of 4 quadrant-rank descending orders, each row
//! carrying the isometry/class pair `newclass` reads off it) and `mapping` (the 8x8
//! isometry-composition table Fisher and Saupe-Fisher use to combine "isometry that
//! canonicalises the range" with "isometry stored with the domain").
//!
//! Both are copied byte-for-byte from `globals.h` — this is data, not logic, so there is
//! nothing to port beyond transcription.

/// `ordering[j] = [a, b, c, d, isom, clas]`: row `j` is the descending-quadrant-order
/// permutation `(a,b,c,d)` of `{0,1,2,3}`, paired with the isometry and one-of-3 class
/// `newclass` returns when a block's own quadrant order matches this row.
pub const ORDERING: [[i32; 6]; 24] = [
    [3, 2, 1, 0, 3, 0],
    [2, 3, 1, 0, 5, 1],
    [2, 1, 3, 0, 5, 2],
    [3, 1, 2, 0, 7, 0],
    [1, 3, 2, 0, 1, 1],
    [1, 2, 3, 0, 1, 2],
    [3, 2, 0, 1, 3, 1],
    [2, 3, 0, 1, 5, 0],
    [2, 1, 0, 3, 2, 2],
    [3, 1, 0, 2, 7, 1],
    [1, 3, 0, 2, 1, 0],
    [1, 2, 0, 3, 4, 2],
    [3, 0, 2, 1, 3, 2],
    [2, 0, 3, 1, 2, 0],
    [2, 0, 1, 3, 2, 1],
    [3, 0, 1, 2, 7, 2],
    [1, 0, 2, 3, 4, 1],
    [1, 0, 3, 2, 4, 0],
    [0, 3, 2, 1, 6, 2],
    [0, 2, 3, 1, 6, 1],
    [0, 2, 1, 3, 6, 0],
    [0, 3, 1, 2, 0, 2],
    [0, 1, 3, 2, 0, 1],
    [0, 1, 2, 3, 0, 0],
];

/// `mapping[a][b]`: the isometry to apply when composing "isometry `a` that canonicalises
/// the range" with "isometry `b` stored with the domain" (Fisher/Saupe-Fisher).
pub const MAPPING: [[u8; 8]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [2, 0, 3, 1, 7, 6, 4, 5],
    [1, 3, 0, 2, 6, 7, 5, 4],
    [3, 2, 1, 0, 5, 4, 7, 6],
    [4, 7, 6, 5, 0, 3, 2, 1],
    [5, 6, 7, 4, 3, 0, 1, 2],
    [6, 4, 5, 7, 1, 2, 0, 3],
    [7, 5, 4, 6, 2, 1, 3, 0],
];

/// Find the row of [`ORDERING`] whose first four entries equal `order` — `match()` in
/// `index_func.c`. Panics if none matches, exactly as the C `fatal()` call does: `order`
/// is always a permutation of `{0,1,2,3}` by construction, so all 24 rows are reachable
/// and a non-match means a caller bug, not bad input.
pub fn match_order(order: [i32; 4]) -> usize {
    ORDERING
        .iter()
        .position(|row| row[0..4] == order)
        .expect("order is always a permutation of 0..=3, which ORDERING covers exhaustively")
}
