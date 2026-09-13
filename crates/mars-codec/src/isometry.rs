//! §9's eight isometries, shared by the decoder ([`crate::ifs`]) and the encoder's
//! exhaustive search ([`crate::encode`]) — both need the same destination `(i, j)` for a
//! source `(u, v)`, one to reconstruct pixels, the other to correlate a domain block
//! against a range block before committing to an isometry.

/// §9: destination `(i, j)` for source `(u, v)`, by isometry code (0-7).
pub fn map(k: u8, u: usize, v: usize, size: usize) -> (usize, usize) {
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

/// All eight isometry codes, in order — for iterating the exhaustive search.
pub const ALL: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
