//! Partition, search, fit, encode, decode, bitstream.
//!
//! Group B (Step 5) began here with a read-only `.ifs` parser and a minimal iterative
//! decoder — see [`ifs`]. Step 6 adds the exhaustive encoder ([`encode`]) and a writer
//! (`ifs::write`), both cross-checked against `decmars`. Nothing here implements Mars 2's
//! own format yet; that arrives at Step 10.

pub mod color;
pub mod dct;
pub mod encode;
pub mod ifs;
pub mod isometry;
pub mod mars_format;
pub mod postprocess;
pub mod progressive;
pub mod quant;
pub mod rate;
pub mod residual;
