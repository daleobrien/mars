//! Partition, search, fit, encode, decode, bitstream.
//!
//! Group B (Step 5) begins here with a read-only `.ifs` parser and a minimal iterative
//! decoder — see [`ifs`]. Nothing here writes a Mars 1 bitstream or implements Mars 2's
//! own format; those arrive at Steps 6 and 10 respectively.

pub mod ifs;
