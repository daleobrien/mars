//! NEON kernels for `mars-codec`'s exhaustive-search moment accumulation -- Step 11.
//!
//! Targets NEON directly, no runtime dispatch: NEON is unconditional on aarch64, and
//! `implementation-plan.md` §2.2 commits this project to one machine, Apple Silicon.
//! Every kernel here has an exact-equality differential test against a scalar reference
//! (see `moments`'s module doc for why "exact," not an epsilon, is the right bar for
//! integer moment sums).

pub mod moments;
