//! Command-line front ends. `marsbench` today; `encmars`/`decmars` at Step 6 and beyond.
//!
//! [`human`] holds the human-adaptive encoding support (face/facial-feature regions) and
//! [`scrfd`] the SCRFD face detection behind `encmars --human-adaptive`.

pub mod human;
pub mod scrfd;
