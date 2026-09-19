//! Command-line front ends. `marsbench` today; `encmars`/`decmars` at Step 6 and beyond.
//!
//! [`human`] holds the human-adaptive encoding support (face/facial-feature regions),
//! [`scrfd`] the SCRFD face detection behind `encmars --human-adaptive`, and [`overlay`]
//! the `--debug-regions` visualisation of the regions it finds.

pub mod human;
pub mod overlay;
pub mod scrfd;
