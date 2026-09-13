//! Mars 2 core: image types, image IO, and **the** implementation of every quality
//! metric the project reports (§M1 of the measurement contract).
//!
//! Nothing in this crate knows anything about fractal coding. That is deliberate:
//! the metrics must be computable from two files on disk and nothing else.

pub mod image;
pub mod io;
pub mod metrics;
pub mod tolerance;

pub use crate::image::{ColorSpace, Image, Plane};
pub use crate::io::{read_image, ImageError};
