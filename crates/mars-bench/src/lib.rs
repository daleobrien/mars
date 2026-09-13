//! **The harness.** Built first, before any Mars 2 codec code exists (§0).
//!
//! `mars-bench` owns the things that make a number a result: the metric orchestration
//! (§M1), BD-rate with its mandatory interval (§M3), the provenance block (§M7), and the
//! append-only result store (§M7). The metric mathematics itself lives in `mars-core`,
//! so that exactly one implementation exists.

pub mod bdrate;
pub mod measure;
pub mod pchip;
pub mod provenance;
pub mod report;
pub mod store;

pub use bdrate::{bd_metrics, BdResult, RdCurve, RdPoint};
pub use measure::{measure, MeasureRequest, Measurement};
pub use provenance::{Machine, Provenance, HARNESS_VERSION};
pub use report::Report;
pub use store::{read_rows, ResultStore, Row};
