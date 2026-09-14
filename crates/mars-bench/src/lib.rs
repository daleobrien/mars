//! **The harness.** Built first, before any Mars 2 codec code exists (§0).
//!
//! `mars-bench` owns the things that make a number a result: the metric orchestration
//! (§M1), BD-rate with its mandatory interval (§M3), the provenance block (§M7), and the
//! append-only result store (§M7). The metric mathematics itself lives in `mars-core`,
//! so that exactly one implementation exists.

pub mod anchors;
pub mod anchors_report;
pub mod bdrate;
pub mod gate_c;
pub mod gpu_search;
pub mod ifs_check;
pub mod mars1;
pub mod mars1_fixtures;
pub mod mars1_report;
pub mod mars_format_gate;
pub mod measure;
pub mod mode_gate;
pub mod oracle;
pub mod parallel_bench;
pub mod pchip;
pub mod provenance;
pub mod recall;
pub mod report;
pub mod rd_opt;
pub mod rust_encoder;
pub mod simd_bench;
pub mod store;
pub mod sweep;

pub use bdrate::{bd_metrics, BdResult, RdCurve, RdPoint};
pub use mars1::{
    decode as mars1_decode, encode as mars1_encode, DecodeMode, DecodeParams, EncodeParams,
    Mars1Binaries, Method,
};
pub use measure::{measure, MeasureRequest, Measurement};
pub use provenance::{Machine, Provenance, HARNESS_VERSION};
pub use report::Report;
pub use store::{read_rows, ResultStore, Row};
