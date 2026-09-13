//! Every numeric tolerance in the project, declared once (§A2).
//!
//! Tolerances live here and **only** here so that widening one is a visible diff in a
//! file whose entire purpose is to be watched, rather than a character change buried in
//! a test. Changing any constant in this file requires a `CONTRACT-CHANGE:` trailer on
//! the commit, enforced by `scripts/check-contract.sh`.

/// Hand-computed MSE known-answer vectors must match to this absolute tolerance (Step 1).
pub const MSE_KNOWN_ANSWER_ABS: f64 = 1e-9;

/// `SSIM(x, x)` must equal 1.0 to this absolute tolerance.
pub const SSIM_SELF_ABS: f64 = 1e-12;

/// PSNR cross-validation against an independent implementation, in dB (Step 1).
pub const PSNR_CROSSVAL_DB: f64 = 0.01;

/// SSIM cross-validation against an independent implementation (Step 1).
pub const SSIM_CROSSVAL_ABS: f64 = 0.001;

/// MS-SSIM cross-validation against an independent implementation (Step 1).
pub const MSSSIM_CROSSVAL_ABS: f64 = 0.001;

/// BD-rate of a curve against itself must be 0 to this tolerance, in percent.
pub const BDRATE_SELF_PCT: f64 = 1e-6;

/// BD-rate of an analytically shifted curve must match the closed form, in percent.
pub const BDRATE_ANALYTIC_PCT: f64 = 1e-6;
