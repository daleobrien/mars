//! `marsbench recall` — top-k recall and RMS regret against the oracle (Step 8).
//!
//! Step 9's actual search methods (`CandidateRetriever`) don't exist yet, so this module
//! only defines and consumes a documented input format for "a method's chosen
//! candidates" — a method-agnostic contract Step 9 can produce for real once it exists.
//! For now it can be generated from the oracle's own top-1 pick (see
//! `oracle_top1_as_picks` below), which is exactly how this module's own tests and
//! `marsbench recall`'s smoke-test path exercise it.
//!
//! **Input format: JSONL, one [`MethodPick`] object per line.** Plain JSON objects
//! (`serde_json`) rather than CSV, since this project's other append-only data
//! (`results/*.jsonl`) is already JSONL and every language a future `CandidateRetriever`
//! might be prototyped in has a JSON encoder. Fields: `row, col, size, dom_row, dom_col,
//! isometry, qalfa, qbeta, rms` — `(row, col, size)` locates the range block (must be
//! grid-aligned per the oracle's own grid for that size, see
//! `oracle::SizeCache::block_at`), and the rest is the method's chosen encoding for it.
//!
//! **What "recall" means here.** Per the brief: "match on the discrete tuple, not on
//! `rms` proximity." A pick's `(dom_row, dom_col, isometry, qalfa, qbeta)` must appear
//! *exactly* at rank r among the oracle's up-to-32 candidates for top-r recall to count
//! it, at whatever rank it actually occupies -- finding a different domain/isometry that
//! happens to fit almost as well is not recall, it is a separate (regret) measurement.

use std::io::BufRead;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::oracle::OracleCache;

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct MethodPick {
    pub row: u32,
    pub col: u32,
    pub size: u32,
    pub dom_row: u32,
    pub dom_col: u32,
    pub isometry: u8,
    pub qalfa: u32,
    pub qbeta: u32,
    pub rms: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum RecallError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{path} line {line}: {source}")]
    Json {
        path: String,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
}

/// Reads a `MethodPick` JSONL file. Blank lines are skipped (a common artefact of
/// hand-edited or `join`-generated fixtures); every non-blank line must parse.
pub fn read_picks(path: &Path) -> Result<Vec<MethodPick>, RecallError> {
    let text = std::fs::read_to_string(path).map_err(|source| RecallError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let pick: MethodPick = serde_json::from_str(line).map_err(|source| RecallError::Json {
            path: path.display().to_string(),
            line: i + 1,
            source,
        })?;
        out.push(pick);
    }
    Ok(out)
}

pub fn write_picks(path: &Path, picks: &[MethodPick]) -> Result<(), RecallError> {
    use std::fmt::Write as _;
    let mut out = String::new();
    for p in picks {
        let line = serde_json::to_string(p).expect("MethodPick always serializes");
        writeln!(out, "{line}").expect("String write is infallible");
    }
    std::fs::write(path, out).map_err(|source| RecallError::Io {
        path: path.display().to_string(),
        source,
    })
}

/// Generates a `MethodPick` list from the oracle's own top-1 (rank-0) entry at every
/// block of the given size. This is deliberately the "method" that must score 100%
/// top-1 recall / 0 dB regret against its own oracle -- the harness self-test this
/// module's own tests use, and a template for what Step 9's real methods will produce.
pub fn oracle_top1_as_picks(cache: &OracleCache, size: u32) -> Vec<MethodPick> {
    let Some(sc) = cache.size(size) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for by in 0..sc.num_blocks_y {
        for bx in 0..sc.num_blocks_x {
            let block = &sc.blocks[(by * sc.num_blocks_x + bx) as usize];
            let best = block[0];
            if !best.valid {
                continue;
            }
            out.push(MethodPick {
                row: by * size,
                col: bx * size,
                size,
                dom_row: best.dom_row,
                dom_col: best.dom_col,
                isometry: best.isometry,
                qalfa: best.qalfa,
                qbeta: best.qbeta,
                rms: f64::from(best.rms),
            });
        }
    }
    out
}

/// `regret_db = 20 * log10(rms_method / rms_oracle_best)`, exactly
/// `psnr_oracle_best - psnr_method` under this project's pinned PSNR convention
/// (`mars_core::metrics::psnr_from_mse`'s `20*log10(255/rms)` form). `None` when either
/// `rms` is not strictly positive -- a flat block can have an exact-zero `rms`, and
/// `log10(0)` or `0/0` must never reach a reported statistic (mirrors
/// `psnr_from_mse`'s own `(mse > 0.0).then(...)` guard).
pub fn regret_db(rms_method: f64, rms_oracle_best: f64) -> Option<f64> {
    // Explicit NaN check rather than `!(x > 0.0)`: clippy's `neg_cmp_op_on_partial_ord`
    // flags the negated form, but a plain `<= 0.0` would silently let a NaN `rms` (which
    // should never happen, but this guard exists precisely for the "should never happen"
    // case) through as "not <= 0.0" and on into `log10`.
    if rms_oracle_best.is_nan()
        || rms_method.is_nan()
        || rms_oracle_best <= 0.0
        || rms_method <= 0.0
    {
        return None;
    }
    Some(20.0 * (rms_method / rms_oracle_best).log10())
}

/// Linear-interpolation percentile over an already-sorted ascending slice (the "common
/// convention" percentile, matching e.g. numpy's default). Neither `bdrate.rs` nor
/// `report.rs` has a percentile helper today (checked before adding this one) -- this is
/// intentionally small and local to the one report that needs it.
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    assert!(!sorted.is_empty(), "percentile of an empty sample");
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RegretStats {
    pub mean_db: f64,
    pub median_db: f64,
    pub p95_db: f64,
    pub n: usize,
}

#[derive(Debug, Clone, Default)]
pub struct RecallReport {
    /// Picks whose `(row, col, size)` located a real oracle block (the denominator for
    /// every rate below). A pick that names a block the oracle has no data for (out of
    /// range, or the oracle found no domain position at all) is counted separately and
    /// excluded, since recall/regret are undefined for it, not zero.
    pub matched: usize,
    pub unmatched: usize,
    pub top1: usize,
    pub top5: usize,
    pub top32: usize,
    /// One entry per matched pick with both a valid oracle best and a valid method rms
    /// (see [`regret_db`]'s guard) -- kept, not just summarised, so [`RecallReport::regret_stats`]
    /// and any future reporting can share the same underlying sample.
    pub regret_db_values: Vec<f64>,
}

impl RecallReport {
    pub fn top1_rate(&self) -> f64 {
        rate(self.top1, self.matched)
    }
    pub fn top5_rate(&self) -> f64 {
        rate(self.top5, self.matched)
    }
    pub fn top32_rate(&self) -> f64 {
        rate(self.top32, self.matched)
    }

    pub fn regret_stats(&self) -> Option<RegretStats> {
        if self.regret_db_values.is_empty() {
            return None;
        }
        let mut sorted = self.regret_db_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;
        Some(RegretStats {
            mean_db: mean,
            median_db: percentile_sorted(&sorted, 50.0),
            p95_db: percentile_sorted(&sorted, 95.0),
            n,
        })
    }
}

fn rate(hits: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        hits as f64 / total as f64
    }
}

/// Scores `picks` against `cache`: for each pick, locates its oracle block at the same
/// size, checks whether the pick's discrete tuple appears in the top-1/5/32 of that
/// block's cached list (exact-tuple match, not `rms` proximity -- module doc), and
/// records the RMS regret against that block's rank-0 entry.
pub fn score(cache: &OracleCache, picks: &[MethodPick]) -> RecallReport {
    let mut report = RecallReport::default();
    for pick in picks {
        let Some(sc) = cache.size(pick.size) else {
            report.unmatched += 1;
            continue;
        };
        let Some(block) = sc.block_at(pick.row, pick.col) else {
            report.unmatched += 1;
            continue;
        };
        let oracle_best = block[0];
        if !oracle_best.valid {
            report.unmatched += 1;
            continue;
        }
        report.matched += 1;

        if let Some(rank) = block.iter().position(|c| {
            c.valid
                && c.dom_row == pick.dom_row
                && c.dom_col == pick.dom_col
                && c.isometry == pick.isometry
                && c.qalfa == pick.qalfa
                && c.qbeta == pick.qbeta
        }) {
            if rank == 0 {
                report.top1 += 1;
            }
            if rank < 5 {
                report.top5 += 1;
            }
            if rank < 32 {
                report.top32 += 1;
            }
        }

        if let Some(r) = regret_db(pick.rms, f64::from(oracle_best.rms)) {
            report.regret_db_values.push(r);
        }
    }
    report
}

/// Streaming convenience for a picks file too large to want fully materialised twice —
/// reads and scores in one pass. Used by `marsbench recall`; `read_picks`/`score` stay
/// available separately for tests and for programmatic use (e.g. `oracle_top1_as_picks`
/// output scored in-process without a round trip through disk).
pub fn score_file(cache: &OracleCache, path: &Path) -> Result<RecallReport, RecallError> {
    let f = std::fs::File::open(path).map_err(|source| RecallError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let reader = std::io::BufReader::new(f);
    let mut picks = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| RecallError::Io {
            path: path.display().to_string(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let pick: MethodPick = serde_json::from_str(&line).map_err(|source| RecallError::Json {
            path: path.display().to_string(),
            line: i + 1,
            source,
        })?;
        picks.push(pick);
    }
    Ok(score(cache, &picks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::{CacheSlot, OracleConfig, SizeCache};

    fn slot(dom_row: u32, isometry: u8, rms: f32) -> CacheSlot {
        CacheSlot {
            valid: true,
            dom_row,
            dom_col: 0,
            isometry,
            qalfa: 1,
            qbeta: 2,
            rms,
        }
    }

    fn cache_with_one_block(entries: &[CacheSlot]) -> OracleCache {
        let mut block = [CacheSlot::INVALID; 32];
        for (i, e) in entries.iter().enumerate() {
            block[i] = *e;
        }
        OracleCache {
            image_name: "t".into(),
            image_sha256_hex: "h".into(),
            config: OracleConfig {
                min_size: 8,
                max_size: 8,
                shift: 4,
                bits_alfa: 4,
                bits_beta: 7,
                max_alfa: 1.0,
            },
            sizes: vec![SizeCache {
                size: 8,
                num_blocks_x: 1,
                num_blocks_y: 1,
                blocks: vec![block],
            }],
        }
    }

    #[test]
    fn oracle_top1_as_picks_scores_perfect_recall_and_zero_regret() {
        let cache = cache_with_one_block(&[slot(4, 0, 1.5), slot(8, 1, 2.0)]);
        let picks = oracle_top1_as_picks(&cache, 8);
        assert_eq!(picks.len(), 1);
        let report = score(&cache, &picks);
        assert_eq!(report.matched, 1);
        assert_eq!(report.top1_rate(), 1.0);
        assert_eq!(report.top5_rate(), 1.0);
        assert_eq!(report.top32_rate(), 1.0);
        let stats = report.regret_stats().unwrap();
        assert!(stats.mean_db.abs() < 1e-9, "self-match must be 0 dB regret");
    }

    #[test]
    fn a_rank2_pick_counts_toward_top5_and_top32_but_not_top1() {
        let cache = cache_with_one_block(&[
            slot(4, 0, 1.0),
            slot(8, 0, 1.1),
            slot(12, 0, 1.2), // the pick below matches this, rank 2
        ]);
        let pick = MethodPick {
            row: 0,
            col: 0,
            size: 8,
            dom_row: 12,
            dom_col: 0,
            isometry: 0,
            qalfa: 1,
            qbeta: 2,
            rms: 1.2,
        };
        let report = score(&cache, &[pick]);
        assert_eq!(report.top1, 0);
        assert_eq!(report.top5, 1);
        assert_eq!(report.top32, 1);
    }

    #[test]
    fn a_pick_matching_no_candidate_counts_toward_no_recall_bucket_but_still_gets_regret() {
        let cache = cache_with_one_block(&[slot(4, 0, 1.0)]);
        let pick = MethodPick {
            row: 0,
            col: 0,
            size: 8,
            dom_row: 999,
            dom_col: 0,
            isometry: 0,
            qalfa: 1,
            qbeta: 2,
            rms: 2.0,
        };
        let report = score(&cache, &[pick]);
        assert_eq!(report.matched, 1);
        assert_eq!(report.top1, 0);
        assert_eq!(report.top5, 0);
        assert_eq!(report.top32, 0);
        let stats = report.regret_stats().unwrap();
        assert!((stats.mean_db - 20.0 * (2.0f64 / 1.0).log10()).abs() < 1e-9);
    }

    #[test]
    fn regret_db_guards_zero_rms_on_either_side() {
        assert_eq!(regret_db(1.0, 0.0), None, "oracle rms == 0 must be guarded");
        assert_eq!(regret_db(0.0, 1.0), None, "method rms == 0 must be guarded");
        assert!(
            regret_db(2.0, 1.0).unwrap() > 0.0,
            "worse than oracle is positive dB regret"
        );
    }

    #[test]
    fn percentile_matches_hand_computed_values() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_sorted(&v, 0.0), 1.0);
        assert_eq!(percentile_sorted(&v, 100.0), 5.0);
        assert_eq!(percentile_sorted(&v, 50.0), 3.0);
    }

    #[test]
    fn picks_round_trip_through_jsonl() {
        let dir = std::env::temp_dir().join(format!("mars-recall-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("picks.jsonl");
        let picks = vec![MethodPick {
            row: 0,
            col: 8,
            size: 8,
            dom_row: 4,
            dom_col: 0,
            isometry: 2,
            qalfa: 3,
            qbeta: 10,
            rms: 1.25,
        }];
        write_picks(&path, &picks).unwrap();
        let back = read_picks(&path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].row, 0);
        assert_eq!(back[0].col, 8);
        assert!((back[0].rms - 1.25).abs() < 1e-12);
        std::fs::remove_dir_all(&dir).ok();
    }
}
