//! The provenance block (§M7).
//!
//! Every result row carries enough to regenerate itself. §M7: "a result you cannot
//! regenerate from its own row is not a result." The machine fingerprint is part of
//! that, because §M4 makes a speedup figure without one not a result either.

use std::process::Command;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Harness schema version. Bump when the row shape changes; readers key off it.
pub const HARNESS_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub harness_version: String,
    pub git_sha: String,
    /// True if the working tree had uncommitted changes. A dirty row is still recorded —
    /// suppressing it would just mean people stop recording.
    pub git_dirty: bool,
    pub build_profile: String,
    pub timestamp_utc: String,
    /// SHA-256 over the corpus manifest, so a changed image set is visible.
    pub corpus_manifest_sha256: Option<String>,
    /// SHA-256 over the canonical JSON of the parameter set.
    pub parameter_set_sha256: Option<String>,
    pub run_index: u32,
    pub machine: Machine,
}

/// §M4's machine fingerprint. "M3" alone is not a fingerprint: M3 / M3 Pro / M3 Max span
/// 8 to 16 CPU cores and 10 to 40 GPU cores, so a timing number without the variant is
/// not comparable to the next one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Machine {
    pub os: String,
    pub os_build: String,
    pub arch: String,
    /// Exact chip string, e.g. "Apple M3 Pro".
    pub cpu_brand: String,
    pub physical_cores: Option<u32>,
    /// Performance-core count. §M4 requires compute benchmarks to pin thread counts to
    /// this rather than to `num_cpus`, which includes efficiency cores and bends the
    /// scaling curve for scheduling reasons rather than algorithmic ones.
    pub p_cores: Option<u32>,
    pub e_cores: Option<u32>,
    pub memory_bytes: Option<u64>,
    pub rustc_version: String,
    /// Free-text: whether the machine was otherwise idle, on mains power, etc. §M4
    /// requires it to be stated; the harness cannot detect it, so the operator declares it.
    pub conditions: Option<String>,
}

fn cmd(bin: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(bin).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn sysctl_u32(key: &str) -> Option<u32> {
    cmd("sysctl", &["-n", key])?.parse().ok()
}

fn sysctl_u64(key: &str) -> Option<u64> {
    cmd("sysctl", &["-n", key])?.parse().ok()
}

impl Machine {
    pub fn detect() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            os_build: cmd("uname", &["-r"]).unwrap_or_default(),
            arch: std::env::consts::ARCH.to_string(),
            cpu_brand: cmd("sysctl", &["-n", "machdep.cpu.brand_string"]).unwrap_or_default(),
            physical_cores: sysctl_u32("hw.physicalcpu"),
            p_cores: sysctl_u32("hw.perflevel0.physicalcpu"),
            e_cores: sysctl_u32("hw.perflevel1.physicalcpu"),
            memory_bytes: sysctl_u64("hw.memsize"),
            rustc_version: option_env!("MARS_RUSTC_VERSION")
                .unwrap_or("unknown")
                .to_string(),
            conditions: std::env::var("MARS_BENCH_CONDITIONS").ok(),
        }
    }
}

impl Provenance {
    /// Collect everything detectable. `run_index` distinguishes repeats within a sweep.
    pub fn detect(run_index: u32) -> Self {
        let git_sha = cmd("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
        let git_dirty = cmd("git", &["status", "--porcelain", "--untracked-files=no"])
            .map(|s| !s.is_empty())
            .unwrap_or(true);
        Self {
            harness_version: HARNESS_VERSION.to_string(),
            git_sha,
            git_dirty,
            build_profile: if cfg!(debug_assertions) {
                "debug".into()
            } else {
                "release".into()
            },
            timestamp_utc: now_iso8601(),
            corpus_manifest_sha256: None,
            parameter_set_sha256: None,
            run_index,
            machine: Machine::detect(),
        }
    }

    pub fn with_corpus_manifest(mut self, manifest_bytes: &[u8]) -> Self {
        self.corpus_manifest_sha256 = Some(sha256_hex(manifest_bytes));
        self
    }

    /// Hash the parameter set from its canonical (key-sorted) JSON, so that a
    /// semantically identical set hashes identically regardless of field order.
    pub fn with_parameter_set<T: Serialize>(mut self, params: &T) -> Self {
        let value = serde_json::to_value(params).expect("parameter set must serialise");
        self.parameter_set_sha256 = Some(sha256_hex(canonical_json(&value).as_bytes()));
        self
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// JSON with object keys sorted recursively, so hashing is stable.
pub fn canonical_json(v: &serde_json::Value) -> String {
    use serde_json::Value;
    match v {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .iter()
                .map(|k| {
                    format!(
                        "{}:{}",
                        Value::String((*k).clone()),
                        canonical_json(&map[*k])
                    )
                })
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

/// ISO-8601 UTC to whole seconds, computed from the epoch directly so the harness needs
/// no date-time dependency.
pub fn now_iso8601() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    format_epoch_utc(secs)
}

fn format_epoch_utc(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Howard Hinnant's `civil_from_days`, the standard branch-free epoch-to-date algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_dates_are_right() {
        assert_eq!(format_epoch_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_epoch_utc(1_000_000_000), "2001-09-09T01:46:40Z");
        // 2026-09-13T00:00:00Z
        assert_eq!(format_epoch_utc(1_789_257_600), "2026-09-13T00:00:00Z");
        // A leap day, which is where a hand-rolled calendar usually breaks.
        assert_eq!(format_epoch_utc(1_709_164_800), "2024-02-29T00:00:00Z");
    }

    #[test]
    fn canonical_json_is_key_order_independent() {
        let a: serde_json::Value =
            serde_json::from_str(r#"{"b":1,"a":{"d":2,"c":[3,4]}}"#).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(r#"{"a":{"c":[3,4],"d":2},"b":1}"#).unwrap();
        assert_eq!(canonical_json(&a), canonical_json(&b));
        // ...but not value-order independent: arrays are ordered data.
        let c: serde_json::Value =
            serde_json::from_str(r#"{"a":{"c":[4,3],"d":2},"b":1}"#).unwrap();
        assert_ne!(canonical_json(&a), canonical_json(&c));
    }

    #[test]
    fn sha256_matches_the_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
