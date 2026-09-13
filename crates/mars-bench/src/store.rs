//! The append-only JSONL result store (§M7).
//!
//! Append-only is enforced here rather than by convention: the writer opens with
//! `append(true)` and offers no way to seek, truncate, or rewrite. §M7 — "nothing is
//! ever overwritten or edited" — is the one rule that makes every other number in the
//! project auditable after the fact.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::provenance::Provenance;

/// The result-row schema. `kind` says what the row is about and `data` carries the
/// payload, so new measurement types can be added without a schema migration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Row {
    /// Schema version of this row's envelope.
    pub schema: u32,
    /// What this row records, e.g. `quality`, `baseline-mars1`, `timing`.
    pub kind: String,
    pub provenance: Provenance,
    /// Row-kind-specific payload.
    pub data: serde_json::Value,
}

/// The current envelope schema version.
pub const SCHEMA: u32 = 1;

impl Row {
    pub fn new<T: Serialize>(
        kind: impl Into<String>,
        provenance: Provenance,
        data: &T,
    ) -> Result<Self, serde_json::Error> {
        Ok(Self {
            schema: SCHEMA,
            kind: kind.into(),
            provenance,
            data: serde_json::to_value(data)?,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}:{line}: malformed JSONL: {source}")]
    Parse {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("serialising row: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// An append-only JSONL file. There is deliberately no `write`, `truncate`, or
/// `overwrite` on this type.
pub struct ResultStore {
    path: PathBuf,
    file: File,
}

impl ResultStore {
    /// Open for appending, creating the file and parent directory if needed.
    pub fn open(path: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let path = path.into();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|source| StoreError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| StoreError::Io {
                path: path.clone(),
                source,
            })?;
        Ok(Self { path, file })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one row as a single line, flushed before returning.
    ///
    /// One JSON object per line with no pretty-printing: a partially written row from a
    /// crashed sweep then costs exactly one line, and `wc -l` is a row count.
    pub fn append(&mut self, row: &Row) -> Result<(), StoreError> {
        let mut line = serde_json::to_string(row)?;
        line.push('\n');
        self.file
            .write_all(line.as_bytes())
            .and_then(|()| self.file.flush())
            .map_err(|source| StoreError::Io {
                path: self.path.clone(),
                source,
            })
    }
}

/// Read every row, failing on the first malformed line rather than skipping it.
pub fn read_rows(path: &Path) -> Result<Vec<Row>, StoreError> {
    let file = File::open(path).map_err(|source| StoreError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut rows = Vec::new();
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|source| StoreError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(&line).map_err(|source| StoreError::Parse {
            path: path.to_path_buf(),
            line: i + 1,
            source,
        })?;
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_accumulate_and_round_trip() {
        let dir = std::env::temp_dir().join(format!("mars-store-{}", std::process::id()));
        let path = dir.join("t.jsonl");
        let _ = std::fs::remove_file(&path);

        let mut store = ResultStore::open(&path).unwrap();
        for i in 0..3u32 {
            let row = Row::new(
                "test",
                Provenance::detect(i),
                &serde_json::json!({ "i": i }),
            )
            .unwrap();
            store.append(&row).unwrap();
        }
        // Reopening must append, never truncate.
        let mut store = ResultStore::open(&path).unwrap();
        store
            .append(&Row::new("test", Provenance::detect(3), &serde_json::json!({"i": 3})).unwrap())
            .unwrap();

        let rows = read_rows(&path).unwrap();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].provenance.run_index, 0);
        assert_eq!(rows[3].data["i"], 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_malformed_line_is_an_error_not_a_skip() {
        let dir = std::env::temp_dir().join(format!("mars-store-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.jsonl");
        // A well-formed row, then garbage: the reader must report line 2 rather than
        // quietly returning the one row it could parse.
        let good = serde_json::to_string(
            &Row::new("test", Provenance::detect(0), &serde_json::json!({"i": 0})).unwrap(),
        )
        .unwrap();
        std::fs::write(&path, format!("{good}\nnot json\n")).unwrap();
        assert!(matches!(
            read_rows(&path),
            Err(StoreError::Parse { line: 2, .. })
        ));
        std::fs::remove_dir_all(&dir).ok();
    }
}
