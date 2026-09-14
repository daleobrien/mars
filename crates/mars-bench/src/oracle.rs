//! The oracle cache — Step 8 (§M6).
//!
//! For each `(image, config)` pair this module runs `mars-gpu`'s top-32 exhaustive
//! search (`GpuSearcher::search_top32`, Step 8's kernel — see `search.wgsl`'s module
//! doc) once per configured block size and persists the result to a compact binary file
//! under `oracle-cache/`. That cache is the project's single most reusable asset (§M6):
//! it is both the RD upper bound for fractal-only mode and the ground truth
//! `marsbench recall` measures every future search method against.
//!
//! **Binary format, not `bincode`.** This project hand-rolls its binary formats
//! elsewhere (`crates/mars-codec/src/ifs.rs`'s bit-level `.ifs` reader/writer) rather
//! than depending on a generic serialization crate, so the cache file here follows the
//! same practice: explicit little-endian fields written and read by hand
//! (`to_le_bytes`/`from_le_bytes`), no new workspace dependency. Every record is a fixed
//! size, so the format is a flat header followed by flat arrays — no variable-length
//! framing inside the hot data.
//!
//! **Manifest hash — the load-bearing part of this module.** `docs/decisions.md`'s own
//! framing (and this module's brief) is blunt about the risk: "silent stale-cache reuse
//! is a textbook agent failure mode." [`manifest_hash`] hashes together the image's own
//! content hash (already tracked per-image in `corpus/standard.images.json`), every
//! config parameter that changes what the search computes, and [`CACHE_FORMAT_VERSION`]
//! (bumped whenever this module's on-disk layout changes incompatibly). [`load`] never
//! silently rebuilds or ignores a mismatch — it hands the caller a
//! [`CacheError::Stale`] with a full diagnostic and lets the caller decide (per the
//! brief: "error loudly ... so a caller decides whether to rebuild rather than the
//! harness masking staleness").

use std::io::{self, Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use mars_core::Plane;
use mars_gpu::{GpuCandidate, GpuSearchParams, GpuSearcher};

/// Bumped whenever the on-disk layout in [`serialize`]/[`parse`] changes incompatibly.
/// Folded into [`manifest_hash`] so a stale binary layout is caught the same way a stale
/// image or config is.
pub const CACHE_FORMAT_VERSION: u32 = 1;

const MAGIC: &[u8; 8] = b"MARSORC1";

/// The `(min_size, max_size, SHIFT, bits_alfa, bits_beta, max_alfa)` tuple M6 names as
/// the oracle's config axes. `max_alfa` is stored as `f64` (matching
/// `mars_codec::encode::EncodeParams`) even though the GPU search itself runs in `f32`
/// (Metal has no fp64, §2.2) — the config value is what a caller supplies and compares
/// against, not a GPU-internal precision detail.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct OracleConfig {
    pub min_size: u32,
    pub max_size: u32,
    pub shift: u32,
    pub bits_alfa: u32,
    pub bits_beta: u32,
    pub max_alfa: f64,
}

impl OracleConfig {
    /// Every power-of-two block size in `[min_size, max_size]`, ascending — the sizes
    /// the encoder's quadtree partition can emit a leaf at, and therefore every size the
    /// oracle must cover for this config (P8.1).
    pub fn sizes(&self) -> Vec<u32> {
        let mut sizes = Vec::new();
        let mut s = self.min_size;
        while s <= self.max_size {
            sizes.push(s);
            s = s.saturating_mul(2);
            if s == 0 {
                break;
            }
        }
        sizes
    }

    fn gpu_params(&self, size: u32) -> GpuSearchParams {
        GpuSearchParams {
            size,
            shift: self.shift,
            bits_alfa: self.bits_alfa,
            bits_beta: self.bits_beta,
            max_alfa: self.max_alfa as f32,
        }
    }
}

/// One labelled config, as `configs/oracle.json` lists them — the same
/// label/description-plus-fields shape `mars1_fixtures::FixtureConfig`'s `Variant` and
/// `sweep::Variant` already use for a labelled config list.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OracleConfigEntry {
    pub label: String,
    pub description: String,
    #[serde(flatten)]
    pub config: OracleConfig,
}

/// `configs/oracle.json`'s top-level shape.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OracleSuite {
    pub name: String,
    pub description: String,
    /// Image-set indexes to draw from, e.g. `corpus/standard.images.json`.
    pub indexes: Vec<std::path::PathBuf>,
    pub configs: Vec<OracleConfigEntry>,
}

impl OracleSuite {
    pub fn read(path: &Path) -> Result<Self, CacheError> {
        let text = std::fs::read_to_string(path).map_err(|source| CacheError::Io {
            path: path.display().to_string(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| CacheError::Config {
            path: path.display().to_string(),
            source,
        })
    }
}

/// Hashes together everything that determines what a cache file's bytes *should* be:
/// the image's own content hash, every parameter the search itself depends on, and the
/// cache's own binary layout version. Two calls with equal arguments always produce the
/// same digest (this is exercised directly in the unit tests below, not only through a
/// round-tripped file), and changing any single argument changes the digest — that is
/// the entire correctness property a "stale cache" check needs.
pub fn manifest_hash(image_sha256_hex: &str, config: &OracleConfig) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(image_sha256_hex.as_bytes());
    h.update(CACHE_FORMAT_VERSION.to_le_bytes());
    h.update(config.min_size.to_le_bytes());
    h.update(config.max_size.to_le_bytes());
    h.update(config.shift.to_le_bytes());
    h.update(config.bits_alfa.to_le_bytes());
    h.update(config.bits_beta.to_le_bytes());
    h.update(config.max_alfa.to_bits().to_le_bytes());
    h.finalize().into()
}

/// One range block's cached candidate, the same shape as `mars_gpu::GpuCandidate` plus
/// the `valid` flag `search_top32` folds into `None`/short `Vec` at the API boundary —
/// kept explicit here because a cache record slot may be padding (fewer than 32 real
/// candidates existed for this block, e.g. an unrealistically small domain pool).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheSlot {
    pub valid: bool,
    pub dom_row: u32,
    pub dom_col: u32,
    pub isometry: u8,
    pub qalfa: u32,
    pub qbeta: u32,
    pub rms: f32,
}

impl CacheSlot {
    pub const INVALID: CacheSlot = CacheSlot {
        valid: false,
        dom_row: 0,
        dom_col: 0,
        isometry: 0,
        qalfa: 0,
        qbeta: 0,
        rms: 0.0,
    };

    const RECORD_LEN: usize = 4 * 7; // 7 u32-sized fields, see (de)serialize below

    fn serialize(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.valid as u32).to_le_bytes());
        out.extend_from_slice(&self.dom_row.to_le_bytes());
        out.extend_from_slice(&self.dom_col.to_le_bytes());
        out.extend_from_slice(&u32::from(self.isometry).to_le_bytes());
        out.extend_from_slice(&self.qalfa.to_le_bytes());
        out.extend_from_slice(&self.qbeta.to_le_bytes());
        out.extend_from_slice(&self.rms.to_bits().to_le_bytes());
    }

    fn parse(bytes: &[u8]) -> Self {
        let u = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        CacheSlot {
            valid: u(0) != 0,
            dom_row: u(1),
            dom_col: u(2),
            isometry: u(3) as u8,
            qalfa: u(4),
            qbeta: u(5),
            rms: f32::from_bits(u(6)),
        }
    }
}

impl From<GpuCandidate> for CacheSlot {
    fn from(c: GpuCandidate) -> Self {
        CacheSlot {
            valid: true,
            dom_row: c.dom_row,
            dom_col: c.dom_col,
            isometry: c.isometry,
            qalfa: c.qalfa,
            qbeta: c.qbeta,
            rms: c.rms,
        }
    }
}

/// One block's up-to-32 candidates, ascending by `rms`, invalid-padded at the tail.
pub type BlockTop32 = [CacheSlot; 32];

/// One configured size's full grid, row-major — matches `GpuSearcher::search_top32`'s
/// own output order exactly.
#[derive(Debug, Clone)]
pub struct SizeCache {
    pub size: u32,
    pub num_blocks_x: u32,
    pub num_blocks_y: u32,
    pub blocks: Vec<BlockTop32>,
}

impl SizeCache {
    /// The block at pixel `(row, col)`, or `None` if it is out of range or not aligned
    /// to this size's grid.
    pub fn block_at(&self, row: u32, col: u32) -> Option<&BlockTop32> {
        if row % self.size != 0 || col % self.size != 0 {
            return None;
        }
        let (bx, by) = (col / self.size, row / self.size);
        if bx >= self.num_blocks_x || by >= self.num_blocks_y {
            return None;
        }
        self.blocks.get((by * self.num_blocks_x + bx) as usize)
    }
}

/// One image x config's full oracle: every configured size's grid, plus enough of the
/// image/config identity to describe what was built (the header also carries the
/// [`manifest_hash`] that actually gates reuse — these fields are for human diagnostics
/// on a mismatch, not the correctness check itself).
#[derive(Debug, Clone)]
pub struct OracleCache {
    pub image_name: String,
    pub image_sha256_hex: String,
    pub config: OracleConfig,
    pub sizes: Vec<SizeCache>,
}

impl OracleCache {
    pub fn size(&self, size: u32) -> Option<&SizeCache> {
        self.sizes.iter().find(|s| s.size == size)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error("config error in {path}: {source}")]
    Config {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path}: not a mars oracle cache file (bad magic bytes)")]
    BadMagic { path: String },
    #[error(
        "{path}: unsupported cache format version {found} (this build writes/reads {expected})"
    )]
    Version {
        path: String,
        found: u32,
        expected: u32,
    },
    #[error("{path}: truncated or corrupt cache file ({detail})")]
    Truncated { path: String, detail: String },
    /// The manifest-hash mismatch this whole module exists to make impossible to miss
    /// (module doc). Never produced silently -- see [`load_and_validate`]. Boxed:
    /// clippy (correctly) flags an error type whose common-path variants are all a few
    /// words wide carrying one rare variant this large.
    #[error(
        "STALE ORACLE CACHE {path}: stored manifest hash does not match the current image + \
         config. stored: image={stored_image} sha256={stored_sha} config={stored_config:?} | \
         current: sha256={current_sha} config={current_config:?}. Rebuild this cache file \
         (delete it and rerun `marsbench oracle-build`) rather than trusting it.",
        path = detail.path,
        stored_image = detail.stored_image,
        stored_sha = detail.stored_sha,
        stored_config = detail.stored_config,
        current_sha = detail.current_sha,
        current_config = detail.current_config,
    )]
    Stale { detail: Box<StaleDetail> },
}

#[derive(Debug)]
pub struct StaleDetail {
    pub path: String,
    pub stored_image: String,
    pub stored_sha: String,
    pub stored_config: OracleConfig,
    pub current_sha: String,
    pub current_config: OracleConfig,
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    path: String,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], path: &str) -> Self {
        Self {
            bytes,
            pos: 0,
            path: path.to_string(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CacheError> {
        if self.pos + n > self.bytes.len() {
            return Err(CacheError::Truncated {
                path: self.path.clone(),
                detail: format!(
                    "expected {n} more bytes at offset {}, only {} remain",
                    self.pos,
                    self.bytes.len() - self.pos
                ),
            });
        }
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, CacheError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64, CacheError> {
        Ok(f64::from_bits(u64::from_le_bytes(
            self.take(8)?.try_into().unwrap(),
        )))
    }

    fn string(&mut self) -> Result<String, CacheError> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| CacheError::Truncated {
            path: self.path.clone(),
            detail: format!("non-utf8 string: {e}"),
        })
    }
}

/// Serialise a cache to bytes. Pure and testable without touching disk — see the
/// round-trip test below. Layout (all little-endian):
///
/// ```text
/// magic: [u8; 8] = "MARSORC1"
/// format_version: u32
/// manifest_hash: [u8; 32]
/// image_name: (u32 len, bytes)
/// image_sha256_hex: (u32 len, bytes)
/// min_size, max_size, shift, bits_alfa, bits_beta: u32 each
/// max_alfa: f64 (bit pattern)
/// num_sizes: u32
/// for each size, ascending:
///     size, num_blocks_x, num_blocks_y: u32 each
///     num_blocks_x * num_blocks_y blocks, each 32 CacheSlot records (28 bytes each:
///         valid, dom_row, dom_col, isometry, qalfa, qbeta as u32, rms as f32 bits)
/// ```
pub fn serialize(cache: &OracleCache) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&CACHE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&manifest_hash(&cache.image_sha256_hex, &cache.config));
    write_string(&mut out, &cache.image_name);
    write_string(&mut out, &cache.image_sha256_hex);
    out.extend_from_slice(&cache.config.min_size.to_le_bytes());
    out.extend_from_slice(&cache.config.max_size.to_le_bytes());
    out.extend_from_slice(&cache.config.shift.to_le_bytes());
    out.extend_from_slice(&cache.config.bits_alfa.to_le_bytes());
    out.extend_from_slice(&cache.config.bits_beta.to_le_bytes());
    out.extend_from_slice(&cache.config.max_alfa.to_bits().to_le_bytes());
    out.extend_from_slice(&(cache.sizes.len() as u32).to_le_bytes());
    for sc in &cache.sizes {
        out.extend_from_slice(&sc.size.to_le_bytes());
        out.extend_from_slice(&sc.num_blocks_x.to_le_bytes());
        out.extend_from_slice(&sc.num_blocks_y.to_le_bytes());
        out.reserve(sc.blocks.len() * 32 * CacheSlot::RECORD_LEN);
        for block in &sc.blocks {
            for slot in block {
                slot.serialize(&mut out);
            }
        }
    }
    out
}

/// The inverse of [`serialize`]. Does **not** check the manifest hash against anything
/// external — that is [`load_and_validate`]'s job, since parsing alone cannot know what
/// the "current" image/config are. `path` is only used to make error messages
/// actionable.
pub fn parse(bytes: &[u8], path: &str) -> Result<OracleCache, CacheError> {
    let mut cur = Cursor::new(bytes, path);
    let magic = cur.take(8)?;
    if magic != MAGIC {
        return Err(CacheError::BadMagic {
            path: path.to_string(),
        });
    }
    let version = cur.u32()?;
    if version != CACHE_FORMAT_VERSION {
        return Err(CacheError::Version {
            path: path.to_string(),
            found: version,
            expected: CACHE_FORMAT_VERSION,
        });
    }
    let stored_hash: [u8; 32] = cur.take(32)?.try_into().unwrap();
    let image_name = cur.string()?;
    let image_sha256_hex = cur.string()?;
    let config = OracleConfig {
        min_size: cur.u32()?,
        max_size: cur.u32()?,
        shift: cur.u32()?,
        bits_alfa: cur.u32()?,
        bits_beta: cur.u32()?,
        max_alfa: cur.f64()?,
    };
    // Internal self-consistency: the hash stored in the header must match what these
    // very fields hash to, independent of whatever the caller later compares against.
    // A mismatch here means the file itself is corrupt (bit flip, truncation that
    // landed on a field boundary), not merely stale.
    if manifest_hash(&image_sha256_hex, &config) != stored_hash {
        return Err(CacheError::Truncated {
            path: path.to_string(),
            detail: "manifest hash does not match the header's own stored fields \
                     (file is corrupt, not merely stale)"
                .to_string(),
        });
    }
    let num_sizes = cur.u32()? as usize;
    let mut sizes = Vec::with_capacity(num_sizes);
    for _ in 0..num_sizes {
        let size = cur.u32()?;
        let num_blocks_x = cur.u32()?;
        let num_blocks_y = cur.u32()?;
        let num_blocks = (num_blocks_x * num_blocks_y) as usize;
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            let mut block = [CacheSlot::INVALID; 32];
            for slot in &mut block {
                let raw = cur.take(CacheSlot::RECORD_LEN)?;
                *slot = CacheSlot::parse(raw);
            }
            blocks.push(block);
        }
        sizes.push(SizeCache {
            size,
            num_blocks_x,
            num_blocks_y,
            blocks,
        });
    }
    Ok(OracleCache {
        image_name,
        image_sha256_hex,
        config,
        sizes,
    })
}

pub fn save(cache: &OracleCache, path: &Path) -> Result<(), CacheError> {
    let bytes = serialize(cache);
    let mut f = std::fs::File::create(path).map_err(|source| CacheError::Io {
        path: path.display().to_string(),
        source,
    })?;
    f.write_all(&bytes).map_err(|source| CacheError::Io {
        path: path.display().to_string(),
        source,
    })
}

/// Loads a cache file with no external validation -- see [`load_and_validate`] for the
/// version that actually checks staleness against a current image/config.
pub fn load(path: &Path) -> Result<OracleCache, CacheError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|source| CacheError::Io {
            path: path.display().to_string(),
            source,
        })?;
    parse(&bytes, &path.display().to_string())
}

/// Loads a cache file and asserts it was built from exactly `current_image_sha256_hex`
/// and `current_config` -- the check the module doc calls "the load-bearing part of this
/// module." Never silently rebuilds or ignores a mismatch (`CacheError::Stale` carries a
/// full diagnostic); the caller decides what to do next.
pub fn load_and_validate(
    path: &Path,
    current_image_sha256_hex: &str,
    current_config: &OracleConfig,
) -> Result<OracleCache, CacheError> {
    let cache = load(path)?;
    let expected = manifest_hash(current_image_sha256_hex, current_config);
    let stored = manifest_hash(&cache.image_sha256_hex, &cache.config);
    if expected != stored {
        return Err(CacheError::Stale {
            detail: Box::new(StaleDetail {
                path: path.display().to_string(),
                stored_image: cache.image_name.clone(),
                stored_sha: cache.image_sha256_hex.clone(),
                stored_config: cache.config,
                current_sha: current_image_sha256_hex.to_string(),
                current_config: *current_config,
            }),
        });
    }
    Ok(cache)
}

/// Runs `search_top32` at every size in `config`'s range and assembles the in-memory
/// cache. `progress` is called once per `(size, block count, elapsed)` so a long build
/// is observable (`marsbench oracle-build`'s CLI wraps this with `eprintln!`, matching
/// `gpu-search-check`'s own per-step progress style).
pub fn build(
    gpu: &GpuSearcher,
    image_name: &str,
    image_sha256_hex: &str,
    image: &Plane,
    config: OracleConfig,
    mut progress: impl FnMut(u32, usize, std::time::Duration),
) -> OracleCache {
    let (width, height) = (image.width() as u32, image.height() as u32);
    let mut sizes = Vec::new();
    for size in config.sizes() {
        let t0 = std::time::Instant::now();
        let params = config.gpu_params(size);
        let results = gpu.search_top32(image, &params);
        let num_blocks_x = width / size;
        let num_blocks_y = height / size;
        let blocks: Vec<BlockTop32> = results
            .into_iter()
            .map(|opt| {
                let mut arr = [CacheSlot::INVALID; 32];
                if let Some(list) = opt {
                    for (slot, cand) in arr.iter_mut().zip(list.into_iter().take(32)) {
                        *slot = CacheSlot::from(cand);
                    }
                }
                arr
            })
            .collect();
        progress(size, blocks.len(), t0.elapsed());
        sizes.push(SizeCache {
            size,
            num_blocks_x,
            num_blocks_y,
            blocks,
        });
    }
    OracleCache {
        image_name: image_name.to_string(),
        image_sha256_hex: image_sha256_hex.to_string(),
        config,
        sizes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> OracleConfig {
        OracleConfig {
            min_size: 8,
            max_size: 16,
            shift: 4,
            bits_alfa: 4,
            bits_beta: 7,
            max_alfa: 1.0,
        }
    }

    #[test]
    fn sizes_covers_every_power_of_two_in_range() {
        assert_eq!(cfg().sizes(), vec![8, 16]);
        let c32 = OracleConfig {
            max_size: 32,
            ..cfg()
        };
        assert_eq!(c32.sizes(), vec![8, 16, 32]);
    }

    #[test]
    fn manifest_hash_is_deterministic_and_sensitive_to_every_field() {
        let h1 = manifest_hash("abc123", &cfg());
        let h2 = manifest_hash("abc123", &cfg());
        assert_eq!(h1, h2, "same inputs must hash identically");

        assert_ne!(
            h1,
            manifest_hash("def456", &cfg()),
            "image hash must matter"
        );
        assert_ne!(
            h1,
            manifest_hash(
                "abc123",
                &OracleConfig {
                    min_size: 4,
                    ..cfg()
                }
            ),
            "min_size must matter"
        );
        assert_ne!(
            h1,
            manifest_hash("abc123", &OracleConfig { shift: 8, ..cfg() }),
            "shift must matter"
        );
        assert_ne!(
            h1,
            manifest_hash(
                "abc123",
                &OracleConfig {
                    max_alfa: 0.5,
                    ..cfg()
                }
            ),
            "max_alfa must matter"
        );
    }

    fn sample_cache() -> OracleCache {
        let mut block0 = [CacheSlot::INVALID; 32];
        block0[0] = CacheSlot {
            valid: true,
            dom_row: 4,
            dom_col: 8,
            isometry: 3,
            qalfa: 2,
            qbeta: 100,
            rms: 1.5,
        };
        block0[1] = CacheSlot {
            valid: true,
            dom_row: 12,
            dom_col: 0,
            isometry: 0,
            qalfa: 1,
            qbeta: 50,
            rms: 2.25,
        };
        OracleCache {
            image_name: "kodim01".to_string(),
            image_sha256_hex: "deadbeef".to_string(),
            config: cfg(),
            sizes: vec![SizeCache {
                size: 8,
                num_blocks_x: 1,
                num_blocks_y: 1,
                blocks: vec![block0],
            }],
        }
    }

    #[test]
    fn round_trips_through_serialize_and_parse() {
        let cache = sample_cache();
        let bytes = serialize(&cache);
        let back = parse(&bytes, "test").expect("valid bytes must parse");
        assert_eq!(back.image_name, cache.image_name);
        assert_eq!(back.image_sha256_hex, cache.image_sha256_hex);
        assert_eq!(back.config, cache.config);
        assert_eq!(back.sizes.len(), 1);
        assert_eq!(back.sizes[0].size, 8);
        assert_eq!(back.sizes[0].blocks[0][0], cache.sizes[0].blocks[0][0]);
        assert_eq!(back.sizes[0].blocks[0][1], cache.sizes[0].blocks[0][1]);
        assert!(!back.sizes[0].blocks[0][2].valid, "padding stays invalid");
    }

    #[test]
    fn load_and_validate_rejects_a_hash_mismatch_loudly() {
        let dir = std::env::temp_dir().join(format!("mars-oracle-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stale.bin");
        save(&sample_cache(), &path).unwrap();

        // Matching image/config: loads cleanly.
        assert!(load_and_validate(&path, "deadbeef", &cfg()).is_ok());

        // A different image hash must be rejected, not silently accepted.
        let err = load_and_validate(&path, "not-the-same-hash", &cfg()).unwrap_err();
        assert!(matches!(err, CacheError::Stale { .. }));

        // A different config must also be rejected.
        let mut other = cfg();
        other.max_size = 32;
        let err = load_and_validate(&path, "deadbeef", &other).unwrap_err();
        assert!(matches!(err, CacheError::Stale { .. }));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupt_bytes_are_rejected_as_truncated_or_bad_magic() {
        assert!(matches!(
            parse(b"not a cache file at all", "x"),
            Err(CacheError::BadMagic { .. })
        ));
        let mut bytes = serialize(&sample_cache());
        bytes.truncate(bytes.len() - 4);
        assert!(matches!(
            parse(&bytes, "x"),
            Err(CacheError::Truncated { .. })
        ));
    }

    #[test]
    fn block_at_respects_grid_alignment_and_bounds() {
        let cache = sample_cache();
        let sc = cache.size(8).unwrap();
        assert_eq!(sc.block_at(0, 0), Some(&cache.sizes[0].blocks[0]));
        assert_eq!(sc.block_at(1, 0), None, "misaligned row");
        assert_eq!(sc.block_at(8, 0), None, "out of the 1x1 grid");
    }
}
