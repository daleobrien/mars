//! Golden `.ifs` fixtures (Step 3).
//!
//! Freezes a set of Mars 1 bitstreams, together with everything needed to prove that a
//! parser has understood them: the encoder's own `transforms` and
//! `Zero_alfa_transformations` counts, and the SHA-256 of the partition rendering
//! (`encmars -Q`) and of both decodes.
//!
//! **Only the `.ifs` files are committed.** The decoded PGMs and partition renderings are
//! recorded as hashes in `fixtures/mars1/manifest.toml` instead — 25 MB of decoded pixels
//! is not "committed, small" (§3), and a hash is a stricter check than a stored file
//! because it cannot be quietly regenerated to match. This mirrors `fixtures/images/`,
//! where the `.raw` inputs are also hash-pinned rather than committed.
//!
//! The consequence that matters: `just gate-3` validates the spec against committed data
//! alone. It needs no C binaries and no image corpus, so the format spec stays checkable
//! long after this machine's toolchain has moved on.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::mars1::{
    decode, encode, DecodeMode, DecodeParams, EncodeParams, EncodeStats, Mars1Binaries, Mars1Error,
    Method,
};
use crate::provenance::sha256_hex;
use crate::sweep::{BaseParams, ImageEntry, ImageSet, Variant};

/// Where the committed bitstreams live.
pub const FIXTURE_DIR: &str = "fixtures/mars1";

// ------------------------------------------------------------------- the config file

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FixtureConfig {
    pub name: String,
    pub description: String,
    /// Image-set indexes to draw from, e.g. `corpus/fixtures.images.json`.
    pub indexes: Vec<PathBuf>,
    /// Images that belong to no corpus index: the 1998 `lena.raw` and the walkthrough
    /// image. Listed here with their hashes so the set is still fully pinned.
    #[serde(default)]
    pub extra_images: Vec<ImageEntry>,
    pub base: BaseParams,
    pub variants: Vec<Variant>,
    pub groups: Vec<CaseGroup>,
}

/// A cross product of images × variants × methods × rates. Several small groups are
/// clearer than one grid with exceptions bolted on, and the config stays readable.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CaseGroup {
    pub label: String,
    pub description: String,
    /// Image names. Empty means every image in every index plus every extra.
    #[serde(default)]
    pub images: Vec<String>,
    pub variants: Vec<String>,
    pub methods: Vec<Method>,
    pub rms: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Case {
    pub id: usize,
    pub group: String,
    pub image: ImageEntry,
    pub variant: String,
    pub params: EncodeParams,
}

impl Case {
    /// The fixture's stem, and therefore its filename. Deterministic and sortable.
    pub fn stem(&self) -> String {
        format!(
            "{}__{}__{}__r{}",
            self.image.name,
            self.variant,
            self.params.method.key(),
            fmt_rms(self.params.t_rms)
        )
    }
}

/// `2.0` -> `2`, `2.5` -> `2p5`. Filenames with dots in the middle invite tooling to
/// mistake the rate for an extension.
fn fmt_rms(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{}", v as i64)
    } else {
        format!("{v}").replace('.', "p")
    }
}

impl FixtureConfig {
    pub fn read(path: &Path) -> Result<Self, FixtureError> {
        let text = std::fs::read_to_string(path).map_err(|source| FixtureError::Io {
            path: path.display().to_string(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| FixtureError::Config {
            path: path.display().to_string(),
            source,
        })
    }

    fn params(&self, variant: &Variant, method: Method, t_rms: f64) -> EncodeParams {
        EncodeParams {
            method,
            t_rms,
            min_size: variant.min_size.unwrap_or(self.base.min_size),
            max_size: variant.max_size.unwrap_or(self.base.max_size),
            shift: variant.shift.unwrap_or(self.base.shift),
            bits_alfa: variant.bits_alfa.unwrap_or(self.base.bits_alfa),
            bits_beta: variant.bits_beta.unwrap_or(self.base.bits_beta),
            max_alfa: variant.max_alfa.unwrap_or(self.base.max_alfa),
            t_ent: None,
            t_var: None,
        }
    }

    /// Read the indexes and expand into cases, in a fixed order: group, image, variant,
    /// method, rate. Fixture filenames are content-addressed by their parameters, so the
    /// order only decides which fixture is generated first — but the manifest is written
    /// in it, and a manifest whose row order wandered between runs would produce a diff
    /// on every regeneration.
    pub fn plan(&self, root: &Path) -> Result<Vec<Case>, FixtureError> {
        let mut images: Vec<ImageEntry> = Vec::new();
        for index in &self.indexes {
            let set = ImageSet::read(&root.join(index)).map_err(|e| FixtureError::ImageSet {
                path: index.display().to_string(),
                message: e.to_string(),
            })?;
            images.extend(set.images);
        }
        images.extend(self.extra_images.iter().cloned());

        let by_name: BTreeMap<&str, &ImageEntry> =
            images.iter().map(|i| (i.name.as_str(), i)).collect();
        if by_name.len() != images.len() {
            return Err(FixtureError::DuplicateImage);
        }

        let mut cases = Vec::new();
        for group in &self.groups {
            let selected: Vec<&ImageEntry> = if group.images.is_empty() {
                images.iter().collect()
            } else {
                group
                    .images
                    .iter()
                    .map(|n| {
                        by_name
                            .get(n.as_str())
                            .copied()
                            .ok_or_else(|| FixtureError::UnknownImage {
                                name: n.clone(),
                                group: group.label.clone(),
                            })
                    })
                    .collect::<Result<_, _>>()?
            };
            for image in selected {
                for label in &group.variants {
                    let variant = self
                        .variants
                        .iter()
                        .find(|v| &v.label == label)
                        .ok_or_else(|| FixtureError::UnknownVariant {
                            label: label.clone(),
                            group: group.label.clone(),
                        })?;
                    for &method in &group.methods {
                        for &t_rms in &group.rms {
                            cases.push(Case {
                                id: cases.len(),
                                group: group.label.clone(),
                                image: image.clone(),
                                variant: label.clone(),
                                params: self.params(variant, method, t_rms),
                            });
                        }
                    }
                }
            }
        }

        let mut stems: Vec<String> = cases.iter().map(Case::stem).collect();
        let before = stems.len();
        stems.sort_unstable();
        stems.dedup();
        if stems.len() != before {
            return Err(FixtureError::DuplicateCase);
        }
        Ok(cases)
    }
}

// ------------------------------------------------------------------- the manifest row

/// What one golden fixture pins. Every field is either an exact integer the C printed or
/// a SHA-256, so every check downstream of it is an equality rather than a tolerance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureRecord {
    pub stem: String,
    pub group: String,
    pub image: String,
    pub image_sha256: String,
    pub width: u32,
    pub height: u32,
    pub variant: String,
    pub params: EncodeParams,
    /// The full `encmars` argument vector, as spawned.
    pub encode_argv: Vec<String>,
    pub ifs_bytes: u64,
    pub ifs_sha256: String,
    /// `Number of transformations`, straight from the encoder's stdout.
    pub transforms: u64,
    /// `Zero_alfa_transformations`.
    pub zero_alfa_transforms: u64,
    /// `Number of comparisons` (§M5). Not used by the parser check; recorded because a
    /// fixture that silently changed method would otherwise look identical.
    pub comparisons: u64,
    /// SHA-256 of the `quadtree.pgm` written by `-Q`: the partition, rendered.
    pub quadtree_pgm_sha256: String,
    /// SHA-256 of `decmars -i` output (PGM, 10 iterations, no postprocessing).
    pub decode_iterative_sha256: String,
    /// SHA-256 of the default pyramidal decode.
    pub decode_pyramidal_sha256: String,
}

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("io error on {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("cannot parse {path}: {source}")]
    Config {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("cannot read image set {path}: {message}")]
    ImageSet { path: String, message: String },
    #[error("two images share a name")]
    DuplicateImage,
    #[error("two cases produce the same fixture name")]
    DuplicateCase,
    #[error("group {group:?} names image {name:?}, which is in no index")]
    UnknownImage { name: String, group: String },
    #[error("group {group:?} names variant {label:?}, which is not defined")]
    UnknownVariant { label: String, group: String },
    #[error("{name}: expected sha256 {want}, found {got} — regenerate the corpus")]
    ImageHashMismatch {
        name: String,
        want: String,
        got: String,
    },
    #[error("{stem}: encmars -Q wrote no quadtree.pgm")]
    NoQuadtreeImage { stem: String },
    #[error(transparent)]
    Mars1(#[from] Mars1Error),
}

// ------------------------------------------------------------------------ generation

/// Generate one fixture: encode with `-Q`, decode both ways, hash everything, and copy
/// the bitstream into `fixtures/mars1/`.
///
/// `-Q` does not change the bitstream — `qtt` is filled in during the walk regardless and
/// the flag only decides whether it is written out — so the committed `.ifs` is exactly
/// what a plain `encmars` invocation produces.
pub fn generate(
    bins: &Mars1Binaries,
    root: &Path,
    scratch: &Path,
    case: &Case,
) -> Result<FixtureRecord, FixtureError> {
    let workdir = scratch.join(format!("c{:04}", case.id));
    std::fs::create_dir_all(&workdir).map_err(|source| FixtureError::Io {
        path: workdir.display().to_string(),
        source,
    })?;

    let src = root.join(&case.image.file);
    let bytes = std::fs::read(&src).map_err(|source| FixtureError::Io {
        path: src.display().to_string(),
        source,
    })?;
    let image_sha256 = sha256_hex(&bytes);
    if image_sha256 != case.image.sha256 {
        return Err(FixtureError::ImageHashMismatch {
            name: case.image.name.clone(),
            want: case.image.sha256.clone(),
            got: image_sha256,
        });
    }
    write(&workdir.join("i.raw"), &bytes)?;

    let qflag = vec!["-Q".to_string()];
    let enc = encode(
        bins,
        &workdir,
        "i.raw",
        "o.ifs",
        case.image.width,
        case.image.height,
        &case.params,
        &qflag,
    )?;
    let EncodeStats {
        transforms,
        zero_alfa_transforms,
        comparisons,
        ..
    } = enc.stats;

    let ifs = read(&workdir.join("o.ifs"))?;
    let quadtree = match std::fs::read(workdir.join("quadtree.pgm")) {
        Ok(b) => b,
        Err(_) => return Err(FixtureError::NoQuadtreeImage { stem: case.stem() }),
    };

    let expect = (case.image.width, case.image.height);
    let mut decodes = Vec::new();
    for (mode, out) in [
        (DecodeMode::Iterative, "d_it.pgm"),
        (DecodeMode::Pyramidal, "d_py.pgm"),
    ] {
        let params = DecodeParams {
            mode,
            iterations: None,
            postprocess: false,
        };
        decode(bins, &workdir, "o.ifs", out, expect, &params)?;
        decodes.push(sha256_hex(&read(&workdir.join(out))?));
    }

    let dir = root.join(FIXTURE_DIR);
    std::fs::create_dir_all(&dir).map_err(|source| FixtureError::Io {
        path: dir.display().to_string(),
        source,
    })?;
    write(&dir.join(format!("{}.ifs", case.stem())), &ifs)?;

    let mut encode_argv = vec!["encmars".to_string()];
    let mut args = case
        .params
        .args(case.image.width, case.image.height, "i.raw", "o.ifs");
    let tail = args.split_off(args.len() - 2);
    encode_argv.extend(args);
    encode_argv.extend(qflag);
    encode_argv.extend(tail);

    Ok(FixtureRecord {
        stem: case.stem(),
        group: case.group.clone(),
        image: case.image.name.clone(),
        image_sha256,
        width: case.image.width,
        height: case.image.height,
        variant: case.variant.clone(),
        params: case.params.clone(),
        encode_argv,
        ifs_bytes: ifs.len() as u64,
        ifs_sha256: sha256_hex(&ifs),
        transforms,
        zero_alfa_transforms,
        comparisons,
        quadtree_pgm_sha256: sha256_hex(&quadtree),
        decode_iterative_sha256: decodes[0].clone(),
        decode_pyramidal_sha256: decodes[1].clone(),
    })
}

fn read(path: &Path) -> Result<Vec<u8>, FixtureError> {
    std::fs::read(path).map_err(|source| FixtureError::Io {
        path: path.display().to_string(),
        source,
    })
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), FixtureError> {
    std::fs::write(path, bytes).map_err(|source| FixtureError::Io {
        path: path.display().to_string(),
        source,
    })
}

// -------------------------------------------------------------------------- manifest

/// Render `fixtures/mars1/manifest.toml`.
///
/// Written by hand rather than through a TOML serialiser so that the header explaining
/// what the file is for travels with it — this manifest is read by a Python script that
/// deliberately knows nothing else about the project.
pub fn manifest_toml(
    config: &FixtureConfig,
    records: &[FixtureRecord],
    build_info: Option<&str>,
) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# Golden Mars 1 `.ifs` fixtures -- Step 3.\n\
         # GENERATED by `just mars1-fixtures` -- do not edit by hand.\n\
         #\n\
         # {}\n\
         #\n\
         # Only the .ifs files are committed. The decoded PGMs and the -Q partition\n\
         # renderings are pinned by SHA-256 here instead; `just gate-3` re-derives them\n\
         # from the bitstream and compares. Every check downstream of this file is an\n\
         # exact equality.\n\
         #\n\
         # `transforms` and `zero_alfa_transforms` are what the 1998 encoder printed. They\n\
         # are the oracle an independent parser is checked against, and they are integers,\n\
         # so the check needs no tolerance and no floating point.\n\n",
        config.description
    ));

    s.push_str("[generator]\n");
    s.push_str(&format!("config_name = {:?}\n", config.name));
    s.push_str(&format!("fixture_count = {}\n", records.len()));
    if let Some(info) = build_info {
        s.push_str("# verbatim target/mars1/build-info.txt\n");
        for line in info.lines().filter(|l| !l.trim().is_empty()) {
            s.push_str(&format!("# {line}\n"));
        }
    }
    s.push('\n');

    for r in records {
        s.push_str("[[fixture]]\n");
        s.push_str(&format!("stem = {:?}\n", r.stem));
        s.push_str(&format!("group = {:?}\n", r.group));
        s.push_str(&format!("file = {:?}\n", format!("{}.ifs", r.stem)));
        s.push_str(&format!("image = {:?}\n", r.image));
        s.push_str(&format!("image_sha256 = {:?}\n", r.image_sha256));
        s.push_str(&format!("width = {}\n", r.width));
        s.push_str(&format!("height = {}\n", r.height));
        s.push_str(&format!("variant = {:?}\n", r.variant));
        s.push_str(&format!("method = {:?}\n", r.params.method.key()));
        s.push_str(&format!("t_rms = {}\n", r.params.t_rms));
        s.push_str(&format!("min_size = {}\n", r.params.min_size));
        s.push_str(&format!("max_size = {}\n", r.params.max_size));
        s.push_str(&format!("shift = {}\n", r.params.shift));
        s.push_str(&format!("bits_alfa = {}\n", r.params.bits_alfa));
        s.push_str(&format!("bits_beta = {}\n", r.params.bits_beta));
        s.push_str(&format!("max_alfa = {}\n", r.params.max_alfa));
        s.push_str(&format!(
            "encode_argv = [{}]\n",
            r.encode_argv
                .iter()
                .map(|a| format!("{a:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        s.push_str(&format!("ifs_bytes = {}\n", r.ifs_bytes));
        s.push_str(&format!("ifs_sha256 = {:?}\n", r.ifs_sha256));
        s.push_str(&format!("transforms = {}\n", r.transforms));
        s.push_str(&format!(
            "zero_alfa_transforms = {}\n",
            r.zero_alfa_transforms
        ));
        s.push_str(&format!("comparisons = {}\n", r.comparisons));
        s.push_str(&format!(
            "quadtree_pgm_sha256 = {:?}\n",
            r.quadtree_pgm_sha256
        ));
        s.push_str(&format!(
            "decode_iterative_sha256 = {:?}\n",
            r.decode_iterative_sha256
        ));
        s.push_str(&format!(
            "decode_pyramidal_sha256 = {:?}\n\n",
            r.decode_pyramidal_sha256
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> FixtureConfig {
        serde_json::from_str(
            r#"{
              "name": "t", "description": "d",
              "indexes": [],
              "extra_images": [
                {"name":"a","file":"a.raw","width":64,"height":64,"sha256":"00"},
                {"name":"b","file":"b.raw","width":64,"height":64,"sha256":"01"}
              ],
              "base": {"min_size":4,"max_size":16,"shift":4,"bits_alfa":4,"bits_beta":7,"max_alfa":1.0},
              "variants": [
                {"label":"default","description":"1998"},
                {"label":"alfa5","description":"-A 5","bits_alfa":5}
              ],
              "groups": [
                {"label":"grid","description":"","images":[],"variants":["default"],
                 "methods":["fisher","masscenter"],"rms":[2.0,8.0]},
                {"label":"stress","description":"","images":["b"],"variants":["alfa5"],
                 "methods":["fisher"],"rms":[8.0]}
              ]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn the_plan_is_the_declared_cross_product_in_a_fixed_order() {
        let cases = cfg().plan(Path::new(".")).unwrap();
        // 2 images x 1 variant x 2 methods x 2 rates, then 1 stress case.
        assert_eq!(cases.len(), 9);
        assert_eq!(cases[0].stem(), "a__default__fisher__r2");
        assert_eq!(cases[8].stem(), "b__alfa5__fisher__r8");
        assert_eq!(cases[8].params.bits_alfa, 5);
        assert_eq!(cases[0].params.bits_alfa, 4);
        // Re-planning gives the identical order; the manifest must not churn.
        let again = cfg().plan(Path::new(".")).unwrap();
        let a: Vec<String> = cases.iter().map(Case::stem).collect();
        let b: Vec<String> = again.iter().map(Case::stem).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn a_variant_that_changes_nothing_would_collide_and_is_rejected() {
        // Two variant labels that resolve to the same parameters still produce distinct
        // filenames, but a repeated *label* in one group would not — and a fixture set
        // with two entries writing one file is a set that silently loses a case.
        let mut c = cfg();
        c.groups[0].variants.push("default".into());
        assert!(matches!(
            c.plan(Path::new(".")),
            Err(FixtureError::DuplicateCase)
        ));
    }

    #[test]
    fn an_unknown_image_or_variant_fails_the_plan_rather_than_being_skipped() {
        let mut c = cfg();
        c.groups[1].images = vec!["nope".into()];
        assert!(matches!(
            c.plan(Path::new(".")),
            Err(FixtureError::UnknownImage { .. })
        ));
        let mut c = cfg();
        c.groups[1].variants = vec!["nope".into()];
        assert!(matches!(
            c.plan(Path::new(".")),
            Err(FixtureError::UnknownVariant { .. })
        ));
    }

    #[test]
    fn rates_render_into_filenames_without_a_second_dot() {
        assert_eq!(fmt_rms(2.0), "2");
        assert_eq!(fmt_rms(32.0), "32");
        assert_eq!(fmt_rms(2.5), "2p5");
        assert!(!fmt_rms(2.5).contains('.'));
    }
}
