//! Step 17 — offline training for `mars_search::learned::Learned`'s MLP.
//!
//! Reads `kodim01`'s oracle cache (`default` config, size 16 only — see this file's own
//! scope-cut reasoning in `docs/predictions.md`'s Step 17 prediction and `docs/decisions.md`),
//! builds a labelled (range block, domain position) dataset (positive = domain position
//! appears in the oracle's true top-32 for that block, negative = a seeded-random sample
//! of the rest of the domain pool), trains a small 2-layer MLP by plain full-batch
//! gradient descent on binary cross-entropy, and writes the trained weights to
//! `crates/mars-search/src/learned_weights.rs` as plain Rust source (`include!`'d by
//! `mars_search::learned::trained`).
//!
//! Run with `cargo run -p mars-bench --example train_learned --release` from the repo
//! root, after `marsbench oracle-build --images kodim01 --configs default`.
//!
//! **Determinism (§2.3): no wall-clock seed.** Every random choice (weight
//! initialisation, negative sampling, epoch shuffling) is drawn from one seeded
//! `SplitMix64` PRNG, `RNG_SEED` below — re-running this example with the same oracle
//! cache reproduces byte-identical weights.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mars_bench::oracle::{self, OracleConfig};
use mars_codec::encode::build_contracted;
use mars_core::io::read_raw;
use mars_search::funnel::stage1_features;
use mars_search::learned::{build_features, mean_std, FEATURE_DIM, HIDDEN_DIM};
use mars_search::{DomainPool, RangeBlock};

/// Explicit seed, recorded here and in `docs/decisions.md`'s Step 17 entry — the whole
/// point of pinning it is that a reviewer can reproduce this exact model.
const RNG_SEED: u64 = 0xC0_FF_EE_17;
const SIZE: u32 = 16;
const NEG_PER_BLOCK: usize = 32;
const EPOCHS: usize = 300;
const LR: f32 = 0.02;

/// SplitMix64 — a small, well-known, dependency-free PRNG. Not cryptographic; this is
/// deterministic sampling for training, not a security context.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// Uniform in `[-scale, scale]`.
    fn next_f32(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
        (u * 2.0 - 1.0) * scale
    }
}

fn fisher_yates_shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.next_usize(i + 1);
        v.swap(i, j);
    }
}

fn main() {
    let cfg = OracleConfig {
        min_size: 8,
        max_size: 16,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
    };
    let image = read_raw(Path::new("corpus/images/kodak-gray/kodim01.raw"), 768, 512)
        .expect("kodim01.raw (run `just corpus-gray` or fetch+convert kodim01 first)");
    let cache = oracle::load_and_validate(
        Path::new("oracle-cache/kodim01__default.bin"),
        "4bcd9402749c018e7a3c5c5c921dfa51ac6e9037bb648310569ecb803be3012d",
        &cfg,
    )
    .expect("valid oracle cache for kodim01 [default] (run `marsbench oracle-build` first)");

    let contracted = build_contracted(&image);
    let (image_width, image_height) = (image.width() as u32, image.height() as u32);
    let pool = DomainPool {
        contracted: &contracted,
        size: SIZE,
        shift: cfg.shift,
        image_width,
        image_height,
    };

    struct DomainEntry {
        row: u32,
        col: u32,
        stage1: mars_search::funnel::Stage1Features,
        std: f64,
    }
    let domains: Vec<DomainEntry> = pool
        .domain_positions()
        .map(|(row, col)| {
            let block = pool.domain_block(row, col);
            let stage1 = stage1_features(&block);
            let (_, std) = mean_std(&block);
            DomainEntry {
                row,
                col,
                stage1,
                std,
            }
        })
        .collect();
    let pos_lookup: HashMap<(u32, u32), usize> = domains
        .iter()
        .enumerate()
        .map(|(i, d)| ((d.row, d.col), i))
        .collect();
    eprintln!("domain pool: {} positions", domains.len());

    let size_cache = cache.size(SIZE).expect("size 16 present in oracle cache");
    eprintln!(
        "range grid: {}x{} = {} blocks",
        size_cache.num_blocks_x,
        size_cache.num_blocks_y,
        size_cache.num_blocks_x * size_cache.num_blocks_y
    );

    let px = image.as_slice();
    let stride = image.width();
    let mut rng = Rng(RNG_SEED);
    let mut samples: Vec<([f32; FEATURE_DIM], f32)> = Vec::new();

    for by in 0..size_cache.num_blocks_y {
        for bx in 0..size_cache.num_blocks_x {
            let row = by * SIZE;
            let col = bx * SIZE;
            let mut pixels = vec![0u8; (SIZE * SIZE) as usize];
            for i in 0..SIZE as usize {
                let src = (row as usize + i) * stride + col as usize;
                pixels[i * SIZE as usize..(i + 1) * SIZE as usize]
                    .copy_from_slice(&px[src..src + SIZE as usize]);
            }
            let range = RangeBlock {
                row,
                col,
                size: SIZE,
                pixels: &pixels,
            };
            let rblock = range.as_block();
            let r1 = stage1_features(&rblock);
            let (_, r_std) = mean_std(&rblock);

            let top32 = size_cache
                .block_at(row, col)
                .expect("every range-grid position has a cache slot");
            let mut pos_positions: Vec<(u32, u32)> = top32
                .iter()
                .filter(|s| s.valid)
                .map(|s| (s.dom_row, s.dom_col))
                .collect();
            pos_positions.sort_unstable();
            pos_positions.dedup();
            let pos_set: HashSet<(u32, u32)> = pos_positions.iter().copied().collect();

            for &(dr, dc) in &pos_positions {
                let Some(&idx) = pos_lookup.get(&(dr, dc)) else {
                    // The oracle's GPU search scans domain positions on the same
                    // (shift-aligned) grid `DomainPool::domain_positions` produces at
                    // this size — a miss here would mean the two grids disagree, a
                    // harness bug worth knowing about rather than silently skipping.
                    panic!("oracle top-32 domain ({dr},{dc}) not in this DomainPool's grid");
                };
                let d = &domains[idx];
                let feats = build_features(
                    &r1,
                    r_std,
                    &d.stage1,
                    d.std,
                    row,
                    col,
                    d.row,
                    d.col,
                    image_width,
                    image_height,
                );
                samples.push((feats, 1.0));
            }

            let mut neg_count = 0usize;
            let mut guard = 0usize;
            while neg_count < NEG_PER_BLOCK && guard < NEG_PER_BLOCK * 50 {
                guard += 1;
                let idx = rng.next_usize(domains.len());
                let d = &domains[idx];
                if pos_set.contains(&(d.row, d.col)) {
                    continue;
                }
                let feats = build_features(
                    &r1,
                    r_std,
                    &d.stage1,
                    d.std,
                    row,
                    col,
                    d.row,
                    d.col,
                    image_width,
                    image_height,
                );
                samples.push((feats, 0.0));
                neg_count += 1;
            }
        }
    }
    eprintln!(
        "training samples: {} ({} positive, {} negative)",
        samples.len(),
        samples.iter().filter(|(_, y)| *y > 0.5).count(),
        samples.iter().filter(|(_, y)| *y < 0.5).count(),
    );

    // --- train: 2-layer MLP, plain full-dataset SGD with per-sample updates ---
    let scale1 = 1.0 / (FEATURE_DIM as f32).sqrt();
    let scale2 = 1.0 / (HIDDEN_DIM as f32).sqrt();
    let mut w1 = [[0f32; FEATURE_DIM]; HIDDEN_DIM];
    for row in w1.iter_mut() {
        for v in row.iter_mut() {
            *v = rng.next_f32(scale1);
        }
    }
    let mut b1 = [0f32; HIDDEN_DIM];
    let mut w2 = [0f32; HIDDEN_DIM];
    for v in w2.iter_mut() {
        *v = rng.next_f32(scale2);
    }
    let mut b2 = 0f32;

    let mut order: Vec<usize> = (0..samples.len()).collect();
    for epoch in 0..EPOCHS {
        fisher_yates_shuffle(&mut order, &mut rng);
        let mut total_loss = 0f64;
        for &si in &order {
            let (feats, label) = samples[si];
            let mut hidden = [0f32; HIDDEN_DIM];
            for h in 0..HIDDEN_DIM {
                let mut acc = b1[h];
                for f in 0..FEATURE_DIM {
                    acc += w1[h][f] * feats[f];
                }
                hidden[h] = acc.tanh();
            }
            let mut out_pre = b2;
            for h in 0..HIDDEN_DIM {
                out_pre += w2[h] * hidden[h];
            }
            let pred = 1.0 / (1.0 + (-out_pre).exp());
            let eps = 1e-7f32;
            total_loss += -(f64::from(label) * f64::from((pred + eps).ln())
                + f64::from(1.0 - label) * f64::from((1.0 - pred + eps).ln()));

            let dout = pred - label; // BCE + sigmoid combined gradient
            let mut dhidden = [0f32; HIDDEN_DIM];
            for h in 0..HIDDEN_DIM {
                dhidden[h] = dout * w2[h] * (1.0 - hidden[h] * hidden[h]);
                w2[h] -= LR * dout * hidden[h];
            }
            b2 -= LR * dout;
            for h in 0..HIDDEN_DIM {
                b1[h] -= LR * dhidden[h];
                for f in 0..FEATURE_DIM {
                    w1[h][f] -= LR * dhidden[h] * feats[f];
                }
            }
        }
        if epoch % 50 == 0 || epoch == EPOCHS - 1 {
            eprintln!(
                "epoch {epoch}: mean BCE loss = {:.5}",
                total_loss / samples.len() as f64
            );
        }
    }

    // --- write learned_weights.rs ---
    let mut out = String::new();
    out.push_str(
        "// GENERATED by `crates/mars-bench/examples/train_learned.rs` -- do not hand-edit.\n",
    );
    out.push_str(&format!(
        "// Trained on kodim01's oracle cache (`default` config, size {SIZE}), seed 0x{RNG_SEED:X}, {EPOCHS} epochs, lr={LR}.\n"
    ));
    out.push_str(&format!(
        "// {} training samples ({} positive / {} negative). Regenerate: `cargo run -p mars-bench --example train_learned --release`.\n",
        samples.len(),
        samples.iter().filter(|(_, y)| *y > 0.5).count(),
        samples.iter().filter(|(_, y)| *y < 0.5).count(),
    ));
    out.push_str("pub static WEIGHTS: super::MlpWeights = super::MlpWeights {\n");
    // `{v}` (f32 Display), not a fixed-decimal format: Rust's float Display prints the
    // shortest decimal that round-trips to the exact same f32 bits, which is also what
    // clippy's `excessive_precision` lint wants -- a `{v:.8}` fixed format was rejected
    // by that lint (more digits than the value needs).
    out.push_str("    w1: [\n");
    for row in &w1 {
        out.push_str("        [");
        for v in row {
            out.push_str(&format!("{v}f32, "));
        }
        out.push_str("],\n");
    }
    out.push_str("    ],\n");
    out.push_str("    b1: [");
    for v in &b1 {
        out.push_str(&format!("{v}f32, "));
    }
    out.push_str("],\n");
    out.push_str("    w2: [");
    for v in &w2 {
        out.push_str(&format!("{v}f32, "));
    }
    out.push_str("],\n");
    out.push_str(&format!("    b2: {b2}f32,\n"));
    out.push_str("};\n");

    let dest = Path::new("crates/mars-search/src/learned_weights.rs");
    std::fs::write(dest, out).expect("write learned_weights.rs");
    eprintln!("wrote {}", dest.display());
}
