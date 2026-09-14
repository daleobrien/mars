//! Domain/range classification, ported from `reference/mars1/index_func.c`,
//! `coding_func.c`'s `flips`, and `split_func.c`'s `variance_2` — the pieces every
//! candidate-restriction method needs regardless of which candidate set it builds.
//!
//! **Scale invariance note.** Mars 1 classifies over `contract[][]`, which
//! `contraction()` fills with the *mean* of each 2x2 pixel quadruple. `mars_codec`'s
//! [`mars_codec::encode::Contracted`] instead stores the *sum* `D = 4*mean` (§9 of the
//! format doc). Every classifier below only ever compares quadrant sums/variances to each
//! other or takes a ratio of two quantities that both scale identically with the block
//! (`newclass`, `hurtgen_class`, `variance_class`, `ComputeMc`'s `row/mass`, `col/mass`),
//! so classifying on `D` instead of `D/4` produces the *identical* classification — same
//! order permutation, same threshold-comparison bits, same polar angle — as a naive port
//! that divided by 4 first would. This lets every function here take `D`-scaled blocks
//! directly, with no behavioural difference, and is recorded in `docs/decisions.md`.

use std::f64::consts::PI;

const TWOPI: f64 = 2.0 * PI;

/// A `size x size` block of `f64` samples, row-major — the common currency between a
/// domain's contracted samples and a range block's raw pixels once cast to `f64`.
#[derive(Clone)]
pub struct Block {
    pub data: Vec<f64>,
    pub size: usize,
}

impl Block {
    pub fn new(size: usize, data: Vec<f64>) -> Self {
        debug_assert_eq!(data.len(), size * size);
        Self { data, size }
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.size + j]
    }
}

/// `[TL, TR, BL, BR]` quadrant sums over the top-level `size x size` block — shared by
/// `newclass` (sums) and `hurtgen_class` (sums, then divided into means).
fn quadrant_sums(b: &Block) -> [f64; 4] {
    let h = b.size / 2;
    let mut a = [0.0f64; 4];
    for i in 0..h {
        for j in 0..h {
            a[0] += b.at(i, j);
            a[1] += b.at(i, j + h);
            a[2] += b.at(i + h, j);
            a[3] += b.at(i + h, j + h);
        }
    }
    a
}

/// `index_func.c`'s exact 3-pass descending bubble sort over 4 elements, carrying the
/// index permutation alongside the values — `newclass`/`variance_class` both need this
/// *exact* algorithm (not just "sorted descending") because ties must break the same way
/// the reference's `match()` lookup expects, and the comparisons are scale-invariant (see
/// module doc) so this reproduces the reference's tie-breaking exactly, not just its
/// asymptotic behaviour.
fn bubble_sort_desc_with_order(a: &mut [f64; 4]) -> [i32; 4] {
    let mut order = [0i32, 1, 2, 3];
    for i in (0..=2).rev() {
        for j in 0..=i {
            if a[j] < a[j + 1] {
                a.swap(j, j + 1);
                order.swap(j, j + 1);
            }
        }
    }
    order
}

/// `variance_2(size, block, atx, aty)`: `mean2 - mean^2` over one `size x size`
/// sub-block starting at `(atx, aty)` of a larger block.
fn variance_2(b: &Block, atx: usize, aty: usize, size: usize) -> f64 {
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for i in atx..atx + size {
        for j in aty..aty + size {
            let v = b.at(i, j);
            sum += v;
            sum2 += v * v;
        }
    }
    let n = (size * size) as f64;
    sum2 / n - (sum / n) * (sum / n)
}

/// `newclass()`: the symmetry operation that brings a domain (or range) block into
/// canonical orientation, plus its one-of-3 shape class (Fisher's outer classification
/// axis).
pub fn newclass(b: &Block) -> (u8, u8) {
    let mut a = quadrant_sums(b);
    let order = bubble_sort_desc_with_order(&mut a);
    let idx = crate::tables::match_order(order);
    let row = crate::tables::ORDERING[idx];
    (row[4] as u8, row[5] as u8)
}

/// `variance_class()`: the one-of-24 class from the descending order of the four
/// quadrant variances — used standalone by Hurtgen/MassCenter's second axis and after
/// canonicalisation by Fisher/Saupe-Fisher.
pub fn variance_class(b: &Block) -> u8 {
    let h = b.size / 2;
    let mut a = [
        variance_2(b, 0, 0, h),
        variance_2(b, 0, h, h),
        variance_2(b, h, 0, h),
        variance_2(b, h, h, h),
    ];
    let order = bubble_sort_desc_with_order(&mut a);
    crate::tables::match_order(order) as u8
}

/// `hurtgen_class()`: a 4-bit code, one bit per quadrant, set when that quadrant's mean
/// exceeds the block's overall mean.
pub fn hurtgen_class(b: &Block) -> u8 {
    let a = quadrant_sums(b);
    let h = b.size / 2;
    let total = a.iter().sum::<f64>();
    let mean = total / (b.size * b.size) as f64;
    let hh = (h * h) as f64;
    let mut clas = 0u8;
    for &qs in &a {
        clas <<= 1;
        if mean < qs / hh {
            clas |= 1;
        }
    }
    clas
}

/// `flips()`'s per-isometry source index — `flip_block[i][j] = block[source(i, j)]`.
/// Note this is the *destination-reads-from-source* direction, the opposite of
/// [`mars_codec::isometry::map`]'s source-writes-to-destination convention; both encode
/// the same eight isometries; this one matches `coding_func.c::flips` literally.
fn flip_source(iso: u8, i: usize, j: usize, size: usize) -> (usize, usize) {
    match iso {
        0 => (i, j),                       // IDENTITY
        1 => (j, size - i - 1),            // L_ROTATE90
        2 => (size - j - 1, i),            // R_ROTATE90
        3 => (size - i - 1, size - j - 1), // ROTATE180
        4 => (i, size - j - 1),            // R_VERTICAL
        5 => (size - i - 1, j),            // R_HORIZONTAL
        6 => (j, i),                       // F_DIAGONAL
        7 => (size - j - 1, size - i - 1), // S_DIAGONAL
        _ => unreachable!("isometry is 0..=7"),
    }
}

/// `flips(size, block, flip_block, iso)`.
pub fn flips(b: &Block, iso: u8) -> Block {
    let size = b.size;
    let mut out = vec![0.0f64; size * size];
    for i in 0..size {
        for j in 0..size {
            let (si, sj) = flip_source(iso, i, j, size);
            out[i * size + j] = b.at(si, sj);
        }
    }
    Block::new(size, out)
}

/// `ComputeMc()`: the block's centre-of-mass, in the `(x, y)` convention `index_func.c`
/// uses (note the axes are swapped and offset relative to a naive row/col centroid — this
/// is copied exactly, not simplified).
pub fn compute_mc(b: &Block) -> (f64, f64) {
    let size = b.size;
    let mut row = 0.0;
    let mut col = 0.0;
    let mut mass = 0.0;
    for i in 0..size {
        let mut a = 0.0;
        let mut bb = 0.0;
        for j in 0..size {
            a += b.at(i, j);
            bb += b.at(j, i);
            mass += b.at(j, i);
        }
        row += a * (i as f64 + 1.0);
        col += bb * (i as f64 + 1.0);
    }
    if mass != 0.0 {
        let y = (size as f64 + 1.0) / 2.0 - row / mass;
        let x = col / mass - (size as f64 + 1.0) / 2.0;
        (x, y)
    } else {
        (0.0, 0.0)
    }
}

fn theta_of(x: f64, y: f64) -> f64 {
    let theta = y.atan2(x);
    if theta >= 0.0 {
        theta
    } else {
        TWOPI + theta
    }
}

/// `ComputeMcVectors()`: the two polar-angle features used by MassCenter/Mc-Saupe.
///
/// **Simplification (`docs/decisions.md`):** the reference optionally box-downsamples
/// the block first via `ShrunkBlock`/`average_factor_mc[tip]`, controlled by
/// `shrunk_factor_mc` (default `1`). `ComputeAverageFactorMc` resolves that default to
/// `average_factor_mc[tip] == 1` for every `tip` actually reachable at `max_size <= 16`,
/// i.e. the shrink never fires under the 1998 defaults this project runs — so it is
/// omitted here rather than ported unused.
pub fn compute_mc_vectors(b: &Block) -> (f64, f64) {
    let (xx, yy) = compute_mc(b);
    let theta0 = theta_of(xx, yy);

    let size = b.size;
    let n = (size * size) as f64;
    let mean = b.data.iter().sum::<f64>() / n;
    let mut sq = vec![0.0f64; size * size];
    for (k, v) in b.data.iter().enumerate() {
        let d = v - mean;
        sq[k] = d * d;
    }
    let sq_block = Block::new(size, sq);
    let (xx2, yy2) = compute_mc(&sq_block);
    let theta1 = theta_of(xx2, yy2);
    (theta0, theta1)
}

/// `ShrunkBlock()`: box-downsample by `factor` (average `factor x factor` sub-blocks).
fn shrunk_block(b: &Block, factor: usize) -> Block {
    let newsize = b.size / factor;
    let mut out = vec![0.0f64; newsize * newsize];
    for i in 0..newsize {
        for j in 0..newsize {
            let mut tmp = 0.0;
            for k in 0..factor {
                for w in 0..factor {
                    tmp += b.at(i * factor + k, j * factor + w);
                }
            }
            out[i * newsize + j] = tmp / (factor * factor) as f64;
        }
    }
    Block::new(newsize, out)
}

/// `ComputeSaupeVectors()`: zero-mean, unit-L2-norm feature vector, after an optional
/// `ShrunkBlock` box-downsample when `average_factor > 1` (`ComputeFeatVectDimSaupe`,
/// [`feature_dims`]).
pub fn compute_saupe_vector(b: &Block, average_factor: usize) -> Vec<f32> {
    let shrunk;
    let block = if average_factor > 1 {
        shrunk = shrunk_block(b, average_factor);
        &shrunk
    } else {
        b
    };
    let n = (block.size * block.size) as f64;
    let sum: f64 = block.data.iter().sum();
    let sum2: f64 = block.data.iter().map(|v| v * v).sum();
    let s = sum / n;
    let v = (sum2 - sum * sum / n).sqrt();
    block.data.iter().map(|&x| ((x - s) / v) as f32).collect()
}

/// `ComputeFeatVectDimSaupe()`: for every `tip` (`size = 2^tip`) up to `max_size`, the
/// Saupe feature dimension and the `ShrunkBlock` factor that gets there, at the 1998
/// default `n_features = 16` and `shrunk_factor_saupe = 0` (disabled — the dimension
/// budget is fixed at 16, not derived from a fixed shrink factor).
///
/// Returns `(feat_dim, average_factor)` indexed by `tip` (index 0 unused, as in the C
/// array, since `tip == 0` is the no-search size-1 special case).
pub fn feature_dims(max_size: u32, n_features: u32) -> Vec<(usize, usize)> {
    let max_tip = (max_size as f64).log2().round() as u32;
    let mut out = vec![(0usize, 1usize); (max_tip + 1) as usize];
    for tip in 1..=max_tip {
        let size = 1u32 << tip;
        out[tip as usize] = saupe_dim_and_factor(size, n_features);
    }
    out
}

/// One size's Saupe feature dimension and `ShrunkBlock` factor — the per-`tip` body of
/// [`feature_dims`], exposed standalone so a single method's `index()` (which only ever
/// sees its own `DomainPool::size`, not the whole `min_size..=max_size` range) can
/// compute its own dimension without reconstructing the full table.
pub fn saupe_dim_and_factor(size: u32, n_features: u32) -> (usize, usize) {
    if size * size > n_features {
        let average_factor = (size as f64 / (n_features as f64).sqrt()).round() as usize;
        (n_features as usize, average_factor.max(1))
    } else {
        ((size * size) as usize, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_dims_match_1998_defaults_at_max_size_16() {
        let dims = feature_dims(16, 16);
        // tip=1 (size=2): 4 <= 16 -> dim=4, factor=1
        assert_eq!(dims[1], (4, 1));
        // tip=2 (size=4): 16 <= 16 (not >) -> dim=16, factor=1
        assert_eq!(dims[2], (16, 1));
        // tip=3 (size=8): 64 > 16 -> dim=16, factor=8/4=2
        assert_eq!(dims[3], (16, 2));
        // tip=4 (size=16): 256 > 16 -> dim=16, factor=16/4=4
        assert_eq!(dims[4], (16, 4));
    }

    #[test]
    fn newclass_is_identity_for_a_flat_block() {
        let b = Block::new(4, vec![10.0; 16]);
        // All quadrant sums equal -> ties broken by the exact bubble-sort order, giving
        // order [0,1,2,3] since no swap ever fires -> ORDERING row 23 (0,1,2,3,0,0).
        let (isom, clas) = newclass(&b);
        assert_eq!(isom, 0);
        assert_eq!(clas, 0);
    }

    #[test]
    fn flips_identity_is_a_no_op() {
        let data: Vec<f64> = (0..16).map(|x| x as f64).collect();
        let b = Block::new(4, data.clone());
        let f = flips(&b, 0);
        assert_eq!(f.data, data);
    }

    #[test]
    fn compute_saupe_vector_is_zero_mean_unit_norm() {
        let b = Block::new(2, vec![1.0, 2.0, 3.0, 4.0]);
        let v = compute_saupe_vector(&b, 1);
        let sum: f32 = v.iter().sum();
        assert!(sum.abs() < 1e-5, "sum={sum}");
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
    }
}
