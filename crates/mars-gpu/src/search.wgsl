// Step 7's exhaustive search kernel — one workgroup per range block.
//
// This is a line-for-line port of `crates/mars-codec/src/encode.rs`'s `search` +
// `fit_f32`, not an independent reimplementation: the moment accumulation stays
// non-negative-integer (`u32` — see docs/predictions.md P7.1: the worst case, `s2_x16`
// for a 32x32 block, is ~1.07e9, comfortably under u32::MAX ~4.29e9, so no 64-bit
// emulation across two lanes is needed), and the fit below mirrors `fit_f32`'s operation
// order exactly, because floating point is not associative and a reordered-but-equivalent
// expression is not the same computation.
//
// Winner selection must match the CPU's `best.is_none_or(|b| rms < b.rms)` — strict
// less-than, first-enumerated wins on an exact tie, enumerated `dom_row` outer, `dom_col`
// middle, isometry inner. Rather than special-casing the tie-break in the parallel
// reduction, every candidate carries a `key = domRowIdx * num_dom_cols * 8 + domColIdx * 8
// + isometry` — exactly the CPU's enumeration order — and "better" is defined as the
// lexicographic order on `(rms, key)`. That relation is a strict total order (no two
// distinct candidates in a block share a `key`, and `rms` is never NaN: the fit's `sum`
// is `max`-clamped to 0.0 before `sqrt`), and picking its minimum by repeated pairwise
// comparison is associative and commutative regardless of how the reduction tree is
// shaped — so any reduction order reproduces the CPU's tie-break, not just one
// particular schedule.

struct Params {
    width: u32,
    height: u32,
    size: u32,
    shift: u32,
    contracted_stride: u32,
    num_blocks_x: u32,
    num_dom_rows: u32, // 0 => no valid domain position at this size (CPU's checked_sub underflow)
    num_dom_cols: u32,
    bits_alfa: u32,
    bits_beta: u32,
    max_alfa: f32,
    _pad: u32,
};

struct Candidate {
    dom_row: u32,
    dom_col: u32,
    isometry: u32,
    qalfa: u32,
    qbeta: u32,
    rms_bits: u32, // bitcast<u32>(rms) -- see note on ordering below
    valid: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> image_px: array<u32>;
@group(0) @binding(2) var<storage, read> contracted: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<Candidate>;

const MAX_SIZE2: u32 = 1024u; // 32x32, this project's max_size (configs/*.json)
const WG_SIZE: u32 = 256u;

var<workgroup> range_block: array<u32, MAX_SIZE2>;
var<workgroup> shared_t0: u32;
var<workgroup> shared_t2: u32;

// Per-thread local winners, reduced by thread 0 after the barrier. `key` is the
// candidate's position in the CPU's canonical enumeration order (see module doc); it is
// what makes the reduction's tie-break well-defined regardless of scan order.
var<workgroup> local_rms: array<f32, WG_SIZE>;
var<workgroup> local_key: array<u32, WG_SIZE>;
var<workgroup> local_domrow: array<u32, WG_SIZE>;
var<workgroup> local_domcol: array<u32, WG_SIZE>;
var<workgroup> local_isom: array<u32, WG_SIZE>;
var<workgroup> local_qalfa: array<u32, WG_SIZE>;
var<workgroup> local_qbeta: array<u32, WG_SIZE>;
var<workgroup> local_has: array<u32, WG_SIZE>;

// §9's eight isometries — verbatim from `crates/mars-codec/src/isometry.rs::map`.
fn isometry_map(k: u32, u: u32, v: u32, size: u32) -> vec2<u32> {
    switch k {
        case 0u: { return vec2<u32>(u, v); }
        case 1u: { return vec2<u32>(size - 1u - v, u); }
        case 2u: { return vec2<u32>(v, size - 1u - u); }
        case 3u: { return vec2<u32>(size - 1u - u, size - 1u - v); }
        case 4u: { return vec2<u32>(u, size - 1u - v); }
        case 5u: { return vec2<u32>(size - 1u - u, v); }
        case 6u: { return vec2<u32>(v, u); }
        default: { return vec2<u32>(size - 1u - v, size - 1u - u); } // 7: S_DIAGONAL
    }
}

// `int(0.5 + x)` truncated toward zero then clamped -- verbatim `quantise_f32`. `x >= 0`
// always holds here, same as the CPU (alfa/beta are pre-clamped non-negative).
fn quantise(x: f32, max_v: f32) -> f32 {
    return clamp(trunc(0.5 + x), 0.0, max_v);
}

// Verbatim port of `fit_f32`. `s1_x4`/`s2_x16`/`t1_x4` are the raw accumulator sums
// (4x/16x/4x their mathematical value -- see `crates/mars-codec/src/encode.rs`'s module
// doc); `fit_f32` divides them out before using them, and skipping that step here was
// this kernel's first bug (caught by `mars-bench`'s synthetic differential test, not by
// inspection). Returns (qalfa, qbeta, rms).
fn fit(s0: f32, s1_x4: f32, s2_x16: f32, t0: f32, t1_x4: f32, t2: f32) -> vec3<f32> {
    let s1 = s1_x4 / 4.0;
    let s2 = s2_x16 / 16.0;
    let t1 = t1_x4 / 4.0;
    let det = s0 * s2 - s1 * s1;
    var alfa: f32;
    if (det == 0.0) {
        alfa = 0.0;
    } else {
        alfa = (s0 * t1 - s1 * t0) / det;
    }
    if (alfa < 0.0) {
        alfa = 0.0;
    }

    let bits_alfa_scale = f32(1u << params.bits_alfa);
    let max_qalfa = bits_alfa_scale - 1.0;
    let qalfa = quantise(alfa / params.max_alfa * bits_alfa_scale, max_qalfa);
    let alfa2 = qalfa / bits_alfa_scale * params.max_alfa;

    var beta = (t0 - alfa2 * s1) / s0;
    if (alfa2 > 0.0) {
        beta = beta + alfa2 * 255.0;
    }
    let max_qbeta = f32((1u << params.bits_beta) - 1u);
    let qbeta = quantise(beta / ((1.0 + abs(alfa2)) * 255.0) * max_qbeta, max_qbeta);
    var beta2 = qbeta / max_qbeta * ((1.0 + abs(alfa2)) * 255.0);
    if (alfa2 > 0.0) {
        beta2 = beta2 - alfa2 * 255.0;
    }

    let sum = t2 - 2.0 * alfa2 * t1 - 2.0 * beta2 * t0
        + alfa2 * alfa2 * s2
        + 2.0 * alfa2 * beta2 * s1
        + s0 * beta2 * beta2;
    let rms = sqrt(max(sum / s0, 0.0));
    return vec3<f32>(qalfa, qbeta, rms);
}

// True iff `a` beats `b` under the CPU's tie-break: strict rms less-than, else the
// earlier-enumerated (`key`) candidate. `has_a`/`has_b` let an "empty" local winner
// (no domain positions assigned to this thread) participate in the reduction safely.
fn better(rms_a: f32, key_a: u32, has_a: bool, rms_b: f32, key_b: u32, has_b: bool) -> bool {
    if (!has_a) {
        return false;
    }
    if (!has_b) {
        return true;
    }
    if (rms_a != rms_b) {
        return rms_a < rms_b;
    }
    return key_a < key_b;
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {
    let bx = wg_id.x;
    let by = wg_id.y;
    let size = params.size;
    let row0 = by * size;
    let col0 = bx * size;
    let block_index = by * params.num_blocks_x + bx;

    let size2 = size * size;
    // Thread 0 loads the range block and its own moments once; every other thread reads
    // this back after the barrier. `size2 <= 1024` so this is cheap next to the domain
    // sweep below.
    if (lidx == 0u) {
        var t0: u32 = 0u;
        var t2: u32 = 0u;
        for (var i: u32 = 0u; i < size; i = i + 1u) {
            let src = (row0 + i) * params.width + col0;
            for (var j: u32 = 0u; j < size; j = j + 1u) {
                let r = image_px[src + j];
                range_block[i * size + j] = r;
                t0 = t0 + r;
                t2 = t2 + r * r;
            }
        }
        shared_t0 = t0;
        shared_t2 = t2;
    }
    workgroupBarrier();

    let s0 = f32(size2);
    let t0f = f32(shared_t0);
    let t2f = f32(shared_t2);

    var has = false;
    var best_rms: f32 = 0.0;
    var best_key: u32 = 0u;
    var best_domrow: u32 = 0u;
    var best_domcol: u32 = 0u;
    var best_isom: u32 = 0u;
    var best_qalfa: u32 = 0u;
    var best_qbeta: u32 = 0u;

    let total_dom = params.num_dom_rows * params.num_dom_cols;
    var dom_idx = lidx;
    loop {
        if (dom_idx >= total_dom) {
            break;
        }
        let dom_row_idx = dom_idx / params.num_dom_cols;
        let dom_col_idx = dom_idx % params.num_dom_cols;
        let dom_row = dom_row_idx * params.shift;
        let dom_col = dom_col_idx * params.shift;
        let dr = dom_row / 2u;
        let dc = dom_col / 2u;

        // (Sum D, Sum D^2) over this domain position's size x size samples -- isometry
        // independent, computed once per domain position exactly as the CPU's
        // `domain_sums`.
        var s1: u32 = 0u;
        var s2: u32 = 0u;
        for (var u: u32 = 0u; u < size; u = u + 1u) {
            let base = (dr + u) * params.contracted_stride + dc;
            for (var v: u32 = 0u; v < size; v = v + 1u) {
                let d = contracted[base + v];
                s1 = s1 + d;
                s2 = s2 + d * d;
            }
        }
        let s1f = f32(s1);
        let s2f = f32(s2);

        for (var k: u32 = 0u; k < 8u; k = k + 1u) {
            var t1: u32 = 0u;
            for (var u: u32 = 0u; u < size; u = u + 1u) {
                let base = (dr + u) * params.contracted_stride + dc;
                for (var v: u32 = 0u; v < size; v = v + 1u) {
                    let d = contracted[base + v];
                    let ij = isometry_map(k, u, v, size);
                    t1 = t1 + range_block[ij.x * size + ij.y] * d;
                }
            }
            let fitted = fit(s0, s1f, s2f, t0f, f32(t1), t2f);
            let key = dom_idx * 8u + k;
            if (better(fitted.z, key, true, best_rms, best_key, has)) {
                has = true;
                best_rms = fitted.z;
                best_key = key;
                best_domrow = dom_row;
                best_domcol = dom_col;
                best_isom = k;
                best_qalfa = u32(fitted.x);
                best_qbeta = u32(fitted.y);
            }
        }
        dom_idx = dom_idx + WG_SIZE;
    }

    local_rms[lidx] = best_rms;
    local_key[lidx] = best_key;
    local_domrow[lidx] = best_domrow;
    local_domcol[lidx] = best_domcol;
    local_isom[lidx] = best_isom;
    local_qalfa[lidx] = best_qalfa;
    local_qbeta[lidx] = best_qbeta;
    local_has[lidx] = select(0u, 1u, has);
    workgroupBarrier();

    // A plain linear scan by thread 0: WG_SIZE (64) comparisons is negligible next to the
    // domain sweep each thread just did, and it sidesteps having to prove a tree
    // reduction preserves the tie-break -- the `better()` total order already guarantees
    // that for *any* merge order, so the simplest merge is also a correct one.
    if (lidx == 0u) {
        var win = 0u;
        var win_has = local_has[0] != 0u;
        for (var i: u32 = 1u; i < WG_SIZE; i = i + 1u) {
            let i_has = local_has[i] != 0u;
            if (better(local_rms[i], local_key[i], i_has, local_rms[win], local_key[win], win_has)) {
                win = i;
                win_has = i_has;
            }
        }
        var out: Candidate;
        if (win_has) {
            out.dom_row = local_domrow[win];
            out.dom_col = local_domcol[win];
            out.isometry = local_isom[win];
            out.qalfa = local_qalfa[win];
            out.qbeta = local_qbeta[win];
            out.rms_bits = bitcast<u32>(local_rms[win]);
            out.valid = 1u;
        } else {
            out.dom_row = 0u;
            out.dom_col = 0u;
            out.isometry = 0u;
            out.qalfa = 0u;
            out.qbeta = 0u;
            out.rms_bits = 0u;
            out.valid = 0u;
        }
        out._pad = 0u;
        results[block_index] = out;
    }
}
