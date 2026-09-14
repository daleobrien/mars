//! A manual diagnostic for Gate C's D33 finding (docs/decisions.md), not part of the
//! default suite -- needs `target/mars1/encmars` built (`just mars1`), which CI/other
//! sandboxes may not have. Parses Mars 1's own `.ifs` output and diffs it leaf-by-leaf
//! against `mars_search`'s Fisher leaves for the same image/params, to see whether the
//! two encoders agree on the quadtree partition shape and, where blocks match, on which
//! domain/isometry/qalfa/qbeta they picked. Run with:
//! `cargo test -p mars-bench --release --test gate_c_leaf_diff -- --ignored --nocapture`
//!
//! D33's own run of this found: of ~19k same-shape blocks, only ~51% picked the same
//! domain/isometry -- ruling out tie-breaking (reversing Fisher's bucket insertion order,
//! `crates/mars-search/src/fisher.rs`, changed nothing) and pointing at the candidate
//! *sets* themselves differing, not just which of an agreed-upon set wins. **Root-caused
//! in D34**: a genuine bug in the 1998 reference's `contraction()`
//! (`reference/mars1/index_func.c`) compares a column index against `image_height`
//! instead of `image_width`, corrupting `contract[][]` for roughly the right third of any
//! non-square image (Kodak is 768x512) and sending those domains to the wrong Fisher
//! bucket in the *real* C binary -- `mars_search::classify` was independently confirmed
//! correct three ways (source audit, raw-byte Python recomputation, and the instrumented
//! C binary's own printed values for the *range* side, which match Rust exactly). Not a
//! `mars_search` bug; see D34 for the full trace and why it explains the PSNR gap.

use std::collections::HashMap;
use std::path::Path;

use mars_bench::mars1::{encode, EncodeParams as Mars1EncodeParams, Mars1Binaries};
use mars_codec::encode::build_contracted;
use mars_codec::ifs;
use mars_core::io::read_raw;
use mars_search::fisher::Fisher;
use mars_search::SizedRetrievers;

#[test]
#[ignore = "needs target/mars1/encmars built via `just mars1`; manual Gate C diagnostic, not part of the default suite"]
fn leaf_diff_kodim01() {
    let root = Path::new("../..");
    let bins = Mars1Binaries::from_dir(root.join("target/mars1")).expect("mars1 built");
    let workdir = root.join("target/gate-c-leafdiff");
    std::fs::create_dir_all(&workdir).unwrap();
    let src = root.join("corpus/images/kodak-gray/kodim01.raw");
    std::fs::copy(&src, workdir.join("i.raw")).unwrap();

    let m1_params = Mars1EncodeParams::default();
    let outcome =
        encode(&bins, &workdir, "i.raw", "o.ifs", 768, 512, &m1_params, &[]).expect("mars1 encode");
    eprintln!(
        "mars1: transforms={} comparisons={}",
        outcome.stats.transforms, outcome.stats.comparisons
    );
    let ifs_bytes = std::fs::read(workdir.join("o.ifs")).unwrap();
    let (m1_hdr, m1_leaves) = ifs::parse(&ifs_bytes).expect("parse mars1 .ifs");

    let plane = read_raw(&src, 768, 512).expect("kodim01.raw");
    let contracted = build_contracted(&plane);
    let mars2_params = mars_codec::encode::EncodeParams {
        min_size: m1_params.min_size,
        max_size: m1_params.max_size,
        shift: m1_params.shift,
        bits_alfa: m1_params.bits_alfa,
        bits_beta: m1_params.bits_beta,
        max_alfa: m1_params.max_alfa,
        t_rms: m1_params.t_rms,
        zero_threshold: 0,
    };
    let retrievers = SizedRetrievers::build(
        &contracted,
        768,
        512,
        mars2_params.shift,
        mars2_params.min_size,
        mars2_params.max_size,
        || Box::new(Fisher::default()),
    );
    let (m2_hdr, m2_leaves, m2_evals, _picks) =
        mars_search::encode_image(&plane, &mars2_params, &retrievers);
    eprintln!("mars2: transforms={} evals={}", m2_leaves.len(), m2_evals);
    assert_eq!(m1_hdr.width, m2_hdr.width);
    assert_eq!(m1_hdr.height, m2_hdr.height);

    let key = |l: &ifs::Leaf| (l.row, l.col, l.size);
    let m1_map: HashMap<_, _> = m1_leaves.iter().map(|l| (key(l), l)).collect();
    let m2_map: HashMap<_, _> = m2_leaves.iter().map(|l| (key(l), l)).collect();

    let mut same_shape = 0usize;
    let mut same_pick = 0usize;
    let mut diff_pick = 0usize;
    let mut by_size: HashMap<u32, (usize, usize)> = HashMap::new();
    let mut diff_pick_examples: Vec<String> = Vec::new();
    for (k, m1_leaf) in &m1_map {
        if let Some(m2_leaf) = m2_map.get(k) {
            same_shape += 1;
            let same = m1_leaf.qalfa == m2_leaf.qalfa
                && m1_leaf.qbeta == m2_leaf.qbeta
                && m1_leaf.isometry == m2_leaf.isometry
                && m1_leaf.dom_row == m2_leaf.dom_row
                && m1_leaf.dom_col == m2_leaf.dom_col;
            let entry = by_size.entry(k.2).or_insert((0, 0));
            if same {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
            if same {
                same_pick += 1;
            } else {
                diff_pick += 1;
                if diff_pick_examples.len() < 10 {
                    diff_pick_examples.push(format!(
                        "{k:?}: mars1(qalfa={} qbeta={} iso={} dom=({},{})) vs mars2(qalfa={} qbeta={} iso={} dom=({},{}))",
                        m1_leaf.qalfa, m1_leaf.qbeta, m1_leaf.isometry, m1_leaf.dom_row, m1_leaf.dom_col,
                        m2_leaf.qalfa, m2_leaf.qbeta, m2_leaf.isometry, m2_leaf.dom_row, m2_leaf.dom_col,
                    ));
                }
            }
        }
    }
    let m1_only = m1_map.keys().filter(|k| !m2_map.contains_key(*k)).count();
    let m2_only = m2_map.keys().filter(|k| !m1_map.contains_key(*k)).count();

    eprintln!(
        "shape: m1_leaves={} m2_leaves={} same_shape_blocks={} m1_only={} m2_only={}",
        m1_leaves.len(),
        m2_leaves.len(),
        same_shape,
        m1_only,
        m2_only
    );
    eprintln!("of same-shape blocks: same_pick={same_pick} diff_pick={diff_pick}");
    let mut sizes: Vec<_> = by_size.keys().copied().collect();
    sizes.sort();
    for s in sizes {
        let (same, diff) = by_size[&s];
        eprintln!(
            "  size={s}: same={same} diff={diff} ({:.1}% same)",
            100.0 * same as f64 / (same + diff) as f64
        );
    }
    for e in &diff_pick_examples {
        eprintln!("  {e}");
    }

    // Decisive check on the first diff example, (84,548,4): mars1 picked dom=(444,660)
    // iso=6. Recompute, using this crate's own classify functions, what bucket the range
    // lands in and what bucket domain (444,660) lands in -- if they match, mars2's search
    // *did* consider mars1's winning domain and just scored it worse (a fit/dist bug, not
    // a classification bug); if they don't match, mars2 never even looked at it (a
    // classification bug).
    {
        use mars_search::classify::{flips, newclass, variance_class};
        use mars_search::RangeBlock;
        let (row, col, size) = (84u32, 548u32, 4u32);
        let px = plane.as_slice();
        let stride = plane.width();
        let mut pixels = vec![0u8; (size * size) as usize];
        for i in 0..size as usize {
            let src = (row as usize + i) * stride + col as usize;
            pixels[i * size as usize..(i + 1) * size as usize]
                .copy_from_slice(&px[src..src + size as usize]);
        }
        let range = RangeBlock {
            row,
            col,
            size,
            pixels: &pixels,
        };
        let rblock = range.as_block();
        let (r_isom, r_clas) = newclass(&rblock);
        let r_flipped = flips(&rblock, r_isom);
        let r_var = variance_class(&r_flipped);
        eprintln!("range (84,548,4): isom={r_isom} clas={r_clas} var_class={r_var}");

        let (dom_row, dom_col) = (444u32, 660u32);
        let (dr, dc) = ((dom_row / 2) as usize, (dom_col / 2) as usize);
        let mut ddata = vec![0.0f64; (size * size) as usize];
        for u in 0..size as usize {
            for v in 0..size as usize {
                ddata[u * size as usize + v] = f64::from(contracted.at(dr + u, dc + v));
            }
        }
        let dblock = mars_search::classify::Block::new(size as usize, ddata);
        let (d_iso, d_clas) = newclass(&dblock);
        let d_flipped = flips(&dblock, d_iso);
        let d_var = variance_class(&d_flipped);
        eprintln!(
            "domain (444,660,4) [mars1's pick]: dom_iso={d_iso} clas={d_clas} var_class={d_var}"
        );
        eprintln!(
            "same bucket as range? {}",
            r_clas == d_clas && r_var == d_var
        );
        eprintln!(
            "MAPPING[{r_isom}][{d_iso}] = {} (mars1 reported iso=6)",
            mars_search::tables::MAPPING[r_isom as usize][d_iso as usize]
        );

        // Is range's own bucket [clas=1][var_class=2] (pre-fallback) actually populated,
        // or would the "no empty class" fallback have overwritten it with some other
        // bucket's contents entirely? Recompute the full size-4 bucket population by hand.
        let mut bucket_counts = vec![vec![0usize; 24]; 3];
        let max_row = 512u32 - 2 * size;
        let max_col = 768u32 - 2 * size;
        let mut r = 0u32;
        while r <= max_row {
            let mut c = 0u32;
            while c <= max_col {
                let (dr2, dc2) = ((r / 2) as usize, (c / 2) as usize);
                let mut data2 = vec![0.0f64; (size * size) as usize];
                for u in 0..size as usize {
                    for v in 0..size as usize {
                        data2[u * size as usize + v] = f64::from(contracted.at(dr2 + u, dc2 + v));
                    }
                }
                let b2 = mars_search::classify::Block::new(size as usize, data2);
                let (iso2, clas2) = newclass(&b2);
                let flipped2 = flips(&b2, iso2);
                let var2 = variance_class(&flipped2);
                bucket_counts[clas2 as usize][var2 as usize] += 1;
                c += 4;
            }
            r += 4;
        }
        eprintln!(
            "bucket[clas=1][var=2] population (pre-fallback): {}",
            bucket_counts[1][2]
        );
        eprintln!(
            "bucket[clas=2][var=2] population (domain's own bucket): {}",
            bucket_counts[2][2]
        );
        let total: usize = bucket_counts.iter().flatten().sum();
        eprintln!("total domains indexed at size=4: {total}");
        for (ci, row) in bucket_counts.iter().enumerate() {
            for (vi, &n) in row.iter().enumerate() {
                if n == 0 {
                    eprintln!("  EMPTY bucket [{ci}][{vi}]");
                }
            }
        }
    }
}
