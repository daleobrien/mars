//! Bounded, corpus-free APCC ordering, fitting, and production integration tests.

use mars_codec::encode::{
    self, build_contracted, encode_image_with_search, Candidate as Fit, EncodeOptions,
    EncodeOutcome, EncodeParams, RawMoments, SearchOutcome, SearchProvider, SearchRequest,
};
use mars_codec::{ifs, mars_format};
use mars_core::Plane;
use mars_search::apcc::{apcc_key, Apcc, ApccConfig, ApccSearchProvider};
use mars_search::classify::{flips, newclass, variance_class, Block};
use mars_search::tables::MAPPING;
use mars_search::{search_block_fitted, Candidate, CandidateRetriever, DomainPool, RangeBlock};

fn image(w: usize, h: usize) -> Plane {
    let mut state = 0xc0ffee_u64;
    Plane::from_vec(
        w,
        h,
        (0..w * h)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect(),
    )
}

fn params(lambda: Option<f64>) -> EncodeParams {
    EncodeParams {
        min_size: 4,
        max_size: 16,
        shift: 2,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda,
    }
}

fn pixels(image: &Plane, row: u32, col: u32, size: u32) -> Vec<u8> {
    (row as usize..(row + size) as usize)
        .flat_map(|r| {
            image.as_slice()
                [r * image.width() + col as usize..r * image.width() + (col + size) as usize]
                .iter()
                .copied()
        })
        .collect()
}

fn indexed(pool: &DomainPool<'_>, budget: usize) -> Apcc {
    let mut index = Apcc::new(ApccConfig { budget });
    index.index(pool);
    index
}

fn positions(candidates: &[Candidate]) -> Vec<(u32, u32)> {
    assert_eq!(candidates.len() % 8, 0);
    candidates
        .chunks_exact(8)
        .map(|group| {
            let pos = (group[0].dom_row, group[0].dom_col);
            assert!(group.iter().all(|c| (c.dom_row, c.dom_col) == pos));
            let mut isos = group.iter().map(|c| c.isometry).collect::<Vec<_>>();
            isos.sort_unstable();
            assert_eq!(isos, (0..8).collect::<Vec<_>>());
            pos
        })
        .collect()
}

fn assert_fit(a: Option<Fit>, b: Option<Fit>) {
    match (a, b) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(
                (a.dom_row, a.dom_col, a.isometry, a.qalfa, a.qbeta),
                (b.dom_row, b.dom_col, b.isometry, b.qalfa, b.qbeta)
            );
            assert_eq!(a.rms.to_bits(), b.rms.to_bits());
            assert_eq!(a.moments, b.moments);
            assert!(a.rms.is_finite());
        }
        other => panic!("different fits: {other:?}"),
    }
}

// Independent brute-force fit driver: computes real cross terms for every proposed
// orientation, rather than using search_block_fitted or trusting returned moments.
fn brute_fit(
    pool: &DomainPool<'_>,
    range: &RangeBlock<'_>,
    candidates: &[Candidate],
    p: &EncodeParams,
) -> Option<Fit> {
    let mut best: Option<Fit> = None;
    for c in candidates {
        let (dr, dc, n) = (
            (c.dom_row / 2) as usize,
            (c.dom_col / 2) as usize,
            range.size as usize,
        );
        let (s1_x4, s2_x16) = encode::domain_sums(pool.contracted, dr, dc, n);
        let (t0, t2) = range.t0_t2();
        let moments = RawMoments {
            s0: i64::from(range.size * range.size),
            s1_x4,
            s2_x16,
            t0,
            t2,
            t1_x4: encode::cross_term(pool.contracted, dr, dc, n, c.isometry, range.pixels),
        };
        let (qalfa, qbeta, rms) = encode::fit_f64(moments, p.max_alfa, p.bits_alfa, p.bits_beta);
        if best.is_none_or(|b| rms < b.rms) {
            best = Some(Fit {
                dom_row: c.dom_row,
                dom_col: c.dom_col,
                isometry: c.isometry,
                qalfa,
                qbeta,
                rms,
                moments,
            });
        }
    }
    best
}

// Independently reconstruct the full matching bucket, with the documented synthetic
// reference and canonicalization. No fallback and no APCC candidate selection here.
fn bucket(pool: &DomainPool<'_>, range: &RangeBlock<'_>) -> Vec<Candidate> {
    let (isom, clas) = newclass(&range.as_block());
    let canonical = flips(&range.as_block(), isom);
    let var = variance_class(&canonical);
    let h = range.size as usize / 2;
    let ranks = [
        [3.0, 2.0, 1.0, 0.0],
        [3.0, 2.0, 0.0, 1.0],
        [3.0, 1.0, 0.0, 2.0],
    ][clas as usize];
    let reference = Block::new(
        range.size as usize,
        (0..range.size as usize)
            .flat_map(|r| {
                (0..range.size as usize).map(move |c| {
                    4.0 * ranks[2 * (r / h) + c / h] + ((r % h) * h + c % h) as f64 / (h * h) as f64
                })
            })
            .collect(),
    );
    if apcc_key(&canonical, &reference).is_none() {
        return Vec::new();
    }
    let mut domains = Vec::new();
    for (row, col) in pool.domain_positions() {
        let block = pool.domain_block(row, col);
        let (iso, class) = newclass(&block);
        let canonical = flips(&block, iso);
        if class == clas && variance_class(&canonical) == var {
            if let Some(key) = apcc_key(&canonical, &reference) {
                domains.push((key, row, col, iso));
            }
        }
    }
    domains.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| (a.1, a.2).cmp(&(b.1, b.2)))
    });
    domains
        .into_iter()
        .flat_map(|(_, dom_row, dom_col, iso)| {
            let first = MAPPING[isom as usize][iso as usize];
            std::iter::once(first)
                .chain((0..8).filter(move |&i| i != first))
                .map(move |isometry| Candidate {
                    dom_row,
                    dom_col,
                    isometry,
                })
        })
        .collect()
}

#[test]
fn pinned_order_ties_lower_bound_and_wrapped_neighborhood() {
    // Five nonoverlapping 4x4 domains, each an expanded 2x2 block. Last is flat.
    let blocks = [
        [12, 8, 4, 0],
        [12, 11, 1, 0],
        [12, 6, 5, 0],
        [12, 8, 4, 0],
        [7; 4],
    ];
    let image = Plane::from_vec(
        20,
        4,
        (0..80)
            .map(|i| {
                let (r, c) = (i / 20, i % 20);
                blocks[c / 4][2 * (r / 2) + (c % 4) / 2]
            })
            .collect(),
    );
    let contracted = build_contracted(&image);
    let pool = DomainPool {
        contracted: &contracted,
        size: 2,
        shift: 4,
        image_width: 20,
        image_height: 4,
    };
    let range = RangeBlock {
        row: 0,
        col: 0,
        size: 2,
        pixels: &blocks[0],
    };
    let full = indexed(&pool, usize::MAX)
        .candidates(&range)
        .collect::<Vec<_>>();
    assert_eq!(positions(&full), [(0, 4), (0, 8), (0, 0), (0, 12)]);
    assert!(full.chunks_exact(8).all(|group| group[0].isometry == 0));
    for (budget, expected) in [
        (0, vec![]),
        (1, vec![(0, 0)]),
        (2, vec![(0, 8), (0, 0)]),
        (3, vec![(0, 8), (0, 0), (0, 12)]),
        (4, vec![(0, 4), (0, 8), (0, 0), (0, 12)]),
        (5, vec![(0, 4), (0, 8), (0, 0), (0, 12)]),
    ] {
        assert_eq!(
            positions(
                &indexed(&pool, budget)
                    .candidates(&range)
                    .collect::<Vec<_>>()
            ),
            expected
        );
    }
    let low = RangeBlock {
        pixels: &blocks[1],
        ..range
    };
    assert_eq!(
        positions(&indexed(&pool, 2).candidates(&low).collect::<Vec<_>>()),
        [(0, 4), (0, 12)]
    );
    // Key above every bucket key: insertion at len wraps to zero as well.
    let above = RangeBlock {
        pixels: &[12, 8, 4, 0],
        ..range
    };
    let only_low = Plane::from_vec(
        4,
        4,
        (0..16)
            .map(|i| blocks[1][2 * (i / 8) + (i % 4) / 2])
            .collect(),
    );
    let low_contracted = build_contracted(&only_low);
    let low_pool = DomainPool {
        contracted: &low_contracted,
        image_width: 4,
        ..pool
    };
    assert_eq!(
        positions(&indexed(&low_pool, 1).candidates(&above).collect::<Vec<_>>()),
        [(0, 0)]
    );
    // Above the final key in a nontrivial bucket: insertion len wraps, selecting
    // the final and first entries, then restoring bucket order.
    let repeated = Plane::from_vec(
        12,
        4,
        (0..48)
            .map(|i| blocks[1][2 * (i / 24) + (i % 4) / 2])
            .collect(),
    );
    let repeated_contracted = build_contracted(&repeated);
    let repeated_pool = DomainPool {
        contracted: &repeated_contracted,
        image_width: 12,
        ..pool
    };
    assert_eq!(
        positions(
            &indexed(&repeated_pool, 2)
                .candidates(&above)
                .collect::<Vec<_>>()
        ),
        [(0, 0), (0, 8)]
    );
    // Absent outer class: no Fisher-style replacement.
    let absent = RangeBlock {
        pixels: &[12, 0, 4, 8],
        ..range
    };
    assert!(indexed(&pool, usize::MAX)
        .candidates(&absent)
        .next()
        .is_none());
}

#[test]
fn finite_keys_flat_domains_flat_ranges_and_reindexing() {
    let reference = Block::new(2, vec![12.0, 8.0, 4.0, 0.0]);
    let inverse = Block::new(2, vec![0.0, 4.0, 8.0, 12.0]);
    assert!((apcc_key(&inverse, &reference).unwrap() - 1.0).abs() < 1e-15);
    assert_eq!(
        apcc_key(&inverse, &reference),
        apcc_key(&reference, &reference)
    );
    for value in [0.0, 173.0, f64::NAN, f64::INFINITY] {
        let flat = Block::new(2, vec![value; 4]);
        assert_eq!(apcc_key(&flat, &reference), None);
        assert_eq!(apcc_key(&reference, &flat), None);
    }
    assert_eq!(
        apcc_key(&Block::new(0, vec![]), &Block::new(0, vec![])),
        None
    );
    assert_eq!(apcc_key(&Block::new(1, vec![1.0]), &reference), None);
    let image = image(17, 19);
    let contracted = build_contracted(&image);
    let pool = DomainPool {
        contracted: &contracted,
        size: 2,
        shift: 2,
        image_width: 17,
        image_height: 19,
    };
    let mut index = indexed(&pool, usize::MAX);
    let flat_range = RangeBlock {
        row: 0,
        col: 0,
        size: 2,
        pixels: &[173; 4],
    };
    assert_eq!(index.candidates(&flat_range).count(), 0);
    let flat = Plane::from_vec(17, 19, vec![173; 17 * 19]);
    let contracted = build_contracted(&flat);
    index.index(&DomainPool {
        contracted: &contracted,
        ..pool
    });
    let varied = RangeBlock {
        pixels: &[12, 8, 4, 0],
        ..flat_range
    };
    assert_eq!(index.candidates(&varied).count(), 0);
    let result = search_block_fitted(&index, &contracted, &varied, 1.0, 4, 7);
    assert!(result.candidate.is_none());
    assert_eq!(result.evals, 0);
    assert_eq!(
        Apcc::new(ApccConfig { budget: 5 })
            .candidates(&varied)
            .count(),
        0
    );
}

#[test]
fn all_budgets_nested_full_bucket_exhaustive_evals_and_real_moments() {
    let image = image(20, 18);
    let contracted = build_contracted(&image);
    let p = params(None);
    let mut saw_nonempty = false;
    for shift in [2, 4] {
        for size in [2, 4, 8, 16] {
            let pool = DomainPool {
                contracted: &contracted,
                size,
                shift,
                image_width: 20,
                image_height: 18,
            };
            for (row, col) in [(0, 0), (2, 2)] {
                let px = pixels(&image, row, col, size);
                let range = RangeBlock {
                    row,
                    col,
                    size,
                    pixels: &px,
                };
                let full = bucket(&pool, &range);
                let full_positions = positions(&full);
                saw_nonempty |= !full.is_empty();
                let mut previous = Vec::new();
                for budget in (0..=full_positions.len() + 1).chain(std::iter::once(usize::MAX)) {
                    let index = indexed(&pool, budget);
                    let candidates = index.candidates(&range).collect::<Vec<_>>();
                    let selected = positions(&candidates);
                    assert_eq!(selected.len(), budget.min(full_positions.len()));
                    assert!(previous.iter().all(|pos| selected.contains(pos)));
                    let indices = selected
                        .iter()
                        .map(|pos| full_positions.iter().position(|p| p == pos).unwrap())
                        .collect::<Vec<_>>();
                    assert!(indices.windows(2).all(|w| w[0] < w[1]));
                    let got = search_block_fitted(
                        &index,
                        &contracted,
                        &range,
                        p.max_alfa,
                        p.bits_alfa,
                        p.bits_beta,
                    );
                    assert_eq!(got.evals, 8 * selected.len() as u64);
                    assert_fit(got.candidate, brute_fit(&pool, &range, &candidates, &p));
                    if budget >= full_positions.len() {
                        assert_eq!(candidates, full);
                        assert_fit(got.candidate, brute_fit(&pool, &range, &full, &p));
                    }
                    previous = selected;
                }
            }
        }
    }
    assert!(saw_nonempty);
}

#[test]
fn negative_correlation_retrieves_but_does_not_invent_negative_contrast() {
    // All quadrants have identical sums/variances, so inversion preserves the bucket
    // and canonicalization. Identity covariance is strictly negative.
    let domain = [10_u8, 230, 170, 90];
    let image = Plane::from_vec(
        8,
        8,
        (0..64)
            .map(|i| domain[2 * ((i / 16) % 2) + (i % 8) / 2 % 2])
            .collect(),
    );
    let contracted = build_contracted(&image);
    let pool = DomainPool {
        contracted: &contracted,
        size: 4,
        shift: 2,
        image_width: 8,
        image_height: 8,
    };
    let px = (0..16)
        .map(|i| 250 - domain[2 * ((i / 4) % 2) + i % 2])
        .collect::<Vec<_>>();
    let range = RangeBlock {
        row: 0,
        col: 0,
        size: 4,
        pixels: &px,
    };
    let index = indexed(&pool, 1);
    let candidates = index.candidates(&range).collect::<Vec<_>>();
    assert_eq!(positions(&candidates), [(0, 0)]);
    assert_eq!(candidates[0].isometry, 0);
    let p = params(None);
    let negative = brute_fit(&pool, &range, &candidates[..1], &p).unwrap();
    let m = negative.moments;
    assert!(m.s0 * m.t1_x4 - m.s1_x4 * m.t0 < 0);
    assert_eq!(negative.qalfa, 0);
    let got = search_block_fitted(
        &index,
        &contracted,
        &range,
        p.max_alfa,
        p.bits_alfa,
        p.bits_beta,
    );
    assert_eq!(got.evals, 8);
    assert_fit(got.candidate, brute_fit(&pool, &range, &candidates, &p));
    let fitted = got.candidate.unwrap();
    assert!(fitted.qalfa < 1 << p.bits_alfa);
    assert_eq!(
        (fitted.qalfa, fitted.qbeta, fitted.rms),
        encode::fit_f64(fitted.moments, p.max_alfa, p.bits_alfa, p.bits_beta)
    );
}

#[test]
fn provider_query_order_threads_and_independent_density_indexes() {
    use rayon::prelude::*;
    let image = image(65, 67);
    let contracted = build_contracted(&image);
    let p = EncodeParams {
        min_size: 32,
        max_size: 32,
        shift: 16,
        ..params(Some(10.0))
    };
    let options = EncodeOptions {
        adaptive_density: true,
        ..EncodeOptions::default()
    };
    let provider =
        ApccSearchProvider::build(&image, &contracted, &p, &options, ApccConfig { budget: 3 });
    let queries = [2, 4, 8, 16, 32]
        .into_iter()
        .flat_map(|size| [16, 32].map(|shift| (size, shift)))
        .collect::<Vec<_>>();
    let run = |&(size, shift): &(u32, u32)| {
        let request = SearchRequest {
            image: &image,
            contracted: &contracted,
            params: &p,
            row: 0,
            col: 0,
            size,
            shift,
        };
        let got = provider.search(&request);
        let direct = ApccSearchProvider::build(
            &image,
            &contracted,
            &EncodeParams { shift, ..p },
            &EncodeOptions::default(),
            ApccConfig { budget: 3 },
        );
        let expected = direct.search(&request);
        assert_eq!(got.evals, expected.evals);
        assert_fit(got.candidate, expected.candidate);
        got
    };
    let baseline = queries.iter().map(run).collect::<Vec<_>>();
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let reverse = queries.iter().copied().rev().collect::<Vec<_>>();
        let results = pool.install(|| reverse.par_iter().map(run).collect::<Vec<SearchOutcome>>());
        for (a, b) in results.iter().rev().zip(&baseline) {
            assert_eq!(a.evals, b.evals);
            assert_fit(a.candidate, b.candidate);
        }
    }
}

fn roundtrip(result: &EncodeOutcome) -> Vec<u8> {
    let bytes = mars_format::write(&result.header, &result.leaves).unwrap();
    let (header, leaves) = mars_format::read(&bytes).unwrap();
    assert_eq!(header, result.header);
    assert_eq!(leaves, result.leaves);
    assert_eq!(bytes, mars_format::write(&header, &leaves).unwrap());
    assert_eq!(
        ifs::decode_iterative(&header, &leaves, 3).as_slice(),
        ifs::decode_iterative(&result.header, &result.leaves, 3).as_slice()
    );
    bytes
}

#[test]
fn production_threshold_rd_density_mode3_all_sizes_tiny_odd_and_threads() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let two = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let mut saw_residual = false;
    let mut saw_search = false;
    for (w, h, size) in [
        (1, 9, 1),
        (9, 1, 1),
        (2, 2, 2),
        (9, 11, 2),
        (17, 19, 4),
        (17, 19, 8),
        (33, 35, 16),
        (65, 67, 32),
    ] {
        let image = image(w, h);
        for lambda in [None, Some(0.0), Some(10.0)] {
            let p = EncodeParams {
                min_size: size,
                max_size: size,
                shift: 8,
                ..params(lambda)
            };
            for adaptive_density in [false, true] {
                let options = EncodeOptions {
                    adaptive_density,
                    allowed_modes: [false, false, false, true],
                    ..EncodeOptions::default()
                };
                for budget in [0, 3] {
                    let run = || {
                        encode_image_with_search(&image, &p, &options, |contracted| {
                            Box::new(ApccSearchProvider::build(
                                &image,
                                contracted,
                                &p,
                                &options,
                                ApccConfig { budget },
                            ))
                        })
                    };
                    let a = one.install(run);
                    let b = two.install(run);
                    assert_eq!(roundtrip(&a), roundtrip(&b));
                    assert_eq!(a.counters.search_evals, b.counters.search_evals);
                    assert_eq!(a.counters.warmup_evals, b.counters.warmup_evals);
                    assert_eq!(a.mode_stats.leaf_modes, b.mode_stats.leaf_modes);
                    assert_eq!(a.mode_stats.split_decisions, b.mode_stats.split_decisions);
                    if budget == 0 {
                        assert_eq!(a.counters.search_evals, 0);
                    }
                    saw_search |= a.counters.search_evals > 0;
                    saw_residual |= a
                        .leaves
                        .iter()
                        .any(|leaf| leaf.mode == 3 && !leaf.residual.is_empty());
                }
            }
        }
    }
    assert!(saw_search && saw_residual);
}
