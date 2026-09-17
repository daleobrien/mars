//! Bounded synthetic regressions for the versioned grayscale random provider.

use mars_codec::encode::{
    self, build_contracted, encode_image_with_search, Candidate, EncodeOptions, EncodeOutcome,
    EncodeParams, ExhaustiveSearch, SearchProvider, SearchRequest,
};
use mars_codec::{ifs, mars_format};
use mars_core::Plane;
use mars_search::random::{image_fingerprint, RandomConfig, RandomSearchProvider, RNG_VERSION};
use mars_search::DomainPool;

fn image(w: usize, h: usize) -> Plane {
    Plane::from_vec(
        w,
        h,
        (0..w * h).map(|i| ((i * 37 + 11) % 256) as u8).collect(),
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

fn assert_candidate(a: Option<Candidate>, b: Option<Candidate>) {
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
        other => panic!("candidate mismatch: {other:?}"),
    }
}

fn assert_encoding(a: &EncodeOutcome, b: &EncodeOutcome) {
    assert_eq!(a.header, b.header);
    assert_eq!(a.leaves, b.leaves);
    assert_eq!(a.counters.search_evals, b.counters.search_evals);
    assert_eq!(a.counters.warmup_evals, b.counters.warmup_evals);
    assert_eq!(a.mode_stats.leaf_modes, b.mode_stats.leaf_modes);
    assert_eq!(a.mode_stats.split_decisions, b.mode_stats.split_decisions);
    assert_eq!(a.mode_stats.leaf_decisions, b.mode_stats.leaf_decisions);
    assert_eq!(
        mars_format::write(&a.header, &a.leaves).unwrap(),
        mars_format::write(&b.header, &b.leaves).unwrap()
    );
}

fn roundtrip(result: &EncodeOutcome) {
    let bytes = mars_format::write(&result.header, &result.leaves).unwrap();
    let (header, leaves) = mars_format::read(&bytes).unwrap();
    assert_eq!(header, result.header);
    assert_eq!(leaves, result.leaves);
    assert_eq!(bytes, mars_format::write(&header, &leaves).unwrap());
    assert_eq!(
        ifs::decode_iterative(&header, &leaves, 3).as_slice(),
        ifs::decode_iterative(&result.header, &result.leaves, 3).as_slice()
    );
}

#[test]
fn pinned_version_fingerprint_query_seed_and_samples() {
    assert_eq!(RNG_VERSION, "mars-random-gray-fnv1a64-splitmix64-fy-v1");
    let image = image(20, 18);
    assert_eq!(image_fingerprint(&image), 0xe61cee0d282e5ea9);
    let contracted = build_contracted(&image);
    let p = params(None);
    let provider = RandomSearchProvider::build(
        &image,
        &contracted,
        &p,
        &EncodeOptions::default(),
        RandomConfig {
            budget: 8,
            seed: 42,
        },
    );
    assert_eq!(provider.query_seed(2, 4, 2, 2), 0xa278cb2e9c77e249);
    // Independently computed with a dense forward Fisher–Yates reference.
    assert_eq!(
        provider.sampled_positions(2, 4, 2, 2),
        vec![
            (0, 4),
            (0, 12),
            (4, 4),
            (4, 16),
            (6, 2),
            (6, 6),
            (12, 14),
            (14, 12)
        ]
    );
    let reshaped = Plane::from_vec(18, 20, image.as_slice().to_vec());
    assert_ne!(image_fingerprint(&image), image_fingerprint(&reshaped));
    let mut changed = image.as_slice().to_vec();
    changed[0] ^= 1;
    assert_ne!(
        image_fingerprint(&image),
        image_fingerprint(&Plane::from_vec(20, 18, changed))
    );
    for (row, col, size, shift) in [(4, 4, 2, 2), (2, 6, 2, 2), (2, 4, 4, 2), (2, 4, 2, 4)] {
        assert_ne!(
            provider.query_seed(row, col, size, shift),
            provider.query_seed(2, 4, 2, 2)
        );
    }
    let other = RandomSearchProvider::build(
        &image,
        &contracted,
        &p,
        &EncodeOptions::default(),
        RandomConfig {
            budget: 8,
            seed: 43,
        },
    );
    assert_ne!(
        other.query_seed(2, 4, 2, 2),
        provider.query_seed(2, 4, 2, 2)
    );
    assert_ne!(
        other.sampled_positions(2, 4, 2, 2),
        provider.sampled_positions(2, 4, 2, 2)
    );
}

#[test]
fn every_budget_is_nested_unique_legal_and_exactly_eight_evals_per_position() {
    let image = image(20, 18);
    let contracted = build_contracted(&image);
    let p = params(Some(10.0));
    let options = EncodeOptions {
        adaptive_density: true,
        ..EncodeOptions::default()
    };
    for shift in [2, 4] {
        for size in [2, 4, 8, 16] {
            let legal = DomainPool {
                contracted: &contracted,
                size,
                shift,
                image_width: 20,
                image_height: 18,
            }
            .domain_positions()
            .collect::<Vec<_>>();
            for seed in [0, 42, u64::MAX] {
                let mut previous = Vec::new();
                for budget in 0..=legal.len() + 1 {
                    let provider = RandomSearchProvider::build(
                        &image,
                        &contracted,
                        &p,
                        &options,
                        RandomConfig { budget, seed },
                    );
                    let samples = provider.sampled_positions(0, 0, size, shift);
                    assert_eq!(samples.len(), budget.min(legal.len()));
                    assert!(samples.windows(2).all(|pair| pair[0] < pair[1]));
                    assert!(samples.iter().all(|pos| legal.binary_search(pos).is_ok()));
                    assert!(previous
                        .iter()
                        .all(|pos| samples.binary_search(pos).is_ok()));
                    if budget >= legal.len() {
                        assert_eq!(samples, legal);
                    }
                    let request = SearchRequest {
                        image: &image,
                        contracted: &contracted,
                        params: &p,
                        row: 0,
                        col: 0,
                        size,
                        shift,
                    };
                    let outcome = provider.search(&request);
                    assert_eq!(outcome.evals, 8 * samples.len() as u64);
                    assert_eq!(outcome.candidate.is_none(), samples.is_empty());
                    if let Some(c) = outcome.candidate {
                        assert!(samples.contains(&(c.dom_row, c.dom_col)));
                        assert!(c.isometry < 8);
                        assert_eq!(
                            (c.qalfa, c.qbeta, c.rms),
                            encode::fit_f64(c.moments, p.max_alfa, p.bits_alfa, p.bits_beta)
                        );
                    }
                    if budget >= legal.len() {
                        let exhaustive = ExhaustiveSearch.search(&request);
                        assert_eq!(outcome.evals, exhaustive.evals);
                        assert_candidate(outcome.candidate, exhaustive.candidate);
                    }
                    previous = samples;
                }
            }
        }
    }
}

#[test]
fn constant_ties_choose_first_reference_position_and_identity_isometry() {
    let image = Plane::from_vec(20, 18, vec![173; 360]);
    let contracted = build_contracted(&image);
    let p = params(None);
    let provider = RandomSearchProvider::build(
        &image,
        &contracted,
        &p,
        &EncodeOptions::default(),
        RandomConfig { budget: 7, seed: 1 },
    );
    let first = provider.sampled_positions(0, 0, 4, 2)[0];
    let outcome = provider.search(&SearchRequest {
        image: &image,
        contracted: &contracted,
        params: &p,
        row: 0,
        col: 0,
        size: 4,
        shift: 2,
    });
    assert_eq!(outcome.evals, 56);
    let c = outcome.candidate.unwrap();
    assert_eq!((c.dom_row, c.dom_col, c.isometry), (first.0, first.1, 0));
}

#[test]
fn query_order_and_parallel_scheduling_do_not_change_samples_or_fits() {
    use rayon::prelude::*;
    fn require_sync<T: Sync>() {}
    require_sync::<RandomSearchProvider>();
    let image = image(33, 31);
    let contracted = build_contracted(&image);
    let p = params(Some(10.0));
    let options = EncodeOptions {
        adaptive_density: true,
        ..EncodeOptions::default()
    };
    let provider = RandomSearchProvider::build(
        &image,
        &contracted,
        &p,
        &options,
        RandomConfig {
            budget: 5,
            seed: 42,
        },
    );
    let queries = [(0, 0, 4, 2), (4, 8, 8, 4), (2, 6, 2, 2), (0, 0, 16, 4)];
    let run = |&(row, col, size, shift): &(u32, u32, u32, u32)| {
        (
            provider.sampled_positions(row, col, size, shift),
            provider.search(&SearchRequest {
                image: &image,
                contracted: &contracted,
                params: &p,
                row,
                col,
                size,
                shift,
            }),
        )
    };
    let baseline = queries.iter().map(run).collect::<Vec<_>>();
    for threads in [1, 2] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let reverse = queries.into_iter().rev().collect::<Vec<_>>();
        let results = pool.install(|| reverse.par_iter().map(run).collect::<Vec<_>>());
        for ((samples, result), (want_samples, want)) in results.iter().rev().zip(&baseline) {
            assert_eq!(samples, want_samples);
            assert_eq!(result.evals, want.evals);
            assert_candidate(result.candidate, want.candidate);
        }
    }
}

#[test]
fn full_budget_production_bytes_match_exhaustive_with_rd_density_and_residual() {
    let mut saw_residual = false;
    for (w, h) in [(1, 9), (9, 1), (2, 2), (8, 8), (17, 19)] {
        let image = image(w, h);
        for lambda in [None, Some(0.0), Some(10.0)] {
            let p = params(lambda);
            for adaptive_density in [false, true] {
                for allowed_modes in [[true; 4], [false, false, false, true]] {
                    let options = EncodeOptions {
                        adaptive_density,
                        allowed_modes,
                        ..EncodeOptions::default()
                    };
                    let reference = encode_image_with_search(&image, &p, &options, |_| {
                        Box::new(ExhaustiveSearch)
                    });
                    let result = encode_image_with_search(&image, &p, &options, |contracted| {
                        Box::new(RandomSearchProvider::build(
                            &image,
                            contracted,
                            &p,
                            &options,
                            RandomConfig {
                                budget: usize::MAX,
                                seed: 42,
                            },
                        ))
                    });
                    assert_encoding(&result, &reference);
                    roundtrip(&result);
                    saw_residual |= result
                        .leaves
                        .iter()
                        .any(|leaf| leaf.mode == 3 && !leaf.residual.is_empty());
                }
            }
        }
    }
    assert!(saw_residual);
}

#[test]
fn bounded_and_zero_budget_production_is_thread_deterministic_without_fallback() {
    let image = image(17, 19);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let two = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for lambda in [None, Some(0.0), Some(10.0)] {
        let p = params(lambda);
        let options = EncodeOptions {
            adaptive_density: true,
            allowed_modes: [false, false, false, true],
            ..EncodeOptions::default()
        };
        let reference =
            encode_image_with_search(&image, &p, &options, |_| Box::new(ExhaustiveSearch));
        for budget in [0, 1, 7] {
            let run = || {
                encode_image_with_search(&image, &p, &options, |contracted| {
                    Box::new(RandomSearchProvider::build(
                        &image,
                        contracted,
                        &p,
                        &options,
                        RandomConfig { budget, seed: 12 },
                    ))
                })
            };
            let a = one.install(run);
            let b = two.install(run);
            assert_encoding(&a, &b);
            roundtrip(&a);
            assert_eq!(a.counters.warmup_evals, reference.counters.warmup_evals);
            if budget == 0 {
                assert_eq!(a.counters.search_evals, 0);
            }
        }
    }
}

#[test]
fn configured_large_boundary_and_size_one_only_geometry() {
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
    let provider = RandomSearchProvider::build(
        &image,
        &contracted,
        &p,
        &options,
        RandomConfig {
            budget: 3,
            seed: 42,
        },
    );
    for size in [2, 4, 8, 16, 32] {
        for shift in [16, 32] {
            let result = provider.search(&SearchRequest {
                image: &image,
                contracted: &contracted,
                params: &p,
                row: 0,
                col: 0,
                size,
                shift,
            });
            assert!(result.evals > 0 && result.evals <= 24);
        }
    }
    let p = EncodeParams {
        min_size: 1,
        max_size: 1,
        ..params(None)
    };
    let tiny = Plane::from_vec(1, 1, vec![173]);
    let result = encode_image_with_search(&tiny, &p, &options, |contracted| {
        Box::new(RandomSearchProvider::build(
            &tiny,
            contracted,
            &p,
            &options,
            RandomConfig {
                budget: 3,
                seed: 42,
            },
        ))
    });
    assert_eq!(result.counters.search_evals, 0);
    assert_eq!(result.leaves[0].qbeta, 86);
    roundtrip(&result);
}
