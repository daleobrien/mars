//! Small deterministic provider/codec integration tests; no corpus or global thread pool.

use mars_codec::encode::{
    self, build_contracted, encode_image_with_options, encode_image_with_search, Candidate,
    EncodeOptions, EncodeOutcome, EncodeParams, ExhaustiveSearch, SearchOutcome, SearchProvider,
    SearchRequest,
};
use mars_codec::{ifs, mars_format};
use mars_core::Plane;
use mars_search::{
    search_block, search_block_fitted, DomainPool, IndexedSearchProvider, MethodName, RangeBlock,
    SizedRetrievers,
};

fn image(width: usize, height: usize, flat: bool) -> Plane {
    let mut state = 0xC0FFEE_u64;
    let pixels = (0..width * height)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if flat {
                173
            } else {
                state as u8
            }
        })
        .collect();
    Plane::from_vec(width, height, pixels)
}

fn params(lambda: Option<f64>) -> EncodeParams {
    EncodeParams {
        min_size: 4,
        max_size: 16,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda,
    }
}

fn assert_candidate_eq(a: Option<Candidate>, b: Option<Candidate>) {
    match (a, b) {
        (None, None) => {}
        (Some(a), Some(b)) => {
            assert_eq!(
                (a.dom_row, a.dom_col, a.isometry),
                (b.dom_row, b.dom_col, b.isometry)
            );
            assert_eq!((a.qalfa, a.qbeta), (b.qalfa, b.qbeta));
            assert_eq!(a.rms.to_bits(), b.rms.to_bits());
            assert_eq!(a.moments, b.moments);
            assert!(a.rms.is_finite());
        }
        other => panic!("candidate mismatch: {other:?}"),
    }
}

fn assert_outcome_eq(a: &EncodeOutcome, b: &EncodeOutcome) {
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

fn encode_method(
    image: &Plane,
    p: &EncodeParams,
    options: &EncodeOptions,
    method: MethodName,
) -> EncodeOutcome {
    encode_image_with_search(image, p, options, |contracted| {
        Box::new(IndexedSearchProvider::build(
            image, contracted, p, options, method,
        ))
    })
}

#[test]
fn fitted_search_retains_moments_and_tuple_projection_for_every_method() {
    for flat in [false, true] {
        let image = image(17, 19, flat);
        let contracted = build_contracted(&image);
        let p = params(None);
        for method in MethodName::ALL {
            for size in [2, 4, 8, 16] {
                let pool = DomainPool {
                    contracted: &contracted,
                    size,
                    shift: p.shift,
                    image_width: 17,
                    image_height: 19,
                };
                let mut retriever = method.new_retriever();
                retriever.index(&pool);
                let pixels = (0..size as usize)
                    .flat_map(|r| {
                        image.as_slice()[r * 17..r * 17 + size as usize]
                            .iter()
                            .copied()
                    })
                    .collect::<Vec<_>>();
                let range = RangeBlock {
                    row: 0,
                    col: 0,
                    size,
                    pixels: &pixels,
                };
                let result = search_block_fitted(
                    &*retriever,
                    &contracted,
                    &range,
                    p.max_alfa,
                    p.bits_alfa,
                    p.bits_beta,
                );
                let (legacy, evals) = search_block(
                    &*retriever,
                    &contracted,
                    &range,
                    p.max_alfa,
                    p.bits_alfa,
                    p.bits_beta,
                );
                assert_eq!(evals, result.evals);
                assert_eq!(evals, retriever.candidates(&range).count() as u64);
                assert_eq!(
                    legacy,
                    result.candidate.map(|c| (
                        mars_search::Candidate {
                            dom_row: c.dom_row,
                            dom_col: c.dom_col,
                            isometry: c.isometry
                        },
                        c.qalfa,
                        c.qbeta,
                        c.rms
                    ))
                );
                if let Some(c) = result.candidate {
                    let (qa, qb, rms) =
                        encode::fit_f64(c.moments, p.max_alfa, p.bits_alfa, p.bits_beta);
                    assert_eq!((qa, qb, rms), (c.qalfa, c.qbeta, c.rms));
                    assert!(rms.is_finite());
                    assert_eq!(c.moments.s0, i64::from(size * size));
                    assert_eq!((c.moments.t0, c.moments.t2), range.t0_t2());
                    let (dr, dc) = ((c.dom_row / 2) as usize, (c.dom_col / 2) as usize);
                    assert_eq!(
                        (c.moments.s1_x4, c.moments.s2_x16),
                        encode::domain_sums(&contracted, dr, dc, size as usize)
                    );
                    assert_eq!(
                        c.moments.t1_x4,
                        encode::cross_term(&contracted, dr, dc, size as usize, c.isometry, &pixels)
                    );
                    if flat {
                        let first = retriever.candidates(&range).next().unwrap();
                        assert_eq!(
                            (c.dom_row, c.dom_col, c.isometry),
                            (first.dom_row, first.dom_col, first.isometry)
                        );
                    }
                }
                if method == MethodName::Exhaustive {
                    let (reference, evals) =
                        encode::search_block(&image, &contracted, 0, 0, size, &p);
                    assert_eq!(result.evals, evals);
                    assert_candidate_eq(result.candidate, reference);
                }
            }
        }
    }
}

#[test]
fn density_indexes_match_independently_built_pools_at_boundary_and_large_sizes() {
    fn require_sync<T: Sync>() {}
    require_sync::<IndexedSearchProvider>();
    for (width, height, min_size, max_size, sizes) in [
        (17, 19, 8, 16, vec![2, 4, 8, 16]),
        (65, 67, 32, 32, vec![2, 32]),
    ] {
        let image = image(width, height, false);
        let contracted = build_contracted(&image);
        let p = EncodeParams {
            min_size,
            max_size,
            shift: 16,
            ..params(Some(10.0))
        };
        let options = EncodeOptions {
            adaptive_density: true,
            ..EncodeOptions::default()
        };
        for method in MethodName::ALL {
            let provider = IndexedSearchProvider::build(&image, &contracted, &p, &options, method);
            for shift in [p.shift, p.shift * 2] {
                let direct_params = EncodeParams { shift, ..p };
                for &size in &sizes {
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
                    let direct = IndexedSearchProvider::build(
                        &image,
                        &contracted,
                        &direct_params,
                        &EncodeOptions::default(),
                        method,
                    );
                    let expected = direct.search(&SearchRequest {
                        params: &direct_params,
                        ..request
                    });
                    assert_eq!(got.evals, expected.evals);
                    assert_candidate_eq(got.candidate, expected.candidate);
                    if method == MethodName::Exhaustive {
                        let reference = ExhaustiveSearch.search(&request);
                        assert_eq!(got.evals, reference.evals);
                        assert_candidate_eq(got.candidate, reference.candidate);
                    }
                }
            }
        }
    }
}

#[test]
fn exhaustive_adapter_matches_default_production_bytes_and_historical_evals() {
    for flat in [false, true] {
        let image = image(17, 19, flat);
        for lambda in [None, Some(10.0)] {
            let p = params(lambda);
            for adaptive_density in [false, true] {
                let options = EncodeOptions {
                    adaptive_density,
                    ..EncodeOptions::default()
                };
                let (header, leaves, evals, stats) =
                    encode_image_with_options(&image, &p, &options);
                let result = encode_method(&image, &p, &options, MethodName::Exhaustive);
                let direct =
                    encode_image_with_search(&image, &p, &options, |_| Box::new(ExhaustiveSearch));
                assert_outcome_eq(&result, &direct);
                assert_eq!(result.header, header);
                assert_eq!(result.leaves, leaves);
                assert_eq!(result.counters.search_evals, evals);
                assert_eq!(result.mode_stats.leaf_modes, stats.leaf_modes);
                assert_eq!(
                    mars_format::write(&result.header, &result.leaves).unwrap(),
                    mars_format::write(&header, &leaves).unwrap()
                );
            }
        }
    }
}

fn assert_roundtrip(result: &EncodeOutcome, image: &Plane) {
    let mut coverage = vec![0u8; image.width() * image.height()];
    for leaf in &result.leaves {
        assert!(leaf.row + leaf.size <= image.height() as u32);
        assert!(leaf.col + leaf.size <= image.width() as u32);
        if leaf.qalfa != 0 {
            assert!(leaf.dom_row + 2 * leaf.size <= image.height() as u32);
            assert!(leaf.dom_col + 2 * leaf.size <= image.width() as u32);
            assert_eq!(leaf.dom_row % result.header.shift, 0);
            assert_eq!(leaf.dom_col % result.header.shift, 0);
            assert!(leaf.isometry < 8);
        }
        for r in leaf.row..leaf.row + leaf.size {
            for c in leaf.col..leaf.col + leaf.size {
                coverage[r as usize * image.width() + c as usize] += 1;
            }
        }
    }
    assert!(coverage.iter().all(|&n| n == 1));
    let bytes = mars_format::write(&result.header, &result.leaves).unwrap();
    let (header, leaves) = mars_format::read(&bytes).unwrap();
    assert_eq!(header, result.header);
    assert_eq!(leaves, result.leaves);
    assert_eq!(mars_format::write(&header, &leaves).unwrap(), bytes);
    let decoded = ifs::decode_iterative(&header, &leaves, 3);
    assert_eq!(
        (decoded.width(), decoded.height()),
        (image.width(), image.height())
    );
    assert_eq!(
        decoded.as_slice(),
        ifs::decode_iterative(&result.header, &result.leaves, 3).as_slice()
    );
}

#[test]
fn all_nine_methods_production_threshold_rd_density_residual_and_threads() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let two = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let mut residual_methods = [false; 9];
    for (width, height, flat) in [
        (1, 9, true),
        (9, 1, false),
        (2, 2, true),
        (8, 8, false),
        (17, 19, false),
        (17, 19, true),
    ] {
        let image = image(width, height, flat);
        for (lambda, density) in [(None, false), (Some(0.0), false), (Some(0.0), true)] {
            let p = params(lambda);
            // Mode 3 competes alone when a positive-alfa fit exists; codec DC fallback
            // still handles empty pools and constants. This guarantees residual coverage.
            let options = EncodeOptions {
                adaptive_density: density,
                allowed_modes: [false, false, false, true],
                ..EncodeOptions::default()
            };
            let baseline = one.install(|| {
                encode_image_with_search(&image, &p, &options, |_| Box::new(ExhaustiveSearch))
            });
            for (index, method) in MethodName::ALL.into_iter().enumerate() {
                let result = one.install(|| encode_method(&image, &p, &options, method));
                let parallel = two.install(|| encode_method(&image, &p, &options, method));
                assert_outcome_eq(&result, &parallel);
                assert_roundtrip(&result, &image);
                assert_eq!(result.counters.warmup_evals, baseline.counters.warmup_evals);
                assert_eq!(
                    result.counters.total_evals(),
                    result.counters.search_evals + result.counters.warmup_evals
                );
                if lambda.is_none() {
                    assert_eq!(result.counters.warmup_evals, 0);
                }
                residual_methods[index] |= result
                    .leaves
                    .iter()
                    .any(|leaf| leaf.mode == 3 && !leaf.residual.is_empty());
            }
        }
    }
    assert!(
        residual_methods.into_iter().all(|covered| covered),
        "each method must exercise residual mode 3"
    );
}

#[test]
fn diagnostic_singletons_use_scaled_seven_bit_dc_like_production() {
    for max_size in [1, 16] {
        let image = image(1, 9, true);
        let p = EncodeParams {
            min_size: 1,
            max_size,
            ..params(None)
        };
        let contracted = build_contracted(&image);
        for method in MethodName::ALL {
            let retrievers =
                SizedRetrievers::build(&contracted, 1, 9, p.shift, p.min_size, p.max_size, || {
                    method.new_retriever()
                });
            let (_, leaves, evals, _) = mars_search::encode_image(&image, &p, &retrievers);
            let result = encode_method(&image, &p, &EncodeOptions::default(), method);
            assert_eq!(leaves, result.leaves);
            assert_eq!(evals, 0);
            assert_eq!(result.counters.search_evals, 0);
            assert!(leaves.iter().all(|leaf| leaf.size == 1 && leaf.qbeta == 86));
            assert_roundtrip(&result, &image);
        }
    }
}

#[test]
fn empty_provider_pool_returns_no_fit_and_zero_evals() {
    let image = image(8, 8, false);
    let p = params(Some(10.0));
    let contracted = build_contracted(&image);
    for method in MethodName::ALL {
        let provider = IndexedSearchProvider::build(
            &image,
            &contracted,
            &p,
            &EncodeOptions::default(),
            method,
        );
        let SearchOutcome { candidate, evals } = provider.search(&SearchRequest {
            image: &image,
            contracted: &contracted,
            params: &p,
            row: 0,
            col: 0,
            size: 8,
            shift: p.shift,
        });
        assert!(candidate.is_none());
        assert_eq!(evals, 0);
    }
}
