//! Bounded synthetic coverage: no image corpus, random seeds, or timing assertions.

use mars_codec::encode::{build_contracted, EncodeParams};
use mars_codec::{ifs, mars_format};
use mars_core::Plane;
use mars_search::classify::{compute_saupe_vector, Block};
use mars_search::{
    encode_image, search_block, DomainPool, MethodName, RangeBlock, SizedRetrievers,
};

fn image(width: usize, height: usize, constant: bool) -> Plane {
    let pixels = (0..width * height)
        .map(|i| {
            if constant {
                173
            } else {
                let (r, c) = (i / width, i % width);
                ((r * 37 + c * 19 + r * c * 11) % 256) as u8
            }
        })
        .collect();
    Plane::from_vec(width, height, pixels)
}

fn params(min_size: u32, bits_beta: u32) -> EncodeParams {
    EncodeParams {
        min_size,
        max_size: 16,
        shift: 4,
        bits_alfa: 4,
        bits_beta,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda: None,
    }
}

#[test]
fn domain_positions_match_legal_row_major_scan() {
    for (width, height) in [(2, 2), (3, 9), (9, 3), (8, 8), (11, 13), (17, 15)] {
        let image = image(width, height, false);
        let contracted = build_contracted(&image);
        for size in [2, 4, 8, 16] {
            for shift in [1, 2, 4, 7] {
                let pool = DomainPool {
                    contracted: &contracted,
                    size,
                    shift,
                    image_width: width as u32,
                    image_height: height as u32,
                };
                let mut expected = Vec::new();
                for row in (0..height as u32).step_by(shift as usize) {
                    for col in (0..width as u32).step_by(shift as usize) {
                        if row + 2 * size <= height as u32 && col + 2 * size <= width as u32 {
                            expected.push((row, col));
                        }
                    }
                }
                assert_eq!(pool.domain_positions().collect::<Vec<_>>(), expected);
                for (row, col) in expected {
                    let block = pool.domain_block(row, col);
                    for r in 0..size as usize {
                        for c in 0..size as usize {
                            assert_eq!(
                                block.at(r, c),
                                f64::from(
                                    contracted.at(row as usize / 2 + r, col as usize / 2 + c)
                                )
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn oversized_domain_diameter_cannot_overflow_into_a_candidate() {
    let image = image(8, 8, false);
    let contracted = build_contracted(&image);
    for size in [8, 16, u32::MAX / 2 + 1, u32::MAX] {
        let pool = DomainPool {
            contracted: &contracted,
            size,
            shift: 4,
            image_width: 8,
            image_height: 8,
        };
        assert_eq!(pool.domain_positions().next(), None);
    }
}

#[test]
fn every_method_returns_legal_deterministic_candidates_and_clears_empty_indexes() {
    for constant in [false, true] {
        let image = image(17, 15, constant);
        let contracted = build_contracted(&image);
        for method in MethodName::ALL {
            let mut retriever = method.new_retriever();
            // Reuse each instance: nonempty -> empty -> nonempty -> empty.
            for size in [2, 8, 4, 16] {
                let pool = DomainPool {
                    contracted: &contracted,
                    size,
                    shift: 2,
                    image_width: 17,
                    image_height: 15,
                };
                retriever.index(&pool);
                let legal = pool.domain_positions().collect::<Vec<_>>();
                for flat_range in [false, true] {
                    let pixels = (0..size * size)
                        .map(|i| {
                            if flat_range {
                                173
                            } else {
                                ((i * 43 + 7) % 256) as u8
                            }
                        })
                        .collect::<Vec<_>>();
                    let range = RangeBlock {
                        row: 0,
                        col: 0,
                        size,
                        pixels: &pixels,
                    };
                    let candidates = retriever.candidates(&range).collect::<Vec<_>>();
                    assert_eq!(
                        candidates,
                        retriever.candidates(&range).collect::<Vec<_>>(),
                        "{method:?}, size {size}"
                    );
                    assert_eq!(
                        candidates.is_empty(),
                        legal.is_empty(),
                        "{method:?}, size {size}"
                    );
                    if method == MethodName::Exhaustive {
                        let expected = legal
                            .iter()
                            .flat_map(|&(dom_row, dom_col)| {
                                (0..8).map(move |isometry| mars_search::Candidate {
                                    dom_row,
                                    dom_col,
                                    isometry,
                                })
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(candidates, expected);
                    }
                    for c in &candidates {
                        assert!(legal.contains(&(c.dom_row, c.dom_col)), "{method:?}: {c:?}");
                        assert!(c.isometry < 8);
                    }
                    let (best, evals) = search_block(&*retriever, &contracted, &range, 1.0, 4, 7);
                    assert_eq!(evals, candidates.len() as u64);
                    assert_eq!(best.is_none(), legal.is_empty());
                    if let Some((candidate, _, _, rms)) = best {
                        assert!(rms.is_finite(), "{method:?}, size {size}");
                        if constant && flat_range {
                            assert_eq!(candidate, candidates[0], "first candidate wins equal RMS");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn constant_saupe_features_are_finite_including_after_shrinking() {
    for size in [2, 4, 8, 16] {
        for value in [0.0, 173.0, 1020.0] {
            let block = Block::new(size, vec![value; size * size]);
            for factor in [1, 2] {
                let features = compute_saupe_vector(&block, factor);
                assert_eq!(features, vec![0.0; (size / factor) * (size / factor)]);
            }
        }
    }
}

#[test]
fn empty_polar_grid_search_is_bounded_and_occupied_ring_order_is_unchanged() {
    use mars_search::masscenter::expanding_ring;
    assert!(expanding_ring(0, 0, 0, 0, 1, |_, _| panic!("empty grid")).is_empty());
    for (r, t) in [(0, 1), (1, 3)] {
        let mut probes = 0;
        assert!(expanding_ring(0, 0, 5, r, t, |_, _| {
            probes += 1;
            false
        })
        .is_empty());
        assert!(probes <= 55);
    }
    assert_eq!(
        expanding_ring(0, 0, 5, 1, 3, |a, b| a == 0 && b == 0),
        vec![
            (4, 4),
            (4, 0),
            (4, 1),
            (0, 4),
            (0, 0),
            (0, 1),
            (1, 4),
            (1, 0),
            (1, 1)
        ]
    );
}

#[test]
fn border_splits_index_smaller_sizes_without_changing_configured_sizes() {
    let image = image(11, 13, false);
    let contracted = build_contracted(&image);
    let p = params(8, 7);
    for method in MethodName::ALL {
        let indexed =
            SizedRetrievers::build(&contracted, 11, 13, p.shift, p.min_size, p.max_size, || {
                method.new_retriever()
            });
        for size in [2, 4, 8, 16] {
            let pool = DomainPool {
                contracted: &contracted,
                size,
                shift: p.shift,
                image_width: 11,
                image_height: 13,
            };
            let mut direct = method.new_retriever();
            direct.index(&pool);
            let pixels = vec![173; (size * size) as usize];
            let range = RangeBlock {
                row: 0,
                col: 0,
                size,
                pixels: &pixels,
            };
            assert_eq!(
                indexed.get(size).candidates(&range).collect::<Vec<_>>(),
                direct.candidates(&range).collect::<Vec<_>>(),
                "{method:?}, size {size}"
            );
        }
    }
}

#[test]
fn tiny_odd_images_roundtrip_for_all_methods_at_one_and_two_threads() {
    roundtrip_dimensions(&[(2, 2), (3, 5), (8, 8), (11, 13), (17, 15)], true);
}

#[test]
fn singleton_dimensions_roundtrip_for_all_methods() {
    roundtrip_dimensions(&[(1, 1), (1, 9), (9, 1)], true);
}

#[test]
fn tiny_odd_images_encode_and_serialize_for_all_methods() {
    roundtrip_dimensions(&[(2, 2), (3, 5), (8, 8), (11, 13), (17, 15)], false);
}

fn roundtrip_dimensions(dimensions: &[(usize, usize)], decode: bool) {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let two = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    for &(width, height) in dimensions {
        for constant in [false, true] {
            let image = image(width, height, constant);
            let contracted = build_contracted(&image);
            // 8-bit beta permits exact constant reconstruction even at size-one leaves.
            let p = params(4, if constant { 8 } else { 7 });
            for method in MethodName::ALL {
                let retrievers = SizedRetrievers::build(
                    &contracted,
                    width as u32,
                    height as u32,
                    p.shift,
                    p.min_size,
                    p.max_size,
                    || method.new_retriever(),
                );
                let encoded = one.install(|| encode_image(&image, &p, &retrievers));
                assert_eq!(
                    encoded,
                    two.install(|| encode_image(&image, &p, &retrievers)),
                    "{method:?} {width}x{height}"
                );
                let (header, leaves, _, picks) = encoded;
                let mut coverage = vec![0u8; width * height];
                for leaf in &leaves {
                    assert!(leaf.row + leaf.size <= height as u32);
                    assert!(leaf.col + leaf.size <= width as u32);
                    if leaf.qalfa != 0 {
                        assert!(leaf.dom_row + 2 * leaf.size <= height as u32);
                        assert!(leaf.dom_col + 2 * leaf.size <= width as u32);
                        assert_eq!(leaf.dom_row % p.shift, 0);
                        assert_eq!(leaf.dom_col % p.shift, 0);
                        assert!(leaf.isometry < 8);
                    }
                    for r in leaf.row..leaf.row + leaf.size {
                        for c in leaf.col..leaf.col + leaf.size {
                            coverage[r as usize * width + c as usize] += 1;
                        }
                    }
                }
                assert!(coverage.iter().all(|&n| n == 1));
                for pick in picks {
                    assert!(pick.rms.is_finite());
                    assert!(pick.dom_row + 2 * pick.size <= height as u32);
                    assert!(pick.dom_col + 2 * pick.size <= width as u32);
                }
                let bytes = mars_format::write(&header, &leaves).unwrap();
                let (parsed_header, parsed_leaves) = mars_format::read(&bytes).unwrap();
                assert_eq!(parsed_header.geometry, header);
                assert_eq!(parsed_leaves, leaves);
                assert_eq!(
                    mars_format::write(&parsed_header, &parsed_leaves).unwrap(),
                    bytes
                );
                if !decode {
                    continue;
                }
                let decoded = ifs::decode_iterative(&parsed_header, &parsed_leaves, 3);
                assert_eq!((decoded.width(), decoded.height()), (width, height));
                assert_eq!(
                    decoded.as_slice(),
                    ifs::decode_iterative(&header, &leaves, 3).as_slice()
                );
                if constant {
                    assert_eq!(
                        decoded.as_slice(),
                        image.as_slice(),
                        "{method:?} {width}x{height}"
                    );
                }
            }
        }
    }
}
