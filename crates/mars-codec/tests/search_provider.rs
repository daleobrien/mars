//! Provider injection must preserve pinned pre-refactor streams and codec objectives.
use mars_codec::encode::{
    EncodeOptions, EncodeParams, ResidualQuantisation, encode_image_with_options,
};
use mars_codec::encode::{
    ExhaustiveSearch, SearchOutcome, SearchProvider, SearchRequest, encode_image_with_search,
};
use mars_codec::{ifs, mars_format, quant::ResidualQstep};
use mars_core::Plane;
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};

fn image() -> Plane {
    Plane::from_vec(
        16,
        16,
        (0..256)
            .map(|i| {
                if i % 16 < 8 {
                    100
                } else {
                    ((i * 37 + i / 16 * 13) % 256) as u8
                }
            })
            .collect(),
    )
}

fn params(lambda: Option<f64>) -> EncodeParams {
    EncodeParams {
        min_size: 4,
        max_size: 8,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda,
    }
}

fn cases() -> [(EncodeParams, EncodeOptions); 4] {
    [
        (params(None), EncodeOptions::default()),
        (params(Some(2.0)), EncodeOptions::default()),
        (
            params(Some(2.0)),
            EncodeOptions {
                adaptive_density: true,
                residual_quantisation: ResidualQuantisation::LambdaAdaptive,
                ..EncodeOptions::default()
            },
        ),
        (
            params(Some(50.0)),
            EncodeOptions {
                allowed_modes: [true, false, true, false],
                residual_quantisation: ResidualQuantisation::Fixed(
                    ResidualQstep::new(17.123456789).unwrap(),
                ),
                ..EncodeOptions::default()
            },
        ),
    ]
}

struct NoCandidates {
    requests: Arc<Mutex<Vec<(u32, u32, u32, u32)>>>,
    // Demonstrates the agreed factory can build an owned index from Contracted.
    contracted_copy: Vec<i32>,
}

impl SearchProvider for NoCandidates {
    fn search(&self, request: &SearchRequest<'_>) -> SearchOutcome {
        assert_eq!(request.contracted.raw().0, self.contracted_copy);
        assert!(request.row + request.size <= request.image.height() as u32);
        assert!(request.col + request.size <= request.image.width() as u32);
        assert!(request.size >= 2);
        assert_eq!(request.params.shift, 4);
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for i in 0..request.size {
            for j in 0..request.size {
                let p = f64::from(
                    request.image.as_slice()[(request.row + i) as usize * request.image.width()
                        + (request.col + j) as usize],
                );
                sum += p;
                sum2 += p * p;
            }
        }
        let n = f64::from(request.size * request.size);
        let rms = (sum2 / n - (sum / n).powi(2)).max(0.0).sqrt();
        let expected_shift = if request.params.lambda.is_some() && rms <= 8.0 {
            8
        } else {
            4
        };
        assert_eq!(request.shift, expected_shift);
        self.requests
            .lock()
            .unwrap()
            .push((request.row, request.col, request.size, request.shift));
        SearchOutcome {
            candidate: None,
            evals: 7,
        }
    }
}

#[test]
fn provider_is_used_only_for_final_walk_with_effective_stride_and_separate_counts() {
    let source = image();
    for lambda in [None, Some(2.0)] {
        let mut p = params(lambda);
        p.t_rms = 1000.0; // Warmup must still use its fixed threshold of 8.
        let options = EncodeOptions {
            adaptive_density: true,
            allowed_modes: [true, false, true, false],
            residual_quantisation: ResidualQuantisation::LambdaAdaptive,
        };
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let mut factory_calls = 0;
        let outcome = encode_image_with_search(&source, &p, &options, |contracted| {
            factory_calls += 1;
            Box::new(NoCandidates {
                requests: recorded,
                contracted_copy: contracted.raw().0.to_vec(),
            })
        });
        assert_eq!(factory_calls, 1);
        let requests = requests.lock().unwrap();
        // 4 size-8 parents and 16 size-4 children. Warmup must never reach this provider.
        assert_eq!(requests.len(), 20);
        assert_eq!(outcome.counters.search_evals, 20 * 7);
        assert_eq!(
            outcome.counters.warmup_evals,
            if lambda.is_some() { 608 } else { 0 }
        );
        if lambda.is_some() {
            assert!(requests.iter().any(|r| r.3 == 8));
            assert!(requests.iter().any(|r| r.3 == 4));
        } else {
            assert!(requests.iter().all(|r| r.3 == 4));
        }
        assert!(
            outcome
                .leaves
                .iter()
                .all(|leaf| leaf.mode == 0 && leaf.qalfa == 0)
        );
        assert_eq!(
            outcome.header.residual_qstep,
            lambda.map_or(ResidualQstep::LEGACY, ResidualQstep::from_lambda)
        );
        let bytes = mars_format::write(&outcome.header, &outcome.leaves).unwrap();
        let (header, leaves) = mars_format::read(&bytes).unwrap();
        assert_eq!(header, outcome.header);
        assert_eq!(leaves, outcome.leaves);
        let decoded = ifs::decode_iterative(&header, &leaves, 2);
        for leaf in &leaves {
            let expected = (0.5 + f64::from(leaf.qbeta) / 127.0 * 255.0) as u8;
            for i in 0..leaf.size {
                for j in 0..leaf.size {
                    assert_eq!(
                        decoded.as_slice()[(leaf.row + i) as usize * 16 + (leaf.col + j) as usize],
                        expected
                    );
                }
            }
        }
    }
}

#[test]
fn pinned_pre_provider_streams() {
    // Captured from the existing encoder before introducing SearchProvider.
    let pinned = [
        (
            53,
            608,
            "bb480414035f577917542c7f972cb302f904cecbc79449568d05ad74345e98d8",
        ),
        (
            149,
            1184,
            "5069a669a65b6926bc8a597099e0c40c36ac94f038138480226c9b0385b6ae1c",
        ),
        (
            161,
            864,
            "afbb23207c8e98b70a1b23639701a8c02d46bd10e7299c58cebe60d52cee76d8",
        ),
        (
            53,
            1184,
            "44fd4f749006cddd0054b817ba5afd630f356bfa6c18f5b68473e4a300c3885f",
        ),
    ];
    for ((params, options), (length, expected_evals, hash)) in cases().into_iter().zip(pinned) {
        let (header, leaves, evals, _) = encode_image_with_options(&image(), &params, &options);
        let bytes = mars_format::write(&header, &leaves).unwrap();
        assert_eq!(bytes.len(), length);
        assert_eq!(evals, expected_evals);
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), hash);
        let outcome =
            encode_image_with_search(&image(), &params, &options, |_| Box::new(ExhaustiveSearch));
        assert_eq!(outcome.counters.search_evals, expected_evals);
        assert_eq!(
            outcome.counters.warmup_evals,
            if params.lambda.is_some() { 608 } else { 0 }
        );
        assert_eq!(
            outcome.counters.total_evals(),
            expected_evals + outcome.counters.warmup_evals
        );
        let bytes = mars_format::write(&outcome.header, &outcome.leaves).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), hash);
        let (parsed_header, parsed_leaves) = mars_format::read(&bytes).unwrap();
        assert_eq!(parsed_header, outcome.header);
        assert_eq!(parsed_leaves, outcome.leaves);
    }
}
