//! `gate-cli-c` -- CLI-C of `encmars-decmars-cli-plan.md`: exposing `mars-search`'s nine
//! candidate-restriction methods as `encmars --method`. Fast (one small synthetic image,
//! default params, no corpus RD sweep) -- unlike CLI-A's gate, `mars-search`'s methods are
//! restricted-candidate searches, not exhaustive RD sweeps, so a full run over all nine on
//! one small image is cheap.
//!
//! Two exit criteria, per the plan:
//! 1. For every method, `encmars --method X` produces a `.mars` file `decmars` decodes
//!    without error.
//! 2. The `evals` count `encmars` reports matches what the same method/image/params would
//!    report through the codec's production walk with the same search provider.
//!    Both legacy and RD counters are checked exactly, including warm-up work.
//!
//! Explicit lambda now enables production RD for methods. Colour input remains refused.

use std::path::Path;
use std::process::Command;

use mars_codec::encode::{encode_image_with_search, EncodeOptions, EncodeParams, ExhaustiveSearch};
use mars_core::Plane;

fn encmars_bin() -> &'static str {
    env!("CARGO_BIN_EXE_encmars")
}
fn decmars_bin() -> &'static str {
    env!("CARGO_BIN_EXE_decmars")
}

const METHODS: [(&str, mars_search::MethodName); 9] = [
    ("exhaustive", mars_search::MethodName::Exhaustive),
    ("fisher", mars_search::MethodName::Fisher),
    ("hurtgen", mars_search::MethodName::Hurtgen),
    ("masscenter", mars_search::MethodName::MassCenter),
    ("saupe", mars_search::MethodName::Saupe),
    ("saupe-fisher", mars_search::MethodName::SaupeFisher),
    ("mc-saupe", mars_search::MethodName::McSaupe),
    ("funnel", mars_search::MethodName::Funnel),
    ("learned", mars_search::MethodName::Learned),
];

/// Mirrors CLI-B's own test-image discipline: uniformly textured with two overlapping
/// frequencies, no perfectly flat blocks, small enough that even `exhaustive` (a real
/// brute-force scan) finishes in well under a second.
fn write_test_pgm(path: &Path) {
    let (w, h) = (48usize, 48usize);
    let mut data = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = ((x * 7 + y * 11) % 256) + (((x * x + y) % 37) * 5);
            data[y * w + x] = (v % 256) as u8;
        }
    }
    let mut pgm = format!("P5\n{w} {h}\n255\n").into_bytes();
    pgm.extend_from_slice(&data);
    std::fs::write(path, pgm).expect("write test pgm");
}

/// The same params `encmars`'s own CLI defaults produce (min_size 4, max_size 16, shift
/// 4, bits_alfa 4, bits_beta 7, max_alfa 1.0, t_rms 8.0, zero_threshold 0, lambda: None).
fn default_params() -> EncodeParams {
    EncodeParams {
        min_size: 4,
        max_size: 16,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda: None,
    }
}

fn evals_via_library(
    image: &Plane,
    method: mars_search::MethodName,
    params: &EncodeParams,
) -> (u64, u64, u64) {
    let options = EncodeOptions {
        allowed_modes: [true, false, true, false],
        ..EncodeOptions::default()
    };
    let outcome = encode_image_with_search(image, params, &options, |contracted| {
        if matches!(method, mars_search::MethodName::Exhaustive) {
            Box::new(ExhaustiveSearch)
        } else {
            Box::new(mars_search::IndexedSearchProvider::build(
                image, contracted, params, &options, method,
            ))
        }
    });
    (
        outcome.counters.search_evals,
        outcome.counters.warmup_evals,
        outcome.counters.total_evals(),
    )
}

fn named_counter(stdout: &str, name: &str) -> u64 {
    stdout
        .split_once(&format!("{name}="))
        .expect("counter label")
        .1
        .split(',')
        .next()
        .unwrap()
        .parse()
        .expect("integer counter")
}

fn evals_from_encmars_stdout(stdout: &str) -> u64 {
    // "... method <name>, <N> evals, <M> transforms, ..."
    let idx = stdout.find(" evals,").unwrap_or_else(|| {
        panic!("encmars stdout has no ' evals,' marker to parse evals from: {stdout:?}")
    });
    let before = &stdout[..idx];
    let n_str = before
        .rsplit(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap_or_default();
    n_str
        .parse()
        .unwrap_or_else(|e| panic!("couldn't parse evals count {n_str:?} from {stdout:?}: {e}"))
}

#[test]
fn every_method_decodes_and_matches_the_library_evals_count() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-c-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.pgm");
    write_test_pgm(&input);

    // Load the same pixels encmars will read, for the library-side cross-check.
    let plane = mars_core::io::read_image(&input, None)
        .expect("reading the test pgm")
        .planes()[0]
        .clone();
    let params = default_params();

    for &(name, method) in &METHODS {
        let out = tmp.join(format!("{name}.mars"));
        let output = Command::new(encmars_bin())
            .arg(&input)
            .arg(&out)
            .args(["--method", name])
            .output()
            .expect("encmars runs");
        assert!(
            output.status.success(),
            "encmars --method {name} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let cli_evals = evals_from_encmars_stdout(&stdout);

        let (lib_evals, warmup, total) = evals_via_library(&plane, method, &params);
        assert_eq!(warmup, 0, "legacy encoding has no RD warm-up");
        assert_eq!(named_counter(&stdout, "search_evals"), lib_evals);
        assert_eq!(named_counter(&stdout, "warmup_evals"), warmup);
        assert_eq!(named_counter(&stdout, "total_evals"), total);
        assert_eq!(
            cli_evals, lib_evals,
            "method {name}: CLI-reported evals ({cli_evals}) does not match the library \
             path's evals ({lib_evals}) for the same image/method/params"
        );

        let decoded = tmp.join(format!("{name}.png"));
        let status = Command::new(decmars_bin())
            .arg(&out)
            .arg(&decoded)
            .status()
            .expect("decmars runs");
        assert!(
            status.success(),
            "decmars failed to decode method {name}'s output"
        );
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn method_and_lambda_together_decodes_and_matches_production_counters() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-c-excl-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.pgm");
    write_test_pgm(&input);
    let out = tmp.join("out.mars");

    let output = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out)
        .args(["--method", "fisher"])
        .args(["--lambda", "200"])
        .output()
        .expect("encmars runs");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let image = mars_core::io::read_image(&input, None).expect("read fixture");
    let params = EncodeParams {
        lambda: Some(200.0),
        ..default_params()
    };
    let (search, warmup, total) =
        evals_via_library(&image.planes()[0], mars_search::MethodName::Fisher, &params);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(evals_from_encmars_stdout(&stdout), search);
    assert_eq!(named_counter(&stdout, "search_evals"), search);
    assert_eq!(named_counter(&stdout, "warmup_evals"), warmup);
    assert_eq!(named_counter(&stdout, "total_evals"), total);
    assert!(warmup > 0);
    assert_eq!(total, search + warmup);
    let decoded = Command::new(decmars_bin())
        .arg(&out)
        .arg(tmp.join("out.png"))
        .output()
        .expect("decmars runs");
    assert!(
        decoded.status.success(),
        "{}",
        String::from_utf8_lossy(&decoded.stderr)
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn method_on_colour_input_is_refused() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-c-rgb-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.ppm");
    let (w, h) = (16usize, 16usize);
    let mut data = vec![0u8; w * h * 3];
    for i in 0..w * h {
        data[i * 3] = (i % 256) as u8;
        data[i * 3 + 1] = ((i * 2) % 256) as u8;
        data[i * 3 + 2] = ((i * 3) % 256) as u8;
    }
    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
    ppm.extend_from_slice(&data);
    std::fs::write(&input, ppm).expect("write test ppm");
    let out = tmp.join("out.mars");

    let output = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out)
        .args(["--method", "fisher"])
        .output()
        .expect("encmars runs");
    assert!(
        !output.status.success(),
        "--method on colour input should be refused (grayscale-only for now)"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
