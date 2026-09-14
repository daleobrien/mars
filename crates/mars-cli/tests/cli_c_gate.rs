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
//!    report going straight through the library (`mars_search::SizedRetrievers::build` +
//!    `mars_search::encode_image`, the same two calls `marsbench classical-methods` itself
//!    makes) -- cross-validating the new CLI path against the existing measurement path,
//!    not trusting the new wiring on inspection alone.
//!
//! Also covers the plan's own design-question resolution: `--method` and `--lambda` are
//! mutually exclusive (`mars_search::encode_image` has no RD-pruning implementation), and
//! `--method` on colour input is refused (grayscale-only for now) -- both checked here as
//! explicit CLI behaviour, not left implicit.

use std::path::Path;
use std::process::Command;

use mars_codec::encode::EncodeParams;
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

/// The exact two calls `marsbench classical-methods` itself makes for one method, per
/// `crates/mars-cli/src/bin/marsbench.rs`'s own `classical_methods_cmd`.
fn evals_via_library(image: &Plane, method: mars_search::MethodName, params: &EncodeParams) -> u64 {
    let contracted = mars_codec::encode::build_contracted(image);
    let retrievers = mars_search::SizedRetrievers::build(
        &contracted,
        image.width() as u32,
        image.height() as u32,
        params.shift,
        params.min_size,
        params.max_size,
        || method.new_retriever(),
    );
    let (_hdr, _leaves, evals, _picks) = mars_search::encode_image(image, params, &retrievers);
    evals
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

        let lib_evals = evals_via_library(&plane, method, &params);
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
fn method_and_lambda_together_is_refused() {
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
        !output.status.success(),
        "--method and --lambda together should be refused"
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
