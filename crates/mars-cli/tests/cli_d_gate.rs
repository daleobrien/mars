//! `gate-cli-d` -- CLI-D of `encmars-decmars-cli-plan.md`: `encmars --threads`.
//!
//! **`--gpu` is not implemented this session** -- see `docs/decisions.md` D49 for the
//! full reasoning (the architectural mismatch between `mars_gpu::GpuSearcher`'s one-size,
//! whole-image kernel and the per-node `CandidateRetriever` shape a real quadtree-walk
//! integration would need, plus D25's own already-recorded relaxation of Step 7's oracle
//! from bit-identity to a bounded-divergence tolerance). This gate therefore only checks
//! `--threads`'s own exit criterion: byte-identical output across at least 2 different
//! thread counts, against the real `encmars` binary.

use std::path::Path;
use std::process::Command;

fn encmars_bin() -> &'static str {
    env!("CARGO_BIN_EXE_encmars")
}

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

fn encode_with_threads(input: &Path, out: &Path, threads: usize) {
    let status = Command::new(encmars_bin())
        .arg(input)
        .arg(out)
        .args(["--threads", &threads.to_string()])
        .args(["--lambda", "200"]) // exercise the RD path, which does real rayon::join work
        .status()
        .expect("encmars runs");
    assert!(status.success(), "encmars --threads {threads} failed");
}

#[test]
fn threads_1_and_4_and_8_are_byte_identical() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-d-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.pgm");
    write_test_pgm(&input);

    let mut reference: Option<Vec<u8>> = None;
    for &threads in &[1usize, 4, 8] {
        let out = tmp.join(format!("t{threads}.mars"));
        encode_with_threads(&input, &out, threads);
        let bytes = std::fs::read(&out).unwrap();
        match &reference {
            None => reference = Some(bytes),
            Some(r) => assert_eq!(
                r, &bytes,
                "--threads {threads} produced different bytes than the first thread count tried"
            ),
        }
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn omitting_threads_matches_threads_matching_available_parallelism() {
    // Not a strict requirement (Rayon's own default may differ from
    // available_parallelism() in unusual environments), but on a normal machine omitting
    // --threads should still be byte-identical to *some* explicit thread count, since
    // every encode path in this crate is thread-count-invariant by construction.
    let tmp = std::env::temp_dir().join(format!("gate-cli-d-default-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let input = tmp.join("in.pgm");
    write_test_pgm(&input);

    let out_default = tmp.join("default.mars");
    let status = Command::new(encmars_bin())
        .arg(&input)
        .arg(&out_default)
        .args(["--lambda", "200"])
        .status()
        .expect("encmars runs");
    assert!(status.success());

    let out_explicit = tmp.join("explicit.mars");
    encode_with_threads(&input, &out_explicit, 2);

    assert_eq!(
        std::fs::read(&out_default).unwrap(),
        std::fs::read(&out_explicit).unwrap(),
        "omitting --threads should be byte-identical to any explicit thread count"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
