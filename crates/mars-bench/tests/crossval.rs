//! Step 1's cross-validation: agree with implementations we did not write (§M1, §A3).
//!
//! This is the test that catches the error class §2.1 exists to defend against — a
//! colour-conversion or window-normalisation difference that silently biases every later
//! result by a fraction of a dB. The known-answer vectors in `mars-core/tests` prove the
//! arithmetic; this proves the *definition*.
//!
//! Three independent references, each checking something different:
//!   * **ffmpeg** `-lavfi psnr` — PSNR, a completely separate codebase.
//!   * **scikit-image** `structural_similarity` — SSIM, configured to Wang's original.
//!     This is the canonical implementation of the definition §M2 pins.
//!   * **sewar** `msssim` — the MS-SSIM scale chain and weighting.
//!
//! A note on ffmpeg's `ssim` filter, because it is the obvious thing to reach for and it
//! is the wrong thing: it computes SSIM over **8x8 uniform** windows (the x264-derived
//! variant), not 11x11 Gaussian. It therefore cannot agree with §M2's definition to
//! 0.001, and "fixing" that by widening the tolerance would be precisely the failure A2
//! and A7 forbid. See `docs/decisions.md`.
//!
//! Ignored by default so the CI quick gate stays inside its two-minute budget; run with
//! `just crossval`, which is part of `just gate-a`.

use std::path::{Path, PathBuf};
use std::process::Command;

use mars_bench::measure::{measure, MeasureRequest};
use mars_core::metrics::Downsample;
use mars_core::tolerance::{
    MSE_KNOWN_ANSWER_ABS, MSSSIM_CROSSVAL_ABS, PSNR_CROSSVAL_DB, SSIM_CROSSVAL_ABS,
};

/// §M9/Step 1: ten Kodak images, which is enough for a systematic bias to show up in
/// every row rather than look like one image's quirk.
const IMAGES: usize = 10;
/// Two distortion levels, so the comparison covers more than one point on the curve.
const JPEG_QUALITIES: [u32; 2] = [8, 25];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn require_tool(cmd: &str, args: &[&str], how_to_get_it: &str) {
    let ok = Command::new(cmd)
        .args(args)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    assert!(
        ok,
        "cross-validation needs `{cmd}`, which is not working here.\n{how_to_get_it}\n\
         This test fails rather than skips: a cross-validation that silently passes when \
         the reference is missing is worse than no cross-validation at all (§A2)."
    );
}

fn run(cmd: &mut Command) -> String {
    let out = cmd
        .output()
        .unwrap_or_else(|e| panic!("spawning {cmd:?}: {e}"));
    assert!(
        out.status.success(),
        "{cmd:?} failed with {}:\n{}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn ffmpeg(args: &[&str]) {
    let out = Command::new("ffmpeg")
        .args(["-y", "-v", "error"])
        .args(args)
        .output()
        .expect("ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// PSNR-Y as ffmpeg's `psnr` filter computes it.
fn ffmpeg_psnr(a: &Path, b: &Path) -> f64 {
    let out = Command::new("ffmpeg")
        .args(["-v", "info", "-i"])
        .arg(a)
        .arg("-i")
        .arg(b)
        .args(["-lavfi", "psnr", "-f", "null", "-"])
        .output()
        .expect("ffmpeg psnr");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let line = text
        .lines()
        .rev()
        .find(|l| l.contains("PSNR y:"))
        .unwrap_or_else(|| panic!("no PSNR line in ffmpeg output:\n{text}"));
    let y = line
        .split("y:")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .unwrap_or_else(|| panic!("cannot parse {line:?}"));
    y.parse().unwrap_or_else(|e| panic!("parsing {y:?}: {e}"))
}

#[derive(serde::Deserialize)]
struct Reference {
    mse: f64,
    psnr: Option<f64>,
    ssim: f64,
    ms_ssim_scipy_phase: f64,
}

#[test]
#[ignore = "needs ffmpeg, the pinned python venv, and the Kodak corpus; run via `just crossval`"]
fn metrics_agree_with_independent_implementations_on_kodak() {
    let root = repo_root();
    let python = root.join(".venv-crossval/bin/python");
    let reference_script = root.join("scripts/crossval/reference_metrics.py");
    let work = root.join("target/crossval");
    std::fs::create_dir_all(&work).unwrap();

    require_tool(
        "ffmpeg",
        &["-version"],
        "install it with `brew install ffmpeg`.",
    );
    assert!(
        python.is_file(),
        "missing {}. Create it with `just crossval-setup` \
         (python3.12 -m venv .venv-crossval && pip install -r scripts/crossval/requirements.txt).",
        python.display()
    );

    let mut checked = 0usize;
    let mut worst = (0.0f64, 0.0f64, 0.0f64, String::new());

    for i in 1..=IMAGES {
        let src = root.join(format!("corpus/images/kodak/kodim{i:02}.png"));
        assert!(
            src.is_file(),
            "missing {}. Fetch the corpus with `just corpus`.",
            src.display()
        );

        // A grayscale original, because §M2 computes SSIM and MS-SSIM on Y and the
        // reference scripts work on single-plane input.
        let orig = work.join(format!("kodim{i:02}.gray.png"));
        ffmpeg(&[
            "-i",
            src.to_str().unwrap(),
            "-vf",
            "format=gray",
            orig.to_str().unwrap(),
        ]);

        for q in JPEG_QUALITIES {
            // Real codec distortion rather than synthetic noise: blocking and ringing are
            // what the structural metrics are supposed to be sensitive to.
            let jpg = work.join(format!("kodim{i:02}.q{q}.jpg"));
            let dec = work.join(format!("kodim{i:02}.q{q}.png"));
            ffmpeg(&[
                "-i",
                orig.to_str().unwrap(),
                "-q:v",
                &q.to_string(),
                jpg.to_str().unwrap(),
            ]);
            ffmpeg(&[
                "-i",
                jpg.to_str().unwrap(),
                "-vf",
                "format=gray",
                dec.to_str().unwrap(),
            ]);

            let reference: Reference = serde_json::from_str(&run(Command::new(&python)
                .arg(&reference_script)
                .arg(&orig)
                .arg(&dec)))
            .expect("reference_metrics.py must emit JSON");

            let ours = measure(&MeasureRequest::new(&orig, &dec)).expect("measure");
            let mut scipy_phase = MeasureRequest::new(&orig, &dec);
            scipy_phase.ms_ssim.downsample = Downsample::ScipyUniform2;
            let ours_scipy = measure(&scipy_phase).expect("measure");

            let tag = format!("kodim{i:02} q{q}");

            // --- MSE, exactly --------------------------------------------------
            let d_mse = (ours.mse[0] - reference.mse).abs();
            assert!(
                d_mse < MSE_KNOWN_ANSWER_ABS,
                "{tag}: MSE {} vs numpy {} (diff {d_mse:.3e})",
                ours.mse[0],
                reference.mse
            );

            // --- PSNR against numpy and, separately, ffmpeg ---------------------
            let ours_psnr = ours.psnr_y.expect("distorted image cannot be identical");
            let d_np = (ours_psnr - reference.psnr.expect("reference psnr")).abs();
            let d_ff = (ours_psnr - ffmpeg_psnr(&orig, &dec)).abs();
            assert!(
                d_np < PSNR_CROSSVAL_DB,
                "{tag}: PSNR {ours_psnr} vs numpy, diff {d_np:.4} dB"
            );
            assert!(
                d_ff < PSNR_CROSSVAL_DB,
                "{tag}: PSNR {ours_psnr} vs ffmpeg, diff {d_ff:.4} dB"
            );

            // --- SSIM against scikit-image -------------------------------------
            let ours_ssim = ours.ssim.expect("SSIM available at this size");
            let d_ssim = (ours_ssim - reference.ssim).abs();
            assert!(
                d_ssim < SSIM_CROSSVAL_ABS,
                "{tag}: SSIM {ours_ssim} vs skimage {}, diff {d_ssim:.3e}. \
                 Investigate; do not widen the tolerance (§A7).",
                reference.ssim
            );

            // --- MS-SSIM against sewar, phase-matched --------------------------
            let ours_ms = ours_scipy.ms_ssim.expect("MS-SSIM available at this size");
            let d_ms = (ours_ms - reference.ms_ssim_scipy_phase).abs();
            assert!(
                d_ms < MSSSIM_CROSSVAL_ABS,
                "{tag}: MS-SSIM {ours_ms} vs sewar {}, diff {d_ms:.3e}",
                reference.ms_ssim_scipy_phase
            );

            // The pinned Box2x2 result must differ from the phase-matched one, or the
            // comparison above was not actually testing the pinned configuration's
            // scale chain against anything.
            let pinned = ours.ms_ssim.expect("MS-SSIM available");
            assert_ne!(
                pinned, ours_ms,
                "{tag}: the two decimation phases collapsed"
            );

            if d_ssim > worst.1 {
                worst = (d_np.max(d_ff), d_ssim, d_ms, tag.clone());
            }
            checked += 1;
        }
    }

    assert_eq!(
        checked,
        IMAGES * JPEG_QUALITIES.len(),
        "every pair must be checked"
    );
    eprintln!(
        "cross-validated {checked} pairs; worst disagreement was on {}: \
         PSNR {:.2e} dB, SSIM {:.2e}, MS-SSIM {:.2e}",
        worst.3, worst.0, worst.1, worst.2
    );
}
