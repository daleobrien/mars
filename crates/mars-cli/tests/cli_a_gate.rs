//! `gate-cli-a` -- CLI-A of `encmars-decmars-cli-plan.md`: exposing Step 16's
//! `adaptive_density` flag on `encmars` itself, per the plan's own exit criterion --
//! **a re-measurement at CLI scope, not a re-use of `gate-16`'s (D43) library-level
//! number.** `gate-16`'s `density_gate.rs` calls `mars_codec::encode::
//! encode_image_rd_with_modes_and_density` directly; this gate instead shells out to the
//! actual `encmars` binary (`CARGO_BIN_EXE_encmars`, since `mars-cli` owns that binary
//! target) with and without `--adaptive-density`, so any effect from `color.rs`'s
//! YCbCr/subsampling wrapping, or from the real `.mars` bitstream round trip, that a
//! library-level gate might not exercise is caught rather than assumed away.
//!
//! **This is exactly what happened.** The first run of this gate (pre-D48) found a real
//! bitstream-corruption bug in the densify half of `adaptive_density` -- invisible to
//! `gate-16` because that gate never actually decoded from the bytes it wrote, only from
//! the search's in-memory leaves. See `docs/decisions.md` D48 for the full root-cause
//! writeup and the fix (the densify branch was removed; D43's originally-measured
//! -6.82% is withdrawn). Decoding here uses the library (`mars_codec::color::
//! decode_color_image`) directly -- only the *encode* path is what `--adaptive-density`
//! threads through, so only the encode side needs to run through the real binary; the
//! actual bitstream bytes are still what gets decoded, so a corruption bug on the decode
//! side would still be caught.
//!
//! Same corpus scope (kodim01/kodim02) and lambda grid as `gate-16`, for the same
//! session-time reasons `density_gate.rs` states.

use std::path::Path;
use std::process::Command;

use mars_bench::bdrate::{bd_metrics, RdCurve, RdPoint};
use mars_core::metrics::psnr;

const LAMBDA_GRID: [f64; 4] = [50.0, 200.0, 800.0, 3200.0];
const KODAK_WIDTH: usize = 768;
const KODAK_HEIGHT: usize = 512;

/// **D48 CONTRACT-CHANGE.** D43's originally-measured -6.82% mean BD-rate came from a
/// branch that D48 found corrupted the bitstream and removed; the surviving mechanism
/// re-measures at essentially BD-rate-neutral (`gate-16`'s own post-fix number: mean
/// -0.14%). This is now a regression ceiling, not an improvement floor -- mirroring
/// `density_gate.rs`'s own post-D48 `BD_RATE_CEILING_PCT`, not its original value.
const BD_RATE_CEILING_PCT: f64 = 2.0;

fn encmars_bin() -> &'static str {
    env!("CARGO_BIN_EXE_encmars")
}

/// Encode `kodimNN.raw` at one lambda via the real `encmars` binary, returning the
/// produced `.mars` bytes and the file's bpp.
fn encode_via_cli(
    image_path: &Path,
    out_path: &Path,
    lambda: f64,
    adaptive_density: bool,
) -> (Vec<u8>, f64) {
    let mut cmd = Command::new(encmars_bin());
    cmd.arg(image_path)
        .arg(out_path)
        .args(["--raw-width", &KODAK_WIDTH.to_string()])
        .args(["--raw-height", &KODAK_HEIGHT.to_string()])
        .args(["--lambda", &lambda.to_string()])
        // Preserve this historical four-mode density A/B despite the new CLI mask default.
        .args(["--modes", "0,1,2,3"])
        // Match `density_gate.rs`'s own `BASE` exactly (`t_rms: 0.0`) -- `encmars`'s CLI
        // default is 8.0 (a sensible default for the legacy `--t-rms`-only path), but
        // `--lambda`'s own doc says `t_rms` still seeds the rate-estimation warm-up pass
        // even when it no longer drives the split decision, so leaving the CLI default in
        // place here would make this a different measurement than D43's, not a CLI-scope
        // re-measurement of the *same* one. (First attempt at this gate left the CLI
        // default in place and measured a spurious multi-dB PSNR collapse at low lambda,
        // traced to exactly this mismatch -- see the commit history / docs/decisions.md.)
        .args(["--t-rms", "0"]);
    if adaptive_density {
        cmd.arg("--adaptive-density");
    }
    let status = cmd.status().expect("encmars binary runs");
    assert!(status.success(), "encmars exited with {status}");

    let bytes = std::fs::read(out_path).expect("encmars wrote the output file");
    let bpp = 8.0 * bytes.len() as f64 / (KODAK_WIDTH * KODAK_HEIGHT) as f64;
    (bytes, bpp)
}

fn curve_for(label: &str, image_path: &Path, tmp: &Path, adaptive_density: bool) -> RdCurve {
    let original =
        mars_core::io::read_raw(image_path, KODAK_WIDTH, KODAK_HEIGHT).unwrap_or_else(|e| {
            panic!(
                "{}: {e} (run `just corpus-gray` first)",
                image_path.display()
            )
        });

    let mut points = Vec::new();
    for (i, &lambda) in LAMBDA_GRID.iter().enumerate() {
        let out_path = tmp.join(format!(
            "{label}-{}-{i}.mars",
            if adaptive_density {
                "adaptive"
            } else {
                "fixed"
            }
        ));
        let (bytes, bpp) = encode_via_cli(image_path, &out_path, lambda, adaptive_density);
        let decoded = mars_codec::color::decode_color_image(&bytes, 10)
            .expect("decode of what encmars just wrote");
        let p =
            psnr(&original, &decoded.planes()[0]).expect("non-identical planes give finite PSNR");
        points.push(RdPoint { bpp, psnr: p });
    }
    RdCurve::new(label, points)
}

/// Opt in with `MARS_RUN_CLI_A_GATE=1` (`just gate-cli-a` does this automatically) --
/// mirrors every other RD gate's convention in this project (each full sweep here runs
/// `encmars` as a real subprocess, on top of the RD search cost `gate-16` already pays).
#[test]
fn adaptive_density_flag_reproduces_a_real_improvement_at_cli_scope() {
    if std::env::var("MARS_RUN_CLI_A_GATE").as_deref() != Ok("1") {
        eprintln!(
            "skipping: set MARS_RUN_CLI_A_GATE=1 to run gate-cli-a's multi-minute CLI-scope \
             RD sweep (`just gate-cli-a` does this automatically)"
        );
        return;
    }

    let tmp = std::env::temp_dir().join(format!("gate-cli-a-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");

    let mut bd_rates = Vec::new();
    for n in [1u32, 2] {
        let label = format!("kodim{n:02}");
        let image_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../corpus/images/kodak-gray/{label}.raw"));

        let fixed = curve_for(&format!("{label} fixed (CLI)"), &image_path, &tmp, false);
        let adaptive = curve_for(&format!("{label} adaptive (CLI)"), &image_path, &tmp, true);

        for p in &fixed.points {
            eprintln!("  fixed:    bpp={:.4} psnr={:.3}", p.bpp, p.psnr);
        }
        for p in &adaptive.points {
            eprintln!("  adaptive: bpp={:.4} psnr={:.3}", p.bpp, p.psnr);
        }

        let bd = bd_metrics(&fixed, &adaptive)
            .unwrap_or_else(|e| panic!("{label}: BD-rate computation failed: {e}"));
        eprintln!(
            "  {label} BD-rate (adaptive vs fixed, CLI scope): {:.2}% BD-PSNR: {:.3} dB (bpp overlap {:?})",
            bd.bd_rate_pct, bd.bd_psnr_db, bd.bpp_interval
        );
        bd_rates.push((label, bd.bd_rate_pct));
    }

    let mean_bd_rate: f64 = bd_rates.iter().map(|(_, r)| *r).sum::<f64>() / bd_rates.len() as f64;
    eprintln!(
        "mean BD-rate across {} image(s) at CLI scope: {mean_bd_rate:.2}% (regression \
         ceiling <= {BD_RATE_CEILING_PCT:.1}%; expected near 0% post-D48, not D43's \
         withdrawn -6.82% -- see docs/decisions.md D48)",
        bd_rates.len()
    );
    assert!(
        mean_bd_rate <= BD_RATE_CEILING_PCT,
        "mean BD-rate {mean_bd_rate:.2}% exceeds this gate's CLI-scope regression ceiling \
         of {BD_RATE_CEILING_PCT:.1}% (per-image: {bd_rates:?})"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// This project's own convention (per `docs/decisions.md`'s D43 note that `--adaptive-
/// density` must stay an opt-in default-off flag): omitting the flag must still produce
/// exactly what it always has. Fast -- no `MARS_RUN_CLI_A_GATE` gate needed.
#[test]
fn omitting_adaptive_density_matches_explicit_false_byte_for_byte() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-a-default-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let image_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/images/kodak-gray/kodim01.raw");
    if !image_path.exists() {
        eprintln!("skipping: corpus not fetched (run `just corpus-gray`)");
        return;
    }

    // Both arms must pin the same explicit non-default configuration (four modes,
    // `t_rms` 0, matching `encode_via_cli` below) so this test isolates the
    // `--adaptive-density` flag alone. With the D51 CLI-default switch, a bare
    // `--lambda 200` invocation resolves to modes 0,2 / `t_rms` 8, which would make
    // this a modes-vs-modes comparison instead of a density-flag one.
    let out_default = tmp.join("default.mars");
    let status = Command::new(encmars_bin())
        .arg(&image_path)
        .arg(&out_default)
        .args(["--raw-width", &KODAK_WIDTH.to_string()])
        .args(["--raw-height", &KODAK_HEIGHT.to_string()])
        .args(["--lambda", "200"])
        .args(["--modes", "0,1,2,3"])
        .args(["--t-rms", "0"])
        .status()
        .expect("encmars runs");
    assert!(status.success());

    let out_explicit = tmp.join("explicit-off.mars");
    let (bytes_explicit, _) = encode_via_cli(&image_path, &out_explicit, 200.0, false);
    let bytes_default = std::fs::read(&out_default).unwrap();

    assert_eq!(
        bytes_default, bytes_explicit,
        "omitting --adaptive-density must be byte-identical to passing it off explicitly"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
