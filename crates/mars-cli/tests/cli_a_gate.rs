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
use mars_bench::provenance::sha256_hex;
use mars_core::metrics::psnr;

const LAMBDA_GRID: [f64; 4] = [50.0, 200.0, 800.0, 3200.0];
const KODAK_WIDTH: usize = 768;
const KODAK_HEIGHT: usize = 512;

/// Side of the synthetic plane [`omitting_adaptive_density_reproduces_the_pinned_default_stream`]
/// encodes.
///
/// A synthetic input rather than a corpus crop is deliberate and load-bearing. Post-D48,
/// `--adaptive-density` only reroutes a block whose `block_rms` sits below `DENSITY_LOW_RMS`
/// onto a doubled domain-search stride, and on natural imagery almost no block qualifies.
/// Measured directly at the args the check below pins: kodim01's 128x128 top-left encodes
/// **byte-identically** with and without the flag. A real-image input therefore cannot
/// observe the default flipping, so any check built on one is inert however it asserts. A
/// gradient this gentle puts *every* block below the threshold, making the reroute
/// unconditional and the flag observable. Fast, and hermetic -- no corpus fetch, so unlike
/// the opt-in sweep above it runs in CI's quick job.
const DEFAULT_CHECK_SIZE: usize = 128;

/// SHA-256 of the `.mars` stream `encmars` writes for [`DEFAULT_CHECK_SIZE`]'s synthetic
/// plane when `--adaptive-density` is *omitted*. Captured from the current default-off path;
/// per `docs/decisions.md` D43 that path must stay opt-in and byte-for-byte stable. Supplying
/// the flag on this input yields a different stream, so a default that silently flipped to
/// `true` would move the omitted stream off this hash and fail the check.
const PINNED_DEFAULT_STREAM_SHA256: &str =
    "55ccaacf9b9c0e952673992404cfdf8ebb21e8503ae3d45a67b4e82a5eed3544";

/// Write a `size`x`size` gentle gradient whose every block is flat enough to route through
/// `adaptive_shift`'s sparsify branch -- the same construction `mars_codec::encode`'s
/// `adaptive_density_reduces_evals_on_a_uniformly_low_rms_image` uses to make the flag
/// observable.
fn write_low_rms_gradient(path: &Path, size: usize) {
    let mut data = vec![0u8; size * size];
    for r in 0..size {
        for c in 0..size {
            data[r * size + c] = (100 + (r + c) % 5) as u8;
        }
    }
    std::fs::write(path, &data).expect("writing the synthetic gradient plane");
}

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
    (width, height): (usize, usize),
    lambda: f64,
    adaptive_density: bool,
) -> (Vec<u8>, f64) {
    let mut cmd = Command::new(encmars_bin());
    cmd.arg(image_path)
        .arg(out_path)
        .args(["--raw-width", &width.to_string()])
        .args(["--raw-height", &height.to_string()])
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
    let bpp = 8.0 * bytes.len() as f64 / (width * height) as f64;
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
        let (bytes, bpp) = encode_via_cli(
            image_path,
            &out_path,
            (KODAK_WIDTH, KODAK_HEIGHT),
            lambda,
            adaptive_density,
        );
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
///
/// The claim is pinned by stream hash rather than by an "explicit false" arm, which cannot
/// exist for a boolean flag: both arms would have to *omit* `--adaptive-density`, making the
/// comparison vacuous (see the previous revision of this test). Pinning alone would be just
/// as hollow on an input where the flag does nothing, so the observable-effect assertion
/// below is what gives the hash teeth -- and, per [`DEFAULT_CHECK_SIZE`], only the synthetic
/// plane supports it. `encode_via_cli` pins the same explicit non-default configuration as
/// the sweep above (four modes, `t_rms` 0), so the flag is the one differing input.
#[test]
fn omitting_adaptive_density_reproduces_the_pinned_default_stream() {
    let tmp = std::env::temp_dir().join(format!("gate-cli-a-default-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("scratch dir");
    let image_path = tmp.join("gradient.raw");
    let size = DEFAULT_CHECK_SIZE;
    write_low_rms_gradient(&image_path, size);

    let (off, _) = encode_via_cli(
        &image_path,
        &tmp.join("off.mars"),
        (size, size),
        200.0,
        false,
    );
    let (on, _) = encode_via_cli(&image_path, &tmp.join("on.mars"), (size, size), 200.0, true);
    let (off_hash, on_hash) = (sha256_hex(&off), sha256_hex(&on));

    assert_ne!(
        off_hash, on_hash,
        "--adaptive-density must be observable on this input, or pinning the omitted stream \
         would prove nothing about which branch omitting selects"
    );
    assert_eq!(
        off_hash, PINNED_DEFAULT_STREAM_SHA256,
        "omitting --adaptive-density must still reproduce the pinned historical default stream"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}
