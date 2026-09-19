//! `just gate-18`'s own exit criteria (§A1) for Step 18 (colour).
//!
//! This is deliberately **not** a full BD-rate-vs-anchors gate: that measurement
//! (`docs/predictions.md`'s Step 18 outcome) took several minutes of exhaustive-encoder
//! wall time for a single image at four rate points in two subsampling modes, which is
//! too slow for a routine gate. What this file checks instead, mirroring gate-13's own
//! "scoped to what a session can actually run" precedent:
//!
//! 1. RGB -> YCbCr -> RGB round trip is lossless for gray-equivalent input and bounded
//!    (<= 2 LSB) in general -- covered by `mars_core::metrics`'s own unit tests, not
//!    repeated here.
//! 2. A synthetic colour image round-trips through the full `.mars` colour container
//!    (encode -> write -> read -> decode) with reasonable PSNR-Y, in both 4:4:4 and
//!    4:2:0, and 4:2:0 does not cost more total bytes than 4:4:4 at matched `t_rms`.
//! 3. If the real Kodak corpus is present (`corpus/images/kodak/kodim01.png` -- fetched
//!    by `scripts/fetch-corpus.sh`, not committed), a 256x256 crop of one real colour
//!    photograph round-trips through encode/decode with PSNR-Y above a sanity floor, in
//!    both subsampling modes, and 4:2:0 again does not exceed 4:4:4's total bytes. If the
//!    corpus is absent this check is skipped with a message, not failed -- the same
//!    convention `funnel_gate.rs` and `classical_methods_gate.rs` already use for
//!    corpus/oracle-dependent checks. The crop is deliberate: the encoded frame size does
//!    not enter this gate's contract (container round trip + chroma byte ordering), while
//!    the full 768x512 frame dominated the whole `cargo test -p mars-codec` run -- ~90s
//!    even at the dev profile's opt-level 2, and ~23 minutes at opt-level 0.
//!
//! Does **not** check: BD-rate against any anchor codec, the full 24-image corpus, or
//! MS-SSIM-based comparisons -- all open per `docs/decisions.md` D38's scope-cut note.

use std::path::Path;

use mars_codec::color::{decode_color_image, encode_color_image, ColorEncodeParams, Subsampling};
use mars_codec::encode::EncodeParams;
use mars_core::image::{ColorSpace, Image, Plane};
use mars_core::io::read_image;
use mars_core::metrics::{quality, MsSsimConfig, SsimConfig};

fn params(t_rms: f64) -> EncodeParams {
    EncodeParams {
        min_size: 4,
        max_size: 16,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms,
        zero_threshold: 0,
        lambda: None,
    }
}

fn synthetic_rgb(w: usize, h: usize) -> Image {
    let mut r = Vec::with_capacity(w * h);
    let mut g = Vec::with_capacity(w * h);
    let mut b = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            r.push(((x * 5 + y) % 256) as u8);
            g.push(((y * 5 + x / 2) % 256) as u8);
            b.push((((x + y) * 3) % 256) as u8);
        }
    }
    Image::rgb(
        Plane::from_vec(w, h, r),
        Plane::from_vec(w, h, g),
        Plane::from_vec(w, h, b),
    )
}

fn round_trip_psnr_y_and_bytes(img: &Image, subsampling: Subsampling) -> (f64, usize) {
    let cfg = ColorEncodeParams {
        y: params(8.0),
        chroma: params(8.0),
        subsampling,
        adaptive_density: false,
        allowed_modes: [true; 4],
        rd_candidates: 1,
        lambda_regions: Vec::new(),
        color_regions: Vec::new(),
        color_desaturate: 1.0,
        color_desaturate_ramp: None,
    };
    let (bytes, _stats) = encode_color_image(img, &cfg);
    let decoded = decode_color_image(&bytes, 10).expect("decode of what we just encoded");
    let q = quality(
        img,
        &decoded,
        None,
        &SsimConfig::default(),
        &MsSsimConfig::default(),
    );
    (
        q.psnr_y.expect("non-identical images give finite PSNR"),
        bytes.len(),
    )
}

#[test]
fn synthetic_colour_image_round_trips_in_both_subsampling_modes() {
    let img = synthetic_rgb(96, 80);

    let (psnr_444, bytes_444) = round_trip_psnr_y_and_bytes(&img, Subsampling::Yuv444);
    let (psnr_420, bytes_420) = round_trip_psnr_y_and_bytes(&img, Subsampling::Yuv420);

    assert!(psnr_444 > 20.0, "4:4:4 PSNR-Y {psnr_444} looks too low");
    assert!(psnr_420 > 20.0, "4:2:0 PSNR-Y {psnr_420} looks too low");
    // The *coded* Y stream is byte-identical between subsampling modes (chroma
    // subsampling never touches it), but `quality()`'s PSNR-Y is measured on the
    // *reconstructed RGB*, re-converted back to YCbCr -- so noisier chroma (4:2:0) does
    // perturb the recomputed Y slightly through the inverse transform's per-channel
    // clamping, even though the decoded Y plane itself is identical. This was a genuine
    // (if small) surprise caught by writing this gate before assuming exact equality --
    // recorded in `docs/decisions.md` D38 rather than silently loosened after the fact.
    // The bound below (1 dB) is generous; on kodim01 the gap measured under 0.01 dB.
    assert!(
        (psnr_444 - psnr_420).abs() < 1.0,
        "PSNR-Y should be nearly identical between subsampling modes \
         (444={psnr_444}, 420={psnr_420})"
    );
    assert!(
        bytes_420 <= bytes_444,
        "4:2:0 ({bytes_420} bytes) should not exceed 4:4:4 ({bytes_444} bytes) at matched t_rms"
    );
}

#[test]
fn gray_image_round_trips_through_the_colour_container() {
    let mut data = vec![0u8; 32 * 32];
    for (i, v) in data.iter_mut().enumerate() {
        *v = (i % 256) as u8;
    }
    let img = Image::gray(Plane::from_vec(32, 32, data));
    let cfg = ColorEncodeParams {
        y: params(8.0),
        chroma: params(8.0),
        subsampling: Subsampling::Yuv444,
        adaptive_density: false,
        allowed_modes: [true; 4],
        rd_candidates: 1,
        lambda_regions: Vec::new(),
        color_regions: Vec::new(),
        color_desaturate: 1.0,
        color_desaturate_ramp: None,
    };
    let (bytes, _stats) = encode_color_image(&img, &cfg);
    let decoded = decode_color_image(&bytes, 10).unwrap();
    assert_eq!(decoded.color(), ColorSpace::Gray);
    assert_eq!((decoded.width(), decoded.height()), (32, 32));
}

#[test]
fn real_kodak_image_round_trips_if_the_corpus_is_present() {
    let path = Path::new("../../corpus/images/kodak/kodim01.png");
    if !path.exists() {
        eprintln!(
            "skipping: no corpus at {path:?} (run `scripts/fetch-corpus.sh` / `just corpus`)"
        );
        return;
    }
    let img = read_image(path, None).expect("reading the Kodak fixture");
    assert_eq!(img.color(), ColorSpace::Rgb);
    // The encoder configuration below is what this gate pins; the frame size is incidental
    // (see the module doc). Cropping keeps every container/chroma code path and real
    // photographic content at a fraction of the cost.
    let img = img.crop_top_left(256, 256);
    assert_eq!((img.width(), img.height()), (256, 256));

    let (psnr_444, bytes_444) = round_trip_psnr_y_and_bytes(&img, Subsampling::Yuv444);
    let (psnr_420, bytes_420) = round_trip_psnr_y_and_bytes(&img, Subsampling::Yuv420);

    assert!(
        psnr_444 > 25.0,
        "4:4:4 PSNR-Y {psnr_444} too low on a real photograph"
    );
    assert!(
        psnr_420 > 25.0,
        "4:2:0 PSNR-Y {psnr_420} too low on a real photograph"
    );
    assert!(
        bytes_420 <= bytes_444,
        "4:2:0 ({bytes_420} bytes) should not exceed 4:4:4 ({bytes_444} bytes) at matched t_rms \
         on a real photograph"
    );
}
