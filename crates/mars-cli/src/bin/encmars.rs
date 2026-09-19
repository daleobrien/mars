//! `encmars` -- compress an image into Mars 2's `.mars` container (Step 10 format), via
//! the Step 6 exhaustive Rust encoder. Colour input (Step 18) is encoded as independent
//! Y/Cb/Cr planes, each via the same single-plane encoder, wrapped in the small colour
//! container `mars_codec::color` defines -- see that module's doc for the design.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use mars_codec::color::{
    ColorEncodeParams, Subsampling, encode_color_image_with_residual_quantisation, wrap_gray_stream,
};
use mars_codec::encode::{
    EncodeOptions, EncodeParams, ExhaustiveSearch, LambdaRegion, ResidualQuantisation,
    encode_image_with_search,
};
use mars_core::io::read_image;

/// Candidate providers for the codec's production partition walk. Names match
/// `mars_search::MethodName::key()`; exhaustive uses the codec's own search for byte identity.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum MethodArg {
    Exhaustive,
    Fisher,
    Hurtgen,
    Masscenter,
    Saupe,
    #[value(name = "saupe-fisher")]
    SaupeFisher,
    #[value(name = "mc-saupe")]
    McSaupe,
    Funnel,
    Learned,
    Random,
    Apcc,
}

impl MethodArg {
    fn indexed_method(self) -> Option<mars_search::MethodName> {
        Some(match self {
            MethodArg::Exhaustive => mars_search::MethodName::Exhaustive,
            MethodArg::Fisher => mars_search::MethodName::Fisher,
            MethodArg::Hurtgen => mars_search::MethodName::Hurtgen,
            MethodArg::Masscenter => mars_search::MethodName::MassCenter,
            MethodArg::Saupe => mars_search::MethodName::Saupe,
            MethodArg::SaupeFisher => mars_search::MethodName::SaupeFisher,
            MethodArg::McSaupe => mars_search::MethodName::McSaupe,
            MethodArg::Funnel => mars_search::MethodName::Funnel,
            MethodArg::Learned => mars_search::MethodName::Learned,
            MethodArg::Random => return None,
            MethodArg::Apcc => return None,
        })
    }

    fn key(self) -> &'static str {
        match self {
            MethodArg::Random => "random",
            MethodArg::Apcc => "apcc",
            other => other
                .indexed_method()
                .map_or("unnamed", |method| method.key()),
        }
    }
}

/// Chroma subsampling mode (Step 18); ignored for grayscale input.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum SubsamplingArg {
    #[value(name = "444")]
    Yuv444,
    #[value(name = "420")]
    Yuv420,
}

impl From<SubsamplingArg> for Subsampling {
    fn from(v: SubsamplingArg) -> Self {
        match v {
            SubsamplingArg::Yuv444 => Subsampling::Yuv444,
            SubsamplingArg::Yuv420 => Subsampling::Yuv420,
        }
    }
}

/// Compress an image into a `.mars` bitstream.
#[derive(Parser)]
struct Cli {
    /// Input image: .png, .jpg/.jpeg, .pgm, .ppm, or headerless .raw/.y/.gray (needs
    /// --raw-width/--raw-height). PNG and JPEG are autodetected from the file's contents.
    input: PathBuf,
    /// Output `.mars` bitstream path.
    output: PathBuf,

    /// Select legacy threshold partitioning: split blocks whose best-fit RMS exceeds
    /// this value. Higher = smaller files/lower quality. Without a threshold or --method,
    /// the default is RD at lambda=200. Ignored when --lambda is explicitly supplied;
    /// RD rate-estimation warm-up always uses a fixed threshold of 8. The implicit
    /// legacy threshold is also 8.
    #[arg(short = 'r', long)]
    t_rms: Option<f64>,

    /// Split threshold for the Cb/Cr planes of a colour image (Step 18: independent
    /// per-plane quality control). Like --t-rms, selects legacy partitioning unless
    /// --lambda is explicitly supplied, in which case it is ignored (including during
    /// the fixed-threshold-8 RD warm-up). Defaults to the luma threshold (8 if omitted).
    /// Chroma very
    /// commonly wants a looser (higher) threshold than luma, but the encoder never picks
    /// that automatically; the caller sets it explicitly.
    #[arg(long)]
    chroma_t_rms: Option<f64>,

    /// Chroma subsampling for colour input: `444` (no subsampling) or `420`
    /// (half-resolution Cb/Cr, box-filtered). Ignored for grayscale input.
    #[arg(long, value_enum, default_value = "444")]
    subsampling: SubsamplingArg,

    /// Step 14's `J = D + lambda*R` quality knob: the encoder searches every candidate
    /// block bottom-up and keeps whichever of "one leaf here" or "the four children" has
    /// the smaller distortion-plus-lambda-times-estimated-bits. This is the RD-optimal
    /// replacement for `--t-rms`/`-r`'s top-down threshold -- an RD curve is a lambda
    /// sweep, not a t_rms sweep. When given, `--t-rms` and `--chroma-t-rms` are ignored;
    /// the internal rate-estimation warm-up always uses a fixed threshold of 8.
    /// Larger lambda = more weight on rate = fewer/larger blocks. Defaults to 200
    /// unless --t-rms, --chroma-t-rms or --method selects the legacy path.
    #[arg(long)]
    lambda: Option<f64>,

    /// Experimental lambda-derived residual step. Fixed step 8 remains the default:
    /// the initial Kodak measurement regressed. Requires --lambda; stored in the stream
    /// so decmars needs no matching option. Requires mode 3, e.g. --modes 0,2,3.
    /// Applies to grayscale, colour and progressive.
    #[arg(long, requires = "lambda")]
    adaptive_residual: bool,

    /// Smallest range-block size: a power of two in 1..=128, no larger than --max-size.
    #[arg(long, default_value_t = 4)]
    min_size: u32,
    /// Largest range-block size: a power of two in 1..=128.
    #[arg(long, default_value_t = 16)]
    max_size: u32,
    /// Domain-pool stride: even, in 2..=254 (odd-origin contraction is unsupported).
    #[arg(long, default_value_t = 4)]
    shift: u32,
    /// Bits for the quantised contrast (alfa) coefficient: 2..=24.
    #[arg(long, default_value_t = 4)]
    bits_alfa: u32,
    /// Bits for the quantised brightness (beta) coefficient: 1..=24.
    #[arg(long, default_value_t = 7)]
    bits_beta: u32,
    /// Maximum contrast magnitude: an exact multiple of 1/32 in 1/32..=255/32,
    /// representable without rounding in the stream header.
    #[arg(long, default_value_t = 1.0)]
    max_alfa: f64,
    /// `-z`: qalfa threshold for the zero-alfa (flat block) override.
    #[arg(long, default_value_t = 0)]
    zero_threshold: u32,

    /// Width for headerless raw input.
    #[arg(long)]
    raw_width: Option<usize>,
    /// Height for headerless raw input.
    #[arg(long)]
    raw_height: Option<usize>,

    /// Step 16's content-adaptive domain-pool density: search a sparser domain-pool
    /// stride where a block's local pixel-domain RMS is low (near-flat), instead of
    /// always using the run's fixed `--shift`. An eval/wall-clock optimisation, not a
    /// quality knob: measured essentially BD-rate-neutral (mean -0.14% on kodim01/
    /// kodim02) but genuinely faster (~0.87x encode time) -- `docs/decisions.md` D48,
    /// which also documents why an earlier "denser where RMS is high" branch was removed
    /// (it could corrupt the bitstream; D43's originally-claimed -6.82% BD-rate number is
    /// withdrawn, see D48). Default off, matching every `encmars` invocation before this
    /// flag existed -- byte-identical output either way when omitted. Requires the RD
    /// path; with thresholds or --method, supply explicit --lambda to select RD.
    #[arg(long, default_value_t = false)]
    adaptive_density: bool,

    /// P5c: let this many of the search's best-RMS domain candidates compete for modes
    /// 2/3, instead of only the single minimum-RMS winner. `1` (the default) is the
    /// historical behaviour, byte-identical. Values above 1 require the RD path and the
    /// exhaustive search (omit `--method`, or use `--method exhaustive`). They add per-leaf
    /// mode competition -- real work, not a free retrieval widening -- but no stream
    /// syntax, so the chosen leaf's coordinate/coefficient fields are already fully priced.
    /// Not an output-preserving optimization: a wider list can change the partition and
    /// therefore the bytes. Applies to every plane; the P5c measurement was grayscale.
    #[arg(long, default_value_t = 1)]
    rd_candidates: usize,

    /// Step 15's per-leaf mode mask, a diagnostic/comparison knob: comma-separated mode
    /// numbers to allow, from 0 (flat), 1 (affine), 2 (fractal), 3 (fractal + residual) --
    /// e.g. `--modes 0,1,2` disables mode 3, `--modes 2` forces fractal-only. Only affects
    /// the RD path (the default); an explicit mask is rejected on the legacy path.
    /// With --method, supply --lambda to select RD. Default on RD: 0,2 (flat and fractal),
    /// the better measured combination on
    /// kodim01/02. Use --modes 0,1,2,3 to enable all four modes. Explicit masks are strict:
    /// if the final partition needs a mode outside the mask, encoding fails without
    /// writing output. Enable mode 0 for mandatory DC fallback at borders or where
    /// the selected search has no usable domain candidate.
    #[arg(long, value_delimiter = ',')]
    modes: Vec<u8>,

    /// Select a candidate-search provider inside the codec's production partition walk.
    /// Method alone retains legacy threshold partitioning at RMS 8; explicit --lambda
    /// enables RD, including --modes, --adaptive-density and --adaptive-residual.
    /// `exhaustive` uses the codec's own exhaustive provider, producing identical bytes
    /// to no-method encoding with the same threshold/lambda and options. Other methods
    /// use mars-search's indexed candidate providers. `learned` has baked-in weights.
    /// `random` is opt-in and requires --budget; --seed defaults to 0. Its budget
    /// limits production queries only, not the exhaustive RD warm-up, which may
    /// dominate total work.
    /// `apcc` is opt-in and requires --budget (no seed); the same budget limits apply.
    /// Grayscale only; mutually exclusive with --progressive for now.
    /// Reported `evals` counts production search only; `warmup_evals` and `total_evals`
    /// separately expose RD rate-estimation work.
    #[arg(long, value_enum)]
    method: Option<MethodArg>,

    /// Positive domain-position budget per production search query (8 isometry fits
    /// per position, at most 8*K fits). Required with --method random or apcc; rejected
    /// for other methods. Does not limit exhaustive RD warm-up fits.
    #[arg(long)]
    budget: Option<usize>,

    /// Reproducible random-search seed (default 0). Only valid with --method random.
    #[arg(long)]
    seed: Option<u64>,

    /// Pin Rayon's global thread pool to this many threads instead of Rayon's own
    /// default (`std::thread::available_parallelism()`). Matches `marsbench`'s own
    /// `rayon::ThreadPoolBuilder` usage exactly, so the two binaries' thread-count
    /// semantics don't silently diverge. Must be set before any parallel work runs
    /// (Rayon lazily builds an unpinned default pool on first use otherwise), so this is
    /// applied immediately after option validation. Determinism (Step 12/14's own
    /// discipline): every encode path in this crate is bit-identical across thread
    /// counts by construction, so this flag exists for controlling wall-clock/CPU usage,
    /// never for reproducing a different result.
    #[arg(long)]
    threads: Option<usize>,

    /// Write Step 19's 4-layer progressive container (base -> partition refinement ->
    /// fractal refinement -> residual refinement) instead of the single-layer `.mars` v0
    /// format -- so a `decmars --layer N` can decode any prefix independently, the whole
    /// point of a progressive stream. **Grayscale only** for this first cut: whether/how
    /// a progressive container composes with `mars_codec::color`'s per-plane YCbCr/
    /// subsampling wrapping (`progressive.rs` was not designed against it) is unresolved
    /// and explicitly out of scope here -- refused on colour input rather than silently
    /// encoding only the luma plane. Mutually exclusive with `--method` (the progressive
    /// encoder takes an already-built `(Header, Vec<Leaf>)`, which `--progressive` gets
    /// from the same RD/legacy partition every other `encmars` invocation uses --
    /// composing it with `mars-search`'s own candidate-restricted partition is unattempted
    /// and not wired). Composes with `--lambda`/`--t-rms`/`--adaptive-density`/`--modes`
    /// normally -- those choose the partition; `--progressive` only changes how that same
    /// partition is serialised.
    #[arg(long)]
    progressive: bool,

    /// Human-adaptive encoding: detect faces, then lower the RD lambda inside each face
    /// region -- and further inside its eyes, nose, and mouth -- so the encoder spends more
    /// bits where human viewers notice errors most. Needs the RD path (the default; omit
    /// --t-rms/--method, or supply --lambda). Regions apply to the luma/grayscale plane;
    /// chroma keeps the run's uniform lambda. Face detection needs a build with the
    /// `face-detect` feature (ONNX Runtime) and an SCRFD ONNX model via --scrfd-model.
    #[arg(long, default_value_t = false)]
    human_adaptive: bool,

    /// SCRFD ONNX model used by --human-adaptive (for example `det_500m.onnx`).
    #[arg(long, requires = "human_adaptive")]
    scrfd_model: Option<PathBuf>,

    /// Lambda multiplier inside a detected face box (default 0.5, i.e. half the run's
    /// lambda: more bits, lower error, inside the face).
    #[arg(long, default_value_t = 0.5, requires = "human_adaptive")]
    face_lambda_scale: f64,

    /// Lambda multiplier inside a detected eye, nose, or mouth box (default 0.25, applied
    /// on top of the face region: still more detail around the features within a face).
    #[arg(long, default_value_t = 0.25, requires = "human_adaptive")]
    feature_lambda_scale: f64,

    /// Detector confidence threshold for --human-adaptive (default 0.25, SCRFD's own).
    #[arg(long, default_value_t = 0.25, requires = "human_adaptive")]
    face_confidence: f32,

    /// Maximum number of faces --human-adaptive will adapt for (default 8).
    #[arg(long, default_value_t = 8, requires = "human_adaptive")]
    max_faces: usize,

    /// Debug aid for --human-adaptive: also write a copy of the input with the active
    /// regions outlined -- the face box, and the eye/nose/mouth boxes inside it. On colour
    /// input the face outline is green and the feature outlines red; on grayscale, face
    /// outlines are white and features mid-grey. The file format follows the extension, so
    /// pass a `.png` path.
    #[arg(long, value_name = "PATH", requires = "human_adaptive")]
    debug_regions: Option<PathBuf>,
}

impl Cli {
    fn effective_lambda(&self) -> Option<f64> {
        // Only an explicitly selected legacy path suppresses the RD default. Keeping
        // the parsed lambda optional preserves method-alone legacy and explicit RD precedence.
        self.lambda.or_else(|| {
            (self.t_rms.is_none() && self.chroma_t_rms.is_none() && self.method.is_none())
                .then_some(200.0)
        })
    }

    fn effective_t_rms(&self) -> f64 {
        self.t_rms.unwrap_or(8.0)
    }

    fn effective_modes(&self) -> &[u8] {
        // An omitted mask must not make ordinary legacy/method invocations fail.
        if self.modes.is_empty() {
            &[0, 2]
        } else {
            &self.modes
        }
    }

    fn validate_leaf_modes(&self, leaves: &[mars_codec::ifs::Leaf]) -> Result<()> {
        if !self.modes.is_empty() && leaves.iter().any(|leaf| !self.modes.contains(&leaf.mode)) {
            let modes = self
                .modes
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",");
            bail!(
                "--modes {modes} cannot represent this image with selected search; enable mode 0 for DC fallback"
            );
        }
        Ok(())
    }

    fn validate_container_modes(&self, bytes: &[u8]) -> Result<()> {
        if self.modes.is_empty() {
            return Ok(());
        }
        // color::read_container is private. Inspect each length-prefixed MARS plane
        // from our own MARC encoder output without decoding pixels or changing bytes.
        let header = bytes.get(..7).context("encoded MARC header truncated")?;
        if &header[..4] != b"MARC" || header[4] != 0 {
            bail!("unexpected encoded MARC header");
        }
        let mut remaining = &bytes[7..];
        for _ in 0..header[6] {
            let length = remaining
                .get(..4)
                .context("encoded MARC plane length truncated")?;
            let length = u32::from_le_bytes([length[0], length[1], length[2], length[3]]) as usize;
            remaining = &remaining[4..];
            let stream = remaining
                .get(..length)
                .context("encoded MARC plane truncated")?;
            let (_, leaves) = mars_codec::mars_format::read(stream)
                .context("reading encoded plane to validate --modes")?;
            self.validate_leaf_modes(&leaves)?;
            remaining = &remaining[length..];
        }
        if !remaining.is_empty() {
            bail!("unexpected trailing encoded MARC data");
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        match self.method {
            Some(MethodArg::Random) => {
                if self.budget.is_none_or(|budget| budget == 0) {
                    bail!("--method random requires an explicit positive --budget");
                }
            }
            Some(MethodArg::Apcc) => {
                if self.budget.is_none_or(|budget| budget == 0) {
                    bail!("--method apcc requires an explicit positive --budget");
                }
                if self.seed.is_some() {
                    bail!("--seed is only valid with --method random");
                }
            }
            _ => {
                if self.budget.is_some() {
                    bail!("--budget is only valid with --method random or apcc");
                }
                if self.seed.is_some() {
                    bail!("--seed is only valid with --method random");
                }
            }
        }
        for (name, value) in [
            ("--lambda", self.lambda),
            ("--t-rms", self.t_rms),
            ("--chroma-t-rms", self.chroma_t_rms),
        ] {
            if value.is_some_and(|v| !v.is_finite() || v < 0.0) {
                bail!("{name} must be finite and nonnegative");
            }
        }
        for (name, size) in [("--min-size", self.min_size), ("--max-size", self.max_size)] {
            if !size.is_power_of_two() || size > u32::from(u8::MAX) {
                bail!("{name} must be a power of two in 1..=128 (8-bit geometry field)");
            }
        }
        if self.min_size > self.max_size {
            bail!("--min-size must not exceed --max-size");
        }
        if self.shift == 0 || self.shift > u32::from(u8::MAX) || self.shift % 2 != 0 {
            bail!("--shift must be even and in 2..=254: odd-origin contraction is unsupported");
        }
        // Match mars_format's bounds; alfa=1 also gives the entropy coder an invalid
        // one-symbol fractal alphabet on the progressive path.
        if !(2..=24).contains(&self.bits_alfa) {
            bail!("--bits-alfa must be in 2..=24");
        }
        if !(1..=24).contains(&self.bits_beta) {
            bail!("--bits-beta must be in 1..=24");
        }
        // The fitter uses max_alfa directly, whereas the decoder uses int_max_alfa / 32.
        // Reject lossy header rounding rather than silently fitting a different model.
        let int_max_alfa = self.max_alfa * 32.0;
        if !int_max_alfa.is_finite()
            || !(1.0..=255.0).contains(&int_max_alfa)
            || int_max_alfa.fract() != 0.0
        {
            bail!(
                "--max-alfa must be finite and exactly representable in the header: a multiple of 1/32 in 1/32..=255/32"
            );
        }
        if self.progressive && self.method.is_some() {
            bail!("--progressive and --method are mutually exclusive");
        }

        if self.modes.iter().any(|&mode| mode > 3) {
            bail!("--modes must contain only mode numbers 0-3");
        }
        if self.effective_lambda().is_none() {
            if !self.modes.is_empty() {
                bail!(
                    "--modes requires the RD path; supply --lambda with legacy thresholds or --method"
                );
            }
            if self.adaptive_density {
                bail!(
                    "--adaptive-density requires the RD path; supply --lambda with legacy thresholds or --method"
                );
            }
        }
        if self.adaptive_residual && !self.effective_modes().contains(&3) {
            bail!("--adaptive-residual requires mode 3 in --modes (for example --modes 0,2,3)");
        }
        if self.rd_candidates < 1 {
            bail!("--rd-candidates must be at least 1");
        }
        if self.rd_candidates > 1 {
            if self.effective_lambda().is_none() {
                bail!(
                    "--rd-candidates requires the RD path; supply --lambda with legacy thresholds or --method"
                );
            }
            if !matches!(self.method, None | Some(MethodArg::Exhaustive)) {
                bail!(
                    "--rd-candidates > 1 requires the exhaustive search (omit --method or use --method exhaustive); other methods propose one candidate"
                );
            }
        }
        if self.human_adaptive {
            if self.effective_lambda().is_none() {
                bail!(
                    "--human-adaptive requires the RD path; supply --lambda with legacy thresholds or --method"
                );
            }
            for (name, value) in [
                ("--face-lambda-scale", self.face_lambda_scale),
                ("--feature-lambda-scale", self.feature_lambda_scale),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    bail!("{name} must be finite and positive");
                }
            }
            if !self.face_confidence.is_finite() || !(0.0..=1.0).contains(&self.face_confidence) {
                bail!("--face-confidence must be in 0.0..=1.0");
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    cli.validate()?;

    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("pinning Rayon's global thread pool to --threads")?;
    }

    let raw_dims = match (cli.raw_width, cli.raw_height) {
        (Some(w), Some(h)) => Some((w, h)),
        (None, None) => None,
        _ => bail!("--raw-width and --raw-height must be given together"),
    };
    let image = read_image(&cli.input, raw_dims)
        .with_context(|| format!("reading {}", cli.input.display()))?;
    if image.planes().len() != 1 && image.planes().len() != 3 {
        bail!(
            "{}: expected 1 (gray) or 3 (RGB) planes, got {}",
            cli.input.display(),
            image.planes().len()
        );
    }

    let base = EncodeParams {
        min_size: cli.min_size,
        max_size: cli.max_size,
        shift: cli.shift,
        bits_alfa: cli.bits_alfa,
        bits_beta: cli.bits_beta,
        max_alfa: cli.max_alfa,
        t_rms: cli.effective_t_rms(),
        zero_threshold: cli.zero_threshold,
        lambda: cli.effective_lambda(),
    };
    let mut allowed_modes = [false; 4];
    for &mode in cli.effective_modes() {
        allowed_modes[usize::from(mode)] = true;
    }
    let lambda_regions = human_adaptive_regions(&cli, &image)?;
    let options = EncodeOptions {
        allowed_modes,
        adaptive_density: cli.adaptive_density,
        residual_quantisation: residual_policy(&cli),
        rd_candidates: cli.rd_candidates,
        lambda_regions: lambda_regions.clone(),
    };

    if let Some(method_arg) = cli.method {
        if image.planes().len() != 1 {
            bail!(
                "{}: --method only supports grayscale input for now (got {} planes) -- \
                 see --help for --method",
                cli.input.display(),
                image.planes().len()
            );
        }
        return run_with_method(&cli, method_arg, &image, &base, &options);
    }

    if cli.progressive && image.planes().len() != 1 {
        bail!(
            "{}: --progressive only supports grayscale input for now (got {} planes) -- \
             see --help for --progressive",
            cli.input.display(),
            image.planes().len()
        );
    }

    let chroma = EncodeParams {
        t_rms: cli.chroma_t_rms.unwrap_or(cli.effective_t_rms()),
        ..base
    };
    if cli.progressive {
        return run_progressive(&cli, &base, &options, &image);
    }

    let params = ColorEncodeParams {
        y: base,
        chroma,
        subsampling: cli.subsampling.into(),
        adaptive_density: options.adaptive_density,
        allowed_modes: options.allowed_modes,
        rd_candidates: cli.rd_candidates,
        lambda_regions,
    };

    let (width, height) = (image.width(), image.height());
    let (bytes, stats) = encode_color_image_with_residual_quantisation(
        &image,
        &params,
        options.residual_quantisation,
    );
    cli.validate_container_modes(&bytes)?;

    std::fs::write(&cli.output, &bytes)
        .with_context(|| format!("writing {}", cli.output.display()))?;

    let bpp = 8.0 * bytes.len() as f64 / (width * height) as f64;
    if image.planes().len() == 1 {
        println!(
            "{width}x{height} gray -> {} ({} bytes, {bpp:.3} bpp, {} evals)",
            cli.output.display(),
            bytes.len(),
            stats.y_evals,
        );
    } else {
        println!(
            "{width}x{height} rgb ({:?}) -> {} ({} bytes total: Y {} + Cb {} + Cr {}, \
             {bpp:.3} bpp, evals Y {} / Cb {} / Cr {})",
            cli.subsampling,
            cli.output.display(),
            bytes.len(),
            stats.y_bytes,
            stats.cb_bytes,
            stats.cr_bytes,
            stats.y_evals,
            stats.cb_evals,
            stats.cr_evals,
        );
    }
    Ok(())
}

/// Select only the search provider; partitioning and serialization use the production
/// codec, with the same validated parameters and options as the no-method path.
fn run_with_method(
    cli: &Cli,
    method_arg: MethodArg,
    image: &mars_core::image::Image,
    params: &EncodeParams,
    options: &EncodeOptions,
) -> Result<()> {
    let plane = &image.planes()[0];
    let (outcome, search_details) = match method_arg {
        MethodArg::Random => {
            let config = mars_search::random::RandomConfig {
                budget: cli.budget.context("--method random requires --budget")?,
                seed: cli.seed.unwrap_or(0),
            };
            let details = format!(
                "budget={}, seed={}, rng_version={}, ",
                config.budget,
                config.seed,
                mars_search::random::RNG_VERSION
            );
            let outcome = encode_image_with_search(plane, params, options, |contracted| {
                Box::new(mars_search::random::RandomSearchProvider::build(
                    plane, contracted, params, options, config,
                ))
            });
            (outcome, details)
        }
        MethodArg::Apcc => {
            let budget = cli.budget.context("--method apcc requires --budget")?;
            let details = format!("budget={budget}, ");
            let outcome = encode_image_with_search(plane, params, options, |contracted| {
                Box::new(mars_search::apcc::ApccSearchProvider::build(
                    plane,
                    contracted,
                    params,
                    options,
                    mars_search::apcc::ApccConfig { budget },
                ))
            });
            (outcome, details)
        }
        other => {
            let Some(method) = other.indexed_method() else {
                bail!("no search provider wired for --method {}", other.key());
            };
            (
                encode_image_with_search(plane, params, options, |contracted| {
                    if matches!(other, MethodArg::Exhaustive) {
                        Box::new(ExhaustiveSearch)
                    } else {
                        Box::new(mars_search::IndexedSearchProvider::build(
                            plane, contracted, params, options, method,
                        ))
                    }
                }),
                String::new(),
            )
        }
    };
    cli.validate_leaf_modes(&outcome.leaves)?;
    let transforms = outcome.leaves.len() as u64;
    let evals = outcome.counters.search_evals;
    let warmup_evals = outcome.counters.warmup_evals;
    let total_evals = outcome.counters.total_evals();

    let plane_bytes = mars_codec::mars_format::write(&outcome.header, &outcome.leaves)
        .context("serializing the production partition with the selected search provider")?;
    let bytes = wrap_gray_stream(plane_bytes);

    std::fs::write(&cli.output, &bytes)
        .with_context(|| format!("writing {}", cli.output.display()))?;

    let (width, height) = (image.width(), image.height());
    let bpp = 8.0 * bytes.len() as f64 / (width * height) as f64;
    println!(
        "{width}x{height} gray -> {} ({} bytes, {bpp:.3} bpp, method {}, {evals} evals, \
         {search_details}search_evals={evals}, warmup_evals={warmup_evals}, total_evals={total_evals}, \
         {transforms} transforms, {:.2} evals/transform)",
        cli.output.display(),
        bytes.len(),
        method_arg.key(),
        evals as f64 / transforms.max(1) as f64,
    );
    Ok(())
}

fn residual_policy(cli: &Cli) -> ResidualQuantisation {
    if cli.adaptive_residual {
        ResidualQuantisation::LambdaAdaptive
    } else {
        ResidualQuantisation::default()
    }
}

/// Encode the same RD/legacy partition, including the selected residual policy,
/// into the progressive container instead of the single-layer representation.
fn run_progressive(
    cli: &Cli,
    base: &EncodeParams,
    options: &EncodeOptions,
    image: &mars_core::image::Image,
) -> Result<()> {
    let plane = &image.planes()[0];
    let (hdr, leaves, evals, _stats) =
        mars_codec::encode::encode_image_with_options(plane, base, options);
    cli.validate_leaf_modes(&leaves)?;
    let bytes = mars_codec::progressive::encode(plane, &hdr, &leaves)
        .context("building the progressive container")?;

    std::fs::write(&cli.output, &bytes)
        .with_context(|| format!("writing {}", cli.output.display()))?;

    let offsets = mars_codec::progressive::layer_end_offsets(&bytes)
        .expect("bytes this call just produced are always a valid progressive stream");
    let (width, height) = (image.width(), image.height());
    let bpp = 8.0 * bytes.len() as f64 / (width * height) as f64;
    println!(
        "{width}x{height} gray -> {} ({} bytes, {bpp:.3} bpp, progressive, {evals} evals, \
         {} leaves, layer end-offsets {offsets:?})",
        cli.output.display(),
        bytes.len(),
        leaves.len(),
    );
    Ok(())
}

/// Build the region-local lambda set for `--human-adaptive`, or an empty set when the flag
/// is off. Detection itself is behind the `face-detect` feature; without that feature the
/// flag fails with a build instruction rather than silently encoding without adaptation.
///
/// `--debug-regions` writes an overlay of the same plan the encoder is handed, so the file
/// shows exactly which rectangles were active.
#[cfg(feature = "face-detect")]
fn human_adaptive_regions(cli: &Cli, image: &mars_core::image::Image) -> Result<Vec<LambdaRegion>> {
    use mars_cli::human::RegionKind;

    if !cli.human_adaptive {
        return Ok(Vec::new());
    }
    let model = cli
        .scrfd_model
        .as_ref()
        .context("--human-adaptive requires --scrfd-model")?;
    let faces = mars_cli::scrfd::detect_faces(model, image, cli.face_confidence, cli.max_faces)
        .with_context(|| format!("detecting faces in {}", cli.input.display()))?;
    let plan = mars_cli::human::plan_regions(
        &faces,
        image.width() as u32,
        image.height() as u32,
        cli.face_lambda_scale,
        cli.feature_lambda_scale,
    );

    if let Some(path) = &cli.debug_regions {
        let annotated = mars_cli::overlay::draw_regions(image, &plan);
        mars_cli::overlay::write_image(&annotated, path)
            .with_context(|| format!("writing the region overlay to {}", path.display()))?;
    }
    let faces_found = plan
        .iter()
        .filter(|planned| planned.kind == RegionKind::Face)
        .count();
    let overlay = match &cli.debug_regions {
        Some(path) => format!(", overlay -> {}", path.display()),
        None => String::new(),
    };
    println!(
        "human-adaptive: {faces_found} face(s), {} region(s){overlay}",
        plan.len()
    );
    if cli.debug_regions.is_some() {
        for planned in &plan {
            println!(
                "  {:?} rows {}..{} cols {}..{} (lambda x{})",
                planned.kind,
                planned.region.row,
                planned.region.row + planned.region.height,
                planned.region.col,
                planned.region.col + planned.region.width,
                planned.region.scale,
            );
        }
    }

    Ok(plan.into_iter().map(|planned| planned.region).collect())
}

#[cfg(not(feature = "face-detect"))]
fn human_adaptive_regions(
    cli: &Cli,
    _image: &mars_core::image::Image,
) -> Result<Vec<LambdaRegion>> {
    if cli.human_adaptive {
        bail!(
            "--human-adaptive needs the `face-detect` build feature (ONNX Runtime); \
             rebuild with `cargo build -p mars-cli --features face-detect`"
        );
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(
            ["encmars", "input.png", "output.mars"]
                .into_iter()
                .chain(args.iter().copied()),
        )
        .expect("valid arguments")
    }

    #[test]
    fn defaults_select_rd_and_measured_modes_without_other_experiments() {
        let cli = parse(&[]);
        assert_eq!(cli.effective_lambda(), Some(200.0));
        assert_eq!(cli.effective_t_rms(), 8.0);
        assert!(cli.modes.is_empty());
        assert_eq!(cli.effective_modes(), [0, 2]);
        assert!(matches!(cli.subsampling, SubsamplingArg::Yuv444));
        assert!(!cli.adaptive_density && !cli.adaptive_residual && !cli.progressive);
    }

    #[test]
    fn explicit_thresholds_and_methods_keep_the_legacy_path() {
        for args in [
            vec!["--t-rms", "8"],
            vec!["-r", "4"],
            vec!["--chroma-t-rms", "16"],
            vec!["--method", "fisher"],
        ] {
            assert_eq!(parse(&args).effective_lambda(), None);
        }
        assert_eq!(parse(&["-r", "4"]).effective_t_rms(), 4.0);
    }

    #[test]
    fn format_boundaries_are_not_replaced_with_arbitrary_limits() {
        for args in [
            vec!["--bits-alfa", "24", "--bits-beta", "24"],
            vec!["--min-size", "1", "--max-size", "128", "--shift", "254"],
            vec!["--max-alfa", "0.03125"],
            vec!["--max-alfa", "7.96875"],
            vec!["--t-rms", "1e300", "--zero-threshold", "4294967295"],
        ] {
            parse(&args).validate().expect("supported parameter bounds");
        }
    }

    #[test]
    fn rd_candidates_above_one_require_the_rd_exhaustive_path() {
        assert_eq!(parse(&[]).rd_candidates, 1);
        assert_eq!(parse(&["--rd-candidates", "1"]).rd_candidates, 1);
        // Above 1 needs the RD path and the exhaustive search.
        assert!(parse(&["--rd-candidates", "3"]).validate().is_ok());
        assert!(parse(&["--lambda", "200", "--rd-candidates", "3"])
            .validate()
            .is_ok());
        assert!(parse(&[
            "--method",
            "exhaustive",
            "--lambda",
            "200",
            "--rd-candidates",
            "3"
        ])
        .validate()
        .is_ok());
        // A legacy threshold, a candidate-restricted method, and zero are all refused.
        assert!(parse(&["--rd-candidates", "3", "-r", "8"])
            .validate()
            .is_err());
        assert!(parse(&[
            "--method",
            "fisher",
            "--lambda",
            "200",
            "--rd-candidates",
            "3"
        ])
        .validate()
        .is_err());
        assert!(parse(&["--rd-candidates", "0"]).validate().is_err());
    }

    #[test]
    fn explicit_lambda_overrides_threshold_and_modes_replace_defaults() {
        let cli = parse(&[
            "--lambda",
            "50",
            "--t-rms",
            "0",
            "--chroma-t-rms",
            "16",
            "--modes",
            "0,1,2,3",
        ]);
        assert_eq!(cli.effective_lambda(), Some(50.0));
        assert_eq!(cli.effective_t_rms(), 0.0);
        assert_eq!(cli.modes, [0, 1, 2, 3]);
        assert_eq!(parse(&["--lambda", "0"]).effective_lambda(), Some(0.0));
        assert_eq!(parse(&["--modes", "3"]).modes, [3]);
        assert_eq!(parse(&["--progressive"]).effective_lambda(), Some(200.0));
    }

    #[test]
    fn human_adaptive_requires_the_rd_path_and_positive_scales() {
        let cli = parse(&["--human-adaptive"]);
        assert!(cli.human_adaptive);
        // The default RD path satisfies the requirement.
        assert!(cli.validate().is_ok());
        // A legacy threshold leaves no RD path to adapt.
        assert!(parse(&["--human-adaptive", "-r", "8"]).validate().is_err());
        // Both scales must be positive and finite.
        assert!(parse(&["--human-adaptive", "--face-lambda-scale", "0"])
            .validate()
            .is_err());
        assert!(parse(&["--human-adaptive", "--feature-lambda-scale=-1"])
            .validate()
            .is_err());
        // Confidence is a probability.
        assert!(parse(&["--human-adaptive", "--face-confidence", "1.5"])
            .validate()
            .is_err());
        // The detector knobs are meaningless without the flag, and clap refuses them.
        assert!(
            Cli::try_parse_from(["encmars", "in.png", "out.mars", "--scrfd-model", "m.onnx"])
                .is_err()
        );
    }
}
