//! rANS entropy coding for the `.mars` container -- Step 10.
//!
//! [`AdaptiveModel`] is a small order-0 adaptive context model backed by plain integer
//! counts, quantised to a fixed-point CDF with pure integer arithmetic (no floats, so the
//! quantisation is bit-identical across platforms -- see `implementation-plan.md` §2.3).
//! [`Coder`] drives a `constriction` `AnsCoder` stack: because ANS is LIFO, encoding
//! records a forward pass of `(context, symbol)` events against evolving per-context
//! models, then pushes the recorded (symbol, fixed-point-interval) pairs onto the stack in
//! *reverse*, so that decoding -- which replays the same per-context models forward,
//! popping in forward order -- reconstructs the identical symbol sequence.

use std::collections::HashMap;
use std::num::NonZeroU32;

use constriction::stream::model::{DecoderModel, EncoderModel, EntropyModel};
use constriction::stream::stack::DefaultAnsCoder;
use constriction::stream::{Decode, Encode};

/// Fixed-point precision shared by every model in this crate. `u32` probabilities give 8
/// bits of headroom above this, which is what `constriction`'s own `Default` aliases use.
pub const PRECISION: usize = 24;

const PRECISION_TOTAL: u64 = 1u64 << PRECISION;

/// An order-0 adaptive model over `0..alphabet_size`, seeded uniform (Laplace: every
/// symbol starts with count 1) and updated after every symbol it codes.
///
/// The fixed-point CDF is derived from the raw counts by pure integer division (the same
/// technique range coders like LZMA use): symbol `i`'s quantised left-cumulative is
/// `scaled(count_before_i) + i`, reserving one fixed-point slot per symbol before scaling
/// the rest proportionally to its count. This guarantees every symbol keeps a nonzero
/// probability and the whole computation is exact integer arithmetic -- deterministic
/// across threads and platforms, unlike the floating-point path `constriction`'s own
/// `Categorical` models take.
#[derive(Debug, Clone)]
pub struct AdaptiveModel {
    counts: Vec<u32>,
    total: u64,
}

/// Above this total, halve every count (floor at 1) so the model keeps adapting to local
/// statistics instead of a whole-image running average drowning out later context.
const RESCALE_THRESHOLD: u64 = 1 << 16;

impl AdaptiveModel {
    pub fn new(alphabet_size: u32) -> Self {
        assert!(
            alphabet_size >= 2,
            "a 1-symbol alphabet carries no information"
        );
        assert!(
            u64::from(alphabet_size) < PRECISION_TOTAL,
            "alphabet too large for PRECISION"
        );
        Self {
            counts: vec![1u32; alphabet_size as usize],
            total: u64::from(alphabet_size),
        }
    }

    pub fn update(&mut self, symbol: u32) {
        self.counts[symbol as usize] += 32;
        self.total += 32;
        if self.total >= RESCALE_THRESHOLD {
            self.rescale();
        }
    }

    fn rescale(&mut self) {
        let mut total = 0u64;
        for c in &mut self.counts {
            *c = (*c >> 1).max(1);
            total += u64::from(*c);
        }
        self.total = total;
    }

    /// The quantised cumulative for a raw count-space cumulative `c` (`0..=self.total`).
    fn scaled(&self, c: u64) -> u32 {
        let free_weight = PRECISION_TOTAL - self.counts.len() as u64;
        ((u128::from(c) * u128::from(free_weight)) / u128::from(self.total)) as u32
    }

    /// `(left_cumulative, probability)` for `symbol`, both in fixed-point PRECISION space.
    fn left_and_prob(&self, symbol: usize) -> (u32, u32) {
        let before: u64 = self.counts[..symbol].iter().map(|&c| u64::from(c)).sum();
        let after = before + u64::from(self.counts[symbol]);
        let left = self.scaled(before) + symbol as u32;
        let right = self.scaled(after) + symbol as u32 + 1;
        (left, right - left)
    }

    /// Step 14: the bits it would cost to code `symbol` under this model's *current*
    /// state, without mutating it -- `-log2(probability / 2^PRECISION)`. A pure query (no
    /// `update`), so it is safe to call from multiple threads on a shared, read-only
    /// model -- exactly what the rate estimator needs to price a candidate without
    /// committing to it.
    pub fn bits_for(&self, symbol: u32) -> f64 {
        let (_, prob) = self.left_and_prob(symbol as usize);
        -(f64::from(prob) / PRECISION_TOTAL as f64).log2()
    }

    /// The inverse of [`Self::left_and_prob`]: the symbol whose interval contains
    /// `quantile`, plus that interval.
    fn symbol_for_quantile(&self, quantile: u32) -> (u32, u32, u32) {
        let mut cum = 0u64;
        for (i, &c) in self.counts.iter().enumerate() {
            let next = cum + u64::from(c);
            let left = self.scaled(cum) + i as u32;
            let right = self.scaled(next) + i as u32 + 1;
            if quantile < right {
                return (i as u32, left, right - left);
            }
            cum = next;
        }
        unreachable!(
            "quantile {quantile} out of range for a model with total {}",
            self.total
        )
    }
}

impl EntropyModel<PRECISION> for AdaptiveModel {
    type Symbol = u32;
    type Probability = u32;
}

impl EncoderModel<PRECISION> for AdaptiveModel {
    fn left_cumulative_and_probability(
        &self,
        symbol: impl std::borrow::Borrow<u32>,
    ) -> Option<(u32, NonZeroU32)> {
        let symbol = *symbol.borrow();
        if symbol as usize >= self.counts.len() {
            return None;
        }
        let (left, prob) = self.left_and_prob(symbol as usize);
        Some((
            left,
            NonZeroU32::new(prob).expect("probability is always >= 1"),
        ))
    }
}

impl DecoderModel<PRECISION> for AdaptiveModel {
    fn quantile_function(&self, quantile: u32) -> (u32, u32, NonZeroU32) {
        let (symbol, left, prob) = self.symbol_for_quantile(quantile);
        (
            symbol,
            left,
            NonZeroU32::new(prob).expect("probability is always >= 1"),
        )
    }
}

/// A single symbol's already-computed fixed-point interval, used to encode onto the rANS
/// stack in reverse order without re-deriving it from a model whose state has since moved
/// on. See the module doc for why this indirection exists.
#[derive(Debug, Clone, Copy)]
struct FixedInterval {
    left: u32,
    prob: NonZeroU32,
}

impl EntropyModel<PRECISION> for FixedInterval {
    type Symbol = u32;
    type Probability = u32;
}

impl EncoderModel<PRECISION> for FixedInterval {
    fn left_cumulative_and_probability(
        &self,
        _symbol: impl std::borrow::Borrow<u32>,
    ) -> Option<(u32, NonZeroU32)> {
        Some((self.left, self.prob))
    }
}

/// Identifies one context: a field kind plus a small integer disambiguator (depth, size
/// class, ...). Distinct keys get independent, independently-adapting models.
pub type ContextKey = (u8, u32);

/// One `(context, alphabet size, symbol)` event in the canonical traversal order --
/// recorded during encoding (the symbols are already known) and produced one at a time
/// during decoding (each is decoded before the next is requested, since later events'
/// context and even existence can depend on earlier symbols, e.g. a split flag).
#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub ctx: ContextKey,
    pub alphabet: u32,
    pub symbol: u32,
}

/// Records a forward pass of events against evolving per-context [`AdaptiveModel`]s, then
/// pushes them onto a fresh `AnsCoder` in reverse, and returns the compressed bytes.
///
/// `events` must be in the exact order a matching [`Decoder`] will ask for them --
/// typically produced by walking the same recursive structure the decoder will walk,
/// reading symbols from already-known data instead of decoding them.
pub fn encode(events: &[Event]) -> Vec<u8> {
    encode_recorded(record(events).0)
}

/// The compressed bytes, plus each event's own information content under the *live*,
/// evolving per-context model -- `-log2(p_i)` for the exact fixed-point probability the
/// coder used. This is what the stream really paid per event, as distinct from any frozen
/// estimate a caller compared it against (research plan §8 P5a).
#[derive(Debug, Clone, PartialEq)]
pub struct Coded {
    pub bytes: Vec<u8>,
    /// `bits_per_event[i]` belongs to `events[i]`, in the same order.
    pub bits_per_event: Vec<f64>,
    /// `sum(bits_per_event)`. Bounded above by `8 * bytes.len()`: the difference is the
    /// coder's final state and word-alignment padding, not hidden cost.
    pub bits_total: f64,
}

impl Coded {
    /// The byte-aligned size of the compressed payload, in bits.
    pub fn byte_aligned_bits(&self) -> f64 {
        self.bytes.len() as f64 * 8.0
    }
}

/// [`encode`], additionally reporting each event's own cost under the evolving models.
/// Produces byte-identical output to [`encode`] on the same events -- the recording loop is
/// shared, and the extra vector is a query, never a mutation.
pub fn encode_with_bits(events: &[Event]) -> Coded {
    let (recorded, bits_per_event) = record(events);
    let bits_total = bits_per_event.iter().sum();
    Coded {
        bytes: encode_recorded(recorded),
        bits_per_event,
        bits_total,
    }
}

/// The shared forward pass: evolve one model per context, recording each symbol's
/// fixed-point interval and the `-log2 p` cost that interval represents.
fn record(events: &[Event]) -> (Vec<(u32, FixedInterval)>, Vec<f64>) {
    let mut models: HashMap<ContextKey, AdaptiveModel> = HashMap::new();
    let mut recorded: Vec<(u32, FixedInterval)> = Vec::with_capacity(events.len());
    let mut bits = Vec::with_capacity(events.len());
    for ev in events {
        let model = models
            .entry(ev.ctx)
            .or_insert_with(|| AdaptiveModel::new(ev.alphabet));
        let (left, prob) = model.left_and_prob(ev.symbol as usize);
        bits.push(-(f64::from(prob) / PRECISION_TOTAL as f64).log2());
        recorded.push((
            ev.symbol,
            FixedInterval {
                left,
                prob: NonZeroU32::new(prob).expect("probability is always >= 1"),
            },
        ));
        model.update(ev.symbol);
    }
    (recorded, bits)
}

/// Step 14: replay `events` against fresh per-context [`AdaptiveModel`]s exactly the way
/// [`encode`] does, but return the resulting models instead of encoding -- the rate
/// estimator's "warm-up" snapshot (`mars-codec`'s `rate` module) is built by handing this
/// the event stream of a representative (not necessarily RD-optimal) partition, so its
/// per-context statistics are real, observed frequencies rather than a constant-bits
/// stand-in. Shares the exact model-building loop `encode` uses, so the snapshot is
/// bit-for-bit what `encode` itself would have converged to on the same event stream.
pub fn build_models(events: &[Event]) -> HashMap<ContextKey, AdaptiveModel> {
    let mut models: HashMap<ContextKey, AdaptiveModel> = HashMap::new();
    for ev in events {
        let model = models
            .entry(ev.ctx)
            .or_insert_with(|| AdaptiveModel::new(ev.alphabet));
        model.update(ev.symbol);
    }
    models
}

fn encode_recorded(recorded: Vec<(u32, FixedInterval)>) -> Vec<u8> {
    let mut ans = DefaultAnsCoder::new();
    for &(symbol, interval) in recorded.iter().rev() {
        ans.encode_symbol(symbol, interval)
            .expect("fixed-point interval is always valid");
    }
    let words = ans
        .into_compressed()
        .expect("Vec backend never fails to write");
    bytemuck_words_to_bytes(&words)
}

/// [`Decoder::new`]'s only failure mode: an empty compressed payload is fine, but a
/// nonempty one ending in an all-zero word cannot have come from [`encode`] and is
/// rejected rather than misinterpreted.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("compressed payload ends in a trailing all-zero word, which `encode` never produces")]
pub struct InvalidCompressedData;

/// The decode-side counterpart of [`encode`]. Call [`Decoder::next`] once per event, in
/// the same order [`encode`]'s caller produced them, passing that event's context and
/// alphabet size (both must be derivable from already-decoded data, matching the
/// encoder's traversal exactly).
pub struct Decoder {
    ans: DefaultAnsCoder,
    models: HashMap<ContextKey, AdaptiveModel>,
}

impl Decoder {
    /// Fails only if `bytes` (word-aligned to 4 bytes, dropping any trailing partial
    /// word) ends in an all-zero word -- the one input `AnsCoder::from_compressed` itself
    /// refuses, since it cannot represent trailing zero words. Never panics, so a
    /// `.mars` fuzz target can feed it arbitrary bytes.
    pub fn new(bytes: &[u8]) -> Result<Self, InvalidCompressedData> {
        let words = bytes_to_words(bytes);
        let ans = DefaultAnsCoder::from_compressed(words).map_err(|_| InvalidCompressedData)?;
        Ok(Self {
            ans,
            models: HashMap::new(),
        })
    }

    /// Decode the next symbol from context `ctx`, whose alphabet has `alphabet` symbols.
    pub fn next(&mut self, ctx: ContextKey, alphabet: u32) -> u32 {
        let model = self
            .models
            .entry(ctx)
            .or_insert_with(|| AdaptiveModel::new(alphabet));
        let symbol = self
            .ans
            .decode_symbol(&*model)
            .expect("decoding from a `Vec` backend is infallible");
        model.update(symbol);
        symbol
    }
}

fn bytemuck_words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_skewed_stream() {
        // A single context, heavily skewed toward symbol 0 -- the case adaptive coding
        // should compress well and, more importantly, must still round-trip exactly.
        let symbols: Vec<u32> = (0..2000).map(|i| if i % 7 == 0 { 1 } else { 0 }).collect();
        let events: Vec<Event> = symbols
            .iter()
            .map(|&s| Event {
                ctx: (0, 0),
                alphabet: 2,
                symbol: s,
            })
            .collect();
        let bytes = encode(&events);

        let mut dec = Decoder::new(&bytes).unwrap();
        let decoded: Vec<u32> = symbols.iter().map(|_| dec.next((0, 0), 2)).collect();
        assert_eq!(decoded, symbols);

        // Skewed binary data at ~2000 symbols should be well under 1 bit/symbol.
        assert!(
            bytes.len() < 2000 / 8 * 3,
            "got {} bytes for 2000 symbols",
            bytes.len()
        );
    }

    #[test]
    fn round_trips_multiple_contexts_and_alphabets() {
        let mut events = Vec::new();
        let mut expected = Vec::new();
        for i in 0..500u32 {
            let ctx: ContextKey = (u8::try_from(i % 3).unwrap(), i % 5);
            let alphabet = 4 + (i % 3) * 10;
            let symbol = i % alphabet;
            events.push(Event {
                ctx,
                alphabet,
                symbol,
            });
            expected.push((ctx, alphabet, symbol));
        }
        let bytes = encode(&events);
        let mut dec = Decoder::new(&bytes).unwrap();
        for (ctx, alphabet, symbol) in expected {
            assert_eq!(dec.next(ctx, alphabet), symbol);
        }
    }

    #[test]
    fn single_symbol_alphabet_is_rejected() {
        let result = std::panic::catch_unwind(|| AdaptiveModel::new(1));
        assert!(result.is_err());
    }

    #[test]
    fn encode_with_bits_is_byte_identical_and_its_cost_bounds_the_payload() {
        let symbols: Vec<u32> = (0..4000).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();
        let events: Vec<Event> = symbols
            .iter()
            .enumerate()
            .map(|(i, &s)| Event {
                ctx: (u8::try_from(i % 2).unwrap(), 0),
                alphabet: 2,
                symbol: s,
            })
            .collect();

        let plain = encode(&events);
        let coded = encode_with_bits(&events);
        assert_eq!(
            coded.bytes, plain,
            "the accounting pass must not change the bytes"
        );
        assert_eq!(coded.bits_per_event.len(), events.len());
        assert!(coded
            .bits_per_event
            .iter()
            .all(|b| b.is_finite() && *b >= 0.0));

        // The recorded costs are the real payload: byte alignment, the coder's final
        // state, and word padding are the only difference, never a large hidden term.
        let slack = coded.byte_aligned_bits() - coded.bits_total;
        assert!(
            slack >= -1e-9 && slack < 128.0,
            "cost {} vs payload {} bits (slack {slack})",
            coded.bits_total,
            coded.byte_aligned_bits()
        );
    }

    #[test]
    fn encode_with_bits_reports_a_deterministic_cost_for_a_repeated_event() {
        // The same context/symbol pair gets strictly cheaper as its model adapts, which is
        // the property the audit relies on to compare a frozen estimate to a live cost.
        let events: Vec<Event> = (0..64)
            .map(|_| Event {
                ctx: (0, 0),
                alphabet: 8,
                symbol: 3,
            })
            .collect();
        let coded = encode_with_bits(&events);
        assert!(coded.bits_per_event[0] > coded.bits_per_event[63]);
        assert_eq!(
            coded.bits_per_event,
            encode_with_bits(&events).bits_per_event
        );
    }
}
