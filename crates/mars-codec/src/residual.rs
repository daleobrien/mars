//! Context-modelled rANS event vocabulary for Step 15's residual-mode DCT coefficients.
//!
//! One leaf's residual is `size*size` quantised [`crate::dct`] coefficients, in the same
//! row-major order [`crate::dct::forward_dct2d`] produces them (index 0 is DC). Coding
//! them coefficient-by-coefficient rather than run-length/EOB (JPEG's classic scheme) is
//! the simpler design the brief asks Step 15 to start with: two events per position
//! (nonzero flag, then a signed magnitude only if nonzero), each keyed by a
//! `(size_class, frequency band)` context so DC/low-frequency and high-frequency
//! coefficients -- which have very different statistics -- get independently adapted
//! models via `mars_entropy`'s existing per-context machinery, without inventing a new
//! entropy backend for this step.
//!
//! Levels are clamped to `+-LEVEL_CLAMP` before coding: a bounded alphabet keeps the event
//! vocabulary simple (no escape codes) at the cost of a documented approximation -- an
//! outlier coefficient beyond the clamp is coded at the clamp, which is a real, bounded
//! distortion contribution the RD search's own SSE accounting sees and prices like any
//! other, so it cannot silently make residual mode look better than it is.

use mars_entropy::{ContextKey, Decoder, Event};

/// Field id for the "is this coefficient nonzero" bit, in the same `u8` id space
/// `mars_format`'s `FIELD_*` constants use (that module owns the full field list; this
/// value must not collide with any of them -- checked by the shared-context regression
/// test in `mars_format`'s own test module).
pub(crate) const FIELD_RESID_NZ: u8 = 10;
/// Field id for a nonzero coefficient's signed magnitude.
pub(crate) const FIELD_RESID_MAG: u8 = 11;

/// Quantised levels are clamped to `[-LEVEL_CLAMP, LEVEL_CLAMP]` before coding (see module
/// doc). `63` keeps the magnitude alphabet (`2*LEVEL_CLAMP + 1 = 127`) comfortably inside
/// `mars_entropy::PRECISION`'s headroom while covering every level a `residual_qstep`
/// derived from any lambda this project sweeps actually produces in practice.
pub const LEVEL_CLAMP: i32 = 63;

/// Number of frequency bands a `size x size` coefficient block is bucketed into for
/// context selection -- coarse (four bands: DC-heavy through high-frequency) rather than
/// per-position, so each context still accumulates enough observations to adapt
/// meaningfully within one image.
const BANDS: u32 = 4;

fn band_of(index: usize, n: usize) -> u32 {
    (((index * BANDS as usize) / n) as u32).min(BANDS - 1)
}

fn nz_ctx(size_class: u32, band: u32) -> ContextKey {
    (FIELD_RESID_NZ, size_class * BANDS + band)
}

fn mag_ctx(size_class: u32, band: u32) -> ContextKey {
    (FIELD_RESID_MAG, size_class * BANDS + band)
}

fn zigzag(x: i32) -> u32 {
    ((x << 1) ^ (x >> 31)) as u32
}

fn unzigzag(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

/// Append the events coding `levels` (already dead-zone-quantised, one per DCT coefficient
/// in row-major order) for a `size_class` leaf.
pub(crate) fn encode_events(levels: &[i32], size_class: u32, events: &mut Vec<Event>) {
    let n = levels.len();
    for (i, &level) in levels.iter().enumerate() {
        let band = band_of(i, n);
        let clamped = level.clamp(-LEVEL_CLAMP, LEVEL_CLAMP);
        let nonzero = clamped != 0;
        events.push(Event {
            ctx: nz_ctx(size_class, band),
            alphabet: 2,
            symbol: u32::from(nonzero),
        });
        if nonzero {
            events.push(Event {
                ctx: mag_ctx(size_class, band),
                alphabet: (2 * LEVEL_CLAMP + 1) as u32,
                symbol: zigzag(clamped),
            });
        }
    }
}

/// Decode `n` coefficient levels for a `size_class` leaf from `dec`, in the same order
/// [`encode_events`] wrote them.
pub(crate) fn decode_values(dec: &mut Decoder, size_class: u32, n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let band = band_of(i, n);
        let nonzero = dec.next(nz_ctx(size_class, band), 2) == 1;
        if nonzero {
            let symbol = dec.next(mag_ctx(size_class, band), (2 * LEVEL_CLAMP + 1) as u32);
            out.push(unzigzag(symbol));
        } else {
            out.push(0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Round-trip identity through the real `mars_entropy` coder -- the strongest
    /// available oracle for a lossless entropy stage (verification-discipline).
    #[test]
    fn residual_events_round_trip_through_the_real_entropy_coder() {
        let mut rng = Rng(0x5EED_1122_3344_5566);
        for &size in &[2usize, 4, 8, 16] {
            let n = size * size;
            let size_class = (size as u32).trailing_zeros();
            let levels: Vec<i32> = (0..n)
                .map(|_| {
                    let r = (rng.next_u64() % 100) as i32;
                    if r < 60 {
                        0
                    } else {
                        (rng.next_u64() % 40) as i32 - 20
                    }
                })
                .collect();

            let mut events = Vec::new();
            encode_events(&levels, size_class, &mut events);
            let bytes = mars_entropy::encode(&events);
            let mut dec = Decoder::new(&bytes).unwrap();
            let decoded = decode_values(&mut dec, size_class, n);

            let expected: Vec<i32> = levels.iter().map(|&l| l.clamp(-LEVEL_CLAMP, LEVEL_CLAMP)).collect();
            assert_eq!(decoded, expected, "size {size}");
        }
    }

    #[test]
    fn all_zero_block_round_trips_to_all_zero() {
        let size_class = 3;
        let levels = vec![0i32; 64];
        let mut events = Vec::new();
            encode_events(&levels, size_class, &mut events);
        let bytes = mars_entropy::encode(&events);
        let mut dec = Decoder::new(&bytes).unwrap();
        let decoded = decode_values(&mut dec, size_class, 64);
        assert_eq!(decoded, levels);
    }

    #[test]
    fn levels_beyond_the_clamp_round_trip_to_the_clamp() {
        let size_class = 2;
        let levels = vec![500, -500, 0, 63, -63, 64, -64];
        let mut events = Vec::new();
            encode_events(&levels, size_class, &mut events);
        let bytes = mars_entropy::encode(&events);
        let mut dec = Decoder::new(&bytes).unwrap();
        let decoded = decode_values(&mut dec, size_class, levels.len());
        assert_eq!(decoded, vec![63, -63, 0, 63, -63, 64i32.min(LEVEL_CLAMP), (-64i32).max(-LEVEL_CLAMP)]);
    }
}
