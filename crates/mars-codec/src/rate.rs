//! Rate estimation for Step 14's `J = D + λR` mode decision.
//!
//! **The approximation, stated up front (the brief's own warning: getting this subtly
//! wrong "would bias every decision toward the modes with cheap headers").** Pricing a
//! candidate leaf exactly would mean knowing the *live* per-context [`mars_entropy`]
//! model's state at the exact point in the final, RD-chosen traversal where that leaf
//! would be coded -- but that traversal order is exactly what the bottom-up search is
//! still deciding, so the true live state is not available yet (a chicken-and-egg
//! problem: the cost of a decision depends on decisions not yet made). Two designs were
//! considered:
//!
//! 1. **A mutable running model, updated in true decision order.** Rejected: `walk`'s
//!    bottom-up recursion decides a parent only *after* its children are fully resolved,
//!    but the real bitstream (`mars_format::write`) emits the parent's split flag
//!    *before* its children -- so "decision order" and "bitstream order" disagree, and a
//!    single mutable model can only match one of them. Worse, sharing one mutable model
//!    across the `rayon::join` parallel subtrees would make the estimate (and hence the
//!    RD decision, and hence the bitstream) depend on thread scheduling -- exactly the
//!    nondeterminism §2.3 forbids.
//! 2. **A frozen, read-only snapshot, taken once per image before the RD search runs.**
//!    Chosen. A cheap "warm-up" encode of the same image, using the *old* threshold-driven
//!    top-down partition at a fixed `t_rms` (unrelated to this run's `λ`), is run first;
//!    its real leaf events are replayed through [`mars_entropy::build_models`] to obtain
//!    one [`AdaptiveModel`] per `(field, size_class)` context, exactly as the real
//!    encoder would have converged them on that partition. This snapshot is then used
//!    **read-only** for every `bits_for` query the entire RD search makes -- trivially
//!    `Sync`, so parallel evaluation is bit-identical at any thread count (nothing is
//!    mutated after it is built).
//!
//! **What this costs in fidelity, honestly:** the snapshot reflects one representative
//! partition's context statistics, not this λ's own (possibly quite different) leaf-size
//! mix, and it is not updated as the RD search's own decisions accumulate -- there is no
//! iteration to a fixed point. A leaf whose true final position in the bitstream would
//! see a colder or warmer model than the snapshot's frozen state is priced slightly off.
//! This is a deliberate, bounded, and documented approximation (`docs/decisions.md`), not
//! an oversight -- and it is a world away from a constant-bits stand-in: two leaves with
//! different `(field, size_class, symbol)` triples still get genuinely different,
//! data-derived prices, which is what keeps the mode decision from being biased toward
//! "whichever mode has a cheap header regardless of what actually gets coded."
//!
//! **A second, narrower approximation, confined to two fields.** [`mars_format::emit_leaf`]
//! codes each domain-referencing leaf's row/col as a zigzag delta from the *previous*
//! leaf's domain position in final traversal order -- again, not yet known during
//! bottom-up search. Rate estimation for `FIELD_DOM_ROW`/`FIELD_DOM_COL` therefore uses a
//! *fresh* (zeroed) predictor per candidate (see [`mars_format::leaf_events`]), pricing
//! the coordinate's own zigzag value against the snapshot's delta-shaped histogram for
//! that context -- a known mismatch in distribution shape, but confined to two of the six
//! to eight fields a leaf emits, and it does not affect `FIELD_MODE`/`FIELD_QALFA`/
//! `FIELD_QBETA`/`FIELD_ISOMETRY`, which dominate a typical leaf's bit cost.

use std::collections::HashMap;

use mars_entropy::{AdaptiveModel, ContextKey};

use crate::ifs::{Header, Leaf};
use crate::mars_format::{events_for_leaves, MarsFormatError};

/// A frozen, read-only set of per-context entropy models -- Step 14's rate-estimation
/// snapshot. See the module doc for exactly what approximation this represents and why.
pub struct RateModels {
    models: HashMap<ContextKey, AdaptiveModel>,
}

impl RateModels {
    /// Build the snapshot from a representative partition's own leaves, by replaying the
    /// exact event stream `mars_format::write` would encode for them.
    pub fn from_leaves(hdr: &Header, leaves: &[Leaf]) -> Result<Self, MarsFormatError> {
        let events = events_for_leaves(hdr, leaves)?;
        Ok(Self {
            models: mars_entropy::build_models(&events),
        })
    }

    /// Estimated bits to code `symbol` in context `ctx` with alphabet size `alphabet`.
    /// Falls back to a fresh, uniform (Laplace) model when the snapshot never observed
    /// this context -- the same "no data yet" state a live model starts every image in,
    /// so an unseen context is priced at its honest information-theoretic default
    /// (`log2(alphabet)`-ish) rather than zero or some constant.
    pub fn bits_for(&self, ctx: ContextKey, alphabet: u32, symbol: u32) -> f64 {
        match self.models.get(&ctx) {
            Some(m) => m.bits_for(symbol),
            None => AdaptiveModel::new(alphabet.max(2)).bits_for(symbol),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ifs::Leaf;

    fn hdr() -> Header {
        Header {
            bits_alfa: 4,
            bits_beta: 7,
            min_size: 4,
            max_size: 16,
            shift: 4,
            width: 32,
            height: 32,
            int_max_alfa: 32,
        }
    }

    #[test]
    fn a_context_with_skewed_observations_is_cheaper_than_an_unseen_one() {
        // Build a snapshot where every leaf is DC-only (qalfa == 0), heavily skewed
        // toward one qbeta value -- FIELD_MODE's symbol 0 should end up far cheaper than
        // an unseen context's uniform default, which is the whole point of not using a
        // constant-bits estimate.
        let h = hdr();
        let mut leaves = Vec::new();
        for i in 0..16u32 {
            let row = (i / 4) * 8;
            let col = (i % 4) * 8;
            leaves.push(Leaf {
                row,
                col,
                size: 8,
                mode: 0,
                qalfa: 0,
                qbeta: 64,
                isometry: 0,
                dom_row: 0,
                dom_col: 0,
                qgx: 0,
                qgy: 0,
                residual: Vec::new(),
            });
        }
        let snapshot = RateModels::from_leaves(&h, &leaves).unwrap();
        let size_class = 8u32.trailing_zeros();
        let mode0_bits = snapshot.bits_for((crate::mars_format::FIELD_MODE, size_class), 2, 0);
        let mode1_bits = snapshot.bits_for((crate::mars_format::FIELD_MODE, size_class), 2, 1);
        assert!(
            mode0_bits < 1.0 && mode0_bits < mode1_bits,
            "observed-heavy symbol should be cheap: mode0={mode0_bits} mode1={mode1_bits}"
        );

        // A context never observed at all still gets a sensible (uniform, not zero)
        // estimate rather than panicking or silently returning 0.
        let unseen = snapshot.bits_for((crate::mars_format::FIELD_ISOMETRY, size_class), 8, 3);
        assert!(
            (unseen - 3.0).abs() < 0.5,
            "unseen 8-way context ~3 bits, got {unseen}"
        );
    }
}
