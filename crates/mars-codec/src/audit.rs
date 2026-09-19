//! P5a: the RD surrogate's estimated versus actual bits, by category.
//!
//! Step 14's `J = D + λR` search decides with a **frozen, read-only** rate snapshot taken
//! from a representative warm-up partition (`crate::rate`'s module doc documents exactly
//! which approximations that implies). The stream that ships is coded by the **live**,
//! evolving entropy models instead. Those are two different price lists for the same event
//! stream, and P5a's job is to quantify the difference instead of assuming it away:
//!
//! * `estimated_bits` -- what the frozen snapshot predicted for the final event stream,
//!   which is the objective the RD search actually minimised.
//! * `actual_bits` -- what each event really cost under the live model
//!   ([`mars_entropy::encode_with_bits`]), summed. This is the payload's own information
//!   content; the byte-aligned stream is slightly larger by the coder's final state and
//!   word padding, which [`RateAudit::overhead_bits`] isolates rather than burying.
//!
//! Bits are attributed to the plan's five categories by the event's field id, so a
//! compensation between categories cannot hide in the total.
//!
//! This is a measurement, not an optimisation: nothing here changes any encode. The
//! additive surrogate is known to be approximate (it prices domain coordinates against a
//! zeroed predictor, and never re-snapshots as decisions accumulate), and the honest exit
//! for P5a is stating how large the resulting error is, per category, on real images.

use crate::ifs::Leaf;
use crate::mars_format::{
    self, Header, MarsFormatError, FIELD_DOM_COL, FIELD_DOM_ROW, FIELD_GX, FIELD_GY,
    FIELD_ISOMETRY, FIELD_MODE, FIELD_QALFA, FIELD_QBETA, FIELD_QBETA_DC, FIELD_SPLIT,
};
use crate::rate::RateModels;
use crate::residual::{FIELD_RESID_MAG, FIELD_RESID_NZ};

/// The five cost categories P5a reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Quadtree split flags (`FIELD_SPLIT`).
    Partition,
    /// Per-leaf mode symbols (`FIELD_MODE`).
    Modes,
    /// Domain-position fields, the sequential-predictor-coded pair.
    Coordinates,
    /// Per-leaf quantised parameters: contrast, brightness, orientation, affine gradients.
    Coefficients,
    /// DCT residual coefficients (nonzero flags and signed magnitudes).
    Residuals,
}

impl Category {
    pub const ALL: [Category; 5] = [
        Category::Partition,
        Category::Modes,
        Category::Coordinates,
        Category::Coefficients,
        Category::Residuals,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Category::Partition => "partition",
            Category::Modes => "modes",
            Category::Coordinates => "coordinates",
            Category::Coefficients => "coefficients",
            Category::Residuals => "residuals",
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    /// Which category an event's field id belongs to. `None` for an id this build does not
    /// know, which [`RateAudit::unattributed_events`] records rather than dropping.
    pub fn of_field(field: u8) -> Option<Category> {
        match field {
            FIELD_SPLIT => Some(Category::Partition),
            FIELD_MODE => Some(Category::Modes),
            FIELD_DOM_ROW | FIELD_DOM_COL => Some(Category::Coordinates),
            FIELD_QALFA | FIELD_QBETA | FIELD_QBETA_DC | FIELD_ISOMETRY | FIELD_GX | FIELD_GY => {
                Some(Category::Coefficients)
            }
            FIELD_RESID_NZ | FIELD_RESID_MAG => Some(Category::Residuals),
            _ => None,
        }
    }
}

/// Estimated and actual bits for one category, plus how many events landed there.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CategoryBits {
    pub events: u64,
    pub estimated_bits: f64,
    pub actual_bits: f64,
}

/// One event stream priced twice: by the frozen surrogate the search decided with, and by
/// the live models the stream was actually coded with.
#[derive(Debug, Clone, PartialEq)]
pub struct RateAudit {
    categories: [CategoryBits; 5],
    /// Sum of the category estimates.
    pub estimated_bits: f64,
    /// Information content of the events under the live models.
    pub actual_event_bits: f64,
    /// Whole `.mars` plane stream as written, header and sections included.
    pub serialized_bytes: u64,
    pub serialized_bits: f64,
    /// `serialized_bits - actual_event_bits`: header, section table, byte alignment, the
    /// coder's final state. Never negative in a valid encode.
    pub overhead_bits: f64,
    /// Events whose field id this build did not recognise. Expected to be zero; a nonzero
    /// value means the category totals do not cover the whole stream.
    pub unattributed_events: u64,
    pub unattributed_bits: f64,
    pub events: u64,
    pub leaves: usize,
    pub estimated_provenance: EstimatedProvenance,
}

/// What produced the estimated side, so a reader cannot mistake a threshold partition's
/// reference price for an objective that partition was optimised against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatedProvenance {
    /// The frozen warm-up snapshot the RD search itself priced candidates with.
    RdWarmupSnapshot,
    /// The same snapshot, priced against a partition the RD search did not choose. A
    /// diagnostic reference, not an objective.
    ReferenceOnly,
}

impl RateAudit {
    pub fn category(&self, category: Category) -> CategoryBits {
        self.categories[category.index()]
    }

    pub fn categories(&self) -> &[CategoryBits; 5] {
        &self.categories
    }

    /// `estimated - actual` on the events themselves (excluding container overhead).
    /// Positive means the surrogate over-priced the stream, so the search was slightly
    /// biased toward larger leaves than a perfect price list would pick.
    pub fn estimate_error_bits(&self) -> f64 {
        self.estimated_bits - self.actual_event_bits
    }

    pub fn estimate_error_pct(&self) -> f64 {
        if self.actual_event_bits > 0.0 {
            self.estimate_error_bits() / self.actual_event_bits * 100.0
        } else {
            0.0
        }
    }
}

/// Price one final partition twice: against `rate` (the frozen snapshot) and against the
/// live models a real write uses.
///
/// `estimated_provenance` records which of the two situations this is -- see
/// [`EstimatedProvenance`].
pub fn audit_stream(
    hdr: &Header,
    leaves: &[Leaf],
    rate: &RateModels,
    estimated_provenance: EstimatedProvenance,
) -> Result<RateAudit, MarsFormatError> {
    let events = mars_format::events_for_leaves(&hdr.geometry, leaves)?;
    let coded = mars_entropy::encode_with_bits(&events);

    let mut categories = [CategoryBits::default(); 5];
    let mut estimated_bits = 0.0;
    let mut unattributed_events = 0u64;
    let mut unattributed_bits = 0.0;
    for (i, event) in events.iter().enumerate() {
        let estimated = rate.bits_for(event.ctx, event.alphabet, event.symbol);
        estimated_bits += estimated;
        match Category::of_field(event.ctx.0) {
            Some(category) => {
                let slot = &mut categories[category.index()];
                slot.events += 1;
                slot.estimated_bits += estimated;
                slot.actual_bits += coded.bits_per_event[i];
            }
            None => {
                unattributed_events += 1;
                unattributed_bits += estimated;
            }
        }
    }

    let serialized_bytes = mars_format::write(hdr, leaves)?.len() as u64;
    let serialized_bits = serialized_bytes as f64 * 8.0;
    Ok(RateAudit {
        categories,
        estimated_bits,
        actual_event_bits: coded.bits_total,
        serialized_bytes,
        serialized_bits,
        overhead_bits: serialized_bits - coded.bits_total,
        unattributed_events,
        unattributed_bits,
        events: events.len() as u64,
        leaves: leaves.len(),
        estimated_provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ifs::{Header as GeometryHeader, Leaf};
    use crate::quant::ResidualQstep;

    fn header() -> Header {
        Header {
            geometry: GeometryHeader {
                bits_alfa: 4,
                bits_beta: 7,
                min_size: 4,
                max_size: 8,
                shift: 4,
                width: 8,
                height: 8,
                int_max_alfa: 32,
            },
            residual_qstep: ResidualQstep::LEGACY,
        }
    }

    /// A single flat 8x8 leaf: the smallest valid tree, with one event per relevant field.
    fn flat_leaves() -> Vec<Leaf> {
        vec![Leaf {
            row: 0,
            col: 0,
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
        }]
    }

    #[test]
    fn every_field_id_this_build_writes_maps_to_a_category() {
        for field in 0u8..=11 {
            assert!(
                Category::of_field(field).is_some(),
                "field {field} would vanish from the category totals"
            );
        }
        assert_eq!(Category::of_field(FIELD_SPLIT), Some(Category::Partition));
        assert_eq!(Category::of_field(FIELD_MODE), Some(Category::Modes));
        assert_eq!(
            Category::of_field(FIELD_DOM_ROW),
            Some(Category::Coordinates)
        );
        assert_eq!(
            Category::of_field(FIELD_DOM_COL),
            Some(Category::Coordinates)
        );
        assert_eq!(
            Category::of_field(FIELD_QALFA),
            Some(Category::Coefficients)
        );
        assert_eq!(
            Category::of_field(FIELD_ISOMETRY),
            Some(Category::Coefficients)
        );
        assert_eq!(Category::of_field(FIELD_GX), Some(Category::Coefficients));
        assert_eq!(Category::of_field(FIELD_GY), Some(Category::Coefficients));
        assert_eq!(
            Category::of_field(FIELD_RESID_NZ),
            Some(Category::Residuals)
        );
        assert_eq!(
            Category::of_field(FIELD_RESID_MAG),
            Some(Category::Residuals)
        );
        // An unknown id is reported, never silently bucketed.
        assert_eq!(Category::of_field(200), None);
    }

    #[test]
    fn categories_are_labelled_and_indexed_consistently() {
        for (i, category) in Category::ALL.iter().enumerate() {
            assert_eq!(category.index(), i);
            assert!(!category.label().is_empty());
        }
        assert_eq!(Category::ALL.len(), 5);
    }

    #[test]
    fn a_real_leaf_stream_is_priced_twice_and_every_bit_is_accounted_for() {
        let hdr = header();
        let leaves = flat_leaves();
        // Priced by a snapshot built from the *same* partition, so this test isolates the
        // frozen-vs-live model difference rather than the warm-up mismatch as well.
        let rate = RateModels::from_leaves(&hdr.geometry, &leaves).unwrap();
        let audit =
            audit_stream(&hdr, &leaves, &rate, EstimatedProvenance::RdWarmupSnapshot).unwrap();

        assert_eq!(audit.leaves, 1);
        assert_eq!(audit.unattributed_events, 0);
        assert_eq!(
            audit.events,
            audit.categories().iter().map(|c| c.events).sum::<u64>(),
            "every event must be attributed"
        );
        let category_estimated: f64 = audit.categories().iter().map(|c| c.estimated_bits).sum();
        assert!(
            (category_estimated - audit.estimated_bits).abs() < 1e-9,
            "{category_estimated} vs {}",
            audit.estimated_bits
        );
        let category_actual: f64 = audit.categories().iter().map(|c| c.actual_bits).sum();
        assert!((category_actual - audit.actual_event_bits).abs() < 1e-9);

        // The byte-aligned stream is the payload plus a small, non-negative container
        // and coder slack -- never less than the information content it carries.
        assert!(audit.actual_event_bits >= 0.0);
        assert!(audit.serialized_bits >= audit.actual_event_bits);
        assert!(
            audit.overhead_bits < 8.0 * 64.0,
            "overhead {} bits is implausibly large",
            audit.overhead_bits
        );

        // A mode-0 leaf emits a mode symbol and a DC brightness symbol, and the 8x8 root
        // sits above `min_size`, so it also carries a "do not split" flag.
        assert_eq!(audit.category(Category::Partition).events, 1);
        assert_eq!(audit.category(Category::Modes).events, 1);
        assert_eq!(audit.category(Category::Coefficients).events, 1);
        assert_eq!(audit.category(Category::Coordinates).events, 0);
        assert_eq!(audit.category(Category::Residuals).events, 0);
        assert!(audit.category(Category::Modes).actual_bits > 0.0);
    }

    #[test]
    fn the_error_sign_is_the_surrogate_minus_the_live_cost() {
        let hdr = header();
        let leaves = flat_leaves();
        let rate = RateModels::from_leaves(&hdr.geometry, &leaves).unwrap();
        let audit = audit_stream(&hdr, &leaves, &rate, EstimatedProvenance::ReferenceOnly).unwrap();
        assert!(
            (audit.estimate_error_bits() - (audit.estimated_bits - audit.actual_event_bits)).abs()
                < 1e-12
        );
        assert_eq!(
            audit.estimated_provenance,
            EstimatedProvenance::ReferenceOnly
        );
    }
}
