//! `marsbench report`: Markdown and HTML tables plus RD curves.
//!
//! The plots are hand-written inline SVG. That is not an aesthetic choice — it keeps the
//! figure regeneration path dependency-free and diffable, which matters because with no
//! peer review in the plan (§M9) one-command figure regeneration is the only remaining
//! external check on the numbers.

use std::fmt::Write as _;

use crate::bdrate::{bd_metrics, BdResult, RdCurve};

/// Everything a report needs.
pub struct Report {
    pub title: String,
    pub curves: Vec<RdCurve>,
    /// Curve label used as the BD-rate reference.
    pub reference: Option<String>,
    /// Free-text provenance line reproduced verbatim at the top of the report.
    pub provenance_note: String,
}

impl Report {
    pub fn new(title: impl Into<String>, curves: Vec<RdCurve>) -> Self {
        Self {
            title: title.into(),
            curves,
            reference: None,
            provenance_note: String::new(),
        }
    }

    pub fn with_reference(mut self, label: impl Into<String>) -> Self {
        self.reference = Some(label.into());
        self
    }

    pub fn with_provenance_note(mut self, note: impl Into<String>) -> Self {
        self.provenance_note = note.into();
        self
    }

    /// BD metrics of every other curve against the reference. Curves that cannot be
    /// compared are reported as errors rather than omitted — a silently missing row is
    /// how an inconvenient comparison disappears.
    pub fn bd_table(&self) -> Vec<Result<BdResult, String>> {
        let Some(reference) = self
            .reference
            .as_ref()
            .and_then(|r| self.curves.iter().find(|c| &c.label == r))
        else {
            return Vec::new();
        };
        self.curves
            .iter()
            .filter(|c| c.label != reference.label)
            .map(|c| bd_metrics(reference, c).map_err(|e| e.to_string()))
            .collect()
    }

    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "# {}\n", self.title);
        if !self.provenance_note.is_empty() {
            let _ = writeln!(s, "> {}\n", self.provenance_note);
        }

        let _ = writeln!(s, "## Operating points\n");
        let _ = writeln!(s, "| curve | bpp | PSNR (dB) |");
        let _ = writeln!(s, "|---|---:|---:|");
        for c in &self.curves {
            let mut pts = c.points.clone();
            pts.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).unwrap());
            for p in pts {
                let _ = writeln!(s, "| {} | {:.4} | {:.3} |", c.label, p.bpp, p.psnr);
            }
        }

        let bd = self.bd_table();
        if !bd.is_empty() {
            let reference = self.reference.clone().unwrap_or_default();
            let _ = writeln!(s, "\n## BD-rate vs `{reference}`\n");
            let _ = writeln!(
                s,
                "BD-rate is negative when the test curve needs fewer bits. \
                 §M3: the interval is part of the number."
            );
            let _ = writeln!(
                s,
                "\n| curve | BD-rate % | BD-PSNR dB | PSNR interval (dB) | bpp interval | pts |"
            );
            let _ = writeln!(s, "|---|---:|---:|---|---|---:|");
            for r in &bd {
                match r {
                    Ok(r) => {
                        let _ = writeln!(
                            s,
                            "| {} | {:+.2} | {:+.3} | {:.2} – {:.2} | {:.4} – {:.4} | {} |",
                            r.test,
                            r.bd_rate_pct,
                            r.bd_psnr_db,
                            r.psnr_interval_db.0,
                            r.psnr_interval_db.1,
                            r.bpp_interval.0,
                            r.bpp_interval.1,
                            r.points_test
                        );
                    }
                    Err(e) => {
                        let _ = writeln!(s, "| — | n/a | n/a | n/a | n/a | **{e}** |");
                    }
                }
            }
        }
        s
    }

    pub fn to_html(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "<!doctype html><meta charset=\"utf-8\"><title>{}</title>",
            escape(&self.title)
        );
        let _ = writeln!(s, "<style>{CSS}</style>");
        let _ = writeln!(s, "<h1>{}</h1>", escape(&self.title));
        if !self.provenance_note.is_empty() {
            let _ = writeln!(s, "<p class=prov>{}</p>", escape(&self.provenance_note));
        }
        let _ = writeln!(s, "{}", self.rd_plot_svg(720.0, 440.0));

        let _ = writeln!(s, "<h2>Operating points</h2><table><thead><tr><th>curve<th>bpp<th>PSNR (dB)</thead><tbody>");
        for c in &self.curves {
            let mut pts = c.points.clone();
            pts.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).unwrap());
            for p in pts {
                let _ = writeln!(
                    s,
                    "<tr><td>{}<td class=n>{:.4}<td class=n>{:.3}",
                    escape(&c.label),
                    p.bpp,
                    p.psnr
                );
            }
        }
        let _ = writeln!(s, "</tbody></table>");

        let bd = self.bd_table();
        if !bd.is_empty() {
            let _ = writeln!(
                s,
                "<h2>BD-rate vs <code>{}</code></h2>",
                escape(self.reference.as_deref().unwrap_or(""))
            );
            let _ = writeln!(s, "<table><thead><tr><th>curve<th>BD-rate %<th>BD-PSNR dB<th>PSNR interval (dB)<th>bpp interval</thead><tbody>");
            for r in &bd {
                match r {
                    Ok(r) => {
                        let _ = writeln!(
                            s,
                            "<tr><td>{}<td class=n>{:+.2}<td class=n>{:+.3}<td>{:.2} – {:.2}<td>{:.4} – {:.4}",
                            escape(&r.test),
                            r.bd_rate_pct,
                            r.bd_psnr_db,
                            r.psnr_interval_db.0,
                            r.psnr_interval_db.1,
                            r.bpp_interval.0,
                            r.bpp_interval.1
                        );
                    }
                    Err(e) => {
                        let _ = writeln!(s, "<tr><td colspan=5 class=err>{}", escape(e));
                    }
                }
            }
            let _ = writeln!(s, "</tbody></table>");
        }
        s
    }

    /// RD curves on a log-bpp x axis — the axis BD-rate is defined on, so the picture
    /// and the number agree.
    pub fn rd_plot_svg(&self, w: f64, h: f64) -> String {
        let pts: Vec<(f64, f64)> = self
            .curves
            .iter()
            .flat_map(|c| c.points.iter().map(|p| (p.bpp.log10(), p.psnr)))
            .collect();
        if pts.len() < 2 {
            return String::new();
        }
        let (pad_l, pad_r, pad_t, pad_b) = (64.0, 16.0, 16.0, 48.0);
        let (x0, x1) = minmax(pts.iter().map(|p| p.0));
        let (y0, y1) = minmax(pts.iter().map(|p| p.1));
        let (x0, x1) = pad_range(x0, x1);
        let (y0, y1) = pad_range(y0, y1);
        let sx = |x: f64| pad_l + (x - x0) / (x1 - x0) * (w - pad_l - pad_r);
        let sy = |y: f64| h - pad_b - (y - y0) / (y1 - y0) * (h - pad_t - pad_b);

        let mut s = String::new();
        let _ = write!(
            s,
            "<svg class=rd viewBox=\"0 0 {w} {h}\" width=\"100%\" role=img aria-label=\"rate-distortion curves\">"
        );
        // Grid and axes.
        for i in 0..=4 {
            let y = y0 + (y1 - y0) * i as f64 / 4.0;
            let _ = write!(
                s,
                "<line class=grid x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"/>\
                 <text class=tick x=\"{:.1}\" y=\"{:.1}\" text-anchor=end>{:.1}</text>",
                pad_l,
                sy(y),
                w - pad_r,
                sy(y),
                pad_l - 8.0,
                sy(y) + 4.0,
                y
            );
        }
        for i in 0..=4 {
            let x = x0 + (x1 - x0) * i as f64 / 4.0;
            let _ = write!(
                s,
                "<line class=grid x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"/>\
                 <text class=tick x=\"{:.1}\" y=\"{:.1}\" text-anchor=middle>{:.3}</text>",
                sx(x),
                pad_t,
                sx(x),
                h - pad_b,
                sx(x),
                h - pad_b + 18.0,
                10f64.powf(x)
            );
        }
        let _ = write!(
            s,
            "<text class=axis x=\"{:.1}\" y=\"{:.1}\" text-anchor=middle>bpp (log scale)</text>\
             <text class=axis transform=\"translate(14,{:.1}) rotate(-90)\" text-anchor=middle>PSNR-Y (dB)</text>",
            (pad_l + w - pad_r) / 2.0,
            h - 8.0,
            (pad_t + h - pad_b) / 2.0
        );

        for (i, c) in self.curves.iter().enumerate() {
            let colour = PALETTE[i % PALETTE.len()];
            let mut pts = c.points.clone();
            pts.sort_by(|a, b| a.bpp.partial_cmp(&b.bpp).unwrap());
            let path: Vec<String> = pts
                .iter()
                .map(|p| format!("{:.2},{:.2}", sx(p.bpp.log10()), sy(p.psnr)))
                .collect();
            let _ = write!(
                s,
                "<polyline fill=none stroke=\"{colour}\" stroke-width=2 points=\"{}\"/>",
                path.join(" ")
            );
            for p in &pts {
                let _ = write!(
                    s,
                    "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=3 fill=\"{colour}\"/>",
                    sx(p.bpp.log10()),
                    sy(p.psnr)
                );
            }
            let ly = pad_t + 16.0 + i as f64 * 18.0;
            let _ = write!(
                s,
                "<rect x=\"{:.1}\" y=\"{:.1}\" width=10 height=10 fill=\"{colour}\"/>\
                 <text class=legend x=\"{:.1}\" y=\"{:.1}\">{}</text>",
                w - pad_r - 160.0,
                ly - 9.0,
                w - pad_r - 144.0,
                ly,
                escape(&c.label)
            );
        }
        let _ = write!(s, "</svg>");
        s
    }
}

const PALETTE: [&str; 7] = [
    "#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#a463f2", "#97bbf5",
];

const CSS: &str = "body{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:2rem auto;max-width:56rem;padding:0 1rem}\
table{border-collapse:collapse;margin:1rem 0;font-variant-numeric:tabular-nums}\
th,td{border-bottom:1px solid #0002;padding:.3rem .6rem;text-align:left}\
td.n{text-align:right}.err{color:#b00}.prov{color:#555;font-size:.9em}\
svg.rd{max-width:100%;height:auto;border:1px solid #0002;border-radius:4px;background:#fff}\
.grid{stroke:#0001}.tick,.legend,.axis{font:11px ui-sans-serif,system-ui,sans-serif;fill:#444}";

fn minmax(it: impl Iterator<Item = f64>) -> (f64, f64) {
    it.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    })
}

fn pad_range(lo: f64, hi: f64) -> (f64, f64) {
    if hi > lo {
        let m = (hi - lo) * 0.06;
        (lo - m, hi + m)
    } else {
        (lo - 1.0, hi + 1.0)
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bdrate::RdPoint;

    fn curve(label: &str, shift: f64) -> RdCurve {
        RdCurve::new(
            label,
            [0.125, 0.25, 0.5, 1.0]
                .iter()
                .enumerate()
                .map(|(i, &bpp)| RdPoint {
                    bpp,
                    psnr: 26.0 + 3.3 * i as f64 + shift,
                })
                .collect(),
        )
    }

    #[test]
    fn markdown_contains_the_interval_with_the_number() {
        let r = Report::new("t", vec![curve("a", 0.0), curve("b", 1.0)]).with_reference("a");
        let md = r.to_markdown();
        assert!(md.contains("BD-rate"), "{md}");
        assert!(md.contains("PSNR interval"), "{md}");
        // The reference curve is not compared against itself.
        assert_eq!(md.matches("| b |").count(), 5); // 4 operating points + 1 BD row
    }

    #[test]
    fn html_and_svg_are_well_formed_enough_to_render() {
        let r = Report::new("t", vec![curve("a", 0.0), curve("b", 1.0)]).with_reference("a");
        let html = r.to_html();
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("<svg"));
        assert_eq!(html.matches("<svg").count(), html.matches("</svg>").count());
        assert!(!html.contains("NaN"), "plot scaling produced NaN");
    }

    #[test]
    fn an_uncomparable_curve_is_reported_not_dropped() {
        let bad = RdCurve::new(
            "short",
            vec![
                RdPoint {
                    bpp: 0.1,
                    psnr: 20.0,
                },
                RdPoint {
                    bpp: 0.2,
                    psnr: 22.0,
                },
            ],
        );
        let r = Report::new("t", vec![curve("a", 0.0), bad]).with_reference("a");
        let table = r.bd_table();
        assert_eq!(table.len(), 1);
        assert!(table[0].is_err());
        assert!(r.to_markdown().contains("§M3 requires"));
    }
}
