//! Compare residual changes in two grayscale or color containers without re-encoding.
use anyhow::{Context, Result, ensure};
use mars_codec::ifs::Leaf;
use mars_codec::mars_format::{self, Header};
use std::collections::BTreeMap;

fn planes(path: &str) -> Result<Vec<(Header, Vec<Leaf>)>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() >= 7 && &bytes[..4] == b"MARC", "expected MARC container");
    let mut offset = 7usize;
    let mut result = Vec::new();
    for _ in 0..bytes[6] {
        let length = bytes.get(offset..offset + 4).context("truncated length")?;
        let length = u32::from_le_bytes(length.try_into()?) as usize;
        offset += 4;
        let end = offset.checked_add(length).context("length overflow")?;
        result.push(mars_format::read(bytes.get(offset..end).context("truncated plane")?)?);
        offset = end;
    }
    ensure!(offset == bytes.len(), "trailing bytes");
    Ok(result)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: compare_residual_streams fixed.mars adaptive.mars");
    let fixed = planes(&args[1])?;
    let adaptive = planes(&args[2])?;
    ensure!(fixed.len() == adaptive.len(), "different plane counts");
    for (plane, ((fh, fl), (ah, al))) in fixed.iter().zip(&adaptive).enumerate() {
        let by_pos: BTreeMap<_, _> = fl.iter().map(|l| ((l.row, l.col, l.size), l)).collect();
        let mut changed = 0;
        let mut modes = 0;
        let mut geometry = 0;
        let mut residuals = 0;
        let mut swapped = al.clone();
        for (index, leaf) in al.iter().enumerate() {
            let Some(previous) = by_pos.get(&(leaf.row, leaf.col, leaf.size)) else {
                geometry += 1;
                continue;
            };
            if *previous != leaf {
                changed += 1;
                modes += usize::from(previous.mode != leaf.mode);
                residuals += usize::from(previous.residual != leaf.residual);
                if previous.mode == 3 && leaf.mode == 3 {
                    // Bit-cost ablation only: these coefficients belong to a different qstep.
                    swapped[index].residual.clone_from(&previous.residual);
                }
                println!("plane={plane} block=({},{},{}) mode={}->{} residual_nonzero={}->{}",
                    leaf.row, leaf.col, leaf.size, previous.mode, leaf.mode,
                    previous.residual.iter().filter(|&&v| v != 0).count(),
                    leaf.residual.iter().filter(|&&v| v != 0).count());
            }
        }
        println!("plane={plane} qstep={}->{} leaves={}->{} changed_same_geometry={changed} mode_changes={modes} new_geometry={geometry} residual_changes={residuals} bytes={}->{} adaptive_with_fixed_residual_levels_bytes={} (bit-cost ablation, not a quality comparison)",
            fh.residual_qstep.get(), ah.residual_qstep.get(), fl.len(), al.len(),
            mars_format::write(fh, fl)?.len(), mars_format::write(ah, al)?.len(),
            mars_format::write(ah, &swapped)?.len());
    }
    Ok(())
}
