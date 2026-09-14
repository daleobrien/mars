//! Ad hoc Step 9 measurement: run every method against one corpus image at the exact
//! `EncodeParams` a `results/baseline-mars1.jsonl` row was recorded at, and print
//! evals/transform for comparison against that row's `comparisons`/`transforms`.
//!
//! Not a `marsbench` subcommand (that integration -- CLI wiring, recall scoring against
//! the oracle, a real results/ row -- is still open, see docs/decisions.md) -- this is
//! the fastest path in this session to a real number, run via:
//! `cargo run --release -p mars-search --example evals_check -- <path/to/image.raw> <width> <height>`

use mars_codec::encode::EncodeParams;
use mars_core::io::read_raw;
use mars_search::{MethodName, SizedRetrievers};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let width: usize = args[2].parse().unwrap();
    let height: usize = args[3].parse().unwrap();

    let image = read_raw(std::path::Path::new(path), width, height).expect("read raw image");
    let params = EncodeParams {
        min_size: 4,
        max_size: 16,
        shift: 4,
        bits_alfa: 4,
        bits_beta: 7,
        max_alfa: 1.0,
        t_rms: 8.0,
        zero_threshold: 0,
        lambda: None,
    };
    let contracted = mars_codec::encode::build_contracted(&image);

    println!(
        "{:<14} {:>10} {:>10} {:>12}",
        "method", "evals", "transforms", "evals/xform"
    );
    for method in MethodName::ALL {
        let retrievers = SizedRetrievers::build(
            &contracted,
            image.width() as u32,
            image.height() as u32,
            params.shift,
            params.min_size,
            params.max_size,
            || method.new_retriever(),
        );
        let (_, leaves, evals, _picks) = mars_search::encode_image(&image, &params, &retrievers);
        // `transforms` (§M5): every leaf block emitted, matching `coding_func.c`'s
        // `transforms++` site -- reached even for a size-1 leaf when the forced
        // subdivision path lands on one, so this counts *every* leaf, not just size>1.
        let transforms = leaves.len();
        println!(
            "{:<14} {:>10} {:>10} {:>12.3}",
            method.key(),
            evals,
            transforms,
            evals as f64 / transforms as f64
        );
    }
}
