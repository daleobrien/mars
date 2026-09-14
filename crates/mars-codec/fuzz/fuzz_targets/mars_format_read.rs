#![no_main]

use libfuzzer_sys::fuzz_target;
use mars_codec::mars_format::read;

// Step 10's exit criterion: `mars_format::read` must not panic, OOM, or allocate
// unboundedly on arbitrary input. `read` returning `Ok` is not itself a bug -- some
// random byte strings are structurally valid headers with an empty or trivial tree --
// but if it ever does, decode the resulting leaves too, since a `Leaf` list that
// round-tripped from garbage input is exactly the kind of value a downstream consumer
// (e.g. `ifs::decode_iterative`) would otherwise be handed unchecked.
fuzz_target!(|data: &[u8]| {
    if let Ok((hdr, leaves)) = read(data) {
        let iterations = 1; // one pass is enough to exercise indexing; speed matters here
        let _ = mars_codec::ifs::decode_iterative(&hdr, &leaves, iterations);
    }
});
