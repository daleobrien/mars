# Licensing

**Decision (2026-09-13, Step 0): Mars 2 is GPL-2.0-or-later.**

## The catch, stated deliberately rather than discovered later

Mars 1 is GPL-2.0-or-later (`reference/mars1/COPYING.txt`), and Mars 2 keeps that for
continuity. The interaction worth settling before dependencies accumulate (§9 of the
implementation plan):

**Apache-2.0 code cannot be combined into a GPL-2.0-*only* work.** Apache-2.0's
patent-termination and indemnification clauses are additional restrictions under GPLv2. It
*is* compatible with GPL-3.0.

Consequences, accepted knowingly:

- Because the licence is "**or later**", a recipient may choose v3, so combining
  Apache-2.0 dependencies is lawful — but **the effective licence of the distributed
  binary becomes GPL-3.0**.
- Any Apache-2.0-**only** dependency (no MIT option) forces that outcome with no choice
  left to the recipient.
- Most of the Rust ecosystem is dual-licensed MIT OR Apache-2.0, and MIT alone is
  GPLv2-compatible, so in practice a recipient who wants v2 can usually have it.

**Policy, enforced rather than stated:** [`deny.toml`](../deny.toml) **omits Apache-2.0
from the allow-list entirely.** Almost the whole Rust ecosystem is dual-licensed
`MIT OR Apache-2.0`, so a crate that offers an MIT option passes on MIT and the recipient's
GPLv2 option survives. A crate offering *only* Apache-2.0 fails `cargo deny check licenses`
and therefore fails CI.

This turns "prefer crates with an MIT option" from a preference nobody re-checks into a
build failure. Adding Apache-2.0 to the list means accepting GPL-3.0 as the effective
licence of every distributed binary; that is an entry in
[docs/decisions.md](decisions.md), not a one-line edit to a config file.

The allow-list is also kept minimal — every entry corresponds to a licence actually present
in the dependency graph — so that an unused allowance can never quietly pre-authorise
something nobody examined.

## Current dependency licences

Every runtime dependency as of Step 1 offers an MIT option:

Every crate in the graph at Step 1 resolves to a GPLv2-compatible licence:

| Licence chosen | Crates |
|---|---|
| MIT | the large majority, including `anyhow`, `clap`, `serde`, `serde_json`, `sha2`, `thiserror`, `image`, `png`, `flate2`, `memchr`, `byteorder-lite`, `tiff`, `fax` |
| MIT (or Apache-2.0 or Zlib) | `zune-jpeg`, `zune-core` — `image`'s JPEG codec; MIT chosen |
| MIT (or Apache-2.0) | `image-webp` (`image`'s WebP codec) and its `weezl`, `half`, `quick-error` dependencies; MIT chosen |
| BSD-3-Clause | `moxcms`, `pxfm` (Apache-2.0 OR BSD-3-Clause — pulled in by `image`) |
| Unicode-3.0 | `unicode-ident`, whose expression is `(MIT OR Apache-2.0) AND Unicode-3.0` |
| Zlib | `bytemuck`, `miniz_oxide` (also available under MIT) |

Run `cargo deny check` — or `just deny` — to re-verify. The check is in CI, so a dependency
that changes its terms fails the build rather than an audit six months later.

## Clean-room status

Mars 2 is a **clean-room Rust implementation**. `reference/mars1/` is an unmodified
historical reference used for measurement and format documentation, not something to
mechanically translate. That keeps the copyright story simple regardless of the licence
choice above: the derived work is the *format* documentation and the measured baselines,
and the Rust code is written from the specification in `docs/mars1-format.md` (Step 3).

## The cross-validation references are not dependencies

`scripts/crossval/requirements.txt` pins `numpy` (BSD-3-Clause), `scipy` (BSD-3-Clause),
`scikit-image` (BSD-3-Clause), `Pillow` (MIT-CMU) and `sewar` (MIT). These are test-time
reference implementations invoked as subprocesses; no code from them is linked into or
distributed with Mars 2, so they raise no combination question.
