#!/usr/bin/env bash
# Build the unmodified 1998 Mars 1 C reference with PINNED flags (Step 0).
#
# The flags are not incidental. Mars 1's search is driven by floating-point RMS
# comparisons, so any optimisation that reassociates or contracts FP arithmetic changes
# which domain block wins a tie, which changes the bitstream. `-ffast-math` would make
# the baseline irreproducible across compiler versions.
#
# Target aarch64 (or x86-64) but NEVER 32-bit x86: x87's 80-bit excess precision makes
# the same source produce different comparisons depending on register allocation. On
# aarch64 this hazard class does not exist at all (§M4).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/reference/mars1"
OUT="${MARS1_OUT:-$ROOT/target/mars1}"

CC="${CC:-cc}"
CFLAGS="-O2 -fno-fast-math -fno-unsafe-math-optimizations -fno-associative-math -ffp-contract=off -std=gnu89 -Wno-implicit-function-declaration -Wno-implicit-int -Wno-return-type"
LDLIBS="-lm"

arch="$(uname -m)"
case "$arch" in
  arm64|aarch64|x86_64) ;;
  i386|i686)
    echo "refusing to build on 32-bit x86: x87 excess precision makes the baseline irreproducible" >&2
    exit 1 ;;
  *) echo "warning: untested architecture $arch" >&2 ;;
esac

mkdir -p "$OUT"

ENC_SRCS=(mars_enc.c image_io.c index_func.c coding_func.c miscell.c split_func.c nn_search.c)
DEC_SRCS=(mars_dec.c miscell.c image_io.c)

compile_one() {
  local out="$1"; shift
  local objs=()
  for s in "$@"; do
    local o="$OUT/${s%.c}.o"
    # shellcheck disable=SC2086
    (cd "$SRC" && $CC $CFLAGS -c "$s" -o "$o")
    objs+=("$o")
  done
  # shellcheck disable=SC2086
  $CC -o "$OUT/$out" "${objs[@]}" $LDLIBS
}

compile_one encmars "${ENC_SRCS[@]}"
rm -f "$OUT"/*.o
compile_one decmars "${DEC_SRCS[@]}"
rm -f "$OUT"/*.o

# Record exactly what produced these binaries, so Step 2's baseline rows can cite it.
{
  echo "built:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "arch:     $arch"
  echo "cc:       $CC"
  echo "cc_version: $($CC --version | head -1)"
  echo "cflags:   $CFLAGS"
  echo "ldlibs:   $LDLIBS"
  echo "src_sha256: $(cd "$SRC" && cat ./*.c ./*.h | shasum -a 256 | cut -d' ' -f1)"
} > "$OUT/build-info.txt"

echo "built $OUT/encmars and $OUT/decmars"
cat "$OUT/build-info.txt"
