#!/usr/bin/env bash
# Fetch corpus images listed in a manifest and verify their hashes (§M9).
#
# Images are never committed (they are large and, for Kodak, not ours to redistribute);
# the manifest is. The manifest's hash goes into every result row's provenance block
# (§M7), so a changed image set can never silently change a number.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
MANIFEST="${1:-$ROOT/corpus/kodak.manifest.json}"
RETRIES="${FETCH_RETRIES:-5}"

command -v python3 >/dev/null || { echo "python3 required to read the manifest" >&2; exit 1; }

# name|url|sha256|dir, one per line, separated by US (0x1f) rather than a tab: bash
# collapses runs of IFS *whitespace*, so an absent sha256 would silently shift every
# later field along -- exactly the sort of quiet misparse that yields a plausible wrong
# answer (§2.1).
SEP=$'\x1f'
entries="$(python3 "$HERE/manifest-lines.py" "$MANIFEST")"

fail=0
while IFS="$SEP" read -r name url want dir; do
  [ -n "$name" ] || continue
  dest="$ROOT/$dir/$name"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ] && [ -n "$want" ]; then
    got="$(shasum -a 256 "$dest" | cut -d' ' -f1)"
    if [ "$got" = "$want" ]; then echo "ok       $name"; continue; fi
    echo "rehash   $name (had $got, want $want) -- refetching" >&2
    rm -f "$dest"
  fi
  if [ ! -s "$dest" ]; then
    for attempt in $(seq 1 "$RETRIES"); do
      if curl -fsS --retry 2 --retry-delay 2 --max-time 180 -o "$dest.part" "$url"; then
        mv "$dest.part" "$dest"
        break
      fi
      rm -f "$dest.part"
      echo "retry    $name (attempt $attempt/$RETRIES)" >&2
      sleep $((attempt * 2))
    done
  fi
  if [ ! -s "$dest" ]; then echo "FAILED   $name" >&2; fail=1; continue; fi
  if [ -n "$want" ]; then
    got="$(shasum -a 256 "$dest" | cut -d' ' -f1)"
    if [ "$got" != "$want" ]; then
      echo "MISMATCH $name: got $got, want $want" >&2
      fail=1
      continue
    fi
    echo "fetched  $name"
  else
    echo "fetched  $name (no sha256 pinned yet; run scripts/hash-corpus.py)"
  fi
done <<< "$entries"

exit "$fail"
