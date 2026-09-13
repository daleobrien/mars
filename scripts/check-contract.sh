#!/usr/bin/env bash
# §A2 — "Tests are the contract; agents may not weaken them."
#
# The single most important guard in the plan. The natural failure mode of an agent that
# cannot make code pass is to adjust the assertion, and the natural failure mode of one
# whose cross-validation disagrees is to widen the tolerance. Both are invisible in a
# green CI run, so they are made visible here instead.
#
# This check fails a commit range that does either of the following without an explaining
# trailer in a commit message:
#
#   deletes or loosens an assertion      -> needs  CONTRACT-CHANGE: <why>
#   changes a constant in tolerance.rs   -> needs  CONTRACT-CHANGE: <why>
#   changes a committed golden fixture   -> needs  FORMAT-CHANGE:   <why>   (§M8)
#
# It is deliberately a blunt instrument: it counts assertions and watches one file. A
# check that tried to understand whether a rewritten test is weaker would be a research
# project, and one nobody would trust.
set -euo pipefail

BASE="${1:-origin/main}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --verify --quiet "$BASE" >/dev/null; then
  echo "contract-check: base '$BASE' not found; nothing to compare against." >&2
  echo "contract-check: SKIP (this is expected on the first commit of a repository)." >&2
  exit 0
fi

MERGE_BASE="$(git merge-base "$BASE" HEAD)"
RANGE="$MERGE_BASE..HEAD"

if [ -z "$(git log --format=%H "$RANGE")" ]; then
  echo "contract-check: no commits in $RANGE; PASS"
  exit 0
fi

MESSAGES="$(git log --format='%B' "$RANGE")"
has_trailer() { grep -qE "^[[:space:]]*$1:" <<< "$MESSAGES"; }

fail=0
note() { echo "contract-check: $*" >&2; }

# --- 1. Assertion count, per test file ---------------------------------------
# Counting assertions per file catches the common case (an assert removed or commented
# out) without pretending to judge semantics.
assert_count() {
  # $1 = git revision or empty for worktree, $2 = path
  local content
  if [ -z "$1" ]; then content="$(cat "$2" 2>/dev/null || true)"
  else content="$(git show "$1:$2" 2>/dev/null || true)"; fi
  grep -cE '\bassert(_eq|_ne|_matches)?!|\.unwrap_err\(\)|should_panic' <<< "$content" || true
}

test_files="$(git diff --name-only "$RANGE" -- '*.rs' | grep -E '(^|/)tests/|_test\.rs$|tests\.rs$' || true)"
# Also consider files that merely contain a #[cfg(test)] module.
inline_test_files="$(git diff --name-only "$RANGE" -- '*.rs' | while read -r f; do
  [ -f "$f" ] || continue
  grep -q '#\[cfg(test)\]' "$f" && echo "$f"
done || true)"
all_test_files="$(printf '%s\n%s\n' "$test_files" "$inline_test_files" | sort -u | sed '/^$/d')"

while IFS= read -r f; do
  [ -n "$f" ] || continue
  before="$(assert_count "$MERGE_BASE" "$f")"
  after="$(assert_count "" "$f")"
  if [ "$after" -lt "$before" ]; then
    note "$f: assertions went $before -> $after"
    fail=1
  fi
done <<< "$all_test_files"

# A deleted test file is the most complete way to delete its assertions.
deleted_tests="$(git diff --name-only --diff-filter=D "$RANGE" -- '*.rs' | grep -E '(^|/)tests/' || true)"
if [ -n "$deleted_tests" ]; then
  note "test files deleted:"; echo "$deleted_tests" >&2
  fail=1
fi

# --- 1b. Contract scripts that are not Rust tests -----------------------------
# Step 3's exit criterion lives entirely in a Python file: `validate-ifs.py` is the
# independent implementation that decides whether docs/mars1-format.md is a spec. Its
# checks are as much "the contract" as any #[test], and the Rust-only scan above cannot
# see them, so it is counted separately.
CONTRACT_SCRIPTS="scripts/validate-ifs.py"
py_check_count() {
  local content
  if [ -z "$1" ]; then content="$(cat "$2" 2>/dev/null || true)"
  else content="$(git show "$1:$2" 2>/dev/null || true)"; fi
  grep -cE 'fails\.append\(|raise SpecViolation' <<< "$content" || true
}
for f in $CONTRACT_SCRIPTS; do
  git diff --quiet "$RANGE" -- "$f" && continue
  before="$(py_check_count "$MERGE_BASE" "$f")"
  after="$(py_check_count "" "$f")"
  # A brand-new file has nothing to have been weakened from.
  if [ "$before" -gt 0 ] && [ "$after" -lt "$before" ]; then
    note "$f: checks went $before -> $after"
    fail=1
  fi
done

# --- 2. The central tolerance module -----------------------------------------
TOLERANCE="crates/mars-core/src/tolerance.rs"
if git diff --quiet "$RANGE" -- "$TOLERANCE"; then :; else
  note "$TOLERANCE changed:"
  git diff "$RANGE" -- "$TOLERANCE" | grep -E '^[+-]pub const' >&2 || true
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  if has_trailer CONTRACT-CHANGE; then
    echo "contract-check: contract weakened, but a CONTRACT-CHANGE: trailer explains why. PASS"
    fail=0
  else
    note ""
    note "A contract was weakened with no explanation. Either restore it, or add a trailer:"
    note ""
    note "    CONTRACT-CHANGE: <why this assertion or tolerance had to change>"
    note ""
    note "§A7: widening a tolerance also requires a recorded reason in docs/decisions.md."
  fi
fi

# --- 3. Golden fixtures (§M8) -------------------------------------------------
fixture_changes="$(git diff --name-only "$RANGE" -- 'fixtures/' || true)"
if [ -n "$fixture_changes" ]; then
  if has_trailer FORMAT-CHANGE; then
    echo "contract-check: golden fixtures changed with a FORMAT-CHANGE: trailer. OK"
  else
    note "golden fixtures changed with no FORMAT-CHANGE: trailer:"
    echo "$fixture_changes" >&2
    fail=1
  fi
fi

if [ "$fail" -eq 0 ]; then
  echo "contract-check: PASS"
fi
exit "$fail"
