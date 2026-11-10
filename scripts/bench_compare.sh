#!/usr/bin/env bash
# Reproduce the README before/after numbers.
#
#   scripts/bench_compare.sh [BASE_REF]      (default: the pre-optimisation commit)
#
# Benchmarks BASE_REF in a temporary git worktree, saves it as Criterion
# baseline "before", then benchmarks the current checkout against it.
set -euo pipefail
BASE_REF="${1:-perf-baseline}"
ROOT="$(git rev-parse --show-toplevel)"
WT="$(mktemp -d)"
trap 'git -C "$ROOT" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT

git -C "$ROOT" worktree add --detach "$WT" "$BASE_REF" >/dev/null
# Separate build dirs (sharing one lets Cargo reuse the wrong binary), but one
# Criterion results dir so the "after" run can compare against "before".
export CRITERION_HOME="$ROOT/target/criterion"
# Use the current benchmark harness so both sides measure the same workload.
cp "$ROOT/benches/tracking_loop.rs" "$WT/benches/tracking_loop.rs"
cp "$ROOT/src/scenario.rs" "$WT/src/scenario.rs"

echo "== before: $BASE_REF =="
(cd "$WT" && CARGO_TARGET_DIR="$ROOT/target/bench-base" cargo bench --bench tracking_loop -- --save-baseline before --noplot)
echo "== after: $(git -C "$ROOT" rev-parse --short HEAD) =="
(cd "$ROOT" && CARGO_TARGET_DIR="$ROOT/target/bench-head" cargo bench --bench tracking_loop -- --baseline before --noplot)
