#!/bin/bash
# Full test suite with one pytest process per test file: XLA's compiled OU-JAX steps are never returned to the
# OS within a process, so a single process running every file grows past WSL's memory. Prints each file's result
# and peak RSS, then the totals; exits non-zero if any file fails.
# Each file's pytest is killed if its RSS passes MEM_LIMIT_MB (default 7000), so one file can't take WSL down;
# progress is appended to LOG (default logs/tests_per_file.log) as each file finishes.
#   scripts/run_tests_per_file.sh [pytest args...]
cd "$(dirname "$0")/.." || exit 1
files=$(git ls-files '*test_*.py' | grep -v '^tools/')
LIMIT=${MEM_LIMIT_MB:-7000}
# Persistent XLA compile cache: files that compile the same step reuse it (a file drops from minutes to seconds)
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/oujax_xla}
LOG=${LOG:-logs/tests_per_file.log}
mkdir -p "$(dirname "$LOG")"; echo "=== $(date -Is) $(git rev-parse --short HEAD)" >> "$LOG"
pass=0; fail=0; status=0
for f in $files; do
  tmp=$(mktemp)
  /usr/bin/time -f "%M" pytest -q -p no:cacheprovider "$@" "$f" > "$tmp" 2>&1 &
  tpid=$!
  while kill -0 $tpid 2>/dev/null; do
    rss=$(ps -o rss= --ppid $tpid 2>/dev/null | sort -n | tail -1)  # pytest itself (time's child)
    if [ -n "$rss" ] && [ $((rss / 1024)) -gt "$LIMIT" ]; then
      echo "MEMORY LIMIT: $f passed ${LIMIT} MB, killed" >> "$tmp"; pkill -P $tpid; break
    fi
    sleep 1
  done
  wait $tpid; rc=$?
  out=$(cat "$tmp"); rm -f "$tmp"
  peak_kb=$(echo "$out" | tail -1)
  line=$(echo "$out" | grep -E "passed|failed|error|no tests ran" | tail -1)
  p=$(echo "$line" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+"); p=${p:-0}
  n=$(echo "$line" | grep -oE "[0-9]+ (failed|error)" | grep -oE "^[0-9]+" | paste -sd+ | bc); n=${n:-0}
  pass=$((pass + p)); fail=$((fail + n))
  [ $rc -ne 0 ] && [ $rc -ne 5 ] && { status=1; echo "$out" | grep -E "^(FAILED|ERROR)"; }
  msg=$(printf "%-60s %s  (peak %s MB)" "$f" "$line" $(( ${peak_kb//[!0-9]/} / 1024 )))
  echo "$msg" | tee -a "$LOG"
done
echo "TOTAL: $pass passed, $fail failed" | tee -a "$LOG"
exit $status
