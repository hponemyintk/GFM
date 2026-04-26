#!/usr/bin/env bash
# Sanity check: the bounded job-pool pattern in scripts/pretrain_p4d.sh
# (a) doesn't deadlock, (b) respects the concurrency limit, (c) processes
# every input.
#
# We can't run the real builds in CI (they need GPUs and many GB of
# RelBench data), so this test extracts the pool pattern with stub jobs
# that just sleep + log. Run with:
#
#     bash tests/test_parallel_pool.sh
#
# Asserts via exit code; emits diagnostic to stderr.

set -uo pipefail
shopt -s nullglob  # unmatched globs expand to nothing instead of errors

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

PARALLEL_BUILDS=4
N_JOBS=10
MAX_OBSERVED_FILE="$LOG_DIR/max_observed"
echo 0 > "$MAX_OBSERVED_FILE"

_count_glob() {
    # Print number of files matching the given pattern. With nullglob,
    # an unmatched pattern expands to nothing (count = 0).
    local files=("$@")
    echo "${#files[@]}"
}

stub_job() {
    local id="$1"
    local started="$LOG_DIR/started_$id"
    local finished="$LOG_DIR/finished_$id"
    touch "$started"
    local active
    active=$(_count_glob "$LOG_DIR"/started_*)
    local done_count
    done_count=$(_count_glob "$LOG_DIR"/finished_*)
    local concurrent=$(( active - done_count ))
    local prev
    prev=$(cat "$MAX_OBSERVED_FILE" 2>/dev/null || true)
    prev=${prev:-0}
    if [ "$concurrent" -gt "$prev" ]; then
        echo "$concurrent" > "$MAX_OBSERVED_FILE"
    fi
    sleep 0.2  # let other jobs land in the same window
    touch "$finished"
}

prune_pids() {
    local -n arr=$1
    local kept=()
    local p
    for p in "${arr[@]}"; do
        if kill -0 "$p" 2>/dev/null; then
            kept+=("$p")
        fi
    done
    arr=("${kept[@]}")
}

declare -a PIDS=()
for i in $(seq 1 "$N_JOBS"); do
    while [ ${#PIDS[@]} -ge "$PARALLEL_BUILDS" ]; do
        wait -n 2>/dev/null || true
        prune_pids PIDS
    done
    stub_job "$i" &
    PIDS+=("$!")
done
wait

# Assert: every job finished
N_FINISHED=$(_count_glob "$LOG_DIR"/finished_*)
if [ "$N_FINISHED" -ne "$N_JOBS" ]; then
    echo "FAIL: expected $N_JOBS finished jobs, got $N_FINISHED" >&2
    exit 1
fi

# Assert: peak concurrency never exceeded the limit
MAX_OBSERVED=$(cat "$MAX_OBSERVED_FILE")
if [ "$MAX_OBSERVED" -gt "$PARALLEL_BUILDS" ]; then
    echo "FAIL: peak concurrency $MAX_OBSERVED exceeded limit $PARALLEL_BUILDS" >&2
    exit 1
fi

# Assert: peak concurrency was actually used (else parallelism is broken)
if [ "$MAX_OBSERVED" -lt 2 ]; then
    echo "FAIL: peak concurrency $MAX_OBSERVED < 2 -- pool didn't run jobs in parallel" >&2
    exit 1
fi

echo "PASS: $N_JOBS jobs ran, peak concurrency $MAX_OBSERVED (limit $PARALLEL_BUILDS)"
