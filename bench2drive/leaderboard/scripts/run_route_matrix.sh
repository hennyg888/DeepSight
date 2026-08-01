#!/bin/bash
# Run the route x lighting x obstacle matrix with the PDM-lite expert, in parallel.
#
#   routes:    121, 149, 190
#   lighting:  sun_glare, night
#   obstacle:  clear, car, cyclists, crowd
#   -> 24 runs, all with the 'expert' profile.
#
# Usage: bash leaderboard/scripts/run_route_matrix.sh [out_dir] [n_workers]
#   out_dir    where runs land (default results_matrix). NOT results_custom, so this
#              batch stays separate from one-off runs.
#   n_workers  parallel workers, one GPU + one CARLA server each (default 9).
#
# Each worker k owns GPU k, RPC port 2000+10k and TM port 2500+10k. The port spacing
# is 10 because a CARLA server also binds port+1/port+2 for its streaming channels;
# adjacent workers would collide at a spacing of 1. Servers are started on demand by
# run_custom_route.sh (start_carla.sh is idempotent per port) and are deliberately
# left running afterwards -- stop them with stop_carla.sh <port>.
#
# Routes are dealt round-robin rather than from a shared queue: every route here is
# ~134 m with end_on_encounter, so the runtimes are close enough that a static split
# costs less than the locking a dynamic queue would need.
#
# DRY_RUN=1 prints the plan and exits without touching a GPU.
set -euo pipefail

BENCH2DRIVE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH2DRIVE"

OUT_DIR="${1:-results_matrix}"
N_WORKERS="${2:-9}"
PROFILE=expert

ROUTES=()
for r in 121 149 190; do
    for l in sun_glare night; do
        for o in clear car cyclists crowd; do
            ROUTES+=("leaderboard/data/route${r}_${l}_${o}.xml")
        done
    done
done

# Fail before burning an hour of GPU time on a typo'd filename.
missing=0
for xml in "${ROUTES[@]}"; do
    [ -f "$xml" ] || { echo "ERROR: missing $xml" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1

mkdir -p "$OUT_DIR/logs"

echo "matrix:   ${#ROUTES[@]} routes, profile=$PROFILE"
echo "output:   $OUT_DIR"
echo "workers:  $N_WORKERS (gpu k, rpc 2000+10k, tm 2500+10k)"
echo

for ((k = 0; k < N_WORKERS; k++)); do
    slice=()
    for ((i = k; i < ${#ROUTES[@]}; i += N_WORKERS)); do
        slice+=("${ROUTES[$i]}")
    done
    [ ${#slice[@]} -eq 0 ] && continue
    echo "worker $k (gpu $k, port $((2000 + 10 * k))): ${#slice[@]} runs"
    for xml in "${slice[@]}"; do echo "    $(basename "$xml")"; done
done

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo
    echo "DRY_RUN=1, nothing launched."
    exit 0
fi

echo
echo "launching..."
pids=()
for ((k = 0; k < N_WORKERS; k++)); do
    slice=()
    for ((i = k; i < ${#ROUTES[@]}; i += N_WORKERS)); do
        slice+=("${ROUTES[$i]}")
    done
    [ ${#slice[@]} -eq 0 ] && continue

    (
        port=$((2000 + 10 * k))
        tm_port=$((2500 + 10 * k))
        for xml in "${slice[@]}"; do
            name="$(basename "${xml%.xml}")"
            echo "[worker $k] START $name"
            # Keep going if one route dies: a single bad run should not cost the batch.
            if RESULTS_ROOT="$OUT_DIR" bash leaderboard/scripts/run_custom_route.sh \
                    "$xml" "$PROFILE" "$k" "" "$port" "$tm_port"; then
                echo "[worker $k] OK    $name"
            else
                echo "[worker $k] FAIL  $name (exit $?)" >&2
            fi
        done
    ) > "$OUT_DIR/logs/worker${k}.log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

echo
echo "matrix done (worker failures: $fail). Per-worker logs: $OUT_DIR/logs/worker*.log"
echo "Results:  $OUT_DIR/*/results.json"
grep -h "FAIL " "$OUT_DIR"/logs/worker*.log 2>/dev/null || echo "no per-route failures logged"
echo
echo "CARLA servers are still running. Stop them with:"
for ((k = 0; k < N_WORKERS; k++)); do echo "  bash leaderboard/scripts/stop_carla.sh $((2000 + 10 * k))"; done
