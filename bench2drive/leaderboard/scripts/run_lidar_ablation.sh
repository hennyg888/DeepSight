#!/bin/bash
# Black-LiDAR ablation on the matrix cells where `matrix_lidar_noafz` drove cleanly.
#
# Question: on the cells this checkpoint already handles without a collision, how much of
# that behaviour comes from the LiDAR BEV image? Every job runs the same checkpoint with
# a degraded BEV; the real BEV is still rendered to lidar_bev/ for inspection either way
# (see qwen_b2d_agent_with_lidar.py).
#
# MODE selects the degradation:
#   delay (default) -- the model sees the BEV from LIDAR_DELAY_S seconds ago; the first
#                      LIDAR_DELAY_S of each route are padded with the black BEV.
#   black           -- the BEV is all zeros for the whole route (the earlier ablation).
#
# Cell selection (from logs/matrix_repeats_report.md, the `matrix_lidar_noafz` table):
# obstacle != clear, AND outcome `...` -- clean in all three repeats, no collision.
# That is 5 of the 18 non-clear cells; the other 13 collide in at least one repeat, so a
# blacked-out run there has nothing to degrade from.
#
#   bash leaderboard/scripts/run_lidar_ablation.sh [RUN_DIR]
#
# REPEATS defaults to 3 to match the report's protocol -- single runs on this matrix are
# noise (report §3: cells are bistable and flip by 28-35 points). REPEATS=1 for a quick look.
# GPUS="0 1 2" to leave part of the box free.

set -uo pipefail

BENCH2DRIVE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH2DRIVE"
source "$BENCH2DRIVE/.venv/bin/activate"

# The ablation itself. One of the two must be active: with neither, this script is just a
# re-run of the baseline the report already contains.
MODE="${MODE:-delay}"
case "$MODE" in
  delay) export LIDAR_DELAY_S="${LIDAR_DELAY_S:-2}"; ABLATION="LIDAR_DELAY_S=$LIDAR_DELAY_S" ;;
  black) export BLACK_LIDAR_INPUT=1;                 ABLATION="BLACK_LIDAR_INPUT=1" ;;
  *) echo "unknown MODE '$MODE', expected delay|black" >&2; exit 1 ;;
esac
# 35 m BEV -- the lineage matrix_lidar_noafz was trained on. The agent builds both the BEV
# image and the "N m around the ego" prompt line from this, so it must not drift.
export BEV_RANGE_M=35
# Score-only runs; skip the per-frame anno sweep and the .laz dumps.
export SAVE_ANNO=0

CKPT="$BENCH2DRIVE/.model_mirror/matrix_lidar_noafz_checkpoint-38"
AGENT=team_code/qwen_b2d_agent_with_lidar.py
ROUTE_DIR="$BENCH2DRIVE/leaderboard/data"
PLANNER_TYPE=only_traj

CELLS="route149_night_cyclists route149_sun_glare_cyclists route149_sun_glare_crowd route190_night_cyclists route190_sun_glare_crowd"
REPEATS="${REPEATS:-3}"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5 6 7 8}"
MAX_RETRY=1

RUN_DIR="${1:-results_lidar_ablation/$(date +%m%d_%H%M)_${MODE}}"
mkdir -p "$RUN_DIR"
# 'latest' lives next to the run it points at, so passing a RUN_DIR under a different
# root (results_lidar_delay/...) can't repoint the previous root's symlink at a
# directory that isn't there.
ln -sfn "$(basename "$RUN_DIR")" "$(dirname "$RUN_DIR")/latest"

QUEUE="$RUN_DIR/queue.txt"
LOCK="$RUN_DIR/queue.lock"
touch "$LOCK"

# Rebuilt only when empty, so re-running the same RUN_DIR resumes rather than duplicates.
if [ ! -s "$QUEUE" ]; then
  : > "$QUEUE"
  for rep in $(seq 1 "$REPEATS"); do
    for cell in $CELLS; do
      echo "r${rep} ${cell}" >> "$QUEUE"
    done
  done
fi
echo -e "\033[36m[lidar-ablation] RUN_DIR=$RUN_DIR  jobs=$(wc -l < "$QUEUE")  GPUs=${GPU_LIST[*]}  mode=$MODE ($ABLATION)\033[0m"

worker() {
  local gpu=$1
  local port=$((30000 + gpu*200))
  local tm=$((50000 + gpu*200))
  while true; do
    local job
    job=$(flock "$LOCK" bash -c "head -1 '$QUEUE'; sed -i '1d' '$QUEUE'")
    [ -z "$job" ] && break
    local rep="${job%% *}"
    local route="${job##* }"
    local tag="${rep}__${route}"
    local xml="$ROUTE_DIR/${route}.xml"

    if [ ! -f "$xml" ]; then echo "[gpu$gpu] MISSING $xml"; continue; fi
    if [ -f "$RUN_DIR/$tag.json" ] && python -c \
       "import json,sys;d=json.load(open('$RUN_DIR/$tag.json'));sys.exit(0 if d.get('_checkpoint',{}).get('records') else 1)" 2>/dev/null; then
      echo "[gpu$gpu] $tag already done, skip"; continue
    fi

    mkdir -p "$RUN_DIR/$tag"
    echo "[gpu$gpu] $tag start (port $port)"
    bash leaderboard/scripts/run_evaluation.sh \
      $port $tm True "$xml" "$AGENT" "$CKPT" \
      "$RUN_DIR/$tag.json" "$RUN_DIR/$tag/" "$PLANNER_TYPE" $gpu \
      > "$RUN_DIR/$tag.log" 2>&1

    local status
    status=$(python -c "import json;d=json.load(open('$RUN_DIR/$tag.json'));r=d.get('_checkpoint',{}).get('records',[]);print(r[0]['status'] if r else 'NORECORD')" 2>/dev/null || echo NORECORD)
    if [[ "$status" == "NORECORD" || "$status" == *"Simulation crashed"* ]]; then
      local rt; rt=$(cat "$RUN_DIR/retry_$tag" 2>/dev/null || echo 0)
      if [ "$rt" -lt $MAX_RETRY ]; then
        echo $((rt+1)) > "$RUN_DIR/retry_$tag"; rm -f "$RUN_DIR/$tag.json"
        flock "$LOCK" bash -c "echo '$job' >> '$QUEUE'"
        echo "[gpu$gpu] $tag CARLA issue ($status), requeued"
      else
        echo "[gpu$gpu] $tag failed after retry, giving up"
      fi
    else
      echo "[gpu$gpu] $tag done: $status"
    fi
  done
  echo "[gpu$gpu] queue empty, worker exit"
}

for gpu in "${GPU_LIST[@]}"; do worker $gpu & sleep 4; done
wait
echo -e "\033[32m[lidar-ablation] all done -> $RUN_DIR\033[0m"
