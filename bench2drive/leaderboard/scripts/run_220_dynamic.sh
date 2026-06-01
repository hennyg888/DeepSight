#!/bin/bash
# Run the full bench2drive 220 with dynamic scheduling: 9-GPU work-stealing
# (whichever GPU is free grabs the next route). Each route gets its own
# json / metric_info / log; supports resume (skip done routes) + auto-retry on CARLA crash.
# Auto-computes the first 4 metrics (DS/SR/Efficiency/Comfortness) when finished.
#
# NOTE for collaborators: the absolute paths below (project root, .venv-b2d, ckpt/DeepSight)
# point at /home/s56cai/DeepSight; adjust them to your environment.
#
# Usage:
#   bash leaderboard/scripts/run_220_dynamic.sh                  # fresh run (auto timestamped dir)
#   bash leaderboard/scripts/run_220_dynamic.sh <existing RUN_DIR>   # resume (continue unfinished routes)

export PYTHONPATH=$PYTHONPATH:/home/s56cai/Carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/s56cai/DeepSight
PY=/home/s56cai/DeepSight/.venv-b2d/bin/python   # b2d env (has carla)

# ===== config =====
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=team_code/qwen_b2d_agent.py
TEAM_CONFIG=/home/s56cai/ckpt/DeepSight
PLANNER_TYPE=only_traj
ALGO=qwenbase
GPU_LIST=(0 1 2 3 4 5 6 7 8)
N=220
MAX_RETRY=2

RUN_DIR="${1:-results_220/deepsight_cot_220_$(date +%m%d_%H%M)}"
mkdir -p "$RUN_DIR"
echo -e "\033[36m[run_220] RUN_DIR=$RUN_DIR  GPUs=${GPU_LIST[*]}\033[0m"

# 1. Split into 220 single-route xml (only once, shared across runs)
if [ ! -f "${BASE_ROUTES}_0_${ALGO}_${PLANNER_TYPE}.xml" ]; then
  echo -e "\033[33m[split] splitting into ${N} single-route xml\033[0m"
  $PY tools/split_xml.py "$BASE_ROUTES" $N "$ALGO" "$PLANNER_TYPE"
fi

# 2. Queue + lock (enqueue all; done routes are skipped by the workers)
QUEUE="$RUN_DIR/queue.txt"
LOCK="$RUN_DIR/queue.lock"
touch "$LOCK"
seq 0 $((N-1)) > "$QUEUE"

WALL_START=$(date +%s); echo $WALL_START > "$RUN_DIR/wall_start.txt"

# worker: bound to one GPU, loops pulling routes atomically from the queue
worker() {
  local gpu=$1
  local port=$((30000 + gpu*200))
  local tm_port=$((50000 + gpu*200))
  while true; do
    local idx
    idx=$(flock "$LOCK" bash -c "head -1 '$QUEUE'; sed -i '1d' '$QUEUE'")
    [ -z "$idx" ] && break
    local routes="${BASE_ROUTES}_${idx}_${ALGO}_${PLANNER_TYPE}.xml"
    local ckpt="$RUN_DIR/route_${idx}.json"
    local save="$RUN_DIR/route_${idx}/"
    # resume: skip if a non-empty result already exists
    if [ -f "$ckpt" ] && $PY -c "import json,sys;d=json.load(open('$ckpt'));sys.exit(0 if d.get('_checkpoint',{}).get('records') else 1)" 2>/dev/null; then
      echo "[gpu$gpu] route $idx already has result, skip"; continue
    fi
    mkdir -p "$save"
    echo "[gpu$gpu] route $idx start (port $port)"
    bash leaderboard/scripts/run_evaluation.sh \
      $port $tm_port True "$routes" "$TEAM_AGENT" "$TEAM_CONFIG" \
      "$ckpt" "$save" "$PLANNER_TYPE" $gpu \
      > "$RUN_DIR/route_${idx}.log" 2>&1
    # CARLA crash (no record or Simulation crashed) -> retry; model-side failures are kept (not retried)
    local status
    status=$($PY -c "import json;d=json.load(open('$ckpt'));r=d.get('_checkpoint',{}).get('records',[]);print(r[0]['status'] if r else 'NORECORD')" 2>/dev/null || echo NORECORD)
    if [[ "$status" == "NORECORD" || "$status" == *"Simulation crashed"* ]]; then
      local rt; rt=$(cat "$RUN_DIR/retry_${idx}" 2>/dev/null || echo 0)
      if [ "$rt" -lt $MAX_RETRY ]; then
        echo $((rt+1)) > "$RUN_DIR/retry_${idx}"; rm -f "$ckpt"
        flock "$LOCK" bash -c "echo $idx >> '$QUEUE'"
        echo "[gpu$gpu] route $idx CARLA error ($status), re-queued retry=$((rt+1))"
      else
        echo "[gpu$gpu] route $idx still failing after ${MAX_RETRY} retries, give up"
      fi
    else
      echo "[gpu$gpu] route $idx done: $status"
    fi
  done
  echo "[gpu$gpu] queue empty, worker exits"
}

echo -e "\033[36m[launch] dynamic scheduling $N routes over ${#GPU_LIST[@]} GPUs (staggered start)\033[0m"
for gpu in "${GPU_LIST[@]}"; do worker $gpu & sleep 4; done
wait

WALL_END=$(date +%s); echo $WALL_END > "$RUN_DIR/wall_end.txt"
echo -e "\033[32m[done] total wall-clock $((WALL_END-WALL_START)) s\033[0m"

# 3. Compute DS/SR/Efficiency/Comfortness (Ability via eval_all.sh)
$PY tools/compute_metrics.py --run_dir "$RUN_DIR" --total 220
echo -e "\033[33mTip: for the Ability 5-dim metric run: bash leaderboard/scripts/eval_all.sh $RUN_DIR\033[0m"
