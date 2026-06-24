#!/bin/bash
# Dynamic scheduling to run dev10 (10 routes): 9-GPU work-stealing (whichever GPU is idle grabs the next route) + resume + retry on CARLA crash
# Uses base/checkpoint-3148 (no-CoT model). Same queue/retry logic as run_220_dynamic.sh, only the routes are switched to dev10 and the checkpoint to base.
# WARNING: Companion requirement: the agent's get_prompt CoT flag must be <CoT_flag_False> (already changed), because this is a no-CoT model.
#
# Usage:
#   bash leaderboard/scripts/run_dev10_dynamic.sh                 # fresh run
#   bash leaderboard/scripts/run_dev10_dynamic.sh <existing RUN_DIR>   # resume (skip already completed)

export PYTHONPATH=$PYTHONPATH:/home/s56cai/Carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/s56cai/DeepSight
PY=/home/s56cai/DeepSight/.venv-b2d/bin/python   # b2d environment (has carla)

# ===== Configuration =====
BASE_ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10
TEAM_AGENT=team_code/qwen_b2d_agent.py
TEAM_CONFIG=/home/s56cai/DeepSight/saves/qwen2_5vl-3b/deepsight/base/checkpoint-3148
PLANNER_TYPE=only_traj
ALGO=qwenbase
GPU_LIST=(0 1 2 3 4 5 6 7 8)
N=10
MAX_RETRY=2

RUN_DIR="${1:-results_multi/deepsight_base_dev10_dyn_$(date +%m%d_%H%M)}"
mkdir -p "$RUN_DIR"
echo -e "\033[36m[run_dev10] RUN_DIR=$RUN_DIR  GPUs=${GPU_LIST[*]}\033[0m"

# 1. Split into N single-route xml files (re-split if _0 or _{N-1} does not exist)
if [ ! -f "${BASE_ROUTES}_0_${ALGO}_${PLANNER_TYPE}.xml" ] || [ ! -f "${BASE_ROUTES}_$((N-1))_${ALGO}_${PLANNER_TYPE}.xml" ]; then
  echo -e "\033[33m[split] Splitting into ${N} single-route xml files\033[0m"
  $PY tools/split_xml.py "$BASE_ROUTES" $N "$ALGO" "$PLANNER_TYPE"
fi

# 2. Queue + lock (enqueue everything; already-completed ones will be skipped by the worker)
QUEUE="$RUN_DIR/queue.txt"
LOCK="$RUN_DIR/queue.lock"
touch "$LOCK"
seq 0 $((N-1)) > "$QUEUE"

WALL_START=$(date +%s); echo $WALL_START > "$RUN_DIR/wall_start.txt"

# worker: bind to one GPU, loop and atomically take a route from the queue
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
    if [ -f "$ckpt" ] && $PY -c "import json,sys;d=json.load(open('$ckpt'));sys.exit(0 if d.get('_checkpoint',{}).get('records') else 1)" 2>/dev/null; then
      echo "[gpu$gpu] route $idx already has a result, skipping"; continue
    fi
    mkdir -p "$save"
    echo "[gpu$gpu] route $idx starting (port $port)"
    bash leaderboard/scripts/run_evaluation.sh \
      $port $tm_port True "$routes" "$TEAM_AGENT" "$TEAM_CONFIG" \
      "$ckpt" "$save" "$PLANNER_TYPE" $gpu \
      > "$RUN_DIR/route_${idx}.log" 2>&1
    # CARLA crash (no record or Simulation crashed) -> retry; model-type failures are kept and not retried
    local status
    status=$($PY -c "import json;d=json.load(open('$ckpt'));r=d.get('_checkpoint',{}).get('records',[]);print(r[0]['status'] if r else 'NORECORD')" 2>/dev/null || echo NORECORD)
    if [[ "$status" == "NORECORD" || "$status" == *"Simulation crashed"* ]]; then
      local rt; rt=$(cat "$RUN_DIR/retry_${idx}" 2>/dev/null || echo 0)
      if [ "$rt" -lt $MAX_RETRY ]; then
        echo $((rt+1)) > "$RUN_DIR/retry_${idx}"; rm -f "$ckpt"
        flock "$LOCK" bash -c "echo $idx >> '$QUEUE'"
        echo "[gpu$gpu] route $idx CARLA error ($status), re-enqueued retry=$((rt+1))"
      else
        echo "[gpu$gpu] route $idx still failed after ${MAX_RETRY} retries, giving up"
      fi
    else
      echo "[gpu$gpu] route $idx done: $status"
    fi
  done
  echo "[gpu$gpu] queue empty, worker exiting"
}

echo -e "\033[36m[launch] ${#GPU_LIST[@]} GPUs dynamically scheduling $N routes (staggered start)\033[0m"
for gpu in "${GPU_LIST[@]}"; do worker $gpu & sleep 4; done
wait

WALL_END=$(date +%s); echo $WALL_END > "$RUN_DIR/wall_end.txt"
echo -e "\033[32m[done] Total wall-clock time $((WALL_END-WALL_START)) seconds\033[0m"

# 3. Compute DS/SR/Efficiency/Comfortness (dev10 is only 10 routes; Ability sample is too small to compute)
$PY tools/compute_metrics.py --run_dir "$RUN_DIR"
