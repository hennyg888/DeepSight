#!/bin/bash
# Dynamic scheduling to run full bench2drive 220: base/checkpoint-3148 (no-CoT model)
# 9 GPUs work-stealing (whichever GPU is free grabs the next route) + resume + auto-retry on CARLA crash; computes all 5 metrics after finishing (including Ability)
# ⚠️ Required setup: the agent's get_prompt CoT flag must be <CoT_flag_False> (no-CoT model), already changed
# ⚠️ 220 takes about 14h, make sure to run inside tmux/nohup, disconnecting SSH will kill the process
#
# Usage:
#   bash leaderboard/scripts/run_220_base.sh                 # New run (automatically creates a timestamped directory)
#   bash leaderboard/scripts/run_220_base.sh <existing RUN_DIR>   # Resume (skips already completed)

export PYTHONPATH=$PYTHONPATH:/home/s56cai/Carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/s56cai/DeepSight
PY=/home/s56cai/DeepSight/.venv-b2d/bin/python   # b2d environment (has carla)

# ===== Configuration =====
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=team_code/qwen_b2d_agent_with_lidar.py
TEAM_CONFIG=/home/s56cai/DeepSight/saves/qwen2_5vl-3b/deepsight/base_lidar_sq/checkpoint-3148
PLANNER_TYPE=only_traj
ALGO=qwenbase
GPU_LIST=(0 1 2 3 4 5 6 7 8)
N=220
MAX_RETRY=2

RUN_DIR="${1:-results_220/deepsight_lidar_sq_220_$(date +%m%d_%H%M)}"
mkdir -p "$RUN_DIR"
echo -e "\033[36m[run_220_base] RUN_DIR=$RUN_DIR  GPUs=${GPU_LIST[*]}\033[0m"

# 1. Split into 220 single-route xml files (split only once, shared by all runs)
if [ ! -f "${BASE_ROUTES}_0_${ALGO}_${PLANNER_TYPE}.xml" ] || [ ! -f "${BASE_ROUTES}_$((N-1))_${ALGO}_${PLANNER_TYPE}.xml" ]; then
  echo -e "\033[33m[split] Splitting into ${N} single-route xml files\033[0m"
  $PY tools/split_xml.py "$BASE_ROUTES" $N "$ALGO" "$PLANNER_TYPE"
fi

# 2. Queue + lock (enqueue everything; already completed ones will be skipped by the worker)
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
      echo "[gpu$gpu] route $idx already has results, skipping"; continue
    fi
    mkdir -p "$save"
    echo "[gpu$gpu] route $idx starting (port $port)"
    bash leaderboard/scripts/run_evaluation.sh \
      $port $tm_port True "$routes" "$TEAM_AGENT" "$TEAM_CONFIG" \
      "$ckpt" "$save" "$PLANNER_TYPE" $gpu \
      > "$RUN_DIR/route_${idx}.log" 2>&1
    # CARLA crash (no record or Simulation crashed) -> retry; model-class failures are kept and not retried
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
echo -e "\033[32m[done] total wall clock $((WALL_END-WALL_START)) seconds\033[0m"

# 3. Compute all 5 metrics: DS/SR/Efficiency/Comfortness + Ability (eval_all internally does merge + compute_metrics + ability_benchmark)
echo -e "\033[36m[metrics] Computing all 5 metrics (including Ability, will start CARLA)\033[0m"
bash leaderboard/scripts/eval_all.sh "$RUN_DIR"
