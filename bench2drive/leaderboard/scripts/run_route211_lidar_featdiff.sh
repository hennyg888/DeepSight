#!/bin/bash
# Single-route closed-loop test on Bench2Drive-220 (route index 188 by default).
# Entirely self-contained in hhguo's DeepSight checkout: CARLA at /home/hhguo/Carla,
# python env at bench2drive/.venv (torch/transformers/carla/leaderboard deps installed there).
# Weights are read from local .model_mirror/ (see ../PROFILES.md for how those were built).
#
# Usage:
#   bash leaderboard/scripts/run_route211_lidar_featdiff.sh                              # lidar30, route 188, GPU 0
#   bash leaderboard/scripts/run_route211_lidar_featdiff.sh <profile> <route_idx> <gpu> <checkpoint>
#
#   profile is one of: base | lidar30 | afz -- see ../PROFILES.md for what each
#   one changes (agent script, BEV_RANGE_M, default checkpoint).

export PATH="/home/hhguo/DeepSight/bench2drive/.venv/bin:$PATH"
export PYTHONPATH=$PYTHONPATH:/home/hhguo/Carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/hhguo/DeepSight
PY=/home/hhguo/DeepSight/bench2drive/.venv/bin/python

BASE_ROUTES=leaderboard/data/bench2drive220
ALGO=qwenbase
PLANNER_TYPE=only_traj
PROFILE="${1:-lidar30}"
ROUTE_IDX="${2:-188}"
GPU="${3:-0}"
CHECKPOINT_OVERRIDE="${4:-}"

MIRROR=/home/hhguo/DeepSight/bench2drive/.model_mirror
# One switch drives agent script + BEV encoding + default checkpoint --
# see ../PROFILES.md for the evidence behind these three settings.
case "$PROFILE" in
  base)
    TEAM_AGENT=team_code/qwen_b2d_agent.py
    DEFAULT_CHECKPOINT="$MIRROR/base_checkpoint-3148"
    unset BEV_RANGE_M || true
    ;;
  lidar30)
    TEAM_AGENT=team_code/qwen_b2d_agent_with_lidar.py
    DEFAULT_CHECKPOINT="$MIRROR/base_lidar30_featdiff_checkpoint-3148"
    export BEV_RANGE_M=30
    ;;
  afz)
    TEAM_AGENT=team_code/qwen_b2d_agent_with_lidar.py
    DEFAULT_CHECKPOINT="$MIRROR/base_lidar_sq_afz_checkpoint-300"
    export BEV_RANGE_M=35
    ;;
  *)
    echo "unknown profile '$PROFILE', expected base|lidar30|afz" >&2
    exit 1
    ;;
esac
TEAM_CONFIG="${CHECKPOINT_OVERRIDE:-$DEFAULT_CHECKPOINT}"

# Split into 220 single-route xml files (only once, shared across runs/indices)
ROUTE_FILE="${BASE_ROUTES}_${ROUTE_IDX}_${ALGO}_${PLANNER_TYPE}.xml"
if [ ! -f "$ROUTE_FILE" ]; then
  echo "[split] splitting ${BASE_ROUTES}.xml into 220 single-route files"
  $PY tools/split_xml.py "$BASE_ROUTES" 220 "$ALGO" "$PLANNER_TYPE"
fi

OUT="results_lidar_test/route${ROUTE_IDX}_$(date +%m%d_%H%M)"
mkdir -p "$OUT"
echo "profile: $PROFILE (agent=$TEAM_AGENT${BEV_RANGE_M:+, BEV_RANGE_M=$BEV_RANGE_M})"
echo "checkpoint: $TEAM_CONFIG"
echo "route: $ROUTE_FILE (index $ROUTE_IDX of 220)"
echo "GPU $GPU, port 2000"

bash leaderboard/scripts/run_evaluation.sh \
  2000 2500 True "$ROUTE_FILE" "$TEAM_AGENT" "$TEAM_CONFIG" \
  "$OUT/route${ROUTE_IDX}.json" "$OUT/route${ROUTE_IDX}/" "$PLANNER_TYPE" "$GPU" \
  2>&1 | tee "$OUT/route${ROUTE_IDX}.log"

echo
echo "Done. Results: $OUT/route${ROUTE_IDX}.json, log: $OUT/route${ROUTE_IDX}.log"
echo "Live-generated LiDAR BEV frames: $OUT/route${ROUTE_IDX}/Scenarios/*/lidar_bev/*.png"
