#!/bin/bash
# Quick single-route test for the LiDAR agent —— verifies that the "LiDAR BEV generated in real time at inference" matches the BEV of the training data.
# ⚠️ Verifying BEV consistency [does NOT depend on the model]: make_lidar_bev only uses CARLA point clouds + bev_projection, and is independent of the weights.
#    So you can run it now with [any checkpoint] to first check whether the BEV is correct; after the model finishes training, use the lidar checkpoint to check driving performance.
#
# Usage:
#   bash leaderboard/scripts/run_1route_lidar.sh <any checkpoint path>   # Verify BEV first (recommended, no need to wait for training to finish)
#   bash leaderboard/scripts/run_1route_lidar.sh                       # Default uses base_lidar (after training finishes)

export PYTHONPATH=$PYTHONPATH:/home/s56cai/Carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/s56cai/DeepSight
PY=/home/s56cai/DeepSight/.venv-b2d/bin/python

BASE_ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10
ALGO=qwenbase
PLANNER_TYPE=only_traj
TEAM_AGENT=team_code/qwen_b2d_agent_with_lidar.py
TEAM_CONFIG="${1:-/home/s56cai/DeepSight/saves/qwen2_5vl-3b/deepsight/base_lidar/checkpoint-3148}"

# Split out the single-route xml (if it does not exist)
ROUTE0="${BASE_ROUTES}_0_${ALGO}_${PLANNER_TYPE}.xml"
[ -f "$ROUTE0" ] || $PY tools/split_xml.py "$BASE_ROUTES" 10 "$ALGO" "$PLANNER_TYPE"

OUT=results_lidar_test/$(date +%m%d_%H%M)
mkdir -p "$OUT"
echo "checkpoint: $TEAM_CONFIG"
echo "route: $ROUTE0 (1 route)"
echo "GPU 0, port 2000"

bash leaderboard/scripts/run_evaluation.sh \
  2000 2500 True "$ROUTE0" "$TEAM_AGENT" "$TEAM_CONFIG" \
  "$OUT/route0.json" "$OUT/route0/" "$PLANNER_TYPE" 0 \
  2>&1 | tee "$OUT/route0.log"

echo
echo "✅ After the run finishes, the LiDAR BEV generated in real time at inference is at:"
echo "   $OUT/route0/Scenarios/*/lidar_bev/*.png"
echo "   Send a few of them to Claude to compare against the BEV of the training data, and confirm that ego is centered, the concentric rings, red = tall objects, and no green (intensity=0) are all consistent."
