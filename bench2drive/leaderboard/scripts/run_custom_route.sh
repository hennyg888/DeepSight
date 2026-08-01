#!/bin/bash
# Run one custom route xml against a persistent CARLA server, writing to a fresh
# timestamped output dir every time.
#
# Complements run_route211_lidar_featdiff.sh (which is tied to the 220-route split
# naming and spawns its own throwaway server). Use this one for hand-built route
# files like leaderboard/data/route_149_copy.xml.
#
# Usage:
#   bash leaderboard/scripts/run_custom_route.sh <route_xml> [profile] [gpu] [checkpoint] [port] [tm_port]
#
#   profile is one of: base | lidar30 | afz | expert (default afz) -- see ../../PROFILES.md.
#   The profile sets the agent script, BEV_RANGE_M and the default checkpoint; these
#   MUST match what the checkpoint was trained on, which is the main reason to go
#   through this wrapper instead of calling run_evaluation.sh by hand.
#
#   bash leaderboard/scripts/run_custom_route.sh leaderboard/data/route_149_copy.xml
#   bash leaderboard/scripts/run_custom_route.sh leaderboard/data/route_149_copy.xml afz 0
#
#   'expert' runs fail2drive's PDM-lite privileged autopilot instead of a checkpoint
#   (team_code/pdm_lite_b2d_agent.py) -- no model is loaded, so the checkpoint argument
#   is ignored. Use it to collect training data whose trajectory labels come from the
#   expert rather than from a model imitating its own rollout:
#     bash leaderboard/scripts/run_custom_route.sh leaderboard/data/route121_car_glare.xml expert
#
# anno/*.json.gz (the Bench2Drive per-frame annotations that turn a run into DeepSight
# training samples, see team_code/b2d_anno.py) and lidar/*.laz (the raw point clouds
# behind the rendered lidar_bev pngs) are written by default; SAVE_ANNO=0 disables both.
#
# The CARLA server is started via start_carla.sh if it isn't already up (that script
# is idempotent), and is deliberately left running afterwards. Stop it with:
#   bash leaderboard/scripts/stop_carla.sh [port]
# Set SKIP_CARLA_START=1 to never auto-start.

set -euo pipefail
# On by default here; SAVE_ANNO=0 bash run_custom_route.sh ... turns it off for a run
# where you only care about the score (it costs a full actor sweep plus a ~100 KB .laz
# per frame).
export SAVE_ANNO="${SAVE_ANNO:-1}"

BENCH2DRIVE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH2DRIVE"

ROUTE_XML="${1:?usage: run_custom_route.sh <route_xml> [profile: base|lidar30|afz] [gpu] [checkpoint] [port] [tm_port]}"
PROFILE="${2:-afz}"
GPU="${3:-0}"
CHECKPOINT_OVERRIDE="${4:-}"
PORT="${5:-2000}"
TM_PORT="${6:-2500}"
PLANNER_TYPE=only_traj

if [ ! -f "$ROUTE_XML" ]; then
    echo "ERROR: route xml '$ROUTE_XML' not found" >&2
    exit 1
fi

MIRROR="$BENCH2DRIVE/.model_mirror"
# One switch drives agent script + BEV encoding + default checkpoint --
# see ../../PROFILES.md for the evidence behind these three settings.
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
  expert)
    # PDM-lite privileged autopilot -- drives from simulator state, loads no model, so
    # DEFAULT_CHECKPOINT is a placeholder the agent only reads the '+<save_name>' suffix
    # off. BEV_RANGE_M still applies: it sets the LiDAR BEV encoding of the recorded
    # pngs, which must match the checkpoint you intend to train.
    TEAM_AGENT=team_code/pdm_lite_b2d_agent.py
    DEFAULT_CHECKPOINT="expert"
    export BEV_RANGE_M="${BEV_RANGE_M:-35}"
    ;;
  *)
    echo "unknown profile '$PROFILE', expected base|lidar30|afz|expert" >&2
    exit 1
    ;;
esac
TEAM_CONFIG="${CHECKPOINT_OVERRIDE:-$DEFAULT_CHECKPOINT}"

# Where runs land. Override to keep a batch (e.g. a whole route matrix) out of the
# results_custom pile:  RESULTS_ROOT=results_matrix bash run_custom_route.sh ...
RESULTS_ROOT="${RESULTS_ROOT:-results_custom}"

# Fresh timestamped dir per run: never overwrites a previous run's results, and
# avoids run_evaluation.sh's RESUME=True skipping a route that a stale json
# already recorded as completed.
RUN_NAME="$(basename "${ROUTE_XML%.xml}")"
OUT="$RESULTS_ROOT/${RUN_NAME}_$(date +%m%d_%H%M%S)"
mkdir -p "$OUT"

# Stable path that always points at the newest run, so a file explorer / watch can
# stay parked on <RESULTS_ROOT>/latest and follow the live camera + lidar_bev frames
# as they stream in. Relative target so the link resolves from anywhere.
ln -sfn "$(basename "$OUT")" "$RESULTS_ROOT/latest"

# Attach to a persistent server instead of spawning a per-run one.
export LB_EXTERNAL_CARLA=1
if [ "${SKIP_CARLA_START:-0}" = "0" ]; then
    bash leaderboard/scripts/start_carla.sh "$PORT" "$GPU"
fi

echo
echo "profile:    $PROFILE (agent=$TEAM_AGENT${BEV_RANGE_M:+, BEV_RANGE_M=$BEV_RANGE_M})"
echo "checkpoint: $TEAM_CONFIG"
echo "route:      $ROUTE_XML"
echo "output:     $OUT"
echo "live view:  results_custom/latest/Scenarios/   (symlink -> this run; frames stream in"
echo "                                                once the model finishes loading, ~1 min)"
echo "GPU $GPU, CARLA at localhost:$PORT (persistent; stop_carla.sh to clean up)"
echo

bash leaderboard/scripts/run_evaluation.sh \
  "$PORT" "$TM_PORT" True "$ROUTE_XML" "$TEAM_AGENT" "$TEAM_CONFIG" \
  "$OUT/results.json" "$OUT/" "$PLANNER_TYPE" "$GPU" \
  2>&1 | tee "$OUT/run.log"

echo
echo "Done. Results: $OUT/results.json, log: $OUT/run.log"
echo "LiDAR BEV frames: $OUT/Scenarios/*/lidar_bev/*.png"
echo "CARLA is still running on port $PORT -- stop with:"
echo "  bash leaderboard/scripts/stop_carla.sh $PORT"
