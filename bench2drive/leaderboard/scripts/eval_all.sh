#!/bin/bash
# After a 220 run, compute ALL metrics in one go:
#   DS / SR / Efficiency / Comfortness / Ability (5 dims + Mean)
#
# NOTE for collaborators: absolute paths below point at /home/s56cai/DeepSight; adjust to your env.
#
# Usage: bash leaderboard/scripts/eval_all.sh <RUN_DIR>

export PYTHONPATH=$PYTHONPATH:/home/s56cai/Carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/s56cai/DeepSight
export CARLA_ROOT=/home/s56cai/Carla
PY=/home/s56cai/DeepSight/.venv-b2d/bin/python

RUN_DIR="$1"
[ -z "$RUN_DIR" ] && { echo "Usage: bash $0 <RUN_DIR>"; exit 1; }
[ -d "$RUN_DIR" ] || { echo "dir not found: $RUN_DIR"; exit 1; }

echo -e "\033[36m===== 1. merge all route json -> merged.json =====\033[0m"
$PY tools/merge_route_json.py -f "$RUN_DIR"

echo -e "\033[36m===== 2. DS / SR / Efficiency / Comfortness =====\033[0m"
$PY tools/compute_metrics.py --run_dir "$RUN_DIR" --total 220

echo -e "\033[36m===== 3. Ability 5 dims (needs CARLA, computes junction completion) =====\033[0m"
bash tools/clean_carla.sh
$PY tools/ability_benchmark.py \
    -f leaderboard/data/bench2drive220.xml \
    -r "$RUN_DIR/merged.json"
bash tools/clean_carla.sh

echo -e "\033[32m===== all metrics done. Ability details in $RUN_DIR/ability.json =====\033[0m"
