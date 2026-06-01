#!/bin/bash
# NOTE for collaborators: absolute paths below point at /home/s56cai/DeepSight; adjust to your env.
# PYTHONPATH: make CARLA client and project root importable
export PYTHONPATH=$PYTHONPATH:/home/s56cai/Carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/s56cai/DeepSight

# CARLA server ports
BASE_PORT=2000        # must match the -carla-rpc-port you start ./CarlaUE4.sh with
BASE_TM_PORT=2500     # Traffic Manager port (CARLA-internal; just keep it offset from RPC)
IS_BENCH2DRIVE=True

# Pick routes — strongly recommend dev10 (10 routes) for the first quick run
BASE_ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10
# BASE_ROUTES=leaderboard/data/bench2drive220   # full benchmark (220 routes, 6-24 h)

TEAM_AGENT=team_code/qwen_b2d_agent.py
# Path to your trained checkpoint
TEAM_CONFIG=/home/s56cai/DeepSight/saves/qwen2_5vl-3b/deepsight/base/checkpoint-3148

# Output JSON name (incrementally written per route; set RESUME=True in run_evaluation.sh to continue)
BASE_CHECKPOINT_ENDPOINT=deepsight_base_dev10_$(date +%m%d_%H%M)
SAVE_PATH=results/${BASE_CHECKPOINT_ENDPOINT}/
PLANNER_TYPE=only_traj

# Which GPU runs the Qwen agent inference (avoid the GPU used by CARLA + the training GPUs)
GPU_RANK=0

PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"

mkdir -p $SAVE_PATH
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
