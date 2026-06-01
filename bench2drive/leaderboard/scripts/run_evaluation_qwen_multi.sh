#!/bin/bash
# Multi-GPU eval of dev10: one CARLA + one Qwen inference per GPU, routes split across GPUs.
# Adapted from the official run_evaluation_multi_vad.sh for the qwen base model + dev10 + timing.
#
# NOTE for collaborators: absolute paths below point at /home/s56cai/DeepSight; adjust to your env.

# PYTHONPATH: CARLA client + project root
export PYTHONPATH=$PYTHONPATH:/home/s56cai/Carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/s56cai/DeepSight

# ===== config =====
BASE_PORT=30000        # CARLA rpc start port, +150 per task to avoid clashes
BASE_TM_PORT=50000     # Traffic Manager start port
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10   # without .xml
TEAM_AGENT=team_code/qwen_b2d_agent.py
TEAM_CONFIG=/home/s56cai/ckpt/DeepSight
PLANNER_TYPE=only_traj
ALGO=qwenbase

# GPUs to use (each GPU = one task = one CARLA + one Qwen)
GPU_RANK_LIST=(0 1 2 3 4 5 6 7 8)
TASK_NUM=${#GPU_RANK_LIST[@]}

# unified run name + output dir (all shard json/log kept together for easy stats)
RUN_NAME=deepsight_cot_dev10_multi_$(date +%m%d_%H%M)
OUT_DIR=results_multi/${RUN_NAME}
mkdir -p ${OUT_DIR}

# ===== split routes into TASK_NUM shards =====
echo -e "\033[33m[split] split dev10 into ${TASK_NUM} shards\033[0m"
python tools/split_xml.py ${BASE_ROUTES} ${TASK_NUM} ${ALGO} ${PLANNER_TYPE}

# ===== launch in parallel =====
echo -e "\033[36m[launch] ${TASK_NUM} tasks, GPUs: ${GPU_RANK_LIST[*]}\033[0m"
WALL_START=$(date +%s)
echo "${WALL_START}" > ${OUT_DIR}/wall_start.txt

for (( i=0; i<${TASK_NUM}; i++ )); do
    PORT=$(( BASE_PORT + i * 150 ))
    TM_PORT=$(( BASE_TM_PORT + i * 150 ))
    ROUTES="${BASE_ROUTES}_${i}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${OUT_DIR}/task${i}.json"
    SAVE_PATH="${OUT_DIR}/task${i}/"
    GPU_RANK=${GPU_RANK_LIST[$i]}
    mkdir -p ${SAVE_PATH}
    echo -e "\033[32m task${i} | GPU ${GPU_RANK} | port ${PORT}/${TM_PORT} | $(grep -c '<route ' ${ROUTES}) routes -> ${CHECKPOINT_ENDPOINT}\033[0m"
    bash leaderboard/scripts/run_evaluation.sh \
        ${PORT} ${TM_PORT} ${IS_BENCH2DRIVE} ${ROUTES} ${TEAM_AGENT} ${TEAM_CONFIG} \
        ${CHECKPOINT_ENDPOINT} ${SAVE_PATH} ${PLANNER_TYPE} ${GPU_RANK} \
        > ${OUT_DIR}/task${i}.log 2>&1 &
    sleep 8   # stagger startup so 9 CARLAs don't fight for resources at once
done

echo -e "\033[36m[wait] waiting for all tasks... (live: tail -f ${OUT_DIR}/task0.log)\033[0m"
wait

WALL_END=$(date +%s)
echo "${WALL_END}" > ${OUT_DIR}/wall_end.txt
echo -e "\033[32m[done] total wall-clock: $(( WALL_END - WALL_START )) s\033[0m"

# ===== timing stats + 220 estimate =====
python tools/summarize_timing.py --run_dir ${OUT_DIR} --total_routes 220
