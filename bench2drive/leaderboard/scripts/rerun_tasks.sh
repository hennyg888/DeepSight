#!/bin/bash
# Re-run only the given tasks, writing results back into the same run dir's task{i}.json.
# Each listed task is rerun from scratch; useful to recover CARLA-crashed routes.
#
# NOTE for collaborators: absolute paths below point at /home/s56cai/DeepSight; adjust to your env.
#
# Usage:
#   bash leaderboard/scripts/rerun_tasks.sh [RUN_DIR] [task_i ...]
# Default: rerun task 1 and 5 into the example dev10 dir:
#   bash leaderboard/scripts/rerun_tasks.sh

export PYTHONPATH=$PYTHONPATH:/home/s56cai/Carla/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/s56cai/DeepSight

# arg1 = RUN_DIR (default below), remaining args = task ids to rerun (default: 1 5)
RUN_DIR="${1:-results_multi/deepsight_cot_dev10_multi_0531_1746}"
shift 2>/dev/null
RERUN_TASKS=("$@")
[ ${#RERUN_TASKS[@]} -eq 0 ] && RERUN_TASKS=(1 5)

# ===== config (keep consistent with run_evaluation_qwen_multi.sh) =====
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
TEAM_AGENT=team_code/qwen_b2d_agent.py
TEAM_CONFIG=/home/s56cai/ckpt/DeepSight
PLANNER_TYPE=only_traj
ALGO=qwenbase
BASE_ROUTES=leaderboard/data/drivetransformer_bench2drive_dev10

echo -e "\033[36m[rerun] RUN_DIR=${RUN_DIR}  tasks=${RERUN_TASKS[*]}\033[0m"
for i in "${RERUN_TASKS[@]}"; do
    PORT=$(( BASE_PORT + i * 150 ))
    TM_PORT=$(( BASE_TM_PORT + i * 150 ))
    ROUTES="${BASE_ROUTES}_${i}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${RUN_DIR}/task${i}.json"
    SAVE_PATH="${RUN_DIR}/task${i}/"
    GPU_RANK=$i
    if [ ! -f "${ROUTES}" ]; then echo -e "\033[31m  task$i: shard xml not found ${ROUTES}, skip\033[0m"; continue; fi
    rm -f "${CHECKPOINT_ENDPOINT}"        # drop old empty json, rerun this route from scratch
    mkdir -p "${SAVE_PATH}"
    echo -e "\033[32m  task$i | GPU ${GPU_RANK} | port ${PORT}/${TM_PORT} | $(grep -c '<route ' ${ROUTES}) route -> ${CHECKPOINT_ENDPOINT}\033[0m"
    bash leaderboard/scripts/run_evaluation.sh \
        ${PORT} ${TM_PORT} ${IS_BENCH2DRIVE} ${ROUTES} ${TEAM_AGENT} ${TEAM_CONFIG} \
        ${CHECKPOINT_ENDPOINT} ${SAVE_PATH} ${PLANNER_TYPE} ${GPU_RANK} \
        > "${RUN_DIR}/task${i}.log" 2>&1 &
    sleep 10
done
echo -e "\033[36m[wait] waiting for rerun to finish... (live: tail -f ${RUN_DIR}/task1.log)\033[0m"
wait

echo -e "\033[32m[done] rerun finished, summarize:\033[0m"
python tools/summarize_timing.py --run_dir "${RUN_DIR}" --total_routes 220
