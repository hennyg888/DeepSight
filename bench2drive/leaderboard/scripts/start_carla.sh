#!/bin/bash
# Launch a persistent CARLA server for leaderboard evaluation.
#
# The server is detached (setsid + nohup), so it survives the shell that started it
# and can be reused across many eval runs. Pair with stop_carla.sh to clean up.
#
# Usage: bash leaderboard/scripts/start_carla.sh [port] [gpu]
#   port  RPC port to bind (default 2000)
#   gpu   GPU index for -graphicsadapter (default 0)

export CARLA_ROOT=${CARLA_ROOT:-/home/hhguo/Carla}

PORT=${1:-2000}
GPU=${2:-0}
LOG=${CARLA_LOG:-/tmp/carla_${PORT}.log}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-120}

if [ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]; then
    echo "ERROR: ${CARLA_ROOT}/CarlaUE4.sh not found or not executable" >&2
    exit 1
fi

# Already up on this port? Reuse it rather than starting a second server.
if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
    echo "CARLA already listening on port ${PORT}, reusing it:"
    pgrep -af "carla-rpc-port=${PORT}" | grep -v pgrep
    exit 0
fi

echo "Starting CARLA on port ${PORT} (gpu ${GPU}), log -> ${LOG}"
nohup setsid "${CARLA_ROOT}/CarlaUE4.sh" \
    -RenderOffScreen -nosound \
    -carla-rpc-port="${PORT}" \
    -graphicsadapter="${GPU}" \
    > "${LOG}" 2>&1 &

# Wait for the RPC port to accept connections instead of a blind sleep.
echo -n "Waiting for CARLA to accept connections on ${PORT}"
for _ in $(seq "${BOOT_TIMEOUT}"); do
    if timeout 1 bash -c "</dev/tcp/127.0.0.1/${PORT}" 2>/dev/null; then
        echo
        echo "CARLA is up:"
        pgrep -af "carla-rpc-port=${PORT}" | grep -v pgrep
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo
echo "ERROR: CARLA did not come up within ${BOOT_TIMEOUT}s. Last log lines:" >&2
tail -20 "${LOG}" >&2
exit 1
