#!/bin/bash
# Stop the persistent CARLA server started by start_carla.sh.
#
# Usage: bash leaderboard/scripts/stop_carla.sh [port]
#   port  RPC port of the server to stop (default 2000). Pass "all" to stop every
#         CarlaUE4 server owned by the current user.

TARGET=${1:-2000}

if [ "${TARGET}" = "all" ]; then
    PATTERN="CarlaUE4-Linux-Shipping"
    echo "Stopping ALL CarlaUE4 servers owned by $(whoami)"
else
    PATTERN="carla-rpc-port=${TARGET}"
    echo "Stopping CARLA server on port ${TARGET}"
fi

PIDS=$(pgrep -u "$(whoami)" -f "${PATTERN}")

if [ -z "${PIDS}" ]; then
    echo "No matching CARLA process found - nothing to stop."
else
    echo "Killing: $(echo ${PIDS} | tr '\n' ' ')"
    # SIGKILL directly: CarlaUE4 does not shut down reliably on SIGTERM.
    kill -9 ${PIDS} 2>/dev/null
    sleep 2
fi

# Report anything still alive. A <defunct> entry is a reaped-pending zombie and is
# harmless - it holds no port and disappears once its parent exits.
REMAINING=$(pgrep -u "$(whoami)" -af "${PATTERN}" | grep -v pgrep)
if [ -n "${REMAINING}" ]; then
    echo "Still present (defunct entries are harmless):"
    echo "${REMAINING}"
fi

if [ "${TARGET}" != "all" ]; then
    if timeout 1 bash -c "</dev/tcp/127.0.0.1/${TARGET}" 2>/dev/null; then
        echo "WARNING: something is still listening on port ${TARGET}" >&2
        exit 1
    fi
    echo "Port ${TARGET} is free."
fi
