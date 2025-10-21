#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME=${1:-langflow-dev}

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f "${CONTAINER_NAME}"
  echo "Stopped and removed container ${CONTAINER_NAME}."
else
  echo "No container named ${CONTAINER_NAME} is running."
fi
