#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLOW_DIR="${ROOT_DIR}/artifacts/langflow"
MODEL_DIR="${ROOT_DIR}/models"
DATA_DIR="${ROOT_DIR}/data"
SRC_DIR="${ROOT_DIR}/src"
ENV_FILE="${ROOT_DIR}/.env.langflow"

mkdir -p "${FLOW_DIR}" "${MODEL_DIR}" "${DATA_DIR}"

docker run -d \
  --name langflow-dev \
  --restart unless-stopped \
  --add-host host.docker.internal:host-gateway \
  --env-file "${ENV_FILE}" \
  -e PYTHONPATH=/app/src \
  -v "${SRC_DIR}":/app/src \
  -v "${FLOW_DIR}":/app/exports \
  -v "${ROOT_DIR}/docker/langflow/flows":/app/flows \
  -v "${ROOT_DIR}/docker/langflow/custom_nodes":/app/custom_nodes \
  -v "${MODEL_DIR}":/app/models/trep \
  -v "${DATA_DIR}":/app/data \
  -p 7860:7860 \
  langflowai/langflow:1.5.1 \
  langflow run --host 0.0.0.0 --port 7860 --load /app/flows

echo "LangFlow running at http://127.0.0.1:7860"
