#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/path/to/model}"
COORDINATOR_URL="${COORDINATOR_URL:-http://127.0.0.1:8080}"
STORE_URL="${STORE_URL:-tcp://127.0.0.1:8570}"
NODE_IP="${NODE_IP:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-10000}"
TP="${TP:-1}"
PP="${PP:-1}"

EXTRA_CONFIG=$(cat <<EOF
{"coordinator_url":"${COORDINATOR_URL}","store_url":"${STORE_URL}","node_ip":"${NODE_IP}","base_port":${BASE_PORT},"poll_interval_s":2,"poll_timeout_s":1800,"log_level":1}
EOF
)

vllm serve "${MODEL_PATH}" \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --load-format memfabric_http \
  --model-loader-extra-config "${EXTRA_CONFIG}"
