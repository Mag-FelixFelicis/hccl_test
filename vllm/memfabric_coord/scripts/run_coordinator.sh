#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

python3 /app/coordinator.py --host "${HOST}" --port "${PORT}"
