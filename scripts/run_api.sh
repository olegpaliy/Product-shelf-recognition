#!/usr/bin/env bash
# Stable API start for local demo (no --reload).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="${ROOT}/.venv"
if [[ ! -x "${VENV}/bin/uvicorn" ]]; then
  echo "Missing venv at ${VENV}. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
# Free port 8000 if something stale is listening
if command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -nP -tiTCP:8000 -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${PIDS}" ]]; then
    echo "Stopping PIDs on :8000: ${PIDS}"
    kill ${PIDS} 2>/dev/null || true
    sleep 1
  fi
fi
export PYTHONUNBUFFERED=1
exec "${VENV}/bin/uvicorn" poc.api:app --host 127.0.0.1 --port 8000
