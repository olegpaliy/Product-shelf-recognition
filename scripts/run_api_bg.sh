#!/usr/bin/env bash
# Start API in background (survives terminal close). Logs: /tmp/shelf_api.log
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="${ROOT}/.venv"
LOG="${SHELF_API_LOG:-/tmp/shelf_api.log}"
PIDFILE="${SHELF_API_PIDFILE:-/tmp/shelf_api.pid}"

if [[ ! -x "${VENV}/bin/uvicorn" ]]; then
  echo "Missing venv at ${VENV}. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

if command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -nP -tiTCP:8000 -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${PIDS}" ]]; then
    echo "Port 8000 already in use (PIDs: ${PIDS}). Stop them or use scripts/run_api.sh in foreground."
    exit 1
  fi
fi

export PYTHONUNBUFFERED=1
nohup "${VENV}/bin/uvicorn" poc.api:app --host 127.0.0.1 --port 8000 >>"${LOG}" 2>&1 &
echo $! >"${PIDFILE}"
disown || true
echo "Started uvicorn PID $(cat "${PIDFILE}") — http://127.0.0.1:8000"
echo "Log: ${LOG} (first requests may take ~10s while ML libs load)"
