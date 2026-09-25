#!/usr/bin/env bash
# Fail loudly if YOLO weights are missing (otherwise code falls back to yolov8n).
set -euo pipefail

WEIGHTS="/app/models/sku110k/sku110k-finetuned.pt"
BASE="/app/models/sku110k/sku110k-yolo11-s640.pt"

if [[ ! -f "${WEIGHTS}" && ! -f "${BASE}" ]]; then
  cat <<'EOF' >&2
ERROR: No YOLO weights mounted.

Need at least:

  models/sku110k/sku110k-finetuned.pt

(Git LFS: git lfs pull, then docker compose up --build)
EOF
  exit 1
fi

if [[ -f "${WEIGHTS}" ]]; then
  echo "Detector weights: ${WEIGHTS}"
else
  echo "Detector weights: ${BASE} (finetuned missing — quality may differ from local demo)"
fi

exec python -m uvicorn poc.api:app --host 0.0.0.0 --port 8000
