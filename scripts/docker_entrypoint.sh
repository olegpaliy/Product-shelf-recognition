#!/usr/bin/env bash
# Fail loudly if YOLO weights are missing (otherwise code falls back to yolov8n).
set -euo pipefail

WEIGHTS="/app/models/sku110k/sku110k-finetuned.pt"
BASE="/app/models/sku110k/sku110k-yolo11-s640.pt"

if [[ ! -f "${WEIGHTS}" && ! -f "${BASE}" ]]; then
  cat <<'EOF' >&2
ERROR: No YOLO weights mounted.

Place at least one of these on the host, then re-run compose:

  models/sku110k/sku110k-finetuned.pt   (preferred — same as local demo)
  models/sku110k/sku110k-yolo11-s640.pt

Example:
  mkdir -p models/sku110k
  # copy .pt files from the machine that already runs the PoC
  docker compose up --build
EOF
  exit 1
fi

if [[ -f "${WEIGHTS}" ]]; then
  echo "Detector weights: ${WEIGHTS}"
else
  echo "Detector weights: ${BASE} (finetuned missing — quality may differ from local demo)"
fi

exec python -m uvicorn poc.api:app --host 0.0.0.0 --port 8000
