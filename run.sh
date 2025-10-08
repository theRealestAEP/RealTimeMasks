#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

# Prefer active venv, else use .venv, else myenv (AI Generated kinda sloppy but w/e it works)
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/uvicorn" ]]; then
  VENV_DIR="${VIRTUAL_ENV}"
elif [[ -x "${ROOT_DIR}/.venv/bin/uvicorn" ]]; then
  VENV_DIR="${ROOT_DIR}/.venv"
elif [[ -x "${ROOT_DIR}/myenv/bin/uvicorn" ]]; then
  VENV_DIR="${ROOT_DIR}/myenv"
else
  VENV_DIR="${ROOT_DIR}/.venv"
fi
UVICORN_BIN="${VENV_DIR}/bin/uvicorn"
PYTHON_BIN="${VENV_DIR}/bin/python"

if [[ ! -x "${UVICORN_BIN}" ]]; then
  echo "[ERROR] Uvicorn not found at ${UVICORN_BIN}. Activate/install venv first:" >&2
  echo "       ./setup.sh" >&2
  echo "    or python3 -m venv '${VENV_DIR}' && source '${VENV_DIR}/bin/activate' && pip install -r '${ROOT_DIR}/requirements.txt'" >&2
  exit 1
fi

# Defaults (override via env or flags)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD=""

# YOLO settings (override via env)
# Use the canonical weight alias so Ultralytics auto-downloads on first run
export YOLO_MODEL="${YOLO_MODEL:-yolo11n-seg.pt}"
export YOLO_DEVICE="${YOLO_DEVICE:-}"
export YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
export YOLO_CONF="${YOLO_CONF:-0.25}"
export YOLO_MASK_CONF="${YOLO_MASK_CONF:-${YOLO_CONF}}"
export YOLO_DEFAULT_OVERLAY="${YOLO_DEFAULT_OVERLAY:-${ROOT_DIR}/cat-smile-smiling-cat.png}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--host HOST] [--port PORT] [--reload]

Environment overrides:
  YOLO_MODEL=/abs/path/to/model.pt
  YOLO_DEVICE=cuda|mps|cpu
  YOLO_IMGSZ=640
  YOLO_CONF=0.25
  YOLO_MASK_CONF=0.25
  YOLO_DEFAULT_OVERLAY=/abs/path/to.png
  HOST=0.0.0.0  PORT=8000
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"; shift 2 ;;
    --port)
      PORT="$2"; shift 2 ;;
    --reload)
      RELOAD="--reload"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

cd "${ROOT_DIR}"
echo "[INFO] Starting server on ${HOST}:${PORT}"
echo "[INFO] Model=${YOLO_MODEL} Device=${YOLO_DEVICE:-<default>} ImgSz=${YOLO_IMGSZ} Conf=${YOLO_CONF} MaskConf=${YOLO_MASK_CONF}"

# Prevent mixing global site-packages
export PYTHONNOUSERSITE=1

exec "${UVICORN_BIN}" server:create_app --factory --host "${HOST}" --port "${PORT}" ${RELOAD}


