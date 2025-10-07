#!/usr/bin/env bash
set -euo pipefail

# Setup script for yoloe_cam_filter_webapp
# - Creates a Python venv
# - Installs dependencies
# - Ensures YOLOv11n-seg model is available (auto-download via Ultralytics alias)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
VENV_DIR="${ROOT_DIR}/.venv"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[INFO] Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[INFO] Upgrading pip"
python -m pip install --upgrade pip wheel setuptools

REQ_FILE="${ROOT_DIR}/requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
  cat > "${REQ_FILE}" <<'REQS'
ultralytics>=8.3.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.2.0
fastapi>=0.111.0
uvicorn>=0.30.0
python-multipart>=0.0.9
Pillow>=10.0.0
REQS
fi

echo "[INFO] Installing Python dependencies"
pip install -r "${REQ_FILE}"

echo "[INFO] Verifying model availability (will auto-download if missing)"
python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolo11n-seg.pt')  # downloads to ultralytics cache on first use
print('Model ready:', getattr(model, 'names', None) is not None)
PY

echo "[SUCCESS] Environment ready. Activate with: source ${VENV_DIR}/bin/activate"
echo "[SUCCESS] Run server via: ./run.sh --reload"


