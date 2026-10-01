#!/usr/bin/env bash
# Continue an already-prepared run. Never rerun preparation, download, or smoke.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:?Pass the existing expanded model_run/config.json}"
test -f "$CONFIG"
test -x .venv/bin/python
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
LOG_DIR="runs/train_resume_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/console.log") 2>&1
echo "RESUME_CONFIG=$CONFIG"
echo "LOG_DIRECTORY=$LOG_DIR"
python -u -m factorbridge train --config "$CONFIG"
echo "Training command completed. Read the existing model_run/training/status.json for the actual outcome."
