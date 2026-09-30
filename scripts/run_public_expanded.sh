#!/usr/bin/env bash
# Run inside screen. No test suite, GPU smoke, inference benchmark, or internal data.
set -euo pipefail
cd "$(dirname "$0")/.."
test -x .venv/bin/python
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
SOURCE_CONFIG="${1:-runs/pretraining_20260930T173002_680253Z/config.json}"
test -f "$SOURCE_CONFIG"
LAUNCH_DIR="runs/expanded_launcher_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$LAUNCH_DIR"
exec > >(tee -a "$LAUNCH_DIR/console.log") 2>&1
echo "LOG_DIRECTORY=$LAUNCH_DIR"
# Broad acquisition continues independently of admitted-matrix preparation.
# It records unresolved sources explicitly; only audited recipes enter SFT.
python -u scripts/download_public_catalogue.py \
  --destination data/public/catalogue_archive_v2 --workers 4 \
  > "$LAUNCH_DIR/catalogue_download.log" 2>&1 &
DOWNLOAD_PID=$!
trap 'echo "Catalogue download PID: $DOWNLOAD_PID; log: $LAUNCH_DIR/catalogue_download.log"' EXIT
python -u scripts/prepare_public_expanded.py --source-config "$SOURCE_CONFIG" \
  --run-dir "$LAUNCH_DIR/model_run"
CONFIG="$(cat data/public/expanded_v2/latest_training_config.txt)"
python -u -m factorbridge train --config "$CONFIG"
cat "$LAUNCH_DIR/model_run/training/status.json"
echo "FORMAL_TRAINING_COMMAND_COMPLETED: see actual steps.jsonl and training/adapter"
echo "Waiting for catalogue acquisition; internal killifish has not been accessed."
DOWNLOAD_EXIT=0
wait "$DOWNLOAD_PID" || DOWNLOAD_EXIT=$?
echo "CATALOGUE_ACQUISITION_EXIT=$DOWNLOAD_EXIT (2 means explicitly incomplete; see acquisition_report.json)"
echo "TRAINING_RUN=$LAUNCH_DIR/model_run"
echo "Scientific performance and final killifish test have NOT been evaluated."
