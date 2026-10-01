#!/usr/bin/env bash
# Run inside screen; preserve all old runs. Each stage has real, inspectable outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
SOURCE_CONFIG="${1:?Pass the completed expanded run config.json}"
test -f "$SOURCE_CONFIG"
LAUNCH="runs/semantic_launcher_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$LAUNCH"
exec > >(tee -a "$LAUNCH/console.log") 2>&1
echo "LOG_DIRECTORY=$LAUNCH"
trap 'code=$?; echo "PIPELINE_EXIT=$code LOG=$LAUNCH/console.log"' EXIT
python -u scripts/prepare_functional_annotations.py
python -u scripts/prepare_semantic_run.py --source-config "$SOURCE_CONFIG" --run-dir "$LAUNCH/model_run"
CONFIG="$LAUNCH/model_run/config.json"
python -u -m factorbridge baselines --config "$CONFIG"
# New card schema and longer sequence need their own actual GPU memory/reload gate.
# This two-step check is automatic; no repeated manual test suite is launched here.
python -u -m factorbridge smoke --config "$CONFIG"
python -u -m factorbridge train --config "$CONFIG"
python -u -m factorbridge evaluate --config "$CONFIG" --split validation \
  --methods pca_raw loading_refit stability non_llm semantic_prior qwen_frozen qwen_finetuned qwen_no_semantics qwen_shuffled_text
python -u -m factorbridge programmes --config "$CONFIG" --split train
python -u -m factorbridge programmes --config "$CONFIG" --split validation
echo "PIPELINE_COMPLETED=$LAUNCH/model_run"
echo "Final internal killifish test untouched. Read evaluation_validation/report.json before any performance claim."
