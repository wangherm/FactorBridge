#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
CONFIG="${1:?Provide the completed pretraining config.json path}"
test -x .venv/bin/python
source .venv/bin/activate
LOG_DIR="runs/qwen_smoke_launcher_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$LOG_DIR"
exec > >(tee "$LOG_DIR/console.log") 2>&1

export HF_HOME="${FACTORBRIDGE_HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2

python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import torch
from factorbridge.io import read_json
from factorbridge.llm import prepared
c = read_json(sys.argv[1])
assert c['model_name'] == 'Qwen/Qwen3-4B-Instruct-2507'
assert c['model_revision'] == 'cdbee75f17c01a7cc42f958dc650907174af0554'
run = Path(c['run_dir'])
assert read_json(run / 'pretraining_report.json')['pretraining_checks'] == 'passed'
assert not any((run / p).exists() for p in ['smoke', 'training', 'frozen_protocol.json', 'test_prepared', 'internal_prepared']), 'Existing smoke/training/test state: retain it and inspect before proceeding'
prepared(c)
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'
print('Using original config and prepared data:', run, flush=True)
PY

python scripts/download_qwen_weights.py --hf-home "$HF_HOME" --workers 2
python -m factorbridge smoke --config "$CONFIG"
python - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from factorbridge.io import read_json
c = read_json(sys.argv[1])
path = Path(c['run_dir']) / 'smoke/status.json'
status = read_json(path)
assert status['status'] == 'passed'
print(path)
print(path.read_text())
print('STOP: Qwen smoke completed; formal training NOT executed.')
PY
echo "Launcher log: $LOG_DIR/console.log"
