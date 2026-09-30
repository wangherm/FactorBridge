#!/usr/bin/env bash
# Run from JupyterLab Terminal: bash scripts/run_public_pilot.sh
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
test -f pyproject.toml
python scripts/download_public_pilot.py
config=configs/public_gse124109.local.json
python -m factorbridge environment --config "$config"
python -m factorbridge audit --config "$config"
python -m factorbridge prepare --config "$config"
python -m factorbridge supervision-audit --config "$config"
python -m factorbridge evaluate --config "$config" --split train --methods pca_raw loading_refit stability
echo 'non_llm is not run in this pilot: the audited default labels provide no supervised gene observations. See supervision_audit.json.'
echo 'Public numerical pilot finished. No Qwen training or independent test was performed.'
