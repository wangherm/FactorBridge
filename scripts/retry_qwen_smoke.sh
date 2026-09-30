#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
CONFIG="${1:?Provide the original failed run config.json path}"
source .venv/bin/activate
python scripts/archive_failed_smoke.py --config "$CONFIG"
bash scripts/smoke_qwen_offline.sh "$CONFIG"
