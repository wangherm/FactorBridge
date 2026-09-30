"""Run after pip install -e .; download the pinned GEO public numerical pilot."""
import argparse
import json
from pathlib import Path

from factorbridge.io import read_json, write_json
from factorbridge.public_data import fetch_gse124109


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="data/public/GSE124109")
    parser.add_argument("--config", default="configs/public_gse124109.local.json")
    parser.add_argument("--run-dir", default="runs/public_gse124109")
    args = parser.parse_args()
    config_path = Path(args.config)
    template = Path(__file__).resolve().parents[1] / "configs/stage1.json"
    c = dict(read_json(template), pilot_only=True, manifest=str(Path(args.destination) / "manifest.json"), run_dir=args.run_dir)
    if config_path.exists() and read_json(config_path) != c:
        raise FileExistsError("Existing pilot config differs; use a new --config and --run-dir")
    manifest_path = fetch_gse124109(args.destination)
    write_json(config_path, c)
    print(json.dumps({"manifest": str(manifest_path), "config": str(config_path),
                      "scope": "Single public study; numerical diagnostics only. Qwen training and independent evaluation blocked."}))


if __name__ == "__main__":
    main()
