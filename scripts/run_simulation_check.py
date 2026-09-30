"""One explicit CPU numerical engineering run. Does not load/train any LLM."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factorbridge.io import read_json, write_json, environment
from factorbridge.benchmark import simulate, prepare
from factorbridge.data import audit
from factorbridge.evaluate import baselines, evaluate, freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    if destination.exists(): raise FileExistsError("Use a fresh destination")
    destination.mkdir(parents=True)
    started = time.monotonic()
    c = read_json(Path(__file__).resolve().parents[1] / "configs/stage1.json")
    c.update(manifest=str(simulate(destination / "simulation", studies=18)), run_dir=str(destination / "run"), bootstrap_repeats=8)
    write_json(destination / "config.json", c)
    write_json(destination / "environment.json", environment())
    audit(c)
    prepare(c)
    baselines(c)
    evaluate(c, "validation")
    freeze(c, baseline_only=True)
    result = evaluate(c, "test")
    write_json(destination / "status.json", {"status": "numerical_engineering_check_passed", "data": "controlled_simulation",
               "qwen_loaded": False, "qwen_trained": False, "real_data_evaluated": False, "seconds": time.monotonic() - started,
               "executed_methods": result["executed_methods"], "not_executed_methods": result["not_executed_methods"]})
    print(destination / "status.json")


if __name__ == "__main__": main()
