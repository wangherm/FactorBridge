"""Prepare public data through SFT/tokenizer checks; NEVER call smoke/train/freeze/test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from factorbridge.io import read_json, write_json, lock_environment, code_hash
from factorbridge.public_panel import fetch_public_panel
from factorbridge.pretraining import export_text, tokenizer_check, readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/public/pretraining_panel_v1")
    parser.add_argument("--run-dir")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Explicit text-only run; final report remains blocked for tokenizer checks")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = Path(args.run_dir or "runs/pretraining_" + stamp).resolve()
    if run.exists():
        raise FileExistsError("Use a new run directory; old pilot/preparation results are preserved")
    run.mkdir(parents=True)
    events = run / "pretraining_commands.jsonl"
    def record(name, action):
        event = {"step": name, "started_utc": datetime.now(timezone.utc).isoformat()}
        print(f"START {name}", flush=True)
        try:
            result = action(); event["status"] = "completed"
            return result
        except Exception as exc:
            event.update(status="failed", error=str(exc)); raise
        finally:
            event["finished_utc"] = datetime.now(timezone.utc).isoformat()
            with events.open("a", encoding="utf-8") as f: f.write(json.dumps(event) + "\n")
    try:
        manifest = record("download_and_convert_public_panel", lambda: fetch_public_panel(args.data_dir))
        c = read_json(Path(__file__).resolve().parents[1] / "configs/stage1.json")
        c.update(manifest=str(manifest), run_dir=str(run), pilot_only=False, defer_public_test=True,
                 reference_mode="dense_identity_sparse_refit_v2", reference_gene_recurrence=0.5,
                 model_revision="cdbee75f17c01a7cc42f958dc650907174af0554")
        path = run / "config.json"; write_json(path, c)
        def command(name, extra=()):
            argv = [sys.executable, "-m", "factorbridge", name, "--config", str(path), *extra]
            with (run / "console.log").open("a", encoding="utf-8") as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
        for step in ["environment", "audit", "prepare", "supervision-audit"]:
            record(step, lambda step=step: command(step))
        quality = record("export_train_validation_text", lambda: export_text(c))
        # The gene selector cannot be fitted without observed gene supervision.
        if not quality["train"]["positive_gene_targets"]:
            report = readiness(c, quality, {"status": "not_executed"})
            raise ValueError("No supervised training genes; see label_quality.json and pretraining_report.json")
        record("fit_non_llm_on_train_only", lambda: command("baselines"))
        record("numerical_validation_only", lambda: command("evaluate", ["--split", "validation", "--methods", "pca_raw", "loading_refit", "stability", "non_llm"]))
        token_status = {"status": "explicitly_skipped", "model_weights_loaded": False}
        if not args.skip_tokenizer:
            token_status = record("tokenizer_length_and_completion_mask_check", lambda: tokenizer_check(c))
        report = readiness(c, quality, token_status)
        lock_environment(run / "requirements.resolved.txt")
        write_json(run / "STOP_BEFORE_TRAINING.json", {"status": "stopped_as_requested", "qwen_training_executed": False,
                   "non_llm_baseline_fitted_on_train": True,
                   "config": str(path), "code_hash": code_hash(), "pretraining_checks": report["pretraining_checks"]})
        print(json.dumps({"run_dir": str(run), "report": str(run / "pretraining_report.json"),
                          "pretraining_checks": report["pretraining_checks"], "qwen_training_executed": False}, indent=2))
    except Exception as exc:
        write_json(run / "PREPARATION_FAILED.json", {"error": str(exc), "qwen_training_executed": False,
                   "next_step": "Inspect console.log; retain this run and use a new run directory after fixing the cause"})
        print(f"FAILED: {exc}. Logs: {run}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
