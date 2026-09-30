"""Resume only tokenizer checks after a network failure; never rerun biology or train."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from factorbridge.io import read_json, read_jsonl, write_json, sha256, digest, code_hash, environment, lock_environment
from factorbridge.llm import prepared
from factorbridge.pretraining import tokenizer_check, readiness


def resumable(run):
    events = run / "pretraining_commands.jsonl"
    if not events.is_file():
        return False
    rows = read_jsonl(events)
    failed = any(r["step"] == "tokenizer_length_and_completion_mask_check" and r["status"] == "failed" for r in rows)
    stopped = run / "STOP_BEFORE_TRAINING.json"
    if stopped.exists() and read_json(stopped).get("pretraining_checks") == "passed":
        return False
    skipped = (run / "pretraining_report.json").exists() and read_json(run / "pretraining_report.json")["tokenizer"]["status"] == "explicitly_skipped"
    return failed or skipped


def resume(run, tokenizer_dir):
    run = Path(run).resolve()
    if not resumable(run):
        raise ValueError("Run must have a failed tokenizer step or explicitly skipped tokenizer; completed runs stay unchanged")
    c = read_json(run / "config.json")
    if Path(c["run_dir"]).resolve() != run or c["stage"] != "noise_recovery" or not c.get("defer_public_test"):
        raise ValueError("Only the original deferred-test Stage 1 preparation can be resumed")
    if any((run / p).exists() for p in ("training", "smoke", "frozen_protocol.json", "test_prepared", "internal_prepared", "evaluation_test")):
        raise ValueError("Run has progressed beyond pretraining; refusing to alter it")
    root, _ = prepared(c)  # Validate every prepared artifact and the exact original config.
    if {r["split"] for r in read_jsonl(root / "private/lineage.jsonl")} != {"train", "validation"}:
        raise ValueError("Only train/validation lineage may be resumed")
    evaluation = read_json(run / "evaluation_validation/report.json")
    methods = {"pca_raw", "loading_refit", "stability", "non_llm"}
    if evaluation["config_hash"] != digest(c) or set(evaluation["executed_methods"]) != methods:
        raise ValueError("Original numerical validation is missing or incompatible")
    quality = read_json(run / "label_quality.json")
    if quality["train"]["positive_gene_targets"] < 1:
        raise ValueError("No positive training gene supervision")
    for name in ("non_llm.json", "audit.json", "environment.json"):
        if not (run / name).is_file():
            raise FileNotFoundError(run / name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    attempt = run / ("tokenizer_resume_" + stamp)
    attempt.mkdir()
    for name in ("PREPARATION_FAILED.json", "pretraining_report.json", "STOP_BEFORE_TRAINING.json", "tokenizer_check.json"):
        if (run / name).exists():
            shutil.copyfile(run / name, attempt / ("previous_" + name))
    event = {"started_utc": datetime.now(timezone.utc).isoformat(), "operation": "offline_tokenizer_only",
             "numerical_code_hash": evaluation["code_hash"], "resume_code_hash": code_hash(),
             "config_sha256": sha256(run / "config.json"), "qwen_training_executed": False}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        token_status = tokenizer_check(c, local_dir=tokenizer_dir)
        report = readiness(c, quality, token_status)
        report.update(resumed_tokenizer_only=True, numerical_code_hash=evaluation["code_hash"],
                      previous_failure_is_historical=(run / "PREPARATION_FAILED.json").exists())
        write_json(run / "pretraining_report.json", report)
        write_json(attempt / "environment.json", environment())
        lock_environment(attempt / "requirements.resolved.txt")
        write_json(run / "STOP_BEFORE_TRAINING.json", {"status": "stopped_as_requested", "qwen_training_executed": False,
                   "non_llm_baseline_fitted_on_train": True, "config": str(run / "config.json"),
                   "code_hash": code_hash(), "pretraining_checks": report["pretraining_checks"], "resumed_tokenizer_only": True})
        event["status"] = "completed"
        return {"run_dir": str(run), "report": str(run / "pretraining_report.json"),
                "pretraining_checks": report["pretraining_checks"], "qwen_training_executed": False}
    except Exception as exc:
        event.update(status="failed", error=str(exc))
        raise
    finally:
        event["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(attempt / "result.json", event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--tokenizer-dir", required=True)
    args = parser.parse_args()
    if args.run_dir:
        run = Path(args.run_dir)
    else:
        choices = [p for p in sorted(Path("runs").glob("pretraining_*")) if p.is_dir() and resumable(p)]
        if len(choices) != 1:
            parser.error("Provide --run-dir explicitly; eligible runs: " + str([str(p) for p in choices]))
        run = choices[0]
    print(json.dumps(resume(run, args.tokenizer_dir), indent=2))


if __name__ == "__main__":
    main()
