from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

from .io import config, read_json, write_json, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description="FactorBridge recovery and cross-species training; no missing-data or CPU-Qwen fallback")
    parser.add_argument("command", choices=["environment", "audit", "supervision-audit", "simulate", "prepare", "prepare-internal", "baselines", "pin-model", "smoke", "train", "evaluate", "freeze", "import-protein-tokens", "export-factor-teacher", "regression", "ablation-configs", "select-experiment"])
    parser.add_argument("--config", default="configs/stage1.json")
    parser.add_argument("--destination", default="data/controlled_simulation")
    parser.add_argument("--studies", type=int, default=18)
    parser.add_argument("--split", choices=["train", "validation", "test", "internal_test"], default="validation")
    parser.add_argument("--source")
    parser.add_argument("--species")
    parser.add_argument("--revision")
    parser.add_argument("--provenance")
    parser.add_argument("--factor-cards")
    parser.add_argument("--experiments", nargs="+")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args(argv)
    c = config(args.config)
    run = Path(c["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    event = {"command": args.command, "arguments": vars(args), "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
    try:
        if c.get("pilot_only") and args.command in {"smoke", "train", "freeze", "prepare-internal"}:
            raise ValueError("Single-study numerical pilot only: independent studies required before training or test freezing")
        if args.command == "export-factor-teacher":
            from .distill import export_factor_vectors
            if not all([args.source, args.provenance, args.factor_cards]):
                raise ValueError("--source, --provenance, --factor-cards required")
            export_factor_vectors(args.source, args.provenance, args.factor_cards, args.destination)
            result = {"exported": args.destination}
        elif args.command == "select-experiment":
            from .distill import select_experiment
            if not args.experiments: raise ValueError("--experiments required")
            result = select_experiment(args.experiments, args.destination)
        elif args.command == "ablation-configs":
            if c["stage"] != "cross_species_identity": raise ValueError("Use Stage 2 config")
            destination = Path(args.destination)
            if destination.exists(): raise FileExistsError(destination)
            weights = c["distillation_weights"]
            variants = {"supervised_only": {}, **{k: {k: w} for k, w in weights.items()}, "multi_teacher": weights}
            for name, selected in variants.items():
                variant = dict(c, distillation_weights=selected, run_dir=c["run_dir"] + "_" + name)
                write_json(destination / f"{name}.json", variant)
            result = {"variants": list(variants), "training_executed": False}
        elif args.command == "import-protein-tokens":
            from .cross_species import import_protein_tokens
            if not all([args.source, args.species, args.revision]):
                raise ValueError("--source, --species, --revision required")
            import_protein_tokens(args.source, args.destination, args.species, args.revision)
            result = {"converted": args.destination}
        elif c["stage"] == "cross_species_identity" and args.command in {"prepare", "evaluate", "freeze", "baselines", "regression"}:
            from .cross_species import prepare_pairs, evaluate_pairs, freeze_pairs, train_pair_baseline, regression
            if args.command == "prepare": result = {"prepared_path": str(prepare_pairs(c))}
            elif args.command == "evaluate": result = evaluate_pairs(c, args.split, args.methods)
            elif args.command == "baselines": result = train_pair_baseline(c)
            elif args.command == "regression": result = regression(c)
            else:
                freeze_pairs(c)
                result = {"status": "pair_protocol_frozen"}
        elif c["stage"] == "cross_species_identity" and args.command in {"audit", "prepare-internal", "simulate"}:
            raise ValueError("Use Stage 1 configuration for dataset audit, numerical baselines, simulation and internal recovery")
        elif args.command == "environment":
            result = environment()
            write_json(run / "environment.json", result)
        elif args.command == "regression":
            raise ValueError("Use Stage 2 config for the Stage 1 recovery regression")
        elif args.command == "pin-model":
            from huggingface_hub import HfApi
            c["model_revision"] = HfApi().model_info(c["model_name"], revision=c.get("model_revision") or "main").sha
            write_json(args.config, c)
            result = {"model_name": c["model_name"], "model_revision": c["model_revision"]}
        elif args.command == "audit":
            from .data import audit
            result = audit(c)
        elif args.command == "supervision-audit":
            if c["stage"] != "noise_recovery":
                raise ValueError("Use Stage 1 config for gene supervision audit")
            from .benchmark import supervision_audit
            result = supervision_audit(c)
        elif args.command == "simulate":
            from .benchmark import simulate
            result = {"manifest": str(simulate(args.destination, c["seed"], args.studies)), "source_kind": "controlled_simulation"}
        elif args.command in {"prepare", "prepare-internal"}:
            from .benchmark import prepare
            if args.command == "prepare-internal":
                from .evaluate import check_frozen
                check_frozen(c, args.methods or [])
            result = {"prepared_path": str(prepare(c, internal=args.command == "prepare-internal"))}
        elif args.command == "baselines":
            from .evaluate import baselines
            result = baselines(c)
        elif args.command in {"smoke", "train"}:
            from .llm import smoke, train
            (smoke if args.command == "smoke" else train)(c)
            result = {"status": "executed", "command": args.command}
        elif args.command == "evaluate":
            from .evaluate import evaluate
            result = evaluate(c, args.split, args.methods)
        else:
            from .evaluate import freeze
            freeze(c, args.baseline_only)
            result = {"status": "protocol_frozen", "baseline_only": args.baseline_only}
        event["status"] = "completed"
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as exc:
        event.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        print(f"FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        if args.command in {"smoke", "train"}:
            status = run / ("smoke" if args.command == "smoke" else "training") / "status.json"
            if status.exists() and read_json(status).get("status") == "running":
                write_json(status, {**read_json(status), "status": "failed", "error": str(exc)})
        raise SystemExit(1)
    finally:
        event["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with (run / "commands.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
