"""Preparation/export before Qwen training. No Qwen weights, test evaluation or freeze."""
from collections import Counter
from pathlib import Path

from .contracts import messages, validate_output
from .io import read_jsonl, write_jsonl, write_json, sha256, code_hash
from .llm import prepared, training_data


def export_text(c):
    root, index = prepared(c)
    cards = {r["example_id"]: r["card"] for r in read_jsonl(root / "cards.jsonl")}
    labels = {r["example_id"]: r for r in read_jsonl(root / "private/labels.jsonl")}
    lineage = read_jsonl(root / "private/lineage.jsonl")
    if c.get("defer_public_test") and any(r["split"] not in {"train", "validation"} for r in lineage):
        raise ValueError("Deferred test leaked into development preparation")
    report = {}
    for split in ("train", "validation"):
        selected = [r for r in lineage if r["split"] == split]
        if not selected:
            raise ValueError(f"No {split} examples")
        rows, decisions, positive, sources, axes = [], Counter(), 0, Counter(), Counter()
        for sidecar in selected:
            eid = sidecar["example_id"]; evidence, label = cards[eid], labels[eid]
            target = validate_output(label["target"], evidence, c["min_support"])
            decisions[target["decision"]] += 1
            positive += len(target["supported_gene_slots"])
            sources[label["label_source"]] += 1
            axes.update(target["axis_flags"])
            import json
            rows.append({"prompt": messages(evidence), "completion": [{"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]})
        write_jsonl(Path(c["run_dir"]) / "sft_text" / f"{split}.jsonl", rows)
        report[split] = {"examples": len(rows), "decisions": dict(decisions), "positive_gene_targets": positive,
                         "axis_flags": dict(axes), "sources": dict(sources),
                         "studies": sorted({r["study_id"] for r in selected})}
    write_json(Path(c["run_dir"]) / "label_quality.json", report)
    return report


def tokenizer_check(c):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(c["model_name"], revision=c["model_revision"], trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = training_data(c, tokenizer)  # Reuses exact training-time masks and no-truncation checks.
    target = Path(c["run_dir"]) / "tokenizer"
    tokenizer.save_pretrained(target)
    result = {"status": "passed", "model_name": c["model_name"], "revision": c["model_revision"],
              "max_tokens": {s: max(len(r["input_ids"]) for r in data) for s, data in rows.items()},
              "completion_only_masks_checked": True, "model_weights_loaded": False,
              "files": {p.name: sha256(p) for p in target.iterdir() if p.is_file()}}
    write_json(Path(c["run_dir"]) / "tokenizer_check.json", result)
    return result


def readiness(c, quality, token_status):
    blockers = []
    for split, q in quality.items():
        if not q["positive_gene_targets"]:
            blockers.append(f"{split}: no supported gene targets")
        if not q["decisions"].get("retain") or not q["decisions"].get("uncertain"):
            blockers.append(f"{split}: retain/uncertain decision diversity missing")
    if token_status.get("status") != "passed":
        blockers.append("Exact tokenizer length/mask checks not completed")
    result = {"data_preparation": "completed", "text_contract": "passed", "tokenizer": token_status,
              "pretraining_checks": "blocked" if blockers else "passed", "blockers": blockers,
              "qwen_training_executed": False, "model_weights_loaded": False, "gpu_smoke": "not_executed",
              "non_llm_baseline_fitted_on_train": (Path(c["run_dir"]) / "non_llm.json").is_file(),
              "public_test_prepared": False, "public_test_evaluated": False, "internal_data_used": False,
              "adapter": None, "code_hash": code_hash(), "label_quality": quality,
              "scientific_training_readiness": "Exploratory weak-supervision experiment only; insufficient for all project objectives",
              "limitations": ["One independent study per split; few factor cards are not expanded into fictitious independent observations",
                              "Real weak labels provide no verified negative genes or technical/composition origin gold",
                              "Stable directions may encode cell line, ageing, culture batch or other biology; not causal biological factors",
                              "No model or teacher performance claim; controlled-truth supervision still needed for complete null/confounding objectives"],
              "next_gate": "Review label quality and scope; then a separate actual GPU smoke test before training"}
    write_json(Path(c["run_dir"]) / "pretraining_report.json", result)
    return result
