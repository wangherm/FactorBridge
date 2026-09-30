from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import importlib.metadata


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    return digest({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def config(path):
    c = read_json(path)
    if c["stage"] not in {"noise_recovery", "cross_species_identity"}:
        raise ValueError("Only Stage 1 and explicitly requested Stage 2 are implemented")
    if c.get("reference_mode") not in {None, "legacy_sparse_v1", "dense_identity_sparse_refit_v2"}:
        raise ValueError("Unknown reference mode")
    if c.get("reference_mode") == "dense_identity_sparse_refit_v2" and not 0 <= c.get("reference_gene_recurrence", -1) <= 1:
        raise ValueError("Explicit reference_gene_recurrence in [0,1] required")
    if c["rank"] < 1 or c["rank"] > 12:
        raise ValueError("rank must be in 1..12 for this small-rank pilot")
    if not 0 < c["local_holdout_fraction"] < 0.5:
        raise ValueError("local_holdout_fraction must be in (0, 0.5)")
    if not 0 <= c["missing_fraction"] < 1:
        raise ValueError("missing_fraction must be in [0, 1)")
    if c["min_support"] < 2 or c["card_genes"] < c["min_support"]:
        raise ValueError("Invalid gene-support sizes")
    if c["packing"] or not c["completion_only_loss"]:
        raise ValueError("Only unpacked completion-only SFT is supported")
    if min(c["epochs"], c["smoke_steps"], c["micro_batch_size"], c["gradient_accumulation_steps"], c.get("preprocessing_workers", 1)) < 1:
        raise ValueError("Training step/batch/worker counts must be positive")
    if c["learning_rate"] <= 0 or c.get("distillation_lambda", 0) < 0:
        raise ValueError("Invalid learning rate/distillation weight")
    return c


def environment():
    packages = {}
    for name in ["numpy", "torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface-hub", "tokenizers", "jinja2", "xlrd"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.version, "executable": sys.executable, "packages": packages}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        result["nvidia_smi"] = {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        result["nvidia_smi"] = {"error": str(e)}
    if packages["torch"]:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                      gpu=[{"name": torch.cuda.get_device_name(i), "total_memory": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())])
    return result


def lock_environment(path):
    Path(path).write_text("\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())) + "\n", encoding="utf-8")
