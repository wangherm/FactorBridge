"""Fixed multi-teacher auxiliary regression; independent biological labels stay SFT targets."""
from pathlib import Path
import numpy as np

from .io import read_json, read_jsonl, write_json, sha256


def teacher_keys(c):
    return list(c.get("distillation_weights", {})) if c["stage"] == "cross_species_identity" else []


def make_head(model, c, torch):
    keys = teacher_keys(c)
    if not keys:
        return None
    weights = np.array([c["distillation_weights"][k] for k in keys])
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Distillation teacher weights must be fixed positive finite values")
    return torch.nn.Linear(model.config.hidden_size, len(keys), bias=True).to(device="cuda", dtype=torch.float32)


def loss(model, inputs, rows, c, head, torch):
    output = model(**inputs, output_hidden_states=head is not None)
    supervised = output.loss
    if head is None:
        return supervised, float(supervised.detach().cpu()), 0.0
    # Last PROMPT position, before the answer: auxiliary head cannot peek at labels.
    starts = (inputs["labels"] != -100).int().argmax(dim=1) - 1
    if torch.any(starts < 0):
        raise ValueError("Distillation requires nonempty masked prompt")
    hidden = output.hidden_states[-1][torch.arange(len(rows), device="cuda"), starts].float()
    predicted = head(hidden)
    keys = teacher_keys(c)
    targets = torch.tensor([[r.get("teacher_targets", {}).get(k, 0.0) for k in keys] for r in rows], dtype=torch.float32, device="cuda")
    mask = torch.tensor([[k in r.get("teacher_targets", {}) for k in keys] for r in rows], dtype=torch.float32, device="cuda")
    weights = torch.tensor([c["distillation_weights"][k] for k in keys], dtype=torch.float32, device="cuda")
    weighted = mask * weights
    auxiliary = (((predicted - targets) ** 2) * weighted).sum() / weighted.sum().clamp(min=1)
    combined = supervised + c["distillation_lambda"] * auxiliary
    return combined, float(supervised.detach().cpu()), float(auxiliary.detach().cpu())


def save_head(head, path, c):
    if head is None: return
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(Path(path) / "distillation_head.safetensors"))
    write_json(Path(path) / "distillation.json", {"teachers": teacher_keys(c), "weights": c["distillation_weights"],
               "lambda": c["distillation_lambda"], "teacher_outputs_are_gold": False, "readout": "last_prompt_position"})


def export_factor_vectors(embedding_path, provenance_path, cards_path, destination):
    """Turn actual contextual sample embeddings into factor-associated vectors.

    External model inference must already have been performed on the corresponding
    noisy input. No bulk-to-scRNA reinterpretation or teacher model fallback occurs.
    """
    provenance = read_json(provenance_path)
    required = {"model_name", "model_revision", "checkpoint_sha256", "source_matrix_sha256", "embedding_sha256", "pretraining_species_exposure", "input_assay"}
    if any(not provenance.get(k) for k in required):
        raise ValueError("Complete teacher checkpoint/input provenance required")
    if provenance["embedding_sha256"] != sha256(embedding_path):
        raise ValueError("Teacher embedding hash mismatch")
    with np.load(embedding_path, allow_pickle=False) as a:
        samples, embeddings = a["samples"].astype(str), a["embeddings"].astype(float)
    if embeddings.ndim != 2 or len(embeddings) != len(samples) or len(set(samples)) != len(samples) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid teacher sample embeddings")
    lookup = {s: i for i, s in enumerate(samples)}
    ids, vectors = [], []
    for row in read_jsonl(cards_path):
        if row["noisy_evidence"]["assay"] != provenance["input_assay"]:
            raise ValueError("Teacher assay mismatch; scRNA teacher cannot consume bulk by relabeling")
        if row["provenance"].get("noisy_matrix_sha256") != provenance["source_matrix_sha256"]:
            raise ValueError("Teacher input must be the same noisy matrix as the factor; clean-reference teacher forbidden")
        p = Path(cards_path).parent / row["artifact"]
        with np.load(p, allow_pickle=False) as a:
            sample_ids, z = a["discovery_samples"].astype(str), a["Z_discovery"][:, row["column"]]
        if any(s not in lookup for s in sample_ids):
            raise ValueError("Teacher/sample mismatch; do not fabricate missing unit embeddings")
        e = embeddings[[lookup[s] for s in sample_ids]]
        direction = (z - z.mean()) @ (e - e.mean(axis=0))
        if np.linalg.norm(direction) < 1e-12:
            raise ValueError("Undefined factor-associated teacher vector")
        ids.append(row["factor_id"])
        vectors.append(direction / np.linalg.norm(direction))
    if not ids: raise ValueError("No factors")
    dest = Path(destination)
    if dest.exists(): raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, factor_ids=np.array(ids), embeddings=np.array(vectors))
    write_json(str(dest) + ".provenance.json", {**provenance, "factor_vectors_sha256": sha256(dest),
               "factor_cards_sha256": sha256(cards_path), "construction": "discovery-only score/embedding covariance"})


def select_experiment(paths, destination):
    """Select on public validation only; report separate teacher strengths, no routing."""
    records = []
    common = None
    for config_path in paths:
        c = read_json(config_path)
        run = Path(c["run_dir"])
        if (run / "pairs_test").exists():
            raise ValueError("Do not choose experiments after viewing their test results")
        report = read_json(run / "pairs_validation/report.json")
        from .io import digest
        if report["config_hash"] != digest(c): raise ValueError("Validation config changed")
        key = (report["cards_sha256"], report["lineage_sha256"], c["stage1_config"], c["selection_max_invalid_rate"], c["selection_max_recovery_drop"])
        if common is not None and key != common:
            raise ValueError("Ablations must use identical examples, grouping, replay and selection criteria")
        common = key
        regression = read_json(run / "recovery_regression.json")
        recovery = regression["summary"]["qwen_finetuned"]["recovery_recall"]["mean"]
        baseline = read_json(Path(read_json(c["stage1_config"])["run_dir"]) / "evaluation_validation/report.json")
        old_recovery = baseline["summary"]["qwen_finetuned"]["recovery_recall"]["mean"]
        tuned = report["summary"]["qwen_finetuned"]
        valid = tuned["invalid_rate"] <= c["selection_max_invalid_rate"] and recovery is not None and old_recovery is not None and recovery >= old_recovery - c["selection_max_recovery_drop"]
        records.append({"config": str(config_path), "teacher_weights": c["distillation_weights"], "validation_agreement": tuned["label_agreement"],
                        "recovery_recall": recovery, "stage1_recovery_recall": old_recovery, "eligible": bool(valid), "by_relation": tuned.get("by_relation"),
                        "by_species_pair": tuned.get("by_species_pair")})
    eligible = [r for r in records if r["eligible"]]
    chosen = max(eligible, key=lambda r: r["validation_agreement"]) if eligible else None
    write_json(destination, {"selected": chosen, "experiments": records, "selection_split": "validation", "test_used": False,
               "warning": "No eligible model is a valid outcome; choose none rather than hiding recovery regression"})
    return chosen
