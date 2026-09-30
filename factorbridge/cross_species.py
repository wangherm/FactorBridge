"""Stage 2, explicitly added by user: fixed ESM2/UCE-token evidence, independent labels.

This consumes ESM2 protein embeddings of the type used as UCE gene tokens. It does
NOT claim to execute or distil contextual UCE cell embeddings from bulk matrices.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import re
import numpy as np

from .io import read_json, read_jsonl, write_json, write_jsonl, digest, sha256, code_hash

RELATIONS = {"shared", "partially_shared", "unmatched", "uncertain"}
PAIR_METHODS = ["orthology_only", "protein_only", "non_llm", "qwen_frozen", "qwen_finetuned"]
PAIR_SYSTEM = ("Compare two numerically recovered factors from different species. Return exactly JSON fields relation "
               "(shared, partially_shared, unmatched, uncertain), supported_links, evidence_ids, limitations. "
               "supported_links may contain only supplied orthology link IDs. Embedding similarity is evidence, "
               "not biological truth. Allow uncertain. Missing orthologs do not mean absence. "
               "Loading sign flips are not biological reversal. Do not invent genes or pathway names.")


def validate_pair_card(card):
    from .contracts import validate_card
    expected = {"task", "factor_a", "factor_b", "orthology_links", "protein_embedding_evidence", "available_evidence_ids"}
    if set(card) != expected:
        raise ValueError("Invalid cross-species evidence fields")
    for side in ("factor_a", "factor_b"):
        if card[side].get("task") != "recover_noisy_factor":
            raise ValueError("Nested pair tasks forbidden")
        validate_card(card[side])
    a, b = card["factor_a"], card["factor_b"]
    if a["species"] == b["species"]:
        raise ValueError("Cross-species pair needs different species")
    if (a["assay"], a["resolution"]) != (b["assay"], b["resolution"]):
        raise ValueError("Stage 2 controls assay/resolution; Stage 3 not implemented")
    seen = set()
    for link in card["orthology_links"]:
        if set(link) != {"link_id", "slot_a", "slot_b", "relation"} or link["relation"] != "one_to_one":
            raise ValueError("Only explicitly versioned one-to-one orthology supported")
        if link["link_id"] in seen:
            raise ValueError("Duplicate link ID")
        seen.add(link["link_id"])
        if link["slot_a"] not in {g["slot_id"] for g in a["genes"]} or link["slot_b"] not in {g["slot_id"] for g in b["genes"]}:
            raise ValueError("Link refers to unobserved gene")
    e = card["protein_embedding_evidence"]
    if set(e) != {"evidence_id", "model_family", "absolute_loading_weighted_cosine", "coverage_a", "coverage_b"} or e["model_family"] != "ESM2_protein_tokens":
        raise ValueError("Invalid fixed protein evidence")
    for key in ("absolute_loading_weighted_cosine", "coverage_a", "coverage_b"):
        if not isinstance(e[key], (float, int)) or not np.isfinite(e[key]):
            raise ValueError("Nonfinite embedding evidence")
    if not -1.00001 <= e["absolute_loading_weighted_cosine"] <= 1.00001 or not 0 <= e["coverage_a"] <= 1 or not 0 <= e["coverage_b"] <= 1:
        raise ValueError("Embedding evidence outside range")
    if card["available_evidence_ids"] != ["orthology", "protein"]:
        raise ValueError("Unknown evidence ID")


def validate_relation(output, card):
    validate_pair_card(card)
    if not isinstance(output, dict) or set(output) != {"relation", "supported_links", "evidence_ids", "limitations"} or output["relation"] not in RELATIONS:
        raise ValueError("Invalid relation JSON")
    for key in ("supported_links", "evidence_ids", "limitations"):
        if not isinstance(output[key], list) or any(not isinstance(s, str) for s in output[key]) or len(output[key]) != len(set(output[key])):
            raise ValueError("Invalid relation list")
    if not set(output["supported_links"]) <= {l["link_id"] for l in card["orthology_links"]}:
        raise ValueError("Invented orthology link")
    if not set(output["evidence_ids"]) <= set(card["available_evidence_ids"]):
        raise ValueError("Invented relation evidence")
    return output


def import_protein_tokens(source, destination, species, revision):
    """Convert official ESM2 gene->tensor token dictionary with weights_only loading."""
    import torch
    if not revision or not species:
        raise ValueError("Supply actual source revision and species")
    data = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not data or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in data.items()):
        raise ValueError("Expected a gene-symbol -> ESM2 tensor dictionary")
    genes = sorted(data)
    vectors = np.stack([data[g].detach().float().numpy().reshape(-1) for g in genes])
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Invalid protein vectors")
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, genes=np.array(genes), embeddings=vectors)
    write_json(str(dest) + ".provenance.json", {"source_sha256": sha256(source), "revision": revision, "species": species,
               "embedding_sha256": sha256(dest), "model_family": "ESM2_protein_tokens", "contextual_UCE_executed": False})


def initial_adapter(c):
    from .llm import signature
    stage1 = read_json(c["stage1_config"])
    state = read_json(Path(stage1["run_dir"]) / "training/status.json")
    if state.get("status") != "completed" or state["signature"] != signature(stage1):
        raise ValueError("Stage 2 requires a verified completed Stage 1 adapter")
    for key in ("model_name", "model_revision", "lora_r", "lora_alpha", "lora_target_modules"):
        if c[key] != stage1[key]:
            raise ValueError(f"Stage 1/2 backbone or adapter mismatch: {key}")
    return Path(stage1["run_dir"]) / "training/adapter"


def pooled_vector(factor, vectors):
    present = [g for g in factor["genes"] if g["feature_id"] in vectors]
    if not present:
        raise ValueError("No protein-embedding coverage: cannot fabricate zero embedding")
    x = np.stack([vectors[g["feature_id"]] for g in present])
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    weights = np.abs([g["signed_loading"] for g in present])
    result = weights @ x
    if np.linalg.norm(result) < 1e-12:
        raise ValueError("Undefined pooled protein embedding")
    return result / np.linalg.norm(result), len(present) / len(factor["genes"])


def pair_card(a, b, orthology, vectors_a, vectors_b):
    slot_a = {g["feature_id"]: g["slot_id"] for g in a["genes"]}
    slot_b = {g["feature_id"]: g["slot_id"] for g in b["genes"]}
    links = []
    for ga, gb in orthology:
        if ga in slot_a and gb in slot_b:
            links.append({"link_id": f"l{len(links)+1:03d}", "slot_a": slot_a[ga], "slot_b": slot_b[gb], "relation": "one_to_one"})
    va, ca = pooled_vector(a, vectors_a)
    vb, cb = pooled_vector(b, vectors_b)
    if va.shape != vb.shape:
        raise ValueError("Protein embedding models/dimensions incompatible")
    result = {"task": "cross_species_identity", "factor_a": a, "factor_b": b, "orthology_links": links,
              "protein_embedding_evidence": {"evidence_id": "protein", "model_family": "ESM2_protein_tokens",
                                             "absolute_loading_weighted_cosine": round(float(va @ vb), 5), "coverage_a": ca, "coverage_b": cb},
              "available_evidence_ids": ["orthology", "protein"]}
    validate_pair_card(result)
    return result


def recovered_card(row, directory, limit):
    """Read genuine numerical W from an emitted Factor Card, not a pathway label."""
    original = row["noisy_evidence"]
    with np.load(Path(directory) / row["artifact"], allow_pickle=False) as a:
        w = a["W"][:, row["column"]]
        genes, measured = a["genes"].astype(str), a["measured_mask"]
    chosen = [int(i) for i in np.argsort(-np.abs(w), kind="stable") if abs(w[i]) > 1e-12 and measured[i]][:limit]
    if len(chosen) < 2:
        raise ValueError("Recovered factor has too few measured support genes")
    rec = {g["feature_id"]: g["noisy_view_recurrence"] for g in original["genes"]}
    # Only evidence that had a recorded recurrence can be used; no invented recurrence.
    chosen = [i for i in chosen if genes[i] in rec]
    if len(chosen) < 2:
        raise ValueError("Insufficient recorded recurrence for recovered support")
    result = dict(original)
    result["genes"] = [{"slot_id": f"g{j+1:03d}", "feature_id": str(genes[i]), "loading_rank": j + 1,
                        "signed_loading": round(float(w[i]), 5), "noisy_view_recurrence": rec[genes[i]], "measured": True} for j, i in enumerate(chosen)]
    return result


def prepare_pairs(c):
    from .llm import prepared
    initial_adapter(c)
    path = Path(c["cross_species_manifest"]).resolve()
    specification = read_json(path)
    root = Path(c["run_dir"]) / "prepared"
    if root.exists():
        raise FileExistsError(root)
    def resolve(s):
        p = Path(s)
        return p if p.is_absolute() else path.parent / p
    factors, artifacts = {}, {str(path): sha256(path)}
    for file in specification["factor_card_files"]:
        p = resolve(file["path"] if isinstance(file, dict) else file)
        prefix = file.get("prefix", "") if isinstance(file, dict) else ""
        artifacts[str(p)] = sha256(p)
        for row in read_jsonl(p):
            key = prefix + row["factor_id"]
            if key in factors:
                raise ValueError("Duplicate factor ID across input files; assign explicit unique prefixes upstream")
            artifact = p.parent / row["artifact"]
            artifacts[str(artifact)] = sha256(artifact)
            factors[key] = (row, recovered_card(row, p.parent, c["pair_card_genes"]))
    vectors, revisions = {}, set()
    for entry in specification["protein_embeddings"]:
        p = resolve(entry["path"])
        provenance = read_json(str(p) + ".provenance.json")
        if provenance["species"] != entry["species"] or provenance["embedding_sha256"] != sha256(p) or provenance["model_family"] != "ESM2_protein_tokens":
            raise ValueError("Protein source provenance mismatch")
        revisions.add(provenance["revision"])
        artifacts[str(p)], artifacts[str(p) + ".provenance.json"] = sha256(p), sha256(str(p) + ".provenance.json")
        with np.load(p, allow_pickle=False) as a:
            if len(set(a["genes"].astype(str))) != len(a["genes"]) or not np.isfinite(a["embeddings"]).all():
                raise ValueError("Invalid imported protein embeddings")
            vectors[entry["species"]] = dict(zip(a["genes"].astype(str), a["embeddings"].astype(float)))
    if len(revisions) != 1:
        raise ValueError("Use the same ESM2 revision/representation space for all species")
    teacher_vectors, teacher_revisions = {}, {}
    for entry in specification.get("factor_embeddings", []):
        p = resolve(entry["path"])
        teacher = entry["teacher_id"]
        if teacher == "esm2": raise ValueError("esm2 is reserved for protein-token pooling")
        provenance = read_json(str(p) + ".provenance.json")
        if provenance["factor_vectors_sha256"] != sha256(p): raise ValueError("Teacher vector hash mismatch")
        rev = (provenance["model_name"], provenance["model_revision"], provenance["checkpoint_sha256"])
        if teacher in teacher_revisions and teacher_revisions[teacher] != rev:
            raise ValueError("Incompatible checkpoints for one teacher embedding space")
        teacher_revisions[teacher] = rev
        artifacts[str(p)], artifacts[str(p) + ".provenance.json"] = sha256(p), sha256(str(p) + ".provenance.json")
        with np.load(p, allow_pickle=False) as data:
            for fid, vector in zip(data["factor_ids"].astype(str), data["embeddings"].astype(float)):
                key = entry.get("prefix", "") + fid
                if key not in factors or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    raise ValueError("Teacher factor IDs/values invalid")
                if provenance["source_matrix_sha256"] != factors[key][0]["provenance"]["noisy_matrix_sha256"]:
                    raise ValueError("Teacher factor vector does not come from matching noisy input")
                if key in teacher_vectors.setdefault(teacher, {}): raise ValueError("Duplicate teacher factor ID")
                teacher_vectors[teacher][key] = vector / np.linalg.norm(vector)
    orth = specification["orthology"]
    if not orth.get("version"):
        raise ValueError("Orthology database version required")
    op = resolve(orth["path"])
    artifacts[str(op)] = sha256(op)
    with op.open(encoding="utf-8-sig", newline="") as f:
        orth_rows = list(csv.DictReader(f))
    pair_path = resolve(specification["pairs_path"])
    label_path = resolve(specification["labels_path"])
    artifacts[str(pair_path)], artifacts[str(label_path)] = sha256(pair_path), sha256(label_path)
    pairs = read_jsonl(pair_path)
    labels = {r["pair_id"]: r for r in read_jsonl(label_path)}
    cards, targets, lineage, owners = [], [], [], {}
    def own(kind, key, split):
        if (kind, key) in owners and owners[kind, key] != split:
            raise ValueError(f"Cross-species split leakage: {kind} {key}")
        owners[kind, key] = split
    for i, pair in enumerate(pairs):
        split = pair["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("Stage 2 internal killifish remains excluded")
        ra, a = factors[pair["factor_a"]]
        rb, b = factors[pair["factor_b"]]
        for r, f in [(ra, a), (rb, b)]:
            provenance = r["provenance"]
            if provenance["split"] == "internal_test" or provenance["source_kind"] == "internal_real":
                raise ValueError("Internal data must remain frozen; no Stage 2 training access")
            if provenance["split"] != split:
                raise ValueError("Stage 1/2 split mismatch")
            for kind, key in [("species", f["species"]), ("study", provenance["study_id"]), ("dataset", provenance["parent_dataset"])]:
                own(kind, key, split)
            for factor in provenance["parent_factor"]:
                own("parent_factor", factor, split)
            for unit in provenance["biological_units"]:
                own("unit", unit, split)
        own("programme_family", pair["programme_family"], split)
        mapping = []
        for r in orth_rows:
            if (r["species_a"], r["species_b"]) == (a["species"], b["species"]):
                if r["relation"] == "one_to_one": mapping.append((r["gene_a"], r["gene_b"]))
            elif (r["species_b"], r["species_a"]) == (a["species"], b["species"]):
                if r["relation"] == "one_to_one": mapping.append((r["gene_b"], r["gene_a"]))
        if len({x for x, y in mapping}) != len(mapping) or len({y for x, y in mapping}) != len(mapping):
            raise ValueError("Declared one-to-one mapping is not one-to-one")
        evidence = pair_card(a, b, mapping, vectors[a["species"]], vectors[b["species"]])
        label = labels[pair["pair_id"]]
        if label["label_source"] not in {"controlled_truth", "curated_weak_reference"} or not label.get("source_reference"):
            raise ValueError("Independent label source required; teacher similarity/Qwen answers cannot be gold")
        target = label["target"]
        validate_relation(target, evidence)
        eid = f"pair_{i:06d}"
        cards.append({"example_id": eid, "card": evidence})
        distillation = {}
        for teacher in c.get("distillation_weights", {}):
            if teacher == "esm2":
                distillation[teacher] = evidence["protein_embedding_evidence"]["absolute_loading_weighted_cosine"]
            else:
                if teacher not in teacher_vectors or any(pair[k] not in teacher_vectors[teacher] for k in ("factor_a", "factor_b")):
                    raise ValueError(f"Missing requested teacher {teacher}; no fallback")
                va, vb = [teacher_vectors[teacher][pair[k]] for k in ("factor_a", "factor_b")]
                if va.shape != vb.shape: raise ValueError("Incompatible teacher factor dimensions")
                distillation[teacher] = abs(float(va @ vb))
        targets.append({"example_id": eid, "target": target, "label_source": label["label_source"], "distillation_targets": distillation})
        lineage.append({"example_id": eid, "split": split, "pair_id": pair["pair_id"], "species": [a["species"], b["species"]],
                        "programme_family": pair["programme_family"], "source_reference": label["source_reference"]})
    if {r["split"] for r in lineage} != {"train", "validation", "test"}:
        raise ValueError("Species-disjoint train/validation/test pairs required")
    # Mandatory recovery replay, rejecting conflicts with held-out species/studies.
    stage1 = read_json(c["stage1_config"])
    source, _ = prepared(stage1)
    prior_cards = {r["example_id"]: r for r in read_jsonl(source / "cards.jsonl")}
    prior_labels = {r["example_id"]: r for r in read_jsonl(source / "private/labels.jsonl")}
    replay = []
    for row in read_jsonl(source / "private/lineage.jsonl"):
        if row["split"] not in {"train", "validation"}: continue
        card = prior_cards[row["example_id"]]
        species = card["card"]["species"]
        if owners.get(("species", species), "train") != "train":
            raise ValueError("Stage 1 replay species overlaps Stage 2 held-out species")
        owners["species", species] = "train"
        own("study", row["study_id"], row["split"])
        eid = "replay_" + row["example_id"]
        replay.append((dict(card, example_id=eid), dict(prior_labels[row["example_id"]], example_id=eid), dict(row, example_id=eid)))
    for card, target, row in replay:
        cards.append(card); targets.append(target); lineage.append(row)
    root.mkdir(parents=True)
    write_jsonl(root / "cards.jsonl", cards)
    write_jsonl(root / "private/labels.jsonl", targets)
    write_jsonl(root / "private/lineage.jsonl", lineage)
    write_json(root / "source_hashes.json", artifacts)
    files = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}
    write_json(root / "index.json", {"complete": True, "config_hash": digest(c), "files": files, "stage": "cross_species_identity",
               "examples": len(cards), "recovery_replay_examples": len(replay), "source_kinds": sorted({r["label_source"] for r in targets}),
               "teacher": "fixed ESM2 tokens and verified contextual factor vectors; no teacher-generated biological labels",
               "teacher_revisions": teacher_revisions, "distillation_weights": c.get("distillation_weights", {})})
    return root


def evaluate_pairs(c, split="validation", methods=None):
    from .llm import prepared, infer
    root, index = prepared(c)
    run = Path(c["run_dir"])
    if split not in {"validation", "test"}:
        raise ValueError("Pair evaluation supports public validation/test only")
    if split == "test":
        protocol = read_json(run / "pair_frozen_protocol.json")
        if protocol["config_hash"] != digest(c) or protocol["code_hash"] != code_hash():
            raise ValueError("Pair evaluation protocol changed after freeze")
        for path, expected in protocol["files"].items():
            if sha256(run / path) != expected: raise ValueError("Pair frozen artifact changed")
    teacher_methods = ["teacher:" + t for t in c.get("distillation_weights", {})]
    methods = methods or PAIR_METHODS + teacher_methods
    if not set(methods) <= set(PAIR_METHODS + teacher_methods):
        raise ValueError("Unknown pair method")
    out = run / f"pairs_{split}"
    if out.exists(): raise FileExistsError(out)
    all_cards = {r["example_id"]: r for r in read_jsonl(root / "cards.jsonl")}
    rows = [r for r in read_jsonl(root / "private/lineage.jsonl") if r["split"] == split and "pair_id" in r]
    inputs = [all_cards[r["example_id"]] for r in rows]
    labels = {r["example_id"]: r for r in read_jsonl(root / "private/labels.jsonl")}
    # Inference is completed before truth is read.
    learned = read_json(run / "pair_non_llm.json") if "non_llm" in methods else None
    if learned and learned["config_hash"] != digest(c): raise ValueError("Pair non-LLM model stale")
    preds = {}
    for m in methods:
        if m.startswith("qwen"):
            preds[m] = infer(c, inputs, m)
        elif m.startswith("teacher:"):
            teacher = m.split(":", 1)[1]
            preds[m] = []
            for row in inputs:
                score = labels[row["example_id"]]["distillation_targets"][teacher]
                output = {"relation": "shared" if score >= c["protein_shared_threshold"] else "uncertain", "supported_links": [], "evidence_ids": [], "limitations": []}
                preds[m].append({"example_id": row["example_id"], "valid": True, "output": output, "teacher_similarity": score})
        else:
            preds[m] = [{"example_id": row["example_id"], "valid": True, "output": numerical_relation(row["card"], m, c, learned)} for row in inputs]
    summaries = {}
    for method, values in preds.items():
        write_jsonl(out / f"{method}.jsonl", values)
        correct = sum(p["valid"] and p["output"]["relation"] == labels[p["example_id"]]["target"]["relation"] for p in values)
        summaries[method] = {"label_agreement": correct / len(values), "n_pairs": len(values),
                             "invalid_rate": sum(not p["valid"] for p in values) / len(values),
                             "uncertain_rate": sum(p["valid"] and p["output"]["relation"] == "uncertain" for p in values) / len(values)}
        committed = [p for p in values if p["valid"] and p["output"]["relation"] != "uncertain"]
        summaries[method]["coverage"] = len(committed) / len(values)
        summaries[method]["agreement_when_committed"] = sum(p["output"]["relation"] == labels[p["example_id"]]["target"]["relation"] for p in committed) / len(committed) if committed else None
        from .evaluate import ci_by_study
        families = {r["example_id"]: r["programme_family"] for r in rows}
        summaries[method]["programme_family_ci"] = ci_by_study([
            {"study_id": families[p["example_id"]], "agreement": int(p["valid"] and p["output"]["relation"] == labels[p["example_id"]]["target"]["relation"])} for p in values], "agreement", c["seed"])
        for name, grouping in [("by_relation", {r["example_id"]: labels[r["example_id"]]["target"]["relation"] for r in rows}),
                                ("by_species_pair", {r["example_id"]: " / ".join(sorted(r["species"])) for r in rows})]:
            groups = {}
            for p in values:
                groups.setdefault(grouping[p["example_id"]], []).append(int(p["valid"] and p["output"]["relation"] == labels[p["example_id"]]["target"]["relation"]))
            summaries[method][name] = {k: {"n": len(v), "agreement": float(np.mean(v))} for k, v in groups.items()}
    # Fixed numerical evidence exported for externally audited calibration, not asserted as gold.
    write_jsonl(out / "numeric_evidence.jsonl", [{"example_id": r["example_id"], "protein": r["card"]["protein_embedding_evidence"], "orthology_links": len(r["card"]["orthology_links"])} for r in inputs])
    write_json(out / "report.json", {"summary": summaries, "source_kinds": index["source_kinds"], "config_hash": digest(c),
               "cards_sha256": sha256(root / "cards.jsonl"), "lineage_sha256": sha256(root / "private/lineage.jsonl"),
               "status": "reference agreement only; not proof of transferred cross-species biology", "species": sorted({s for r in rows for s in r["species"]})})
    return summaries


def freeze_pairs(c):
    run = Path(c["run_dir"])
    report = read_json(run / "pairs_validation/report.json")
    if not set(PAIR_METHODS) <= set(report["summary"]):
        raise ValueError("All pair baselines/Qwen variants must be evaluated on validation")
    regression = run / "recovery_regression.json"
    regression_result = read_json(regression)
    if regression_result["regression_adapter"] != str(run / "training/adapter"):
        raise ValueError("Stage 1 recovery regression from current Stage 2 adapter required")
    path = run / "pair_frozen_protocol.json"
    if path.exists(): raise FileExistsError(path)
    files = [run / "pairs_validation/report.json", run / "pair_non_llm.json", run / "prepared/index.json", *[p for p in (run / "training").rglob("*") if p.is_file()]]
    write_json(path, {"config_hash": digest(c), "code_hash": code_hash(), "stage1_regression_sha256": sha256(regression), "files": {str(p.relative_to(run)): sha256(p) for p in files}})


def pair_features(card):
    validate_pair_card(card)
    evidence = card["protein_embedding_evidence"]
    fraction = len(card["orthology_links"]) / max(1, min(len(card["factor_a"]["genes"]), len(card["factor_b"]["genes"])))
    return np.array([1, fraction, evidence["absolute_loading_weighted_cosine"], evidence["coverage_a"], evidence["coverage_b"]], dtype=float)


def train_pair_baseline(c):
    from .llm import prepared
    from .selectors import fit_logistic
    root, _ = prepared(c)
    cards = {r["example_id"]: r["card"] for r in read_jsonl(root / "cards.jsonl")}
    labels = {r["example_id"]: r for r in read_jsonl(root / "private/labels.jsonl")}
    rows = [r for r in read_jsonl(root / "private/lineage.jsonl") if r["split"] == "train" and "pair_id" in r]
    x = [pair_features(cards[r["example_id"]]) for r in rows]
    y = [labels[r["example_id"]]["target"]["relation"] for r in rows]
    models = {relation: fit_logistic(x, [int(v == relation) for v in y]) for relation in sorted(RELATIONS)}
    result = {"config_hash": digest(c), "models": models, "train_pairs": len(rows)}
    write_json(Path(c["run_dir"]) / "pair_non_llm.json", result)
    return result


def numerical_relation(card, method, c, learned=None):
    from .selectors import predict_logistic
    x = pair_features(card)
    if method == "orthology_only":
        relation = "shared" if x[1] >= c["orthology_shared_fraction"] else "uncertain"
    elif method == "protein_only":
        relation = "shared" if x[2] >= c["protein_shared_threshold"] else "uncertain"
    elif method == "non_llm":
        scores = {k: float(predict_logistic(v, x[None])[0]) for k, v in learned["models"].items()}
        relation = max(scores, key=scores.get)
    else:
        raise ValueError(method)
    return {"relation": relation, "supported_links": [], "evidence_ids": [], "limitations": []}


def regression(c):
    from .llm import signature
    from .evaluate import evaluate
    state = read_json(Path(c["run_dir"]) / "training/status.json")
    if state.get("status") != "completed" or state["signature"] != signature(c):
        raise ValueError("Completed Stage 2 training required")
    stage1 = read_json(c["stage1_config"])
    result = evaluate(stage1, "validation", ["qwen_finetuned"], regression_adapter=Path(c["run_dir"]) / "training/adapter",
                      regression_output=Path(c["run_dir"]) / "recovery_regression")
    write_json(Path(c["run_dir"]) / "recovery_regression.json", result)
    return result
