"""Independent numerical evaluation. Oracle accesses truths ONLY while scoring."""
from __future__ import annotations

import csv
import time
from pathlib import Path
import numpy as np

from .io import read_json, read_jsonl, write_json, write_jsonl, digest, sha256, code_hash
from .llm import prepared, infer
from .selectors import train_selector, select
from .contracts import validate_output
from .factors import refit, deduplicate

METHODS = ["pca_raw", "loading_refit", "stability", "non_llm", "qwen_frozen", "qwen_finetuned"]


def assignment(similarity):
    """Exact small-rank maximum-weight matching, with unmatched references allowed."""
    # DP over true-factor subsets; no sign or identity choices feed back to recovery.
    state = {0: (0.0, [])}
    for i in range(similarity.shape[0]):
        nxt = dict(state)
        for mask, (score, pairs) in state.items():
            for j in range(similarity.shape[1]):
                if not mask & (1 << j):
                    key = mask | (1 << j)
                    value = score + float(similarity[i, j])
                    if key not in nxt or value > nxt[key][0]:
                        nxt[key] = value, pairs + [(i, j)]
        state = nxt
    return max(state.values(), key=lambda x: x[0])[1]


def cosine(a, b):
    return np.abs(a.T @ b) / np.maximum(np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)[None, :], 1e-12)


def correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return abs(float(a @ b)) / denom if denom > 1e-12 else None


def factor_metrics(w, z, truth, scores, controlled, threshold):
    sim = cosine(w, truth)
    pairs = assignment(sim)
    similarities, cors, f1s, precisions, recalls = [], [], [], [], []
    for i, j in pairs:
        similarities.append(float(sim[i, j]))
        corr = correlation(z[:, i], scores[:, j])
        if corr is not None:
            cors.append(corr)
        if controlled:
            a, b = np.abs(w[:, i]) > 1e-12, np.abs(truth[:, j]) > 1e-12
            tp = int((a & b).sum())
            precision, recall = tp / max(1, a.sum()), tp / max(1, b.sum())
            precisions.append(float(precision)); recalls.append(float(recall))
            f1s.append(float(2 * precision * recall / max(precision + recall, 1e-12)))
    ntrue = truth.shape[1]
    # Missing predictions count as zero; prevents selective reporting only on matches.
    recovery = sum(s >= threshold for s in similarities) / ntrue if ntrue else None
    subspace = None
    if w.shape[1] and ntrue:
        u, s, _ = np.linalg.svd(w, full_matrices=False)
        v, t, _ = np.linalg.svd(truth, full_matrices=False)
        u, v = u[:, s > 1e-8], v[:, t > 1e-8]
        subspace = float(np.linalg.norm(u.T @ v, "fro") ** 2 / max(1, v.shape[1]))
    return {"n_reference_biological_factors": ntrue, "n_recovered_factors": w.shape[1], "recovery_recall": recovery,
            "mean_loading_similarity_all_references": sum(similarities) / ntrue if ntrue else None,
            "score_correlation_matched_same_heldout_units": float(np.mean(cors)) if cors else None,
            "support_precision_matched": float(np.mean(precisions)) if precisions else None,
            "support_recall_all_references": sum(recalls) / ntrue if controlled and ntrue else None,
            "support_f1_all_references": sum(f1s) / ntrue if controlled and ntrue else None,
            "reference_subspace_coverage": subspace,
            "matched_pairs_scoring_only": [[int(i), int(j)] for i, j in pairs]}


def baselines(c):
    root, _ = prepared(c)
    cards = {r["example_id"]: r["card"] for r in read_jsonl(root / "cards.jsonl")}
    labels = {r["example_id"]: r for r in read_jsonl(root / "private/labels.jsonl")}
    train = [r for r in read_jsonl(root / "private/lineage.jsonl") if r["split"] == "train"]
    model = train_selector([cards[r["example_id"]] for r in train], [labels[r["example_id"]] for r in train])
    write_json(Path(c["run_dir"]) / "non_llm.json", {"config_hash": digest(c), "prepared_hash": digest(read_json(root / "index.json")), "model": model})
    return model


def freeze(c, baseline_only=False):
    if c.get("pilot_only"):
        raise ValueError("Single-study pilot has no independent validation and cannot freeze a test protocol")
    run = Path(c["run_dir"])
    prepared(c)
    report = read_json(run / "evaluation_validation/report.json")
    if report["config_hash"] != digest(c) or report["code_hash"] != code_hash():
        raise ValueError("Validation evidence stale")
    if not baseline_only and not all(m in report["executed_methods"] for m in METHODS):
        raise ValueError("Run all methods on validation before freeze, or explicitly use --baseline-only")
    path = run / "frozen_protocol.json"
    if path.exists():
        raise FileExistsError("Protocol already frozen; never rewrite it after test access")
    artifact_hashes = {}
    for p in [run / "non_llm.json", *sorted((run / "training").rglob("*"))]:
        if p.is_file():
            artifact_hashes[str(p.relative_to(run))] = sha256(p)
    data_files = {}
    if c.get("defer_public_test"):
        from .data import manifest
        data_files[str(Path(c["manifest"]).resolve())] = sha256(c["manifest"])
        for e in manifest(c["manifest"]):
            for field in ("matrix_path", "metadata_path"):
                data_files[e[field]] = sha256(e[field])
    write_json(path, {"config": c, "config_hash": digest(c), "code_hash": code_hash(), "artifacts": artifact_hashes,
               "data_files": data_files,
               "prepared_index_hash": sha256(run / "prepared/index.json"), "baseline_only": baseline_only,
               "validation_report_sha256": sha256(run / "evaluation_validation/report.json"),
               "global_rules": "frozen; internal discovery remains local and unsupervised"})


def check_frozen(c, methods):
    run = Path(c["run_dir"])
    frozen = read_json(run / "frozen_protocol.json")
    if frozen["config_hash"] != digest(c) or frozen["code_hash"] != code_hash():
        raise ValueError("Frozen configuration/code changed. Do not tune against test results.")
    if sha256(run / "prepared/index.json") != frozen["prepared_index_hash"]:
        raise ValueError("Prepared data changed after freeze")
    if sha256(run / "evaluation_validation/report.json") != frozen["validation_report_sha256"]:
        raise ValueError("Validation evidence changed after freeze")
    for p, expected in frozen["artifacts"].items():
        if sha256(run / p) != expected:
            raise ValueError(f"Frozen artifact changed: {p}")
    for p, expected in frozen.get("data_files", {}).items():
        if sha256(p) != expected:
            raise ValueError(f"Frozen data changed: {p}")
    if frozen["baseline_only"] and any(m.startswith("qwen") for m in methods):
        raise ValueError("Baseline-only protocol cannot be extended after test access")


def oracle(view, truth_w, cards, c):
    measured = np.flatnonzero(view["measured"])
    t = truth_w[measured]
    sims = np.zeros((len(cards), t.shape[1]))
    pool = np.zeros_like(sims)
    for i, evidence in enumerate(cards):
        universe = {str(g): k for k, g in enumerate(view["genes"][view["measured"]])}
        available = [universe[g["feature_id"]] for g in evidence["genes"]]
        for j in range(t.shape[1]):
            support = [k for k in available if abs(t[k, j]) > 1e-12]
            pool[i, j] = len(support) >= c["min_support"]
            if len(support) >= c["min_support"]:
                w, _, _, _ = refit(view["X"][view["discovery"]], view["X"][view["heldout"]], support, view["W"][:, i])
                sims[i, j] = cosine(w[:, None], t[:, j:j+1])[0, 0]
    n = t.shape[1]
    pairs = assignment(sims)
    return {"candidate_pool_support_recall": float((pool.max(axis=0) > 0).mean()) if n else None,
            "oracle_support_refit_recovery": sum(sims[i, j] >= c["recovery_similarity"] for i, j in pairs) / n if n else None,
            "oracle_definition": "known-support intersection with visible card genes + identical refit; reference-informed comparator, not a mathematical upper bound over all subsets",
            "raw_candidate_loading_recall": float((cosine(view["W"], t).max(axis=0) >= c["recovery_similarity"]).mean()) if n else None}


def ci_by_study(rows, metric, seed):
    groups = {}
    for r in rows:
        if r.get(metric) is not None:
            groups.setdefault(r["study_id"], []).append(r[metric])
    values = np.array([np.mean(v) for v in groups.values()])
    if not len(values):
        return {"mean": None, "ci95": None, "independent_studies": 0}
    if len(values) < 2:
        return {"mean": float(values.mean()), "ci95": None, "independent_studies": len(values)}
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(boot, [0.025, 0.975]).tolist(), "independent_studies": len(values)}


def evaluate(c, split="validation", methods=None, regression_adapter=None, regression_output=None):
    if c.get("pilot_only") and (split != "train" or regression_adapter is not None or any(m.startswith("qwen") for m in (methods or []))):
        raise ValueError("Single-study pilot permits numerical train diagnostics only; no independent evaluation or Qwen")
    methods = methods or METHODS[:4]
    if len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("Unknown/duplicate methods")
    if split not in {"train", "validation", "test", "internal_test"}:
        raise ValueError("Evaluation split must be train/validation/test/internal_test")
    if split in {"test", "internal_test"}:
        check_frozen(c, methods)
    run = Path(c["run_dir"])
    if regression_adapter is not None and split != "validation":
        raise ValueError("Stage 2 regression may only use Stage 1 public validation")
    if split == "validation" and (run / "frozen_protocol.json").exists() and regression_adapter is None:
        raise ValueError("Validation is frozen; start a new experiment for new rules")
    root, index = prepared(c, internal=split == "internal_test", public_test=split == "test" and c.get("defer_public_test", False))
    out = Path(regression_output) if regression_output else run / (f"evaluation_{split}" if regression_adapter is None else "stage2_regression_validation")
    if out.exists():
        raise FileExistsError("Evaluation output exists; preserve it and use a new run for changes")
    out.mkdir(parents=True)
    all_cards = {r["example_id"]: r["card"] for r in read_jsonl(root / "cards.jsonl")}
    lineage = [r for r in read_jsonl(root / "private/lineage.jsonl") if r["split"] == split]
    if not lineage:
        raise ValueError("No examples in evaluation split")
    learned = None
    if "non_llm" in methods:
        model = read_json(run / "non_llm.json")
        _, train_index = prepared(c)
        if model["config_hash"] != digest(c) or model["prepared_hash"] != digest(train_index):
            raise ValueError("Non-LLM fit stale")
        learned = model["model"]
    predictions = {}
    elapsed = {}
    for method in methods:
        started = time.monotonic()
        if method.startswith("qwen"):
            results = infer(c, [{"example_id": r["example_id"], "card": all_cards[r["example_id"]]} for r in lineage], method, adapter_override=regression_adapter)
        else:
            results = []
            for r in lineage:
                eid = r["example_id"]
                target, scores = select(all_cards[eid], "loading_refit" if method == "pca_raw" else method, c, learned)
                results.append({"example_id": eid, "valid": True, "output": target, "gene_scores": scores})
        predictions[method] = {r["example_id"]: r for r in results}
        write_jsonl(out / f"predictions_{method}.jsonl", results)
        elapsed[method] = time.monotonic() - started
    # Numerical recovery precedes loading references: selector cannot inspect truth.
    metrics, final_cards, coverage_rows = [], [], []
    for view_id in sorted({r["view_id"] for r in lineage}):
        group = sorted([r for r in lineage if r["view_id"] == view_id], key=lambda r: r["candidate_index"])
        row = group[0]
        with np.load(root / "private" / f"{view_id}.npz", allow_pickle=False) as a:
            view = dict(a)
        recovered = {}
        for method in methods:
            ws, zs, heldzs, means, accepted, invalid = [], [], [], [], [], 0
            for r in group:
                eid, j = r["example_id"], r["candidate_index"]
                pred = predictions[method][eid]
                if not pred["valid"]:
                    invalid += 1
                    continue
                target = validate_output(pred["output"], all_cards[eid], c["min_support"])
                if target["decision"] != "retain":
                    continue
                available = {g["slot_id"]: g["feature_id"] for g in all_cards[eid]["genes"]}
                universe = {str(g): k for k, g in enumerate(view["genes"][view["measured"]])}
                support = [universe[available[s]] for s in target["supported_gene_slots"]]
                if method == "pca_raw":
                    w, mean = view["W"][:, j], view["mean"]
                    z, hz = (view["X"][view["discovery"]] - mean) @ w, (view["X"][view["heldout"]] - mean) @ w
                else:
                    w, z, hz, mean = refit(view["X"][view["discovery"]], view["X"][view["heldout"]], support, view["W"][:, j])
                ws.append(w); zs.append(z); heldzs.append(hz); means.append(mean); accepted.append(r)
            keep = deduplicate(ws, zs, c["dedup_cosine"])
            w = np.column_stack([ws[i] for i in keep]) if keep else np.zeros((view["X"].shape[1], 0))
            z = np.column_stack([zs[i] for i in keep]) if keep else np.zeros((len(view["discovery"]), 0))
            hz = np.column_stack([heldzs[i] for i in keep]) if keep else np.zeros((len(view["heldout"]), 0))
            full_w = np.zeros((len(view["genes"]), w.shape[1]))
            full_w[view["measured"]] = w
            artifact = out / "factors" / f"{view_id}_{method}.npz"
            artifact.parent.mkdir(exist_ok=True)
            np.savez_compressed(artifact, W=full_w, Z_discovery=z, Z_heldout=hz, genes=view["genes"],
                                discovery_samples=view["samples"][view["discovery"]], heldout_samples=view["samples"][view["heldout"]],
                                measured_mask=view["measured"], support_mask=np.abs(full_w) > 1e-12, mean=view["mean"],
                                measured_gene_order=view["genes"][view["measured"]])
            for column, i in enumerate(keep):
                final_cards.append({"factor_id": f"{view_id}_{method}_{column}", "source_example": accepted[i]["example_id"],
                                    "method": method, "artifact": str(artifact.relative_to(out)), "column": column,
                                    "selected_support_genes": view["genes"][view["measured"]][np.abs(ws[i]) > 1e-12].tolist(),
                                    "noisy_evidence": all_cards[accepted[i]["example_id"]], "provenance": accepted[i],
                                    "scope": "dataset-local numerical recovery; no causal/pathway identity claim"})
            selected_stabilities = [all_cards[accepted[i]["example_id"]]["replicate_evidence"][0]["bootstrap_cosine"] for i in keep]
            recovered[method] = w, hz, len(accepted), len(keep), invalid, selected_stabilities
        with np.load(root / "private" / f"{row['dataset_key']}_reference.npz", allow_pickle=False) as a:
            refs = dict(a)
        biological = refs["axes"] != "technical"
        tw = refs["W"][:, biological]
        oracle_result = oracle(view, tw, [all_cards[r["example_id"]] for r in group], c)
        controlled = row["source_kind"] == "controlled_simulation"
        for method, (w, hz, selected, unique, invalid, stabilities) in recovered.items():
            m = factor_metrics(w, hz, tw[view["measured"]], refs["Z"][view["heldout"]][:, biological], controlled, c["recovery_similarity"])
            m.update(oracle_result)
            m.update(method=method, view_id=view_id, study_id=row["study_id"], noise_level=row["noise_level"], label_source=row["label_source"],
                     missing_feature_coverage=row["missing_feature_coverage"], candidate_count=len(group), selected_count=selected,
                     coverage=selected / len(group), abstention_rate=1 - selected / len(group), invalid_rate=invalid / len(group),
                     controlled_null_technical_false_positive=float(unique > 0) if controlled and not biological.any() else None,
                     mean_support_size=float(np.mean((np.abs(w) > 1e-12).sum(axis=0))) if unique else 0)
            metrics.append(m)
            # Common noisy-only ordering, not self-reported LLM confidence. Descriptive
            # test curves cannot select a new deployment threshold after freezing.
            for threshold in [0.0, 0.25, 0.5, 0.75, 0.9]:
                take = np.array(stabilities) >= threshold
                sub = w[:, take]
                similarities = cosine(sub, tw[view["measured"]])
                match = assignment(similarities)
                supported = sum(similarities[i, j] >= c["recovery_similarity"] for i, j in match)
                coverage_rows.append({"method": method, "study_id": row["study_id"], "view_id": view_id, "noise_level": row["noise_level"],
                                      "stability_cutoff": threshold, "coverage": sub.shape[1] / len(group),
                                      "error_fraction": 1 - supported / sub.shape[1] if sub.shape[1] else None,
                                      "error_interpretation": "controlled_support_error" if controlled else "weak_reference_nonagreement"})
    write_jsonl(out / "factor_cards.jsonl", final_cards)
    write_jsonl(out / "metrics.jsonl", metrics)
    with (out / "coverage_error.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage_rows[0]))
        writer.writeheader(); writer.writerows(coverage_rows)
    scalar_keys = [k for k, v in metrics[0].items() if not isinstance(v, (dict, list))]
    with (out / "noise_curves.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(metrics)
    summary = {m: {metric: ci_by_study([r for r in metrics if r["method"] == m], metric, c["seed"])
               for metric in ["recovery_recall", "mean_loading_similarity_all_references", "coverage", "controlled_null_technical_false_positive", "invalid_rate"]} for m in methods}
    # Paired study-level differences against each numerical method, without picking a test winner.
    differences = {}
    if "qwen_finetuned" in methods:
        tuned = {r["view_id"]: r for r in metrics if r["method"] == "qwen_finetuned"}
        for method in METHODS[:4]:
            paired = []
            for r in metrics:
                if r["method"] == method and r["recovery_recall"] is not None:
                    paired.append({"study_id": r["study_id"], "delta": tuned[r["view_id"]]["recovery_recall"] - r["recovery_recall"]})
            differences[method] = ci_by_study(paired, "delta", c["seed"])
    report = {"config_hash": digest(c), "code_hash": code_hash(), "split": split, "executed_methods": methods,
              "regression_adapter": str(regression_adapter) if regression_adapter else None,
              "pilot_only": bool(c.get("pilot_only")), "is_independent_evaluation": split != "train",
              "not_executed_methods": [m for m in METHODS if m not in methods], "summary": summary, "paired_recovery_differences": differences,
              "selection_seconds": elapsed, "source_kinds": index["source_kinds"],
              "scientific_status": "requires independent real-study evidence and matched coverage/error analysis; no automatic success claim",
              "limitations": ["Simulator recovery is not real biological validation", "Real reference agreement is weak-reference consistency",
                              "Raw PCA is an unrefitted diagnostic; all selection methods share support-restricted rank-1 refit",
                              "No AUPRC reported from hard gene selections", "Study bootstrap CI unreliable with very few studies",
                              "Composition not supervised without explicit composition truth", "No external annotation evidence in v0.1"]}
    if c.get("pilot_only"):
        report["scientific_status"] = "Single public study, training-fit numerical diagnostic only; no cross-study generalization or causal recovery claim"
        report["limitations"].append("All references are weak and no model selection is permitted in this diagnostic")
        if "non_llm" in methods:
            report["limitations"].append("Non-LLM selector fitted and assessed on same train study")
    write_json(out / "report.json", report)
    return report
