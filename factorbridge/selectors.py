"""Auditable non-LLM selector on the same visible card; no reference features."""
import hashlib
import numpy as np

from .contracts import validate_card


def features(card):
    validate_card(card)
    stability = card["replicate_evidence"][0]["bootstrap_cosine"]
    rows = []
    maximum = max(abs(g["signed_loading"]) for g in card["genes"]) or 1
    for g in card["genes"]:
        identity = np.zeros(32)
        identity[int(hashlib.sha256(g["feature_id"].encode()).hexdigest(), 16) % len(identity)] = 1
        rows.append(np.r_[1., abs(g["signed_loading"]) / maximum, g["signed_loading"],
                          g["noisy_view_recurrence"], stability, 1 / g["loading_rank"], identity])
    numeric = np.array(rows)
    if card.get('task') == 'recover_semantic_programme':
        from .semantic import semantic_features
        return np.column_stack([numeric, semantic_features(card)])
    return numeric


def fit_logistic(x, y, iterations=500):
    x, y = np.asarray(x), np.asarray(y)
    if not len(y):
        raise ValueError("No supervised observations")
    if len(set(y)) < 2:
        # Explicit constant learned from observed labels; reported, never invented negatives.
        return {"kind": "constant", "value": float(y[0]), "n": len(y)}
    beta = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(x @ beta, -30, 30)))
        penalty = beta * 0.01
        penalty[0] = 0
        beta -= 0.2 * (x.T @ (p - y) / len(y) + penalty)
    return {"kind": "logistic", "beta": beta.tolist(), "n": len(y)}


def predict_logistic(model, x):
    if model["kind"] == "constant":
        return np.full(len(x), model["value"])
    return 1 / (1 + np.exp(-np.clip(x @ np.array(model["beta"]), -30, 30)))


def train_selector(cards, labels):
    xs, ys, xd, yd = [], [], [], []
    sources = {}
    for card, label in zip(cards, labels):
        target = label["target"]
        f = features(card)
        xd.append(f.mean(axis=0))
        yd.append(int(target["decision"] == "retain"))
        sources[label["label_source"]] = sources.get(label["label_source"], 0) + 1
        positive = set(target["supported_gene_slots"])
        for row, gene in zip(f, card["genes"]):
            known = label["gene_supervision"] == "complete" or gene["slot_id"] in positive
            if known:
                xs.append(row)
                ys.append(int(gene["slot_id"] in positive))
    return {"genes": fit_logistic(xs, ys), "decision": fit_logistic(xd, yd), "sources": sources,
            "unlabeled_real_genes": "excluded from negative supervision"}


def select(card, method, c, learned=None):
    f = features(card)
    scores = np.abs([g["signed_loading"] for g in card["genes"]])
    retained = True
    if method == "stability":
        scores = np.array([g["noisy_view_recurrence"] for g in card["genes"]])
        retained = card["replicate_evidence"][0]["bootstrap_cosine"] >= c["stability_threshold"]
        keep = scores >= c["gene_threshold"]
    elif method == "semantic_prior":
        terms = card['functional_evidence']['terms']
        supported = {slot for t in terms if t['q'] <= c['semantic']['enrichment_fdr'] for slot in t['member_slots']}
        scores = np.array([g['noisy_view_recurrence'] for g in card['genes']])
        retained = card['replicate_evidence'][0]['bootstrap_cosine'] >= c['stability_threshold']
        # Unannotated stable factors remain eligible; this policy is explicit.
        keep = np.array([scores[i] >= c['gene_threshold'] and (not supported or g['slot_id'] in supported)
                         for i, g in enumerate(card['genes'])])
    elif method == "non_llm":
        scores = predict_logistic(learned["genes"], f)
        retained = predict_logistic(learned["decision"], f.mean(axis=0, keepdims=True))[0] >= c["decision_threshold"]
        keep = scores >= c["gene_threshold"]
    elif method == "loading_refit":
        keep = np.ones(len(f), dtype=bool)
    else:
        raise ValueError(method)
    slots = [g["slot_id"] for g, yes in zip(card["genes"], keep) if yes]
    decision = "retain" if retained and len(slots) >= c["min_support"] else "uncertain"
    return {"decision": decision, "supported_gene_slots": slots if decision == "retain" else [],
            "axis_flags": [], "evidence_ids": [], "limitations": []}, scores.tolist()
