"""Noisy-only evidence and one shared support-restricted numerical estimator."""
from __future__ import annotations

import numpy as np


def signed(w):
    w = w.copy()
    for j in range(w.shape[1]):
        if w[np.argmax(np.abs(w[:, j])), j] < 0:
            w[:, j] *= -1
    return w


def pca(x, rank, n_units):
    mean = x.mean(axis=0)
    xc = x - mean
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    tol = (s[0] if len(s) else 0) * max(xc.shape) * np.finfo(float).eps
    k = min(rank, n_units - 1, int(np.sum(s > tol)))
    if k < 1:
        raise ValueError("No identifiable nonzero factor")
    w = signed(vt[:k].T)
    return w, xc @ w, mean


def candidates(x, units, c, seed):
    w, z, mean = pca(x, c["rank"], len(set(units)))
    n, k = w.shape
    top = min(c["card_genes"], n)
    recurrence, stability = np.zeros((n, k)), np.zeros(k)
    rng = np.random.default_rng(seed)
    unique = sorted(set(units))
    groups = [np.flatnonzero(np.array(units) == u) for u in unique]
    valid = 0
    for _ in range(c["bootstrap_repeats"]):
        draws = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in draws])
        if len(set(draws)) < 2:
            continue
        bw, _, _ = pca(x[idx], k, len(set(draws)))
        sims = np.abs(w.T @ bw)
        for j in range(k):
            b = int(sims[j].argmax())
            stability[j] += sims[j, b]
            recurrence[np.argsort(-np.abs(bw[:, b]), kind="stable")[:top], j] += 1
        valid += 1
    if valid < 2:
        raise ValueError("Insufficient valid grouped bootstraps")
    return w, z, mean, recurrence / valid, stability / valid


def card(w, recurrence, stability, genes, e, c):
    idx = np.argsort(-np.abs(w), kind="stable")[:c["card_genes"]]
    rows = [{"slot_id": f"g{i+1:03d}", "feature_id": str(genes[g]), "loading_rank": i + 1,
             "signed_loading": round(float(w[g]), 5), "noisy_view_recurrence": round(float(recurrence[g]), 3),
             "measured": True} for i, g in enumerate(idx)]
    return {"task": "recover_noisy_factor", "species": e["species"], "assay": e["assay"],
            "resolution": e["resolution"], "data_scale": e["data_scale"], "genes": rows,
            "pathway_evidence": [], "regulon_evidence": [], "qc_associations": [],
            "replicate_evidence": [{"evidence_id": "e001", "bootstrap_cosine": round(float(stability), 3)}],
            "available_evidence_ids": ["e001"],
            "missing_evidence": ["external_annotation", "measured_technical_covariates", "composition_information"]}, idx


def refit(x_discovery, x_holdout, support, anchor):
    support = np.asarray(support, dtype=int)
    if len(set(support)) != len(support) or len(support) < 2:
        raise ValueError("Support needs at least two unique measured genes")
    mean = x_discovery.mean(axis=0)
    w, _, _ = pca(x_discovery[:, support], 1, len(x_discovery))
    full = np.zeros(x_discovery.shape[1])
    full[support] = w[:, 0]
    if full @ anchor < 0:  # alignment uses noisy candidate only
        full *= -1
    return full, (x_discovery - mean) @ full, (x_holdout - mean) @ full, mean


def deduplicate(ws, zs, threshold):
    keep = []
    for i, w in enumerate(ws):
        duplicate = False
        for j in keep:
            cosine = abs(float(w @ ws[j])) / max(float(np.linalg.norm(w) * np.linalg.norm(ws[j])), 1e-12)
            a, b = zs[i] - np.mean(zs[i]), zs[j] - np.mean(zs[j])
            corr = abs(float(a @ b)) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
            if cosine >= threshold or corr >= threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return keep
