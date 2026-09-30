"""Explicit controlled simulation and real-data weak references; truths stay sidecar."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import numpy as np

from .data import manifest, load_dataset, local_split, preprocess, ROLES, assert_lineage
from .factors import candidates, card
from .io import write_json, write_jsonl, digest, sha256


def simulate(destination, seed=42, studies=18):
    """Engineering/simulator benchmark, never a substitute for public biological data."""
    root = Path(destination).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Simulation directory must be empty")
    if studies < 9 or studies % 3:
        raise ValueError("Use at least 9 studies, divisible by 3")
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(studies):
        rng = np.random.default_rng(seed + i * 1009)
        n, p = 48, 96
        kind = ["mixed", "null", "technical", "nonlinear", "correlated", "mixed"][i % 6]
        k = 0 if kind == "null" else 1 if kind == "technical" else 3
        w = np.zeros((p, k))
        for j in range(k):
            support = rng.choice(p, 12, replace=False)
            w[support, j] = rng.choice([-1, 1], 12) * rng.uniform(0.5, 1.5, 12)
            w[:, j] /= np.linalg.norm(w[:, j])
        z = rng.normal(size=(n, k)) * 4
        if kind == "correlated":
            z[:, 1] = 0.8 * z[:, 0] + 0.6 * z[:, 1]
        if kind == "nonlinear":
            z[:, 0] = 4 * np.tanh(z[:, 0])
            z[:, 1] = (z[:, 1] ** 2 - 16) / 5
        x = z @ w.T + rng.normal(size=(n, p)) * rng.uniform(0.3, 1.0, p)
        if kind == "nonlinear":
            mask = rng.random(x.shape) < 0.02
            x += mask * rng.normal(0, 5, x.shape)  # outliers violate isotropic PCA noise
        name = f"simulation_{i:03d}"
        samples = np.array([f"sample_{j:03d}" for j in range(n)])
        genes = np.array([f"feature_{j:03d}" for j in range(p)])
        np.savez_compressed(root / f"{name}.npz", X=x, genes=genes, samples=samples)
        axes = (["technical"] if kind == "technical" else ["biological", "biological", "technical"]) if k else []
        np.savez_compressed(root / f"{name}.truth.npz", W=w, Z=z, axes=np.array(axes, dtype=str))
        with (root / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "biological_unit", "condition"])
            writer.writerows((s, f"{name}_unit_{j:03d}", "simulated") for j, s in enumerate(samples))
        # Contiguous study blocks, before any noise view generation.
        split = ["public_train", "public_validation", "public_test"][min(2, i * 3 // studies)]
        entries.append({"dataset_id": name, "study_id": name, "parent_dataset": name,
                        "matrix_path": f"{name}.npz", "metadata_path": f"{name}.csv", "truth_path": f"{name}.truth.npz",
                        "species": "controlled_simulation", "assay": "simulated_expression", "resolution": "biological_unit",
                        "data_scale": "log_expression", "biological_unit_col": "biological_unit", "condition_col": "condition",
                        "role": split, "source_kind": "controlled_simulation", "simulation_design": kind})
    write_json(root / "manifest.json", {"datasets": entries, "warning": "Simulation only; not real expression data"})
    return root / "manifest.json"


def noise_view(raw, scale, discovery, level, missing_fraction, rng):
    if scale == "raw_counts":
        keep = 1 / (1 + level)
        noisy = preprocess(rng.binomial(raw.astype(np.int64), keep), scale)
        operator = {"kind": "binomial_thinning", "keep_probability": keep}
    else:
        sigma = np.std(raw[discovery], axis=0, ddof=1)
        noisy = raw + rng.normal(size=raw.shape) * sigma * level
        operator = {"kind": "additive_log_scale_gaussian", "sigma_multiplier": level, "depth_thinning": "not_applicable"}
    measured = np.ones(raw.shape[1], dtype=bool)
    if level > 0:
        mask = rng.choice(raw.shape[1], int(raw.shape[1] * missing_fraction), replace=False)
        measured[mask] = False
    # Masked columns are excluded, never silently presented as biological zeros.
    return noisy[:, measured], measured, operator


def label(candidate, indices, measured_indices, loading, truth_w, axes, source, c):
    answer = {"decision": "uncertain", "supported_gene_slots": [], "axis_flags": [], "evidence_ids": [], "limitations": ["insufficient_support"]}
    status = "controlled_truth" if source == "controlled_simulation" else "real_weak_reference"
    if truth_w.shape[1] == 0:
        if source == "controlled_simulation":
            answer.update(decision="reject_null", limitations=["controlled_null"])
        return answer, {"label_source": status, "matched_reference": None, "gene_supervision": "complete" if source == "controlled_simulation" else "positive_unlabeled"}
    restricted = truth_w[measured_indices]
    norms = np.linalg.norm(restricted, axis=0)
    sims = np.abs(loading @ restricted) / np.maximum(norms, 1e-12)
    best = int(sims.argmax())
    flags = sorted(set(str(axes[j]) for j in np.flatnonzero(sims >= 0.35)))
    if sims[best] >= c["reference_similarity"]:
        support = set(np.flatnonzero(np.abs(restricted[:, best]) > 0))
        slots = [g["slot_id"] for g, idx in zip(candidate["genes"], indices) if idx in support]
        answer.update(axis_flags=flags, supported_gene_slots=slots)
        if axes[best] == "technical" and source == "controlled_simulation":
            answer.update(decision="reject_null", supported_gene_slots=[], limitations=["technical_dominated"])
        elif len(slots) >= c["min_support"]:
            answer.update(decision="retain", limitations=["weak_reference"] if status == "real_weak_reference" else [])
    return answer, {"label_source": status, "matched_reference": best,
                    "reference_similarity": float(sims[best]), "gene_supervision": "complete" if source == "controlled_simulation" else "positive_unlabeled"}


def prepare(c, internal=False):
    root = Path(c["run_dir"]) / ("internal_prepared" if internal else "prepared")
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite prepared lineage: {root}; use a new run_dir")
    entries = manifest(c["manifest"])
    entries = [e for e in entries if (e["role"] == "internal_test") == internal]
    if not entries:
        raise ValueError("No eligible datasets")
    strata = {(e["species"], e["assay"], e["resolution"], e["data_scale"]) for e in entries}
    if len(strata) != 1:
        raise ValueError("Stage 1 pilot requires a single species/assay/resolution/scale per run")
    if not internal and {e["role"] for e in entries} != {"public_train", "public_validation", "public_test"}:
        raise ValueError("Independent study roles train/validation/test required; never split views randomly")
    cards, labels, sidecars, splits = [], [], [], []
    root.mkdir(parents=True)
    try:
        for number, e in enumerate(entries):
            raw, genes, samples, meta = load_dataset(e)
            ds_seed = c["seed"] + number * 10007
            disc, held = local_split(meta, e["biological_unit_col"], c["local_holdout_fraction"], ds_seed)
            units = [r[e["biological_unit_col"]] for r in meta]
            clean = preprocess(raw, e["data_scale"])
            if e["source_kind"] == "controlled_simulation":
                with np.load(e["truth_path"], allow_pickle=False) as a:
                    tw, tz, axes = a["W"], a["Z"], a["axes"]
                if tw.shape[0] != len(genes) or tz.shape != (len(samples), tw.shape[1]):
                    raise ValueError("Simulation truth dimensions do not match matrix")
            else:
                rw, _, rmean, rr, rs = candidates(clean[disc], [units[i] for i in disc], c, ds_seed)
                selected = np.flatnonzero(rs >= c["reference_stability"])
                # Fixed positive-support definition; unselected genes are NOT verified negatives.
                tw = np.zeros((len(genes), len(selected)))
                for j, source in enumerate(selected):
                    idx = np.argsort(-np.abs(rw[:, source]), kind="stable")[:c["card_genes"]]
                    tw[idx, j] = rw[idx, source]
                    tw[:, j] /= max(np.linalg.norm(tw[:, j]), 1e-12)
                tz = (clean - rmean) @ tw
                axes = np.array(["biological"] * len(selected))
            ds_key = f"d{number:04d}"
            (root / "private").mkdir(exist_ok=True)
            np.savez_compressed(root / "private" / f"{ds_key}_reference.npz", W=tw, Z=tz, axes=axes, genes=genes)
            for r, sample in enumerate(samples):
                splits.append({"dataset_id": e["dataset_id"], "study_id": e["study_id"], "sample_id": str(sample), "biological_unit": units[r],
                               "split": ROLES[e["role"]], "local_role": "discovery" if r in disc else "heldout_projection"})
            for v, level in enumerate(c["noise_levels"]):
                noisy, measured, operator = noise_view(raw, e["data_scale"], disc, level, c["missing_fraction"], np.random.default_rng(ds_seed + v + 1))
                mw = np.flatnonzero(measured)
                w, z, mean, recur, stable = candidates(noisy[disc], [units[i] for i in disc], c, ds_seed + 100 + v)
                view_id = f"{ds_key}_v{v:02d}"
                np.savez_compressed(root / "private" / f"{view_id}.npz", X=noisy, measured=measured, genes=genes,
                                    samples=samples, discovery=disc, heldout=held, W=w, mean=mean)
                noisy_hash = digest({"shape": list(noisy.shape), "values_sha256": hashlib.sha256(noisy.tobytes()).hexdigest(), "samples": samples.tolist(), "genes": genes[measured].tolist()})
                for j in range(w.shape[1]):
                    eid = f"{view_id}_c{j:02d}"
                    evidence, idx = card(w[:, j], recur[:, j], stable[j], genes[measured], e, c)
                    target, quality = label(evidence, idx, mw, w[:, j], tw, axes, e["source_kind"], c)
                    if e["source_kind"] != "controlled_simulation":
                        # Stability does not establish causal biology or absence of confounding.
                        target["axis_flags"] = []
                    cards.append({"example_id": eid, "card": evidence})
                    labels.append({"example_id": eid, "target": target, **quality})
                    sidecars.append({"example_id": eid, "view_id": view_id, "dataset_key": ds_key,
                                     "candidate_index": j, "dataset_id": e["dataset_id"], "study_id": e["study_id"],
                                     "parent_dataset": e["parent_dataset"],
                                     "parent_factor": [e["parent_dataset"] + f":factor_{q}" for q in range(tw.shape[1])] or [e["parent_dataset"] + ":null"],
                                     "biological_units": [e["study_id"] + ":" + u for u in sorted(set(units))],
                                     "split": ROLES[e["role"]], "source_kind": e["source_kind"], "noise_level": level,
                                     "noise_operator": operator, "missing_feature_coverage": float(measured.mean()),
                                     "noisy_matrix_sha256": noisy_hash,
                                     "label_source": quality["label_source"]})
        assert_lineage(sidecars)
        write_jsonl(root / "cards.jsonl", cards)
        write_jsonl(root / "private" / "labels.jsonl", labels)
        write_jsonl(root / "private" / "lineage.jsonl", sidecars)
        write_jsonl(root / "splits.jsonl", splits)
        fingerprint = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
        write_json(root / "index.json", {"complete": True, "config_hash": digest(c), "files": fingerprint,
                   "studies": len(entries), "units": sum(e["n_units"] for e in entries), "examples": len(cards),
                   "source_kinds": sorted({e["source_kind"] for e in entries}), "stage": "noise_recovery"})
    except Exception:
        write_json(root / "FAILED.json", {"complete": False, "message": "Preparation failed; incomplete outputs must not be used"})
        raise
    return root
