"""Strict manifests and study-first splits. No accession-to-data inference."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from .io import read_json, sha256, write_json

ROLES = {"public_train": "train", "public_validation": "validation", "public_test": "test", "internal_test": "internal_test"}
REQUIRED = {"dataset_id", "study_id", "parent_dataset", "matrix_path", "metadata_path", "species", "assay", "resolution", "data_scale", "biological_unit_col", "condition_col", "role", "source_kind"}


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def manifest(path):
    entries = read_json(path).get("datasets")
    if not entries:
        raise ValueError("Manifest has no datasets; supply real matrix and metadata paths. No fallback.")
    base = Path(path).resolve().parent
    ids, studies, parents, units, matrix_owners = set(), {}, {}, {}, {}
    for e in entries:
        missing = sorted(k for k in REQUIRED if not e.get(k))
        if missing:
            raise ValueError(f"Missing manifest fields: {missing}")
        if e["dataset_id"] in ids:
            raise ValueError("Duplicate dataset_id")
        ids.add(e["dataset_id"])
        if e["role"] not in ROLES or e["data_scale"] not in {"raw_counts", "log_expression"}:
            raise ValueError("Unrecognised role or data_scale")
        if e["source_kind"] not in {"public_real", "internal_real", "controlled_simulation"}:
            raise ValueError("Unrecognised source_kind")
        if e["source_kind"] == "internal_real" and e["role"] != "internal_test":
            raise ValueError("Internal data are frozen test only in this implementation")
        for mapping, key in [(studies, e["study_id"]), (parents, e["parent_dataset"])]:
            if key in mapping and mapping[key] != e["role"]:
                raise ValueError(f"Split leakage: {key} crosses roles")
            mapping[key] = e["role"]
        for key in ("matrix_path", "metadata_path"):
            p = Path(e[key])
            p = p if p.is_absolute() else base / p
            if not p.is_file():
                raise FileNotFoundError(f"{e['dataset_id']} {key}: {p}")
            e[key] = str(p.resolve())
        content_hash = sha256(e["matrix_path"])
        if content_hash in matrix_owners and matrix_owners[content_hash] != e["role"]:
            raise ValueError("Identical matrix files cross splits under different dataset names")
        matrix_owners[content_hash] = e["role"]
        meta = table(e["metadata_path"])
        if not meta:
            raise ValueError("Empty metadata")
        for row in meta:
            for col in ("sample_id", e["biological_unit_col"], e["condition_col"]):
                if not row.get(col, "").strip():
                    raise ValueError(f"Missing metadata {col} in {e['dataset_id']}")
            # Cross-study reuse must be explicitly linked with global_unit_id.
            unit = row.get("global_unit_id") or e["study_id"] + ":" + row[e["biological_unit_col"]]
            if unit in units and units[unit] != e["role"]:
                raise ValueError(f"Biological unit crosses splits: {unit}")
            units[unit] = e["role"]
        if len({r["sample_id"] for r in meta}) != len(meta):
            raise ValueError("Duplicate metadata sample_id")
        e["n_units"] = len({r[e["biological_unit_col"]] for r in meta})
        if e["n_units"] < 8:
            raise ValueError("At least 8 independent units required for discovery/local holdout pilot")
        if e["source_kind"] == "controlled_simulation":
            p = Path(e.get("truth_path", ""))
            p = p if p.is_absolute() else base / p
            if not p.is_file():
                raise FileNotFoundError("Explicit simulation truth_path required")
            e["truth_path"] = str(p.resolve())
    return entries


def audit(c):
    entries = manifest(c["manifest"])
    records = []
    for e in entries:
        r = dict(e)
        r["file_sha256"] = {k: sha256(e[k]) for k in ("matrix_path", "metadata_path")}
        if e["role"] == "internal_test":
            r["matrix_status"] = "existence/hash only; frozen boundary; no expression analysis"
        else:
            x, genes, samples, meta = load_dataset(e)
            r.update(matrix_status="validated", shape=list(x.shape), n_genes=len(genes), n_samples=len(samples))
        records.append(r)
    report = {"datasets": records, "roles": {s: sum(e["role"] == s for e in entries) for s in ROLES},
              "no_accessions_assumed_downloaded": True, "scope": "single-dataset local extraction; same-species/assay cohorts only"}
    write_json(Path(c["run_dir"]) / "audit.json", report)
    return report


def load_dataset(e):
    p = Path(e["matrix_path"])
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as a:
            x = a["X"].astype(float)
            genes, samples = a["genes"].astype(str), a["samples"].astype(str)
    elif p.suffix == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        genes = np.array(rows[0][1:])
        samples = np.array([r[0] for r in rows[1:]])
        x = np.array([r[1:] for r in rows[1:]], dtype=float)
    else:
        raise ValueError("Use sample-by-gene .npz (X/genes/samples) or CSV; convert explicitly upstream")
    if x.ndim != 2 or x.shape != (len(samples), len(genes)) or not np.isfinite(x).all():
        raise ValueError("Matrix dimensions/finite values invalid; missingness needs explicit mask, not NaN")
    if len(set(genes)) != len(genes) or len(set(samples)) != len(samples):
        raise ValueError("Duplicate gene or sample IDs")
    if e["data_scale"] == "raw_counts" and (np.any(x < 0) or np.any(x != np.floor(x))):
        raise ValueError("raw_counts must be nonnegative integers")
    meta = {r["sample_id"]: r for r in table(e["metadata_path"])}
    if set(samples) != set(meta):
        raise ValueError("Matrix/metadata sample IDs differ; no silent alignment/drop")
    return x, genes, samples, [meta[s] for s in samples]


def local_split(meta, unit_col, fraction, seed):
    units = sorted({r[unit_col] for r in meta})
    if len(units) < 8:
        raise ValueError("Insufficient biological units")
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(units)[:max(2, int(len(units) * fraction))])
    test = np.array([r[unit_col] in held for r in meta])
    return np.flatnonzero(~test), np.flatnonzero(test)


def preprocess(x, scale):
    if scale == "raw_counts":
        totals = x.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("Zero-depth samples after thinning; explicit QC needed")
        return np.log1p(x / totals[:, None] * 10000)
    if scale == "log_expression":
        return x.copy()
    raise ValueError(scale)


def assert_lineage(rows):
    owners = {}
    for r in rows:
        for kind in ("study_id", "parent_dataset", "parent_factor"):
            vals = r[kind] if isinstance(r[kind], list) else [r[kind]]
            for value in vals:
                key = kind, value
                if key in owners and owners[key] != r["split"]:
                    raise ValueError(f"Lineage crosses splits: {key}")
                owners[key] = r["split"]
        for u in r["biological_units"]:
            key = "unit", u
            if key in owners and owners[key] != r["split"]:
                raise ValueError("Biological unit crosses splits")
            owners[key] = r["split"]
