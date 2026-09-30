"""Pinned public GEO pilot: download real files, join by title, audit, never invent units."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
from pathlib import Path
import re
import shutil
import urllib.request

import numpy as np

from .data import manifest
from .io import read_json, sha256, write_json

ACCESSION = "GSE124109"
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE124nnn/GSE124109/"
# Observed file hashes, retrieved from NCBI 2026-09-30; not publisher signatures.
RESOURCES = {
    "GSE124109_family.soft.gz": ("soft/", "d2f4feca901b21f5a5e77e7ace6194ba7d799ceea2b65904a6b78a86d5349e77"),
    "GSE124109_genes.processed.fpkm_table.txt.gz": ("suppl/", "e07c984115f583eda9e0281a2c32066d1f672020d238fdbd322f39e6effc6b07"),
}


def download_verified(url, path, expected):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != expected:
            raise ValueError(f"Cached file hash mismatch: {path}; investigate instead of silently replacing")
        return "verified_cache"
    part = path.with_name(path.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FactorBridge-public-data/0.1"})
    with urllib.request.urlopen(request, timeout=90) as source, part.open("wb") as target:
        shutil.copyfileobj(source, target)
    if sha256(part) != expected:
        raise ValueError(f"Downloaded file hash mismatch: {part}; no conversion performed")
    part.replace(path)
    return "downloaded"


def soft_samples(path):
    samples, current = [], None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith("^"):
                current = None
                if line.startswith("^SAMPLE = "):
                    current = {"accession": [line.split(" = ", 1)[1]]}
                    samples.append(current)
            elif current is not None and line.startswith("!Sample_") and " = " in line:
                key, value = line.split(" = ", 1)
                current.setdefault(key.removeprefix("!Sample_"), []).append(value)
    return samples


def convert_gse124109(matrix_path, soft_path):
    """Use explicit sample names, never GEO row order (Control_2 precedes Control_1)."""
    samples = soft_samples(soft_path)
    titles = [s["title"][0] for s in samples]
    if len(titles) != 30 or len(set(titles)) != 30:
        raise ValueError("Expected 30 uniquely titled samples in this pinned GEO series")
    by_title = dict(zip(titles, samples))
    with gzip.open(matrix_path, "rt", encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream, delimiter="\t")
        header = next(rows)
        if header[0] != "Gene" or len(set(header[1:])) != 30 or set(header[1:]) != set(titles):
            raise ValueError("Matrix columns do not exactly match GEO sample titles")
        genes, values = [], []
        for row in rows:
            if len(row) != 31 or not row[0]:
                raise ValueError("Malformed gene row")
            genes.append(row[0])
            values.append([float(v) for v in row[1:]])
    x = np.asarray(values, dtype=np.float64).T
    if len(set(genes)) != len(genes) or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Duplicate genes, missing values, or invalid FPKM; no silent imputation")
    metadata = []
    for title in header[1:]:
        s = by_title[title]
        if s["organism_ch1"] != ["Rattus norvegicus"] or s["library_strategy"] != ["RNA-Seq"]:
            raise ValueError("Unexpected species or assay")
        match = re.fullmatch(r"(Control|Day(2|3|4|6|8|10|12|14|16))_([123])", title)
        if not match:
            raise ValueError(f"Unrecognised culture/condition title: {title}")
        day = int(match[2] or 0)
        condition = "cell without serum starvation" if day == 0 else f"cell with serum starvation for {day} days"
        if f"cell type: {condition}" not in s["characteristics_ch1"]:
            raise ValueError("Sample title and deposited condition disagree")
        biosamples = [v.rsplit("/", 1)[-1] for v in s.get("relation", []) if v.startswith("BioSample: ")]
        if len(biosamples) != 1:
            raise ValueError("Exactly one deposited BioSample is required")
        metadata.append({"sample_id": s["accession"][0], "biological_unit": biosamples[0],
                         "sample_title": title, "condition": condition, "starvation_days": day,
                         "replicate_within_condition": match[3], "source_name": s["source_name_ch1"][0]})
    if len({m["biological_unit"] for m in metadata}) != 30:
        raise ValueError("BioSamples are reused; inspect culture lineage before splitting")
    return np.log1p(x), np.asarray(genes), np.asarray([m["sample_id"] for m in metadata]), metadata


def fetch_gse124109(destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = root / "download_status.json"
    write_json(state, {"status": "running", "accession": ACCESSION})
    try:
        resources = []
        for name, (folder, expected) in RESOURCES.items():
            path, url = root / "raw" / name, BASE + folder + name
            action = download_verified(url, path, expected)
            resources.append({"url": url, "file": "raw/" + name, "sha256": expected,
                              "bytes": path.stat().st_size, "action": action})
        record = root / "provenance.json"
        if record.exists():
            saved = read_json(record)
            for name, expected in saved["derived_sha256"].items():
                if sha256(root / name) != expected:
                    raise ValueError(f"Converted artifact changed: {name}")
            manifest(root / "manifest.json")
        else:
            if (root / "matrix.npz").exists() or (root / "manifest.json").exists():
                raise FileExistsError("Incomplete prior conversion; inspect it and use a new destination")
            x, genes, samples, metadata = convert_gse124109(
                root / "raw/GSE124109_genes.processed.fpkm_table.txt.gz", root / "raw/GSE124109_family.soft.gz")
            np.savez_compressed(root / "matrix.npz", X=x, genes=genes, samples=samples)
            with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
                writer.writeheader()
                writer.writerows(metadata)
            write_json(root / "manifest.json", {"datasets": [{
                "dataset_id": ACCESSION, "study_id": ACCESSION, "parent_dataset": ACCESSION,
                "matrix_path": "matrix.npz", "metadata_path": "metadata.csv", "species": "Rattus norvegicus",
                "assay": "bulk_RNA_seq", "resolution": "bulk_population", "data_scale": "log_expression",
                "biological_unit_col": "biological_unit", "condition_col": "condition",
                "role": "public_train", "source_kind": "public_real", "source_url":
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124109"}],
                "scope": "Single-study numerical pilot, no independent validation/test; all views remain train"})
            manifest(root / "manifest.json")
            saved = {"accession": ACCESSION, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                     "resources": resources, "shape": list(x.shape), "transformation": "natural log1p(FPKM); no count thinning",
                     "sample_join": "Exact matrix column title to GEO Sample_title, then BioSample biological unit",
                     "feature_selection": "None; retain every deposited feature including all-zero genes",
                     "reference_kind": "Stability-derived weak references only, not causal biological truth",
                     "unit_evidence": "GEO reports 3 biological replicates at each of 10 destructive RNA collection times; 30 distinct BioSamples",
                     "limitations": ["No donor, passage, or cross-time culture-batch lineage supplied; local holdout is a culture-sample diagnostic only",
                                      "All samples are one study and one cell-line system, not 30 independent studies",
                                      "Deposited processing mentions both rn19 alignment and rn6 genome; retain original feature names without inferred remapping",
                                      "FPKM already normalized across submitted samples; local projection does not constitute independent preprocessing validation"],
                     "derived_sha256": {name: sha256(root / name) for name in ["matrix.npz", "metadata.csv", "manifest.json"]}}
            write_json(record, saved)
        write_json(state, {"status": "completed", "resources": resources, "shape": saved["shape"]})
        return root / "manifest.json"
    except Exception as exc:
        write_json(state, {"status": "failed", "error": str(exc), "error_type": type(exc).__name__})
        raise
