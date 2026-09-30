"""Verified public human bulk panel. Fixed study roles; conservative culture grouping."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path
import re

import numpy as np

from .data import manifest
from .io import read_json, write_json, sha256
from .public_data import download_verified, soft_samples

SOURCES = {
    "GSE193258": {"role": "public_train", "n": 60,
        "soft_sha": "5f64725f8bbb5c23cd976789356391c00dfb40e601a514230be96fa71c03a33c",
        "matrix": "GSE193258_RNAseq_estimated_counts.tsv.gz",
        "matrix_sha": "51c92720b7bf4a0d7d29f6b7f33e304bd9f858ba8182572260ddf7f1c4de6d33",
        "transform": "log1p(1e6 * estimated counts / sample total); fractional estimated counts retained before normalization"},
    "GSE63577": {"role": "public_validation", "n": 30,
        "soft_sha": "97b36366f674ccf166ed1cd556ad5b0061ded1b3e76adcdc4d4ffd9f8012d85f",
        "matrix": "GSE63577_counts_rpkm_exvivo_jenage_data.xls.gz",
        "matrix_sha": "48b525e052a5cb17f157b875cb466cc9603b706ef6e94358e83079ef84ca5375",
        "transform": "log1p(1e6 * counts / sample total); counts sheet only; 30 columns in this file, other 18 series samples excluded"},
    "GSE113957": {"role": "public_test", "n": 143,
        "soft_sha": "041e5adfd00beb232d0e73503bfd8f51a0b5018e3849b7a5143e608aac6a86f4",
        "matrix": "GSE113957_fpkm.txt.gz",
        "matrix_sha": "999c9d720ff7f0dfe59675f347803d3a0181231313aace2f8406f08528aa123d",
        "transform": "log1p(FPKM); deposited RefSeq transcript identifiers retained as gene proxies"},
}


def characteristics(sample):
    result = {}
    for item in sample.get("characteristics_ch1", []):
        key, value = item.split(": ", 1)
        if key.lower() in result:
            raise ValueError("Duplicate characteristic")
        result[key.lower()] = value
    return result


def column_key(acc, title):
    if acc == "GSE193258":
        # The count-file names omit the explicitly documented washout durations.
        title = title.removesuffix(" [RNA-seq]")
        return re.sub(r"_long_washout_(72h|96h|7d|10d)_", "_long_wash_", title).replace("_short_washout_", "_short_wash_")
    if acc == "GSE63577":
        return title.replace("-", "_")
    return title


def metadata_rows(acc, columns, samples):
    lookup = {}
    for s in samples:
        title = s["title"][0]; key = column_key(acc, title)
        if key in lookup:
            raise ValueError("Ambiguous sample title normalization")
        lookup[key] = s
    if len(set(columns)) != len(columns) or not set(columns) <= set(lookup):
        raise ValueError(f"{acc}: matrix columns cannot be mapped exactly to sample metadata: {set(columns)-set(lookup)}")
    rows = []
    for name in columns:
        s = lookup[name]; ch = characteristics(s)
        if s.get("organism_ch1") != ["Homo sapiens"] or s.get("library_strategy") != ["RNA-Seq"]:
            raise ValueError("Unexpected organism or assay")
        bios = [v.rsplit("/", 1)[-1] for v in s.get("relation", []) if v.startswith("BioSample: ")]
        if len(bios) != 1:
            raise ValueError("Exactly one public BioSample required")
        if acc == "GSE193258":
            unit = ch["cell line"] + ":replicate_" + name.rsplit("_", 1)[1]
            condition = ch["treatment"]
        elif acc == "GSE63577":
            replicate = re.search(r"([123])$", name)
            if not replicate:
                raise ValueError("Missing culture replicate identifier")
            unit = ch["cell type"] + ":replicate_" + replicate[1]
            condition = "PD=" + ch["population doublings"]
        else:
            unit = ch["cell id"]
            condition = "age=" + ch["age"] + ";disease=" + ch["disease"]
        rows.append({"sample_id": s["accession"][0], "biological_unit": unit,
                     "global_unit_id": ("Coriell_or_PRF:" + unit) if acc == "GSE113957" else acc + ":" + unit,
                     "biosample": bios[0], "condition": condition, "matrix_column": name,
                     "sample_title": s["title"][0], "platform_id": s["platform_id"][0]})
    if len(rows) != SOURCES[acc]["n"] or len({r["biosample"] for r in rows}) != len(rows):
        raise ValueError("Unexpected number of samples or duplicated BioSamples")
    return rows, sorted(set(lookup) - set(columns))


def read_matrix(acc, path):
    if acc == "GSE63577":
        import xlrd
        book = xlrd.open_workbook(file_contents=gzip.decompress(Path(path).read_bytes()), on_demand=True)
        sheet = book.sheet_by_name("counts")
        header = sheet.row_values(0)
        columns = header[4:]
        genes, values = [], []
        for i in range(1, sheet.nrows):
            row = sheet.row_values(i)
            genes.append(str(row[0])); values.append(row[4:])
        book.release_resources()
    else:
        offset = 8 if acc == "GSE113957" else 1
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader); columns = header[offset:]
            genes, values = [], []
            for row in reader:
                if len(row) != len(header):
                    raise ValueError("Malformed expression row")
                genes.append(row[0]); values.append(row[offset:])
    x = np.asarray(values, dtype=np.float64).T
    if not genes or len(set(genes)) != len(genes) or any(not g for g in genes):
        raise ValueError("Missing or duplicate feature identifiers")
    if x.shape != (len(columns), len(genes)) or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Invalid matrix values; no imputation")
    if acc != "GSE113957":
        totals = x.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("Zero-depth sample")
        x = x / totals[:, None] * 1e6
    return np.log1p(x), np.array(genes), columns


def fetch_public_panel(destination):
    root = Path(destination).resolve(); root.mkdir(parents=True, exist_ok=True)
    state = root / "status.json"
    write_json(state, {"status": "running", "training_executed": False})
    entries = []
    try:
        for acc, spec in SOURCES.items():
            folder = root / acc; folder.mkdir(exist_ok=True)
            base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/"
            resources = []
            for subdir, name, expected in [("soft", acc + "_family.soft.gz", spec["soft_sha"]), ("suppl", spec["matrix"], spec["matrix_sha"])]:
                url = base + subdir + "/" + name
                download_verified(url, folder / "raw" / name, expected)
                resources.append({"url": url, "sha256": expected})
            provenance = folder / "provenance.json"
            if provenance.exists():
                saved = read_json(provenance)
                for name, digest in saved["derived_sha256"].items():
                    if sha256(folder / name) != digest:
                        raise ValueError(f"Converted artifact changed: {acc}/{name}")
                entry = saved["entry"]
            else:
                if (folder / "matrix.npz").exists():
                    raise FileExistsError("Incomplete conversion; inspect and use a new destination")
                x, genes, columns = read_matrix(acc, folder / "raw" / spec["matrix"])
                rows, excluded = metadata_rows(acc, columns, soft_samples(folder / "raw" / (acc + "_family.soft.gz")))
                np.savez_compressed(folder / "matrix.npz", X=x, genes=genes, samples=np.array([r["sample_id"] for r in rows]))
                with (folder / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                entry = {"dataset_id": acc, "study_id": acc, "parent_dataset": acc,
                         "matrix_path": acc + "/matrix.npz", "metadata_path": acc + "/metadata.csv",
                         "species": "Homo sapiens", "assay": "bulk_RNA_seq", "resolution": "bulk_population",
                         "data_scale": "log_expression", "biological_unit_col": "biological_unit", "condition_col": "condition",
                         "role": spec["role"], "source_kind": "public_real"}
                write_json(provenance, {"entry": entry, "shape": list(x.shape), "resources": resources,
                    "transformation": spec["transform"], "excluded_metadata_titles": excluded,
                    "unit_policy": "Same cell line and replicate across perturbations/PD stays grouped; test donors use deposited cell id",
                    "limitations": ["Study-level generalization design with only one study per split",
                                    "Culture grouping conservative; not independent human donors in cancer/cell-line studies",
                                    "Different annotations and quantification pipelines retained; no cross-platform capability claim",
                                    "GSE63577 is partly reanalysis; GSE64553 and source accessions must not be added as independent studies without lineage audit"],
                    "derived_sha256": {name: sha256(folder / name) for name in ["matrix.npz", "metadata.csv"]}})
            entries.append(entry)
        value = {"datasets": entries, "scope": "Fixed study split; public test numerical preparation deferred until freeze; internal killifish absent"}
        target = root / "manifest.json"
        if target.exists() and read_json(target) != value:
            raise ValueError("Existing manifest changed; use new destination")
        write_json(target, value)
        validated = manifest(target)
        write_json(state, {"status": "completed", "studies": 3, "units": {e["dataset_id"]: e["n_units"] for e in validated},
                           "roles": {acc: spec["role"] for acc, spec in SOURCES.items()}, "training_executed": False})
        return target
    except Exception as exc:
        write_json(state, {"status": "failed", "error": str(exc), "training_executed": False})
        raise
