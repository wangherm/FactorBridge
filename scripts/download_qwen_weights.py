"""Fetch official ModelScope bytes into the exact pinned HF snapshot, with SHA256 checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

from factorbridge.io import read_json, write_json, sha256

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
HF_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MS_REVISION = "2de2439ea21be1dc5cb21f22f88af07e43393cbb"


def manifest():
    spec = read_json(Path(__file__).resolve().parents[1] / "configs/qwen3_4b_snapshot.json")
    if (spec["model_name"], spec["hf_revision"], spec["modelscope_revision"]) != (MODEL, HF_REVISION, MS_REVISION):
        raise ValueError("Unexpected model snapshot identity")
    names = set()
    for item in spec["files"]:
        name = item["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."} or name in names:
            raise ValueError("Invalid or duplicate snapshot filename")
        if not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]) or item["size"] < 1:
            raise ValueError("Invalid file size or SHA256")
        names.add(name)
    return spec


def verify_file(path, item):
    if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
        raise ValueError(f"File size/SHA256 mismatch: {path}; retained for investigation, not replaced")


def download_file(item, directory, revision=MS_REVISION, verify_only=False, opener=urllib.request.urlopen):
    path = Path(directory) / item["name"]
    url = "https://modelscope.cn/api/v1/models/" + MODEL + "/repo?" + urllib.parse.urlencode(
        {"Revision": revision, "FilePath": item["name"]})
    if path.exists():
        verify_file(path, item)
        print("VERIFIED CACHE " + path.name, flush=True)
        return {"name": path.name, "status": "verified_cache", "sha256": item["sha256"]}
    if verify_only:
        raise FileNotFoundError(path)
    part = path.with_name(path.name + ".part")
    for attempt in range(1, 4):
        offset = part.stat().st_size if part.exists() else 0
        if offset > item["size"]:
            raise ValueError(f"Oversized partial download: {part}")
        if offset == item["size"]:
            break
        headers = {"User-Agent": "FactorBridge-verified-weights/0.1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(f"DOWNLOAD {path.name} from {offset}/{item['size']} bytes, attempt {attempt}", flush=True)
        try:
            with opener(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != item["size"]:
                        raise ValueError("Unexpected HTTP range; refusing to append mismatched bytes")
                elif response.status == 200:
                    if offset:
                        print(f"Server ignored Range; explicitly restarting partial file {path.name}", flush=True)
                    offset = 0
                else:
                    raise OSError(f"Unexpected HTTP status {response.status}")
                last = time.monotonic()
                with part.open("ab" if offset else "wb") as stream:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        offset += len(chunk)
                        if offset > item["size"]:
                            raise ValueError(f"Response larger than fixed snapshot file: {path.name}")
                        if time.monotonic() - last >= 10:
                            print(f"{path.name}: {offset / item['size']:.1%}", flush=True)
                            last = time.monotonic()
            if offset != item["size"]:
                raise OSError(f"Incomplete response: {offset}/{item['size']} bytes")
            break
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as exc:
            print(f"NETWORK FAILURE {path.name}: {exc}; partial file retained", flush=True)
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
    verify_file(part, item)
    part.replace(path)
    print("VERIFIED DOWNLOAD " + path.name, flush=True)
    return {"name": path.name, "status": "downloaded_and_verified", "url": url, "sha256": item["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="Explicit diagnostic: skip weights and report NOT ready for model loading")
    args = parser.parse_args()
    spec = manifest()
    home = Path(args.hf_home).resolve()
    target = home / "hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / HF_REVISION
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    report_path = home / "factorbridge_downloads" / (stamp + ".json")
    items = [r for r in spec["files"] if not args.metadata_only or not r["name"].endswith(".safetensors")]
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "snapshot": str(target),
              "model_name": MODEL, "hf_revision": HF_REVISION, "modelscope_revision": MS_REVISION,
              "metadata_only": args.metadata_only, "model_weights_loaded": False, "qwen_training_executed": False, "files": []}
    write_json(report_path, report)
    try:
        needed = sum(max(0, r["size"] - ((target / (r["name"] + ".part")).stat().st_size
                     if (target / (r["name"] + ".part")).exists() else 0))
                     for r in items if not (target / r["name"]).exists())
        if not args.verify_only and shutil.disk_usage(target).free < needed + 1024**3:
            raise OSError(f"Insufficient free space: need download remainder {needed} bytes plus 1 GiB")
        print(f"Snapshot {HF_REVISION}; remaining download approximately {needed / 1e9:.2f} GB", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(download_file, r, target, MS_REVISION, args.verify_only) for r in items]
            for future in as_completed(futures):
                report["files"].append(future.result())
                write_json(report_path, report)
        if not args.metadata_only:
            expected_shards = {r["name"] for r in items if r["name"].endswith(".safetensors")}
            actual_shards = set(read_json(target / "model.safetensors.index.json")["weight_map"].values())
            if actual_shards != expected_shards:
                raise ValueError("Weight index/shard manifest mismatch")
        report["status"] = "metadata_only_not_ready" if args.metadata_only else "weights_verified_ready_for_smoke"
        print(json.dumps({"status": report["status"], "report": str(report_path), "snapshot": str(target)}, indent=2))
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)


if __name__ == "__main__":
    main()
