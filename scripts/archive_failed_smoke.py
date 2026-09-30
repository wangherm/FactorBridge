"""Preserve a confirmed failed smoke attempt before an explicitly requested retry."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from factorbridge.io import read_json, write_json, sha256


def archive_failed_smoke(config_path):
    config_path = Path(config_path).resolve()
    c = read_json(config_path)
    run = Path(c["run_dir"]).resolve()
    if config_path.parent != run:
        raise ValueError("Use the original config.json inside its run directory")
    if any((run / name).exists() for name in ("training", "frozen_protocol.json", "test_prepared", "internal_prepared")):
        raise ValueError("Run has progressed beyond smoke; cannot retry here")
    source = run / "smoke"
    if source.is_symlink() or source.resolve().parent != run:
        raise ValueError("Smoke source must stay inside the original run")
    status = read_json(source / "status.json")
    if status.get("status") != "failed":
        raise ValueError("Only status=failed may be archived; never move a running or passed smoke")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    target = run / "smoke_attempts" / ("failed_" + stamp)
    if not target.resolve().is_relative_to(run) or target.exists():
        raise ValueError("Archive target must be a new path inside this run")
    receipt = {"archived_utc": datetime.now(timezone.utc).isoformat(), "source": str(source), "archive": str(target),
               "previous_status": status, "previous_status_sha256": sha256(source / "status.json"),
               "config_sha256": sha256(config_path), "old_adapter_reused_for_training": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    write_json(target / "archive_receipt.json", receipt)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("Preserved failed smoke:", archive_failed_smoke(args.config))
