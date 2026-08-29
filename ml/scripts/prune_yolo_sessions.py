#!/usr/bin/env python3
"""Remove visually rejected source sessions from a generated YOLO dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Refusing path outside dataset root: {resolved}")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--session", action="append", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    split_path = root / "splits_v1.json"
    report_path = root / "public_source_report.json"
    manifest = json.loads(split_path.read_text("utf-8"))
    rejected = set(args.session)
    known = set(manifest["session_splits"])
    unknown = sorted(rejected - known)
    if unknown:
        raise ValueError(f"Unknown sessions: {', '.join(unknown)}")

    kept, removed = [], []
    for sample in manifest["samples"]:
        if sample["session"] not in rejected:
            kept.append(sample)
            continue
        image = contained(Path(sample["image"]), root)
        label = contained(Path(sample["label"]), root)
        image.unlink(missing_ok=True)
        label.unlink(missing_ok=True)
        removed.append(sample)

    manifest["samples"] = kept
    manifest["session_splits"] = {
        key: value for key, value in manifest["session_splits"].items()
        if key not in rejected
    }
    manifest["split_counts"] = dict(Counter(item["split"] for item in kept))
    manifest.setdefault("curation", []).append({
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "action": "remove_sessions",
        "sessions": sorted(rejected),
        "reason": args.reason,
        "removed_images": len(removed),
        "removed_boxes": sum(item["runner_count"] for item in removed),
    })
    atomic_json(split_path, manifest)

    report = json.loads(report_path.read_text("utf-8"))
    report["accepted_images"] = len(kept)
    report["accepted_runner_boxes"] = sum(item["runner_count"] for item in kept)
    report["accepted_images_by_session"] = dict(sorted(Counter(
        item["session"] for item in kept
    ).items()))
    report["splits"] = manifest["split_counts"]
    report.setdefault("curation", []).append(manifest["curation"][-1])
    report.setdefault("rejections", {})["manual_false_positive_session"] = (
        report.get("rejections", {}).get("manual_false_positive_session", 0)
        + len(removed)
    )
    atomic_json(report_path, report)
    (root / "dataset_report.json").unlink(missing_ok=True)
    print(f"Removed {len(removed)} images from {len(rejected)} sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
