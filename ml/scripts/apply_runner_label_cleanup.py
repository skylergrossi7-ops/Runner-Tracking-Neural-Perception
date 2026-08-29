#!/usr/bin/env python3
"""Apply a reviewed runner-label cleanup plan with backups and audit logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def yolo_row(box, width, height):
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"


def label_lookup(root):
    result = {}
    for path in root.rglob("*.txt"):
        if path.name in result:
            raise ValueError(f"Duplicate label filename: {path.name}")
        result[path.name] = path
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    plan = json.loads(args.plan.read_text("utf-8"))
    audit = json.loads(args.audit_report.read_text("utf-8"))
    labels = label_lookup(root / "labels")
    manifest_path = root / "splits_v1.json"
    report_path = root / "public_source_report.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    samples_by_label = {Path(item["label"]).name: item for item in manifest["samples"]}
    changes = {}

    def stage_change(label_path, rows, reason):
        before = [row for row in label_path.read_text("utf-8").splitlines() if row.strip()]
        changes[label_path] = {
            "before": before,
            "after": list(rows),
            "reasons": changes.get(label_path, {}).get("reasons", []) + [reason],
        }

    desert = plan["desert_relabel"]
    proposals_path = Path(desert["proposals_json"])
    proposals = json.loads(proposals_path.read_text("utf-8"))
    y_min, y_max = desert["center_y_range"]
    h_min, h_max = desert["height_range"]
    for record in proposals:
        label_name = Path(record["label"]).name
        if desert["session_contains"] not in label_name:
            continue
        sample = samples_by_label[label_name]
        frame = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Unable to read {sample['image']}")
        height, width = frame.shape[:2]
        candidates = []
        for proposal in record["proposals"]:
            box = proposal["xyxy"]
            normalized_width = (box[2] - box[0]) / width
            normalized_height = (box[3] - box[1]) / height
            center_y = ((box[1] + box[3]) / 2.0) / height
            if (
                y_min <= center_y <= y_max
                and h_min <= normalized_height <= h_max
                and normalized_width <= desert["maximum_width"]
            ):
                score = (
                    proposal["confidence"]
                    + desert["height_score_weight"] * normalized_height
                )
                candidates.append((score, proposal))
        if not candidates:
            raise RuntimeError(f"No valid desert runner proposal for {label_name}")
        selected = max(candidates, key=lambda item: item[0])[1]
        stage_change(labels[label_name], [yolo_row(selected["xyxy"], width, height)], desert["reason"])

    unsupported = {
        Path(issue["image"]).with_suffix(".txt").name: issue
        for issue in audit["issues"]
        if issue.get("type") == "label_without_strong_person_support"
        and issue.get("predictions")
    }
    for operation in plan["replace_from_audit_prediction"]:
        name = operation["label_suffix"]
        issue = unsupported.get(name)
        if issue is None:
            raise ValueError(f"Missing audit prediction for {name}")
        sample = samples_by_label[name]
        frame = cv2.imread(sample["image"], cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Unable to read {sample['image']}")
        height, width = frame.shape[:2]
        selected = max(issue["predictions"], key=lambda item: item["confidence"])
        stage_change(labels[name], [yolo_row(selected["xyxy"], width, height)], operation["reason"])

    for operation in plan["keep_indices"]:
        name = operation["label_suffix"]
        label_path = labels[name]
        rows = [row for row in label_path.read_text("utf-8").splitlines() if row.strip()]
        kept = [rows[index] for index in operation["indices"]]
        stage_change(label_path, kept, operation["reason"])

    args.backup_dir.mkdir(parents=True, exist_ok=False)
    audit_log = []
    for label_path, change in sorted(changes.items(), key=lambda item: item[0].name):
        relative = label_path.relative_to(root)
        backup_path = args.backup_dir / relative
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, backup_path)
        label_path.write_text("\n".join(change["after"]) + "\n", encoding="utf-8")
        audit_log.append({
            "label": label_path.as_posix(),
            "backup": backup_path.as_posix(),
            **change,
        })

    for sample in manifest["samples"]:
        label_path = Path(sample["label"])
        sample["runner_count"] = sum(
            1 for row in label_path.read_text("utf-8").splitlines() if row.strip()
        )
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "action": "runner_label_cleanup",
        "changed_label_files": len(audit_log),
        "boxes_before": sum(len(item["before"]) for item in audit_log),
        "boxes_after": sum(len(item["after"]) for item in audit_log),
        "plan": args.plan.as_posix(),
        "audit_report": args.audit_report.as_posix(),
    }
    manifest.setdefault("curation", []).append(event)
    atomic_json(manifest_path, manifest)

    source_report = json.loads(report_path.read_text("utf-8"))
    source_report["accepted_runner_boxes"] = sum(
        item["runner_count"] for item in manifest["samples"]
    )
    source_report["accepted_images_by_session"] = dict(sorted(Counter(
        item["session"] for item in manifest["samples"]
    ).items()))
    source_report.setdefault("curation", []).append(event)
    atomic_json(report_path, source_report)
    (root / "dataset_report.json").unlink(missing_ok=True)

    payload = {"schema_version": 1, **event, "changes": audit_log}
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_log, payload)
    print(json.dumps(event, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
