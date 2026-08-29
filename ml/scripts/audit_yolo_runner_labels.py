#!/usr/bin/env python3
"""Audit runner labels against overlap rules and fresh YOLO pose detections."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
from ultralytics import YOLO


def iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def read_labels(path, width, height):
    boxes = []
    for row in path.read_text("utf-8").splitlines():
        fields = row.split()
        if len(fields) != 5:
            continue
        class_id, cx, cy, bw, bh = map(float, fields)
        boxes.append({
            "class_id": int(class_id),
            "xyxy": [
                (cx - bw / 2.0) * width,
                (cy - bh / 2.0) * height,
                (cx + bw / 2.0) * width,
                (cy + bh / 2.0) * height,
            ],
            "yolo": row,
        })
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pose-model", default="yolov8n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.25)
    parser.add_argument("--duplicate-iou", type=float, default=0.65)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    manifest = json.loads((root / "splits_v1.json").read_text("utf-8"))
    model = YOLO(args.pose_model)
    issues = []
    session_counts = Counter()

    for number, sample in enumerate(manifest["samples"], start=1):
        image_path = Path(sample["image"])
        label_path = Path(sample["label"])
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            issues.append({"type": "image_read_failed", "image": str(image_path)})
            continue
        height, width = frame.shape[:2]
        labels = read_labels(label_path, width, height)
        prediction = model.predict(
            source=frame,
            conf=args.confidence,
            imgsz=args.imgsz,
            device="cpu",
            verbose=False,
        )[0]
        predicted = []
        if prediction.boxes is not None:
            for box, confidence in zip(
                prediction.boxes.xyxy.detach().cpu().tolist(),
                prediction.boxes.conf.detach().cpu().tolist(),
            ):
                predicted.append({
                    "xyxy": [float(value) for value in box],
                    "confidence": float(confidence),
                })

        base = {
            "session": sample["session"],
            "split": sample["split"],
            "image": image_path.as_posix(),
            "label": label_path.as_posix(),
            "frame_index": sample["frame_index"],
        }
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                overlap = iou(labels[left]["xyxy"], labels[right]["xyxy"])
                if overlap >= args.duplicate_iou:
                    issues.append({
                        **base,
                        "type": "overlapping_duplicate_labels",
                        "label_indices": [left, right],
                        "iou": overlap,
                        "labels": [labels[left]["yolo"], labels[right]["yolo"]],
                    })
                    session_counts[(sample["session"], "duplicate")] += 1

        for label_index, label in enumerate(labels):
            best = max(
                (iou(label["xyxy"], item["xyxy"]) for item in predicted),
                default=0.0,
            )
            if best < args.match_iou:
                issues.append({
                    **base,
                    "type": "label_without_strong_person_support",
                    "label_index": label_index,
                    "label": label["yolo"],
                    "best_prediction_iou": best,
                    "predictions": predicted,
                })
                session_counts[(sample["session"], "unsupported")] += 1

        for predicted_index, item in enumerate(predicted):
            best = max(
                (iou(item["xyxy"], label["xyxy"]) for label in labels),
                default=0.0,
            )
            if best < args.match_iou:
                issues.append({
                    **base,
                    "type": "strong_person_without_label",
                    "prediction_index": predicted_index,
                    "prediction": item,
                    "best_label_iou": best,
                })
                session_counts[(sample["session"], "unlabeled_person")] += 1
        if number % 25 == 0:
            print(f"Audited {number}/{len(manifest['samples'])}", flush=True)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_root": root.as_posix(),
        "settings": {
            "pose_model": args.pose_model,
            "imgsz": args.imgsz,
            "confidence": args.confidence,
            "match_iou": args.match_iou,
            "duplicate_iou": args.duplicate_iou,
        },
        "images_audited": len(manifest["samples"]),
        "issue_counts": dict(Counter(item["type"] for item in issues)),
        "session_issue_counts": [
            {"session": key[0], "type": key[1], "count": value}
            for key, value in sorted(session_counts.items())
        ],
        "issues": issues,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["issue_counts"], indent=2))
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
