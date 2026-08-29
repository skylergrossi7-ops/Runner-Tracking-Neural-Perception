#!/usr/bin/env python3
"""Render detector proposals beside current labels without modifying a dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def normalized_box(row, width, height):
    _, cx, cy, bw, bh = map(float, row.split())
    return [
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        (cx + bw / 2) * width,
        (cy + bh / 2) * height,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--session-contains", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.01)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    manifest = json.loads((root / "splits_v1.json").read_text("utf-8"))
    samples = sorted(
        (
            item for item in manifest["samples"]
            if args.session_contains in item["session"]
        ),
        key=lambda item: item["frame_index"],
    )
    model = YOLO(args.model)
    records, cards = [], []
    for sample in samples:
        image_path = Path(sample["image"])
        label_path = Path(sample["label"])
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Unable to read {image_path}")
        height, width = frame.shape[:2]
        old_boxes = [
            normalized_box(row, width, height)
            for row in label_path.read_text("utf-8").splitlines()
            if row.strip()
        ]
        prediction = model.predict(
            source=frame, classes=[0], conf=args.confidence,
            imgsz=args.imgsz, device="cpu", verbose=False,
        )[0]
        proposals = []
        if prediction.boxes is not None:
            for box, confidence in zip(
                prediction.boxes.xyxy.detach().cpu().tolist(),
                prediction.boxes.conf.detach().cpu().tolist(),
            ):
                proposals.append({
                    "xyxy": [float(value) for value in box],
                    "confidence": float(confidence),
                })
        overlay = frame.copy()
        for box in old_boxes:
            cv2.rectangle(
                overlay, (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])), (0, 0, 255),
                max(5, width // 450),
            )
        for proposal in proposals:
            box = proposal["xyxy"]
            cv2.rectangle(
                overlay, (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])), (0, 255, 0),
                max(5, width // 450),
            )
            cv2.putText(
                overlay, f"{proposal['confidence']:.2f}",
                (int(box[0]), max(20, int(box[1]) - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                cv2.LINE_AA,
            )
        resized = cv2.resize(overlay, (640, 360), interpolation=cv2.INTER_AREA)
        card = np.zeros((400, 640, 3), dtype=np.uint8)
        card[40:] = resized
        cv2.putText(
            card,
            f"frame {sample['frame_index']} | red=current green=proposal",
            (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        cards.append(card)
        records.append({
            "image": image_path.as_posix(),
            "label": label_path.as_posix(),
            "frame_index": sample["frame_index"],
            "old_boxes": old_boxes,
            "proposals": proposals,
        })

    columns = 3
    rows = math.ceil(len(cards) / columns)
    blank = np.zeros_like(cards[0])
    while len(cards) < rows * columns:
        cards.append(blank.copy())
    sheet = cv2.vconcat([
        cv2.hconcat(cards[row * columns:(row + 1) * columns])
        for row in range(rows)
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / "proposals.jpg"), sheet)
    (args.output_dir / "proposals.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Rendered {len(records)} proposal comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
