#!/usr/bin/env python3
"""Evaluate a trained YOLO runner detector on a video and save review artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--label-prefix", default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def load_ground_truth(path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            continue
        cx, cy, bw, bh = (float(value) for value in fields[1:])
        boxes.append([
            (cx - bw / 2.0) * width,
            (cy - bh / 2.0) * height,
            (cx + bw / 2.0) * width,
            (cy + bh / 2.0) * height,
        ])
    return boxes


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def main() -> None:
    args = parse_args()
    for path in (args.weights, args.video):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    maximum_frames = source_frames
    if args.max_seconds > 0.0:
        maximum_frames = min(source_frames, int(round(args.max_seconds * fps)))

    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {args.output_video}")

    model = YOLO(str(args.weights))
    # Ultralytics may replace a one-class dataset name with ``item`` when an
    # older run used single_cls=True. Preserve the project's semantic label in
    # review videos and downstream results.
    model.model.names = {0: "runner"}
    inference_ms: list[float] = []
    confidences: list[float] = []
    detected_frames = 0
    current_gap = 0
    longest_gap = 0
    labeled_frames = 0
    matched_labeled_frames = 0
    labeled_ious: list[float] = []
    processed = 0
    started = time.perf_counter()

    while processed < maximum_frames:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        inference_ms.append(float(result.speed.get("inference", 0.0)))
        boxes = result.boxes
        predicted = boxes.xyxy.cpu().tolist() if boxes is not None else []
        frame_confidences = boxes.conf.cpu().tolist() if boxes is not None else []
        if predicted:
            detected_frames += 1
            current_gap = 0
            confidences.extend(float(value) for value in frame_confidences)
        else:
            current_gap += 1
            longest_gap = max(longest_gap, current_gap)

        if args.labels_dir:
            label = args.labels_dir / f"{args.label_prefix}_{processed:07d}.txt"
            truth = load_ground_truth(label, width, height)
            if truth:
                labeled_frames += 1
                best_iou = max((box_iou(p, t) for p in predicted for t in truth), default=0.0)
                labeled_ious.append(best_iou)
                if best_iou >= 0.50:
                    matched_labeled_frames += 1

        annotated = result.plot(line_width=max(2, round(width / 640)))
        status = f"runner detected | conf {max(frame_confidences):.2f}" if frame_confidences else "runner not detected"
        color = (0, 200, 0) if frame_confidences else (0, 0, 255)
        cv2.putText(annotated, status, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        writer.write(annotated)
        processed += 1

    elapsed = time.perf_counter() - started
    capture.release()
    writer.release()

    report = {
        "weights": str(args.weights.resolve()),
        "video": str(args.video.resolve()),
        "evaluation_scope": "session-held-out validation recording",
        "configuration": {
            "confidence_threshold": args.conf,
            "iou_threshold": args.iou,
            "inference_size": args.imgsz,
            "device": args.device,
            "max_seconds": args.max_seconds,
        },
        "video": {
            "fps": fps,
            "width": width,
            "height": height,
            "processed_frames": processed,
            "processed_seconds": processed / fps if fps else 0.0,
        },
        "detection": {
            "detected_frames": detected_frames,
            "frame_detection_rate": detected_frames / processed if processed else 0.0,
            "mean_confidence": statistics.fmean(confidences) if confidences else 0.0,
            "minimum_confidence": min(confidences, default=0.0),
            "maximum_confidence": max(confidences, default=0.0),
            "longest_detection_gap_frames": longest_gap,
            "longest_detection_gap_seconds": longest_gap / fps if fps else 0.0,
        },
        "labeled_frame_check": {
            "labeled_frames_evaluated": labeled_frames,
            "matched_at_iou_0_50": matched_labeled_frames,
            "match_rate": matched_labeled_frames / labeled_frames if labeled_frames else None,
            "mean_best_iou": statistics.fmean(labeled_ious) if labeled_ious else None,
        },
        "performance": {
            "wall_seconds": elapsed,
            "effective_processing_fps": processed / elapsed if elapsed else 0.0,
            "mean_inference_ms": statistics.fmean(inference_ms) if inference_ms else 0.0,
            "p95_inference_ms": percentile(inference_ms, 0.95),
        },
        "artifacts": {
            "annotated_video": str(args.output_video.resolve()),
            "report": str(args.output_json.resolve()),
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
