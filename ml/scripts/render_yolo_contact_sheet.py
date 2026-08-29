#!/usr/bin/env python3
"""Render annotated, paginated contact sheets from a YOLO dataset."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def render_card(sample, card_width=480, card_height=270, title_height=34):
    """Load one sample and render its labels plus audit-friendly identity."""
    image_path = Path(sample["image"])
    label_path = Path(sample["label"])
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Unable to read {image_path}")
    height, width = frame.shape[:2]
    box_count = 0
    for row in label_path.read_text("utf-8").splitlines():
        fields = row.split()
        if len(fields) != 5:
            continue
        _, cx, cy, bw, bh = map(float, fields)
        x1 = int((cx - bw / 2) * width)
        y1 = int((cy - bh / 2) * height)
        x2 = int((cx + bw / 2) * width)
        y2 = int((cy + bh / 2) * height)
        cv2.rectangle(
            frame, (x1, y1), (x2, y2), (0, 0, 255),
            max(5, width // 450),
        )
        box_count += 1
    resized = cv2.resize(
        frame, (card_width, card_height), interpolation=cv2.INTER_AREA
    )
    card = np.zeros(
        (card_height + title_height, card_width, 3), dtype=np.uint8
    )
    card[title_height:] = resized
    session = sample["session"].removeprefix("local_pexels_")
    title = (
        f"{sample['split']} | {session} | frame {sample['frame_index']} "
        f"| boxes {box_count}"
    )
    cv2.putText(
        card, title[:76], (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return card


def render_sheet(samples, columns):
    """Render one rectangular sheet, padding its final row if necessary."""
    cards = [render_card(sample) for sample in samples]
    rows = math.ceil(len(cards) / columns)
    blank = np.zeros_like(cards[0])
    while len(cards) < rows * columns:
        cards.append(blank.copy())
    return cv2.vconcat([
        cv2.hconcat(cards[row * columns:(row + 1) * columns])
        for row in range(rows)
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    destinations = parser.add_mutually_exclusive_group(required=True)
    destinations.add_argument(
        "--output", type=Path,
        help="Single sheet containing one representative image per session.",
    )
    destinations.add_argument(
        "--output-dir", type=Path,
        help="Directory for paginated sheets containing every selected image.",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=24)
    parser.add_argument("--split", choices=("all", "train", "val"), default="all")
    parser.add_argument(
        "--session-contains",
        help="Optionally render only sessions containing this text.",
    )
    parser.add_argument(
        "--minimum-boxes", type=int, default=0,
        help="Optionally render only images with at least this many boxes.",
    )
    parser.add_argument(
        "--audit-report", type=Path,
        help="Optionally restrict images to those named in an audit report.",
    )
    parser.add_argument(
        "--issue-type", action="append",
        help="With --audit-report, include only these issue types.",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    manifest = json.loads((root / "splits_v1.json").read_text("utf-8"))
    audited_images = None
    if args.audit_report is not None:
        audit = json.loads(args.audit_report.read_text("utf-8"))
        issue_types = set(args.issue_type or ())
        audited_images = {
            Path(issue["image"]).name for issue in audit.get("issues", [])
            if "image" in issue
            and (not issue_types or issue.get("type") in issue_types)
        }
    selected = [
        sample for sample in manifest["samples"]
        if (args.split == "all" or sample["split"] == args.split)
        and (
            not args.session_contains
            or args.session_contains in sample["session"]
        )
        and sum(
            1 for row in Path(sample["label"]).read_text("utf-8").splitlines()
            if row.strip()
        ) >= args.minimum_boxes
        and (audited_images is None or Path(sample["image"]).name in audited_images)
    ]
    if not selected:
        raise ValueError(f"No samples selected for split {args.split}")
    selected.sort(key=lambda item: (
        item["split"], item["session"], item["frame_index"]
    ))

    if args.output_dir is not None:
        if args.page_size < 1:
            raise ValueError("--page-size must be positive")
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        page_records = []
        for start in range(0, len(selected), args.page_size):
            page_samples = selected[start:start + args.page_size]
            page_number = len(page_records) + 1
            page_path = output_dir / f"contact_sheet_{page_number:03d}.jpg"
            sheet = render_sheet(page_samples, max(1, args.columns))
            if not cv2.imwrite(str(page_path), sheet):
                raise RuntimeError(f"Unable to write {page_path}")
            page_records.append({
                "page": page_number,
                "file": page_path.as_posix(),
                "sample_count": len(page_samples),
                "images": [Path(item["image"]).name for item in page_samples],
            })
        index = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "dataset_root": root.as_posix(),
            "split": args.split,
            "images": len(selected),
            "pages": page_records,
        }
        (output_dir / "contact_sheet_index.json").write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Rendered {len(selected)} images across {len(page_records)} sheets")
        print(f"Review folder: {output_dir}")
        return 0

    grouped = defaultdict(list)
    for sample in selected:
        grouped[sample["session"]].append(sample)
    representatives = []
    for session, samples in sorted(grouped.items()):
        representatives.append(sorted(
            samples, key=lambda item: item["frame_index"]
        )[len(samples) // 2])
    sheet = render_sheet(representatives, max(1, args.columns))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"Unable to write {args.output}")
    print(f"Rendered {len(grouped)} sessions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
