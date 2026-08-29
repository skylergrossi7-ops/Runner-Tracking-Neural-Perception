#!/usr/bin/env python3
"""Generate review-required YOLO runner labels from a teacher detector."""

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.webp'}


def format_detection(class_id, xywhn):
    """Format one normalized detection as a YOLO label row."""
    values = [float(value) for value in xywhn]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError('Detection must contain four finite xywh values')
    x, y, width, height = values
    if not (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    ):
        raise ValueError(f'Invalid normalized detection: {values}')
    return f'{int(class_id)} {x:.8f} {y:.8f} {width:.8f} {height:.8f}'


def parse_arguments(argv=None):
    """Parse model-assisted annotation settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--session', action='append', default=[])
    parser.add_argument('--source-class', type=int, default=0)
    parser.add_argument('--target-class', type=int, default=0)
    parser.add_argument('--confidence', type=float, default=0.15)
    parser.add_argument('--review-threshold', type=float, default=0.35)
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--review-manifest', default='annotation_review.json')
    return parser.parse_args(argv)


def _atomic_json(path, payload):
    """Atomically write a JSON review artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def _selected_images(dataset_root, requested_sessions):
    """Return source images from requested session directories."""
    image_root = dataset_root / 'images'
    if not image_root.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_root}')
    sessions = requested_sessions or sorted(
        path.name for path in image_root.iterdir() if path.is_dir()
    )
    images = []
    for session in sessions:
        directory = image_root / session
        if not directory.is_dir():
            raise FileNotFoundError(f'Session not found: {directory}')
        images.extend(
            path for path in sorted(directory.rglob('*'))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not images:
        raise ValueError('No images were selected for pre-annotation')
    return images


def preannotate(arguments):
    """Run teacher inference, write provisional labels, and queue review."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            f'Ultralytics is required for pre-annotation: {error}'
        ) from error

    dataset_root = arguments.dataset_root.expanduser().resolve()
    images = _selected_images(dataset_root, arguments.session)
    pending = []
    skipped = []
    for image in images:
        relative = image.relative_to(dataset_root / 'images')
        label = (dataset_root / 'labels' / relative).with_suffix('.txt')
        if label.exists() and not arguments.overwrite:
            skipped.append(image)
        else:
            pending.append(image)
    if not pending:
        raise ValueError(
            'Every selected image already has a label; use --overwrite to '
            'replace provisional labels'
        )

    model = YOLO(arguments.model)
    results = model.predict(
        source=[str(path) for path in pending],
        classes=[arguments.source_class],
        conf=arguments.confidence,
        imgsz=arguments.image_size,
        device=arguments.device,
        stream=True,
        verbose=False,
    )
    review_frames = []
    frame_records = []
    confidence_values = []
    for image, result in zip(pending, results):
        relative = image.relative_to(dataset_root / 'images')
        label = (dataset_root / 'labels' / relative).with_suffix('.txt')
        label.parent.mkdir(parents=True, exist_ok=True)
        detections = []
        if result.boxes is not None:
            normalized = result.boxes.xywhn.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            for xywhn, confidence in zip(normalized, confidences):
                detections.append({
                    'confidence': float(confidence),
                    'xywhn': [float(value) for value in xywhn],
                })
        detections.sort(key=lambda item: item['confidence'], reverse=True)
        rows = [
            format_detection(arguments.target_class, item['xywhn'])
            for item in detections
        ]
        label.write_text(
            ''.join(f'{row}\n' for row in rows), encoding='utf-8'
        )
        confidence_values.extend(
            item['confidence'] for item in detections
        )
        reasons = []
        if not detections:
            reasons.append('no_detection_verify_negative')
        if any(
            item['confidence'] < arguments.review_threshold
            for item in detections
        ):
            reasons.append('low_confidence')
        if len(detections) > 1:
            reasons.append('multiple_people_select_runner')
        record = {
            'image': image.as_posix(),
            'label': label.as_posix(),
            'detections': detections,
            'review_reasons': reasons,
        }
        frame_records.append(record)
        if reasons:
            review_frames.append(record)

    report_path = Path(arguments.review_manifest)
    if not report_path.is_absolute():
        report_path = dataset_root / report_path
    report = {
        'schema_version': 1,
        'teacher_model': arguments.model,
        'source_class': arguments.source_class,
        'target_class': arguments.target_class,
        'confidence_threshold': arguments.confidence,
        'review_threshold': arguments.review_threshold,
        'processed_images': len(frame_records),
        'skipped_existing_labels': len(skipped),
        'review_required_images': len(review_frames),
        'detections': sum(
            len(record['detections']) for record in frame_records
        ),
        'confidence': {
            'minimum': min(confidence_values, default=None),
            'median': (
                statistics.median(confidence_values)
                if confidence_values else None
            ),
            'maximum': max(confidence_values, default=None),
        },
        'review_reason_counts': dict(Counter(
            reason for record in review_frames
            for reason in record['review_reasons']
        )),
        'review_queue': review_frames,
        'frames': frame_records,
        'status': 'provisional_labels_require_human_review',
    }
    _atomic_json(report_path, report)
    metadata_path = dataset_root / 'session_metadata.json'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text('utf-8'))
        selected_sessions = {
            image.relative_to(dataset_root / 'images').parts[0]
            for image in pending
        }
        for session in selected_sessions:
            record = metadata.get('sessions', {}).get(session)
            if record is not None:
                record['annotation_status'] = (
                    'preannotated_requires_human_review'
                )
        _atomic_json(metadata_path, metadata)
    return report_path, report


def main(argv=None):
    """Generate provisional runner labels for human correction."""
    arguments = parse_arguments(argv)
    try:
        report_path, report = preannotate(arguments)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    print(
        f'Pre-annotated {report["processed_images"]} images; '
        f'{report["review_required_images"]} require priority review.'
    )
    print(f'Review manifest: {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
