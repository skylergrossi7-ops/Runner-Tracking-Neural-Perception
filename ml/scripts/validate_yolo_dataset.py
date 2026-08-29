#!/usr/bin/env python3
"""Validate YOLO labels, source images, sessions, and annotation coverage."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.webp'}


def validate_label_line(line, class_count):
    """Validate one YOLO detection row and return its numeric values."""
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f'expected 5 fields, found {len(fields)}')
    try:
        class_value = float(fields[0])
        coordinates = [float(value) for value in fields[1:]]
    except ValueError as error:
        raise ValueError('all fields must be numeric') from error
    if not class_value.is_integer():
        raise ValueError('class ID must be an integer')
    class_id = int(class_value)
    if not 0 <= class_id < class_count:
        raise ValueError(
            f'class ID {class_id} outside configured range '
            f'0..{class_count - 1}'
        )
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError('coordinates must be finite')
    x, y, width, height = coordinates
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError('box center must be normalized to [0, 1]')
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        raise ValueError(
            'box width and height must be normalized and positive'
        )
    tolerance = 1.0e-6
    if (
        x - width / 2.0 < -tolerance
        or x + width / 2.0 > 1.0 + tolerance
        or y - height / 2.0 < -tolerance
        or y + height / 2.0 > 1.0 + tolerance
    ):
        raise ValueError('box extends outside normalized image boundaries')
    return class_id, coordinates


def parse_arguments(argv=None):
    """Parse dataset QA options."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--class-count', type=int, default=1)
    parser.add_argument('--allow-unlabeled', action='store_true')
    parser.add_argument(
        '--mark-reviewed',
        action='store_true',
        help=(
            'After a passing validation, record explicit human-review '
            'attestation for every discovered session.'
        ),
    )
    parser.add_argument('--report', default='dataset_report.json')
    return parser.parse_args(argv)


def _sha256(path):
    """Return the SHA-256 digest of a source image."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(arguments):
    """Inspect all image/label pairs and return a machine-readable report."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            f'OpenCV is required for image QA: {error}'
        ) from error

    root = arguments.dataset_root.expanduser().resolve()
    image_root = root / 'images'
    label_root = root / 'labels'
    if arguments.class_count < 1:
        raise ValueError('--class-count must be positive')
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            f'Expected images/ and labels/ directories below {root}'
        )
    images = sorted(
        path for path in image_root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f'No supported images found below {image_root}')

    errors = []
    warnings = []
    class_counts = Counter()
    per_session = defaultdict(lambda: {
        'images': 0, 'labeled_images': 0, 'negative_images': 0, 'objects': 0
    })
    digest_paths = defaultdict(list)
    widths = []
    heights = []
    unlabeled = 0
    empty_labels = 0
    objects = 0
    for image in images:
        relative = image.relative_to(image_root)
        session = relative.parts[0] if len(relative.parts) > 1 else 'flat'
        stats = per_session[session]
        stats['images'] += 1
        decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if decoded is None:
            errors.append({
                'image': image.as_posix(),
                'error': 'decode_failed',
            })
            continue
        height, width = decoded.shape[:2]
        widths.append(int(width))
        heights.append(int(height))
        digest_paths[_sha256(image)].append(image.as_posix())

        label = (label_root / relative).with_suffix('.txt')
        if not label.is_file():
            unlabeled += 1
            if not arguments.allow_unlabeled:
                errors.append({
                    'image': image.as_posix(),
                    'error': 'missing_label',
                })
            continue
        stats['labeled_images'] += 1
        lines = [
            line.strip() for line in label.read_text('utf-8').splitlines()
            if line.strip()
        ]
        if not lines:
            empty_labels += 1
            stats['negative_images'] += 1
        for line_number, line in enumerate(lines, start=1):
            try:
                class_id, _coordinates = validate_label_line(
                    line, arguments.class_count
                )
            except ValueError as error:
                errors.append({
                    'label': label.as_posix(),
                    'line': line_number,
                    'error': str(error),
                })
                continue
            class_counts[class_id] += 1
            objects += 1
            stats['objects'] += 1

    duplicate_groups = [
        paths for paths in digest_paths.values() if len(paths) > 1
    ]
    if duplicate_groups:
        warnings.append({
            'warning': 'exact_duplicate_images',
            'groups': duplicate_groups,
        })
    metadata_path = root / 'session_metadata.json'
    metadata = None
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text('utf-8'))
        known = set(metadata.get('sessions', {}))
        discovered = set(per_session)
        missing_metadata = sorted(discovered - known)
        if missing_metadata:
            warnings.append({
                'warning': 'sessions_missing_metadata',
                'sessions': missing_metadata,
            })

    report = {
        'schema_version': 1,
        'dataset_root': root.as_posix(),
        'passed': not errors,
        'images': len(images),
        'labeled_images': len(images) - unlabeled,
        'unlabeled_images': unlabeled,
        'negative_images': empty_labels,
        'objects': objects,
        'class_counts': {
            str(class_id): count
            for class_id, count in sorted(class_counts.items())
        },
        'image_dimensions': {
            'minimum_width': min(widths, default=None),
            'maximum_width': max(widths, default=None),
            'minimum_height': min(heights, default=None),
            'maximum_height': max(heights, default=None),
        },
        'sessions': dict(sorted(per_session.items())),
        'duplicate_image_groups': len(duplicate_groups),
        'errors': errors,
        'warnings': warnings,
        'session_metadata_present': metadata is not None,
    }
    report_path = Path(arguments.report)
    if not report_path.is_absolute():
        report_path = root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + '.tmp')
    temporary.write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8'
    )
    temporary.replace(report_path)

    if arguments.mark_reviewed:
        if not report['passed']:
            raise ValueError(
                'Cannot mark annotations reviewed while validation errors '
                'exist'
            )
        if arguments.allow_unlabeled:
            raise ValueError(
                'Cannot mark annotations reviewed with --allow-unlabeled'
            )
        if not metadata_path.is_file():
            raise ValueError(
                'Cannot mark annotations reviewed without '
                'session_metadata.json'
            )
        metadata = json.loads(metadata_path.read_text('utf-8'))
        for session in per_session:
            record = metadata.get('sessions', {}).get(session)
            if record is None:
                raise ValueError(
                    f'Cannot mark unknown session {session!r} reviewed'
                )
            record['annotation_status'] = 'human_reviewed'
        metadata_temporary = metadata_path.with_name(
            metadata_path.name + '.tmp'
        )
        metadata_temporary.write_text(
            json.dumps(metadata, indent=2) + '\n', encoding='utf-8'
        )
        metadata_temporary.replace(metadata_path)
    return report_path, report


def main(argv=None):
    """Validate the raw or reviewed runner dataset."""
    arguments = parse_arguments(argv)
    try:
        report_path, report = validate_dataset(arguments)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    status = 'PASS' if report['passed'] else 'FAIL'
    print(
        f'{status}: {report["images"]} images, {report["objects"]} objects, '
        f'{report["unlabeled_images"]} unlabeled images.'
    )
    print(f'Dataset report: {report_path}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
