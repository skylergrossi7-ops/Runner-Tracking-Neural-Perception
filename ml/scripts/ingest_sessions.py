#!/usr/bin/env python3
"""Batch-ingest independent ROS 2 bags into the runner_raw dataset."""

import argparse
import json
import re
import sys
from pathlib import Path

from extract_rosbag_images import (
    _load_session_metadata,
    bag_digest,
    extract,
)


STORAGE_EXTENSIONS = {'.db3', '.mcap'}
SESSION_PATTERN = re.compile(r'^bag_(?P<number>\d{3,})$')
CONDITION_CHOICES = {
    'distance': ('near', 'mid', 'far'),
    'lighting': ('normal', 'backlit', 'low'),
    'occlusion': ('none', 'partial'),
}
CONDITION_ALIASES = {
    'lighting': {'lowlight': 'low', 'day': 'normal'},
    'occlusion': {'clear': 'none', 'occluded': 'partial'},
}


def parse_arguments(argv=None):
    """Parse batch-ingestion settings."""
    parser = argparse.ArgumentParser(
        description=(
            'Discover ROS 2 bag folders, extract independent image sessions, '
            'and update runner_raw/session_metadata.json.'
        )
    )
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--topic', default='/camera/image_raw')
    parser.add_argument('--every-nth-frame', type=int, default=5)
    parser.add_argument('--maximum-frames', type=int, default=0)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument(
        '--metadata-file',
        type=Path,
        help=(
            'Optional JSON mapping bag folder names or relative paths to '
            'distance, lighting, and occlusion values.'
        ),
    )
    parser.add_argument(
        '--non-interactive',
        action='store_true',
        help='Fail rather than prompt when a bag has no metadata entry.',
    )
    parser.add_argument(
        '--environment',
        choices=('real', 'simulation'),
        default='real',
    )
    parser.add_argument('--report', default='ingestion_report.json')
    return parser.parse_args(argv)


def scan_bag_directories(input_dir):
    """Find directories that directly contain rosbag2 storage files."""
    root = input_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Input directory not found: {root}')
    directories = {
        path.parent
        for path in root.rglob('*')
        if path.is_file() and path.suffix.lower() in STORAGE_EXTENSIONS
    }
    if any(
        path.is_file() and path.suffix.lower() in STORAGE_EXTENSIONS
        for path in root.iterdir()
    ):
        directories.add(root)
    return sorted(directories, key=lambda path: path.as_posix().lower())


def load_metadata_file(path):
    """Load optional batch tag definitions."""
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f'Metadata file not found: {resolved}')
    payload = json.loads(resolved.read_text('utf-8'))
    records = payload.get('sessions', payload)
    if not isinstance(records, dict):
        raise ValueError('Metadata JSON must contain a session mapping')
    return records


def normalize_metadata(record):
    """Validate and normalize one session's environmental tags."""
    if not isinstance(record, dict):
        raise ValueError('Each metadata entry must be a JSON object')
    conditions = record.get('conditions', record)
    normalized = {}
    for name, choices in CONDITION_CHOICES.items():
        if name not in conditions:
            raise ValueError(f'Metadata is missing required field {name!r}')
        value = str(conditions[name]).strip().lower().replace(' ', '_')
        value = CONDITION_ALIASES.get(name, {}).get(value, value)
        if value not in choices:
            allowed = ', '.join(choices)
            raise ValueError(
                f'Invalid {name}={value!r}; expected one of: {allowed}'
            )
        normalized[name] = value
    quality_flags = record.get('quality_flags', [])
    if not isinstance(quality_flags, list):
        raise ValueError('quality_flags must be a JSON list when provided')
    return {
        'conditions': normalized,
        'quality_flags': sorted({str(value) for value in quality_flags}),
        'session_id': record.get('session_id'),
    }


def _prompt_choice(session_name, field, choices):
    """Prompt until the operator supplies one controlled metadata value."""
    options = '/'.join(choices)
    while True:
        answer = input(
            f'{session_name} - {field} ({options}): '
        ).strip().lower()
        answer = CONDITION_ALIASES.get(field, {}).get(answer, answer)
        if answer in choices:
            return answer
        print(f'Choose one of: {", ".join(choices)}', file=sys.stderr)


def prompt_metadata(session_name):
    """Interactively collect controlled environmental tags."""
    return {
        'conditions': {
            field: _prompt_choice(session_name, field, choices)
            for field, choices in CONDITION_CHOICES.items()
        },
        'quality_flags': [],
        'session_id': None,
    }


def metadata_for_bag(
    bag_path, input_root, metadata_records, non_interactive
):
    """Resolve metadata by relative path, folder name, or operator prompt."""
    relative = bag_path.relative_to(input_root).as_posix()
    record = metadata_records.get(relative)
    if record is None:
        record = metadata_records.get(bag_path.name)
    if record is not None:
        return normalize_metadata(record)
    if non_interactive or not sys.stdin.isatty():
        raise ValueError(
            f'No metadata entry for {relative!r}; provide --metadata-file '
            'or run interactively'
        )
    return prompt_metadata(relative)


def next_session_id(existing_ids, reserved_ids=()):
    """Allocate the next collision-free bag_NNN session identifier."""
    occupied = set(existing_ids) | set(reserved_ids)
    numbers = [
        int(match.group('number'))
        for session_id in occupied
        if (match := SESSION_PATTERN.fullmatch(session_id))
    ]
    candidate = max(numbers, default=0) + 1
    while f'bag_{candidate:03d}' in occupied:
        candidate += 1
    return f'bag_{candidate:03d}'


def validate_session_output(session_dir, session_id):
    """Validate image decoding, file size, indices, and naming convention."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            f'OpenCV is required for ingestion validation: {error}'
        ) from error
    if not SESSION_PATTERN.fullmatch(session_id):
        raise ValueError(
            f'Session ID {session_id!r} must match bag_NNN'
        )
    images = sorted(session_dir.glob('*.jpg'))
    if not images:
        raise ValueError(f'No JPEG images extracted into {session_dir}')
    filename_pattern = re.compile(
        rf'^{re.escape(session_id)}_frame_(\d{{6}})_t(\d+)\.jpg$'
    )
    dimensions = set()
    for expected_index, image in enumerate(images):
        match = filename_pattern.fullmatch(image.name)
        if not match:
            raise ValueError(f'Invalid extracted image name: {image.name}')
        if int(match.group(1)) != expected_index:
            raise ValueError(
                f'Non-contiguous frame index in {image.name}; expected '
                f'{expected_index:06d}'
            )
        if image.stat().st_size == 0:
            raise ValueError(f'Extracted image is empty: {image}')
        decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise ValueError(f'Extracted image cannot be decoded: {image}')
        dimensions.add((int(decoded.shape[1]), int(decoded.shape[0])))
    return {
        'image_count': len(images),
        'dimensions': [
            {'width': width, 'height': height}
            for width, height in sorted(dimensions)
        ],
        'naming_valid': True,
        'images_non_empty_and_decodable': True,
    }


def _atomic_json(path, payload):
    """Write one JSON result atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def ingest(arguments):
    """Scan, tag, extract, validate, and register new bag sessions."""
    input_root = arguments.input_dir.expanduser().resolve()
    dataset_root = arguments.dataset_root.expanduser().resolve()
    if arguments.every_nth_frame < 1:
        raise ValueError('--every-nth-frame must be at least 1')
    if arguments.maximum_frames < 0:
        raise ValueError('--maximum-frames cannot be negative')

    bag_directories = scan_bag_directories(input_root)
    if not bag_directories:
        raise ValueError(
            f'No .mcap or .db3 rosbag storage found below {input_root}'
        )
    metadata_records = load_metadata_file(arguments.metadata_file)
    registry_path = dataset_root / 'session_metadata.json'
    registry = _load_session_metadata(registry_path)
    existing_hashes = {
        record.get('source_sha256'): session_id
        for session_id, record in registry['sessions'].items()
        if record.get('source_sha256')
    }
    reserved_ids = set()
    results = []

    for bag_path in bag_directories:
        result = {'source_bag': bag_path.as_posix()}
        try:
            source_digest = bag_digest(bag_path)
            if source_digest in existing_hashes:
                result.update({
                    'status': 'skipped_already_ingested',
                    'session_id': existing_hashes[source_digest],
                    'source_sha256': source_digest,
                })
                results.append(result)
                continue
            tags = metadata_for_bag(
                bag_path,
                input_root,
                metadata_records,
                arguments.non_interactive,
            )
            session_id = tags['session_id'] or next_session_id(
                registry['sessions'], reserved_ids
            )
            if not SESSION_PATTERN.fullmatch(session_id):
                raise ValueError(
                    f'Configured session_id {session_id!r} must match bag_NNN'
                )
            if (
                session_id in registry['sessions']
                or session_id in reserved_ids
            ):
                raise ValueError(f'Session ID already exists: {session_id}')
            reserved_ids.add(session_id)
            conditions = dict(tags['conditions'])
            conditions['environment'] = arguments.environment
            extract_arguments = argparse.Namespace(
                bag=bag_path,
                dataset_root=dataset_root,
                session_id=session_id,
                topic=arguments.topic,
                sample_hz=3.0,
                every_nth_frame=arguments.every_nth_frame,
                maximum_frames=arguments.maximum_frames,
                jpeg_quality=arguments.jpeg_quality,
                condition=[
                    f'{name}={value}'
                    for name, value in sorted(conditions.items())
                ],
                quality_flag=tags['quality_flags'],
                overwrite=False,
                source_sha256=source_digest,
            )
            image_dir, _metadata_path, count = extract(extract_arguments)
            validation = validate_session_output(image_dir, session_id)
            result.update({
                'status': 'ingested',
                'session_id': session_id,
                'source_sha256': source_digest,
                'conditions': conditions,
                'extracted_frame_count': count,
                'validation': validation,
            })
            existing_hashes[source_digest] = session_id
        except (
            FileExistsError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as error:
            result.update({'status': 'failed', 'error': str(error)})
        results.append(result)

    report = {
        'schema_version': 1,
        'input_directory': input_root.as_posix(),
        'dataset_root': dataset_root.as_posix(),
        'sampling': {
            'mode': 'frame_stride',
            'every_nth_frame': arguments.every_nth_frame,
        },
        'bags_discovered': len(bag_directories),
        'sessions_ingested': sum(
            result['status'] == 'ingested' for result in results
        ),
        'sessions_skipped': sum(
            result['status'] == 'skipped_already_ingested'
            for result in results
        ),
        'sessions_failed': sum(
            result['status'] == 'failed' for result in results
        ),
        'results': results,
    }
    report_path = Path(arguments.report)
    if not report_path.is_absolute():
        report_path = dataset_root / report_path
    _atomic_json(report_path, report)
    return report_path, report


def main(argv=None):
    """Run batch ingestion and return a process-friendly status code."""
    arguments = parse_arguments(argv)
    try:
        report_path, report = ingest(arguments)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    print(
        f'Discovered {report["bags_discovered"]} bags: '
        f'{report["sessions_ingested"]} ingested, '
        f'{report["sessions_skipped"]} skipped, '
        f'{report["sessions_failed"]} failed.'
    )
    print(f'Ingestion report: {report_path}')
    return 1 if report['sessions_failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
