#!/usr/bin/env python3
"""Extract a timestamped, session-scoped image set from one ROS 2 bag."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


SESSION_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def parse_condition(entry):
    """Parse and normalize one ``NAME=VALUE`` session condition."""
    if '=' not in entry:
        raise ValueError(f'Invalid condition {entry!r}; expected NAME=VALUE')
    name, value = (part.strip().lower() for part in entry.split('=', 1))
    if not name or not value:
        raise ValueError(f'Invalid condition {entry!r}; expected NAME=VALUE')
    return name, value.replace(' ', '_')


def detect_storage_id(bag_path):
    """Infer the rosbag2 storage plugin from files inside a bag directory."""
    if any(bag_path.glob('*.mcap')):
        return 'mcap'
    if any(bag_path.glob('*.db3')):
        return 'sqlite3'
    raise ValueError(f'No .mcap or .db3 storage file found in {bag_path}')


def bag_digest(bag_path):
    """Hash rosbag metadata and storage files for dataset provenance."""
    digest = hashlib.sha256()
    files = sorted(
        path for path in bag_path.iterdir()
        if path.is_file() and path.suffix.lower() in ('.mcap', '.db3', '.yaml')
    )
    for path in files:
        digest.update(path.name.encode('utf-8'))
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def parse_arguments(argv=None):
    """Parse rosbag extraction settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True, type=Path)
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--topic', default='/camera/image_raw')
    parser.add_argument('--sample-hz', type=float, default=3.0)
    parser.add_argument(
        '--every-nth-frame',
        type=int,
        default=0,
        help=(
            'Save every Nth image message instead of time-based sampling; '
            'zero disables frame-stride sampling.'
        ),
    )
    parser.add_argument('--maximum-frames', type=int, default=0)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument(
        '--condition', action='append', default=[], metavar='NAME=VALUE'
    )
    parser.add_argument(
        '--quality-flag',
        action='append',
        default=[],
        help=(
            'Repeatable collection-quality warning stored in session metadata.'
        ),
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(argv)


def _atomic_json(path, payload):
    """Atomically write JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def _load_session_metadata(path):
    """Load or initialize the project session registry."""
    if not path.exists():
        return {'schema_version': 1, 'sessions': {}}
    payload = json.loads(path.read_text('utf-8'))
    if not isinstance(payload.get('sessions'), dict):
        raise ValueError(f'Invalid session registry: {path}')
    return payload


def _message_to_bgr(message, message_type, bridge, cv2, numpy):
    """Convert raw or compressed ROS image messages into BGR arrays."""
    if message_type == 'sensor_msgs/msg/CompressedImage':
        encoded = numpy.frombuffer(message.data, dtype=numpy.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('OpenCV could not decode a compressed frame')
        return image
    if message_type == 'sensor_msgs/msg/Image':
        return bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
    raise ValueError(
        f'{message_type} is not a supported ROS image message type'
    )


def extract(arguments):
    """Extract sampled frames and update the session registry."""
    try:
        import cv2
        from cv_bridge import CvBridge
        import numpy
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise RuntimeError(
            'Run this script from a sourced ROS 2 environment with OpenCV, '
            f'cv_bridge, and rosbag2_py available: {error}'
        ) from error

    bag_path = arguments.bag.expanduser().resolve()
    dataset_root = arguments.dataset_root.expanduser().resolve()
    if not bag_path.is_dir():
        raise FileNotFoundError(f'Bag directory not found: {bag_path}')
    if not SESSION_PATTERN.fullmatch(arguments.session_id):
        raise ValueError(
            '--session-id may contain only letters, digits, dot, dash, and '
            'underscore and may not begin with punctuation'
        )
    every_nth_frame = getattr(arguments, 'every_nth_frame', 0)
    if every_nth_frame < 0:
        raise ValueError('--every-nth-frame cannot be negative')
    if every_nth_frame == 0 and arguments.sample_hz <= 0.0:
        raise ValueError('--sample-hz must be positive')
    if not 1 <= arguments.jpeg_quality <= 100:
        raise ValueError('--jpeg-quality must be between 1 and 100')

    conditions = dict(parse_condition(value) for value in arguments.condition)
    session_image_dir = dataset_root / 'images' / arguments.session_id
    session_label_dir = dataset_root / 'labels' / arguments.session_id
    metadata_path = dataset_root / 'session_metadata.json'
    metadata = _load_session_metadata(metadata_path)
    if (
        arguments.session_id in metadata['sessions']
        and not arguments.overwrite
    ):
        raise FileExistsError(
            f'Session metadata already exists for {arguments.session_id}'
        )
    if session_image_dir.exists() and not arguments.overwrite:
        raise FileExistsError(
            f'Session output already exists: {session_image_dir}'
        )
    if arguments.overwrite:
        for old_image in session_image_dir.glob('*.jpg'):
            old_image.unlink()
        for old_label in session_label_dir.glob('*.txt'):
            old_label.unlink()
    session_image_dir.mkdir(parents=True, exist_ok=True)
    session_label_dir.mkdir(parents=True, exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_path), storage_id=detect_storage_id(bag_path)
        ),
        rosbag2_py.ConverterOptions('', ''),
    )
    topic_types = {
        metadata.name: metadata.type
        for metadata in reader.get_all_topics_and_types()
    }
    if arguments.topic not in topic_types:
        available = ', '.join(sorted(topic_types))
        raise ValueError(
            f'Topic {arguments.topic!r} not present. Available: {available}'
        )
    message_type_name = topic_types[arguments.topic]
    message_class = get_message(message_type_name)
    bridge = CvBridge()
    minimum_period_ns = None
    if every_nth_frame == 0:
        minimum_period_ns = round(1_000_000_000 / arguments.sample_hz)
    last_saved_timestamp = None
    first_timestamp = None
    final_timestamp = None
    saved = 0
    source_messages = 0

    while reader.has_next():
        topic, serialized, timestamp = reader.read_next()
        if topic != arguments.topic:
            continue
        source_messages += 1
        if every_nth_frame:
            if (source_messages - 1) % every_nth_frame:
                continue
        elif (
            last_saved_timestamp is not None
            and timestamp - last_saved_timestamp < minimum_period_ns
        ):
            continue
        message = deserialize_message(serialized, message_class)
        image = _message_to_bgr(
            message, message_type_name, bridge, cv2, numpy
        )
        filename = (
            f'{arguments.session_id}_frame_{saved:06d}_t{timestamp}.jpg'
        )
        output = session_image_dir / filename
        written = cv2.imwrite(
            str(output),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, arguments.jpeg_quality],
        )
        if not written:
            raise RuntimeError(f'Failed to write extracted image: {output}')
        saved += 1
        last_saved_timestamp = timestamp
        if first_timestamp is None:
            first_timestamp = timestamp
        final_timestamp = timestamp
        if arguments.maximum_frames and saved >= arguments.maximum_frames:
            break

    if saved == 0:
        raise RuntimeError(f'No frames were extracted from {arguments.topic}')

    metadata['sessions'][arguments.session_id] = {
        'source_bag': bag_path.as_posix(),
        'source_sha256': (
            getattr(arguments, 'source_sha256', None)
            or bag_digest(bag_path)
        ),
        'topic': arguments.topic,
        'message_type': message_type_name,
        'conditions': conditions,
        'quality_flags': sorted(set(arguments.quality_flag)),
        'source_message_count': source_messages,
        'extracted_frame_count': saved,
        'sampling': (
            {
                'mode': 'frame_stride',
                'every_nth_frame': every_nth_frame,
            }
            if every_nth_frame
            else {
                'mode': 'frequency',
                'sample_hz': arguments.sample_hz,
            }
        ),
        'sample_hz': None if every_nth_frame else arguments.sample_hz,
        'every_nth_frame': every_nth_frame or None,
        'first_timestamp_ns': first_timestamp,
        'last_timestamp_ns': final_timestamp,
        'annotation_status': 'unlabeled',
    }
    _atomic_json(metadata_path, metadata)
    return session_image_dir, metadata_path, saved


def main(argv=None):
    """Extract one recording session into the raw runner dataset."""
    arguments = parse_arguments(argv)
    try:
        image_dir, metadata_path, count = extract(arguments)
    except (
        FileExistsError, FileNotFoundError, RuntimeError, ValueError
    ) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    print(f'Extracted {count} frames to {image_dir}')
    print(f'Session metadata: {metadata_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
