#!/usr/bin/env python3
"""Create leakage-safe YOLO train/validation manifests by recording session.

Example:
    python3 ml/scripts/prepare_dataset.py \
      --dataset-root data/runner_raw \
      --output-dir data/runner_v1 \
      --session-regex '^(?P<session>bag_[0-9]+)_'

The expected source layout is ``images/**`` mirrored by ``labels/**``. Images
may be grouped into session directories, or a session can be extracted from a
flat filename with ``--session-regex``. The script writes image-list files; it
does not copy or mutate source data.
"""

import argparse
import itertools
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_EXTENSIONS = {
    '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'
}

DEFAULT_CONDITION_REGEXES = {
    'distance': (
        r'(?:^|[/_.-])(near|mid|middle|far)(?:$|[/_.-])'
    ),
    'lighting': (
        r'(?:^|[/_.-])'
        r'(day|bright|overcast|lowlight|low_light|night|backlit)'
        r'(?:$|[/_.-])'
    ),
    'occlusion': (
        r'(?:^|[/_.-])'
        r'(clear|partial|partially_occluded|occluded)'
        r'(?:$|[/_.-])'
    ),
}


@dataclass
class Sample:
    """One source image and its mirrored YOLO label."""

    image: Path
    label: Path


@dataclass
class Session:
    """All samples and condition tags belonging to one recording session."""

    session_id: str
    samples: list[Sample] = field(default_factory=list)
    conditions: set[str] = field(default_factory=set)


def parse_arguments(argv=None):
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            'Split a YOLO dataset by complete recording session and generate '
            'Ultralytics manifests without copying images.'
        )
    )
    parser.add_argument('--dataset-root', required=True, type=Path)
    parser.add_argument('--images-dir', default='images')
    parser.add_argument('--labels-dir', default='labels')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--manifest', default='splits_v1.json')
    parser.add_argument('--yaml', default='runner_v1.yaml')
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--session-regex',
        help=(
            'Regex applied to each path relative to images/. Use a named '
            '"session" group or make the first capture group the session ID.'
        ),
    )
    parser.add_argument(
        '--session-metadata',
        type=Path,
        help=(
            'Optional JSON mapping session IDs to conditions. Accepted form: '
            '{"sessions": {"bag_01": {"conditions": {...}}}}.'
        ),
    )
    parser.add_argument(
        '--condition-regex',
        action='append',
        default=[],
        metavar='NAME=REGEX',
        help=(
            'Condition parser applied to relative paths; repeat for distance, '
            'lighting, occlusion, etc. The first capture is the value.'
        ),
    )
    parser.add_argument(
        '--class-names', nargs='+', default=['runner'],
        help='YOLO class names in numeric ID order.',
    )
    parser.add_argument(
        '--path-mode', choices=('absolute', 'relative'), default='absolute',
        help='How paths are written to image lists and generated YAML.',
    )
    parser.add_argument('--allow-missing-labels', action='store_true')
    parser.add_argument('--allow-unbalanced-conditions', action='store_true')
    return parser.parse_args(argv)


def _compile_named_regexes(entries):
    """Return condition name/compiled-regex pairs from CLI entries."""
    source = DEFAULT_CONDITION_REGEXES if not entries else {}
    if entries:
        for entry in entries:
            if '=' not in entry:
                raise ValueError(
                    f'Invalid --condition-regex {entry!r}; expected NAME=REGEX'
                )
            name, expression = entry.split('=', 1)
            name = name.strip()
            if not name or not expression:
                raise ValueError(
                    f'Invalid --condition-regex {entry!r}; expected NAME=REGEX'
                )
            source[name] = expression
    return {
        name: re.compile(expression, re.IGNORECASE)
        for name, expression in source.items()
    }


def _normalize_condition(value):
    """Normalize common aliases so equivalent conditions share one stratum."""
    value = value.lower().replace('-', '_')
    aliases = {
        'middle': 'mid',
        'low_light': 'lowlight',
        'partially_occluded': 'partial',
    }
    return aliases.get(value, value)


def extract_conditions(relative_path, condition_regexes):
    """Extract ``name=value`` condition tags from an image path."""
    text = relative_path.as_posix().lower()
    tags = set()
    for name, expression in condition_regexes.items():
        match = expression.search(text)
        if match:
            value = match.group(1) if match.lastindex else match.group(0)
            tags.add(f'{name}={_normalize_condition(value)}')
    return tags


def load_metadata(path):
    """Load optional session condition metadata."""
    if path is None:
        return {}
    payload = json.loads(path.expanduser().resolve().read_text('utf-8'))
    sessions = payload.get('sessions', payload)
    result = {}
    for session_id, record in sessions.items():
        conditions = record.get('conditions', record)
        tags = set()
        if isinstance(conditions, dict):
            tags.update(
                f'{name}={_normalize_condition(str(value))}'
                for name, value in conditions.items()
            )
        elif isinstance(conditions, list):
            tags.update(str(value) for value in conditions)
        else:
            raise ValueError(
                f'Conditions for session {session_id!r} must be a map or list'
            )
        result[str(session_id)] = tags
    return result


def parse_session_id(relative_path, session_regex=None):
    """Extract a stable source-session ID from a relative image path."""
    relative_text = relative_path.as_posix()
    if session_regex is not None:
        match = session_regex.search(relative_text)
        if not match:
            raise ValueError(
                f'Path {relative_text!r} does not match --session-regex'
            )
        if 'session' in match.groupdict():
            return match.group('session')
        if match.lastindex:
            return match.group(1)
        return match.group(0)

    if len(relative_path.parts) > 1:
        return relative_path.parts[0]

    stem = relative_path.stem
    patterns = (
        r'(?P<session>.+?)[_-](?:frame|image|img)[_-]?[0-9]+$',
        r'(?P<session>.+?)[_-][0-9]{4,}$',
    )
    for pattern in patterns:
        match = re.match(pattern, stem, re.IGNORECASE)
        if match:
            return match.group('session')
    raise ValueError(
        'Cannot infer a session ID from flat filename '
        f'{relative_path.name!r}. '
        'Place images in session subdirectories or provide --session-regex.'
    )


def discover_sessions(arguments):
    """Discover paired YOLO samples and group them by recording session."""
    dataset_root = arguments.dataset_root.expanduser().resolve()
    image_root = (dataset_root / arguments.images_dir).resolve()
    label_root = (dataset_root / arguments.labels_dir).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f'Image directory not found: {image_root}')
    if not label_root.is_dir():
        raise FileNotFoundError(f'Label directory not found: {label_root}')

    session_regex = None
    if arguments.session_regex:
        session_regex = re.compile(arguments.session_regex)
    condition_regexes = _compile_named_regexes(arguments.condition_regex)
    metadata = load_metadata(arguments.session_metadata)
    sessions = {}
    missing_labels = []
    images = sorted(
        path for path in image_root.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f'No supported images found below {image_root}')

    for image in images:
        relative = image.relative_to(image_root)
        label = (label_root / relative).with_suffix('.txt')
        if not label.is_file() and not arguments.allow_missing_labels:
            missing_labels.append(label)
            continue
        session_id = parse_session_id(relative, session_regex)
        session = sessions.setdefault(session_id, Session(session_id))
        session.samples.append(Sample(image.resolve(), label.resolve()))
        session.conditions.update(
            extract_conditions(relative, condition_regexes)
        )

    if missing_labels:
        preview = '\n'.join(f'  - {path}' for path in missing_labels[:10])
        suffix = '' if len(missing_labels) <= 10 else '\n  - ...'
        raise FileNotFoundError(
            f'{len(missing_labels)} image(s) have no mirrored YOLO label:\n'
            f'{preview}{suffix}\nUse --allow-missing-labels only for '
            'intentional background/negative images.'
        )
    for session_id, tags in metadata.items():
        if session_id not in sessions:
            raise ValueError(
                f'Metadata references unknown session {session_id!r}'
            )
        sessions[session_id].conditions.update(tags)
    if len(sessions) < 2:
        raise ValueError('At least two independent sessions are required')
    return sorted(sessions.values(), key=lambda session: session.session_id)


def _candidate_score(val_indices, sessions, val_fraction):
    """Score a validation-session candidate; lower is better."""
    val_indices = set(val_indices)
    session_ratio = len(val_indices) / len(sessions)
    total_images = sum(len(session.samples) for session in sessions)
    val_images = sum(len(sessions[index].samples) for index in val_indices)
    image_ratio = val_images / total_images

    condition_counts = Counter(
        tag for session in sessions for tag in session.conditions
    )
    val_condition_counts = Counter(
        tag
        for index in val_indices
        for tag in sessions[index].conditions
    )
    missing = []
    balance_error = 0.0
    for tag, total in condition_counts.items():
        if total < 2:
            continue
        val_count = val_condition_counts[tag]
        if val_count == 0 or val_count == total:
            missing.append(tag)
        balance_error += abs((val_count / total) - val_fraction)

    score = (
        1000.0 * len(missing)
        + 5.0 * abs(image_ratio - val_fraction)
        + 3.0 * abs(session_ratio - val_fraction)
        + balance_error
    )
    return score, missing, image_ratio, session_ratio


def choose_validation_sessions(sessions, val_fraction, seed):
    """Choose a deterministic, condition-aware validation session set."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('--val-fraction must be strictly between 0 and 1')
    count = len(sessions)
    candidates = []
    if count <= 16:
        for size in range(1, count):
            candidates.extend(itertools.combinations(range(count), size))
    else:
        randomizer = random.Random(seed)
        target = max(1, min(count - 1, round(count * val_fraction)))
        radius = max(2, round(count * 0.10))
        sizes = range(max(1, target - radius), min(count, target + radius + 1))
        seen = set()
        for size in sizes:
            baseline = tuple(range(size))
            seen.add(baseline)
            candidates.append(baseline)
        for _ in range(50000):
            size = randomizer.choice(tuple(sizes))
            candidate = tuple(sorted(randomizer.sample(range(count), size)))
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

    def ordering(candidate):
        score = _candidate_score(candidate, sessions, val_fraction)[0]
        ids = tuple(sessions[index].session_id for index in candidate)
        return score, ids

    selected = min(candidates, key=ordering)
    score, missing, image_ratio, session_ratio = _candidate_score(
        selected, sessions, val_fraction
    )
    return set(selected), {
        'score': score,
        'unbalanced_representable_conditions': missing,
        'actual_val_image_fraction': image_ratio,
        'actual_val_session_fraction': session_ratio,
    }


def _display_path(path, output_dir, mode):
    """Render an absolute or output-relative portable path."""
    if mode == 'absolute':
        return path.resolve().as_posix()
    relative = os.path.relpath(path.resolve(), output_dir.resolve())
    return Path(relative).as_posix()


def _atomic_write(path, content):
    """Atomically replace a generated text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(content, encoding='utf-8', newline='\n')
    temporary.replace(path)


def write_outputs(arguments, sessions, val_indices, balance):
    """Write lists, split manifest, and an Ultralytics dataset YAML."""
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(arguments.manifest)
    if not manifest_path.is_absolute():
        manifest_path = output_dir / manifest_path
    yaml_path = Path(arguments.yaml)
    if not yaml_path.is_absolute():
        yaml_path = output_dir / yaml_path

    split_sessions = {
        'train': [
            session for index, session in enumerate(sessions)
            if index not in val_indices
        ],
        'val': [
            session for index, session in enumerate(sessions)
            if index in val_indices
        ],
    }
    split_records = {}
    for split, selected_sessions in split_sessions.items():
        samples = sorted(
            (
                sample
                for session in selected_sessions
                for sample in session.samples
            ),
            key=lambda sample: sample.image.as_posix(),
        )
        image_paths = [
            _display_path(sample.image, output_dir, arguments.path_mode)
            for sample in samples
        ]
        label_paths = [
            _display_path(sample.label, output_dir, arguments.path_mode)
            for sample in samples
        ]
        _atomic_write(
            output_dir / f'{split}.txt',
            ''.join(f'{path}\n' for path in image_paths),
        )
        split_records[split] = {
            'sessions': [session.session_id for session in selected_sessions],
            'images': image_paths,
            'labels': label_paths,
        }

    condition_report = {}
    all_conditions = sorted(
        {tag for session in sessions for tag in session.conditions}
    )
    for tag in all_conditions:
        condition_report[tag] = {
            split: sorted(
                session.session_id
                for session in selected_sessions
                if tag in session.conditions
            )
            for split, selected_sessions in split_sessions.items()
        }

    manifest = {
        'schema_version': 1,
        'seed': arguments.seed,
        'target_val_fraction': arguments.val_fraction,
        'source': {
            'dataset_root': (
                arguments.dataset_root.expanduser().resolve().as_posix()
            ),
            'images': (
                arguments.dataset_root.expanduser().resolve()
                / arguments.images_dir
            ).as_posix(),
            'labels': (
                arguments.dataset_root.expanduser().resolve()
                / arguments.labels_dir
            ).as_posix(),
        },
        'strategy': {
            'grouping': 'exclusive_recording_session',
            'balancing': 'session_and_image_ratio_with_condition_coverage',
            'path_mode': arguments.path_mode,
        },
        'balance': {
            **balance,
            'condition_sessions': condition_report,
        },
        'sessions': {
            session.session_id: {
                'sample_count': len(session.samples),
                'conditions': sorted(session.conditions),
            }
            for session in sessions
        },
        'splits': split_records,
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2) + '\n')

    if arguments.path_mode == 'absolute':
        yaml_root = output_dir.as_posix()
    else:
        yaml_root = Path(
            os.path.relpath(output_dir, yaml_path.parent.resolve())
        ).as_posix()
    yaml_lines = [
        '# Generated by ml/scripts/prepare_dataset.py; do not edit by hand.',
        f'path: {json.dumps(yaml_root)}',
        'train: "train.txt"',
        'val: "val.txt"',
        'names:',
    ]
    yaml_lines.extend(
        f'  {index}: {json.dumps(name)}'
        for index, name in enumerate(arguments.class_names)
    )
    _atomic_write(yaml_path, '\n'.join(yaml_lines) + '\n')
    return manifest_path.resolve(), yaml_path.resolve(), manifest


def main(argv=None):
    """Prepare a leakage-safe YOLO dataset split."""
    try:
        arguments = parse_arguments(argv)
        sessions = discover_sessions(arguments)
        val_indices, balance = choose_validation_sessions(
            sessions, arguments.val_fraction, arguments.seed
        )
        missing = balance['unbalanced_representable_conditions']
        if missing and not arguments.allow_unbalanced_conditions:
            raise RuntimeError(
                'No candidate split represented these conditions in both '
                f'train and val: {", ".join(missing)}. Add sessions, provide '
                'better metadata, or use --allow-unbalanced-conditions.'
            )
        manifest_path, yaml_path, manifest = write_outputs(
            arguments, sessions, val_indices, balance
        )
    except (FileNotFoundError, ValueError, RuntimeError, re.error) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2

    train = manifest['splits']['train']
    val = manifest['splits']['val']
    print(
        f'Prepared {len(train["images"])} train images from '
        f'{len(train["sessions"])} sessions and {len(val["images"])} val '
        f'images from {len(val["sessions"])} sessions.'
    )
    print(
        'Validation fractions: '
        f'{balance["actual_val_session_fraction"]:.1%} sessions, '
        f'{balance["actual_val_image_fraction"]:.1%} images.'
    )
    if not any(session.conditions for session in sessions):
        print(
            'WARNING: No condition tags were discovered; provide session '
            'metadata or --condition-regex rules for stratified balancing.',
            file=sys.stderr,
        )
    elif balance['unbalanced_representable_conditions']:
        print(
            'WARNING: Unbalanced conditions: '
            + ', '.join(balance['unbalanced_representable_conditions']),
            file=sys.stderr,
        )
    print(f'Manifest: {manifest_path}')
    print(f'Ultralytics YAML: {yaml_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
