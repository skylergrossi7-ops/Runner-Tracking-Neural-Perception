#!/usr/bin/env python3
"""Acquire public runner media and build a provenance-aware YOLO dataset.

Remote providers are deliberately identifier-driven: the operator chooses and
reviews a specific Roboflow Universe version or Kaggle dataset instead of this
script scraping arbitrary web video. Every remote run requires an explicit
license acknowledgement.

The verifier uses a pretrained COCO-keypoint model. It retains only full-body
human poses whose leg geometry is consistent with running. These are high-
precision *provisional* annotations and still require human review.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.webp'}
VIDEO_EXTENSIONS = {'.avi', '.m4v', '.mkv', '.mov', '.mp4', '.webm'}
HUMAN_CLASS_NAMES = {
    'athlete', 'human', 'jogger', 'jogging', 'pedestrian', 'person',
    'runner', 'running', 'running person', 'running-person',
}
COCO_HEAD = (0, 1, 2, 3, 4)
COCO_SHOULDERS = (5, 6)
COCO_ELBOWS = (7, 8)
COCO_WRISTS = (9, 10)
COCO_HIPS = (11, 12)
COCO_KNEES = (13, 14)
COCO_ANKLES = (15, 16)
REQUIRED_BODY = COCO_SHOULDERS + COCO_HIPS + COCO_KNEES + COCO_ANKLES


@dataclass(frozen=True)
class PoseCandidate:
    """One pose-model result expressed in image pixel coordinates."""

    box: tuple[float, float, float, float]
    confidence: float
    keypoints: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class AcceptedSample:
    """One staged image and its verified runner boxes."""

    session: str
    image: Path
    label: Path
    source: str
    frame_index: int | None
    timestamp_seconds: float | None
    runner_count: int
    minimum_run_score: float


def utc_now():
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def slug(value):
    """Return a conservative identifier suitable for filenames and sessions."""
    value = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(value)).strip('_').lower()
    return value or 'source'


def parse_arguments(argv=None):
    """Parse acquisition, filtering, and output options."""
    parser = argparse.ArgumentParser(
        description=(
            'Download or scan public runner media, verify full-body running '
            'poses, and create an Ultralytics YOLO dataset.'
        )
    )
    parser.add_argument(
        '--provider', required=True, choices=('local', 'roboflow', 'kaggle')
    )
    parser.add_argument('--input', type=Path)
    parser.add_argument(
        '--roboflow-version',
        help='Universe identifier in workspace/project/version form.',
    )
    parser.add_argument(
        '--roboflow-api-key-env', default='ROBOFLOW_API_KEY'
    )
    parser.add_argument(
        '--kaggle-dataset', help='Kaggle identifier in owner/dataset form.'
    )
    parser.add_argument('--cache-dir', type=Path, default=Path('data/public_cache'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-url')
    parser.add_argument('--license-name')
    parser.add_argument('--license-url')
    parser.add_argument(
        '--accept-license',
        action='store_true',
        help='Confirm that the selected source license permits this use.',
    )
    parser.add_argument('--pose-model', default='yolov8n-pose.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--person-confidence', type=float, default=0.45)
    parser.add_argument('--keypoint-confidence', type=float, default=0.35)
    parser.add_argument('--run-score-threshold', type=float, default=0.58)
    parser.add_argument('--minimum-box-height', type=int, default=45)
    parser.add_argument('--frame-interval-seconds', type=float, default=0.5)
    parser.add_argument(
        '--minimum-running-sequence-seconds',
        type=float,
        default=2.0,
        help=(
            'Require accepted video poses to persist for this duration. '
            'This rejects isolated static poses that resemble a running gait.'
        ),
    )
    parser.add_argument(
        '--maximum-running-gap-seconds',
        type=float,
        default=2.0,
        help='Maximum gap joining accepted poses into one running sequence.',
    )
    parser.add_argument('--maximum-frames-per-video', type=int)
    parser.add_argument('--maximum-source-images', type=int)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--include-negatives',
        action='store_true',
        help='Keep inspected frames that contain no accepted runner.',
    )
    parser.add_argument(
        '--allow-single-session-frame-split',
        action='store_true',
        help=(
            'Allow frame-level splitting when only one source session exists. '
            'This is convenient but permits temporal leakage.'
        ),
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args(argv)


def validate_arguments(arguments):
    """Reject ambiguous providers, unsafe thresholds, and unknown licenses."""
    if arguments.provider == 'local' and arguments.input is None:
        raise ValueError('--input is required for --provider local')
    if arguments.provider == 'roboflow' and not arguments.roboflow_version:
        raise ValueError('--roboflow-version is required for Roboflow')
    if arguments.provider == 'kaggle' and not arguments.kaggle_dataset:
        raise ValueError('--kaggle-dataset is required for Kaggle')
    if arguments.provider != 'local':
        missing = [
            name for name, value in (
                ('--license-name', arguments.license_name),
                ('--license-url', arguments.license_url),
            ) if not value
        ]
        if missing:
            raise ValueError(
                'Remote sources require explicit provenance: '
                + ', '.join(missing)
            )
        if not arguments.accept_license:
            raise ValueError(
                'Review the source terms, then pass --accept-license.'
            )
    for name in (
        'person_confidence', 'keypoint_confidence', 'run_score_threshold',
        'val_fraction',
    ):
        value = float(getattr(arguments, name))
        if not 0.0 < value < 1.0:
            raise ValueError(f'--{name.replace("_", "-")} must be in (0, 1)')
    if arguments.frame_interval_seconds <= 0.0:
        raise ValueError('--frame-interval-seconds must be positive')
    if arguments.minimum_running_sequence_seconds < 0.0:
        raise ValueError(
            '--minimum-running-sequence-seconds cannot be negative'
        )
    if arguments.maximum_running_gap_seconds <= 0.0:
        raise ValueError('--maximum-running-gap-seconds must be positive')
    if arguments.minimum_box_height < 1:
        raise ValueError('--minimum-box-height must be positive')


def safe_prepare_output(output, overwrite):
    """Create an output root without silently replacing curated data."""
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f'Output is not empty: {output}; pass --overwrite explicitly'
            )
        if output == Path(output.anchor) or len(output.parts) < 3:
            raise ValueError(f'Refusing to recursively replace {output}')
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _run_checked(command):
    """Execute an external provider CLI without shell interpolation."""
    completed = subprocess.run(
        [str(value) for value in command],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RuntimeError(
            f'Command failed with {completed.returncode}: {command[0]}\n{tail}'
        )
    return completed


def acquire_source(arguments):
    """Return a local directory containing provider media and annotations."""
    if arguments.provider == 'local':
        source = arguments.input.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        return source

    cache = arguments.cache_dir.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if arguments.provider == 'kaggle':
        executable = shutil.which('kaggle')
        if executable is None:
            raise RuntimeError(
                'Kaggle CLI is not installed. Install the `kaggle` package.'
            )
        destination = cache / ('kaggle_' + slug(arguments.kaggle_dataset))
        destination.mkdir(parents=True, exist_ok=True)
        _run_checked([
            executable, 'datasets', 'metadata', arguments.kaggle_dataset,
            '--path', destination,
        ])
        command = [
            executable, 'datasets', 'download', arguments.kaggle_dataset,
            '--path', destination, '--unzip', '--quiet',
        ]
        if arguments.overwrite:
            command.append('--force')
        _run_checked(command)
        return destination

    fields = arguments.roboflow_version.strip('/').split('/')
    if len(fields) != 3 or not fields[2].isdigit():
        raise ValueError(
            '--roboflow-version must be workspace/project/version'
        )
    api_key = os.environ.get(arguments.roboflow_api_key_env)
    if not api_key:
        raise RuntimeError(
            f'Missing API key environment variable '
            f'{arguments.roboflow_api_key_env}'
        )
    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise RuntimeError(
            'Roboflow SDK is not installed. Install the `roboflow` package.'
        ) from error
    workspace_name, project_name, version_number = fields
    destination = cache / ('roboflow_' + slug('_'.join(fields)))
    destination.mkdir(parents=True, exist_ok=True)
    client = Roboflow(api_key=api_key)
    version = (
        client.workspace(workspace_name)
        .project(project_name)
        .version(int(version_number))
    )
    downloaded = version.download(
        'yolov8', location=str(destination), overwrite=arguments.overwrite
    )
    location = Path(getattr(downloaded, 'location', destination)).resolve()
    return location if location.exists() else destination


def load_class_maps(source_root):
    """Load class ID/name mappings from nearby YOLO dataset YAML files."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError('PyYAML is required') from error
    mappings = []
    for path in sorted(source_root.rglob('*.yaml')):
        try:
            payload = yaml.safe_load(path.read_text('utf-8')) or {}
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        names = payload.get('names')
        if isinstance(names, list):
            names = {index: str(value) for index, value in enumerate(names)}
        elif isinstance(names, dict):
            names = {int(key): str(value) for key, value in names.items()}
        else:
            continue
        mappings.append((path.parent.resolve(), names, path.resolve()))
    mappings.sort(key=lambda item: len(item[0].parts), reverse=True)
    return mappings


def class_map_for_image(image, mappings):
    """Select the deepest dataset YAML that owns an image path."""
    resolved = image.resolve()
    for root, names, yaml_path in mappings:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return names, yaml_path
    return {}, None


def matching_label_path(image):
    """Find a YOLO label paired with an image in common export layouts."""
    parts = list(image.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == 'images':
            candidate = Path(*parts[:index], 'labels', *parts[index + 1:])
            candidate = candidate.with_suffix('.txt')
            if candidate.is_file():
                return candidate
    sibling = image.with_suffix('.txt')
    return sibling if sibling.is_file() else None


def parse_yolo_labels(path, width, height, class_names):
    """Parse human-class YOLO boxes from one source annotation file."""
    if path is None:
        return []
    boxes = []
    for line_number, line in enumerate(path.read_text('utf-8').splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f'{path}:{line_number}: expected 5 fields')
        class_id = int(float(fields[0]))
        values = [float(value) for value in fields[1:]]
        name = class_names.get(class_id, '').strip().lower()
        if class_names and name not in HUMAN_CLASS_NAMES:
            continue
        center_x, center_y, box_width, box_height = values
        if not all(math.isfinite(value) for value in values):
            continue
        x1 = (center_x - box_width / 2.0) * width
        y1 = (center_y - box_height / 2.0) * height
        x2 = (center_x + box_width / 2.0) * width
        y2 = (center_y + box_height / 2.0) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def intersection_over_union(first, second):
    """Return axis-aligned bounding-box IoU."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _angle(first, middle, last):
    """Return the smaller joint angle in degrees."""
    left = (first[0] - middle[0], first[1] - middle[1])
    right = (last[0] - middle[0], last[1] - middle[1])
    denominator = math.hypot(*left) * math.hypot(*right)
    if denominator <= 1.0e-6:
        return 180.0
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right)) / denominator))
    return math.degrees(math.acos(cosine))


def running_pose_score(keypoints, minimum_confidence):
    """Score stride, knee drive, and limb asymmetry for a COCO human pose."""
    if len(keypoints) < 17:
        return 0.0
    confident = [
        point for point in keypoints if point[2] >= minimum_confidence
    ]
    if len(confident) < 10:
        return 0.0
    ys = [point[1] for point in confident]
    body_height = max(ys) - min(ys)
    if body_height <= 1.0:
        return 0.0
    left_knee = _angle(keypoints[11], keypoints[13], keypoints[15])
    right_knee = _angle(keypoints[12], keypoints[14], keypoints[16])
    stride = math.dist(keypoints[15][:2], keypoints[16][:2]) / body_height
    knee_drive = abs(keypoints[13][1] - keypoints[14][1]) / body_height
    flexion = max(0.0, 165.0 - min(left_knee, right_knee)) / 75.0
    asymmetry = abs(left_knee - right_knee) / 65.0
    arm_drive = 0.0
    arm_indices = COCO_SHOULDERS + COCO_ELBOWS + COCO_WRISTS
    if all(keypoints[index][2] >= minimum_confidence for index in arm_indices):
        left_elbow = _angle(keypoints[5], keypoints[7], keypoints[9])
        right_elbow = _angle(keypoints[6], keypoints[8], keypoints[10])
        arm_drive = max(0.0, 160.0 - min(left_elbow, right_elbow)) / 80.0
    score = (
        0.34 * min(1.0, stride / 0.38)
        + 0.31 * min(1.0, flexion)
        + 0.20 * min(1.0, asymmetry)
        + 0.10 * min(1.0, knee_drive / 0.25)
        + 0.05 * min(1.0, arm_drive)
    )
    return max(0.0, min(1.0, score))


def full_body_box_is_valid(candidate, image_shape, minimum_keypoint_confidence,
                           minimum_box_height):
    """Require head, torso, knees, and ankles inside a non-clipped box."""
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = candidate.box
    if x2 <= x1 or y2 <= y1 or y2 - y1 < minimum_box_height:
        return False
    border = 2.0
    if x1 <= border or y1 <= border or x2 >= image_width - border or y2 >= image_height - border:
        return False
    points = candidate.keypoints
    if len(points) < 17:
        return False
    if not any(points[index][2] >= minimum_keypoint_confidence for index in COCO_HEAD):
        return False
    if not all(points[index][2] >= minimum_keypoint_confidence for index in REQUIRED_BODY):
        return False
    required = [
        points[index] for index in (*COCO_HEAD, *REQUIRED_BODY)
        if points[index][2] >= minimum_keypoint_confidence
    ]
    return all(x1 <= x <= x2 and y1 <= y <= y2 for x, y, _ in required)


def infer_pose_candidates(model, frame, arguments):
    """Run Ultralytics pose inference and return framework-neutral results."""
    prediction = model.predict(
        source=frame,
        conf=arguments.person_confidence,
        imgsz=arguments.imgsz,
        device=arguments.device,
        verbose=False,
    )[0]
    if prediction.boxes is None or prediction.keypoints is None:
        return []
    boxes = prediction.boxes.xyxy.detach().cpu().tolist()
    confidences = prediction.boxes.conf.detach().cpu().tolist()
    coordinates = prediction.keypoints.xy.detach().cpu().tolist()
    if prediction.keypoints.conf is None:
        keypoint_confidences = [
            [1.0] * len(points) for points in coordinates
        ]
    else:
        keypoint_confidences = (
            prediction.keypoints.conf.detach().cpu().tolist()
        )
    candidates = []
    for box, confidence, points, point_confidences in zip(
        boxes, confidences, coordinates, keypoint_confidences
    ):
        keypoints = tuple(
            (float(point[0]), float(point[1]), float(point_confidence))
            for point, point_confidence in zip(points, point_confidences)
        )
        candidates.append(PoseCandidate(
            box=tuple(float(value) for value in box),
            confidence=float(confidence),
            keypoints=keypoints,
        ))
    return candidates


def verify_runners(frame, pose_candidates, source_boxes, arguments):
    """Return verified runner boxes with scores and rejection statistics."""
    accepted = []
    rejections = Counter()
    for candidate in pose_candidates:
        if not full_body_box_is_valid(
            candidate,
            frame.shape,
            arguments.keypoint_confidence,
            arguments.minimum_box_height,
        ):
            rejections['not_full_body_or_clipped'] += 1
            continue
        score = running_pose_score(
            candidate.keypoints, arguments.keypoint_confidence
        )
        if score < arguments.run_score_threshold:
            rejections['pose_not_running'] += 1
            continue
        output_box = candidate.box
        if source_boxes:
            matches = sorted(
                (
                    (intersection_over_union(candidate.box, source_box), source_box)
                    for source_box in source_boxes
                ),
                reverse=True,
            )
            if not matches or matches[0][0] < 0.30:
                rejections['source_annotation_mismatch'] += 1
                continue
            output_box = matches[0][1]
            source_candidate = PoseCandidate(
                box=output_box,
                confidence=candidate.confidence,
                keypoints=candidate.keypoints,
            )
            if not full_body_box_is_valid(
                source_candidate,
                frame.shape,
                arguments.keypoint_confidence,
                arguments.minimum_box_height,
            ):
                rejections['source_box_not_full_body'] += 1
                continue
        accepted.append((output_box, score, candidate.confidence))
    return accepted, rejections


def yolo_row(box, image_width, image_height):
    """Convert an xyxy pixel box to one normalized runner-class YOLO row."""
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))
    center_x = (x1 + x2) / (2.0 * image_width)
    center_y = (y1 + y2) / (2.0 * image_height)
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return f'0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}'


def infer_image_session(image, source_root, provider):
    """Group exported image sequences by their containing source directory."""
    relative = image.relative_to(source_root)
    parents = [part for part in relative.parent.parts if part.lower() != 'images']
    group = '_'.join(parents[-2:]) if parents else 'images'
    return slug(f'{provider}_{group}')


def stable_session_splits(session_counts, val_fraction, seed):
    """Assign complete sessions to train/val while approximating image ratio."""
    sessions = sorted(session_counts)
    if len(sessions) < 2:
        raise ValueError('At least two independent source sessions are required')
    ordered = sorted(
        sessions,
        key=lambda value: hashlib.sha256(
            f'{seed}:{value}'.encode('utf-8')
        ).hexdigest(),
    )
    total_images = sum(session_counts.values())
    target = total_images * val_fraction
    possibilities = {0: ()}
    for session in ordered:
        additions = {}
        for count, subset in possibilities.items():
            next_count = count + session_counts[session]
            additions.setdefault(next_count, subset + (session,))
        for count, subset in additions.items():
            possibilities.setdefault(count, subset)
    candidates = [
        (count, subset)
        for count, subset in possibilities.items()
        if 0 < count < total_images
    ]
    validation_count, validation_sessions = min(
        candidates,
        key=lambda item: (
            abs(item[0] - target),
            abs(len(item[1]) - len(sessions) * val_fraction),
            item[1],
        ),
    )
    del validation_count
    validation = set(validation_sessions)
    return {
        session: ('val' if session in validation else 'train')
        for session in sessions
    }


def frame_splits(samples, val_fraction, seed):
    """Leakage-prone fallback used only after explicit operator approval."""
    result = {}
    for sample in samples:
        digest = hashlib.sha256(
            f'{seed}:{sample.image.name}'.encode('utf-8')
        ).digest()
        value = int.from_bytes(digest[:8], 'big') / 2**64
        result[sample.image] = 'val' if value < val_fraction else 'train'
    if samples and all(value == 'val' for value in result.values()):
        result[samples[0].image] = 'train'
    if len(samples) > 1 and all(value == 'train' for value in result.values()):
        result[samples[-1].image] = 'val'
    return result


def _atomic_json(path, payload):
    """Write a JSON artifact atomically."""
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', 'utf-8')
    temporary.replace(path)


def write_yaml(output):
    """Write an Ultralytics-compatible runner dataset configuration."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError('PyYAML is required') from error
    payload = {
        'path': output.as_posix(),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': {0: 'runner'},
    }
    path = output / 'runner_v1.yaml'
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )
    return path


def process_media(arguments, source_root, output, model):
    """Filter source images/video frames and stage accepted YOLO pairs."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('OpenCV is required') from error
    mappings = load_class_maps(source_root if source_root.is_dir() else source_root.parent)
    staging = output / '_staging'
    staged_images = staging / 'images'
    staged_labels = staging / 'labels'
    staged_images.mkdir(parents=True, exist_ok=True)
    staged_labels.mkdir(parents=True, exist_ok=True)
    files = [source_root] if source_root.is_file() else sorted(
        path for path in source_root.rglob('*') if path.is_file()
    )
    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    videos = [path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
    if arguments.maximum_source_images is not None:
        images = images[:arguments.maximum_source_images]
    samples = []
    rejection_counts = Counter()
    inspected = Counter()
    seen_digests = set()

    def inspect(frame, session, source_name, item_index, timestamp, source_boxes):
        inspected['frames'] += 1
        pose_candidates = infer_pose_candidates(model, frame, arguments)
        verified, rejected = verify_runners(
            frame, pose_candidates, source_boxes, arguments
        )
        rejection_counts.update(rejected)
        if not verified and not arguments.include_negatives:
            rejection_counts['frame_without_verified_runner'] += 1
            return
        encoded_ok, encoded = cv2.imencode(
            '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if not encoded_ok:
            rejection_counts['jpeg_encode_failed'] += 1
            return
        content = encoded.tobytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen_digests:
            rejection_counts['duplicate_frame'] += 1
            return
        seen_digests.add(digest)
        filename = f'{session}_{item_index:07d}.jpg'
        image_path = staged_images / filename
        label_path = staged_labels / Path(filename).with_suffix('.txt')
        image_path.write_bytes(content)
        height, width = frame.shape[:2]
        rows = [yolo_row(box, width, height) for box, _, _ in verified]
        label_path.write_text(('\n'.join(rows) + '\n') if rows else '', 'utf-8')
        samples.append(AcceptedSample(
            session=session,
            image=image_path,
            label=label_path,
            source=source_name,
            frame_index=item_index,
            timestamp_seconds=timestamp,
            runner_count=len(rows),
            minimum_run_score=min((item[1] for item in verified), default=0.0),
        ))

    source_base = source_root if source_root.is_dir() else source_root.parent
    for image_index, image in enumerate(images):
        frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if frame is None:
            rejection_counts['image_decode_failed'] += 1
            continue
        names, _yaml_path = class_map_for_image(image, mappings)
        label_path = matching_label_path(image)
        source_boxes = parse_yolo_labels(
            label_path, frame.shape[1], frame.shape[0], names
        )
        session = infer_image_session(image, source_base, arguments.provider)
        inspect(frame, session, image.as_posix(), image_index, None, source_boxes)

    for video in videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            rejection_counts['video_open_failed'] += 1
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0.0:
            fps = 30.0
        stride = max(1, round(fps * arguments.frame_interval_seconds))
        session = slug(f'{arguments.provider}_{video.stem}')
        frame_index = 0
        sampled = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                inspect(
                    frame,
                    session,
                    video.as_posix(),
                    frame_index,
                    frame_index / fps,
                    [],
                )
                sampled += 1
                if (
                    arguments.maximum_frames_per_video is not None
                    and sampled >= arguments.maximum_frames_per_video
                ):
                    break
            frame_index += 1
        capture.release()
        inspected['videos'] += 1
    return samples, inspected, rejection_counts


def filter_short_running_sequences(samples, minimum_duration, maximum_gap):
    """Reject isolated video poses that do not form a running sequence.

    A single-frame pose can resemble running while a subject bends, stretches,
    or steps over an object. Video samples therefore need temporal persistence.
    Still-image sources have no timestamp and remain eligible for independent
    human review.
    """
    if minimum_duration <= 0.0:
        return list(samples), 0
    grouped = defaultdict(list)
    keep = set()
    for sample in samples:
        if sample.timestamp_seconds is None:
            keep.add(sample.image)
        else:
            grouped[sample.session].append(sample)
    for session_samples in grouped.values():
        ordered = sorted(
            session_samples,
            key=lambda sample: (sample.timestamp_seconds, sample.frame_index),
        )
        sequence = []

        def retain_if_long_enough(items):
            if not items:
                return
            duration = (
                items[-1].timestamp_seconds
                - items[0].timestamp_seconds
            )
            if duration >= minimum_duration:
                keep.update(item.image for item in items)

        for sample in ordered:
            if (
                sequence
                and sample.timestamp_seconds
                - sequence[-1].timestamp_seconds > maximum_gap
            ):
                retain_if_long_enough(sequence)
                sequence = []
            sequence.append(sample)
        retain_if_long_enough(sequence)
    filtered = [sample for sample in samples if sample.image in keep]
    return filtered, len(samples) - len(filtered)


def finalize_dataset(arguments, output, samples):
    """Assign leakage-safe splits, move staged pairs, and write manifests."""
    if not samples:
        raise RuntimeError('No frames passed the strict runner filter')
    session_counts = Counter(sample.session for sample in samples)
    frame_split = None
    if len(session_counts) == 1 and arguments.allow_single_session_frame_split:
        frame_split = frame_splits(samples, arguments.val_fraction, arguments.seed)
        session_splits = {next(iter(session_counts)): 'mixed_frame_split'}
        leakage_safe = False
    else:
        session_splits = stable_session_splits(
            session_counts, arguments.val_fraction, arguments.seed
        )
        leakage_safe = True
    finalized = []
    for sample in samples:
        split = (
            frame_split[sample.image] if frame_split is not None
            else session_splits[sample.session]
        )
        image_dir = output / 'images' / split
        label_dir = output / 'labels' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        destination_image = image_dir / sample.image.name
        destination_label = label_dir / sample.label.name
        sample.image.replace(destination_image)
        sample.label.replace(destination_label)
        finalized.append({
            **asdict(sample),
            'image': destination_image.as_posix(),
            'label': destination_label.as_posix(),
            'split': split,
        })
    shutil.rmtree(output / '_staging')
    manifest = {
        'schema_version': 1,
        'created_at': utc_now(),
        'leakage_safe': leakage_safe,
        'session_splits': session_splits,
        'split_counts': dict(Counter(item['split'] for item in finalized)),
        'samples': finalized,
    }
    _atomic_json(output / 'splits_v1.json', manifest)
    return manifest


def run(arguments):
    """Execute provider acquisition, neural filtering, and YOLO formatting."""
    validate_arguments(arguments)
    output = safe_prepare_output(arguments.output, arguments.overwrite)
    source_root = acquire_source(arguments)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('Ultralytics is required') from error
    model = YOLO(arguments.pose_model)
    samples, inspected, rejections = process_media(
        arguments, source_root, output, model
    )
    candidates_before_temporal = len(samples)
    candidate_sessions_before_temporal = dict(sorted(Counter(
        sample.session for sample in samples
    ).items()))
    samples, temporal_rejections = filter_short_running_sequences(
        samples,
        arguments.minimum_running_sequence_seconds,
        arguments.maximum_running_gap_seconds,
    )
    if temporal_rejections:
        rejections['short_or_isolated_running_sequence'] += temporal_rejections
    manifest = finalize_dataset(arguments, output, samples)
    yaml_path = write_yaml(output)
    report = {
        'schema_version': 1,
        'created_at': utc_now(),
        'status': 'passed',
        'provider': arguments.provider,
        'provider_identifier': (
            arguments.roboflow_version
            if arguments.provider == 'roboflow'
            else arguments.kaggle_dataset
            if arguments.provider == 'kaggle'
            else str(arguments.input)
        ),
        'source_root': source_root.as_posix(),
        'source_url': arguments.source_url,
        'license': {
            'name': arguments.license_name,
            'url': arguments.license_url,
            'acknowledged': bool(arguments.accept_license),
        },
        'verification': {
            'model': arguments.pose_model,
            'method': (
                'COCO full-body keypoints, running-gait score, and temporal '
                'sequence persistence'
            ),
            'device': arguments.device,
            'inference_image_size': arguments.imgsz,
            'frame_interval_seconds': arguments.frame_interval_seconds,
            'person_confidence': arguments.person_confidence,
            'keypoint_confidence': arguments.keypoint_confidence,
            'run_score_threshold': arguments.run_score_threshold,
            'minimum_box_height': arguments.minimum_box_height,
            'minimum_running_sequence_seconds': (
                arguments.minimum_running_sequence_seconds
            ),
            'maximum_running_gap_seconds': arguments.maximum_running_gap_seconds,
            'annotation_status': 'provisional_requires_human_review',
        },
        'inspected': dict(inspected),
        'candidate_images_before_temporal_filter': candidates_before_temporal,
        'candidate_sessions_before_temporal_filter': (
            candidate_sessions_before_temporal
        ),
        'accepted_images': len(samples),
        'accepted_images_by_session': dict(sorted(Counter(
            sample.session for sample in samples
        ).items())),
        'accepted_runner_boxes': sum(sample.runner_count for sample in samples),
        'rejections': dict(rejections),
        'splits': manifest['split_counts'],
        'leakage_safe': manifest['leakage_safe'],
        'dataset_yaml': yaml_path.as_posix(),
    }
    _atomic_json(output / 'public_source_report.json', report)
    return report


def main(argv=None):
    """CLI entry point with concise operator-facing failures."""
    try:
        report = run(parse_arguments(argv))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    print(
        f"PASS: {report['accepted_images']} images and "
        f"{report['accepted_runner_boxes']} runner boxes"
    )
    print(f"Dataset YAML: {report['dataset_yaml']}")
    print('Annotations are provisional and require human review.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
