#!/usr/bin/env python3
"""Train and evaluate the runner detector with reproducible safeguards."""

import argparse
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path


METRIC_NAMES = ('precision', 'recall', 'map50', 'map50_95')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def parse_arguments(argv=None):
    """Parse training overrides without duplicating the YAML configuration."""
    default_config = Path(__file__).parent / 'config' / 'train_robust.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=default_config)
    parser.add_argument('--model')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--name')
    parser.add_argument('--device')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch', type=float)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-unreviewed', action='store_true')
    parser.add_argument('--no-custom-augmentations', action='store_true')
    return parser.parse_args(argv)


def _load_yaml(path):
    """Load one YAML mapping with an actionable dependency error."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            'Install PyYAML to load training settings'
        ) from error
    payload = yaml.safe_load(path.read_text('utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')
    return payload


def _resolve_path(value, base_directory):
    """Resolve a configuration-relative filesystem path."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve()


def load_training_config(path):
    """Load, validate, and resolve the robust training configuration."""
    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Training config not found: {config_path}')
    config = _load_yaml(config_path)
    required = {
        'model', 'data', 'project', 'name', 'train',
        'robust_augmentations', 'evaluation',
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(
            f'Training config is missing: {", ".join(missing)}'
        )
    if config.get('schema_version') != 1:
        raise ValueError('Unsupported training configuration schema')
    for section in ('train', 'evaluation', 'robust_augmentations'):
        if not isinstance(config[section], dict):
            raise ValueError(f'{section} must be a YAML mapping')
    config['_config_path'] = config_path
    config['data'] = _resolve_path(config['data'], config_path.parent)
    config['project'] = _resolve_path(config['project'], config_path.parent)
    model_candidate = _resolve_path(config['model'], config_path.parent)
    config['model'] = (
        str(model_candidate)
        if model_candidate.is_file()
        else str(config['model'])
    )
    return config


def apply_overrides(config, arguments):
    """Apply only explicit CLI overrides to a loaded configuration."""
    if arguments.model is not None:
        model_path = Path(arguments.model).expanduser()
        config['model'] = (
            str(model_path.resolve()) if model_path.is_file()
            else arguments.model
        )
    if arguments.data is not None:
        config['data'] = arguments.data.expanduser().resolve()
    if arguments.project is not None:
        config['project'] = arguments.project.expanduser().resolve()
    if arguments.name is not None:
        config['name'] = arguments.name
    for name in ('device', 'epochs', 'workers'):
        value = getattr(arguments, name)
        if value is not None:
            config['train'][name] = value
    if arguments.batch is not None:
        config['train']['batch'] = (
            int(arguments.batch)
            if arguments.batch.is_integer()
            else arguments.batch
        )
    if arguments.no_custom_augmentations:
        config['robust_augmentations']['enabled'] = False
    return config


def _read_image_list(path):
    """Resolve and validate every image listed by one YOLO split file."""
    if not path.is_file():
        raise FileNotFoundError(f'Dataset split list not found: {path}')
    entries = [
        line.strip() for line in path.read_text('utf-8').splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(f'Dataset split is empty: {path}')
    missing = []
    resolved = []
    for entry in entries:
        image = Path(entry).expanduser()
        if not image.is_absolute():
            image = path.parent / image
        image = image.resolve()
        resolved.append(image)
        if not image.is_file() or image.stat().st_size == 0:
            missing.append(image)
    if missing:
        preview = ', '.join(str(image) for image in missing[:5])
        raise FileNotFoundError(
            f'{len(missing)} missing or empty images in {path}: {preview}'
        )
    return resolved


def _read_image_source(path):
    """Read a YOLO image-list file or discover images below a directory."""
    if path.is_file():
        return _read_image_list(path)
    if path.is_dir():
        images = sorted(
            image.resolve()
            for image in path.rglob('*')
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f'Dataset split directory is empty: {path}')
        return images
    raise FileNotFoundError(f'Dataset split not found: {path}')


def _validate_review_status(manifest, allow_unreviewed):
    """Require explicit human-review completion for all split sessions."""
    if allow_unreviewed:
        return
    source_root = manifest.get('source', {}).get('dataset_root')
    if not source_root:
        raise ValueError('splits_v1.json has no source dataset_root')
    metadata_path = Path(source_root) / 'session_metadata.json'
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f'Session metadata not found: {metadata_path}'
        )
    metadata = json.loads(metadata_path.read_text('utf-8'))
    selected_sessions = {
        session
        for split in ('train', 'val')
        for session in manifest['splits'][split]['sessions']
    }
    unreviewed = [
        session for session in sorted(selected_sessions)
        if metadata.get('sessions', {}).get(session, {}).get(
            'annotation_status'
        ) != 'human_reviewed'
    ]
    if unreviewed:
        raise ValueError(
            'Training is blocked until these sessions are human-reviewed: '
            + ', '.join(unreviewed)
        )


def validate_dataset_yaml(data_path, allow_unreviewed=False):
    """Validate split files, images, session isolation, and review state."""
    data_path = data_path.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f'Prepared dataset YAML not found: {data_path}. Complete Phase 1 '
            'and generate runner_v1 before training.'
        )
    dataset = _load_yaml(data_path)
    for required in ('train', 'val', 'names'):
        if required not in dataset:
            raise ValueError(f'Dataset YAML is missing {required!r}')
    dataset_root = _resolve_path(
        dataset.get('path', '.'), data_path.parent
    )
    split_images = {}
    for split in ('train', 'val'):
        split_value = dataset[split]
        if not isinstance(split_value, str):
            raise ValueError(
                f'{split} must reference one image-list text file'
            )
        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        split_images[split] = _read_image_source(split_path.resolve())
    overlap = set(split_images['train']) & set(split_images['val'])
    if overlap:
        raise ValueError(
            f'{len(overlap)} images leak across train and validation'
        )

    manifest_path = data_path.parent / 'splits_v1.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'Leakage manifest not found: {manifest_path}'
        )
    manifest = json.loads(manifest_path.read_text('utf-8'))
    if 'splits' in manifest:
        train_sessions = set(manifest['splits']['train']['sessions'])
        val_sessions = set(manifest['splits']['val']['sessions'])
    elif 'session_splits' in manifest:
        train_sessions = {
            session for session, split in manifest['session_splits'].items()
            if split == 'train'
        }
        val_sessions = {
            session for session, split in manifest['session_splits'].items()
            if split == 'val'
        }
    else:
        raise ValueError(
            'splits_v1.json must contain splits or session_splits'
        )
    leaked_sessions = train_sessions & val_sessions
    if leaked_sessions:
        raise ValueError(
            'Recording sessions leak across splits: '
            + ', '.join(sorted(leaked_sessions))
        )
    if 'splits' in manifest:
        _validate_review_status(manifest, allow_unreviewed)
    elif not allow_unreviewed and manifest.get(
        'annotation_status'
    ) != 'human_reviewed':
        raise ValueError(
            'Public-video labels are not marked human-reviewed; pass '
            '--allow-unreviewed only after completing the audit workflow'
        )
    return {
        'data_yaml': data_path.as_posix(),
        'train_images': len(split_images['train']),
        'val_images': len(split_images['val']),
        'train_sessions': len(train_sessions),
        'val_sessions': len(val_sessions),
        'session_leakage': False,
        'human_review_required': not allow_unreviewed,
    }


def _validate_probability(value, name):
    """Return a probability after enforcing its valid range."""
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f'{name} must be between 0 and 1')
    return probability


def build_custom_augmentations(profile, image_size):
    """Build blur, occlusion, and compression transforms from YAML."""
    if not profile.get('enabled', False):
        return None
    try:
        import albumentations as album
    except ImportError as error:
        raise RuntimeError(
            'Custom robust augmentations require albumentations>=1.4'
        ) from error

    blur = profile['blur']
    blur_probability = _validate_probability(
        blur['probability'], 'blur.probability'
    )
    motion_weight = _validate_probability(
        blur['motion_weight'], 'blur.motion_weight'
    )
    transforms = [
        album.OneOf(
            [
                album.MotionBlur(
                    blur_limit=tuple(blur['motion_blur_limit']),
                    p=motion_weight,
                ),
                album.GaussianBlur(
                    blur_limit=tuple(blur['gaussian_blur_limit']),
                    p=1.0 - motion_weight,
                ),
            ],
            p=blur_probability,
        )
    ]

    occlusion = profile['partial_occlusion']
    occlusion_probability = _validate_probability(
        occlusion['probability'], 'partial_occlusion.probability'
    )
    signature = inspect.signature(album.CoarseDropout)
    holes = tuple(int(value) for value in occlusion['holes'])
    heights = tuple(float(value) for value in occlusion['height_fraction'])
    widths = tuple(float(value) for value in occlusion['width_fraction'])
    if 'num_holes_range' in signature.parameters:
        dropout = album.CoarseDropout(
            num_holes_range=holes,
            hole_height_range=heights,
            hole_width_range=widths,
            fill=0,
            p=occlusion_probability,
        )
    else:
        dropout = album.CoarseDropout(
            min_holes=holes[0],
            max_holes=holes[1],
            min_height=max(1, round(heights[0] * image_size)),
            max_height=max(1, round(heights[1] * image_size)),
            min_width=max(1, round(widths[0] * image_size)),
            max_width=max(1, round(widths[1] * image_size)),
            fill_value=0,
            p=occlusion_probability,
        )
    transforms.append(dropout)

    compression = profile['wireless_compression']
    compression_probability = _validate_probability(
        compression['probability'], 'wireless_compression.probability'
    )
    quality = tuple(int(value) for value in compression['quality'])
    compression_signature = inspect.signature(album.ImageCompression)
    compression_arguments = {'p': compression_probability}
    if 'quality_range' in compression_signature.parameters:
        compression_arguments['quality_range'] = quality
    else:
        compression_arguments.update({
            'quality_lower': quality[0],
            'quality_upper': quality[1],
        })
    transforms.append(album.ImageCompression(**compression_arguments))
    return transforms


def robust_trainer_class(custom_transforms):
    """Create a trainer that injects custom transforms into train data only."""
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    class RobustDetectionTrainer(DetectionTrainer):
        """Ultralytics detection trainer with drone-specific augmentation."""

        def build_dataset(self, image_path, mode='train', batch=None):
            if mode == 'train':
                self.args.augmentations = custom_transforms
            return super().build_dataset(image_path, mode, batch)

    return RobustDetectionTrainer


def metric_summary(metrics):
    """Convert Ultralytics detection metrics into serializable values."""
    box = metrics.box
    return {
        'precision': float(box.mp),
        'recall': float(box.mr),
        'map50': float(box.map50),
        'map75': float(box.map75),
        'map50_95': float(box.map),
        'speed_ms_per_image': {
            str(name): float(value)
            for name, value in getattr(metrics, 'speed', {}).items()
        },
    }


def analyze_overfitting(train_metrics, val_metrics, thresholds):
    """Compare unbiased evaluations on train and validation splits."""
    gaps = {
        name: float(train_metrics[name] - val_metrics[name])
        for name in METRIC_NAMES
    }
    warnings = [
        name for name, gap in gaps.items()
        if gap > float(thresholds[name])
    ]
    return {
        'gaps_train_minus_val': gaps,
        'thresholds': {
            name: float(thresholds[name]) for name in METRIC_NAMES
        },
        'warning_metrics': warnings,
        'possible_overfitting': bool(warnings),
    }


def print_metric_comparison(train_metrics, val_metrics, analysis):
    """Print train and validation performance in one aligned table."""
    print('\nPost-training generalization check')
    print(f'{"metric":<12} {"train":>10} {"val":>10} {"gap":>10}')
    print('-' * 44)
    for name in (*METRIC_NAMES, 'map75'):
        gap = train_metrics[name] - val_metrics[name]
        print(
            f'{name:<12} {train_metrics[name]:>10.4f} '
            f'{val_metrics[name]:>10.4f} {gap:>10.4f}'
        )
    status = 'WARNING' if analysis['possible_overfitting'] else 'PASS'
    print(f'Overfitting check: {status}')
    if analysis['warning_metrics']:
        print(
            'Large train/val gaps: '
            + ', '.join(analysis['warning_metrics'])
        )


def _atomic_json(path, payload):
    """Write a JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def _config_digest(path):
    """Hash the source training configuration for provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_training(config, dataset_summary, resume=None):
    """Train, select the best checkpoint, and evaluate both data splits."""
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('Install Ultralytics to train the model') from error

    if resume:
        resume = resume.expanduser().resolve()
        if not resume.is_file():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume}')
    model_source = str(resume) if resume else config['model']
    model = YOLO(model_source)
    train_arguments = dict(config['train'])
    train_arguments.update({
        'data': str(config['data']),
        'project': str(config['project']),
        'name': config['name'],
    })
    custom = build_custom_augmentations(
        config['robust_augmentations'], int(train_arguments['imgsz'])
    )
    if resume:
        train_arguments['resume'] = str(resume)

    if custom is None:
        model.train(**train_arguments)
    else:
        model.train(
            trainer=robust_trainer_class(custom),
            **train_arguments,
        )
    save_dir = Path(model.trainer.save_dir).resolve()
    best_checkpoint = Path(model.trainer.best).resolve()
    if not best_checkpoint.is_file():
        best_checkpoint = Path(model.trainer.last).resolve()
    if not best_checkpoint.is_file():
        raise RuntimeError('Training completed without a usable checkpoint')

    best_model = YOLO(str(best_checkpoint))
    evaluation = config['evaluation']
    common_validation = {
        'data': str(config['data']),
        'imgsz': int(train_arguments['imgsz']),
        'batch': evaluation['split_batch'],
        'workers': evaluation['workers'],
        'device': train_arguments.get('device'),
        'plots': evaluation.get('plots', True),
        'project': str(save_dir / 'evaluation'),
        'exist_ok': True,
        'verbose': False,
    }
    train_result = best_model.val(
        split='train', name='train_split', **common_validation
    )
    val_result = best_model.val(
        split='val', name='val_split', **common_validation
    )
    train_metrics = metric_summary(train_result)
    val_metrics = metric_summary(val_result)
    analysis = analyze_overfitting(
        train_metrics,
        val_metrics,
        evaluation['overfitting_thresholds'],
    )
    print_metric_comparison(train_metrics, val_metrics, analysis)

    report = {
        'schema_version': 1,
        'model_checkpoint': best_checkpoint.as_posix(),
        'run_directory': save_dir.as_posix(),
        'dataset': dataset_summary,
        'train_metrics': train_metrics,
        'validation_metrics': val_metrics,
        'overfitting_analysis': analysis,
        'provenance': {
            'config': config['_config_path'].as_posix(),
            'config_sha256': _config_digest(config['_config_path']),
            'ultralytics_version': ultralytics.__version__,
            'python_version': platform.python_version(),
            'platform': platform.platform(),
        },
    }
    _atomic_json(save_dir / 'overfitting_report.json', report)
    return report


def main(argv=None):
    """Validate inputs and run robust runner-detector training."""
    arguments = parse_arguments(argv)
    try:
        config = apply_overrides(
            load_training_config(arguments.config), arguments
        )
        dataset_summary = validate_dataset_yaml(
            config['data'], arguments.allow_unreviewed
        )
        if arguments.dry_run:
            print(json.dumps({
                'status': 'ready',
                'model': config['model'],
                'project': str(config['project']),
                'name': config['name'],
                'dataset': dataset_summary,
                'custom_augmentations': config[
                    'robust_augmentations'
                ].get('enabled', False),
            }, indent=2))
            return 0
        report = run_training(config, dataset_summary, arguments.resume)
    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    print(f'Best checkpoint: {report["model_checkpoint"]}')
    print(
        'Overfitting report: '
        f'{report["run_directory"]}/overfitting_report.json'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
