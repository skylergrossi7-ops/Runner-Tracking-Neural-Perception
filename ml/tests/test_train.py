"""Tests for robust runner-detector training safeguards."""

import json
import sys
from pathlib import Path


ML_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ML_ROOT))

from train import (  # noqa: E402, I100, I201
    analyze_overfitting,
    build_custom_augmentations,
    load_training_config,
    validate_dataset_yaml,
)
import pytest  # noqa: E402, I100, I201


def _create_prepared_dataset(tmp_path, review_status='human_reviewed'):
    """Create a minimal session-exclusive runner_v1 fixture."""
    prepared = tmp_path / 'runner_v1'
    images = prepared / 'images'
    images.mkdir(parents=True)
    train_image = images / 'train.jpg'
    val_image = images / 'val.jpg'
    train_image.write_bytes(b'train-image')
    val_image.write_bytes(b'val-image')
    (prepared / 'train.txt').write_text(
        'images/train.jpg\n', encoding='utf-8'
    )
    (prepared / 'val.txt').write_text(
        'images/val.jpg\n', encoding='utf-8'
    )
    data_yaml = prepared / 'runner_v1.yaml'
    data_yaml.write_text(
        'path: "."\ntrain: "train.txt"\nval: "val.txt"\n'
        'names:\n  0: "runner"\n',
        encoding='utf-8',
    )
    raw = tmp_path / 'runner_raw'
    raw.mkdir()
    (raw / 'session_metadata.json').write_text(json.dumps({
        'sessions': {
            'bag_001': {'annotation_status': review_status},
            'bag_002': {'annotation_status': review_status},
        }
    }), encoding='utf-8')
    manifest = {
        'source': {'dataset_root': str(raw)},
        'splits': {
            'train': {
                'sessions': ['bag_001'],
                'images': [str(train_image)],
            },
            'val': {
                'sessions': ['bag_002'],
                'images': [str(val_image)],
            },
        },
    }
    (prepared / 'splits_v1.json').write_text(
        json.dumps(manifest), encoding='utf-8'
    )
    return data_yaml


def test_default_training_config_resolves_project_paths():
    """The checked-in profile should resolve independently of the shell CWD."""
    config = load_training_config(
        ML_ROOT / 'config' / 'train_robust.yaml'
    )
    assert config['data'].name == 'runner_v1.yaml'
    assert config['project'].name == 'runs'
    assert config['train']['mosaic'] == 0.75
    assert config['robust_augmentations']['blur']['probability'] == 0.18


def test_validate_dataset_requires_disjoint_reviewed_sessions(tmp_path):
    """A valid prepared dataset should pass every pre-training gate."""
    data_yaml = _create_prepared_dataset(tmp_path)

    summary = validate_dataset_yaml(data_yaml)

    assert summary['train_images'] == 1
    assert summary['val_images'] == 1
    assert summary['session_leakage'] is False


def test_validate_dataset_blocks_unreviewed_labels(tmp_path):
    """Teacher-generated boxes cannot silently become training truth."""
    data_yaml = _create_prepared_dataset(
        tmp_path, review_status='preannotated_requires_human_review'
    )

    with pytest.raises(ValueError, match='human-reviewed'):
        validate_dataset_yaml(data_yaml)


def test_validate_dataset_blocks_session_leakage(tmp_path):
    """A recording session may belong to only one evaluation split."""
    data_yaml = _create_prepared_dataset(tmp_path)
    manifest_path = data_yaml.parent / 'splits_v1.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    manifest['splits']['val']['sessions'] = ['bag_001']
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    with pytest.raises(ValueError, match='sessions leak'):
        validate_dataset_yaml(data_yaml)


def test_validate_public_directory_layout_and_session_manifest(tmp_path):
    """Public-video datasets may use Ultralytics split directories."""
    prepared = tmp_path / 'public_runner_v1'
    for split in ('train', 'val'):
        image_dir = prepared / 'images' / split
        label_dir = prepared / 'labels' / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / f'{split}.jpg').write_bytes(b'image')
        (label_dir / f'{split}.txt').write_text(
            '0 0.5 0.5 0.2 0.4\n', encoding='utf-8'
        )
    data_yaml = prepared / 'runner_v1.yaml'
    data_yaml.write_text(
        'path: .\ntrain: images/train\nval: images/val\n'
        'names:\n  0: runner\n',
        encoding='utf-8',
    )
    (prepared / 'splits_v1.json').write_text(
        json.dumps({
            'annotation_status': 'human_reviewed',
            'session_splits': {
                'public_session_train': 'train',
                'public_session_val': 'val',
            },
        }),
        encoding='utf-8',
    )

    summary = validate_dataset_yaml(data_yaml)

    assert summary['train_images'] == 1
    assert summary['val_images'] == 1
    assert summary['train_sessions'] == 1
    assert summary['val_sessions'] == 1


def test_overfitting_analysis_flags_large_generalization_gap():
    """Large train-minus-validation gaps should identify suspect metrics."""
    train_metrics = {
        'precision': 0.94,
        'recall': 0.90,
        'map50': 0.95,
        'map50_95': 0.78,
    }
    val_metrics = {
        'precision': 0.80,
        'recall': 0.82,
        'map50': 0.74,
        'map50_95': 0.60,
    }
    thresholds = {
        'precision': 0.12,
        'recall': 0.12,
        'map50': 0.12,
        'map50_95': 0.10,
    }

    result = analyze_overfitting(
        train_metrics, val_metrics, thresholds
    )

    assert result['possible_overfitting'] is True
    assert set(result['warning_metrics']) == {
        'precision', 'map50', 'map50_95'
    }


def test_custom_augmentations_can_be_disabled_without_dependency():
    """CPU-only or minimal environments may explicitly disable custom ops."""
    assert build_custom_augmentations({'enabled': False}, 640) is None
