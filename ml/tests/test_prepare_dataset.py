"""Tests for leakage-safe YOLO dataset preparation."""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'prepare_dataset.py'


def _add_sample(root, relative_image):
    """Create one minimal mirrored image/label pair."""
    image = root / 'images' / relative_image
    label = (root / 'labels' / relative_image).with_suffix('.txt')
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b'not-decoded-by-this-tool')
    label.write_text('0 0.5 0.5 0.2 0.4\n', encoding='utf-8')


def test_split_keeps_sessions_exclusive_and_generates_yolo_files(tmp_path):
    """No session may leak across train and validation outputs."""
    source = tmp_path / 'source'
    output = tmp_path / 'prepared'
    conditions = (
        'near_day_clear', 'mid_day_partial', 'far_day_occluded',
        'near_night_partial', 'mid_night_occluded', 'far_night_clear',
        'near_overcast_occluded', 'mid_overcast_clear',
        'far_overcast_partial', 'near_backlit_clear',
        'mid_backlit_partial', 'far_backlit_occluded',
    )
    for index, condition in enumerate(conditions):
        session = f'bag_{index:02d}_{condition}'
        for frame in range(index % 3 + 1):
            _add_sample(source, Path(session) / f'frame_{frame:06d}.jpg')

    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            '--dataset-root', str(source),
            '--output-dir', str(output),
            '--path-mode', 'relative',
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / 'splits_v1.json').read_text('utf-8'))
    train_sessions = set(manifest['splits']['train']['sessions'])
    val_sessions = set(manifest['splits']['val']['sessions'])
    assert train_sessions
    assert val_sessions
    assert train_sessions.isdisjoint(val_sessions)
    assert len(train_sessions | val_sessions) == len(conditions)
    assert not manifest['balance']['unbalanced_representable_conditions']
    for tag, split_sessions in manifest['balance'][
        'condition_sessions'
    ].items():
        if sum(bool(values) for values in split_sessions.values()) == 2:
            assert split_sessions['train'], tag
            assert split_sessions['val'], tag

    train_images = set(manifest['splits']['train']['images'])
    val_images = set(manifest['splits']['val']['images'])
    assert train_images.isdisjoint(val_images)
    assert (output / 'train.txt').is_file()
    assert (output / 'val.txt').is_file()
    yaml_text = (output / 'runner_v1.yaml').read_text('utf-8')
    assert 'train: "train.txt"' in yaml_text
    assert 'val: "val.txt"' in yaml_text
    assert '0: "runner"' in yaml_text


def test_flat_filenames_can_use_explicit_session_regex(tmp_path):
    """A configured regex groups frames from flattened rosbag exports."""
    source = tmp_path / 'source'
    output = tmp_path / 'prepared'
    for session in ('run_a', 'run_b', 'run_c', 'run_d', 'run_e'):
        for frame in range(2):
            _add_sample(source, Path(f'{session}_frame_{frame:06d}.png'))

    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            '--dataset-root', str(source),
            '--output-dir', str(output),
            '--session-regex', r'^(?P<session>run_[a-e])_',
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / 'splits_v1.json').read_text('utf-8'))
    all_sessions = (
        manifest['splits']['train']['sessions']
        + manifest['splits']['val']['sessions']
    )
    assert sorted(all_sessions) == [
        'run_a', 'run_b', 'run_c', 'run_d', 'run_e'
    ]


def test_missing_label_fails_without_explicit_negative_image_flag(tmp_path):
    """Accidental label loss must fail before any manifest is written."""
    source = tmp_path / 'source'
    (source / 'images' / 'bag_a').mkdir(parents=True)
    (source / 'labels').mkdir(parents=True)
    (source / 'images' / 'bag_a' / 'frame_000001.jpg').write_bytes(b'x')
    output = tmp_path / 'prepared'

    completed = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            '--dataset-root', str(source),
            '--output-dir', str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert 'no mirrored YOLO label' in completed.stderr
    assert not (output / 'splits_v1.json').exists()
