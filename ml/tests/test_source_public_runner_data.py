"""Pure regression tests for public runner dataset sourcing utilities."""

import argparse
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from source_public_runner_data import (  # noqa: E402
    AcceptedSample,
    PoseCandidate,
    filter_short_running_sequences,
    full_body_box_is_valid,
    parse_yolo_labels,
    running_pose_score,
    stable_session_splits,
    validate_arguments,
    write_yaml,
    yolo_row,
)


def _running_keypoints():
    points = [
        (100.0, 30.0, 0.95),
        (96.0, 28.0, 0.90),
        (104.0, 28.0, 0.90),
        (92.0, 31.0, 0.85),
        (108.0, 31.0, 0.85),
        (80.0, 60.0, 0.95),
        (120.0, 60.0, 0.95),
        (65.0, 82.0, 0.90),
        (135.0, 82.0, 0.90),
        (82.0, 96.0, 0.90),
        (118.0, 96.0, 0.90),
        (90.0, 120.0, 0.95),
        (110.0, 120.0, 0.95),
        (70.0, 145.0, 0.95),
        (120.0, 155.0, 0.95),
        (80.0, 180.0, 0.95),
        (150.0, 195.0, 0.95),
    ]
    return tuple(points)


def _standing_keypoints():
    points = list(_running_keypoints())
    points[13] = (92.0, 155.0, 0.95)
    points[14] = (108.0, 155.0, 0.95)
    points[15] = (94.0, 195.0, 0.95)
    points[16] = (106.0, 195.0, 0.95)
    points[7] = (78.0, 95.0, 0.90)
    points[8] = (122.0, 95.0, 0.90)
    points[9] = (76.0, 125.0, 0.90)
    points[10] = (124.0, 125.0, 0.90)
    return tuple(points)


def test_running_pose_filter_separates_stride_from_standing():
    assert running_pose_score(_running_keypoints(), 0.35) >= 0.58
    assert running_pose_score(_standing_keypoints(), 0.35) < 0.58


def test_full_body_check_rejects_border_clipping():
    valid = PoseCandidate(
        box=(20.0, 10.0, 180.0, 210.0),
        confidence=0.9,
        keypoints=_running_keypoints(),
    )
    clipped = PoseCandidate(
        box=(0.0, 10.0, 180.0, 210.0),
        confidence=0.9,
        keypoints=_running_keypoints(),
    )

    assert full_body_box_is_valid(valid, (240, 320, 3), 0.35, 45)
    assert not full_body_box_is_valid(clipped, (240, 320, 3), 0.35, 45)


def test_yolo_parser_filters_nonhuman_source_classes(tmp_path):
    label = tmp_path / 'frame.txt'
    label.write_text(
        '0 0.5 0.5 0.25 0.50\n1 0.2 0.2 0.10 0.10\n',
        encoding='utf-8',
    )

    boxes = parse_yolo_labels(
        label, width=320, height=240,
        class_names={0: 'runner', 1: 'bicycle'},
    )

    assert boxes == [(120.0, 60.0, 200.0, 180.0)]


def test_session_split_never_leaks_a_session():
    splits = stable_session_splits(
        {'video_a': 100, 'video_b': 30, 'video_c': 20},
        val_fraction=0.2,
        seed=42,
    )

    assert set(splits) == {'video_a', 'video_b', 'video_c'}
    assert set(splits.values()) == {'train', 'val'}


def test_session_split_chooses_count_closest_to_target():
    splits = stable_session_splits(
        {'large': 15, 'medium': 13, 'small': 5},
        val_fraction=0.2,
        seed=42,
    )

    assert splits['small'] == 'val'
    assert splits['large'] == 'train'
    assert splits['medium'] == 'train'


def test_temporal_filter_rejects_short_pose_bursts(tmp_path):
    def sample(name, timestamp):
        return AcceptedSample(
            session='video_a',
            image=tmp_path / f'{name}.jpg',
            label=tmp_path / f'{name}.txt',
            source='video.mp4',
            frame_index=int(timestamp * 10),
            timestamp_seconds=timestamp,
            runner_count=1,
            minimum_run_score=0.4,
        )

    samples = [
        sample('false_0', 0.0),
        sample('false_1', 0.5),
        sample('run_0', 5.0),
        sample('run_1', 6.0),
        sample('run_2', 7.0),
    ]

    filtered, rejected = filter_short_running_sequences(
        samples, minimum_duration=2.0, maximum_gap=2.0
    )

    assert [item.image.stem for item in filtered] == [
        'run_0', 'run_1', 'run_2'
    ]
    assert rejected == 2


def test_temporal_filter_keeps_still_images_for_review(tmp_path):
    sample = AcceptedSample(
        session='image_session',
        image=tmp_path / 'frame.jpg',
        label=tmp_path / 'frame.txt',
        source='frame.jpg',
        frame_index=0,
        timestamp_seconds=None,
        runner_count=1,
        minimum_run_score=0.7,
    )

    filtered, rejected = filter_short_running_sequences(
        [sample], minimum_duration=2.0, maximum_gap=2.0
    )

    assert filtered == [sample]
    assert rejected == 0


def test_yaml_is_standard_single_class_ultralytics_config(tmp_path):
    path = write_yaml(tmp_path.resolve())
    text = path.read_text('utf-8')

    assert 'train: images/train' in text
    assert 'val: images/val' in text
    assert 'runner' in text
    assert 'nc: 1' in text


def test_yolo_output_is_normalized():
    assert yolo_row((80, 60, 240, 180), 320, 240) == (
        '0 0.50000000 0.50000000 0.50000000 0.50000000'
    )


def test_remote_provider_requires_license_acknowledgement(tmp_path):
    arguments = argparse.Namespace(
        provider='kaggle', input=None, roboflow_version=None,
        kaggle_dataset='owner/dataset', license_name='CC-BY-4.0',
        license_url='https://example.test/license', accept_license=False,
        person_confidence=0.45, keypoint_confidence=0.35,
        run_score_threshold=0.58, val_fraction=0.2,
        frame_interval_seconds=0.5, minimum_box_height=45,
        minimum_running_sequence_seconds=2.0,
        maximum_running_gap_seconds=2.0,
    )

    try:
        validate_arguments(arguments)
    except ValueError as error:
        assert '--accept-license' in str(error)
    else:
        raise AssertionError('Expected license acknowledgement failure')
