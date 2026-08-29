"""Tests for batch ROS 2 session discovery and ingestion validation."""

import argparse
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from ingest_sessions import (  # noqa: E402, I100, I201
    ingest,
    metadata_for_bag,
    next_session_id,
    normalize_metadata,
    scan_bag_directories,
    validate_session_output,
)
import ingest_sessions  # noqa: E402, I100, I201
import cv2  # noqa: E402, I100, I201
import numpy  # noqa: E402, I100, I201
import pytest  # noqa: E402, I100, I201


def test_scan_finds_each_rosbag_directory_once(tmp_path):
    """Chunked bags should resolve to one session directory each."""
    first = tmp_path / 'recording_a'
    first.mkdir()
    (first / 'recording_a_0.mcap').touch()
    (first / 'recording_a_1.mcap').touch()
    second = tmp_path / 'nested' / 'recording_b'
    second.mkdir(parents=True)
    (second / 'recording_b_0.db3').touch()

    assert set(scan_bag_directories(tmp_path)) == {first, second}


def test_next_session_id_uses_bag_naming_and_avoids_reserved_ids():
    """Automatic allocation must never overwrite an existing session."""
    assert next_session_id(
        {'gazebo_runway_001', 'bag_001', 'bag_003'}, {'bag_004'}
    ) == 'bag_005'


def test_metadata_normalization_enforces_controlled_values():
    """Aliases are normalized but unknown collection tags are rejected."""
    normalized = normalize_metadata({
        'distance': 'MID',
        'lighting': 'lowlight',
        'occlusion': 'clear',
    })
    assert normalized['conditions'] == {
        'distance': 'mid',
        'lighting': 'low',
        'occlusion': 'none',
    }
    with pytest.raises(ValueError):
        normalize_metadata({
            'distance': 'unknown',
            'lighting': 'normal',
            'occlusion': 'none',
        })


def test_non_interactive_metadata_lookup_requires_entry(tmp_path):
    """Unattended ingestion must not silently invent environmental tags."""
    bag = tmp_path / 'run_a'
    bag.mkdir()
    with pytest.raises(ValueError):
        metadata_for_bag(bag, tmp_path, {}, non_interactive=True)


def test_validate_session_output_checks_decoding_and_names(tmp_path):
    """A correctly named, decodable image session should pass validation."""
    session = tmp_path / 'bag_001'
    session.mkdir()
    image = numpy.zeros((12, 16, 3), dtype=numpy.uint8)
    output = session / 'bag_001_frame_000000_t123456.jpg'
    assert cv2.imwrite(str(output), image)

    result = validate_session_output(session, 'bag_001')

    assert result['image_count'] == 1
    assert result['dimensions'] == [{'width': 16, 'height': 12}]
    assert result['naming_valid'] is True


def test_validate_session_output_rejects_wrong_filename(tmp_path):
    """Files that cannot be grouped by the splitter must fail ingestion."""
    session = tmp_path / 'bag_002'
    session.mkdir()
    image = numpy.zeros((4, 4, 3), dtype=numpy.uint8)
    assert cv2.imwrite(str(session / 'frame.jpg'), image)

    with pytest.raises(ValueError):
        validate_session_output(session, 'bag_002')


def test_ingest_routes_new_bag_through_extractor_and_validation(
    tmp_path, monkeypatch
):
    """A new tagged bag should be numbered, extracted, and QA checked."""
    inbox = tmp_path / 'inbox'
    source = inbox / 'outdoor_run_01'
    source.mkdir(parents=True)
    (source / 'outdoor_run_01_0.mcap').write_bytes(b'bag')
    dataset = tmp_path / 'runner_raw'
    metadata_file = inbox / 'session_tags.json'
    metadata_file.write_text(json.dumps({
        'outdoor_run_01': {
            'distance': 'mid',
            'lighting': 'normal',
            'occlusion': 'none',
        }
    }), encoding='utf-8')
    captured = {}

    def fake_extract(arguments):
        captured['arguments'] = arguments
        image_dir = dataset / 'images' / arguments.session_id
        image_dir.mkdir(parents=True)
        (dataset / 'labels' / arguments.session_id).mkdir(parents=True)
        image = numpy.zeros((8, 10, 3), dtype=numpy.uint8)
        filename = f'{arguments.session_id}_frame_000000_t100.jpg'
        assert cv2.imwrite(str(image_dir / filename), image)
        return image_dir, dataset / 'session_metadata.json', 1

    monkeypatch.setattr(ingest_sessions, 'bag_digest', lambda _path: 'abc')
    monkeypatch.setattr(ingest_sessions, 'extract', fake_extract)
    arguments = argparse.Namespace(
        input_dir=inbox,
        dataset_root=dataset,
        topic='/camera/image_raw',
        every_nth_frame=5,
        maximum_frames=0,
        jpeg_quality=95,
        metadata_file=metadata_file,
        non_interactive=True,
        environment='real',
        report='ingestion_report.json',
    )

    _report_path, report = ingest(arguments)

    assert report['sessions_ingested'] == 1
    assert report['sessions_failed'] == 0
    assert report['results'][0]['session_id'] == 'bag_001'
    assert report['results'][0]['validation']['image_count'] == 1
    assert captured['arguments'].every_nth_frame == 5
    assert captured['arguments'].condition == [
        'distance=mid',
        'environment=real',
        'lighting=normal',
        'occlusion=none',
    ]
