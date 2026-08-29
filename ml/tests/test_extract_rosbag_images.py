"""Unit tests for session-scoped rosbag extraction helpers."""

import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from extract_rosbag_images import (  # noqa: E402, I100, I201
    detect_storage_id,
    parse_condition,
)
import pytest  # noqa: E402, I100, I201


def test_parse_condition_normalizes_names_and_values():
    """Condition metadata should be stable across recording commands."""
    assert parse_condition(' Lighting = Low Light ') == (
        'lighting', 'low_light'
    )


def test_parse_condition_rejects_missing_value():
    """Malformed collection metadata must fail before extraction."""
    with pytest.raises(ValueError):
        parse_condition('occlusion=')


def test_detect_storage_id_supports_mcap_and_sqlite(tmp_path):
    """Both common rosbag2 storage formats are supported."""
    mcap = tmp_path / 'mcap'
    mcap.mkdir()
    (mcap / 'session_0.mcap').touch()
    assert detect_storage_id(mcap) == 'mcap'

    sqlite = tmp_path / 'sqlite'
    sqlite.mkdir()
    (sqlite / 'session_0.db3').touch()
    assert detect_storage_id(sqlite) == 'sqlite3'
