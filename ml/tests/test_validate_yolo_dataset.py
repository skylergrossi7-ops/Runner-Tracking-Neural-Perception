"""Unit tests for YOLO dataset structural validation."""

import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from validate_yolo_dataset import validate_label_line  # noqa: E402, I100, I201
import pytest  # noqa: E402, I100, I201


def test_validate_label_line_accepts_runner_box():
    """A valid runner box should return its class and coordinates."""
    class_id, coordinates = validate_label_line(
        '0 0.5 0.5 0.2 0.8', class_count=1
    )
    assert class_id == 0
    assert coordinates == [0.5, 0.5, 0.2, 0.8]


@pytest.mark.parametrize(
    'row',
    (
        '1 0.5 0.5 0.2 0.4',
        '0 0.95 0.5 0.2 0.4',
        '0 0.5 0.5 0.0 0.4',
        'runner 0.5 0.5 0.2 0.4',
    ),
)
def test_validate_label_line_rejects_invalid_rows(row):
    """Bad class IDs and geometry must be reported before splitting."""
    with pytest.raises(ValueError):
        validate_label_line(row, class_count=1)
