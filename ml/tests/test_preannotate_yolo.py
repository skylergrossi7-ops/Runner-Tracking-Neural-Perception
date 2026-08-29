"""Unit tests for model-assisted YOLO annotation helpers."""

import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from preannotate_yolo import format_detection  # noqa: E402, I100, I201
import pytest  # noqa: E402, I100, I201


def test_format_detection_produces_yolo_row():
    """A normalized teacher detection should become one YOLO row."""
    assert format_detection(0, [0.5, 0.4, 0.2, 0.6]) == (
        '0 0.50000000 0.40000000 0.20000000 0.60000000'
    )


@pytest.mark.parametrize(
    'box',
    ([0.5, 0.5, 0.0, 0.4], [1.2, 0.5, 0.2, 0.4], [0.5, 0.5, 0.2]),
)
def test_format_detection_rejects_invalid_boxes(box):
    """Invalid teacher predictions must never enter the label set."""
    with pytest.raises(ValueError):
        format_detection(0, box)
