import pytest
from benchmarks.fp2_stage_timestamps import queue_intervals


def test_order_and_single_wrap():
    assert queue_intervals([254, 255, 2], [1, 1, 1], 8, 10, 50) == [0.00001, 0.00003]


@pytest.mark.parametrize('args', [
    ([1, 2], [1, 0], 8, 1, 10),
    ([1, 2], [1], 8, 1, 10),
    ([1, 2], [1, 1], 0, 1, 10),
    ([1, 2], [1, 1], 8, 0, 10),
    ([1, 2], [1, 1], 8, 1, 256),
    ([1, 256], [1, 1], 8, 1, 10),
    ([5, 3, 4], [1, 1, 1], 8, 1, 20),
])
def test_reject_ambiguous_or_invalid(args):
    with pytest.raises(ValueError):
        queue_intervals(*args)
