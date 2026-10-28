"""CPU mock of optional census debug readback."""
from types import SimpleNamespace
import numpy as np
import pytest
from stereoforge.gpu import VulkanStereo


class Buffer:
    def __init__(self, values): self.values = np.array(values, dtype=np.uint32); self.reads = 0
    def to_numpy(self): self.reads += 1; return self.values


def engine():
    e = VulkanStereo.__new__(VulkanStereo)
    e.shape = (1, 2); e.D = 1; e._pending = 7; e._stages = []
    e.last_timings = {}; e.device = SimpleNamespace(wait_for_submit=lambda token: None)
    e.query_pool = SimpleNamespace(get_timestamp_results=lambda start, count: [])
    e.buffers = {name: Buffer([i, i+1]) for i, name in enumerate(
        ('disparity', 'confidence', 'validity', 'ld', 'rd', 'lv', 'rv', 'costs', 'sums'))}
    return e


def test_default_debug_does_not_add_readback():
    e = engine(); out = e.result(7, debug=True)
    assert set(out) == {'disparity', 'confidence', 'valid', 'left_census', 'cost', 'aggregate'}
    assert all(e.buffers[k].reads == 0 for k in ('rd', 'lv', 'rv'))


def test_opt_in_reads_all_census_inputs_and_copies():
    e = engine(); out = e.result(7, debug=True, debug_census=True)
    for key, name in [('right_census','rd'), ('left_census_valid','lv'), ('right_census_valid','rv')]:
        np.testing.assert_array_equal(out[key].ravel(), e.buffers[name].values)
        assert e.buffers[name].reads == 1
        e.buffers[name].values[:] = 999
        assert out[key][0,0] != 999


def test_invalid_opt_in_does_not_consume_submission():
    e = engine()
    with pytest.raises(ValueError): e.result(7, debug_census=True)
    assert e._pending == 7
