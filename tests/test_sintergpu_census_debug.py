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
    np.testing.assert_array_equal(out['raw_validity'].ravel(), e.buffers['validity'].values)
    assert out['raw_validity'].dtype == np.uint32
    for key, name in [('right_census','rd'), ('left_census_valid','lv'), ('right_census_valid','rv')]:
        np.testing.assert_array_equal(out[key].ravel(), e.buffers[name].values)
        assert e.buffers[name].reads == 1
        e.buffers[name].values[:] = 999
        assert out[key][0,0] != 999


def test_invalid_opt_in_does_not_consume_submission():
    e = engine()
    with pytest.raises(ValueError): e.result(7, debug_census=True)
    assert e._pending == 7


def test_pipeline_interval_uses_first_and_last_gpu_timestamps():
    e=engine(); e._stages=['first','last']
    e.query_pool = SimpleNamespace(get_timestamp_results=lambda start, count: [1.0, 1.1, 1.3, 1.5])
    e.result(7)
    assert e.last_timings['gpu_pipeline_ms'] == pytest.approx(500.0)
    assert sum(e.last_timings['gpu_stages_ms'].values()) == pytest.approx(300.0)


def test_fixture_adapter_uses_real_result_with_fake_buffers():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'benchmarks/sintergpu_fixture_execution.py'
    spec = importlib.util.spec_from_file_location('fixture_capture_real_result', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    e=engine(); e.shape=(5,5); e.D=1; e._stages=['first','last']
    e.query_pool=SimpleNamespace(get_timestamp_results=lambda start,count:[1.0,1.1,1.2,1.3])
    e.buffers={name:Buffer(np.arange(25)) for name in e.buffers}
    e.enqueue=lambda left,right,lower=None,upper=None: 7
    image=np.zeros((5,5),dtype=np.uint32)
    result=module.capture(e,image,image)
    assert set(result['outputs']) == set(module.OUTPUTS)
    np.testing.assert_array_equal(result['outputs']['raw_validity'], e.buffers['validity'].values.reshape(5,5))
    assert result['pipeline_ms']==pytest.approx(300.0)
