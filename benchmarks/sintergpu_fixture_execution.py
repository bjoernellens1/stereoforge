"""Fixture adapter for VulkanStereo; no standalone hardware entry point.

Caller must establish a reviewed device/lease/provider admission and control
engine lifetimes. Importing or calling with a fake engine is CPU-only.
"""
from time import perf_counter
import numpy as np

OUTPUTS = ('left_census', 'right_census', 'left_census_valid', 'right_census_valid',
           'cost', 'aggregate', 'disparity', 'confidence', 'valid', 'raw_validity')


def capture(engine, left, right, *, lower=None, upper=None):
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray) or left.ndim != 2 or left.shape != right.shape or min(left.shape) < 5:
        raise ValueError('matching HxW image arrays of size at least 5 are required')
    if (lower is None) != (upper is None):
        raise ValueError('both disparity bounds required')
    start = perf_counter()
    token = engine.enqueue(left, right, lower=lower, upper=upper)
    outputs = engine.result(token, debug=True, debug_census=True)
    elapsed = (perf_counter() - start) * 1000
    if set(outputs) != set(OUTPUTS):
        raise ValueError('incomplete stage outputs')
    timings = engine.last_timings
    return {'outputs': outputs, 'pipeline_ms': timings['gpu_pipeline_ms'],
            'gpu_stages_ms': dict(timings['gpu_stages_ms']),
            'readback_ms': timings['readback_ms'], 'transfer_inclusive_ms': elapsed}
