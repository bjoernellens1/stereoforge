"""CPU-tested measurement collector; hardware entry point deliberately blocked.

This module does not initialize a GPU. A reviewed device-bound executor is a
prerequisite for connecting this collector to actual StereoForge workloads.
"""
import hashlib
import math
import os
import sys
import tempfile

import numpy as np

OUTPUTS = ('left_census', 'right_census', 'left_census_valid', 'right_census_valid',
           'cost', 'aggregate', 'disparity', 'confidence', 'valid')


def _snapshot(record):
    outputs = record.get('outputs')
    if not isinstance(outputs, dict) or set(outputs) != set(OUTPUTS):
        raise ValueError('outputs must contain every required stage')
    snap = {}
    for name in OUTPUTS:
        value = outputs[name]
        if not isinstance(value, np.ndarray) or value.dtype.hasobject:
            raise ValueError(f'output {name} must be a numeric array')
        snap[name] = np.ascontiguousarray(value).copy()
    timings = {}
    for name in ('pipeline_ms', 'transfer_inclusive_ms'):
        value = record.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid timing: {name}')
        timings[name] = float(value)
    stages = record.get('gpu_stages_ms')
    if not isinstance(stages, dict) or not stages:
        raise ValueError('invalid timing: gpu_stages_ms')
    timings['gpu_stages_ms'] = {}
    for name, value in stages.items():
        if not isinstance(name, str) or not name or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid timing: gpu stage')
        timings['gpu_stages_ms'][name] = float(value)
    return snap, timings


def _equal(a, b, *, witness_dir, case, round_index, comparison):
    for name in OUTPUTS:
        x, y = a[name], b[name]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            if witness_dir is not None:
                # Exclusive creation avoids overwriting previous failure evidence.
                with tempfile.NamedTemporaryFile(prefix='mismatch-', suffix='.npz',
                                                 dir=witness_dir, delete=False) as f:
                    np.savez(f, case=case, round_index=round_index, output=name,
                             comparison=comparison, baseline=x, other=y)
                    f.flush()
                    os.fsync(f.fileno())
            raise ValueError(f'output mismatch: {name}')


def collect(run, cases, *, rounds, witness_dir=None):
    """Collect exact stage parity and raw baseline/candidate/baseline timings.

    ``run`` is deliberately injected for CPU mocks. Hardware use requires an
    independently reviewed executor and evidence of device and source identity.
    Returned data is not a qualification verdict.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 3:
        raise ValueError('at least three rounds required')
    cases = list(cases)
    if not cases or any(not isinstance(c, str) or not c for c in cases) or len(set(cases)) != len(cases):
        raise ValueError('unique named cases required')
    result = {'status': 'unqualified', 'cases': {}}
    for case in cases:
        batches = []
        for round_index in range(rounds):
            measurements = []
            for variant in ('baseline', 'candidate', 'baseline'):
                outputs, timings = _snapshot(run(variant, case))
                measurements.append((outputs, timings))
            _equal(measurements[0][0], measurements[1][0], witness_dir=witness_dir, case=case, round_index=round_index, comparison='candidate')
            _equal(measurements[0][0], measurements[2][0], witness_dir=witness_dir, case=case, round_index=round_index, comparison='baseline_after')
            if any(set(m[1]['gpu_stages_ms']) != set(measurements[0][1]['gpu_stages_ms']) for m in measurements[1:]):
                raise ValueError('timing stage mismatch')
            batches.append([{'timings': m[1], 'outputs': {name: {
                'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                'dtype': value.dtype.str, 'shape': list(value.shape)}
                for name, value in m[0].items()}} for m in measurements])
        result['cases'][case] = batches
    return result


def main(argv=None):
    print('Blocked: reviewed physical device selection, lock, identity and teardown executor required; no GPU initialized.', file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
