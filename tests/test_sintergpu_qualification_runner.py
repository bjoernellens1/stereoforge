"""CPU-only tests of qualification collection; no SlangPy device is created."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'benchmarks/sintergpu_qualification_runner.py'
spec = importlib.util.spec_from_file_location('sintergpu_qualification_runner', PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def record(value=1, elapsed=1.0):
    return {'outputs': {key: np.array([value], dtype=np.uint32) for key in runner.OUTPUTS},
            'pipeline_ms': elapsed, 'transfer_inclusive_ms': elapsed + 1, 'readback_ms': elapsed,
            'gpu_stages_ms': {'aggregate': elapsed}}


def test_old_new_old_collection_and_exact_outputs():
    seen = []
    def run(variant, case):
        seen.append((variant, case)); return record()
    result = runner.collect(run, ['odd', 'bounded'], rounds=3)
    assert seen == [(v, case) for case in ['odd', 'bounded'] for _ in range(3)
                    for v in ('baseline', 'candidate', 'baseline')]
    assert len(result['cases']['odd']) == 3
    assert result['status'] == 'unqualified'
    assert len(result['cases']['odd'][0][1]['outputs']['aggregate']['sha256']) == 64
    assert result['cases']['odd'][0][1]['timings']['pipeline_ms'] == 1.0


def test_mismatch_fails_even_if_final_output_matches():
    def run(variant, case):
        r = record()
        if variant == 'candidate': r['outputs']['aggregate'][0] = 2
        return r
    with pytest.raises(ValueError, match='aggregate'):
        runner.collect(run, ['odd'], rounds=3)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0, -1])
def test_bad_timing_rejected(bad):
    with pytest.raises(ValueError, match='timing'):
        runner.collect(lambda *_: record(elapsed=bad), ['odd'], rounds=3)


def test_missing_output_and_insufficient_runs_are_rejected():
    r = record(); del r['outputs']['cost']
    with pytest.raises(ValueError, match='outputs'):
        runner.collect(lambda *_: r, ['odd'], rounds=3)
    with pytest.raises(ValueError, match='rounds'):
        runner.collect(lambda *_: record(), ['odd'], rounds=2)


def test_cli_fails_closed_before_device_creation(capsys):
    assert runner.main([]) != 0
    assert 'physical device' in capsys.readouterr().err


@pytest.mark.parametrize('key', ['right_census', 'left_census_valid', 'right_census_valid'])
def test_census_input_mismatch_saves_witness(tmp_path, key):
    def run(variant, case):
        r = record()
        if variant == 'candidate': r['outputs'][key][0] = 9
        return r
    with pytest.raises(ValueError, match=key):
        runner.collect(run, ['odd'], rounds=3, witness_dir=tmp_path)
    files = list(tmp_path.glob('mismatch-*.npz'))
    assert len(files) == 1
    with np.load(files[0], allow_pickle=False) as data:
        assert data['case'].item() == 'odd'
        assert data['output'].item() == key
        assert data['comparison'].item() == 'candidate'
        assert data['baseline'][0] == 1 and data['other'][0] == 9


def test_missing_right_census_rejected():
    r = record(); del r['outputs']['right_census']
    with pytest.raises(ValueError, match='outputs'):
        runner.collect(lambda *_: r, ['odd'], rounds=3)


def test_raw_validity_change_is_detected_even_when_boolean_matches(tmp_path):
    def run(variant, case):
        r = record()
        r['outputs']['valid'] = np.array([True], dtype=np.bool_)
        r['outputs']['raw_validity'] = np.array([2 if variant == 'candidate' else 1], dtype=np.uint32)
        return r
    with pytest.raises(ValueError, match='raw_validity'):
        runner.collect(run, ['validity'], rounds=3, witness_dir=tmp_path)
    with np.load(next(tmp_path.glob('mismatch-*.npz')), allow_pickle=False) as data:
        assert data['baseline'][0] == 1 and data['other'][0] == 2
