"""CPU contracts and artifact identities for isolated FP2 selection fixture."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT/'build/daxa/fp2_selection_fixture'


def verify(identity, paths):
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity[key]:
            raise ValueError('identity mismatch '+key)


def test_every_identity_mutation_rejected(tmp_path):
    keys = ('host_sha256', 'lut_host_sha256', 'included_host_sha256', 'selection_shader_sha256', 'recurrence_shader_sha256',
            'recurrence_spirv_sha256', 'reset_spirv_sha256', 'select_spirv_sha256', 'lr_spirv_sha256',
            'metric_spirv_sha256', 'compiler_sha256')
    paths = {key: tmp_path/key for key in keys}
    for key, path in paths.items():
        path.write_bytes(key.encode())
    expected = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    verify(expected, paths)
    for key, path in paths.items():
        old = path.read_bytes(); path.write_bytes(old+b'changed')
        with pytest.raises(ValueError, match=key):
            verify(expected, paths)
        path.write_bytes(old)


@pytest.mark.skipif(not BINARY.exists(), reason='optional selection fixture not built')
def test_compiled_identity_and_scalar_contracts():
    identity = json.loads(subprocess.check_output([str(BINARY), '--identity'], text=True))
    shader_dir = Path(identity['shader_dir'])
    verify(identity, {
        'host_sha256': ROOT/'apps/fp2_selection_fixture.cpp',
        'lut_host_sha256': ROOT/'apps/fp2_metric_lut.hpp',
        'included_host_sha256': ROOT/'apps/fp2_prototype_fixture.cpp',
        'selection_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_selection.slang',
        'recurrence_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_banded.slang',
        'recurrence_spirv_sha256': Path(identity['recurrence_spirv_path']),
        'reset_spirv_sha256': shader_dir/'fp2-select-reset_workspace.spv',
        'select_spirv_sha256': shader_dir/'fp2-select-select_packed.spv',
        'lr_spirv_sha256': shader_dir/'fp2-select-consistency.spv',
        'metric_spirv_sha256': shader_dir/'fp2-select-metric.spv',
        'compiler_sha256': Path(identity['compiler_path']),
    })
    output = subprocess.check_output([str(BINARY), '--cpu-only'], text=True)
    assert 'cpu_fixture=7x5' in output and 'cpu_fixture=320x180' in output
    assert 'contracts passed' in output


@pytest.mark.skipif(not BINARY.exists(), reason='optional selection fixture not built')
def test_scalar_witnesses_against_independent_numpy_references():
    import numpy as np
    from benchmarks.compact_sgm_quality import select_eye
    from stereoforge.refinement import lr_consistency

    value = json.loads(subprocess.check_output([str(BINARY), '--cpu-json'], text=True))
    raw = np.array(value['raw'], dtype=np.int64).reshape(2, 1, 12, 7)
    obs = np.array(value['obs'], dtype=np.uint32).reshape(2, 1, 12, 9)
    lo = np.array(value['lo'], dtype=np.uint32).reshape(2, 1, 12)
    hi = np.array(value['hi'], dtype=np.uint32).reshape(2, 1, 12)
    support = np.array(value['support'], dtype=bool).reshape(2, 1, 12)
    disparities, validity = [], []
    for eye in (0, 1):
        data = {'winners': raw[eye, :, :, 0].astype(np.int32)}
        for name, col in [('best', 1), ('second', 2), ('count', 3)]:
            data[name] = raw[eye, :, :, col].astype(np.uint32)
        disparity, confidence, valid = select_eye(data, lo[eye], hi[eye], 128)
        np.testing.assert_array_equal(raw[eye, :, :, 4].astype(np.uint32), disparity.view(np.uint32))
        np.testing.assert_array_equal(raw[eye, :, :, 5].astype(np.uint32), confidence.view(np.uint32))
        np.testing.assert_array_equal(raw[eye, :, :, 6], valid)
        disparities.append(disparity); validity.append(valid)
    disparity, valid = np.stack(disparities), np.stack(validity)
    lr = lr_consistency(disparity, valid, 1.)
    np.testing.assert_array_equal(obs[..., 0], lr)
    with np.errstate(invalid='ignore', divide='ignore'):
        denominator = disparity - np.float32(.25)
        depth = np.float32(63.125) / denominator
    metric = lr & support & (disparity > 0) & (denominator > 0) & np.isfinite(depth) & (depth > 0)
    expected_depth = np.where(metric, depth, np.float32(np.nan)).astype(np.float32)
    np.testing.assert_array_equal(obs[..., 1], expected_depth.view(np.uint32))
    np.testing.assert_array_equal(obs[..., 2], np.where(metric, raw[..., 5], 0).astype(np.uint32))
    np.testing.assert_array_equal(obs[..., 3], metric)
    np.testing.assert_array_equal(obs[..., 4], metric.astype(np.uint32))
    np.testing.assert_array_equal(obs[..., 5], np.where(metric, 0xabcdef01, 0xffffffff).astype(np.uint32))
    np.testing.assert_array_equal(obs[..., 6], np.where(metric, 0x12345678, 0xffffffff).astype(np.uint32))
    assert np.all(obs[..., 7:] == 0)
    assert np.all(metric.sum(axis=(1, 2)) > 0)
    assert not valid[0, 0, 0] and not valid[0, 0, 1]
    assert not np.any(valid[0, 0, 7:11])
    assert valid[0, 0, 2] and not lr[0, 0, 2]
    assert lr[0, 0, 5] and not metric[0, 0, 5]  # difference exactly one
    assert valid[0, 0, 6] and not lr[0, 0, 6]  # difference two


@pytest.mark.skipif(not BINARY.exists(), reason='optional selection fixture not built')
def test_confidence_all_bounded_pairs_exact_rational_nearest_even():
    from fractions import Fraction
    import numpy as np

    bits = json.loads(subprocess.check_output([str(BINARY), '--confidence-all'], text=True))
    assert len(bits) == 224*225//2
    index = 0
    for second in range(1, 225):
        for best in range(second):
            rational = Fraction(second-best, second)
            expected = np.float32(np.float32(second-best) / np.float32(second))
            actual_bits = int(bits[index]); index += 1
            assert actual_bits == int(expected.view(np.uint32)), (best, second)
            actual = np.array(actual_bits, dtype=np.uint32).view(np.float32)
            exact = Fraction.from_float(float(actual))
            lower = np.nextafter(actual, np.float32(-np.inf))
            upper = np.nextafter(actual, np.float32(np.inf))
            distance = abs(exact-rational)
            for neighbor in (lower, upper):
                candidate_distance = abs(Fraction.from_float(float(neighbor))-rational)
                assert distance <= candidate_distance, (best, second)
                if distance == candidate_distance:
                    assert not (actual_bits & 1), (best, second)


@pytest.mark.skipif(not BINARY.exists(), reason='optional selection fixture not built')
@pytest.mark.parametrize('best,second', [(0, 0), (1, 1), (2, 1), (0, 225), (224, 225), (0, 4294967296)])
def test_confidence_outside_contract_fails(best, second):
    result = subprocess.run([str(BINARY), '--confidence-check', str(best), str(second)], capture_output=True)
    assert result.returncode != 0
