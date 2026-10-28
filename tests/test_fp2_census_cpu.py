"""CPU-only identity, scalar contract and independent Census parity tests."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'build/daxa/fp2_census_fixture'


def verify(identity, paths):
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity[key]:
            raise ValueError('identity mismatch ' + key)


def test_identity_mutations_rejected(tmp_path):
    keys = ('host_sha256', 'census_shader_sha256', 'census_helper_sha256', 'census_spirv_sha256',
            'cost_shader_sha256', 'cost_spirv_sha256', 'recurrence_host_sha256',
            'recurrence_shader_sha256', 'recurrence_spirv_sha256', 'compiler_sha256')
    paths = {key: tmp_path / key for key in keys}
    for key, path in paths.items():
        path.write_bytes(key.encode())
    identity = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    verify(identity, paths)
    for key, path in paths.items():
        old = path.read_bytes()
        path.write_bytes(old + b'mutated')
        with pytest.raises(ValueError, match=key):
            verify(identity, paths)
        path.write_bytes(old)


@pytest.mark.skipif(not BINARY.exists(), reason='optional Census fixture not built')
def test_identity_cpu_contract_and_python_reference():
    from stereoforge.refinement import census
    identity = json.loads(subprocess.check_output([str(BINARY), '--identity'], text=True))
    verify(identity, {
        'host_sha256': ROOT / 'apps/fp2_census_fixture.cpp',
        'census_shader_sha256': ROOT / 'python/stereoforge/shaders/daxa/fp2_census.slang',
        'census_helper_sha256': ROOT / 'python/stereoforge/shaders/ops/refinement.slang',
        'census_spirv_sha256': Path(identity['census_spirv_path']),
        'cost_shader_sha256': ROOT / 'python/stereoforge/shaders/daxa/fp2_image_cost.slang',
        'cost_spirv_sha256': Path(identity['cost_spirv_path']),
        'recurrence_host_sha256': ROOT / 'apps/fp2_prototype_fixture.cpp',
        'recurrence_shader_sha256': ROOT / 'python/stereoforge/shaders/daxa/fp2_banded.slang',
        'recurrence_spirv_sha256': Path(identity['recurrence_spirv_path']),
        'compiler_sha256': Path(identity['compiler_path']),
    })
    assert 'contracts passed' in subprocess.check_output([str(BINARY), '--cpu-only'], text=True)
    a = np.array(json.loads(subprocess.check_output([str(BINARY), '--cpu-sample'], text=True)), dtype=np.uint32)
    result = census(a[:, 0].reshape(5, 7), a[:, 1].reshape(5, 7))
    np.testing.assert_array_equal(a[:, 2].reshape(5, 7), result.descriptor)
    np.testing.assert_array_equal(a[:, 3].reshape(5, 7), result.support)
