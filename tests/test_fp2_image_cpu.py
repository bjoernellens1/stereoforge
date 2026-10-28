"""CPU-only identity and reference checks; never invokes the GPU."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'build/daxa/fp2_image_fixture'


def verify(identity, paths):
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity[key]:
            raise ValueError('identity mismatch ' + key)


def test_all_identity_mutations_fail(tmp_path):
    names = ('cost_host_sha256', 'cost_shader_sha256', 'cost_spirv_sha256',
             'recurrence_host_sha256', 'slots_host_sha256', 'recurrence_shader_sha256',
             'recurrence_spirv_sha256', 'compiler_sha256')
    paths = {name: tmp_path / name for name in names}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    identity = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    verify(identity, paths)
    for name, path in paths.items():
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        with pytest.raises(ValueError, match=name):
            verify(identity, paths)
        path.write_bytes(original)


@pytest.mark.skipif(not BINARY.exists(), reason='optional image fixture not built')
def test_compiled_identity_and_cpu_oracle():
    identity = json.loads(subprocess.check_output([str(BINARY), '--identity'], text=True))
    verify(identity, {
        'cost_host_sha256': ROOT / 'apps/fp2_image_fixture.cpp',
        'cost_shader_sha256': ROOT / 'python/stereoforge/shaders/daxa/fp2_image_cost.slang',
        'cost_spirv_sha256': Path(identity['cost_spirv_path']),
        'recurrence_host_sha256': ROOT / 'apps/fp2_prototype_fixture.cpp',
        'slots_host_sha256': ROOT / 'apps/fp2_slots_fixture.cpp',
        'recurrence_shader_sha256': ROOT / 'python/stereoforge/shaders/daxa/fp2_banded.slang',
        'recurrence_spirv_sha256': Path(identity['recurrence_spirv_path']),
        'compiler_sha256': Path(identity['compiler_path']),
    })
    assert 'contracts passed' in subprocess.check_output([str(BINARY), '--cpu-only'], text=True)
