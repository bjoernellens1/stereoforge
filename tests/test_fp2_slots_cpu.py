"""CPU-only identity and contract checks for the isolated two-slot fixture."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'build/daxa/fp2_slots_fixture'


def verify(identity, paths):
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity[key]:
            raise ValueError('identity mismatch ' + key)


def test_identity_mutations_fail(tmp_path):
    paths = {}
    for key in ('slots_source_sha256', 'tiny_source_sha256', 'shader_source_sha256', 'spirv_sha256', 'compiler_sha256'):
        path = tmp_path / key
        path.write_bytes(key.encode())
        paths[key] = path
    identity = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    verify(identity, paths)
    for key, path in paths.items():
        original = path.read_bytes()
        path.write_bytes(original + b'mutation')
        with pytest.raises(ValueError, match=key):
            verify(identity, paths)
        path.write_bytes(original)


@pytest.mark.skipif(not BINARY.exists(), reason='optional two-slot fixture not built')
def test_compiled_identity_and_cpu_contracts():
    identity = json.loads(subprocess.check_output([str(BINARY), '--identity'], text=True))
    verify(identity, {
        'slots_source_sha256': ROOT / 'apps/fp2_slots_fixture.cpp',
        'tiny_source_sha256': ROOT / 'apps/fp2_prototype_fixture.cpp',
        'shader_source_sha256': ROOT / 'python/stereoforge/shaders/daxa/fp2_banded.slang',
        'spirv_sha256': Path(identity['spirv_path']),
        'compiler_sha256': Path(identity['compiler_path']),
    })
    assert 'contracts passed' in subprocess.check_output([str(BINARY), '--cpu-only'], text=True)
