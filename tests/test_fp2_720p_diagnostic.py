"""CPU-only preflight and failure cases for isolated 720p diagnostic."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from benchmarks.fp2_720p_diagnostic import FILES, invoke_host, verify_binary, verify_prepared
from benchmarks.fp2_720p_prepare import sha

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT/'build/daxa/fp2_720p_fixture'


def test_prepared_manifest_and_file_mutations_fail(tmp_path):
    source = tmp_path/'source'; source.write_bytes(b'source')
    eyes = {}
    for eye in ('0', '1'):
        directory = tmp_path/('eye'+eye); directory.mkdir()
        files = {}
        for name in FILES:
            path = directory/name; path.write_bytes((eye+name).encode())
            files[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        eyes[eye] = {'files': files}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'eyes': eyes, 'sources': {str(source): sha(source)}}))
    expected = sha(path)
    verify_prepared(tmp_path, expected)
    (tmp_path/'eye0'/'paths').write_bytes(b'changed')
    with pytest.raises(ValueError, match='prepared file mismatch'):
        verify_prepared(tmp_path, expected)
    with pytest.raises(ValueError, match='manifest hash'):
        verify_prepared(tmp_path, '0'*64)


def test_child_failure_propagates(tmp_path):
    bad = tmp_path/'fail'; bad.write_text('#!/bin/sh\nexit 7\n'); bad.chmod(0o700)
    with pytest.raises(subprocess.CalledProcessError) as error:
        invoke_host(bad, tmp_path, '--check', 0)
    assert error.value.returncode == 7


@pytest.mark.skipif(not BINARY.exists(), reason='optional native 720p host not built')
def test_compiled_identity_and_small_host_bounds(tmp_path):
    verify_binary(BINARY)
    d = tmp_path/'eye0'; d.mkdir(); h, w = 2, 3; n = h*w
    np.zeros(n, dtype='<u4').tofile(d/'lo')
    np.ones(n, dtype='<u4').tofile(d/'hi')
    np.arange(n+1, dtype='<u8').__mul__(16).tofile(d/'offset')
    for name in ('own_luma','other_luma','own_rect_support','other_rect_support','own','other','own_support','other_support','winner','best','second','count'):
        np.zeros(n, dtype='<u4').tofile(d/name)
    np.zeros(n*16*4, dtype='<u4').tofile(d/'paths')
    np.zeros(n*16, dtype='<u4').tofile(d/'sums')
    subprocess.run([str(BINARY), '--check', str(d), '0', str(w), str(h)], check=True)
    np.zeros(n, dtype='<u8').tofile(d/'offset')
    assert subprocess.run([str(BINARY), '--check', str(d), '0', str(w), str(h)], capture_output=True).returncode != 0
