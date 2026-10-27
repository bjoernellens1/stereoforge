"""Native CLI rejects invalid configuration before constructing a Vulkan device."""
import os
from pathlib import Path
import subprocess
import pytest


@pytest.fixture
def executable():
    root=Path(__file__).resolve().parents[1]
    configured=os.environ.get('SF_NATIVE_VIEW_BENCHMARK')
    binary=Path(configured) if configured else root/'build/standalone/native_view_benchmark'
    if configured:
        assert binary.is_file(),f'Explicitly configured native benchmark missing: {binary}'
    elif not binary.is_file():
        pytest.skip('Standalone native_view_benchmark has not been built')
    return binary


@pytest.mark.parametrize('arguments,message',[
    (['--frames','0'],'frames must be positive'),
    (['--mode','unknown'],'mode must be strict or preview'),
    (['--readback','other'],'readback must be gpu or cpu'),
    (['--surprise'],'unknown argument'),
])
def test_invalid_cli_before_gpu(executable,arguments,message):
    result=subprocess.run([str(executable),*arguments],capture_output=True,text=True)
    assert result.returncode!=0 and message in result.stderr
    assert 'create geometry Vulkan device' not in result.stderr


def test_invalid_header_before_gpu(executable,tmp_path):
    (tmp_path/'header.txt').write_text('672 376 672')
    result=subprocess.run([str(executable),'--fixture',str(tmp_path),'--output',str(tmp_path/'result.json')],capture_output=True,text=True)
    assert result.returncode!=0 and 'invalid fixture header' in result.stderr
    assert not (tmp_path/'result.json').exists()


def test_array_size_checked_before_gpu(executable,tmp_path):
    pose='1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1'
    (tmp_path/'header.txt').write_text(f'1 1 1 1 4 4 0 0 1.57 {pose} {pose} 1 synthetic\n')
    (tmp_path/'source_depth.f32').write_bytes(b'bad')
    result=subprocess.run([str(executable),'--fixture',str(tmp_path),'--output',str(tmp_path/'result.json')],capture_output=True,text=True)
    assert result.returncode!=0 and 'fixture array byte count mismatch' in result.stderr
    assert not (tmp_path/'result.json').exists()
