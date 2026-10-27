"""Source dependencies and real GPU differential qualification for the opt in."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import pytest
from stereoforge._reduction_source import reduction_source
import stereoforge

SHADER = Path(stereoforge.__file__).parent / 'shaders/stereo/sgm.slang'


def test_default_has_no_optional_dependency(monkeypatch):
    import builtins
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == 'sintergpu':
            raise AssertionError('default imported optional dependency')
        return original(name, *args, **kwargs)
    monkeypatch.setenv('SINTERGPU_SHARED_REDUCTION', '0')
    monkeypatch.setattr(builtins, '__import__', guard)
    source, helper, identity = reduction_source(SHADER)
    assert source == SHADER.read_bytes() and helper == b''
    assert not identity['shared_reduction']


def test_actual_dependency_bytes_and_cache_identity(monkeypatch, tmp_path):
    module_path = pytest.importorskip("sintergpu").module_path
    monkeypatch.setenv('SINTERGPU_SHARED_REDUCTION', '1')
    source, helper, identity = reduction_source(SHADER)
    assert helper == module_path('common.block_reduce').read_bytes()
    assert source == b'#define SINTERGPU_USE_SHARED_REDUCTION 1\n' + helper + b'\n' + SHADER.read_bytes()
    assert identity['source_sha256'] == sha256(source).hexdigest()
    assert identity['shared_reduction_sha256'] == sha256(helper).hexdigest()
    class Changed:
        def read_bytes(self): return helper + b'\n// changed dependency\n'
    monkeypatch.setattr('sintergpu.module_path', lambda _: Changed())
    assert reduction_source(SHADER)[2]['source_sha256'] != identity['source_sha256']
    monkeypatch.setenv('SINTERGPU_SHARED_REDUCTION', '0')
    assert reduction_source(SHADER)[2]['source_sha256'] != identity['source_sha256']

@pytest.mark.gpu
@pytest.mark.parametrize('D', [1,2,3,31,32,64,128,256])
@pytest.mark.parametrize('paths', [4,8])
def test_shared_sgm_exact_output_bytes(monkeypatch, D, paths):
    pytest.importorskip('sintergpu')
    pytest.importorskip('slangpy')
    from stereoforge.gpu import VulkanStereo
    rng = np.random.default_rng(42)
    left = rng.integers(0,256,(17,33),dtype=np.uint8)
    right = np.roll(left,-2,axis=1)
    results=[]
    for flag in ('0','1'):
        monkeypatch.setenv('SINTERGPU_SHARED_REDUCTION',flag)
        engine=VulkanStereo(max_disparity=D,paths=paths)
        result=engine.debug_stages(left,right)
        # Read raw uint32 validity before the public bool conversion.
        result['validity']=engine.buffers['validity'].to_numpy().copy()
        results.append(result)
        engine.device.wait_for_idle()
    for name in results[0]:
        assert results[0][name].tobytes() == results[1][name].tobytes(), name
