from pathlib import Path
import subprocess
import numpy as np
import pytest
from stereoforge.gpu import VulkanStereo

@pytest.mark.gpu
@pytest.mark.parametrize('paths',[4,8])
def test_cpp_python_share_shader_results(tmp_path,paths):
    executable=Path('build/vulkan/native_stereo')
    if not executable.exists():pytest.skip('native Vulkan target not built; run cmake --preset vulkan')
    rng=np.random.default_rng(19)
    left=rng.integers(0,256,(16,40),dtype=np.uint32)
    right=np.zeros_like(left);right[:,:-4]=left[:,4:]
    left.tofile(tmp_path/'left');right.tofile(tmp_path/'right')
    prefix=tmp_path/'result'
    subprocess.run([str(executable),'--width','40','--height','16','--disparities','16','--paths',str(paths),'--left',str(tmp_path/'left'),'--right',str(tmp_path/'right'),'--output',str(prefix)],check=True,capture_output=True,text=True)
    gpu=VulkanStereo(16,paths=paths)
    expected=gpu.result(gpu.enqueue(left,right))
    np.testing.assert_allclose(np.fromfile(str(prefix)+'.disparity.f32',np.float32).reshape(16,40),expected['disparity'],atol=1e-5,equal_nan=True)
    np.testing.assert_array_equal(np.fromfile(str(prefix)+'.validity.u32',np.uint32).reshape(16,40).astype(bool),expected['valid'])
