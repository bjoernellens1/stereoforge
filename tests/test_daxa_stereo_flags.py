"""Execute the real native flag cross-product, including derived-right selectors."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.gpu
@pytest.mark.parametrize('wave', [32, 64])
def test_descriptor_does_not_enable_disabled_subpixel(wave):
    binary = ROOT / 'build/ros-daxa-native/daxa_stereo_flags_contracts'
    assert binary.is_file(), f'Missing native GPU flag contract: {binary}'
    sdk = ROOT / 'ros_ws/.pixi/envs/default'
    env = dict(os.environ)
    env.update(VK_DRIVER_FILES='/usr/share/vulkan/icd.d/radeon_icd.x86_64.json',
               VK_LAYER_PATH=str(sdk / 'share/vulkan/explicit_layer.d'),
               LD_PRELOAD='/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1',
               LD_LIBRARY_PATH=str(sdk / 'lib'))
    subprocess.run([str(binary), str(wave)], env=env, check=True)
