"""Execute the real wide-range native contract, including beyond-D256 matches."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'build/ros-daxa-native/daxa_stereo_wide_contracts'


def test_native_d288_independent_scalar_and_packed_contract():
    result = subprocess.run([str(BINARY), '--cpu-only'], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
@pytest.mark.parametrize('wave', [32, 64])
def test_native_d288_both_eye_gpu_integer_and_subpixel_contract(wave):
    result = subprocess.run([str(BINARY), '--gpu', str(wave)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
