"""Paired schedule option preflight, with every Vulkan ICD disabled."""
from pathlib import Path
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / 'build/daxa/daxa_depth_benchmark'


def invoke(tmp_path, *options):
    if not BINARY.is_file():
        pytest.skip('offline depth benchmark not built')
    env = os.environ.copy()
    env['VK_DRIVER_FILES'] = env['VK_ICD_FILENAMES'] = str(tmp_path / 'absent-icd.json')
    return subprocess.run([str(BINARY), '--manifest', str(tmp_path / 'missing.json'),
                           '--output', str(tmp_path / 'report.json'), *options],
                          env=env, capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize('mode', ['', 'paired', 'PairedPerPath', 'unknown'])
def test_unknown_schedule_rejected_before_fixture_or_device(tmp_path, mode):
    result = invoke(tmp_path, '--stereo-schedule', mode)
    assert result.returncode == 1
    assert 'unknown depth stereo schedule' in result.stderr
    assert not (tmp_path / 'report.json').exists()


def test_duplicate_schedule_cannot_silently_replace_profile(tmp_path):
    result = invoke(tmp_path, '--stereo-schedule', 'sequential',
                    '--stereo-schedule', 'paired-per-path')
    assert result.returncode == 1 and 'duplicate option' in result.stderr
    assert not (tmp_path / 'report.json').exists()


def test_schedule_requires_value(tmp_path):
    result = invoke(tmp_path, '--stereo-schedule')
    assert result.returncode == 1 and 'missing option value' in result.stderr
    assert not (tmp_path / 'report.json').exists()
