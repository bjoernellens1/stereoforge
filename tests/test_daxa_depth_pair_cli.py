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


@pytest.mark.parametrize('mode', ['sequential', 'paired-per-path', 'layered-per-path', 'shared-census-sequential'])
def test_each_explicit_schedule_reaches_fixture_preflight_without_device(tmp_path, mode):
    result = invoke(tmp_path, '--stereo-schedule', mode)
    assert result.returncode == 1 and 'missing fixture file' in result.stderr
    assert 'unknown depth stereo schedule' not in result.stderr
    assert not (tmp_path / 'report.json').exists()


def test_compiled_shared_and_layered_source_hash_getter_is_cpu_only(tmp_path):
    import hashlib
    import json
    if not BINARY.is_file():
        pytest.skip('offline depth benchmark not built')
    env = os.environ.copy()
    env['VK_DRIVER_FILES'] = env['VK_ICD_FILENAMES'] = str(tmp_path / 'absent-icd.json')
    result = subprocess.run([str(BINARY), '--shader-source-identities'], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    source = ROOT / 'python/stereoforge/shaders'
    sha = lambda path: hashlib.sha256((source / path).read_bytes()).hexdigest()
    assert json.loads(result.stdout) == {
        'stereo_shader_source_sha256_pair': sha('daxa/stereo.slang') + ':' + sha('ops/sgm.slang'),
        'stereo_shared_shader_source_sha256_pair': sha('daxa/stereo_parameters.slang') + ':' + sha('ops/sgm_specialized.slang'),
        'stereo_layered_shader_source_sha256': sha('daxa/stereo_layered.slang'),
    }

@pytest.mark.parametrize('mode', ['', 'derived', 'IndependentSGM', 'unknown'])
def test_unknown_right_mode_rejected_before_fixture(tmp_path, mode):
    result = invoke(tmp_path, '--right-reference-mode', mode)
    assert result.returncode == 1 and 'unknown Right reference mode' in result.stderr

@pytest.mark.parametrize('schedule', ['sequential', 'paired-per-path', 'layered-per-path'])
def test_derived_requires_shared_schedule_before_fixture(tmp_path, schedule):
    result = invoke(tmp_path, '--right-reference-mode', 'left-aggregated-selection',
                    '--stereo-schedule', schedule)
    assert result.returncode == 1 and 'derived Right requires shared-census-sequential' in result.stderr

@pytest.mark.parametrize('mode', ['independent-sgm', 'left-aggregated-selection'])
def test_right_mode_reaches_fixture_preflight(tmp_path, mode):
    result = invoke(tmp_path, '--right-reference-mode', mode,
                    '--stereo-schedule', 'shared-census-sequential')
    assert result.returncode == 1 and 'missing fixture file' in result.stderr
