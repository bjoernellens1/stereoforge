"""CPU-only child diagnostics must veto numeric-only migration success."""
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
import compare_daxa_view as harness


@pytest.mark.parametrize('message,category', [
    ('Validation Error: [ VUID-vkCmdPipelineBarrier-dstStageMask-04091 ]', 'vuid'),
    ('ValidationError: invalid descriptor pool', 'validation_error'),
    ('Validation Warning: deprecated setting', 'warning'),
    ('WARNING: deprecated VK_LAYER_ENABLES', 'warning'),
    ('Daxa assertion failed: compact batches', 'daxa_assertion'),
    ('IMPOSSIBLE CASE! ALL BATCHES MUST BE COMPACT', 'daxa_assertion'),
    ('SYNC-HAZARD-WRITE-AFTER-WRITE', 'synchronization_hazard'),
])
def test_diagnostics_detected(message, category):
    detected = harness.parse_diagnostics(message)
    assert detected['line_count'] == 1
    assert category in detected['matches'][0]['categories']


def test_normal_output_is_clean():
    assert harness.parse_diagnostics('report.json\nGPU timestamp 2.3 ms\n')['line_count'] == 0


def test_zero_exit_and_numeric_parity_do_not_hide_validation_errors(tmp_path):
    child = harness.run_child([sys.executable, '-c',
        'import sys;print("report.json");print("Validation Error: VUID-test-broken", file=sys.stderr)'],
        None, tmp_path, 'strict-rhi')
    assert child['returncode'] == 0
    assert Path(child['stdout_file']).read_text() == 'report.json\n'
    assert 'VUID-test-broken' in Path(child['stderr_file']).read_text()
    assert harness.passed(True, {'strict-rhi': child}, validation_required=True) is False
    assert harness.passed(True, {'strict-rhi': child}, validation_required=False) is True


def test_child_crash_fails_even_without_validation(tmp_path):
    child = harness.run_child([sys.executable, '-c', 'raise SystemExit(3)'], None, tmp_path, 'crash')
    assert not harness.passed(True, {'crash': child}, validation_required=False)


def test_forced_validation_environment():
    assert harness.validation_requested(False, {'VK_INSTANCE_LAYERS': 'VK_LAYER_KHRONOS_validation'})
    assert harness.validation_requested(False, {'SF_DAXA_VALIDATION': '0'})
    assert harness.validation_requested(False, {'VK_LAYER_VALIDATE_SYNC': '1'})
    assert not harness.validation_requested(False, {})


def test_report_counters_and_missing_report_fail():
    clean = dict(returncode=0, stdout_diagnostics=harness.parse_diagnostics(''),
                 stderr_diagnostics=harness.parse_diagnostics(''))
    assert harness.passed(True, {'clean': clean}, True)
    assert not harness.passed(True, {'bad': dict(clean, reported_validation_errors=1)}, True)
    assert not harness.passed(True, {'bad': dict(clean, reported_validation_warnings=1)}, True)
    assert not harness.passed(True, {'bad': dict(clean, report_error='missing report')}, False)
