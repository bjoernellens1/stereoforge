"""Setup-only native depth CLI regressions with Vulkan disabled."""
import os
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'build/daxa/daxa_depth_benchmark'


@pytest.mark.parametrize('arguments,reason',[
    (['--unknown'],'unknown depth option'),
    (['--manifest'],'missing option value'),
    (['--manifest','a','--manifest','b'],'duplicate option'),
    (['--manifest','a','--output','b','--frames','0'],'bounded frames'),
    (['--manifest','a','--output','b','--warmup','31'],'bounded frames'),
    (['--manifest','a','--output','b','--frames','true'],'invalid unsigned option'),
])
def test_invalid_depth_cli_stops_before_gpu(arguments,reason):
    if not BINARY.exists():pytest.skip('optional native depth target not built')
    run=subprocess.run([str(BINARY),*arguments],text=True,capture_output=True,env=dict(os.environ,VK_ICD_FILENAMES='/nonexistent/disabled-vulkan.json'))
    assert run.returncode==1 and reason in run.stderr
    assert 'Vulkan' not in run.stderr and not run.stdout


def test_shared_depth_no_unqualified_float_flags_or_echoed_source():
    source=(ROOT/'apps/daxa_depth_benchmark.cpp').read_text()
    assert 'auto final_source=output.source()' in source
    assert 'metadata(*final_source)' in source
    assert '"metric_compound_float_reference_checked",false' in source
    assert '"stereo_compound_float_reference_checked",false' in source
    assert '"both_stereo_integer_reference_checked",checked' in source
    assert '"both_stereo_packed_penalties_reference_checked",checked' in source
