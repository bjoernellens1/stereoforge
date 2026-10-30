import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("gpu_sharing_summary",ROOT/"benchmarks/gpu_sharing_summary.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def row(mode, ident, latency, units=None, elapsed=None):
    value={"mode":mode,"frame_identity":ident,"latency_ms":latency}
    if units is not None:
        value.update(background_work_units=units,background_elapsed_ms=elapsed)
    return value

def test_summarizes_actual_samples_and_retention():
    rows=[row("background_only",1,0,100,1000),row("background_only",2,0,100,1000),
          row("stereo_60",1,8,70,1000),row("stereo_60",2,10,70,1000)]
    result=mod.compare(mod.summarize(rows))
    assert result["stereo_60"]["latency_p50_ms"]==9
    assert result["stereo_60"]["latency_p99_ms"]==9.98
    assert result["stereo_60"]["background_retention"]==pytest.approx(.7)
    assert result["stereo_60"]["samples"]==2

@pytest.mark.parametrize("records",[
    [row("x",1,-1)],[row("x",1,float("nan"))],
    [row("x",1,1),row("x",1,2)],
    [row("x",True,1)],[row("x",1,1,2,None)],
    [row("x",1,1,-1,100)],[row("x",1,1,2,0)],
    [{"mode":"x","frame_identity":1}],
])
def test_bad_records_rejected(records):
    with pytest.raises(ValueError):mod.summarize(records)

def test_missing_background_measurements_do_not_imply_zero():
    result=mod.compare(mod.summarize([row("stereo",1,8.1)]))
    assert result["stereo"]["background_retention"] is None
    assert result["stereo"]["background_work_units_per_s"] is None
