"""CPU fake-device tests for the real VulkanStereo fixture adapter."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('fixture_execution', Path(__file__).resolve().parents[1]/'benchmarks/sintergpu_fixture_execution.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class Engine:
    D=8
    def __init__(self): self.events=[]; self.last_timings={}
    def enqueue(self, left, right, lower=None, upper=None):
        self.events.append(('enqueue', left, right, lower, upper)); return 42
    def result(self, token, debug=False, debug_census=False):
        self.events.append(('result', token, debug, debug_census))
        self.last_timings={'gpu_pipeline_ms':2.0, 'gpu_stages_ms':{'aggregate':1.0}, 'readback_ms':0.5}
        return {key:np.array([i],dtype=np.uint32) for i,key in enumerate(runner.OUTPUTS)}


def test_actual_engine_adapter_enqueues_and_reads_all_outputs():
    e=Engine(); a=np.ones((5,6),dtype=np.uint8); b=a.copy()
    result=runner.capture(e,a,b)
    assert e.events[0][0]=='enqueue' and e.events[1]==('result',42,True,True)
    assert set(result['outputs'])==set(runner.OUTPUTS)
    assert result['pipeline_ms']==2.0 and result['gpu_stages_ms']=={'aggregate':1.0}
    assert result['transfer_inclusive_ms']>0 and result['readback_ms']==0.5


def test_invalid_fixture_rejected_before_enqueue():
    e=Engine(); a=np.ones((5,6),dtype=np.uint8)
    with pytest.raises(ValueError): runner.capture(e,a,a,lower=np.zeros_like(a))
    assert not e.events


def test_collector_runs_adapter_in_baseline_candidate_baseline_order():
    spec2=importlib.util.spec_from_file_location('collector', Path(__file__).resolve().parents[1]/'benchmarks/sintergpu_qualification_runner.py')
    collector=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(collector)
    base=Engine(); candidate=Engine(); engines={'baseline':base,'candidate':candidate}
    a=np.ones((5,6),dtype=np.uint8)
    result=collector.collect(lambda variant,case:runner.capture(engines[variant],a,a), ['fixture'], rounds=3)
    assert len(base.events)==12 and len(candidate.events)==6
    assert result['status']=='unqualified'
    assert result['cases']['fixture'][0][1]['timings']['readback_ms']==0.5
