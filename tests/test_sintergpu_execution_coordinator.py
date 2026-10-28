"""CPU-only coordinator tests. No hardware provider is imported."""
from types import SimpleNamespace as NS
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
for name in ('sintergpu_device_receipt','sintergpu_qualification_runner','sintergpu_fixture_execution','sintergpu_execution_coordinator'):
    spec=importlib.util.spec_from_file_location(name,ROOT/'benchmarks'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
runner=sys.modules['sintergpu_execution_coordinator']
UUID=bytes(range(1,17)); KINDS=('instance','physical','device')


def setup(monkeypatch):
    handles=[NS(type=t,value=i+1) for i,t in enumerate(KINDS)]
    receipt=NS(schema_version=1,finalized=True,owner_generation=1,device_uuid_known=True,device_uuid=UUID,
        driver_version=7,instance=handles[0],physical_device=handles[1],device=handles[2],
        instance_external=False,physical_device_external=False,device_external=False)
    device=NS(vulkan_creation_receipt=receipt,native_handles=handles,wait_for_idle=lambda:None)
    made=[]
    class Engine:
        def __init__(self,*,device,shared_reduction,**kw):
            self.device=device;self.shared=shared_reduction;self.buffers={};self.kernels={};self.query_pool=object()
            self.shader_source_identity={'source_sha256':('b' if shared_reduction else 'a')*64,'shared_reduction':shared_reduction}
            made.append(self)
    def capture(engine,*a,**kw):
        return {'outputs':{name:np.array([1],dtype=np.uint32) for name in runner.OUTPUTS},
                'pipeline_ms':1.0,'gpu_stages_ms':{'aggregate':1.0},'readback_ms':1.0,'transfer_inclusive_ms':2.0}
    monkeypatch.setattr(runner,'VulkanStereo',Engine);monkeypatch.setattr(runner,'capture',capture)
    return device,made


def test_actual_coordinator_reuses_device_and_releases_engines(monkeypatch,tmp_path):
    d,made=setup(monkeypatch); a=np.ones((5,5),dtype=np.uint8)
    result=runner._execute_with_device(d,UUID,KINDS,7,{'a':(a,a,None,None)}, {},tmp_path,rounds=3,warmups=1)
    assert len(made)==2 and all(e.device is d and e.query_pool is None for e in made)
    assert [e.shared for e in made]==[False,True]
    assert result['status']=='unqualified' and len(result['cases']['a'])==3


def test_uuid_or_driver_mismatch_precedes_engine_creation(monkeypatch,tmp_path):
    d,made=setup(monkeypatch);d.vulkan_creation_receipt.driver_version=9
    a=np.ones((5,5),dtype=np.uint8)
    with pytest.raises(ValueError,match='driver'):
        runner._execute_with_device(d,UUID,KINDS,7,{'a':(a,a,None,None)}, {},tmp_path,rounds=3,warmups=1)
    assert not made


def test_hardware_entry_always_stops_before_device_creation():
    with pytest.raises(RuntimeError,match='provider'):
        runner.run()


def test_midrun_identity_change_aborts(monkeypatch,tmp_path):
    d,made=setup(monkeypatch);a=np.ones((5,5),dtype=np.uint8)
    def changed(*args,**kwargs):
        d.vulkan_creation_receipt.owner_generation += 1
        return {'outputs':{name:np.array([1],dtype=np.uint32) for name in runner.OUTPUTS},
                'pipeline_ms':1.0,'gpu_stages_ms':{'aggregate':1.0},'readback_ms':1.0,'transfer_inclusive_ms':2.0}
    monkeypatch.setattr(runner,'capture',changed)
    with pytest.raises(ValueError,match='identity changed'):
        runner._execute_with_device(d,UUID,KINDS,7,{'a':(a,a,None,None)}, {},tmp_path,rounds=3,warmups=1)
    assert all(e.query_pool is None for e in made)


def test_body_and_idle_failures_are_both_preserved(monkeypatch,tmp_path):
    d,made=setup(monkeypatch);a=np.ones((5,5),dtype=np.uint8)
    def fail_capture(*a,**k): raise ValueError('capture failure')
    def fail_idle(): raise RuntimeError('idle failure')
    monkeypatch.setattr(runner,'capture',fail_capture);d.wait_for_idle=fail_idle
    with pytest.raises(BaseExceptionGroup) as error:
        runner._execute_with_device(d,UUID,KINDS,7,{'a':(a,a,None,None)}, {},tmp_path,rounds=3,warmups=1)
    assert [str(e) for e in error.value.exceptions]==['capture failure','idle failure']
    assert runner.QUARANTINED_ENGINES[-1][0] is d


def test_failed_coordinator_idle_quarantines_outer_device_and_lock(monkeypatch,tmp_path):
    import fcntl
    d,made=setup(monkeypatch);a=np.ones((5,5),dtype=np.uint8);events=[]
    def fail_idle():
        events.append('idle')
        if events.count('idle') == 1: raise RuntimeError('idle failure')
    d.wait_for_idle=fail_idle;d.close=lambda:events.append('close')
    guard=sys.modules['sintergpu_device_receipt'];path=tmp_path/'lock'
    with pytest.raises(RuntimeError,match='idle failure'):
        with guard.admitted_device(lambda:d,UUID,KINDS,_lock_path=path):
            runner._execute_with_device(d,UUID,KINDS,7,{'a':(a,a,None,None)}, {},tmp_path,rounds=3,warmups=1)
    assert events==['idle']
    assert guard.QUARANTINED[-1][1] is d
    assert all(e.query_pool is not None for e in made)
    with path.open('r') as lock:
        with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
