"""CPU tests for locked runner orchestration. No GPU runtime is loaded."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
for name in ('sintergpu_device_receipt','sintergpu_device_selection','sintergpu_qualification_runner','sintergpu_fixture_execution','sintergpu_execution_coordinator','sintergpu_controlled_runner'):
    spec=importlib.util.spec_from_file_location(name,ROOT/'benchmarks'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
runner=sys.modules['sintergpu_controlled_runner']
UUID=bytes(range(1,17));LUID=bytes(range(17,33));KINDS=('instance','physical','device')


def args(tmp_path):
    a=np.ones((5,5),dtype=np.uint8)
    return dict(uuid=UUID,adapter_luid=LUID,vendor_id=0x10de,device_id=17,driver=7,
                cases={'fixture':(a,a,None,None)},config={'max_disparity':8},
                witness_dir=tmp_path,output=tmp_path/'result.json',rounds=3,warmups=1)


def test_production_gate_precedes_runtime_import(monkeypatch,tmp_path):
    monkeypatch.delitem(sys.modules,'slangpy',raising=False)
    with pytest.raises(RuntimeError,match='provider'):
        runner.run(**args(tmp_path))
    assert 'slangpy' not in sys.modules and not (tmp_path/'result.json').exists()


def test_mocked_route_locks_selects_verifies_then_executes_and_closes(monkeypatch,tmp_path):
    events=[];h=[NS(type=k,value=i+1) for i,k in enumerate(KINDS)]
    receipt=NS(schema_version=1,finalized=True,owner_generation=1,device_uuid_known=True,device_uuid=UUID,
        instance=h[0],physical_device=h[1],device=h[2],instance_external=False,physical_device_external=False,device_external=False)
    device=NS(native_handles=h,vulkan_creation_receipt=receipt,wait_for_idle=lambda:events.append('idle'),close=lambda:events.append('close'))
    spy=NS(NativeHandleType=NS(VkInstance=KINDS[0],VkPhysicalDevice=KINDS[1],VkDevice=KINDS[2]))
    monkeypatch.setitem(sys.modules,'slangpy',spy)
    monkeypatch.setattr(runner,'verify_provider',lambda:events.append('provider'))
    monkeypatch.setattr(runner,'create_selected',lambda *a,**k:events.append('create') or device)
    monkeypatch.setattr(runner,'_execute_with_device',lambda dev,*a,**k: events.append('execute') or {'status':'unqualified','cases':{}})
    monkeypatch.setattr(runner,'LOCK_PATH',str(tmp_path/'lock'))
    runner.run(**args(tmp_path))
    assert events==['provider','create','execute','idle','close']
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['status']=='unqualified' and result['uuid']==UUID.hex() and result['driver']==7


def test_existing_result_never_replaced(monkeypatch,tmp_path):
    p=tmp_path/'result.json';p.write_text('existing')
    monkeypatch.setattr(runner,'verify_provider',lambda:None)
    with pytest.raises(FileExistsError):runner.run(**args(tmp_path))
    assert p.read_text()=='existing'


def test_execution_failure_is_not_written_as_success(monkeypatch,tmp_path):
    h=[NS(type=k,value=i+1) for i,k in enumerate(KINDS)]
    receipt=NS(schema_version=1,finalized=True,owner_generation=1,device_uuid_known=True,device_uuid=UUID,
        instance=h[0],physical_device=h[1],device=h[2],instance_external=False,physical_device_external=False,device_external=False)
    events=[];device=NS(native_handles=h,vulkan_creation_receipt=receipt,wait_for_idle=lambda:events.append('idle'),close=lambda:events.append('close'))
    monkeypatch.setitem(sys.modules,'slangpy',NS(NativeHandleType=NS(VkInstance=KINDS[0],VkPhysicalDevice=KINDS[1],VkDevice=KINDS[2])))
    monkeypatch.setattr(runner,'verify_provider',lambda:None)
    monkeypatch.setattr(runner,'create_selected',lambda *a,**k:device)
    def failed(*a,**k):raise ValueError('parity failure')
    monkeypatch.setattr(runner,'_execute_with_device',failed)
    monkeypatch.setattr(runner,'LOCK_PATH',str(tmp_path/'lock'))
    with pytest.raises(ValueError,match='parity failure'):runner.run(**args(tmp_path))
    assert not (tmp_path/'result.json').exists() and events==['idle','close']


@pytest.mark.parametrize('fail_at', [1, 2, 3])
def test_sync_failure_leaves_no_success_output(monkeypatch,tmp_path,fail_at):
    real_fsync=runner.os.fsync; calls=[]
    def injected(fd):
        calls.append(fd)
        if len(calls)==fail_at: raise OSError('sync failure')
        return real_fsync(fd)
    monkeypatch.setattr(runner.os, 'fsync', injected)
    output=tmp_path/'result.json'
    with pytest.raises(OSError,match='sync failure'):
        runner._publish_once(output, {'status':'unqualified'})
    assert not output.exists()
    assert not list(tmp_path.glob('.sintergpu-result-*'))


def test_atomic_publication_does_not_replace_existing_file(tmp_path):
    output=tmp_path/'result.json';output.write_text('existing')
    with pytest.raises(FileExistsError):runner._publish_once(output, {'status':'unqualified'})
    assert output.read_text()=='existing'
    assert not list(tmp_path.glob('.sintergpu-result-*'))
