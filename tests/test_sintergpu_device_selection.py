"""CPU-only adapter-selection contract; no GPU runtime is loaded."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

spec=importlib.util.spec_from_file_location('selection',Path(__file__).resolve().parents[1]/'benchmarks/sintergpu_device_selection.py')
selection=importlib.util.module_from_spec(spec);spec.loader.exec_module(selection)
LUID=bytes(range(1,17)); UUID=bytes(range(17,33))


def backend(adapters):
    calls=[]
    class Device:
        @staticmethod
        def enumerate_adapters(kind): calls.append(('enumerate',kind));return adapters
    def create_device(kind,**kwargs): calls.append(('create',kind,kwargs));return object()
    return NS(Device=Device,DeviceType=NS(vulkan='vulkan'),create_device=create_device),calls


def adapter(luid=LUID,vendor=0x10de,device=17):
    return NS(luid=luid,vendor_id=vendor,device_id=device)


def test_selects_exact_unique_identifier_and_vendor(monkeypatch):
    monkeypatch.delenv('SLANGPY_DEVICE_TYPE_OVERRIDE',raising=False)
    spy,calls=backend([adapter(),adapter(bytes([8])*16)])
    selection.create_selected(spy, adapter_luid=LUID,vendor_id=0x10de,device_id=17)
    assert calls[1][2]['adapter_luid']==list(LUID)
    assert calls[1][2]['enable_cuda_interop'] is False
    assert calls[1][0:2]==('create','vulkan')


@pytest.mark.parametrize('adapters', [[],[adapter(),adapter()],[adapter(vendor=1)],[adapter(device=18)]])
def test_invalid_or_ambiguous_adapter_fails_before_creation(monkeypatch,adapters):
    monkeypatch.delenv('SLANGPY_DEVICE_TYPE_OVERRIDE',raising=False)
    spy,calls=backend(adapters)
    with pytest.raises(ValueError):selection.create_selected(spy,adapter_luid=LUID,vendor_id=0x10de,device_id=17)
    assert not any(c[0]=='create' for c in calls)


def test_override_and_invalid_luid_fail_before_enumeration(monkeypatch):
    spy,calls=backend([adapter()]);monkeypatch.setenv('SLANGPY_DEVICE_TYPE_OVERRIDE','cuda')
    with pytest.raises(ValueError):selection.create_selected(spy,adapter_luid=LUID,vendor_id=0x10de,device_id=17)
    assert calls==[]
    monkeypatch.delenv('SLANGPY_DEVICE_TYPE_OVERRIDE')
    with pytest.raises(ValueError):selection.create_selected(spy,adapter_luid=bytes(16),vendor_id=0x10de,device_id=17)
    assert calls==[]


def test_selected_device_receipt_rejects_wrong_uuid_before_use(monkeypatch,tmp_path):
    import sys
    spec2=importlib.util.spec_from_file_location('sintergpu_device_receipt',Path(__file__).resolve().parents[1]/'benchmarks/sintergpu_device_receipt.py')
    guard=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(guard)
    monkeypatch.delenv('SLANGPY_DEVICE_TYPE_OVERRIDE',raising=False)
    spy,calls=backend([adapter()]);events=[]
    kinds=('instance','physical','device');handles=[NS(type=t,value=i+1) for i,t in enumerate(kinds)]
    receipt=NS(schema_version=1,finalized=True,owner_generation=1,device_uuid_known=True,device_uuid=b'x'*16,
        instance=handles[0],physical_device=handles[1],device=handles[2],instance_external=False,
        physical_device_external=False,device_external=False)
    device=NS(native_handles=handles,vulkan_creation_receipt=receipt,wait_for_idle=lambda:events.append('idle'),close=lambda:events.append('close'))
    def create(kind,**kwargs):calls.append(('create',kind,kwargs));return device
    spy.create_device=create
    with pytest.raises(ValueError,match='receipt'):
        with guard.admitted_device(lambda:selection.create_selected(spy,adapter_luid=LUID,vendor_id=0x10de,device_id=17),UUID,kinds,_lock_path=tmp_path/'lock'):
            pytest.fail('mismatched UUID admitted')
    assert [c[0] for c in calls]==['enumerate','create'] and events==['idle','close']
