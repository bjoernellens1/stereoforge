"""CPU mock of caller-owned device integration; never creates a GPU."""
from types import SimpleNamespace
import sys
import pytest
from stereoforge.gpu import VulkanStereo


class Module:
    def entry_point(self, name): return name


class Device:
    info = SimpleNamespace(adapter_name='verified-device')
    def __init__(self): self.sources = []; self.closed = False
    def load_module_from_source(self, name, source): self.sources.append(source); return Module()
    def link_program(self, modules, entries): return entries
    def create_compute_kernel(self, program): return program
    def create_query_pool(self, *args): return object()
    def close(self): self.closed = True


def test_borrowed_device_is_used_without_creating_or_closing(monkeypatch):
    device = Device()
    def forbidden(*args, **kwargs): pytest.fail('unexpected device creation')
    monkeypatch.setitem(sys.modules, 'slangpy', SimpleNamespace(create_device=forbidden,
        DeviceType=SimpleNamespace(vulkan=object()), QueryType=SimpleNamespace(timestamp=object())))
    engine = VulkanStereo(device=device)
    assert engine.device is device and engine.device_name == 'verified-device'
    assert len(device.sources) == 1 and not device.closed


def test_default_still_creates_one_device(monkeypatch):
    devices = []
    def create(*args, **kwargs):
        devices.append(Device()); return devices[-1]
    monkeypatch.setitem(sys.modules, 'slangpy', SimpleNamespace(create_device=create,
        DeviceType=SimpleNamespace(vulkan=object()), QueryType=SimpleNamespace(timestamp=object())))
    engine = VulkanStereo()
    assert devices == [engine.device]


def test_receipt_admission_precedes_shader_loading(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('receipt_guard_integration', Path(__file__).resolve().parents[1] / 'benchmarks/sintergpu_device_receipt.py')
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    monkeypatch.setitem(sys.modules, 'slangpy', SimpleNamespace(create_device=lambda *a, **k: pytest.fail('extra device'),
        DeviceType=SimpleNamespace(vulkan=object()), QueryType=SimpleNamespace(timestamp=object())))
    kinds = ('instance', 'physical', 'device'); expected = bytes(range(1,17))
    device = Device(); device.wait_for_idle = lambda: None
    device.native_handles = [SimpleNamespace(type=t, value=i+1) for i,t in enumerate(kinds)]
    h = device.native_handles
    device.vulkan_creation_receipt = SimpleNamespace(schema_version=1, finalized=True, owner_generation=1,
        device_uuid_known=True, device_uuid=expected, instance=h[0], physical_device=h[1], device=h[2],
        instance_external=False, physical_device_external=False, device_external=False)
    with guard.admitted_device(lambda: device, expected, kinds, _lock_path=tmp_path/'one') as admitted:
        engine = VulkanStereo(device=admitted)
        assert engine.device is device and len(device.sources) == 1
        del engine
    assert device.closed
    device.closed = False; device.sources.clear()
    device.vulkan_creation_receipt.device_uuid = b'x'*16
    with pytest.raises(ValueError):
        with guard.admitted_device(lambda: device, expected, kinds, _lock_path=tmp_path/'two') as admitted:
            VulkanStereo(device=admitted)
    assert not device.sources and device.closed
