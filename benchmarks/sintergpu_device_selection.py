"""Explicit adapter selection helper, not a hardware admission entry point.

The Slang RHI adapter identifier may be a platform LUID or a UUID. A caller
must establish the mapping independently and compare the created device's
same-device UUID receipt; this helper alone cannot establish physical identity.
The runtime/provider provenance gate remains mandatory and is not implemented.
"""
import os


def create_selected(spy, *, adapter_luid, vendor_id, device_id):
    if 'SLANGPY_DEVICE_TYPE_OVERRIDE' in os.environ:
        raise ValueError('SlangPy device type override is forbidden')
    if not isinstance(adapter_luid, bytes) or len(adapter_luid) != 16 or not any(adapter_luid):
        raise ValueError('nonzero 16-byte adapter identifier required')
    if any(type(value) is not int or not 0 <= value <= 0xffffffff for value in (vendor_id, device_id)):
        raise ValueError('explicit vendor and device IDs required')
    adapters = spy.Device.enumerate_adapters(spy.DeviceType.vulkan)
    matches = [a for a in adapters if bytes(a.luid) == adapter_luid]
    if len(matches) != 1 or matches[0].vendor_id != vendor_id or matches[0].device_id != device_id:
        raise ValueError('adapter identity missing, ambiguous or mismatched')
    return spy.create_device(spy.DeviceType.vulkan, adapter_luid=list(adapter_luid),
                             enable_cuda_interop=False, enable_debug_layers=True)
