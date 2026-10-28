"""Parity coordinator. The public hardware entry remains unconditionally blocked.

The receipt and runtime provider are not authenticated. Internal functions are
for CPU integration tests and must not be used to authorize hardware execution.
"""
from pathlib import Path
import hashlib
import numpy as np
import sys
from stereoforge.gpu import VulkanStereo
from sintergpu_device_receipt import check_receipt, quarantine_resources
from sintergpu_fixture_execution import capture, OUTPUTS
from sintergpu_qualification_runner import collect

# Preserve engine/device objects on unsuccessful idle to avoid premature release.
QUARANTINED_ENGINES = []


def _identity(device, uuid, kinds, driver):
    check_receipt(device, uuid, kinds)
    receipt = device.vulkan_creation_receipt
    if type(driver) is not int or driver < 0 or receipt.driver_version != driver:
        raise ValueError('driver identity mismatch')
    return (receipt.owner_generation, bytes(receipt.device_uuid), receipt.driver_version,
            tuple((h.type, h.value) for h in device.native_handles))



def _fixture_metadata(values):
    names = ('left', 'right', 'lower', 'upper')
    return {name: (None if value is None else {
        'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
        'dtype': value.dtype.str, 'shape': list(value.shape)})
        for name, value in zip(names, values)}


def _freeze_cases(cases):
    frozen = {}
    for name, values in cases.items():
        if not isinstance(name, str) or not name or len(values) != 4:
            raise ValueError('invalid fixture')
        copies = []
        for value in values:
            if value is None:
                copies.append(None)
            else:
                if not isinstance(value, np.ndarray) or value.dtype.hasobject:
                    raise ValueError('fixture must contain numeric arrays')
                array = np.array(value, copy=True, order='C')
                array.setflags(write=False)
                copies.append(array)
        frozen[name] = tuple(copies)
    return frozen

def _execute_with_device(device, uuid, kinds, driver, cases, config, witness_dir, *, rounds, warmups):
    """Run actual StereoForge entry points once hardware admission is available.

    Caller must already own the device and lock, have verified provenance, and
    close the device after this function has returned. Results remain unqualified.
    """
    if not cases or type(warmups) is not int or warmups < 1 or type(rounds) is not int or rounds < 3:
        raise ValueError('fixtures, warmups and at least three rounds required')
    if not Path(witness_dir).is_dir():
        raise ValueError('existing witness directory required')
    cases = _freeze_cases(cases)
    fixture_metadata = {name: _fixture_metadata(values) for name, values in cases.items()}
    initial = _identity(device, uuid, kinds, driver)
    engines = {}
    try:
        for variant, enabled in [('baseline', False), ('candidate', True)]:
            engine = VulkanStereo(device=device, shared_reduction=enabled, **config)
            engines[variant] = engine
            if engine.device is not device or engine.shader_source_identity['shared_reduction'] is not enabled:
                raise ValueError('engine source or device identity mismatch')
        identities = {name: dict(engine.shader_source_identity) for name, engine in engines.items()}
        if identities['baseline']['source_sha256'] == identities['candidate']['source_sha256']:
            raise ValueError('baseline and candidate source identities are identical')
        def invoke(variant, case):
            if _identity(device, uuid, kinds, driver) != initial:
                raise ValueError('device identity changed')
            left, right, lower, upper = cases[case]
            if _fixture_metadata(cases[case]) != fixture_metadata[case]:
                raise ValueError('fixture input changed')
            result = capture(engines[variant], left, right, lower=lower, upper=upper)
            if _identity(device, uuid, kinds, driver) != initial:
                raise ValueError('device identity changed')
            if _fixture_metadata(cases[case]) != fixture_metadata[case]:
                raise ValueError('fixture input changed')
            return result
        result = {'status': 'unqualified', 'cases': {}}
        for case in cases:
            for _ in range(warmups):
                for variant in ('baseline', 'candidate'):
                    invoke(variant, case)
            result['cases'].update(collect(invoke, [case], rounds=rounds, witness_dir=witness_dir)['cases'])
        result['fixtures'] = fixture_metadata
        result['shader_sources'] = identities
        result['warmups_per_variant'] = warmups
        return result
    finally:
        pending = sys.exception()
        try:
            device.wait_for_idle()
        except BaseException as idle_error:
            QUARANTINED_ENGINES.append((device, engines))
            quarantine_resources(device, engines)
            # Capture the propagating workload error before entering the nested
            # handler (sys.exception here refers to idle_error).
            if pending is not None:
                raise BaseExceptionGroup('work and idle failures', [pending, idle_error])
            raise
        for engine in engines.values():
            engine.buffers.clear()
            engine.kernels.clear()
            engine.query_pool = None


def run():
    raise RuntimeError('blocked: runtime and Vulkan provider provenance are not verified')
