"""Blocked StereoForge hardware coordinator with CPU-tested orchestration.

Do not alter the provider gate without reviewed, source-backed runtime and ICD
provenance. This runner records diagnostic evidence only, never qualification.
"""
import json
import os
from pathlib import Path
import tempfile
from sintergpu_device_receipt import admitted_device
from sintergpu_device_selection import create_selected
from sintergpu_execution_coordinator import _execute_with_device

LOCK_PATH = '/tmp/sintergpu-gpu.lock'


def verify_provider():
    raise RuntimeError('blocked: Vulkan runtime provider and ICD provenance are not verified')


def _publish_once(output, payload):
    """Fsync a temporary file, exclusively link, then sync directory metadata."""
    output = Path(output)
    data = (json.dumps(payload, sort_keys=True, allow_nan=False) + '\n').encode()
    dirfd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    temp = None
    linked = False
    identity = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.sintergpu-result-', dir=output.parent, delete=False) as stream:
            temp = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            st = os.fstat(stream.fileno())
            identity = (st.st_dev, st.st_ino)
        os.link(temp, output, follow_symlinks=False)
        linked = True
        os.fsync(dirfd)
        os.unlink(temp)
        temp = None
        os.fsync(dirfd)
    except BaseException as error:
        rollback_errors = []
        if linked:
            try:
                st = os.stat(output, follow_symlinks=False)
                if (st.st_dev, st.st_ino) != identity:
                    raise RuntimeError('published result replaced before rollback')
                os.unlink(output)
                os.fsync(dirfd)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise BaseExceptionGroup('publication and rollback failures', [error, *rollback_errors])
        raise
    finally:
        if temp is not None:
            try: os.unlink(temp)
            except FileNotFoundError: pass
        os.close(dirfd)


def run(*, uuid, adapter_luid, vendor_id, device_id, driver, cases, config,
        witness_dir, output, rounds, warmups):
    """Run is unconditionally blocked before GPU import until provider verified."""
    verify_provider()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('result already exists')
    import slangpy as spy
    kinds = (spy.NativeHandleType.VkInstance, spy.NativeHandleType.VkPhysicalDevice, spy.NativeHandleType.VkDevice)
    with admitted_device(lambda: create_selected(spy, adapter_luid=adapter_luid,
            vendor_id=vendor_id, device_id=device_id), uuid, kinds, _lock_path=LOCK_PATH) as device:
        result = _execute_with_device(device, uuid, kinds, driver, cases, config, witness_dir,
                                      rounds=rounds, warmups=warmups)
    payload = {'status':'unqualified', 'uuid':uuid.hex(), 'adapter_identifier':adapter_luid.hex(),
               'vendor_id':vendor_id, 'device_id':device_id, 'driver':driver,
               'config':config, 'result':result,
               'scope':'diagnostic only; provider, ICD and full qualification pending'}
    _publish_once(output, payload)
