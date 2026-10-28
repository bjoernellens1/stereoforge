"""Receipt identity/lifetime scaffold; not yet a hardware admission mechanism.

The required receipt exists only in an unqualified instrumented runtime. Its
provenance and same-loader/provider chain need independent verification before
this code can authorize physical GPU use. Do not use with untrusted receipts.
"""
from contextlib import contextmanager
import fcntl
import os
import stat

LOCK_PATH = '/tmp/sintergpu-gpu.lock'


def check_receipt(device, expected_uuid, handle_types):
    if not isinstance(expected_uuid, bytes) or len(expected_uuid) != 16 or not any(expected_uuid):
        raise ValueError('expected UUID must be a nonzero 16-byte value')
    receipt = getattr(device, 'vulkan_creation_receipt', None)
    if (receipt is None or receipt.schema_version != 1 or not receipt.finalized
            or receipt.owner_generation <= 0 or not receipt.device_uuid_known
            or receipt.device_uuid != expected_uuid
            or receipt.instance_external or receipt.physical_device_external or receipt.device_external):
        raise ValueError('device receipt missing or mismatched')
    handles = device.native_handles
    recorded = (receipt.instance, receipt.physical_device, receipt.device)
    if len(handles) != 3 or len(handle_types) != 3:
        raise ValueError('invalid native handles')
    for actual, saved, kind in zip(handles, recorded, handle_types):
        if actual.type != kind or saved.type != kind or not actual.value or actual.value != saved.value:
            raise ValueError('device receipt native handles mismatch')


# Keeping references prevents Python destructors and preserves the advisory lock
# after a teardown failure. Process termination still releases OS resources.
QUARANTINED = []


def _verify_lock(fd, path):
    held = os.fstat(fd)
    current = os.stat(path, follow_symlinks=False)
    if (not stat.S_ISREG(held.st_mode) or held.st_nlink != 1 or
            (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino)):
        raise RuntimeError('lock inode changed')
    with open(f'/proc/self/fdinfo/{fd}') as info:
        lines = info.read().splitlines()
    if not any(line.startswith('lock:') and 'FLOCK' in line and 'WRITE' in line for line in lines):
        raise RuntimeError('lock lease missing')


@contextmanager
def admitted_device(factory, expected_uuid, handle_types, *, _lock_path=LOCK_PATH):
    """CPU-only scaffold; receipt/provider authenticity remains unverified."""
    fd = os.open(_lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    device = None
    quarantine = False
    errors = []
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _verify_lock(fd, _lock_path)
        device = factory()
        try:
            check_receipt(device, expected_uuid, handle_types)
            yield device
        except BaseException as error:
            errors.append(error)
        try:
            device.wait_for_idle()
        except BaseException as error:
            errors.append(error)
            quarantine = True
        try:
            _verify_lock(fd, _lock_path)
        except BaseException as error:
            errors.append(error)
            quarantine = True
        if not quarantine:
            try:
                device.close()
            except BaseException as error:
                errors.append(error)
                quarantine = True
            try:
                _verify_lock(fd, _lock_path)
            except BaseException as error:
                errors.append(error)
                quarantine = True
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise BaseExceptionGroup('work and teardown failures', errors)
    finally:
        if quarantine:
            QUARANTINED.append((fd, device))
        else:
            os.close(fd)
