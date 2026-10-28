"""CPU mock checks for same-device receipt and lifecycle; no Vulkan calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

spec = importlib.util.spec_from_file_location('receipt_guard', Path(__file__).resolve().parents[1] / 'benchmarks/sintergpu_device_receipt.py')
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
TYPES = ('instance', 'physical', 'device')
UUID = bytes(range(1, 17))


def mock_device():
    handles = [NS(type=t, value=i+1) for i, t in enumerate(TYPES)]
    receipt = NS(schema_version=1, finalized=True, owner_generation=1, device_uuid_known=True,
                 device_uuid=UUID, instance=handles[0], physical_device=handles[1], device=handles[2],
                 instance_external=False, physical_device_external=False, device_external=False)
    events = []
    d = NS(native_handles=handles, vulkan_creation_receipt=receipt,
           wait_for_idle=lambda: events.append('idle'), close=lambda: events.append('close'))
    return d, events


def test_matching_same_device_and_close(tmp_path):
    d, events = mock_device()
    with guard.admitted_device(lambda: d, UUID, TYPES, _lock_path=tmp_path / 'lock') as got:
        assert got is d
    assert events == ['idle', 'close']


@pytest.mark.parametrize('mutation', ['uuid', 'handle', 'type', 'missing', 'external', 'unknown'])
def test_reject_receipt_and_still_close(mutation, tmp_path):
    d, events = mock_device(); r = d.vulkan_creation_receipt
    if mutation == 'uuid': r.device_uuid = bytes([9])*16
    elif mutation == 'handle': r.physical_device = NS(type='physical', value=999)
    elif mutation == 'type': d.native_handles[1] = NS(type='device', value=2)
    elif mutation == 'missing': d.vulkan_creation_receipt = None
    elif mutation == 'external': r.device_external = True
    elif mutation == 'unknown': r.device_uuid_known = False
    with pytest.raises(ValueError):
        with guard.admitted_device(lambda: d, UUID, TYPES, _lock_path=tmp_path / 'lock'): pytest.fail('must not enter')
    assert events == ['idle', 'close']


def test_idle_failure_quarantines_device_and_lock(tmp_path):
    d, events = mock_device()
    def fail(): events.append('idle'); raise RuntimeError('idle failed')
    d.wait_for_idle = fail
    with pytest.raises(RuntimeError, match='idle failed'):
        with guard.admitted_device(lambda: d, UUID, TYPES, _lock_path=tmp_path / 'lock'): pass
    assert events == ['idle']
    assert guard.QUARANTINED[-1][1] is d
    with pytest.raises(BlockingIOError):
        with guard.admitted_device(lambda: pytest.fail('created'), UUID, TYPES, _lock_path=tmp_path / 'lock'): pass


def test_busy_lock_fails_before_factory(tmp_path):
    import fcntl
    path = tmp_path / 'lock'
    with path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with guard.admitted_device(lambda: pytest.fail('created device'), UUID, TYPES, _lock_path=path):
                pytest.fail('entered')


def test_body_and_idle_failures_are_preserved(tmp_path):
    d, events = mock_device()
    def fail(): events.append('idle'); raise RuntimeError('idle failed')
    d.wait_for_idle = fail
    with pytest.raises(BaseExceptionGroup) as error:
        with guard.admitted_device(lambda: d, UUID, TYPES, _lock_path=tmp_path / 'lock'):
            raise ValueError('work failed')
    assert [str(e) for e in error.value.exceptions] == ['work failed', 'idle failed']
    assert events == ['idle']


def test_lock_replacement_quarantines_before_close(tmp_path):
    d, events = mock_device(); path = tmp_path / 'lock'
    with pytest.raises(RuntimeError, match='lock'):
        with guard.admitted_device(lambda: d, UUID, TYPES, _lock_path=path):
            path.unlink(); path.touch()
    assert events == ['idle']
    assert guard.QUARANTINED[-1][1] is d


def test_close_and_body_errors_are_preserved(tmp_path):
    d, events = mock_device()
    def fail(): events.append('close'); raise RuntimeError('close failed')
    d.close = fail
    with pytest.raises(BaseExceptionGroup) as error:
        with guard.admitted_device(lambda: d, UUID, TYPES, _lock_path=tmp_path / 'lock'):
            raise ValueError('work failed')
    assert [str(e) for e in error.value.exceptions] == ['work failed', 'close failed']
    assert events == ['idle', 'close']
