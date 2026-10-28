"""CPU-only source guards for the bounded in-process GPU consumer.

These checks do not execute Vulkan, validate external consumers, or measure latency.
"""
from pathlib import Path
import re
import pytest
from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/daxa_gpu_consumer.cpp'


def check_lifetime_and_barriers(source):
    submit = body(source, 'void submit(Slot& s,const DaxaObservationFrame& input)')
    available = body(source, 'std::shared_ptr<Slot>& available()')
    collect = body(source, 'size_t DaxaGpuConsumer::collect()')
    retain = body(source, 'void DaxaGpuConsumer::Frame::retain_until(uint64_t ticket)')
    assert submit.index('s.producer=input') < submit.index('d.submit_commands('), 'retain source before submission'
    assert 's->consumer_ticket>s->copy_ticket?s->consumer_ticket:s->copy_ticket' in available
    assert 'detail::available_slot(s.use_count(),last,boundary)' in available, 'slot requires completed GPU use and released leases'
    assert 'detail::completed_submission(s.copy_ticket,boundary)' in collect, 'producer lease must survive copy completion'
    assert 'ticket>slot_->copy_ticket&&ticket<=latest' in retain
    assert 'slot_->consumer_ticket=ticket' in retain
    before = submit.split('recorder.copy_buffer_to_buffer(', 1)[0]
    for access in ('COMPUTE_SHADER_WRITE', 'TRANSFER_WRITE', 'COMPUTE_SHADER_READ', 'TRANSFER_READ'):
        assert access in before, 'destination reuse and producer dependency required'
    assert '.dst_access=daxa::AccessConsts::TRANSFER_READ|daxa::AccessConsts::TRANSFER_WRITE' in before, 'destination write dependency required'
    after = submit.rsplit('recorder.copy_buffer_to_buffer(', 1)[1]
    assert '.src_access=daxa::AccessConsts::TRANSFER_WRITE' in after
    assert 'COMPUTE_SHADER_READ|daxa::AccessConsts::TRANSFER_READ' in after
    assert 'if(latest>before)s.copy_ticket=latest' in submit, 'uncertain submit must retain source'
    assert 'catch(...){std::terminate();}' in submit, 'unknown queue state must fail closed'
    assert not re.search(r'wait_on_submit\s*\(|wait_idle\s*\(|memcpy\s*\(|buffer_host_address\s*\(', submit)


def test_lifetime_and_barriers():
    check_lifetime_and_barriers(SOURCE.read_text())


@pytest.mark.parametrize('old,new,diagnostic', [
    ('s.producer=input', '/* source lease lost */', 'retain source'),
    ('.dst_access=daxa::AccessConsts::TRANSFER_READ|daxa::AccessConsts::TRANSFER_WRITE', '.dst_access=daxa::AccessConsts::TRANSFER_READ', 'destination write'),
    ('detail::completed_submission(s.copy_ticket,boundary)', 'true', 'producer lease'),
    ('if(latest>before)s.copy_ticket=latest', 'if(latest>before)s.copy_ticket=0', 'uncertain submit'),
])
def test_guard_detects_regressions(old, new, diagnostic):
    source = SOURCE.read_text()
    assert old in source
    with pytest.raises((AssertionError, ValueError), match=diagnostic if old != 's.producer=input' else None):
        check_lifetime_and_barriers(source.replace(old, new, 1))


def test_no_application_wait_or_full_readback_in_enqueue():
    source = SOURCE.read_text()
    entry = body(source, 'DaxaGpuConsumer::Frame DaxaGpuConsumer::enqueue(')
    assert entry.index('require_thread()') < entry.index('e.available()')
    assert 'e.context->device.collect_garbage()' in entry
    assert not re.search(r'wait_on_submit\s*\(|wait_idle\s*\(|memcpy\s*\(|buffer_host_address\s*\(', entry)


def check_teardown_and_capacity(source):
    slot = body(source, '~Slot()noexcept')
    impl = body(source, '~Impl()noexcept')
    frame = body(source, 'DaxaGpuConsumer::Frame::~Frame()noexcept')
    consumer = body(source, 'DaxaGpuConsumer::~DaxaGpuConsumer()noexcept')
    for teardown in (slot, impl, frame, consumer):
        assert 'std::this_thread::get_id()' in teardown and 'std::terminate()' in teardown, 'thread-affine teardown must fail closed'
    capacity = body(source, 'bool DaxaGpuConsumer::has_capacity()const')
    assert capacity.index('require_thread()') < capacity.index('oldest_pending_submit_index()')
    assert 'detail::available_slot(s.use_count(),last,boundary)' in capacity
    assert 'catch(' not in capacity, 'completion failures must propagate'
    assert not re.search(r'wait_on_submit\s*\(|wait_idle\s*\(', capacity)


def test_thread_teardown_and_capacity_fail_closed():
    check_teardown_and_capacity(SOURCE.read_text())


@pytest.mark.parametrize('old,new', [
    ('if(thread!=std::this_thread::get_id())std::terminate();', ''),
    ('auto boundary=impl_->context->device.oldest_pending_submit_index();', 'try { auto boundary=impl_->context->device.oldest_pending_submit_index(); } catch(...) { return false; }'),
])
def test_thread_and_capacity_guard_detect_regressions(old, new):
    source = SOURCE.read_text()
    assert old in source
    with pytest.raises(AssertionError):
        check_teardown_and_capacity(source.replace(old, new, 1))
