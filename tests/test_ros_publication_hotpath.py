"""Static checks over the actual native subscriber and bounded trace paths.

These checks do not instrument ROS middleware, the OS clocks, or GPU work.
"""
from pathlib import Path
import re
import pytest
from test_kernel_variant_hotpath import body

ROOT=Path(__file__).resolve().parents[1]


def check_callback(source):
    callback=body(source, 'auto receive=')
    forbidden=r'\b(?:new|make_shared|make_unique|resize|reserve|readback|wait_idle|wait_on_submit|fopen|open|write|dump|memcpy)\s*(?:<[^;()]*>)?\s*\('
    assert not re.search(forbidden, callback), 'allocation, full copy, I/O or wait in callback'
    assert 'trace.append(r)' in callback
    assert 'image->data.size()' in callback and 'image->header.stamp' in callback


def check_buffer(source):
    append=body(source, 'bool append(')
    assert not re.search(r'\b(?:new|reserve|resize|malloc|calloc|realloc|wait|write|dump)\s*(?:<[^;()]*>)?\s*\(',append)
    assert 'records_.size()>=limit_-2' in append
    assert append.count('records_.push_back(record)')==1


def test_actual_subscriber_callback_and_trace_append():
    check_callback((ROOT/'apps/ros2_delivery_probe.cpp').read_text())
    check_buffer((ROOT/'include/stereoforge/ros_delivery_trace.hpp').read_text())


def test_mutants_reintroducing_callback_work_are_rejected():
    src=(ROOT/'apps/ros2_delivery_probe.cpp').read_text()
    with pytest.raises(AssertionError): check_callback(src.replace('auto at=ns_now<steady>();','std::memcpy(nullptr,nullptr,0); auto at=ns_now<steady>();',1))
    src=(ROOT/'include/stereoforge/ros_delivery_trace.hpp').read_text()
    with pytest.raises(AssertionError): check_buffer(src.replace('records_.push_back(record);','records_.reserve(limit_); records_.push_back(record);',1))


def check_publisher(source):
    loop=body(source,'while(rclcpp::ok()&&!trace_full')
    forbidden=r'\b(?:new|make_shared|make_unique|resize|reserve|readback|wait_idle|wait_on_submit|fopen|open|write|dump)\s*(?:<[^;()]*>)?\s*\('
    assert not re.search(forbidden,loop), 'explicit allocation, full readback, I/O or device wait in publisher loop'
    assert 'capture->read(' in loop and 'pipeline->enqueue(f)' in loop and 'preprocess->enqueue(f)' in loop
    assert 'count_pending()?std::chrono::milliseconds(1):std::chrono::milliseconds(5)' in loop
    assert 'capture->read(capture_wait)' in loop
    retire=body(source,'auto retire=')
    assert not re.search(forbidden,retire)
    assert 'if(drain)p.output.wait()' in retire and retire.count('.wait()')==1
    assert 'std::memcpy(image.data.data(),bytes.data(),bytes.size())' in retire
    assert 'publisher->publish(image)' in retire and 'trace.append(r)' in retire


def test_native_publisher_path_uses_explicit_boundary_and_drain_only_wait():
    check_publisher((ROOT/'apps/ros2_depth_publisher.cpp').read_text())


def test_replay_backpressures_and_host_endianness_is_checked_before_export():
    source=(ROOT/'apps/ros2_depth_publisher.cpp').read_text()
    run=body(source,'void run(')
    assert 'if(o.source=="replay")capture_config.queue_policy=CaptureQueuePolicy::wait_for_consumer;' in run
    assert run.index('std::endian::native!=std::endian::little') < run.index('initialize_image(')
    assert 'image.is_bigendian=0' in body(source,'void initialize_image(')


def check_replay_and_discovery(source):
    loop=body(source,'while(rclcpp::ok()&&!trace_full')
    gate='if(o.source=="replay"&&count_pending()==pending.size())'
    assert gate in loop and loop.index(gate)<loop.index('capture->read(capture_wait)')
    assert 'std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;' in loop
    assert '++counters.ring_drops;continue;' in loop
    parse=body(source,'Options parse(')
    assert '--subscriber-wait-seconds' in parse and 'std::isfinite(o.subscriber_wait)' in parse
    assert 'o.subscriber_wait<0||o.subscriber_wait>60' in parse
    run=body(source,'void run(')
    assert run.index('if(o.subscriber_wait>0)') < run.index('auto capture=o.source')
    assert run.index('if(o.subscriber_wait>0)') < run.index('auto started=now_ns<steady>()')
    assert '(depth&&depth_pub->get_subscription_count()==0)||(rgb&&rgb_pub->get_subscription_count()==0)' in run
    assert 'subscriber discovery timed out' in run and 'std::this_thread::sleep_until' in run
    assert '{"subscriber_wait_seconds",o.subscriber_wait}' in run


def test_discovery_and_replay_policy():
    check_replay_and_discovery((ROOT/'apps/ros2_depth_publisher.cpp').read_text())


def test_replay_and_discovery_regressions_are_rejected():
    source=(ROOT/'apps/ros2_depth_publisher.cpp').read_text()
    with pytest.raises(AssertionError):
        check_replay_and_discovery(source.replace('o.source=="replay"&&count_pending()==pending.size()', 'false', 1))
    with pytest.raises(AssertionError):
        check_replay_and_discovery(source.replace('(rgb&&rgb_pub->get_subscription_count()==0)', 'false', 1))


def test_publisher_mutants_reintroducing_sync_are_rejected():
    src=(ROOT/'apps/ros2_depth_publisher.cpp').read_text()
    with pytest.raises(AssertionError):
        check_publisher(src.replace('auto f=capture->read(capture_wait);','wait_idle(); auto f=capture->read(capture_wait);',1))
    with pytest.raises(AssertionError):
        check_publisher(src.replace('count_pending()?std::chrono::milliseconds(1):std::chrono::milliseconds(5)',
                                    'std::chrono::milliseconds(5)',1))


def check_export_submission(source):
    forbidden=r'\b(?:wait_idle|wait_on_submit|submit_and_wait|wait_for_idle|wait_for_fences)\s*\('
    for signature in ('void submit(Slot&',
                      'DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaObservationFrame&',
                      'DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaFrame&'):
        assert not re.search(forbidden, body(source, signature)), 'blocking wait in export submission'
    assert 'd.submit_commands(' in body(source, 'void submit(Slot&')


def test_export_submission_is_nonblocking():
    check_export_submission((ROOT/'src/daxa_host_export.cpp').read_text())


def test_export_submission_wait_mutant_is_rejected():
    source=(ROOT/'src/daxa_host_export.cpp').read_text()
    mutant=source.replace('s.copy_ticket=d.submit_commands(', 'd.wait_idle(); s.copy_ticket=d.submit_commands(', 1)
    assert mutant != source
    with pytest.raises(AssertionError, match='blocking wait'):
        check_export_submission(mutant)


def test_publisher_records_enqueue_boundaries_without_extra_work():
    src=(ROOT/'apps/ros2_depth_publisher.cpp').read_text()
    loop=body(src,'while(rclcpp::ok()&&!trace_full')
    assert loop.count('index->producer_enqueue_begin=now_ns<steady>()') == 1
    assert loop.count('index->producer_enqueue_end=now_ns<steady>()') == 2
    assert loop.count('index->export_enqueue_end=now_ns<steady>()') == 1
    assert loop.index('index->producer_enqueue_begin=now_ns<steady>()') < loop.index('pipeline->enqueue(f)')
    assert loop.index('index->producer_enqueue_end=now_ns<steady>()') < loop.index('exporter->enqueue(')
    assert loop.index('exporter->enqueue(') < loop.index('index->export_enqueue_end=now_ns<steady>()')


def check_thread_cpu_instrumentation(source):
    emit=body(source, 'auto emit=')
    sample='if(o.thread_cpu_timing)cpu_ok &= thread_cpu_now('
    assert emit.count(sample)==4
    positions=[emit.index(sample+'r.thread_cpu_copy_begin_ns)'), emit.index('std::memcpy('),
               emit.index(sample+'r.thread_cpu_copy_end_ns)'),
               emit.index(sample+'r.thread_cpu_publish_begin_ns)'), emit.index('publisher->publish(image)'),
               emit.index(sample+'r.thread_cpu_publish_end_ns)')]
    assert positions==sorted(positions)
    sampler=body(source, 'bool thread_cpu_now(')
    assert 'CLOCK_THREAD_CPUTIME_ID' in sampler and 'noexcept' in source[source.index('bool thread_cpu_now('):source.index('bool thread_cpu_now(')+70]
    assert not re.search(r'\b(?:throw|new|malloc|write|wait|dump)\b', sampler)
    assert 'RosThreadCpuStatus::ClockReadFailed' in emit and 'RosThreadCpuStatus::Nonmonotonic' in emit


def test_thread_cpu_samples_are_opt_in_and_surround_operations():
    source=(ROOT/'apps/ros2_depth_publisher.cpp').read_text()
    check_thread_cpu_instrumentation(source)
    with pytest.raises(AssertionError):
        check_thread_cpu_instrumentation(source.replace('if(o.thread_cpu_timing)cpu_ok &= thread_cpu_now(r.thread_cpu_copy_begin_ns)', 'thread_cpu_now(r.thread_cpu_copy_begin_ns)', 1))
