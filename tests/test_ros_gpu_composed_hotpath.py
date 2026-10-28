"""CPU source guards only; runtime GPU, ROS and driver allocation remain unmeasured."""
from pathlib import Path
import re
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'apps/ros2_gpu_composed.cpp'


def check_composed_hotpath(source):
    loop = source.split('while(rclcpp::ok()&&admitted', 1)[1].split('if(!rclcpp::ok())reason=', 1)[0]
    callback = source.split('create_subscription<Adapted>', 1)[1].split('rclcpp::executors::SingleThreadedExecutor', 1)[0]
    assert 'SingleThreadedExecutor' in source
    assert 'use_intra_process_comms(true)' in source
    assert 'std::array<Pending,2>' in source
    assert 'frame->observation()' in callback
    assert 'consumer->has_capacity()' in loop and 'consumer->has_capacity()' in callback
    assert 'source=="replay"' in loop
    assert 'executor.spin_some()' in loop
    assert 'unexpected GPU message identity' in callback
    assert 'GPU message was not delivered synchronously' in loop
    assert source.index('subscriber.reset();publisher.reset()') < source.index('consumer.reset();pipeline.reset()')
    assert 'metadata only' in source and 'same-device GPU-to-GPU copies' in source
    assert 'sidecars' in source

    # One explicit per-frame message allocation is intentional; header string
    # assignment may also allocate. These checks do not see ROS, executor,
    # Daxa command-recorder, allocator, or standard-library internals.
    message = 'std::make_unique<RosGpuFrame>(depth.observations(),std::move(header))'
    assert loop.count(message) == 1, 'exactly one expected message allocation'
    assert 'std_msgs::msg::Header header;' in loop and 'header.frame_id=o.frame;' in loop
    scrubbed_loop = loop.replace(message, '')
    for name, section in [('loop', scrubbed_loop), ('callback', callback)]:
        assert not re.search(r'\b(?:memcpy|memmove|readback|readback_images|readback_stereo|buffer_host_address|map_memory)\s*\(', section), f'{name}: host readback'
        assert not re.search(r'(?:\.|->)\s*(?:wait|wait_idle|wait_on_submit|drain)\s*\(', section), f'{name}: blocking wait'
        assert not re.search(r'\b(?:wait_idle|wait_on_submit|vkDeviceWaitIdle|vkQueueWaitIdle)\s*\(', section), f'{name}: blocking wait'
        assert not re.search(r'\b(?:new|malloc|calloc|realloc|make_unique|make_shared)\b', section), f'{name}: unexpected allocation'
        assert not re.search(r'\b(?:std::vector|std::map|std::unordered_map|std::deque)\s*<', section), f'{name}: unexpected dynamic container'
        assert not re.search(r'(?:\.|->)\s*(?:resize|reserve)\s*\(', section), f'{name}: unexpected container allocation'
    # Bounded replay pacing and capture polling are explicit CPU waits.
    assert loop.count('std::this_thread::sleep_for(std::chrono::milliseconds(1))') == 1
    assert 'capture->read(full?std::chrono::milliseconds(1):std::chrono::milliseconds(5))' in loop


def test_composed_path_keeps_gpu_leases_and_avoids_host_pixel_copies():
    check_composed_hotpath(SOURCE.read_text())


@pytest.mark.parametrize('old,new,diagnostic', [
    ('executor.spin_some();retire();', 'executor.spin_some();gpu.wait();retire();', 'blocking wait'),
    ('++received;if(frame->observation()', 'gpu.wait_on_submit();++received;if(frame->observation()', 'blocking wait'),
    ('executor.spin_some();retire();', 'memcpy(dst,src,n);executor.spin_some();retire();', 'host readback'),
    ('++received;if(frame->observation()', 'auto pixels=readback(frame);++received;if(frame->observation()', 'host readback'),
    ('executor.spin_some();retire();', 'auto extra=std::make_unique<int>();executor.spin_some();retire();', 'unexpected allocation'),
    ('++received;if(frame->observation()', 'auto extra=std::make_shared<int>();++received;if(frame->observation()', 'unexpected allocation'),
    ('executor.spin_some();retire();', 'std::vector<int> extra;executor.spin_some();retire();', 'dynamic container'),
    ('executor.spin_some();retire();', 'buffer.resize(32);executor.spin_some();retire();', 'container allocation'),
    ('std::make_unique<RosGpuFrame>(depth.observations(),std::move(header))', 'std::make_unique<RosGpuFrame>()', 'message allocation'),
])
def test_guard_detects_regressions(old, new, diagnostic):
    source = SOURCE.read_text()
    assert old in source
    with pytest.raises(AssertionError, match=diagnostic):
        check_composed_hotpath(source.replace(old, new, 1))


def test_gpu_composed_target_is_opt_in():
    cmake = (ROOT / 'cmake/ROS2.cmake').read_text()
    assert 'option(STEREOFORGE_ROS2_GPU_COMPOSED' in cmake
    assert 'add_library(stereoforge_daxa_gpu_consumer STATIC src/daxa_gpu_consumer.cpp)' in cmake
    assert 'target_link_libraries(stereoforge_daxa_gpu_consumer PUBLIC stereoforge_daxa_depth)' in cmake
    assert 'target_link_libraries(stereoforge_ros2_gpu_composed stereoforge_daxa_gpu_consumer stereoforge_capture)' in cmake
    assert 'target_link_libraries(daxa_gpu_consumer_check PRIVATE stereoforge_daxa_gpu_consumer stereoforge_daxa_view)' in cmake
    assert 'add_executable(stereoforge_ros2_gpu_composed apps/ros2_gpu_composed.cpp)' in cmake
    assert 'add_executable(daxa_gpu_consumer_check apps/daxa_gpu_consumer_check.cpp)' in cmake


def test_trace_explicitly_marks_unverified_clock_and_partial_memory():
    source = SOURCE.read_text()
    assert '{"clock_mapping_status","unverified"}' in source
    assert '{"observation_age",{{"status","unavailable"}' in source
    assert '{"memory_accounting",{{"status","partial"}' in source
    assert '{"producer_and_consumer_bytes","requested GPU buffers only"}' in source
    for category in ('capture', 'imu', 'ros', 'driver'):
        assert '{"' + category + '","unavailable"}' in source
    assert 'local copy demonstration only; downstream robotics consumption unverified' in source
    assert 'are retained in producer observation but are not copied to consumer buffers' in source
