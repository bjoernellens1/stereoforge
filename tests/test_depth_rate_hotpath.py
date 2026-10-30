"""Static regression checks for depth-rate admission on the native ROS delivery path.

These checks prevent accidental bypasses; they are not live camera/GPU tests.
"""
from pathlib import Path

from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]


def publisher_loop(source: str) -> str:
    return body(source, "while(rclcpp::ok()&&!trace_full")


def check_admission(source: str) -> None:
    loop = publisher_loop(source)
    assert "if(!valid_source(f,stamp))" in loop
    assert "scene_monitor.observe(f->packed_yuyv" in loop
    assert "rate_controller.observe(f->camera_timestamp_ns,activity).due" in loop
    assert "rate_controller.reject_busy()" in loop
    assert "rate_controller.admitted(f->camera_timestamp_ns)" in loop
    assert "scene_monitor.depth_admitted()" in loop
    assert loop.index("if(!valid_source(f,stamp))") < loop.index("scene_monitor.observe(")
    assert loop.index("scene_monitor.observe(") < loop.index("rate_controller.observe(")
    assert loop.index("rate_controller.observe(") < loop.index("std::find_if(pending.begin()")
    assert loop.index("std::find_if(pending.begin()") < loop.index("pipeline->enqueue(f)")
    assert loop.index("pipeline->enqueue(f)") < loop.index("rate_controller.admitted(")
    assert loop.index("exporter->enqueue(") < loop.index("rate_controller.admitted(")
    assert "if(!rate_controller.observe(f->camera_timestamp_ns,activity).due)continue;" in loop
    assert "++counters.ring_drops;continue;" in loop
    assert "if(o.depth_rate==\"adaptive\")scene_monitor.depth_admitted();" in loop


def test_depth_rate_gate_precedes_gpu_submission() -> None:
    source = (ROOT / "apps/ros2_depth_publisher.cpp").read_text()
    check_admission(source)


def test_rate_gate_mutations_are_rejected() -> None:
    source = (ROOT / "apps/ros2_depth_publisher.cpp").read_text()
    mutant = source.replace(
        "if(!rate_controller.observe(f->camera_timestamp_ns,activity).due)continue;",
        "rate_controller.observe(f->camera_timestamp_ns,activity);",
        1,
    )
    assert mutant != source
    try:
        check_admission(mutant)
    except AssertionError:
        pass
    else:
        raise AssertionError("rate-gate removal was not detected")


def test_rate_configuration_and_telemetry_are_separate_from_ros_trace() -> None:
    source = (ROOT / "apps/ros2_depth_publisher.cpp").read_text()
    parser = body(source, "Options parse(")
    assert "--depth-rate" in parser and "--capture-fps" in parser
    assert "--consumer-min-depth-fps" in parser and "--max-depth-age-ms" in parser
    assert "depth_rate_config(o);" in parser
    run = body(source, "void run(")
    loop = publisher_loop(source)
    assert "DepthRateController rate_controller(" in run
    assert "YuyvSceneMonitor scene_monitor;" in run
    assert "stereoforge.depth_rate_report.v1" in run
    assert "stereoforge.depth_rate_report.v1" not in loop
    assert "intentional_skips" in run and "slot_rejections" in run


def test_scene_monitor_is_cpu_only_and_fixed_storage() -> None:
    monitor = (ROOT / "include/stereoforge/depth_rate_controller.hpp").read_text()
    assert "std::array<std::uint8_t, samples> previous_" in monitor
    assert "yuyv[py*stride + 2*px]" in monitor
    assert "void depth_admitted() noexcept" in monitor
    assert "Fail open: never throttle on missing evidence" in monitor
    assert not any(token in monitor for token in ("daxa::", "cuda", "hipLaunch", "new uint8_t"))
