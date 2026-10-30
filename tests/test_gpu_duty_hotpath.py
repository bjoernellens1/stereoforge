"""Verify optional GPU budget admission and completion stay out of expensive paths."""
from pathlib import Path
from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]


def test_budget_is_checked_after_depth_rate_and_before_gpu_enqueue():
    src = (ROOT / "apps/ros2_depth_publisher.cpp").read_text()
    loop = body(src, "while(rclcpp::ok()&&!trace_full")
    assert loop.index("rate_controller.observe(f->camera_timestamp_ns,activity)") < \
           loop.index("gpu_budget.may_admit(f->camera_timestamp_ns)") < \
           loop.index("std::find_if(pending.begin()") < loop.index("pipeline->enqueue(f)")
    assert "rate_controller.reject_budget();continue;" in loop
    assert loop.index("exporter->enqueue(") < loop.index("gpu_budget.reserve(f->frame_id,f->camera_timestamp_ns)")
    assert "rate_controller.admitted(f->camera_timestamp_ns)" in loop


def test_budget_is_charged_on_retire_not_by_cpu_wall_time():
    src = (ROOT / "apps/ros2_depth_publisher.cpp").read_text()
    retire = body(src, "auto retire=")
    assert "const auto timings=p.depth.timings();" in retire
    assert "timings.combined_gpu_timing_available" in retire
    assert "gpu_budget.complete(p.capture->frame_id,measured_gpu_ms)" in retire
    assert retire.index("p.output.wait()") < retire.index("gpu_budget.complete(") < retire.index("p.output={};")
    assert "gpu_budget.complete" not in body(src, "while(rclcpp::ok()&&!trace_full")


def test_gpu_budget_exposes_explicit_partial_timing_scope():
    src = (ROOT / "apps/ros2_depth_publisher.cpp").read_text()
    assert 'experimental?"configured_cost_estimate_only":"upload_through_metric_no_host_export"' in src
    assert "gpu_budget.stats().missing_timings" in src
    assert "--gpu-duty-percent" in src
    assert "--gpu-estimate-us" in src
    assert "--gpu-burst-ms" in src
    assert '"budget_rejections",stats.budget_rejections' in src
