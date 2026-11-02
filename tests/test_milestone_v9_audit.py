"""Milestone v9 verification tests: launch/CLI wiring, fixed-region accuracy audit,
selective 4x refinement budget & parity, and end-to-end ROS 2 delivery qualification."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_v9_publisher_and_launch_cli_wiring() -> None:
    pub_cpp = (ROOT / "apps/ros2_depth_publisher.cpp").read_text()
    for flag in (
        "--descriptor-mode",
        "--fast-mode",
        "--coarse-schedule",
        "--coarse-right-reference-mode",
        "--refinement-subpixel",
        "--selective-repair",
    ):
        assert flag in pub_cpp, f"Missing {flag} in apps/ros2_depth_publisher.cpp"

    for launch_rel in (
        "ros_ws/launch/stereoforge_publisher.launch.py",
        "ros_ws/launch/stereoforge_standard.launch.py",
    ):
        launch_txt = (ROOT / launch_rel).read_text()
        for arg_name in (
            "descriptor_mode",
            "fast_mode",
            "coarse_schedule",
            "coarse_right_reference_mode",
            "refinement_subpixel",
            "selective_repair",
        ):
            assert arg_name in launch_txt, f"Missing {arg_name} in {launch_rel}"


def test_step8_9_visual_accuracy_and_selective_repair_evidence() -> None:
    report_path = ROOT / "docs/evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json"
    assert report_path.exists()
    doc = json.loads(report_path.read_text())
    assert doc["schema"] == "stereoforge.visual_accuracy_and_selective_repair_v9.v1"
    assert doc["consensus_reference_coverage_pct"] > 10.0

    fixed_summary = doc["fixed_regions_summary"]
    assert len(fixed_summary["fixed_planar_20tiles_coords_yxwh"]) == 20
    assert fixed_summary["fixed_planar_20tiles_pixels"] == 20 * 32 * 32

    pipelines = doc["pipelines"]
    for name, p in pipelines.items():
        assert p["fixed_planar_20tiles"]["fixed_tiles_total"] == 20, name
        assert p["fixed_planar_20tiles"]["fixed_tiles_with_min_32px_support"] == 20, name
        assert "all_interior" in p["regions_vs_consensus_ref"]
        assert "fixed_planar_20tiles" in p["regions_vs_consensus_ref"]
        assert "fixed_boundary_bands" in p["regions_vs_consensus_ref"]
        assert "fixed_low_texture_tiles" in p["regions_vs_consensus_ref"]
        assert "fixed_repetitive_texture_tiles" in p["regions_vs_consensus_ref"]

    # Verify Selective Repair improves independent & derived 4x coherent coverage within 240 FPS budget
    eff = doc["selective_repair_efficiency_v9"]
    for variant in ("independent_4x", "derived_4x"):
        v = eff[variant]
        assert v["within_240fps_4_167ms_budget_p50_and_p95"] is True
        assert v["selective_repair_host_wall_p50_ms"] <= 4.167
        assert v["selective_repair_host_wall_p95_ms"] <= 4.167
        assert v["delta_lr_valid_pct"] > 8.0
        assert v["delta_coherent_coverage_pct"] > 10.0
        assert v["all_30_planes_exact_between_variants"] is True

    # Verify visual diagnostic PNG artifacts exist and decode cleanly
    panels_img = cv2.imread(str(ROOT / doc["visual_artifacts"]["panels_png"]))
    regions_img = cv2.imread(str(ROOT / doc["visual_artifacts"]["regions_overlay_png"]))
    assert panels_img is not None and panels_img.shape[0] > 500 and panels_img.shape[1] > 1000
    assert regions_img is not None and regions_img.shape == (720, 1280, 3)


def test_step7_10_live_and_e2e_qualification_evidence() -> None:
    report_path = ROOT / "docs/evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json"
    assert report_path.exists()
    doc = json.loads(report_path.read_text())
    assert doc["schema"] == "stereoforge.live_and_e2e_qualification_v9.v1"

    ros_runs = doc["ros2_e2e_60hz_delivery_runs"]
    for key in (
        "4x_independent_180f",
        "4x_derived_180f",
        "4x_selective_repair_independent_180f",
        "4x_selective_repair_derived_180f",
    ):
        r = ros_runs[key]
        assert r["published_depth_frames"] == 180
        assert r["published_rgb_frames"] == 180
        assert r["unique_source_ids"] == 180
        assert r["synchronized_depth_rgb_pairs"] == 180
        assert r["stamp_mismatch_count"] == 0
        assert r["source_ids_contiguous_no_gaps"] is True
        assert r["hidden_backlog_or_drops"] == 0
        assert r["rate_controller_summary"]["final_target_fps"] == 60
        assert r["rate_controller_summary"]["slot_rejections"] == 0

    gpu_q = doc["gpu_240fps_e2e_latency_qualification"]
    for key, g in gpu_q.items():
        assert g["meets_240fps_4_167ms_p50"] is True, key
        assert g["all_30_planes_exact_between_variants"] is True, key
        assert g["validation_errors_after_teardown"] == 0, key
        assert g["validation_warnings_after_teardown"] == 0, key


@pytest.mark.gpu
def test_v9_gpu_hardware_execution_ortho_hybrid_and_selective_repair(tmp_path: Path) -> None:
    """Execute native Vulkan/Daxa GPU contract and 4x selective-repair benchmark on hardware."""
    env = dict(os.environ)
    env.update(
        {
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
            "LD_PRELOAD": "/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1",
            "LD_LIBRARY_PATH": str(ROOT / "ros_ws/.pixi/envs/default/lib"),
        }
    )
    contracts_bin = ROOT / "build/ros-daxa-native/daxa_refinement_contracts"
    subprocess.run([str(contracts_bin), "--gpu"], env=env, check=True)

    coarse_m = ROOT / "artifacts/depth-video/v8-review-milestone/fixtures/zed2_320x180/manifest.json"
    if coarse_m.exists():
        out_json = tmp_path / "gpu_live_dual4x_repair.json"
        dual_bin = ROOT / "build/ros-daxa-native/daxa_dual_preprocess_benchmark"
        subprocess.run(
            [
                str(dual_bin),
                "--coarse-manifest",
                str(coarse_m),
                "--target-manifest",
                str(ROOT / "configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json"),
                "--coarse-disparities",
                "32",
                "--descriptor-mode",
                "ortho-hybrid",
                "--fast-mode",
                "1",
                "--selective-repair",
                "1",
                "--coarse-schedule",
                "shared-census-sequential",
                "--coarse-right-reference-mode",
                "left-aggregated-selection",
                "--frames",
                "10",
                "--warmup",
                "3",
                "--validation",
                "--output",
                str(out_json),
            ],
            env=env,
            check=True,
        )
        res = json.loads(out_json.read_text())
        assert res["all_30_planes_exact_between_variants"] is True
        assert res["validation_errors_after_teardown"] == 0
        assert res["validation_warnings_after_teardown"] == 0

