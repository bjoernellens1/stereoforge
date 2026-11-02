"""Milestone v9.1 verification tests (Gates 1-5):
- Launch/CLI wiring
- Non-circular accuracy audit (Reference A: 5-scene 1280x720 GT suite on Vulkan GPU; Reference B: 5-frame ZNCC epipolar reference)
- Disjoint cohort decomposition for newly admitted selective_repair pixels
- Live 60 Hz physical ZED 2 camera (--source camera) + concurrent ROS 2 subscriber E2E qualification
- Multi-trial interleaved hardware GPU timestamp & host-wall 240 FPS qualification
- Un-skippable live GPU pytest execution and negative/mutation rejection tests."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.lr_confidence_audit_v6 import read_fixture_luma
from benchmarks.visual_accuracy_audit_v9 import (
    build_independent_zncc_epipolar_reference,
    decompose_selective_repair_cohorts,
)

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
    assert doc["schema"] == "stereoforge.visual_accuracy_and_selective_repair_v9_1.v2"
    assert "consensus_reference_coverage_pct" not in doc, "Circular consensus reference must not be used"

    # Gate 2 Reference A: 5-Scene 1280x720 Known Ground Truth Suite on Native Vulkan GPU
    gt_suite = doc["known_ground_truth_suite_720p"]
    assert gt_suite["scenes_evaluated"] == 5
    agg_gt = gt_suite["aggregate_across_5_scenes"]
    for pname in (
        "4x Refinement (Independent)",
        "4x Selective Repair (Independent)",
        "4x Refinement (Derived Right)",
        "4x Selective Repair (Derived Right)",
    ):
        assert agg_gt[pname]["mean_bad_1px_pct"] <= 0.5, pname
        assert agg_gt[pname]["mean_unconditional_good_1px_completeness_pct"] >= 93.0, pname

    assert (
        agg_gt["4x Selective Repair (Independent)"]["mean_unconditional_good_1px_completeness_pct"]
        > agg_gt["4x Refinement (Independent)"]["mean_unconditional_good_1px_completeness_pct"] + 3.0
    )
    assert (
        agg_gt["4x Selective Repair (Independent)"]["mean_bad_1px_pct"]
        <= agg_gt["4x Refinement (Independent)"]["mean_bad_1px_pct"]
    )
    assert (
        agg_gt["4x Selective Repair (Derived Right)"]["mean_unconditional_good_1px_completeness_pct"]
        > agg_gt["4x Refinement (Derived Right)"]["mean_unconditional_good_1px_completeness_pct"] + 3.0
    )
    assert (
        agg_gt["4x Selective Repair (Derived Right)"]["mean_bad_1px_pct"]
        <= agg_gt["4x Refinement (Derived Right)"]["mean_bad_1px_pct"]
    )

    # Gate 2 Reference B: 5-Frame ZED 2 Independent ZNCC Epipolar Audit
    f5 = doc["five_frame_zed2_zncc_epipolar_audit"]
    assert f5["frames_evaluated"] == 5
    m5 = f5["mean_across_5_zed2_frames"]
    assert m5["4x_ind_rep"]["mean_coverage_pct"] > m5["4x_ind_base"]["mean_coverage_pct"] + 1.0
    assert m5["4x_ind_rep"]["mean_zncc_mae_px"] < m5["4x_ind_base"]["mean_zncc_mae_px"]
    assert m5["4x_ind_rep"]["mean_zncc_bad_1px_pct"] < m5["4x_ind_base"]["mean_zncc_bad_1px_pct"]
    assert m5["4x_der_rep"]["mean_coverage_pct"] > m5["4x_der_base"]["mean_coverage_pct"] + 1.0
    assert m5["4x_der_rep"]["mean_zncc_mae_px"] < m5["4x_der_base"]["mean_zncc_mae_px"]
    assert m5["4x_der_rep"]["mean_zncc_bad_1px_pct"] < m5["4x_der_base"]["mean_zncc_bad_1px_pct"]

    # Gate 5: Newly Admitted Pixel Cohort Decomposition
    cohorts = doc["newly_admitted_pixel_cohorts_gate5"]
    for ckey in ("independent_4x", "derived_4x"):
        c = cohorts[ckey]
        assert c["partition_is_exact_and_disjoint"] is True
        assert c["newly_admitted_total"]["pixels"] > 10000
        assert c["newly_admitted_total"]["bad_1px_pct"] < 8.0
        assert c["newly_admitted_total"]["bad_2px_pct"] < 2.5

    fixed_summary = doc["fixed_regions_summary"]
    assert len(fixed_summary["fixed_planar_20tiles_coords_yxwh"]) == 20
    assert fixed_summary["fixed_planar_20tiles_pixels"] == 20 * 32 * 32

    pipelines = doc["pipelines"]
    for name, p in pipelines.items():
        assert p["fixed_planar_20tiles"]["fixed_tiles_total"] == 20, name
        assert p["fixed_planar_20tiles"]["fixed_tiles_with_min_32px_support"] == 20, name
        assert "all_interior" in p["regions_vs_independent_zncc_ref"]
        assert "fixed_planar_20tiles" in p["regions_vs_independent_zncc_ref"]
        assert "fixed_boundary_bands" in p["regions_vs_independent_zncc_ref"]
        assert "fixed_low_texture_tiles" in p["regions_vs_independent_zncc_ref"]
        assert "fixed_repetitive_texture_tiles" in p["regions_vs_independent_zncc_ref"]
        assert "cross_estimator_diagnostics_non_gt" in p

    eff = doc["selective_repair_efficiency_v9"]
    assert eff["terminology"] == "conditional_per_pixel_selective_repair"
    for variant in ("independent_4x", "derived_4x"):
        v = eff[variant]
        assert v["within_240fps_4_167ms_budget_p50_and_p95"] is True, variant
        assert v["selective_repair_host_wall_p50_ms"] <= 4.167, variant
        assert v["selective_repair_host_wall_p95_ms"] <= 4.167, variant
        assert v["selective_repair_gpu_combined_p50_ms"] <= 4.167, variant
        assert v["selective_repair_gpu_combined_p95_ms"] <= 4.167, variant
        assert v["delta_lr_valid_pct"] > 1.0, variant
        assert v["delta_coherent_coverage_pct"] > 1.5, variant
        assert v["dual_vs_separate_preprocess_30_planes_exact"] is True, variant
        assert v["disparities_differ_from_baseline"] is True, variant

    panels_img = cv2.imread(str(ROOT / doc["visual_artifacts"]["panels_png"]))
    regions_img = cv2.imread(str(ROOT / doc["visual_artifacts"]["regions_overlay_png"]))
    assert panels_img is not None and panels_img.shape[0] > 500 and panels_img.shape[1] > 1000
    assert regions_img is not None and regions_img.shape == (720, 1280, 3)


def test_step7_10_live_and_e2e_qualification_evidence() -> None:
    report_path = ROOT / "docs/evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json"
    assert report_path.exists()
    doc = json.loads(report_path.read_text())
    assert doc["schema"] == "stereoforge.live_and_e2e_qualification_v9_1.v2"
    assert doc["zed2_hardware_probe"]["live_camera_ready"] is True

    # Gate 1: Real Live Physical Camera (--source camera) + Concurrent Subscriber Runs
    live_runs = doc["live_physical_camera_60hz_ros2_e2e_runs"]
    for key in (
        "4x_independent_180f",
        "4x_derived_180f",
        "4x_selective_repair_independent_180f",
        "4x_selective_repair_derived_180f",
    ):
        r = live_runs[key]
        assert r["capture_mode"] == "live_physical_zed2_camera_60hz", key
        assert r["published_depth_frames"] == 180, key
        assert r["published_rgb_frames"] == 180, key
        assert r["received_depth_frames"] >= 165, key
        assert r["received_rgb_frames"] >= 165, key
        assert r["matched_pub_sub_depth_pairs"] == r["received_depth_frames"], key
        assert r["matched_pub_sub_rgb_pairs"] == r["received_rgb_frames"], key
        assert r["unique_source_ids"] == 180, key
        assert r["synchronized_depth_rgb_pairs"] == 180, key
        assert r["stamp_mismatch_count"] == 0, key
        assert r["source_ids_contiguous_no_gaps"] is True, key
        assert r["sequence_gaps"] == 0, key
        assert r["hidden_backlog_or_drops"] == 0, key
        assert r["frame_accounting"]["invalid_subscriber_payloads"] == 0, key
        assert r["frame_accounting"]["subscriber_delivery_ratio_depth"] >= 0.90, key
        assert r["rate_controller_summary"]["final_target_fps"] == 60, key
        assert r["rate_controller_summary"]["slot_rejections"] == 0, key
        assert r["latency_ms"]["ros2_dds_publish_to_subscriber_arrival_ms"] is not None, key
        assert r["latency_ms"]["host_ingress_to_subscriber_arrival_e2e_ms"] is not None, key
        assert r["latency_ms"]["camera_interframe_interval_ms"] is not None, key

    # Replay runs explicitly labeled unpaced_offline_replay
    replay_runs = doc["unpaced_offline_replay_ros2_e2e_runs"]
    for key, r in replay_runs.items():
        assert r["capture_mode"] == "unpaced_offline_replay", key
        assert r["received_depth_frames"] > 0, key
        assert r["matched_pub_sub_depth_pairs"] == r["received_depth_frames"], key

    # Gate 4: Hardware GPU Timestamps + Multi-Trial Interleaved Qualification
    gpu_q = doc["gpu_240fps_e2e_latency_qualification"]
    for key, g in gpu_q.items():
        assert g["meets_240fps_4_167ms_host_p50"] is True, key
        assert g["meets_240fps_4_167ms_gpu_p50"] is True, key
        assert g["dual_vs_separate_preprocess_30_planes_exact"] is True, key
        assert g["validation_errors_after_teardown"] == 0, key
        assert g["validation_warnings_after_teardown"] == 0, key
        assert "gpu_stage_timings_ms" in g, key
        assert "multi_trial_5frame_interleaved" in g, key


@pytest.mark.gpu
def test_v9_gpu_hardware_execution_ortho_hybrid_and_selective_repair(tmp_path: Path) -> None:
    """Gate 3: Un-skippable live Vulkan/Daxa GPU contract and 4x selective-repair qualification."""
    env = dict(os.environ)
    env.update(
        {
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
            "LD_PRELOAD": "/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1",
            "LD_LIBRARY_PATH": str(ROOT / "ros_ws/.pixi/envs/default/lib"),
        }
    )
    contracts_bin = ROOT / "build/ros-daxa-native/daxa_refinement_contracts"
    dual_bin = ROOT / "build/ros-daxa-native/daxa_dual_preprocess_benchmark"
    refine_bin = ROOT / "build/ros-daxa-native/daxa_refinement_benchmark"
    coarse_m = ROOT / "artifacts/depth-video/v8-review-milestone/fixtures/zed2_320x180/manifest.json"
    target_m = ROOT / "configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json"

    assert contracts_bin.exists(), f"Missing {contracts_bin}"
    assert dual_bin.exists(), f"Missing {dual_bin}"
    assert refine_bin.exists(), f"Missing {refine_bin}"
    assert coarse_m.exists(), f"Missing {coarse_m}"
    assert target_m.exists(), f"Missing {target_m}"

    # 1. Execute C++ GPU contract suite (including 1280x720 contract & 3-px plateau rejection)
    subprocess.run([str(contracts_bin), "--gpu"], env=env, check=True)

    # 2. Execute baseline (--selective-repair 0) and selective_repair (--selective-repair 1) on GPU
    base_pref = tmp_path / "ref4x_base"
    rep_pref = tmp_path / "ref4x_rep"
    for pref, rep_flag in ((base_pref, "0"), (rep_pref, "1")):
        subprocess.run(
            [
                str(refine_bin),
                "--coarse-manifest", str(coarse_m),
                "--target-manifest", str(target_m),
                "--variant", "raw_borrow",
                "--frames", "10",
                "--warmup", "2",
                "--coarse-disparities", "32",
                "--coarse-schedule", "layered-per-path",
                "--coarse-right-reference-mode", "independent-sgm",
                "--descriptor-mode", "ortho-hybrid",
                "--fast-mode", "1",
                "--selective-repair", rep_flag,
                "--radius", "2",
                "--subpixel", "1",
                "--dump-prefix", str(pref),
                "--output", str(pref) + ".json",
            ],
            env=env,
            check=True,
        )

    luma_l, luma_r = read_fixture_luma(target_m.parent)
    d_zncc, v_zncc = build_independent_zncc_epipolar_reference(luma_l, luma_r)
    cohorts = decompose_selective_repair_cohorts(base_pref, rep_pref, d_zncc, v_zncc)
    assert cohorts["partition_is_exact_and_disjoint"] is True
    assert cohorts["newly_admitted_total"]["pixels"] > 10000
    assert cohorts["newly_admitted_total"]["bad_1px_pct"] < 8.0
    assert cohorts["newly_admitted_total"]["bad_2px_pct"] < 2.5

    # 3. Negative/mutation check: corrupt 30% of newly admitted pixels by +5.0 px and verify gate catches it
    mut_pref = tmp_path / "ref4x_mutated"
    for suffix in (".json", ".validity.u32", ".lower.u32", ".upper.u32", ".reason.u32"):
        (Path(str(mut_pref) + suffix)).write_bytes((Path(str(rep_pref) + suffix)).read_bytes())
    d_rep_arr = np.fromfile(str(rep_pref) + ".disparity.f32", dtype="<f4").copy()
    v_base_arr = np.fromfile(str(base_pref) + ".validity.u32", dtype="<u4") != 0
    v_rep_arr = np.fromfile(str(rep_pref) + ".validity.u32", dtype="<u4") != 0
    new_idx = np.flatnonzero((~v_base_arr) & v_rep_arr)
    d_rep_arr[new_idx[::2]] += 5.0
    d_rep_arr.tofile(str(mut_pref) + ".disparity.f32")
    mut_cohorts = decompose_selective_repair_cohorts(base_pref, mut_pref, d_zncc, v_zncc)
    assert mut_cohorts["newly_admitted_total"]["bad_1px_pct"] > 25.0, "Mutation gate failed to detect injected bad matches"

    # 4. Execute dual preprocess + selective repair benchmark on GPU and verify 240 FPS budget & 30-plane exactness
    out_json = tmp_path / "gpu_live_dual4x_repair.json"
    subprocess.run(
        [
            str(dual_bin),
            "--coarse-manifest", str(coarse_m),
            "--target-manifest", str(target_m),
            "--coarse-disparities", "32",
            "--descriptor-mode", "ortho-hybrid",
            "--fast-mode", "1",
            "--selective-repair", "1",
            "--radius", "2",
            "--subpixel", "1",
            "--coarse-schedule", "shared-census-sequential",
            "--coarse-right-reference-mode", "left-aggregated-selection",
            "--frames", "60",
            "--warmup", "15",
            "--validation",
            "--output", str(out_json),
        ],
        env=env,
        check=True,
    )
    res = json.loads(out_json.read_text())
    assert res["dual_vs_separate_preprocess_30_planes_exact"] is True
    assert float(res["gpu_combined_queue_span_ms"]["p50"]) <= 4.167
    assert float(res["host_wall_ms"]["p50"]) <= 4.167
    assert res["validation_errors_after_teardown"] == 0
    assert res["validation_warnings_after_teardown"] == 0
