"""Milestone v9.2 verification tests (Gates 1-5):
- Launch/CLI wiring (including --qos-depth and --verify-payload-checksum)
- Controlled unit tests for ZED 2 MCU VID:PID verification and recovery
- Non-circular accuracy audit (Reference A: 5-scene 1280x720 GT suite on Vulkan GPU; Reference B: 5-frame localized 7x7+9x9 ZNCC epipolar reference)
- Disjoint cohort decomposition for newly admitted selective_repair pixels with explicit null on zero reference support + 100%-coverage dense epipolar reprojection audit
- Live 60 Hz physical ZED 2 camera (--source camera) + concurrent ROS 2 subscriber E2E qualification (KeepLast(16) lossless 180f & 600f endurance, KeepLast(1) latest-frame, shared-GPU contention, and 64-bit FNV-1a payload checksums)
- Multi-trial interleaved hardware GPU single-interval (refinement.end_tick - dual_preprocess.start_tick) & stage-sum timestamp + host-wall 240 FPS qualification
- Programmatic Markdown-to-JSON equivalence check
- Un-skippable live GPU pytest execution and negative/mutation rejection tests."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.live_and_e2e_qualification_v9 import (
    ensure_zed2_camera_active,
    find_verified_zed2_hidraw,
    render_milestone_v9_markdown_from_json,
)
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
        "--planar-recovery",
        "--completed-depth-topic",
        "--qos-depth",
        "--verify-payload-checksum",
    ):
        assert flag in pub_cpp, f"Missing {flag} in apps/ros2_depth_publisher.cpp"

    probe_cpp = (ROOT / "apps/ros2_delivery_probe.cpp").read_text()
    for token in ("--qos-depth", "fnv1a64_words", "valid_pixels"):
        assert token in probe_cpp, f"Missing {token} in apps/ros2_delivery_probe.cpp"

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
            "planar_recovery",
            "completed_depth_topic",
        ):
            assert arg_name in launch_txt, f"Missing {arg_name} in {launch_rel}"


def test_zed2_hidraw_vid_pid_verification_and_controlled_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Verify C++ driver sources enforce single-device 2B03:F780 VID:PID check and active-stream timeout recovery
    for cpp_rel in ("src/capture.cpp", "src/zed_capture.cpp"):
        cpp_txt = (ROOT / cpp_rel).read_text()
        assert "2B03:0000F780" in cpp_txt, f"Missing VID:PID check in {cpp_rel}"
        assert "resetSensorModule" in cpp_txt, f"Missing resetSensorModule in {cpp_rel}"
        assert ">= 2000" in cpp_txt or ">=2000" in cpp_txt, f"Missing bounded timeout recovery in {cpp_rel}"

    # 2. Controlled sysfs tests for find_verified_zed2_hidraw and ensure_zed2_camera_active
    sysfs_root = tmp_path / "sys_class_hidraw"
    dev_root = tmp_path / "dev"
    sysfs_root.mkdir()
    dev_root.mkdir()
    (dev_root / "hidraw0").write_bytes(b"")
    (dev_root / "hidraw2").write_bytes(b"")
    (dev_root / "hidraw3").write_bytes(b"")

    # Case A: Zero matching ZED 2 devices (only unrelated USB mouse 046D:C52B)
    h0 = sysfs_root / "hidraw0" / "device"
    h0.mkdir(parents=True)
    (h0 / "uevent").write_text("DRIVER=hid-generic\nHID_ID=0003:0000046D:0000C52B\n")
    assert find_verified_zed2_hidraw(sysfs_root, dev_root) is None
    res_zero = ensure_zed2_camera_active(force_reset=True, sysfs_hidraw_root=sysfs_root, dev_root=dev_root)
    assert res_zero["status"] == "error"
    assert res_zero["action"] == "refused_no_verified_single_zed2_hidraw"

    # Case B: Exactly one matching ZED 2 MCU (2B03:F780)
    h2 = sysfs_root / "hidraw2" / "device"
    h2.mkdir(parents=True)
    (h2 / "uevent").write_text("DRIVER=hid-generic\nHID_ID=0003:00002B03:0000F780\n")
    assert find_verified_zed2_hidraw(sysfs_root, dev_root) == dev_root / "hidraw2"

    # Case C: Multiple matching ZED 2 MCUs -> refuses ambiguous reset
    h3 = sysfs_root / "hidraw3" / "device"
    h3.mkdir(parents=True)
    (h3 / "uevent").write_text("DRIVER=hid-generic\nHID_ID=0003:00002B03:0000F780\n")
    assert find_verified_zed2_hidraw(sysfs_root, dev_root) is None
    res_multi = ensure_zed2_camera_active(force_reset=True, sysfs_hidraw_root=sysfs_root, dev_root=dev_root)
    assert res_multi["status"] == "error"
    assert res_multi["action"] == "refused_no_verified_single_zed2_hidraw"

    # Remove duplicate h3 so single verified device remains
    (h3 / "uevent").unlink()
    h3.rmdir()
    (sysfs_root / "hidraw3").rmdir()

    # Case D: When hidraw open raises PermissionError, error status is surfaced without silent suppression
    def fake_open_perm_error(*_args: object, **_kwargs: object) -> int:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "open", fake_open_perm_error)
    res_perm = ensure_zed2_camera_active(force_reset=True, sysfs_hidraw_root=sysfs_root, dev_root=dev_root)
    assert res_perm["status"] == "error"
    assert res_perm["action"] == "hidraw_open_failed"
    assert res_perm["errno"] == 13


def test_zero_reference_support_cohort_returns_null_not_zero(tmp_path: Path) -> None:
    h, w = 64, 64
    base_pref = tmp_path / "syn_base"
    rep_pref = tmp_path / "syn_rep"

    v_base = np.zeros((2, h, w), dtype=np.uint32)
    v_base[0, 10:30, 10:30] = 1
    v_rep = v_base.copy()
    v_rep[0, 30:45, 20:40] = 1  # 300 newly admitted pixels

    d_base = np.full((2, h, w), 12.0, dtype=np.float32)
    d_rep = np.full((2, h, w), 12.0, dtype=np.float32)
    lower = np.full((2, h, w), 10, dtype=np.uint32)
    upper = np.full((2, h, w), 14, dtype=np.uint32)
    reason_base = np.where(v_base == 1, 1, 32).astype(np.uint32)
    reason_rep = np.where(v_rep == 1, 1, 32).astype(np.uint32)

    for pref, v_arr, d_arr, r_arr in (
        (base_pref, v_base, d_base, reason_base),
        (rep_pref, v_rep, d_rep, reason_rep),
    ):
        Path(str(pref) + ".json").write_text(json.dumps({"width": w, "height": h}))
        v_arr.tofile(str(pref) + ".validity.u32")
        d_arr.tofile(str(pref) + ".disparity.f32")
        lower.tofile(str(pref) + ".lower.u32")
        upper.tofile(str(pref) + ".upper.u32")
        r_arr.tofile(str(pref) + ".reason.u32")

    # Zero ZNCC reference support everywhere
    d_zncc_empty = np.zeros((h, w), dtype=np.float32)
    v_zncc_empty = np.zeros((h, w), dtype=bool)
    rng = np.random.default_rng(42)
    luma_r = rng.uniform(40, 200, size=(h, w)).astype(np.float32)
    luma_l = np.roll(luma_r, 12, axis=1)

    res = decompose_selective_repair_cohorts(
        base_pref, rep_pref, d_zncc_empty, v_zncc_empty, luma_l=luma_l, luma_r=luma_r
    )
    na = res["newly_admitted_total"]
    assert na["pixels"] == 300
    assert na["joint_independent_ref_pixels"] == 0
    assert na["disparity_mae_px"] is None
    assert na["bad_1px_pct"] is None
    assert na["bad_2px_pct"] is None
    assert na["evaluation_status"] == "not_evaluated_zero_reference_support"
    assert na["dense_epipolar_evaluated_pixels"] == 300
    assert na["dense_photometric_reproj_mae_luma"] == pytest.approx(0.0, abs=1e-4)
    assert na["dense_patch_zncc_5x5_mean"] > 0.95
    assert na["dense_local_3x3_planar_inlier_pct"] == pytest.approx(100.0, abs=1e-4)


def test_step8_9_visual_accuracy_and_selective_repair_evidence() -> None:
    report_path = ROOT / "docs/evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json"
    assert report_path.exists()
    doc = json.loads(report_path.read_text())
    assert doc["schema"] == "stereoforge.visual_accuracy_and_selective_repair_v9_2.v3"
    assert "consensus_reference_coverage_pct" not in doc, "Circular consensus reference must not be used"
    assert doc["provenance"]["compiled_spv_count"] >= 20
    assert len(doc["provenance"]["git_head_at_generation"]) == 40

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

    # Gate 2 Reference B: 5-Frame ZED 2 Localized Multi-Scale (7x7 + 9x9) ZNCC Epipolar Audit
    f5 = doc["five_frame_zed2_zncc_epipolar_audit"]
    assert f5["frames_evaluated"] == 5
    m5 = f5["mean_across_5_zed2_frames"]
    assert m5["4x_ind_rep"]["mean_coverage_pct"] > m5["4x_ind_base"]["mean_coverage_pct"] + 1.0
    assert m5["4x_ind_rep"]["mean_zncc_mae_px"] < m5["4x_ind_base"]["mean_zncc_mae_px"]
    assert m5["4x_ind_rep"]["mean_zncc_bad_1px_pct"] < m5["4x_ind_base"]["mean_zncc_bad_1px_pct"]
    assert m5["4x_der_rep"]["mean_coverage_pct"] > m5["4x_der_base"]["mean_coverage_pct"] + 1.0
    assert m5["4x_der_rep"]["mean_zncc_mae_px"] < m5["4x_der_base"]["mean_zncc_mae_px"]
    assert m5["4x_der_rep"]["mean_zncc_bad_1px_pct"] < m5["4x_der_base"]["mean_zncc_bad_1px_pct"]

    # Gate 5: Newly Admitted Pixel Cohort Decomposition & Dense Epipolar Reprojection
    cohorts = doc["newly_admitted_pixel_cohorts_gate5"]
    for ckey in ("independent_4x", "derived_4x"):
        c = cohorts[ckey]
        assert c["partition_is_exact_and_disjoint"] is True
        na = c["newly_admitted_total"]
        bv = c["baseline_valid"]
        assert na["pixels"] > 10000
        assert na["bad_1px_pct"] < 12.5
        assert na["bad_2px_pct"] < 4.0
        assert na["dense_epipolar_evaluated_pixels"] > 16000
        assert na["dense_photometric_reproj_mae_luma"] < bv["dense_photometric_reproj_mae_luma"]
        assert na["dense_patch_zncc_5x5_mean"] > bv["dense_patch_zncc_5x5_mean"]
        assert na["dense_local_3x3_planar_inlier_pct"] > bv["dense_local_3x3_planar_inlier_pct"]

    # Zero-reference-support cohort must explicitly report None (null) rather than 0.0
    der_ca = cohorts["derived_4x"]["newly_admitted_by_mechanism"]["cohort_a_recovered_coarse_hole_or_span"]
    assert der_ca["joint_independent_ref_pixels"] == 0
    assert der_ca["disparity_mae_px"] is None
    assert der_ca["bad_1px_pct"] is None
    assert der_ca["bad_2px_pct"] is None
    assert der_ca["evaluation_status"] == "not_evaluated_zero_reference_support"
    assert der_ca["dense_epipolar_evaluated_pixels"] >= 100

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
        assert v["selective_repair_single_interval_upload_to_metric_gpu_p50_ms"] <= 4.167, variant
        assert v["selective_repair_single_interval_upload_to_metric_gpu_p95_ms"] <= 4.167, variant
        assert v["selective_repair_stage_sum_gpu_p50_ms"] <= 4.167, variant
        assert v["selective_repair_stage_sum_gpu_p95_ms"] <= 4.167, variant
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


def test_step7_10_live_and_e2e_qualification_evidence_and_markdown_sync() -> None:
    report_path = ROOT / "docs/evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json"
    acc_path = ROOT / "docs/evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json"
    md_path = ROOT / "docs/milestone-v9-live-and-selective-refinement-evidence.md"
    assert report_path.exists()
    assert acc_path.exists()
    assert md_path.exists()

    doc = json.loads(report_path.read_text())
    acc_doc = json.loads(acc_path.read_text())
    assert doc["schema"] == "stereoforge.live_and_e2e_qualification_v9_2.v3"
    assert doc["provenance"]["compiled_spv_count"] >= 20
    assert doc["provenance"]["git_head_at_generation"] == acc_doc["provenance"]["git_head_at_generation"]
    assert doc["zed2_hardware_probe"]["live_camera_ready"] is True

    # Programmatic Markdown-to-JSON zero-drift check
    expected_md = render_milestone_v9_markdown_from_json(doc, acc_doc)
    assert md_path.read_text() == expected_md

    # Gate 1: Primary Lossless KeepLast(16) Live Physical Camera (--source camera) Runs
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
        assert r["received_depth_frames"] == 180, key
        assert r["received_rgb_frames"] >= 179, key
        assert r["matched_pub_sub_depth_pairs"] == 180, key
        assert r["matched_pub_sub_rgb_pairs"] == r["received_rgb_frames"], key
        assert r["unique_source_ids"] == 180, key
        assert r["synchronized_depth_rgb_pairs"] == 180, key
        assert r["stamp_mismatch_count"] == 0, key
        assert r["source_ids_contiguous_no_gaps"] is True, key
        assert r["sequence_gaps"] == 0, key
        assert r["hidden_backlog_or_drops"] == 0, key
        fa = r["frame_accounting"]
        assert fa["qos_depth"] == 16, key
        assert fa["subscriber_delivery_ratio_depth"] == 1.0, key
        assert fa["verified_payload_checksum_matches_depth"] == 180, key
        assert fa["verified_payload_checksum_mismatches_depth"] == 0, key
        assert fa["verified_payload_checksum_matches_rgb"] == r["received_rgb_frames"], key
        assert fa["verified_payload_checksum_mismatches_rgb"] == 0, key
        assert fa["invalid_subscriber_payloads"] == 0, key
        assert r["rate_controller_summary"]["final_target_fps"] == 60, key
        assert r["rate_controller_summary"]["slot_rejections"] == 0, key
        assert r["latency_ms"]["ros2_dds_publish_to_subscriber_arrival_ms"] is not None, key
        assert r["latency_ms"]["host_ingress_to_subscriber_arrival_e2e_ms"] is not None, key
        assert r["latency_ms"]["camera_interframe_interval_ms"] is not None, key

    # Gate 1 Extended Regimes: KeepLast(1) latest-frame, 600-frame (10.0 s) endurance, and shared-GPU contention
    qos_stress = doc["live_camera_qos_and_stress_runs"]
    q1 = qos_stress["keep_last_1_latest_frame_180f"]
    assert q1["frame_accounting"]["qos_depth"] == 1
    assert q1["frame_accounting"]["verified_payload_checksum_matches_depth"] == q1["received_depth_frames"]
    assert q1["frame_accounting"]["verified_payload_checksum_mismatches_depth"] == 0
    assert q1["frame_accounting"]["invalid_subscriber_payloads"] == 0

    endurance = qos_stress["extended_600f_10s_endurance"]
    assert endurance["published_depth_frames"] == 600
    assert endurance["received_depth_frames"] == 600
    assert endurance["received_rgb_frames"] == 600
    assert endurance["frame_accounting"]["qos_depth"] == 16
    assert endurance["frame_accounting"]["subscriber_delivery_ratio_depth"] == 1.0
    assert endurance["frame_accounting"]["verified_payload_checksum_matches_depth"] == 600
    assert endurance["frame_accounting"]["verified_payload_checksum_mismatches_depth"] == 0
    assert endurance["hidden_backlog_or_drops"] == 0

    contention = qos_stress["shared_gpu_contention_180f"]
    assert contention["published_depth_frames"] == 180
    assert contention["received_depth_frames"] == 180
    assert contention["frame_accounting"]["qos_depth"] == 16
    assert contention["frame_accounting"]["subscriber_delivery_ratio_depth"] == 1.0
    assert contention["frame_accounting"]["verified_payload_checksum_matches_depth"] == 180
    assert contention["frame_accounting"]["verified_payload_checksum_mismatches_depth"] == 0
    assert contention["hidden_backlog_or_drops"] == 0

    # Replay runs explicitly labeled unpaced_offline_replay
    replay_runs = doc["unpaced_offline_replay_ros2_e2e_runs"]
    for key, r in replay_runs.items():
        assert r["capture_mode"] == "unpaced_offline_replay", key
        assert r["received_depth_frames"] > 0, key
        assert r["matched_pub_sub_depth_pairs"] == r["received_depth_frames"], key
        assert r["frame_accounting"]["verified_payload_checksum_matches_depth"] == r["received_depth_frames"], key
        assert r["frame_accounting"]["verified_payload_checksum_mismatches_depth"] == 0, key

    # Gate 4: Hardware GPU Single-Interval & Stage-Sum Timestamps + Multi-Trial Interleaved Qualification
    gpu_q = doc["gpu_240fps_e2e_latency_qualification"]
    for key, g in gpu_q.items():
        assert g["meets_240fps_4_167ms_host_p50"] is True, key
        assert g["meets_240fps_4_167ms_gpu_p50"] is True, key
        assert g["meets_240fps_4_167ms_gpu_p95"] is True, key
        assert g["gpu_single_interval_upload_to_metric_ms"]["p50"] <= 4.167, key
        assert g["gpu_stage_sum_ms"]["p50"] <= 4.167, key
        assert g["dual_vs_separate_preprocess_30_planes_exact"] is True, key
        assert g["validation_errors_after_teardown"] == 0, key
        assert g["validation_warnings_after_teardown"] == 0, key
        assert "gpu_single_interval_upload_to_metric_ms" in g, key
        assert "gpu_stage_sum_ms" in g, key
        assert "gpu_stage_timings_ms" in g, key
        mt = g["multi_trial_5frame_interleaved"]
        assert "gpu_single_interval_upload_to_metric_ms" in mt, key
        assert "gpu_stage_sum_ms" in mt, key
        assert mt["gpu_single_interval_upload_to_metric_ms"]["p50"] <= 4.167, key
        assert mt["gpu_stage_sum_ms"]["p50"] <= 4.167, key


@pytest.mark.gpu
def test_v9_gpu_hardware_execution_ortho_hybrid_and_selective_repair(tmp_path: Path) -> None:
    """Gate 3: Un-skippable live Vulkan/Daxa GPU contract and 4x selective-repair qualification."""
    env = dict(os.environ)
    env.update(
        {
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
            "VK_LAYER_PATH": str(ROOT / "ros_ws/.pixi/envs/default/share/vulkan/explicit_layer.d"),
            "LD_PRELOAD": "/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1",
            "LD_LIBRARY_PATH": str(ROOT / "ros_ws/.pixi/envs/default/lib"),
        }
    )
    contracts_bin = ROOT / "build/ros-daxa-native/daxa_refinement_contracts"
    dual_contracts_bin = ROOT / "build/ros-daxa-native/daxa_dual_preprocess_contracts"
    dual_bin = ROOT / "build/ros-daxa-native/daxa_dual_preprocess_benchmark"
    refine_bin = ROOT / "build/ros-daxa-native/daxa_refinement_benchmark"
    coarse_m = ROOT / "artifacts/depth-video/v8-review-milestone/fixtures/zed2_320x180/manifest.json"
    target_m = ROOT / "configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json"

    assert contracts_bin.exists(), f"Missing {contracts_bin}"
    assert dual_contracts_bin.exists(), f"Missing {dual_contracts_bin}"
    assert dual_bin.exists(), f"Missing {dual_bin}"
    assert refine_bin.exists(), f"Missing {refine_bin}"
    assert coarse_m.exists(), f"Missing {coarse_m}"
    assert target_m.exists(), f"Missing {target_m}"

    # 1. Execute C++ GPU contract suites (refinement & dual preprocess with Khronos validation)
    subprocess.run([str(contracts_bin), "--gpu"], env=env, check=True)
    subprocess.run([str(dual_contracts_bin), "--gpu", "--validation"], env=env, check=True)

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
    cohorts = decompose_selective_repair_cohorts(base_pref, rep_pref, d_zncc, v_zncc, luma_l=luma_l, luma_r=luma_r)
    assert cohorts["partition_is_exact_and_disjoint"] is True
    na = cohorts["newly_admitted_total"]
    bv = cohorts["baseline_valid"]
    assert na["pixels"] > 10000
    assert na["bad_1px_pct"] < 12.5
    assert na["bad_2px_pct"] < 4.0
    assert na["dense_epipolar_evaluated_pixels"] > 16000
    assert na["dense_photometric_reproj_mae_luma"] < bv["dense_photometric_reproj_mae_luma"]
    assert na["dense_patch_zncc_5x5_mean"] > bv["dense_patch_zncc_5x5_mean"]
    assert na["dense_local_3x3_planar_inlier_pct"] > bv["dense_local_3x3_planar_inlier_pct"]

    # 3. Negative/mutation check: corrupt 50% of newly admitted pixels by +5.0 px and verify both ZNCC & dense epipolar gates catch it
    mut_pref = tmp_path / "ref4x_mutated"
    for suffix in (".json", ".validity.u32", ".lower.u32", ".upper.u32", ".reason.u32"):
        (Path(str(mut_pref) + suffix)).write_bytes((Path(str(rep_pref) + suffix)).read_bytes())
    d_rep_arr = np.fromfile(str(rep_pref) + ".disparity.f32", dtype="<f4").copy()
    v_base_arr = np.fromfile(str(base_pref) + ".validity.u32", dtype="<u4") != 0
    v_rep_arr = np.fromfile(str(rep_pref) + ".validity.u32", dtype="<u4") != 0
    new_idx = np.flatnonzero((~v_base_arr) & v_rep_arr)
    d_rep_arr[new_idx[::2]] += 5.0
    d_rep_arr.tofile(str(mut_pref) + ".disparity.f32")
    mut_cohorts = decompose_selective_repair_cohorts(base_pref, mut_pref, d_zncc, v_zncc, luma_l=luma_l, luma_r=luma_r)
    assert mut_cohorts["newly_admitted_total"]["bad_1px_pct"] > 25.0, "Mutation gate failed to detect injected bad matches"
    assert (
        mut_cohorts["newly_admitted_total"]["dense_photometric_reproj_mae_luma"]
        > na["dense_photometric_reproj_mae_luma"] + 1.5
    ), "Dense epipolar reprojection gate failed to detect injected bad matches"

    # 4. Execute dual preprocess + selective repair benchmark on GPU and verify single-interval & stage-sum 240 FPS budget + 30-plane exactness
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
    assert res["diagnostic_readback"]["producer_output_preserved"] is True
    assert res["diagnostic_readback"]["fast_output_equivalence_checked"] is True
    assert float(res["gpu_combined_queue_span_ms"]["p50"]) <= 4.167
    assert float(res["gpu_stage_sum_ms"]["p50"]) <= 4.167
    assert float(res["host_wall_ms"]["p50"]) <= 4.167
    assert float(res["gpu_combined_queue_span_ms"]["p50"]) >= float(res["gpu_stage_sum_ms"]["p50"])
    assert res["validation_errors_after_teardown"] == 0
    assert res["validation_warnings_after_teardown"] == 0
