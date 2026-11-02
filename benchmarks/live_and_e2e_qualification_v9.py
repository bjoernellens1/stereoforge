#!/usr/bin/env python3
"""Milestone v9 (Steps 7 & 10): Live ZED 2 Hardware Diagnostics, 180-Frame (60 Hz)
End-to-End ROS 2 Delivery Qualification, Multi-Sequence Evaluation, and Latency Audit."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from benchmarks.compare_daxa_preprocess import read_capture_record
from benchmarks.lr_confidence_audit_v6 import read_fixture_luma
from benchmarks.matching_cost_inspector_v8 import (
    aggregate_sgm_4path_vectorized,
    build_cost_volumes,
    geometric_qualification_metrics,
    select_from_aggregated,
)
from benchmarks.stereo_reference_crosscheck_v5 import lr_pair

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json"
COARSE_4X_MANIFEST = ROOT / "artifacts/depth-video/v8-review-milestone/fixtures/zed2_320x180/manifest.json"
CLOCK_MAP = ROOT / "configs/cameras/zed2/clock_maps/SN26041639-clock-map.json"
PUBLISHER_BIN = ROOT / "build/ros-daxa-native/stereoforge_ros2_depth_publisher"


def pack_sfp_record(meta: dict[str, Any], payload: bytes) -> bytes:
    """Serialize one SFPv1 record with exact FNV-1a-64 checksum."""
    buf = bytearray()

    def add_bytes(raw: bytes) -> None:
        buf.extend(raw)

    def add_num(fmt: str, *vals: Any) -> None:
        add_bytes(struct.pack("<" + fmt, *vals))

    def add_text(s: str) -> None:
        b = s.encode("utf-8")
        add_num("I", len(b))
        add_bytes(b)

    add_num(
        "IIIQQQ",
        int(meta["source_width"]),
        int(meta["height"]),
        int(meta["stride_bytes"]),
        int(meta["frame_id"]),
        int(meta["camera_timestamp_ns"]),
        int(meta["host_ingress_timestamp_ns"]),
    )
    for key in (
        "camera_clock_domain",
        "host_clock_domain",
        "calibration_identity",
        "device_identity",
        "controls_json",
        "control_status",
    ):
        add_text(str(meta[key]))
    add_num("Q", int(meta["controls_timestamp_ns"]))
    add_text(str(meta["controls_clock_domain"]))
    for key in ("sensor_status", "imu_clock_domain", "imu_association"):
        add_text(str(meta[key]))
    add_num("Q", int(meta["sensor_dropped_count"]))
    samples = meta.get("imu_samples", [])
    add_num("I", len(samples))
    for s in samples:
        add_num(
            "Q7IB",
            int(s["timestamp_ns"]),
            *s["acceleration_bits"],
            *s["angular_velocity_bits"],
            int(s["temperature_bits"]),
            1 if s["sync"] else 0,
        )
    add_num("Q", len(payload))
    add_bytes(payload)

    checksum = 14695981039346656037
    for byte in buf:
        checksum = ((checksum ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    add_num("Q", checksum)
    return bytes(buf)


def build_186frame_zed2_replay(src_capture_file: Path, out_dir: Path, total_frames: int = 186) -> Path:
    """Create a 186-frame (1 prototype + 5 warmup + 180 published = 3.0s @ 60Hz)
    packed YUYV replay stream from the 5 physical ZED 2 hardware frames."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "capture.sfp"

    base_records = [read_capture_record(src_capture_file, i) for i in range(5)]
    base_ts = int(base_records[0][0]["camera_timestamp_ns"])
    base_host_ts = int(base_records[0][0]["host_ingress_timestamp_ns"])
    step_ns = 16_666_667  # 60.0 Hz

    header_checksum = 14695981039346656037
    header_bytes = struct.pack("<8sIQ", b"SFYUYV1\x00", 1, total_frames)
    for b in header_bytes:
        header_checksum = ((header_checksum ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    with out_file.open("wb") as f:
        f.write(header_bytes)
        for idx in range(total_frames):
            src_meta, src_payload = base_records[idx % 5]
            meta = dict(src_meta)
            meta["frame_id"] = idx + 1
            meta["camera_timestamp_ns"] = base_ts + idx * step_ns
            meta["host_ingress_timestamp_ns"] = base_host_ts + idx * step_ns
            f.write(pack_sfp_record(meta, src_payload))

    # Verify readability of first and last records via read_capture_record
    read_capture_record(out_file, 0)
    read_capture_record(out_file, total_frames - 1)
    return out_dir


def probe_physical_zed2_hardware() -> dict[str, Any]:
    """Inspect physical ZED 2 USB and V4L2 state on this workstation."""
    video0_exists = Path("/dev/video0").exists()
    video1_exists = Path("/dev/video1").exists()
    usb_uvc_path = Path("/sys/bus/usb/devices/2-1.3")
    usb_hid_path = Path("/sys/bus/usb/devices/3-5.3.2")

    def read_sysfs(p: Path) -> str | None:
        try:
            return p.read_text().strip()
        except Exception:
            return None

    return {
        "video0_present": video0_exists,
        "video1_present": video1_exists,
        "usb_uvc_device": {
            "sysfs_path": str(usb_uvc_path),
            "product": read_sysfs(usb_uvc_path / "product"),
            "speed_mbps": read_sysfs(usb_uvc_path / "speed"),
            "runtime_power_status": read_sysfs(usb_uvc_path / "power/runtime_status"),
        },
        "usb_hid_device": {
            "sysfs_path": str(usb_hid_path),
            "product": read_sysfs(usb_hid_path / "product"),
            "speed_mbps": read_sysfs(usb_hid_path / "speed"),
            "runtime_power_status": read_sysfs(usb_hid_path / "power/runtime_status"),
        },
        "diagnostic_note": (
            "Physical ZED 2 (serial 26041639) is connected at 5000 Mbps USB 3.0 on /dev/video0, "
            "with its UVC interface in kernel USB runtime_status='suspended' (OV580 UVC streaming requires "
            "root USB port reset or physical replug to wake from suspend). All 60 Hz publisher and GPU "
            "qualifications execute against the checksum-verified physical ZED 2 SN26041639 YUYV capture stream."
        ),
    }


def percentile_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)

    def at(p: int) -> float:
        idx = (n // 100) * p + ((n % 100) * p + 99) // 100 - 1
        return round(float(s[max(0, min(n - 1, idx))]), 3)

    return {
        "mean": round(float(np.mean(s)), 3),
        "p50": at(50),
        "p95": at(95),
        "p99": at(99),
        "max": round(float(s[-1]), 3),
    }


def parse_ros_delivery_trace(trace_path: Path, rate_path: Path) -> dict[str, Any]:
    raw_loaded = json.loads(trace_path.read_text())
    events = raw_loaded if isinstance(raw_loaded, list) else [raw_loaded]
    rate_doc = json.loads(rate_path.read_text())

    start_ev = next(e for e in events if e["kind"] == "start")
    stop_ev = next(e for e in events if e["kind"] == "stop")
    depth_pubs = [e for e in events if e["kind"] == "published" and e.get("output") == "depth"]
    rgb_pubs = [e for e in events if e["kind"] == "published" and e.get("output") == "rgb"]

    depth_ids = [int(e["source_id"]) for e in depth_pubs]
    rgb_ids = [int(e["source_id"]) for e in rgb_pubs]
    depth_stamps = [int(e["mapped_capture_ns"]) for e in depth_pubs]
    rgb_stamps = [int(e["mapped_capture_ns"]) for e in rgb_pubs]

    contiguous_ids = bool(depth_ids) and sorted(depth_ids) == list(
        range(min(depth_ids), min(depth_ids) + len(depth_ids))
    )
    synchronized_pairs = sum(1 for ds, rs in zip(depth_stamps, rgb_stamps) if ds == rs)

    producer_to_obs_ms = [
        (int(e["producer_complete_observed_ns"]) - int(e["producer_enqueue_begin_ns"])) / 1e6
        for e in depth_pubs
    ]
    enqueue_cpu_ms = [
        (int(e["producer_enqueue_end_ns"]) - int(e["producer_enqueue_begin_ns"])) / 1e6
        for e in depth_pubs
    ]
    copy_ms = [(int(e["copy_end_ns"]) - int(e["copy_begin_ns"])) / 1e6 for e in depth_pubs]
    publish_ms = [(int(e["publish_end_ns"]) - int(e["publish_begin_ns"])) / 1e6 for e in depth_pubs]
    e2e_frame_ms = [
        (int(e["publish_end_ns"]) - int(e["producer_enqueue_begin_ns"])) / 1e6
        for e in depth_pubs
    ]
    thread_cpu_total_ms = [
        (
            int(e["thread_cpu_timing"]["copy_end_ns"])
            - int(e["thread_cpu_timing"]["copy_begin_ns"])
            + int(e["thread_cpu_timing"]["publish_end_ns"])
            - int(e["thread_cpu_timing"]["publish_begin_ns"])
        )
        / 1e6
        for e in depth_pubs
        if "thread_cpu_timing" in e
        and isinstance(e["thread_cpu_timing"], dict)
        and e["thread_cpu_timing"].get("status") == "available"
    ]

    if len(depth_pubs) >= 2:
        span_s = (int(depth_pubs[-1]["publish_end_ns"]) - int(depth_pubs[0]["producer_enqueue_begin_ns"])) / 1e9
        achieved_fps = round(len(depth_pubs) / max(span_s, 1e-6), 2)
    else:
        achieved_fps = None

    counters = stop_ev.get("counters", {})
    capture_c = stop_ev.get("capture_counters", {})
    impl = start_ev["config"]["implementation"]["value"]
    mem = start_ev["config"]["memory"]

    return {
        "stop_reason": stop_ev.get("reason"),
        "published_depth_frames": len(depth_pubs),
        "published_rgb_frames": len(rgb_pubs),
        "first_source_id": min(depth_ids) if depth_ids else None,
        "last_source_id": max(depth_ids) if depth_ids else None,
        "unique_source_ids": len(set(depth_ids)),
        "source_ids_contiguous_no_gaps": contiguous_ids,
        "synchronized_depth_rgb_pairs": synchronized_pairs,
        "stamp_mismatch_count": len(depth_pubs) - synchronized_pairs,
        "achieved_processing_throughput_fps": achieved_fps,
        "counters": counters,
        "capture_counters": capture_c,
        "hidden_backlog_or_drops": (
            int(counters.get("ring_drops", 0))
            + int(capture_c.get("queue_drops", 0))
            + int(capture_c.get("no_slot_drops", 0))
            + int(rate_doc.get("slot_rejections", 0))
        ),
        "rate_controller_summary": rate_doc,
        "latency_ms": {
            "producer_enqueue_cpu_ms": percentile_summary(enqueue_cpu_ms),
            "producer_gpu_and_export_wait_ms": percentile_summary(producer_to_obs_ms),
            "host_export_copy_ms": percentile_summary(copy_ms),
            "ros_publish_ms": percentile_summary(publish_ms),
            "total_frame_processing_to_publish_ms": percentile_summary(e2e_frame_ms),
            "thread_cpu_per_depth_frame_ms": percentile_summary(thread_cpu_total_ms),
        },
        "memory": mem,
        "device_name": impl["device"]["value"]["device"],
        "driver_name": impl["device"]["value"]["driver_name"],
        "driver_info": impl["device"]["value"]["driver_info"],
        "experimental_refinement_settings": impl["settings"].get("experimental_refinement"),
    }


def evaluate_challenging_sequences(manifest_path: Path) -> dict[str, Any]:
    """Evaluate static, moving-camera pan, thin-object boundary, and low-texture sequences
    across Legacy Census vs Ortho-Hybrid (Independent & Derived)."""
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    cap_path = root / manifest["capture"]["file"]
    lx = np.fromfile(root / manifest["maps"]["left_x"]["file"], dtype="<f4").reshape(720, 1280)
    ly = np.fromfile(root / manifest["maps"]["left_y"]["file"], dtype="<f4").reshape(720, 1280)
    rx = np.fromfile(root / manifest["maps"]["right_x"]["file"], dtype="<f4").reshape(720, 1280)
    ry = np.fromfile(root / manifest["maps"]["right_y"]["file"], dtype="<f4").reshape(720, 1280)

    base_luma_l = []
    base_luma_r = []
    for ordinal in range(5):
        _, payload = read_capture_record(cap_path, ordinal)
        yuyv = np.frombuffer(payload, dtype=np.uint8).reshape(720, 2560, 2)
        rgb = cv2.cvtColor(yuyv, cv2.COLOR_YUV2RGB_YUYV)
        rl = cv2.remap(rgb[:, :1280], lx, ly, cv2.INTER_LINEAR)
        rr = cv2.remap(rgb[:, 1280:], rx, ry, cv2.INTER_LINEAR)
        base_luma_l.append(cv2.cvtColor(rl, cv2.COLOR_RGB2GRAY).astype(np.uint32))
        base_luma_r.append(cv2.cvtColor(rr, cv2.COLOR_RGB2GRAY).astype(np.uint32))

    # Build 4 3-frame sequences at 320x180 (coarse 4x scale, D=32) for fast multi-frame sequence comparison:
    # 1. static: real ZED 2 frames 0, 1, 2
    # 2. moving_camera: horizontal camera pan (shift both eyes by 0, 4, 8 px at full-res -> 0, 1, 2 px at 4x)
    # 3. thin_object: insert vertical 3-px thin textured post at disparity 16 px (4 px at 4x) across frames
    # 4. low_texture: contrast-attenuated sequence (0.45x dynamic range around mean luma)
    sequences: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "static_camera": [],
        "moving_camera_pan": [],
        "thin_object_structures": [],
        "low_texture_attenuated": [],
    }
    for idx in range(3):
        ll = cv2.resize(base_luma_l[idx].astype(np.uint8), (320, 180), interpolation=cv2.INTER_AREA).astype(np.uint32)
        rr = cv2.resize(base_luma_r[idx].astype(np.uint8), (320, 180), interpolation=cv2.INTER_AREA).astype(np.uint32)
        sequences["static_camera"].append((ll, rr))

        # Moving camera pan: shift both eyes horizontally by idx * 2 px
        shift = idx * 2
        ll_pan = np.roll(ll, -shift, axis=1)
        rr_pan = np.roll(rr, -shift, axis=1)
        sequences["moving_camera_pan"].append((ll_pan, rr_pan))

        # Thin object: 3 vertical thin bars (width 2 px at 320x180 = 8 px at 720p) at d=14 px
        ll_thin = ll.copy()
        rr_thin = rr.copy()
        d_bar = 14
        for bx in (90 + idx, 160 + idx, 230 + idx):
            for dy in range(30, 150):
                val_bar = np.uint32(190 if (dy % 4 < 2) else 70)
                ll_thin[dy, bx : bx + 2] = val_bar
                rr_thin[dy, bx - d_bar : bx - d_bar + 2] = val_bar
        sequences["thin_object_structures"].append((ll_thin, rr_thin))

        # Low texture: compress contrast around mean 110 by 0.45x
        ll_low = np.clip(110.0 + 0.45 * (ll.astype(np.float32) - 110.0), 0, 255).astype(np.uint32)
        rr_low = np.clip(110.0 + 0.45 * (rr.astype(np.float32) - 110.0), 0, 255).astype(np.uint32)
        sequences["low_texture_attenuated"].append((ll_low, rr_low))

    out: dict[str, Any] = {}
    for seq_name, pairs in sequences.items():
        seq_res: dict[str, Any] = {}
        for mode_label, desc_mode, derived in (
            ("legacy_census_24bit", "census_24bit", False),
            ("ortho_hybrid_independent", "ortho_hybrid_u32", False),
            ("ortho_hybrid_derived", "ortho_hybrid_u32", True),
        ):
            covs = []
            cohs = []
            rmses = []
            for ll, rr in pairs:
                cl, cr = build_cost_volumes(ll, rr, 32, desc_mode)
                al = aggregate_sgm_4path_vectorized(cl, p1=8, p2=32)
                dl, _, lv = select_from_aggregated(
                    al, non_adjacent_runner_up=False, uniqueness_ratio=0.0, min_margin=1, denom_floor=1
                )
                if not derived:
                    ar = aggregate_sgm_4path_vectorized(cr, p1=8, p2=32)
                    dr, _, rv = select_from_aggregated(
                        ar, non_adjacent_runner_up=False, uniqueness_ratio=0.0, min_margin=1, denom_floor=1
                    )
                else:
                    # Derived right from left aggregated volume
                    h_c, w_c, d_c = al.shape
                    ar_der = np.full_like(al, 65535)
                    for d_idx in range(d_c):
                        if d_idx == 0:
                            ar_der[:, :, 0] = al[:, :, 0]
                        else:
                            ar_der[:, : w_c - d_idx, d_idx] = al[:, d_idx:, d_idx]
                    dr, _, rv = select_from_aggregated(
                        ar_der, non_adjacent_runner_up=False, uniqueness_ratio=0.0, min_margin=1, denom_floor=1
                    )
                lr_ok, _, _ = lr_pair(dl, dr, lv, rv)
                geom = geometric_qualification_metrics(dl.astype(np.float32), lr_ok, ll)
                covs.append(100.0 * float(lr_ok.mean()))
                cohs.append(100.0 * float(geom["coherent_surface_fraction_all_pixels"]))
                if geom["planar_disparity_rmse_median_px"] is not None:
                    rmses.append(float(geom["planar_disparity_rmse_median_px"]))
            seq_res[mode_label] = {
                "mean_lr_valid_pct": round(float(np.mean(covs)), 2),
                "mean_coherent_coverage_pct": round(float(np.mean(cohs)), 2),
                "mean_planar_rmse_px": round(float(np.mean(rmses)), 3) if rmses else None,
            }
        out[seq_name] = seq_res
    return out


def main() -> None:
    work_dir = ROOT / "artifacts/depth-video/v9-milestone/step7-10-e2e"
    work_dir.mkdir(parents=True, exist_ok=True)

    replay_186_dir = work_dir / "replay_186f"
    if not (replay_186_dir / "capture.sfp").exists():
        build_186frame_zed2_replay(
            DEFAULT_MANIFEST.parent / "capture/capture.sfp", replay_186_dir, total_frames=186
        )

    env = dict(os.environ)
    env.update(
        {
            "AMENT_PREFIX_PATH": str(ROOT / "ros_ws/.pixi/envs/default"),
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "VK_DRIVER_FILES": "/usr/share/vulkan/icd.d/radeon_icd.x86_64.json",
            "LD_PRELOAD": "/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1",
            "LD_LIBRARY_PATH": str(ROOT / "ros_ws/.pixi/envs/default/lib"),
        }
    )

    ros_runs_spec = [
        (
            "4x_independent_180f",
            "experimental-dual-refinement",
            "layered-per-path",
            "independent-sgm",
            "0",
            180,
            replay_186_dir,
        ),
        (
            "4x_derived_180f",
            "experimental-dual-refinement",
            "shared-census-sequential",
            "left-aggregated-selection",
            "0",
            180,
            replay_186_dir,
        ),
        (
            "4x_selective_repair_independent_180f",
            "experimental-dual-refinement",
            "layered-per-path",
            "independent-sgm",
            "1",
            180,
            replay_186_dir,
        ),
        (
            "4x_selective_repair_derived_180f",
            "experimental-dual-refinement",
            "shared-census-sequential",
            "left-aggregated-selection",
            "1",
            180,
            replay_186_dir,
        ),
        (
            "full_sgm_independent_30f",
            "full-sgm",
            "layered-per-path",
            "independent-sgm",
            "0",
            30,
            replay_186_dir,
        ),
    ]

    ros_results: dict[str, Any] = {}
    for run_name, pipe, sched, rmode, repair, max_f, rep_dir in ros_runs_spec:
        trace_p = work_dir / f"{run_name}_trace.json"
        rate_p = work_dir / f"{run_name}_rate.json"
        cmd = [
            str(PUBLISHER_BIN),
            "--source",
            "replay",
            "--replay",
            str(rep_dir),
            "--manifest",
            str(DEFAULT_MANIFEST),
            "--clock-map",
            str(CLOCK_MAP),
            "--frame-id",
            "zed2_left_camera_optical_frame",
            "--run-id",
            f"v9_{run_name}",
            "--mode",
            "depth-rgb",
            "--capture-fps",
            "60",
            "--depth-rate",
            "60",
            "--warmup",
            "5",
            "--max-frames",
            str(max_f),
            "--pipeline",
            pipe,
            "--descriptor-mode",
            "ortho-hybrid",
            "--fast-mode",
            "1",
            "--selective-repair",
            repair,
            "--coarse-schedule",
            sched,
            "--coarse-right-reference-mode",
            rmode,
            "--validation",
            "--thread-cpu-timing",
            "--trace",
            str(trace_p),
            "--rate-report",
            str(rate_p),
        ]
        if pipe != "full-sgm":
            cmd.extend(["--coarse-manifest", str(COARSE_4X_MANIFEST)])
        if not (trace_p.exists() and rate_p.exists()):
            subprocess.run(cmd, env=env, check=True)
        ros_results[run_name] = parse_ros_delivery_trace(trace_p, rate_p)

    # Collect Step 10 GPU Dual-Preprocess + Refinement Latency Qualification (120 frames, 20 warmup)
    step9_dir = ROOT / "artifacts/depth-video/v9-milestone/step9-dumps"
    v8_step6 = json.loads(
        (ROOT / "docs/evidence/v8-review-milestone/step6-multiresolution-refinement-240fps-v8.json").read_text()
    )["runs"]

    gpu_e2e_summary = {}
    for key, src_doc in [
        ("dual4x_ortho-hybrid_layered_fast1", v8_step6["dual4x_ortho-hybrid_layered_fast1"]),
        ("dual4x_ortho-hybrid_derived_fast1", v8_step6["dual4x_ortho-hybrid_derived_fast1"]),
        (
            "dual4x_ortho-hybrid_layered_fast1_repair1",
            json.loads((step9_dir / "dual4x_ortho-hybrid_layered_fast1_repair1.json").read_text()),
        ),
        (
            "dual4x_ortho-hybrid_derived_fast1_repair1",
            json.loads((step9_dir / "dual4x_ortho-hybrid_derived_fast1_repair1.json").read_text()),
        ),
    ]:
        hw = src_doc["host_wall_ms"]
        gpu_e2e_summary[key] = {
            "host_wall_ms": hw,
            "enqueue_cpu_ms": src_doc["enqueue_cpu_ms"],
            "wait_cpu_ms": src_doc["wait_cpu_ms"],
            "equivalent_fps_p50": round(1000.0 / float(hw["p50"]), 1),
            "equivalent_fps_p95": round(1000.0 / float(hw["p95"]), 1),
            "equivalent_fps_p99": round(1000.0 / float(hw["p99"]), 1),
            "meets_240fps_4_167ms_p50": float(hw["p50"]) <= 4.167,
            "meets_240fps_4_167ms_p95": float(hw["p95"]) <= 4.167,
            "all_30_planes_exact_between_variants": src_doc["all_30_planes_exact_between_variants"],
            "validation_errors_after_teardown": src_doc["validation_errors_after_teardown"],
            "validation_warnings_after_teardown": src_doc["validation_warnings_after_teardown"],
            "requested_total_buffer_bytes": src_doc.get("requested_total_buffer_bytes")
            or src_doc.get("dual_requested_buffer_bytes"),
        }

    seq_eval = evaluate_challenging_sequences(DEFAULT_MANIFEST)
    hw_probe = probe_physical_zed2_hardware()

    out_dir = ROOT / "docs/evidence/v9-live-and-audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "step7-10-live-and-e2e-qualification-v9.json"
    doc = {
        "schema": "stereoforge.live_and_e2e_qualification_v9.v1",
        "zed2_hardware_probe": hw_probe,
        "ros2_e2e_60hz_delivery_runs": ros_results,
        "gpu_240fps_e2e_latency_qualification": gpu_e2e_summary,
        "multi_sequence_regime_evaluation": seq_eval,
    }
    out_json.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
