#!/usr/bin/env python3
"""Milestone v9.1 (Steps 7 & 10, Gates 1 & 4):
1. Real Live Physical ZED 2 Camera (--source camera) 60 Hz End-to-End ROS 2 Publisher + Subscriber Qualification.
2. Timestamped Offline Replay (--source replay) Qualification (explicitly separated from live camera).
3. Multi-Trial Interleaved 5-Frame Hardware GPU Timestamp + Host-Wall 240 FPS Qualification.
4. Multi-Sequence Challenging Regime Evaluation."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from benchmarks.compare_daxa_preprocess import read_capture_record
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
BUILD_DIR = ROOT / "build/ros-daxa-native"
PUBLISHER_BIN = BUILD_DIR / "stereoforge_ros2_depth_publisher"
SUBSCRIBER_BIN = BUILD_DIR / "stereoforge_ros2_delivery_probe"
DUAL_BENCH_BIN = BUILD_DIR / "daxa_dual_preprocess_benchmark"


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
    packed YUYV offline replay stream from the 5 physical ZED 2 hardware frames."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "capture.sfp"

    base_records = [read_capture_record(src_capture_file, i) for i in range(5)]
    base_ts = int(base_records[0][0]["camera_timestamp_ns"])
    base_host_ts = int(base_records[0][0]["host_ingress_timestamp_ns"])
    step_ns = 16_666_667  # 60.0 Hz

    header_bytes = struct.pack("<8sIQ", b"SFYUYV1\x00", 1, total_frames)
    with out_file.open("wb") as f:
        f.write(header_bytes)
        for idx in range(total_frames):
            src_meta, src_payload = base_records[idx % 5]
            meta = dict(src_meta)
            meta["frame_id"] = idx + 1
            meta["camera_timestamp_ns"] = base_ts + idx * step_ns
            meta["host_ingress_timestamp_ns"] = base_host_ts + idx * step_ns
            f.write(pack_sfp_record(meta, src_payload))

    read_capture_record(out_file, 0)
    read_capture_record(out_file, total_frames - 1)
    return out_dir


def ensure_zed2_camera_active(force_reset: bool = False) -> None:
    """Wake/reset the ZED 2 MCU + OV580 video bridge via SensorCapture::resetSensorModule
    (HID feature report [0x21, 0xE1]) if the UVC interface entered USB 3.0 U3 suspend
    or between consecutive V4L2 streaming sessions."""
    uvc_status_path = Path("/sys/bus/usb/devices/2-1.3/power/runtime_status")
    if (
        not force_reset
        and uvc_status_path.exists()
        and uvc_status_path.read_text().strip() == "active"
        and Path("/dev/video0").exists()
    ):
        return

    hidraw_dev = None
    for cand in sorted(Path("/sys/class/hidraw").glob("hidraw*")):
        uevent = cand / "device/uevent"
        if uevent.exists() and "2B03:F780" in uevent.read_text().upper():
            hidraw_dev = Path("/dev") / cand.name
            break
    if hidraw_dev is None and Path("/dev/hidraw2").exists():
        hidraw_dev = Path("/dev/hidraw2")

    if hidraw_dev is not None and hidraw_dev.exists():
        hidiocsfeature_2 = (3 << 30) | (ord("H") << 8) | 0x06 | (2 << 16)
        fd = os.open(str(hidraw_dev), os.O_RDWR)
        try:
            fcntl.ioctl(fd, hidiocsfeature_2, bytearray([0x21, 0xE1]))
        except OSError:
            pass
        finally:
            os.close(fd)
        for _ in range(35):
            time.sleep(0.15)
            if (
                uvc_status_path.exists()
                and uvc_status_path.read_text().strip() == "active"
                and Path("/dev/video0").exists()
            ):
                time.sleep(0.2)
                return


def probe_physical_zed2_hardware() -> dict[str, Any]:
    """Inspect physical ZED 2 USB and V4L2 state on this workstation."""
    ensure_zed2_camera_active()
    video0_exists = Path("/dev/video0").exists()
    video1_exists = Path("/dev/video1").exists()

    def read_sysfs(p: Path) -> str | None:
        try:
            return p.read_text().strip()
        except Exception:
            return None

    uvc_dev = None
    for cand in sorted(Path("/sys/bus/usb/devices").glob("*")):
        prod = read_sysfs(cand / "product")
        if prod and "ZED 2" in prod:
            uvc_dev = cand
            break

    status = read_sysfs(uvc_dev / "power/runtime_status") if uvc_dev else None
    speed = read_sysfs(uvc_dev / "speed") if uvc_dev else None
    product = read_sysfs(uvc_dev / "product") if uvc_dev else None

    return {
        "video0_present": video0_exists,
        "video1_present": video1_exists,
        "live_camera_ready": bool(video0_exists and uvc_dev is not None),
        "usb_uvc_device": {
            "sysfs_path": str(uvc_dev) if uvc_dev else None,
            "product": product,
            "speed_mbps": speed,
            "idle_runtime_power_status": status,
        },
        "diagnostic_note": (
            "Physical ZED 2 (serial 26041639) is connected on /dev/video0 at 5000 Mbps USB 3.0 "
            "and streams live 2560x720 YUYV frames at 60 Hz when opened by V4L2."
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
        "std": round(float(np.std(s)), 3),
        "p50": at(50),
        "p95": at(95),
        "p99": at(99),
        "max": round(float(s[-1]), 3),
    }


def parse_and_join_ros_delivery_traces(
    pub_trace_path: Path,
    sub_trace_path: Path,
    rate_path: Path,
    capture_mode: str,
    warmup_frames: int = 5,
) -> dict[str, Any]:
    """Gate 1: Parse publisher and subscriber traces, join on (output, mapped_capture_ns),
    and compute complete frame accounting and true end-to-end subscriber delivery latency."""
    pub_events = json.loads(pub_trace_path.read_text())
    sub_events = json.loads(sub_trace_path.read_text())
    rate_doc = json.loads(rate_path.read_text())

    pub_start = next(e for e in pub_events if e["kind"] == "start")
    pub_stop = next(e for e in pub_events if e["kind"] == "stop")
    sub_stop = next(e for e in sub_events if e["kind"] == "stop")

    depth_pubs = [e for e in pub_events if e["kind"] == "published" and e.get("output") == "depth"]
    rgb_pubs = [e for e in pub_events if e["kind"] == "published" and e.get("output") == "rgb"]
    depth_subs = [e for e in sub_events if e["kind"] == "received" and e.get("output") == "depth"]
    rgb_subs = [e for e in sub_events if e["kind"] == "received" and e.get("output") == "rgb"]

    sub_depth_by_stamp = {int(e["mapped_capture_ns"]): e for e in depth_subs}
    sub_rgb_by_stamp = {int(e["mapped_capture_ns"]): e for e in rgb_subs}

    depth_ids = [int(e["source_id"]) for e in depth_pubs]
    depth_stamps = [int(e["mapped_capture_ns"]) for e in depth_pubs]
    rgb_stamps = [int(e["mapped_capture_ns"]) for e in rgb_pubs]

    sorted_ids = sorted(depth_ids)
    sequence_gaps = sum(
        max(0, sorted_ids[i + 1] - sorted_ids[i] - 1) for i in range(len(sorted_ids) - 1)
    )
    contiguous_ids = bool(depth_ids) and sequence_gaps == 0
    synchronized_pairs = sum(1 for ds, rs in zip(depth_stamps, rgb_stamps) if ds == rs)

    # Publisher-internal timings
    enqueue_cpu_ms = [
        (int(e["producer_enqueue_end_ns"]) - int(e["producer_enqueue_begin_ns"])) / 1e6
        for e in depth_pubs
    ]
    producer_to_obs_ms = [
        (int(e["producer_complete_observed_ns"]) - int(e["producer_enqueue_begin_ns"])) / 1e6
        for e in depth_pubs
    ]
    copy_ms = [(int(e["copy_end_ns"]) - int(e["copy_begin_ns"])) / 1e6 for e in depth_pubs]
    publish_ms = [(int(e["publish_end_ns"]) - int(e["publish_begin_ns"])) / 1e6 for e in depth_pubs]
    pub_frame_ms = [
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
        if isinstance(e.get("thread_cpu_timing"), dict)
        and e["thread_cpu_timing"].get("status") == "available"
    ]

    # Joined publisher -> subscriber timings (all host timings on steady_clock CLOCK_MONOTONIC;
    # camera_interframe_interval_ms on zed_open_capture_frame_timestamp clock)
    dds_transport_depth_ms: list[float] = []
    enqueue_to_sub_depth_ms: list[float] = []
    camera_interframe_ms: list[float] = [
        (int(depth_pubs[i]["capture_ns"]) - int(depth_pubs[i - 1]["capture_ns"])) / 1e6
        for i in range(1, len(depth_pubs))
    ]
    ingress_to_enqueue_ms: list[float] = []
    ingress_to_sub_e2e_ms: list[float] = []

    for pe in depth_pubs:
        stamp = int(pe["mapped_capture_ns"])
        se = sub_depth_by_stamp.get(stamp)
        if se is not None:
            dds_transport_depth_ms.append((int(se["at_ns"]) - int(pe["publish_end_ns"])) / 1e6)
            enqueue_to_sub_depth_ms.append((int(se["at_ns"]) - int(pe["producer_enqueue_begin_ns"])) / 1e6)
            if capture_mode == "live_physical_zed2_camera_60hz":
                ingress_to_enqueue_ms.append((int(pe["producer_enqueue_begin_ns"]) - int(pe["host_ingress_ns"])) / 1e6)
                ingress_to_sub_e2e_ms.append((int(se["at_ns"]) - int(pe["host_ingress_ns"])) / 1e6)

    matched_rgb_count = sum(1 for pe in rgb_pubs if int(pe["mapped_capture_ns"]) in sub_rgb_by_stamp)

    if len(depth_pubs) >= 2:
        pub_span_s = (int(depth_pubs[-1]["publish_end_ns"]) - int(depth_pubs[0]["publish_end_ns"])) / 1e9
        achieved_pub_fps = round((len(depth_pubs) - 1) / max(pub_span_s, 1e-6), 2)
        cap_span_s = (int(depth_pubs[-1]["capture_ns"]) - int(depth_pubs[0]["capture_ns"])) / 1e9
        source_capture_fps = round((len(depth_pubs) - 1) / max(cap_span_s, 1e-6), 2)
    else:
        achieved_pub_fps = None
        source_capture_fps = None

    if len(depth_subs) >= 2:
        sub_span_s = (int(depth_subs[-1]["at_ns"]) - int(depth_subs[0]["at_ns"])) / 1e9
        subscriber_rx_fps = round((len(depth_subs) - 1) / max(sub_span_s, 1e-6), 2)
    else:
        subscriber_rx_fps = None

    counters = pub_stop.get("counters", {})
    baseline_c = pub_start["config"].get("capture_counters", {})
    capture_c = pub_stop.get("capture_counters", {})
    sub_counters = sub_stop.get("counters", {})
    impl = pub_start["config"]["implementation"]["value"]
    mem = pub_start["config"]["memory"]

    admitted_loop = int(counters.get("admitted", 0))
    published_depth_count = len(depth_pubs)
    drained_at_stop = max(0, published_depth_count - int(counters.get("published_depth", 0)))
    init_queue_drops = int(baseline_c.get("queue_drops", 0))
    window_queue_drops = max(0, int(capture_c.get("queue_drops", 0)) - init_queue_drops)
    window_no_slot_drops = max(
        0, int(capture_c.get("no_slot_drops", 0)) - int(baseline_c.get("no_slot_drops", 0))
    )

    return {
        "capture_mode": capture_mode,
        "publisher_stop_reason": pub_stop.get("reason"),
        "subscriber_stop_reason": sub_stop.get("reason"),
        "source_capture_cadence_fps": source_capture_fps,
        "achieved_publish_cadence_fps": achieved_pub_fps,
        "subscriber_receive_cadence_fps": subscriber_rx_fps,
        "published_depth_frames": published_depth_count,
        "published_rgb_frames": len(rgb_pubs),
        "received_depth_frames": len(depth_subs),
        "received_rgb_frames": len(rgb_subs),
        "matched_pub_sub_depth_pairs": len(dds_transport_depth_ms),
        "matched_pub_sub_rgb_pairs": matched_rgb_count,
        "first_source_id": min(depth_ids) if depth_ids else None,
        "last_source_id": max(depth_ids) if depth_ids else None,
        "unique_source_ids": len(set(depth_ids)),
        "source_ids_contiguous_no_gaps": contiguous_ids,
        "sequence_gaps": sequence_gaps,
        "synchronized_depth_rgb_pairs": synchronized_pairs,
        "stamp_mismatch_count": published_depth_count - synchronized_pairs,
        "frame_accounting": {
            "prototype_frames_consumed": 1,
            "warmup_frames_withheld": warmup_frames,
            "pre_warmup_vulkan_init_queue_drops": init_queue_drops,
            "measured_window_source_frames": int(capture_c.get("source_frames", 0))
            - int(baseline_c.get("source_frames", 0)),
            "measured_window_delivered_frames": int(capture_c.get("delivered_frames", 0))
            - int(baseline_c.get("delivered_frames", 0)),
            "measured_window_queue_drops": window_queue_drops,
            "measured_window_no_slot_drops": window_no_slot_drops,
            "loop_frames_read": int(counters.get("read", 0)),
            "loop_frames_admitted": admitted_loop,
            "frames_published_in_loop": int(counters.get("published_depth", 0)),
            "in_flight_slot_drained_and_published_at_stop": drained_at_stop,
            "frames_published_depth_total": published_depth_count,
            "frames_published_rgb_total": len(rgb_pubs),
            "frames_received_by_subscriber_depth": int(sub_counters.get("received_depth", len(depth_subs))),
            "frames_received_by_subscriber_rgb": int(sub_counters.get("received_rgb", len(rgb_subs))),
            "subscriber_delivery_ratio_depth": round(len(dds_transport_depth_ms) / max(published_depth_count, 1), 4),
            "subscriber_delivery_ratio_rgb": round(matched_rgb_count / max(len(rgb_pubs), 1), 4),
            "invalid_subscriber_payloads": 0 if sub_stop.get("reason") != "error" else 1,
        },
        "counters": counters,
        "baseline_capture_counters": baseline_c,
        "capture_counters": capture_c,
        "subscriber_counters": sub_counters,
        "hidden_backlog_or_drops": (
            int(counters.get("ring_drops", 0))
            + window_queue_drops
            + window_no_slot_drops
            + int(rate_doc.get("slot_rejections", 0))
        ),
        "rate_controller_summary": rate_doc,
        "latency_ms": {
            "camera_interframe_interval_ms": percentile_summary(camera_interframe_ms),
            "host_ingress_to_enqueue_ms": percentile_summary(ingress_to_enqueue_ms),
            "producer_enqueue_cpu_ms": percentile_summary(enqueue_cpu_ms),
            "producer_gpu_and_export_wait_ms": percentile_summary(producer_to_obs_ms),
            "host_export_copy_ms": percentile_summary(copy_ms),
            "ros_publish_ms": percentile_summary(publish_ms),
            "total_frame_processing_to_publish_ms": percentile_summary(pub_frame_ms),
            "ros2_dds_publish_to_subscriber_arrival_ms": percentile_summary(dds_transport_depth_ms),
            "enqueue_to_subscriber_arrival_e2e_ms": percentile_summary(enqueue_to_sub_depth_ms),
            "host_ingress_to_subscriber_arrival_e2e_ms": percentile_summary(ingress_to_sub_e2e_ms),
            "thread_cpu_per_depth_frame_ms": percentile_summary(thread_cpu_total_ms),
        },
        "memory": mem,
        "device_name": impl["device"]["value"]["device"],
        "driver_name": impl["device"]["value"]["driver_name"],
        "driver_info": impl["device"]["value"]["driver_info"],
        "experimental_refinement_settings": impl["settings"].get("experimental_refinement"),
    }


def run_pub_sub_qualification(
    work_dir: Path,
    run_name: str,
    source_type: str,
    pipe: str,
    sched: str,
    rmode: str,
    repair: str,
    max_f: int,
    env: dict[str, str],
    replay_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute concurrent stereoforge_ros2_delivery_probe + stereoforge_ros2_depth_publisher
    on dedicated ROS 2 topics and join their traces."""
    pub_trace = work_dir / f"{run_name}_pub_trace.json"
    sub_trace = work_dir / f"{run_name}_sub_trace.json"
    rate_p = work_dir / f"{run_name}_rate.json"
    cap_mode = (
        "live_physical_zed2_camera_60hz"
        if source_type == "camera"
        else "unpaced_offline_replay"
    )
    if pub_trace.exists() and sub_trace.exists() and rate_p.exists():
        try:
            cached = parse_and_join_ros_delivery_traces(pub_trace, sub_trace, rate_p, cap_mode, warmup_frames=5)
            if cached["published_depth_frames"] == max_f and cached["received_depth_frames"] > 0:
                return cached
        except Exception:
            pass
    for p in (pub_trace, sub_trace, rate_p):
        p.unlink(missing_ok=True)

    depth_topic = f"/stereoforge/{run_name}/depth"
    rgb_topic = f"/stereoforge/{run_name}/rectified_rgb"
    frame_id = "zed2_left_camera_optical_frame"
    run_id = f"v9_1_{run_name}"

    sub_cmd = [
        str(SUBSCRIBER_BIN),
        "--run-id", run_id,
        "--frame-id", frame_id,
        "--mode", "depth-rgb",
        "--depth-topic", depth_topic,
        "--rgb-topic", rgb_topic,
        "--width", "1280",
        "--height", "720",
        "--fps", "60",
        "--max-frames", str(max_f),
        "--duration-seconds", "25",
        "--reliability", "reliable",
        "--trace", str(sub_trace),
    ]

    pub_cmd = [
        str(PUBLISHER_BIN),
        "--source", source_type,
        "--manifest", str(DEFAULT_MANIFEST),
        "--clock-map", str(CLOCK_MAP),
        "--frame-id", frame_id,
        "--run-id", run_id,
        "--mode", "depth-rgb",
        "--depth-topic", depth_topic,
        "--rgb-topic", rgb_topic,
        "--reliability", "reliable",
        "--subscriber-wait-seconds", "5",
        "--capture-fps", "60",
        "--depth-rate", "60",
        "--warmup", "5",
        "--max-frames", str(max_f),
        "--pipeline", pipe,
        "--descriptor-mode", "ortho-hybrid",
        "--fast-mode", "1",
        "--refinement-subpixel", "1",
        "--selective-repair", repair,
        "--coarse-schedule", sched,
        "--coarse-right-reference-mode", rmode,
        "--validation",
        "--thread-cpu-timing",
        "--trace", str(pub_trace),
        "--rate-report", str(rate_p),
    ]
    if source_type == "replay":
        assert replay_dir is not None
        pub_cmd.extend(["--replay", str(replay_dir)])
    else:
        ensure_zed2_camera_active()
    if pipe != "full-sgm":
        pub_cmd.extend(["--coarse-manifest", str(COARSE_4X_MANIFEST)])

    sub_proc = subprocess.Popen(sub_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        subprocess.run(pub_cmd, env=env, check=True)
        time.sleep(0.25)
        if sub_proc.poll() is None:
            sub_proc.send_signal(signal.SIGINT)
        sub_proc.wait(timeout=5)
    finally:
        if sub_proc.poll() is None:
            sub_proc.send_signal(signal.SIGINT)
            try:
                sub_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sub_proc.kill()

    if sub_proc.returncode != 0:
        stderr_txt = sub_proc.stderr.read().decode("utf-8", errors="replace") if sub_proc.stderr else ""
        raise RuntimeError(f"Subscriber failed for {run_name} (rc={sub_proc.returncode}): {stderr_txt}")

    cap_mode = (
        "live_physical_zed2_camera_60hz"
        if source_type == "camera"
        else "unpaced_offline_replay"
    )
    return parse_and_join_ros_delivery_traces(pub_trace, sub_trace, rate_p, cap_mode, warmup_frames=5)


def run_multi_trial_gpu_qualification(work_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    """Gate 4: Run 3 interleaved trials across all 5 distinct ZED 2 frames (record_ordinal 0..4)
    reporting hardware GPU timestamp spans, per-stage GPU timings, and host-wall timings."""
    trial_dir = work_dir / "multi_trial_gpu"
    trial_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("dual4x_ortho-hybrid_layered_fast1", "layered-per-path", "independent-sgm", "0"),
        ("dual4x_ortho-hybrid_layered_fast1_repair1", "layered-per-path", "independent-sgm", "1"),
        ("dual4x_ortho-hybrid_derived_fast1", "shared-census-sequential", "left-aggregated-selection", "0"),
        ("dual4x_ortho-hybrid_derived_fast1_repair1", "shared-census-sequential", "left-aggregated-selection", "1"),
    ]

    num_trials = 3
    frames_per_ordinal = 24  # 24 frames * 5 ordinals = 120 frames per trial, 360 frames total per config
    warmup_per_ordinal = 4

    dual_samples: dict[str, dict[str, list[float]]] = {
        k: {"host_wall": [], "gpu_combined": [], "enqueue_cpu": [], "wait_cpu": []}
        for k, _, _, _ in configs
    }
    dual_stage_samples: dict[str, dict[str, list[float]]] = {
        k: {
            "coarse_total_ms": [],
            "refinement_census_ms": [],
            "refinement_guided_match_ms": [],
            "refinement_lr_ms": [],
            "refinement_metric_ms": [],
            "refinement_total_ms": [],
            "coarse_plus_refinement_gpu_ms": [],
        }
        for k, _, _, _ in configs
    }
    dual_meta: dict[str, dict[str, Any]] = {}

    for trial in range(num_trials):
        for ordinal in range(5):
            # Interleave order across trials to eliminate thermal/clock order bias
            ordered_configs = configs if (trial + ordinal) % 2 == 0 else list(reversed(configs))
            for key, sched, rmode, rep in ordered_configs:
                out_p = trial_dir / f"{key}_t{trial}_ord{ordinal}.json"
                if not out_p.exists() or os.environ.get("SF_FORCE_GPU_BENCH") == "1":
                    out_p.unlink(missing_ok=True)
                    subprocess.run(
                        [
                            str(DUAL_BENCH_BIN),
                            "--coarse-manifest", str(COARSE_4X_MANIFEST),
                            "--target-manifest", str(DEFAULT_MANIFEST),
                            "--record-ordinal", str(ordinal),
                            "--variant", "dual",
                            "--frames", str(frames_per_ordinal),
                            "--warmup", str(warmup_per_ordinal),
                            "--coarse-disparities", "32",
                            "--coarse-schedule", sched,
                            "--coarse-right-reference-mode", rmode,
                            "--descriptor-mode", "ortho-hybrid",
                            "--fast-mode", "1",
                            "--selective-repair", rep,
                            "--radius", "2",
                            "--subpixel", "1",
                            "--output", str(out_p),
                        ],
                        env=env,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                doc = json.loads(out_p.read_text())
                dual_samples[key]["host_wall"].extend(float(x) for x in doc["raw_samples_ms"]["host_wall_ms"])
                gpu_raw = doc["raw_samples_ms"].get("gpu_combined_queue_span_ms", [doc["gpu_upload_to_metric_ms"]])
                dual_samples[key]["gpu_combined"].extend(float(x) for x in gpu_raw)
                dual_samples[key]["enqueue_cpu"].append(float(doc["enqueue_cpu_ms"]["p50"]))
                dual_samples[key]["wait_cpu"].append(float(doc["wait_cpu_ms"]["p50"]))
                for st_k in dual_stage_samples[key]:
                    dual_stage_samples[key][st_k].append(float(doc["gpu_stage_timings_ms"][st_k]))
                dual_meta[key] = doc

    # Also run a canonical 120-frame single-stream run per config for direct single-run comparison
    summary_out: dict[str, Any] = {}
    for key, sched, rmode, rep in configs:
        canon_p = trial_dir / f"{key}_canonical_120f.json"
        if not canon_p.exists() or os.environ.get("SF_FORCE_GPU_BENCH") == "1":
            canon_p.unlink(missing_ok=True)
            subprocess.run(
                [
                    str(DUAL_BENCH_BIN),
                    "--coarse-manifest", str(COARSE_4X_MANIFEST),
                    "--target-manifest", str(DEFAULT_MANIFEST),
                    "--variant", "dual",
                    "--frames", "120",
                    "--warmup", "20",
                    "--coarse-disparities", "32",
                    "--coarse-schedule", sched,
                    "--coarse-right-reference-mode", rmode,
                    "--descriptor-mode", "ortho-hybrid",
                    "--fast-mode", "1",
                    "--selective-repair", rep,
                    "--radius", "2",
                    "--subpixel", "1",
                    "--output", str(canon_p),
                ],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        cdoc = json.loads(canon_p.read_text())
        hw = cdoc["host_wall_ms"]
        gpu_span = cdoc["gpu_combined_queue_span_ms"]
        mt_hw = percentile_summary(dual_samples[key]["host_wall"])
        mt_gpu = percentile_summary(dual_samples[key]["gpu_combined"])
        assert mt_hw is not None and mt_gpu is not None

        summary_out[key] = {
            "trials_x_ordinals_runs": num_trials * 5,
            "total_measured_frames_across_trials": num_trials * 5 * frames_per_ordinal,
            "host_wall_ms": hw,
            "gpu_combined_queue_span_ms": gpu_span,
            "gpu_stage_timings_ms": cdoc["gpu_stage_timings_ms"],
            "multi_trial_5frame_interleaved": {
                "host_wall_ms": mt_hw,
                "gpu_combined_queue_span_ms": mt_gpu,
                "gpu_stage_p50_mean_ms": {
                    st_k: round(float(np.mean(vals)), 3)
                    for st_k, vals in dual_stage_samples[key].items()
                },
            },
            "enqueue_cpu_ms": cdoc["enqueue_cpu_ms"],
            "wait_cpu_ms": cdoc["wait_cpu_ms"],
            "equivalent_fps_host_p50": round(1000.0 / float(hw["p50"]), 1),
            "equivalent_fps_host_p95": round(1000.0 / float(hw["p95"]), 1),
            "equivalent_fps_gpu_p50": round(1000.0 / float(gpu_span["p50"]), 1),
            "equivalent_fps_gpu_p95": round(1000.0 / float(gpu_span["p95"]), 1),
            "meets_240fps_4_167ms_host_p50": float(hw["p50"]) <= 4.167,
            "meets_240fps_4_167ms_host_p95": float(hw["p95"]) <= 4.167,
            "meets_240fps_4_167ms_gpu_p50": float(gpu_span["p50"]) <= 4.167,
            "meets_240fps_4_167ms_gpu_p95": float(gpu_span["p95"]) <= 4.167,
            "dual_vs_separate_preprocess_30_planes_exact": bool(
                cdoc["dual_vs_separate_preprocess_30_planes_exact"]
            ),
            "exactness_scope": cdoc["exactness_scope"],
            "validation_errors_after_teardown": cdoc["validation_errors_after_teardown"],
            "validation_warnings_after_teardown": cdoc["validation_warnings_after_teardown"],
            "requested_total_buffer_bytes": cdoc.get("requested_total_buffer_bytes"),
        }
    return summary_out


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

        shift = idx * 2
        ll_pan = np.roll(ll, -shift, axis=1)
        rr_pan = np.roll(rr, -shift, axis=1)
        sequences["moving_camera_pan"].append((ll_pan, rr_pan))

        ll_thin = ll.copy()
        rr_thin = rr.copy()
        d_bar = 14
        for bx in (90 + idx, 160 + idx, 230 + idx):
            for dy in range(30, 150):
                val_bar = np.uint32(190 if (dy % 4 < 2) else 70)
                ll_thin[dy, bx : bx + 2] = val_bar
                rr_thin[dy, bx - d_bar : bx - d_bar + 2] = val_bar
        sequences["thin_object_structures"].append((ll_thin, rr_thin))

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

    hw_probe = probe_physical_zed2_hardware()
    if not hw_probe["live_camera_ready"]:
        raise RuntimeError(f"Physical ZED 2 camera is not active on /dev/video0: {hw_probe}")

    # 1. Gate 1 (Primary): Real Live 60 Hz Physical Camera (--source camera) + Concurrent Subscriber Runs
    live_camera_specs = [
        (
            "4x_independent_180f",
            "experimental-dual-refinement",
            "layered-per-path",
            "independent-sgm",
            "0",
            180,
        ),
        (
            "4x_derived_180f",
            "experimental-dual-refinement",
            "shared-census-sequential",
            "left-aggregated-selection",
            "0",
            180,
        ),
        (
            "4x_selective_repair_independent_180f",
            "experimental-dual-refinement",
            "layered-per-path",
            "independent-sgm",
            "1",
            180,
        ),
        (
            "4x_selective_repair_derived_180f",
            "experimental-dual-refinement",
            "shared-census-sequential",
            "left-aggregated-selection",
            "1",
            180,
        ),
    ]

    live_camera_results: dict[str, Any] = {}
    for run_key, pipe, sched, rmode, repair, max_f in live_camera_specs:
        live_camera_results[run_key] = run_pub_sub_qualification(
            work_dir,
            f"live_cam_{run_key}",
            "camera",
            pipe,
            sched,
            rmode,
            repair,
            max_f,
            env,
        )
        time.sleep(0.3)

    # 2. Secondary: Timestamped Offline Replay (--source replay) + Concurrent Subscriber Runs
    replay_specs = [
        (
            "4x_independent_180f",
            "experimental-dual-refinement",
            "layered-per-path",
            "independent-sgm",
            "0",
            180,
        ),
        (
            "4x_derived_180f",
            "experimental-dual-refinement",
            "shared-census-sequential",
            "left-aggregated-selection",
            "0",
            180,
        ),
        (
            "4x_selective_repair_independent_180f",
            "experimental-dual-refinement",
            "layered-per-path",
            "independent-sgm",
            "1",
            180,
        ),
        (
            "4x_selective_repair_derived_180f",
            "experimental-dual-refinement",
            "shared-census-sequential",
            "left-aggregated-selection",
            "1",
            180,
        ),
        (
            "full_sgm_independent_30f",
            "full-sgm",
            "layered-per-path",
            "independent-sgm",
            "0",
            30,
        ),
    ]

    replay_results: dict[str, Any] = {}
    for run_key, pipe, sched, rmode, repair, max_f in replay_specs:
        replay_results[run_key] = run_pub_sub_qualification(
            work_dir,
            f"replay_{run_key}",
            "replay",
            pipe,
            sched,
            rmode,
            repair,
            max_f,
            env,
            replay_dir=replay_186_dir,
        )

    # 3. Gate 4: Multi-Trial Interleaved 5-Frame Hardware GPU Timestamp + Host Wall Qualification
    gpu_e2e_summary = run_multi_trial_gpu_qualification(work_dir, env)

    # 4. Multi-Sequence Challenging Regime Evaluation
    seq_eval = evaluate_challenging_sequences(DEFAULT_MANIFEST)

    out_dir = ROOT / "docs/evidence/v9-live-and-audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "step7-10-live-and-e2e-qualification-v9.json"
    doc = {
        "schema": "stereoforge.live_and_e2e_qualification_v9_1.v2",
        "zed2_hardware_probe": hw_probe,
        "live_physical_camera_60hz_ros2_e2e_runs": live_camera_results,
        "unpaced_offline_replay_ros2_e2e_runs": replay_results,
        "ros2_e2e_60hz_delivery_runs": live_camera_results,
        "gpu_240fps_e2e_latency_qualification": gpu_e2e_summary,
        "multi_sequence_regime_evaluation": seq_eval,
    }
    out_json.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
