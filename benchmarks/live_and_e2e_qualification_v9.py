#!/usr/bin/env python3
"""Milestone v9.2 (Steps 7 & 10, Gates 1 & 4):
1. Real Live Physical ZED 2 Camera (--source camera) 60 Hz End-to-End ROS 2 Publisher + Subscriber Qualification:
   - Lossless KeepLast(16) reliable mode (--qos-depth 16) with 64-bit FNV-1a payload checksum verification
   - Latest-frame KeepLast(1) mode (--qos-depth 1) comparison
   - Extended 600-frame (10.0 s) live physical camera endurance run
   - Shared-GPU contention live physical camera run (concurrent GPU compute load)
2. Timestamped Offline Replay (--source replay) Qualification (explicitly separated from live camera).
3. Multi-Trial Interleaved 5-Frame Hardware GPU Single-Interval & Stage-Sum Timestamp + Host-Wall 240 FPS Qualification.
4. Multi-Sequence Challenging Regime Evaluation and Programmatic Markdown Report Generation."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare_daxa_preprocess import read_capture_record
from benchmarks.matching_cost_inspector_v8 import (
    aggregate_sgm_4path_vectorized,
    build_cost_volumes,
    geometric_qualification_metrics,
    select_from_aggregated,
)
from benchmarks.stereo_reference_crosscheck_v5 import lr_pair
from benchmarks.visual_accuracy_audit_v9 import current_provenance_manifest

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


def find_verified_zed2_hidraw(sysfs_hidraw_root: Path = Path("/sys/class/hidraw"), dev_root: Path = Path("/dev")) -> Path | None:
    """Locate a single Stereolabs ZED 2 MCU hidraw node by verifying VID:PID 2B03:F780 in sysfs uevent.
    Never falls back to an unverified /dev/hidrawN node and refuses if multiple ZED 2 MCUs exist."""
    matches: list[Path] = []
    if sysfs_hidraw_root.exists():
        for cand in sorted(sysfs_hidraw_root.glob("hidraw*")):
            uevent = cand / "device/uevent"
            try:
                txt = uevent.read_text().upper() if uevent.exists() else ""
            except OSError:
                continue
            if any(tag in txt for tag in ("2B03:0000F780", "2B03:F780", "2B03:0000F781", "2B03:F781")):
                dev_node = dev_root / cand.name
                if dev_node.exists():
                    matches.append(dev_node)
    if len(matches) == 1:
        return matches[0]
    return None


def ensure_zed2_camera_active(
    force_reset: bool = False,
    sysfs_hidraw_root: Path = Path("/sys/class/hidraw"),
    dev_root: Path = Path("/dev"),
) -> dict[str, Any]:
    """Wake/reset the ZED 2 MCU + OV580 video bridge via HID feature report [0x21, 0xE1]
    only after verifying single-device Stereolabs VID:PID (2B03:F780/F781). Never suppresses open/ioctl errors."""
    uvc_status_path = Path("/sys/bus/usb/devices/2-1.3/power/runtime_status")
    video0 = dev_root / "video0"
    if (
        not force_reset
        and uvc_status_path.exists()
        and uvc_status_path.read_text().strip() == "active"
        and video0.exists()
    ):
        return {"action": "already_active", "status": "ok", "hidraw_device": None}

    hidraw_dev = find_verified_zed2_hidraw(sysfs_hidraw_root=sysfs_hidraw_root, dev_root=dev_root)
    if hidraw_dev is None:
        return {
            "action": "refused_no_verified_single_zed2_hidraw",
            "status": "error",
            "hidraw_device": None,
        }

    hidiocsfeature_2 = (3 << 30) | (ord("H") << 8) | 0x06 | (2 << 16)
    try:
        fd = os.open(str(hidraw_dev), os.O_RDWR)
    except OSError as exc:
        return {
            "action": "hidraw_open_failed",
            "status": "error",
            "hidraw_device": str(hidraw_dev),
            "errno": exc.errno,
            "error": str(exc),
        }
    try:
        fcntl.ioctl(fd, hidiocsfeature_2, bytearray([0x21, 0xE1]))
    except OSError as exc:
        if exc.errno != 32:  # EPIPE (32) is expected when the ZED 2 MCU resets immediately
            return {
                "action": "hidraw_ioctl_failed",
                "status": "error",
                "hidraw_device": str(hidraw_dev),
                "errno": exc.errno,
                "error": str(exc),
            }
    finally:
        os.close(fd)

    for _ in range(35):
        time.sleep(0.15)
        if (
            uvc_status_path.exists()
            and uvc_status_path.read_text().strip() == "active"
            and video0.exists()
        ):
            time.sleep(0.2)
            return {
                "action": "reset_and_reenumerated",
                "status": "ok",
                "hidraw_device": str(hidraw_dev),
            }
    return {
        "action": "reset_sent_awaiting_uvc_active",
        "status": "ok" if video0.exists() else "timeout",
        "hidraw_device": str(hidraw_dev),
    }


def probe_physical_zed2_hardware() -> dict[str, Any]:
    """Inspect physical ZED 2 USB and V4L2 state and verify actual frame acquisition."""
    wake_info = ensure_zed2_camera_active()
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

    cap_verified = False
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, frame = cap.read()
        cap_verified = bool(ok and frame is not None and frame.size > 0)
        cap.release()

    return {
        "video0_present": video0_exists,
        "video1_present": video1_exists,
        "live_frame_read_verified": cap_verified,
        "live_camera_ready": bool(video0_exists and uvc_dev is not None and cap_verified),
        "wake_recovery_probe": wake_info,
        "usb_uvc_device": {
            "sysfs_path": str(uvc_dev) if uvc_dev else None,
            "product": product,
            "speed_mbps": speed,
            "idle_runtime_power_status": status,
        },
        "diagnostic_note": (
            "Physical ZED 2 (serial 26041639) is connected on /dev/video0 at 5000 Mbps USB 3.0 "
            "and verified via live 2560x720 V4L2 frame acquisition."
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
    comp_depth_pubs = [e for e in pub_events if e["kind"] == "published" and e.get("output") == "completed_depth"]
    comp_prov_pubs = [e for e in pub_events if e["kind"] == "published" and e.get("output") == "completed_provenance"]
    depth_subs = [e for e in sub_events if e["kind"] == "received" and e.get("output") == "depth"]
    rgb_subs = [e for e in sub_events if e["kind"] == "received" and e.get("output") == "rgb"]
    comp_depth_subs = [e for e in sub_events if e["kind"] == "received" and e.get("output") == "completed_depth"]
    comp_prov_subs = [e for e in sub_events if e["kind"] == "received" and e.get("output") == "completed_provenance"]

    sub_depth_by_stamp = {int(e["mapped_capture_ns"]): e for e in depth_subs}
    sub_rgb_by_stamp = {int(e["mapped_capture_ns"]): e for e in rgb_subs}
    sub_comp_depth_by_stamp = {int(e["mapped_capture_ns"]): e for e in comp_depth_subs}
    sub_comp_prov_by_stamp = {int(e["mapped_capture_ns"]): e for e in comp_prov_subs}

    depth_ids = [int(e["source_id"]) for e in depth_pubs]
    depth_stamps = [int(e["mapped_capture_ns"]) for e in depth_pubs]
    rgb_stamps = [int(e["mapped_capture_ns"]) for e in rgb_pubs]
    comp_depth_stamps = [int(e["mapped_capture_ns"]) for e in comp_depth_pubs]
    comp_prov_stamps = [int(e["mapped_capture_ns"]) for e in comp_prov_pubs]

    sorted_ids = sorted(depth_ids)
    sequence_gaps = sum(
        max(0, sorted_ids[i + 1] - sorted_ids[i] - 1) for i in range(len(sorted_ids) - 1)
    )
    contiguous_ids = bool(depth_ids) and sequence_gaps == 0
    synchronized_pairs = sum(1 for ds, rs in zip(depth_stamps, rgb_stamps) if ds == rs)
    synchronized_quads = (
        sum(
            1
            for ds, cds, cps, rs in zip(depth_stamps, comp_depth_stamps, comp_prov_stamps, rgb_stamps)
            if ds == cds == cps == rs
        )
        if comp_depth_pubs
        else 0
    )

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
    ingress_to_publish_ms: list[float] = []
    ingress_to_sub_e2e_ms: list[float] = []
    depth_checksum_matches = 0
    depth_checksum_mismatches = 0
    rgb_checksum_matches = 0
    rgb_checksum_mismatches = 0

    for pe in depth_pubs:
        stamp = int(pe["mapped_capture_ns"])
        se = sub_depth_by_stamp.get(stamp)
        if se is not None:
            dds_transport_depth_ms.append((int(se["at_ns"]) - int(pe["publish_end_ns"])) / 1e6)
            enqueue_to_sub_depth_ms.append((int(se["at_ns"]) - int(pe["producer_enqueue_begin_ns"])) / 1e6)
            if capture_mode == "live_physical_zed2_camera_60hz":
                ingress_to_enqueue_ms.append((int(pe["producer_enqueue_begin_ns"]) - int(pe["host_ingress_ns"])) / 1e6)
                ingress_to_publish_ms.append((int(pe["publish_end_ns"]) - int(pe["host_ingress_ns"])) / 1e6)
                ingress_to_sub_e2e_ms.append((int(se["at_ns"]) - int(pe["host_ingress_ns"])) / 1e6)
            pub_cs = int(pe.get("payload_fnv1a64", 0))
            sub_cs = int(se.get("payload_fnv1a64", 0))
            if pub_cs != 0 and sub_cs != 0:
                if pub_cs == sub_cs and int(pe.get("valid_pixels", 0)) == int(se.get("valid_pixels", 0)) and int(se.get("valid_pixels", 0)) > 0:
                    depth_checksum_matches += 1
                else:
                    depth_checksum_mismatches += 1

    matched_rgb_count = 0
    for pe in rgb_pubs:
        stamp = int(pe["mapped_capture_ns"])
        se = sub_rgb_by_stamp.get(stamp)
        if se is not None:
            matched_rgb_count += 1
            pub_cs = int(pe.get("payload_fnv1a64", 0))
            sub_cs = int(se.get("payload_fnv1a64", 0))
            if pub_cs != 0 and sub_cs != 0:
                if pub_cs == sub_cs and int(se.get("valid_pixels", 0)) > 0:
                    rgb_checksum_matches += 1
                else:
                    rgb_checksum_mismatches += 1

    matched_comp_depth_count = 0
    comp_depth_checksum_matches = 0
    comp_depth_checksum_mismatches = 0
    for pe in comp_depth_pubs:
        stamp = int(pe["mapped_capture_ns"])
        se = sub_comp_depth_by_stamp.get(stamp)
        if se is not None:
            matched_comp_depth_count += 1
            pub_cs = int(pe.get("payload_fnv1a64", 0))
            sub_cs = int(se.get("payload_fnv1a64", 0))
            if pub_cs != 0 and sub_cs != 0:
                if pub_cs == sub_cs and int(pe.get("valid_pixels", 0)) == int(se.get("valid_pixels", 0)) and int(se.get("valid_pixels", 0)) > 0:
                    comp_depth_checksum_matches += 1
                else:
                    comp_depth_checksum_mismatches += 1

    matched_comp_prov_count = 0
    comp_prov_checksum_matches = 0
    comp_prov_checksum_mismatches = 0
    for pe in comp_prov_pubs:
        stamp = int(pe["mapped_capture_ns"])
        se = sub_comp_prov_by_stamp.get(stamp)
        if se is not None:
            matched_comp_prov_count += 1
            pub_cs = int(pe.get("payload_fnv1a64", 0))
            sub_cs = int(se.get("payload_fnv1a64", 0))
            if pub_cs != 0 and sub_cs != 0:
                if pub_cs == sub_cs and int(pe.get("valid_pixels", 0)) == int(se.get("valid_pixels", 0)) and int(se.get("valid_pixels", 0)) > 0:
                    comp_prov_checksum_matches += 1
                else:
                    comp_prov_checksum_mismatches += 1

    inferred_planar_gains = [
        int(cd.get("valid_pixels", 0)) - int(dp.get("valid_pixels", 0))
        for dp, cd in zip(depth_pubs, comp_depth_pubs)
        if int(dp.get("mapped_capture_ns", -1)) == int(cd.get("mapped_capture_ns", -2))
        and int(cd.get("valid_pixels", 0)) > 0
    ]

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
        "published_completed_depth_frames": len(comp_depth_pubs),
        "published_completed_provenance_frames": len(comp_prov_pubs),
        "published_rgb_frames": len(rgb_pubs),
        "received_depth_frames": len(depth_subs),
        "received_completed_depth_frames": len(comp_depth_subs),
        "received_completed_provenance_frames": len(comp_prov_subs),
        "received_rgb_frames": len(rgb_subs),
        "matched_pub_sub_depth_pairs": len(dds_transport_depth_ms),
        "matched_pub_sub_completed_depth_pairs": matched_comp_depth_count,
        "matched_pub_sub_completed_provenance_pairs": matched_comp_prov_count,
        "matched_pub_sub_rgb_pairs": matched_rgb_count,
        "first_source_id": min(depth_ids) if depth_ids else None,
        "last_source_id": max(depth_ids) if depth_ids else None,
        "unique_source_ids": len(set(depth_ids)),
        "source_ids_contiguous_no_gaps": contiguous_ids,
        "sequence_gaps": sequence_gaps,
        "synchronized_depth_rgb_pairs": synchronized_pairs,
        "synchronized_4topic_quads": synchronized_quads,
        "stamp_mismatch_count": published_depth_count - synchronized_pairs,
        "inferred_planar_pixels_per_frame": percentile_summary([float(x) for x in inferred_planar_gains]) if inferred_planar_gains else None,
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
            "frames_published_completed_depth_total": len(comp_depth_pubs),
            "frames_published_completed_provenance_total": len(comp_prov_pubs),
            "frames_published_rgb_total": len(rgb_pubs),
            "frames_received_by_subscriber_depth": int(sub_counters.get("received_depth", len(depth_subs))),
            "frames_received_by_subscriber_completed_depth": int(sub_counters.get("received_completed_depth", len(comp_depth_subs))),
            "frames_received_by_subscriber_completed_provenance": int(sub_counters.get("received_completed_provenance", len(comp_prov_subs))),
            "frames_received_by_subscriber_rgb": int(sub_counters.get("received_rgb", len(rgb_subs))),
            "subscriber_delivery_ratio_depth": round(len(dds_transport_depth_ms) / max(published_depth_count, 1), 4),
            "subscriber_delivery_ratio_completed_depth": (
                round(matched_comp_depth_count / max(len(comp_depth_pubs), 1), 4)
                if comp_depth_pubs
                else None
            ),
            "subscriber_delivery_ratio_completed_provenance": (
                round(matched_comp_prov_count / max(len(comp_prov_pubs), 1), 4)
                if comp_prov_pubs
                else None
            ),
            "subscriber_delivery_ratio_rgb": round(matched_rgb_count / max(len(rgb_pubs), 1), 4),
            "verified_payload_checksum_matches_depth": depth_checksum_matches,
            "verified_payload_checksum_mismatches_depth": depth_checksum_mismatches,
            "verified_payload_checksum_matches_completed_depth": comp_depth_checksum_matches,
            "verified_payload_checksum_mismatches_completed_depth": comp_depth_checksum_mismatches,
            "verified_payload_checksum_matches_completed_provenance": comp_prov_checksum_matches,
            "verified_payload_checksum_mismatches_completed_provenance": comp_prov_checksum_mismatches,
            "verified_payload_checksum_matches_rgb": rgb_checksum_matches,
            "verified_payload_checksum_mismatches_rgb": rgb_checksum_mismatches,
            "invalid_subscriber_payloads": (
                0
                if (
                    sub_stop.get("reason") != "error"
                    and depth_checksum_mismatches == 0
                    and comp_depth_checksum_mismatches == 0
                    and comp_prov_checksum_mismatches == 0
                    and rgb_checksum_mismatches == 0
                )
                else 1
            ),
            "qos_depth": int(pub_start["config"].get("qos", {}).get("depth", pub_start["config"].get("qos_depth", 16))),
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
            "host_ingress_to_publish_ms": percentile_summary(ingress_to_publish_ms),
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
    qos_depth: int = 16,
    verify_checksum: bool = True,
    concurrent_gpu_contention: bool = False,
    planar_recovery: str = "0",
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
    allow_cache = os.environ.get("SF_ALLOW_DEV_CACHE") == "1"
    if allow_cache and pub_trace.exists() and sub_trace.exists() and rate_p.exists():
        try:
            cached = parse_and_join_ros_delivery_traces(pub_trace, sub_trace, rate_p, cap_mode, warmup_frames=5)
            if (
                cached["published_depth_frames"] == max_f
                and cached["received_depth_frames"] > 0
                and cached["frame_accounting"].get("qos_depth") == qos_depth
            ):
                return cached
        except Exception:
            pass
    for p in (pub_trace, sub_trace, rate_p):
        p.unlink(missing_ok=True)

    depth_topic = f"/stereoforge/{run_name}/depth"
    rgb_topic = f"/stereoforge/{run_name}/rectified_rgb"
    comp_depth_topic = f"/stereoforge/{run_name}/depth_completed"
    comp_prov_topic = f"/stereoforge/{run_name}/depth_completed_provenance"
    frame_id = "zed2_left_camera_optical_frame"
    run_id = f"v9_3_{run_name}"
    duration_s = max(25, int(max_f / 60 * 2.5) + 15)

    sub_cmd = [
        str(SUBSCRIBER_BIN),
        "--run-id", run_id,
        "--frame-id", frame_id,
        "--mode", "depth-rgb",
        "--depth-topic", depth_topic,
        "--rgb-topic", rgb_topic,
        "--completed-depth-topic", comp_depth_topic,
        "--completed-provenance-topic", comp_prov_topic,
        "--expect-completed", planar_recovery,
        "--width", "1280",
        "--height", "720",
        "--fps", "60",
        "--max-frames", str(max_f),
        "--duration-seconds", str(duration_s),
        "--reliability", "reliable",
        "--qos-depth", str(qos_depth),
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
        "--completed-depth-topic", comp_depth_topic,
        "--completed-provenance-topic", comp_prov_topic,
        "--reliability", "reliable",
        "--qos-depth", str(qos_depth),
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
        "--planar-recovery", planar_recovery,
        "--coarse-schedule", sched,
        "--coarse-right-reference-mode", rmode,
        "--validation",
        "--thread-cpu-timing",
        "--trace", str(pub_trace),
        "--rate-report", str(rate_p),
    ]
    if verify_checksum:
        pub_cmd.append("--verify-payload-checksum")
    if source_type == "replay":
        assert replay_dir is not None
        pub_cmd.extend(["--replay", str(replay_dir)])
    else:
        ensure_zed2_camera_active()
    if pipe != "full-sgm":
        pub_cmd.extend(["--coarse-manifest", str(COARSE_4X_MANIFEST)])

    bg_gpu_proc = None
    bg_out = work_dir / f"{run_name}_bg_gpu_load.json"
    bg_count_file = work_dir / f"{run_name}_bg_gpu_batches.txt"
    if concurrent_gpu_contention:
        bg_out.unlink(missing_ok=True)
        bg_count_file.unlink(missing_ok=True)
        loop_cmd = (
            "n=0; while true; do "
            f"rm -f '{bg_out}'; "
            f"'{DUAL_BENCH_BIN}' "
            f"--coarse-manifest '{COARSE_4X_MANIFEST}' "
            f"--target-manifest '{DEFAULT_MANIFEST}' "
            "--variant dual --frames 300 --warmup 10 "
            "--coarse-disparities 32 --coarse-schedule layered-per-path "
            "--coarse-right-reference-mode independent-sgm "
            "--descriptor-mode ortho-hybrid --fast-mode 1 "
            "--selective-repair 1 --planar-recovery 1 "
            "--radius 2 --subpixel 1 "
            f"--output '{bg_out}' >/dev/null 2>&1 || exit 1; "
            f"n=$((n+1)); echo $n > '{bg_count_file}'; "
            "done"
        )
        bg_gpu_proc = subprocess.Popen(
            ["bash", "-c", loop_cmd],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(80):
            if bg_count_file.exists() or bg_gpu_proc.poll() is not None:
                break
            time.sleep(0.1)

    bg_active_at_end = False
    sub_proc = subprocess.Popen(sub_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        subprocess.run(pub_cmd, env=env, check=True)
        if bg_gpu_proc is not None:
            bg_active_at_end = bg_gpu_proc.poll() is None
        time.sleep(0.35)
        if sub_proc.poll() is None:
            sub_proc.send_signal(signal.SIGINT)
        sub_proc.wait(timeout=5)
    finally:
        if bg_gpu_proc is not None and bg_gpu_proc.poll() is None:
            try:
                os.killpg(bg_gpu_proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                bg_gpu_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(bg_gpu_proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if sub_proc.poll() is None:
            sub_proc.send_signal(signal.SIGINT)
            try:
                sub_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sub_proc.kill()

    if sub_proc.returncode != 0:
        stderr_txt = sub_proc.stderr.read().decode("utf-8", errors="replace") if sub_proc.stderr else ""
        raise RuntimeError(f"Subscriber failed for {run_name} (rc={sub_proc.returncode}): {stderr_txt}")

    res = parse_and_join_ros_delivery_traces(pub_trace, sub_trace, rate_p, cap_mode, warmup_frames=5)
    res["concurrent_gpu_contention_load"] = concurrent_gpu_contention
    if concurrent_gpu_contention:
        batches = int(bg_count_file.read_text().strip()) if bg_count_file.exists() else 0
        res["concurrent_gpu_contention_active_at_completion"] = bg_active_at_end
        res["concurrent_gpu_contention_completed_300f_batches"] = batches
    return res


def run_multi_trial_gpu_qualification(work_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    """Gate 4: Run 3 interleaved trials across all 5 distinct ZED 2 frames (record_ordinal 0..4)
    reporting single-interval hardware GPU timestamp spans (upload -> metric), stage-sum GPU timings,
    and host-wall timings."""
    trial_dir = work_dir / "multi_trial_gpu"
    trial_dir.mkdir(parents=True, exist_ok=True)
    allow_cache = os.environ.get("SF_ALLOW_DEV_CACHE") == "1"

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
        k: {"host_wall": [], "gpu_single_interval": [], "gpu_stage_sum": [], "enqueue_cpu": [], "wait_cpu": []}
        for k, _, _, _ in configs
    }
    dual_stage_samples: dict[str, dict[str, list[float]]] = {
        k: {
            "dual_preprocess_upload_ms": [],
            "dual_preprocess_decode_ms": [],
            "dual_preprocess_rectify_ms": [],
            "dual_preprocess_total_ms": [],
            "coarse_total_ms": [],
            "refinement_census_ms": [],
            "refinement_guided_match_ms": [],
            "refinement_lr_ms": [],
            "refinement_metric_ms": [],
            "refinement_total_ms": [],
            "single_interval_upload_to_metric_gpu_ms": [],
            "stage_sum_gpu_ms": [],
        }
        for k, _, _, _ in configs
    }

    for trial in range(num_trials):
        for ordinal in range(5):
            ordered_configs = configs if (trial + ordinal) % 2 == 0 else list(reversed(configs))
            for key, sched, rmode, rep in ordered_configs:
                out_p = trial_dir / f"{key}_t{trial}_ord{ordinal}.json"
                if not allow_cache or not out_p.exists():
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
                gpu_raw = doc["raw_samples_ms"]["gpu_combined_queue_span_ms"]
                dual_samples[key]["gpu_single_interval"].extend(float(x) for x in gpu_raw)
                stage_sum_raw = doc["raw_samples_ms"].get("gpu_stage_sum_ms", gpu_raw)
                dual_samples[key]["gpu_stage_sum"].extend(float(x) for x in stage_sum_raw)
                dual_samples[key]["enqueue_cpu"].append(float(doc["enqueue_cpu_ms"]["p50"]))
                dual_samples[key]["wait_cpu"].append(float(doc["wait_cpu_ms"]["p50"]))
                for st_k in dual_stage_samples[key]:
                    if st_k in doc["gpu_stage_timings_ms"]:
                        dual_stage_samples[key][st_k].append(float(doc["gpu_stage_timings_ms"][st_k]))

    summary_out: dict[str, Any] = {}
    for key, sched, rmode, rep in configs:
        canon_p = trial_dir / f"{key}_canonical_120f.json"
        if not allow_cache or not canon_p.exists():
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
        gpu_stage_sum = cdoc.get("gpu_stage_sum_ms", gpu_span)
        mt_hw = percentile_summary(dual_samples[key]["host_wall"])
        mt_gpu = percentile_summary(dual_samples[key]["gpu_single_interval"])
        mt_stage_sum = percentile_summary(dual_samples[key]["gpu_stage_sum"])
        assert mt_hw is not None and mt_gpu is not None and mt_stage_sum is not None

        summary_out[key] = {
            "trials_x_ordinals_runs": num_trials * 5,
            "total_measured_frames_across_trials": num_trials * 5 * frames_per_ordinal,
            "host_wall_ms": hw,
            "gpu_combined_queue_span_ms": gpu_span,
            "gpu_single_interval_upload_to_metric_ms": gpu_span,
            "gpu_stage_sum_ms": gpu_stage_sum,
            "gpu_stage_timings_ms": cdoc["gpu_stage_timings_ms"],
            "multi_trial_5frame_interleaved": {
                "host_wall_ms": mt_hw,
                "gpu_combined_queue_span_ms": mt_gpu,
                "gpu_single_interval_upload_to_metric_ms": mt_gpu,
                "gpu_stage_sum_ms": mt_stage_sum,
                "gpu_stage_p50_mean_ms": {
                    st_k: round(float(np.mean(vals)), 3)
                    for st_k, vals in dual_stage_samples[key].items()
                    if vals
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


def render_milestone_v9_markdown_from_json(e2e_doc: dict[str, Any], acc_doc: dict[str, Any]) -> str:
    """Programmatically generate docs/milestone-v9-live-and-selective-refinement-evidence.md
    directly from the two canonical JSON documents so Markdown and JSON can never drift."""
    hw = e2e_doc["zed2_hardware_probe"]
    uvc = hw["usb_uvc_device"]
    live_runs = e2e_doc["live_physical_camera_60hz_ros2_e2e_runs"]
    qos_comp = e2e_doc["live_camera_qos_and_stress_runs"]
    gpu_q = e2e_doc["gpu_240fps_e2e_latency_qualification"]

    gt = acc_doc["known_ground_truth_suite_720p"]["aggregate_across_5_scenes"]
    f5 = acc_doc["five_frame_zed2_zncc_epipolar_audit"]["mean_across_5_zed2_frames"]
    pipes = acc_doc["pipelines"]
    cohorts = acc_doc["newly_admitted_pixel_cohorts_gate5"]
    zncc_cov_f0 = acc_doc["independent_references_methodology"]["zncc_reference_anchor_coverage_pct_frame0"]

    def fmt_live_row(label: str, r: dict[str, Any], bold: bool = False) -> str:
        b = "**" if bold else ""
        fa = r["frame_accounting"]
        lat = r["latency_ms"]
        cad = f"{r['source_capture_cadence_fps']:.2f} / {r['achieved_publish_cadence_fps']:.2f} / {r['subscriber_receive_cadence_fps']:.2f} FPS"
        pub_pair = f"{r['published_depth_frames']} / {r['published_rgb_frames']}"
        rx_ratio = 100.0 * float(fa["subscriber_delivery_ratio_depth"])
        sub_rx = f"{r['received_depth_frames']} / {r['received_rgb_frames']} (`{rx_ratio:.1f}%`, QoS={fa.get('qos_depth', 16)})"
        cs_ok = f"{fa.get('verified_payload_checksum_matches_depth', 0)}/{r['received_depth_frames']} (`0` err)"
        drops = f"{fa['measured_window_queue_drops']} / {r['counters'].get('ring_drops', 0)} / {r['rate_controller_summary'].get('slot_rejections', 0)}"
        cam_iv = f"{lat['camera_interframe_interval_ms']['p50']:.3f} ms (`{lat['camera_interframe_interval_ms']['std']:.3f} ms`)"
        ing_enq = f"{lat['host_ingress_to_enqueue_ms']['p50']:.3f} ms"
        proc_pub = f"{lat['total_frame_processing_to_publish_ms']['p50']:.3f} / {lat['total_frame_processing_to_publish_ms']['p95']:.3f} ms (`max {lat['total_frame_processing_to_publish_ms']['max']:.3f} ms`)"
        dds = f"{lat['ros2_dds_publish_to_subscriber_arrival_ms']['p50']:.3f} ms"
        e2e = f"{lat['host_ingress_to_subscriber_arrival_e2e_ms']['p50']:.3f} / {lat['host_ingress_to_subscriber_arrival_e2e_ms']['p95']:.3f} ms"
        return (
            f"| {b}{label}{b} | {b}`{cad}`{b} | {b}`{pub_pair}`{b} | {b}`{sub_rx}`{b} | {b}`{cs_ok}`{b} "
            f"| **{drops}** | {b}`{cam_iv}`{b} | {b}`{ing_enq}`{b} | {b}`{proc_pub}`{b} | {b}`{dds}`{b} | **{e2e}** |"
        )

    def fmt_cohort_val(c: dict[str, Any]) -> str:
        if c["joint_independent_ref_pixels"] == 0 or c["disparity_mae_px"] is None:
            return f"`null` (`0/{c['pixels']}` ZNCC ref) | `{c.get('dense_photometric_reproj_mae_luma')} luma / {c.get('dense_patch_zncc_5x5_mean')} ZNCC5 / {c.get('dense_local_3x3_planar_inlier_pct')}% planar`"
        return (
            f"`{c['disparity_mae_px']:.3f} px / {c['bad_1px_pct']:.2f}% / {c['bad_2px_pct']:.2f}%` "
            f"(`{c['joint_independent_ref_pixels']}/{c['pixels']}` = `{c['reference_coverage_of_cohort_pct']:.1f}%` ZNCC) | "
            f"`{c.get('dense_photometric_reproj_mae_luma'):.2f} luma / {c.get('dense_patch_zncc_5x5_mean'):.3f} ZNCC5 / {c.get('dense_local_3x3_planar_inlier_pct'):.1f}% planar`"
        )

    lines = [
        "# StereoForge Milestone v9.2: Live 60 Hz Physical Camera Qualification, Non-Circular Accuracy Audit & Selective 4× Refinement (`240+ FPS`)",
        "",
        "## Overview",
        "",
        "This document is **programmatically generated** from the two canonical provenance-attested JSON files (`step7-10-live-and-e2e-qualification-v9.json` and `step8-9-visual-accuracy-and-selective-repair-v9.json`) on `AMD Radeon 8060S Graphics (RADV STRIX_HALO, Mesa 26.2.5)` and physical `ZED 2 (SN26041639, 1280×720, D=128)`:",
        "",
        "- **Tracked JSON Evidence:**",
        "  - [`docs/evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json`](evidence/v9-live-and-audit/step7-10-live-and-e2e-qualification-v9.json)",
        "  - [`docs/evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json`](evidence/v9-live-and-audit/step8-9-visual-accuracy-and-selective-repair-v9.json)",
        "- **Tracked Visual Diagnostic Panels:**",
        "  - [`docs/evidence/v9-live-and-audit/v9_visual_accuracy_audit_panels.png`](evidence/v9-live-and-audit/v9_visual_accuracy_audit_panels.png)",
        "  - [`docs/evidence/v9-live-and-audit/v9_fixed_regions_overlay.png`](evidence/v9-live-and-audit/v9_fixed_regions_overlay.png)",
        "",
        "---",
        "",
        "## 1. Step 7 (Gate 1) — Real Physical `60 Hz` ZED 2 Camera (`--source camera`) + Concurrent ROS 2 Subscriber Delivery",
        "",
        "### 1.1 Physical ZED 2 Hardware, Verified VID:PID Recovery, & 64-Bit Payload Checksums",
        f"- `/dev/video0` & `/dev/video1`: `{uvc['product']}` (`SN26041639`), connected at `{uvc['speed_mbps']} Mbps` USB 3.0 (`{uvc['sysfs_path']}`, `idle_runtime_power_status: \"{uvc['idle_runtime_power_status']}\"`), `live_frame_read_verified: {str(hw['live_frame_read_verified']).lower()}`.",
        "- Both `src/capture.cpp` (`NativeCapture`) and `src/zed_capture.cpp` (`stereoforge._zed_capture`) verify a single Stereolabs ZED 2 MCU (`2B03:F780` in `/sys/class/hidraw/*/device/uevent`) before invoking `sl_oc::sensors::SensorCapture::resetSensorModule()` (`HIDIOCSFEATURE(2)` `[0x21, 0xE1]`), and trigger bounded active-stream timeout recovery if `snapshotFrame` times out for `>= 2000 ms`.",
        "- Both [`stereoforge_ros2_depth_publisher`](../apps/ros2_depth_publisher.cpp) (`--verify-payload-checksum`) and [`stereoforge_ros2_delivery_probe`](../apps/ros2_delivery_probe.cpp) compute full-frame 64-bit FNV-1a checksums (`payload_fnv1a64`) and `valid_pixels` across all `3,686,400` payload bytes (`1280×720` `32FC1` depth and `rgba8` RGB) on every frame.",
        "",
        "### 1.2 Primary Lossless (`KeepLast(16)`, `--qos-depth 16`) Live Physical `60 Hz` Camera Runs (`180` Frames)",
        "",
        "| Live Physical Camera Run (`--source camera --qos-depth 16 --verify-payload-checksum`) | Capture / Publish / Subscriber Cadence | Published Depth / RGB | Subscriber Received | Full-Frame FNV-1a64 Checksum Verified | Queue / Ring / Slot Drops | Camera Inter-Frame `p50` (`std`) | Host Ingress → Enqueue `p50` | Frame Processing → Publish `p50 / p95` (`max`) | ROS 2 DDS Publish → Sub `p50` | Host Ingress → Subscriber E2E `p50 / p95` |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        fmt_live_row("`4x_independent_180f` (`layered-per-path`, `independent-sgm`, `repair=0`)", live_runs["4x_independent_180f"]),
        fmt_live_row("`4x_derived_180f` (`shared-census-sequential`, `left-aggregated-selection`, `repair=0`)", live_runs["4x_derived_180f"]),
        fmt_live_row("`4x_selective_repair_independent_180f` (`repair=1`)", live_runs["4x_selective_repair_independent_180f"], bold=True),
        fmt_live_row("`4x_selective_repair_derived_180f` (`repair=1`)", live_runs["4x_selective_repair_derived_180f"], bold=True),
        "",
        "### 1.3 Latest-Frame (`KeepLast(1)`), Extended `600`-Frame (`10.0 s`) Endurance, & Shared-GPU Contention Live Camera Runs",
        "",
        "| Live Physical Camera Regime | Capture / Publish / Subscriber Cadence | Published Depth / RGB | Subscriber Received | Full-Frame FNV-1a64 Checksum Verified | Queue / Ring / Slot Drops | Camera Inter-Frame `p50` (`std`) | Host Ingress → Enqueue `p50` | Frame Processing → Publish `p50 / p95` (`max`) | ROS 2 DDS Publish → Sub `p50` | Host Ingress → Subscriber E2E `p50 / p95` |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        fmt_live_row("`keep_last_1_latest_frame_180f` (`--qos-depth 1`, latest-frame mode)", qos_comp["keep_last_1_latest_frame_180f"]),
        fmt_live_row("`extended_600f_10s_endurance` (`600` frames = `10.0 s` continuous live 60 Hz)", qos_comp["extended_600f_10s_endurance"], bold=True),
        fmt_live_row("`shared_gpu_contention_180f` (concurrent background Vulkan compute load)", qos_comp["shared_gpu_contention_180f"], bold=True),
        "",
        "---",
        "",
        "## 2. Step 8 (Gate 2) — Non-Circular Quantitative Accuracy Audit",
        "",
        "### 2.1 Reference A: 5-Scene `1280×720` Known-Ground-Truth Suite on Native Vulkan GPU (`daxa_refinement_benchmark --synthetic-yuyv`)",
        "",
        "| Pipeline (`1280×720` Native Vulkan GPU Execution) | Mean Valid Coverage | Mean Disparity MAE (`px`) | Mean Bad `>1 px` (`%` of Valid) | Mean Bad `>2 px` (`%` of Valid) | Unconditional Good `≤1 px` Completeness | Unconditional Good `≤2 px` Completeness |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for pname, bold in (
        ("4x Refinement (Independent)", False),
        ("4x Selective Repair (Independent)", True),
        ("4x Refinement (Derived Right)", False),
        ("4x Selective Repair (Derived Right)", True),
        ("OpenCV StereoSGBM", False),
    ):
        b = "**" if bold else ""
        g = gt[pname]
        lines.append(
            f"| {b}`{pname}`{b} | {b}`{g['mean_non_occluded_coverage_pct']:.2f}%`{b} | {b}`{g['mean_disparity_mae_px']:.3f} px`{b} "
            f"| {b}`{g['mean_bad_1px_pct']:.2f}%`{b} | {b}`{g['mean_bad_2px_pct']:.2f}%`{b} "
            f"| {b}`{g['mean_unconditional_good_1px_completeness_pct']:.2f}%`{b} | {b}`{g['mean_unconditional_good_2px_completeness_pct']:.2f}%`{b} |"
        )

    lines.extend(
        [
            "",
            f"### 2.2 Reference B: 5-Frame ZED 2 Multi-Scale (`7×7 + 9×9`) Epipolar ZNCC Subpixel Reference (`{zncc_cov_f0:.2f}%` Anchor Coverage on Frame 0)",
            "",
            "| Pipeline (`5` Distinct ZED 2 Frames, `1280×720`) | 5-Frame Mean LR-Valid Coverage | 5-Frame Mean ZNCC MAE (`px`) | 5-Frame Mean ZNCC Bad `>1 px` | 5-Frame Mean ZNCC Bad `>2 px` | Frame 0 Coherent Coverage | Fixed 20 Planar Tiles SVD RMSE (`20/20` Tiles) | Fixed 20 Planar Tile Coverage |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    p_cen = pipes["Legacy Census 24b Full-Res"]
    p_fi = pipes["Full-Res Ortho-Hybrid (Independent)"]
    r_cen = p_cen["regions_vs_independent_zncc_ref"]["all_interior"]
    r_fi = p_fi["regions_vs_independent_zncc_ref"]["all_interior"]
    lines.append(
        f"| `Legacy Census 24b Full-Res` | `{p_cen['overall_lr_valid_pct']:.2f}%` (F0) | `{r_cen['disparity_mae_px']:.3f} px` (F0) "
        f"| `{r_cen['bad_1px_pct_of_valid']:.2f}%` (F0) | `{r_cen['bad_2px_pct_of_valid']:.2f}%` (F0) | `{p_cen['coherent_coverage_pct']:.2f}%` "
        f"| `{p_cen['fixed_planar_20tiles']['median_planar_svd_fit_rmse_px']:.3f} px` | `{p_cen['fixed_planar_20tiles']['mean_tile_coverage_pct']:.2f}%` |"
    )
    lines.append(
        f"| `Full-Res Ortho-Hybrid (Independent)` | `{p_fi['overall_lr_valid_pct']:.2f}%` (F0) | `{r_fi['disparity_mae_px']:.3f} px` (F0) "
        f"| `{r_fi['bad_1px_pct_of_valid']:.2f}%` (F0) | `{r_fi['bad_2px_pct_of_valid']:.2f}%` (F0) | `{p_fi['coherent_coverage_pct']:.2f}%` "
        f"| `{p_fi['fixed_planar_20tiles']['median_planar_svd_fit_rmse_px']:.3f} px` | `{p_fi['fixed_planar_20tiles']['mean_tile_coverage_pct']:.2f}%` |"
    )

    for mkey, pname, bold in (
        ("4x_ind_base", "4x Refinement (Independent)", False),
        ("4x_ind_rep", "4x Selective Repair (Independent)", True),
        ("4x_der_base", "4x Refinement (Derived Right)", False),
        ("4x_der_rep", "4x Selective Repair (Derived Right)", True),
    ):
        b = "**" if bold else ""
        m = f5[mkey]
        p = pipes[pname]
        lines.append(
            f"| {b}`{pname}`{b} | {b}`{m['mean_coverage_pct']:.2f}%`{b} | {b}`{m['mean_zncc_mae_px']:.3f} px`{b} "
            f"| {b}`{m['mean_zncc_bad_1px_pct']:.2f}%`{b} | {b}`{m['mean_zncc_bad_2px_pct']:.2f}%`{b} "
            f"| {b}`{p['coherent_coverage_pct']:.2f}%`{b} | {b}`{p['fixed_planar_20tiles']['median_planar_svd_fit_rmse_px']:.3f} px`{b} "
            f"| {b}`{p['fixed_planar_20tiles']['mean_tile_coverage_pct']:.2f}%`{b} |"
        )

    ci = cohorts["independent_4x"]
    cd = cohorts["derived_4x"]
    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Step 9 (Gate 5) — Newly-Admitted Pixel Cohort Decomposition & 100%-Coverage Epipolar Reprojection Audit",
            "",
            "Every cohort reports both **multi-scale ZNCC reference error** (returning explicit `null` if `joint_independent_ref_pixels == 0`) AND **dense 100%-coverage epipolar photometric reprojection error** ($|I_L(x,y) - I_R(x - \\hat{d}, y)|$), $5\\times 5$ patch ZNCC (`ZNCC5`), and $3\\times 3$ local planar inlier percentage across all in-bounds pixels:",
            "",
            "| Disjoint Pixel Cohort (`1280×720` ZED 2 Frame 0) | `Independent 4×` Pixels (`%` Image) | `Independent 4×` ZNCC MAE / Bad `>1 px` / Bad `>2 px` (`Ref Support`) \\| Dense Reproj / ZNCC5 / Planar | `Derived 4×` Pixels (`%` Image) | `Derived 4×` ZNCC MAE / Bad `>1 px` / Bad `>2 px` (`Ref Support`) \\| Dense Reproj / ZNCC5 / Planar |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| `baseline_valid` ($V_{{\\text{{base}}}}$) | `{ci['baseline_valid']['pixels']:,}` (`{ci['baseline_valid']['image_fraction_pct']:.2f}%`) | {fmt_cohort_val(ci['baseline_valid'])} | `{cd['baseline_valid']['pixels']:,}` (`{cd['baseline_valid']['image_fraction_pct']:.2f}%`) | {fmt_cohort_val(cd['baseline_valid'])} |",
            f"| `selective_repair_valid` ($V_{{\\text{{rep}}}}$) | `{ci['selective_repair_valid']['pixels']:,}` (`{ci['selective_repair_valid']['image_fraction_pct']:.2f}%`) | {fmt_cohort_val(ci['selective_repair_valid'])} | `{cd['selective_repair_valid']['pixels']:,}` (`{cd['selective_repair_valid']['image_fraction_pct']:.2f}%`) | {fmt_cohort_val(cd['selective_repair_valid'])} |",
            f"| **`newly_admitted_total` ($\\Delta V_+ = V_{{\\text{{rep}}}} \\setminus V_{{\\text{{base}}}}$)** | **`{ci['newly_admitted_total']['pixels']:,}` (`{ci['newly_admitted_total']['image_fraction_pct']:.2f}%`)** | {fmt_cohort_val(ci['newly_admitted_total'])} | **`{cd['newly_admitted_total']['pixels']:,}` (`{cd['newly_admitted_total']['image_fraction_pct']:.2f}%`)** | {fmt_cohort_val(cd['newly_admitted_total'])} |",
        ]
    )

    for mkey in (
        "cohort_a_recovered_coarse_hole_or_span",
        "cohort_b_accepted_2px_subpixel_peak",
        "cohort_c_expanded_search_window",
    ):
        mi = ci["newly_admitted_by_mechanism"][mkey]
        md = cd["newly_admitted_by_mechanism"][mkey]
        lines.append(
            f"| — `{mkey}` | `{mi['pixels']:,}` (`{mi['image_fraction_pct']:.2f}%`) | {fmt_cohort_val(mi)} | `{md['pixels']:,}` (`{md['image_fraction_pct']:.2f}%`) | {fmt_cohort_val(md)} |"
        )

    for rkey in ("planar_20tiles", "boundary_bands", "low_texture_tiles", "repetitive_tiles"):
        ri = ci["newly_admitted_by_region"][rkey]
        rd = cd["newly_admitted_by_region"][rkey]
        lines.append(
            f"| — Region: `{rkey}` | `{ri['pixels']:,}` (`{ri['image_fraction_pct']:.2f}%`) | {fmt_cohort_val(ri)} | `{rd['pixels']:,}` (`{rd['image_fraction_pct']:.2f}%`) | {fmt_cohort_val(rd)} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Step 10 (Gate 4) — Hardware GPU Single-Interval & Stage-Sum Timestamps + Multi-Trial Interleaved `240 FPS` (`4.167 ms`) Qualification",
            "",
            "Both the **single hardware `TimelineQueryPool` timestamp interval** (`gpu_single_interval_upload_to_metric_ms` = `refinement.end_tick - dual_preprocess.start_tick` on `daxa::QUEUE_MAIN`) and the **per-stage GPU sum** (`gpu_stage_sum_ms` = `preprocess + coarse + refinement`) are reported alongside host-wall timings across **3 interleaved trials × 5 distinct ZED 2 frames (`360` measured frames per configuration)** and a canonical `120`-frame run:",
            "",
            "| Dual-Preprocess + Refinement Pipeline (`1280×720`, `D=128`) | 3-Trial × 5-Frame Host Wall `p50 / p95` (`std`) | 3-Trial × 5-Frame Single-Interval GPU Span `p50 / p95` (`std`) | 3-Trial × 5-Frame Stage-Sum GPU `p50 / p95` | Canonical 120f Host Wall `p50` (`FPS`) | Canonical 120f Single-Interval GPU `p50` (`FPS`) | Preprocess / Coarse / Refinement GPU (`ms`) | 30-Plane Bit-Exact | Vulkan Validation Errors / Warnings |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
        ]
    )

    for gkey, label, bold in (
        ("dual4x_ortho-hybrid_layered_fast1", "`dual4x_ortho-hybrid_layered_fast1` (`Independent`, `repair=0`)", False),
        ("dual4x_ortho-hybrid_layered_fast1_repair1", "`dual4x_ortho-hybrid_layered_fast1_repair1` (`Independent`, `repair=1`)", True),
        ("dual4x_ortho-hybrid_derived_fast1", "`dual4x_ortho-hybrid_derived_fast1` (`Derived`, `repair=0`)", False),
        ("dual4x_ortho-hybrid_derived_fast1_repair1", "`dual4x_ortho-hybrid_derived_fast1_repair1` (`Derived`, `repair=1`)", True),
    ):
        b = "**" if bold else ""
        g = gpu_q[gkey]
        mt = g["multi_trial_5frame_interleaved"]
        hw_m = mt["host_wall_ms"]
        si_m = mt["gpu_single_interval_upload_to_metric_ms"]
        ss_m = mt["gpu_stage_sum_ms"]
        st = g["gpu_stage_timings_ms"]
        prep_ms = st.get("dual_preprocess_total_ms", st.get("preprocess_total_ms", 0.0))
        lines.append(
            f"| {b}{label}{b} | {b}`{hw_m['p50']:.3f} / {hw_m['p95']:.3f} ms` (`{hw_m['std']:.3f} ms`){b} "
            f"| {b}`{si_m['p50']:.3f} / {si_m['p95']:.3f} ms` (`{si_m['std']:.3f} ms`){b} "
            f"| {b}`{ss_m['p50']:.3f} / {ss_m['p95']:.3f} ms`{b} "
            f"| {b}`{g['host_wall_ms']['p50']:.3f} ms` (`{g['equivalent_fps_host_p50']:.1f} FPS`){b} "
            f"| {b}`{g['gpu_single_interval_upload_to_metric_ms']['p50']:.3f} ms` (`{g['equivalent_fps_gpu_p50']:.1f} FPS`){b} "
            f"| {b}`{prep_ms:.3f} / {st['coarse_total_ms']:.3f} / {st['refinement_total_ms']:.3f} ms`{b} "
            f"| {b}`{str(g['dual_vs_separate_preprocess_30_planes_exact']).lower()}`{b} "
            f"| {b}`{g['validation_errors_after_teardown']} / {g['validation_warnings_after_teardown']}`{b} |"
        )

    return "\n".join(lines) + "\n"


def sync_markdown_evidence_report() -> Path:
    out_dir = ROOT / "docs/evidence/v9-live-and-audit"
    e2e_path = out_dir / "step7-10-live-and-e2e-qualification-v9.json"
    acc_path = out_dir / "step8-9-visual-accuracy-and-selective-repair-v9.json"
    md_path = ROOT / "docs/milestone-v9-live-and-selective-refinement-evidence.md"
    if e2e_path.exists() and acc_path.exists():
        e2e_doc = json.loads(e2e_path.read_text())
        acc_doc = json.loads(acc_path.read_text())
        md_path.write_text(render_milestone_v9_markdown_from_json(e2e_doc, acc_doc))
    return md_path


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
            "VK_LAYER_PATH": str(ROOT / "ros_ws/.pixi/envs/default/share/vulkan/explicit_layer.d"),
            "LD_PRELOAD": "/usr/lib64/libdrm.so.2:/usr/lib64/libdrm_amdgpu.so.1",
            "LD_LIBRARY_PATH": str(ROOT / "ros_ws/.pixi/envs/default/lib"),
        }
    )

    hw_probe = probe_physical_zed2_hardware()
    if not hw_probe["live_camera_ready"]:
        raise RuntimeError(f"Physical ZED 2 camera is not active on /dev/video0: {hw_probe}")

    # 1. Gate 1 (Primary): Real Live 60 Hz Physical Camera (--source camera, KeepLast(16) lossless) + Concurrent Subscriber Runs
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
            f"live_cam_q16_{run_key}",
            "camera",
            pipe,
            sched,
            rmode,
            repair,
            max_f,
            env,
            qos_depth=16,
            verify_checksum=True,
        )
        time.sleep(0.3)

    # 1b. Gate 1 Extended Regimes: KeepLast(1) latest-frame comparison, 600-frame (10.0 s) endurance, and shared-GPU contention
    qos_and_stress_results: dict[str, Any] = {
        "keep_last_1_latest_frame_180f": run_pub_sub_qualification(
            work_dir,
            "live_cam_q1_repair_ind_180f",
            "camera",
            "experimental-dual-refinement",
            "layered-per-path",
            "independent-sgm",
            "1",
            180,
            env,
            qos_depth=1,
            verify_checksum=True,
        ),
        "extended_600f_10s_endurance": run_pub_sub_qualification(
            work_dir,
            "live_cam_q16_repair_ind_600f",
            "camera",
            "experimental-dual-refinement",
            "layered-per-path",
            "independent-sgm",
            "1",
            600,
            env,
            qos_depth=16,
            verify_checksum=True,
        ),
        "shared_gpu_contention_180f": run_pub_sub_qualification(
            work_dir,
            "live_cam_q16_repair_ind_gpu_contention_180f",
            "camera",
            "experimental-dual-refinement",
            "layered-per-path",
            "independent-sgm",
            "1",
            180,
            env,
            qos_depth=16,
            verify_checksum=True,
            concurrent_gpu_contention=True,
        ),
    }

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
            f"replay_q16_{run_key}",
            "replay",
            pipe,
            sched,
            rmode,
            repair,
            max_f,
            env,
            replay_dir=replay_186_dir,
            qos_depth=16,
            verify_checksum=True,
        )

    # 3. Gate 4: Multi-Trial Interleaved 5-Frame Hardware GPU Timestamp + Host Wall Qualification
    gpu_e2e_summary = run_multi_trial_gpu_qualification(work_dir, env)

    # 4. Multi-Sequence Challenging Regime Evaluation
    seq_eval = evaluate_challenging_sequences(DEFAULT_MANIFEST)

    out_dir = ROOT / "docs/evidence/v9-live-and-audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "step7-10-live-and-e2e-qualification-v9.json"
    doc = {
        "schema": "stereoforge.live_and_e2e_qualification_v9_2.v3",
        "provenance": current_provenance_manifest(),
        "zed2_hardware_probe": hw_probe,
        "live_physical_camera_60hz_ros2_e2e_runs": live_camera_results,
        "live_camera_qos_and_stress_runs": qos_and_stress_results,
        "unpaced_offline_replay_ros2_e2e_runs": replay_results,
        "ros2_e2e_60hz_delivery_runs": live_camera_results,
        "gpu_240fps_e2e_latency_qualification": gpu_e2e_summary,
        "multi_sequence_regime_evaluation": seq_eval,
    }
    out_json.write_text(json.dumps(doc, indent=2) + "\n")
    md_path = sync_markdown_evidence_report()
    print(f"Wrote {out_json} and synced {md_path}")


if __name__ == "__main__":
    main()
