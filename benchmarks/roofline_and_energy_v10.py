"""StereoForge v10.0 — Roofline Model, Per-Stage Memory-Traffic Lower Bounds,
and Joint 3-Metric Profiler (T_frame, E_frame, Delta_T_competing_workload).
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.visual_accuracy_audit_v9 import (
    COARSE_4X_MANIFEST,
    DEFAULT_MANIFEST,
    DUAL_BENCH,
    REFINE_BENCH,
    current_provenance_manifest,
    gpu_env,
)


def _find_amdgpu_hwmon() -> Path | None:
    for p in sorted(Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*")):
        name_f = p / "name"
        if name_f.exists() and "amdgpu" in name_f.read_text().strip():
            return p
    return None


def sample_gpu_telemetry() -> dict[str, float]:
    hw = _find_amdgpu_hwmon()
    out: dict[str, float] = {
        "power_w": 0.0,
        "freq_mhz": 0.0,
        "temp_c": 0.0,
        "gpu_busy_pct": 0.0,
    }
    if hw is not None:
        for pf in ("power1_input", "power1_average"):
            f = hw / pf
            if f.exists():
                try:
                    out["power_w"] = float(int(f.read_text().strip())) * 1e-6
                    break
                except Exception:
                    pass
        ff = hw / "freq1_input"
        if ff.exists():
            try:
                out["freq_mhz"] = float(int(ff.read_text().strip())) * 1e-6
            except Exception:
                pass
        tf = hw / "temp1_input"
        if tf.exists():
            try:
                out["temp_c"] = float(int(tf.read_text().strip())) * 1e-3
            except Exception:
                pass
    for bf in sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent")):
        try:
            out["gpu_busy_pct"] = max(out["gpu_busy_pct"], float(int(bf.read_text().strip())))
        except Exception:
            pass
    return out


def wait_gpu_idle_hysteresis(max_busy_pct: float = 12.0, streak_needed: int = 4, timeout_s: float = 40.0) -> None:
    t0 = time.monotonic()
    streak = 0
    while time.monotonic() - t0 < timeout_s:
        if sample_gpu_telemetry()["gpu_busy_pct"] <= max_busy_pct:
            streak += 1
            if streak >= streak_needed:
                return
        else:
            streak = 0
        time.sleep(0.25)


def compute_stage_memory_traffic_and_roofline(
    stage_timings_ms: dict[str, float],
    width: int = 1280,
    height: int = 720,
    coarse_scale: int = 4,
    coarse_disparities: int = 32,
    effective_peak_bw_gbps: float = 288.0,
) -> dict[str, Any]:
    """Compute analytical global-memory byte traffic and bandwidth-roofline floor per GPU stage."""
    n_full = width * height  # 921,600 px per eye
    n_two_eyes = 2 * n_full  # 1,843,200 px across both eyes
    n_coarse = (width // coarse_scale) * (height // coarse_scale)  # 57,600 px per eye
    n_coarse_two = 2 * n_coarse  # 115,200 px

    # 1. Dual Preprocess (upload YUYV + decode + bilinear remap for 2 eyes at full + 4x coarse)
    # Reads: 2 * n_full * 2B (YUYV) + remap lut reads (2 * (n_full + n_coarse) * 8B)
    # Writes: full RGB+luma+support + coarse luma+support (~32B per full px + 16B per coarse px)
    pre_read_bytes = n_two_eyes * 2 + (n_two_eyes + n_coarse_two) * 8
    pre_write_bytes = n_two_eyes * 24 + n_coarse_two * 12

    # 2. Coarse SGM (320x180, D=32, 4-path SGM aggregation + select_output + LR check)
    # Cost volume + 4 directional path volumes (u16/u32) + coarse output planes
    coarse_read_bytes = n_coarse_two * coarse_disparities * 4 * 4
    coarse_write_bytes = n_coarse_two * coarse_disparities * 4 * 2 + n_coarse_two * 16

    # 3. Full-Res Census (ortho-hybrid 32-bit descriptor for 2 eyes)
    # Reads: luma + support (8B/px with halo cache); Writes: census (4B) + census_validity (4B)
    census_read_bytes = int(n_two_eyes * 8 * 1.25)
    census_write_bytes = n_two_eyes * 8

    # 4. Full-Res Guided Match (coarse prior lookup + +-2 search + 8 diagnostic planes + 3 raw planes)
    # Note: writing 8 diagnostic planes (uplane 13..20) = 8 * 4B * 1.843M = 58.98 MB/frame!
    match_read_bytes = n_two_eyes * (8 + 5 * 8) + n_coarse_two * 8
    match_diag_write_bytes = n_two_eyes * 8 * 4  # 58,982,400 B (56.25 MiB) of diagnostic planes
    match_core_write_bytes = n_two_eyes * 3 * 4  # raw_disparity, raw_confidence, raw_validity (22.12 MB)
    match_write_bytes = match_core_write_bytes + match_diag_write_bytes

    # 5. Left-Right Consistency (lr)
    # Reads: raw_disparity + raw_confidence + raw_validity (left & right); Writes: disp, conf, val (+1 diag)
    lr_read_bytes = n_two_eyes * 16
    lr_write_bytes = n_two_eyes * 16

    # 6. Metric Depth + Completed-Buffer Initialization (metric)
    # Reads: disp, conf, val, support (16B * 2 eyes = 29.5 MB)
    # Writes: measured output (z, conf, val, prov, source_id = 24B * 2 eyes = 44.2 MB)
    #       + completed output (z, conf, val, prov, source_id, disp = 28B * 2 eyes = 51.6 MB)
    metric_read_bytes = n_two_eyes * 16
    metric_measured_write_bytes = n_two_eyes * 24
    metric_completed_two_eye_write_bytes = n_two_eyes * 28  # 51.61 MB (25.8 MB is redundant right-eye copy!)
    metric_write_bytes = metric_measured_write_bytes + metric_completed_two_eye_write_bytes

    # 7. Planar Recovery (planar)
    # Workgroup early-exit skips ~55% of tiles; active tiles read luma + disp + coarse + right census
    active_tile_fraction = 0.45
    planar_read_bytes = int(active_tile_fraction * n_full * 28)
    planar_write_bytes = int(0.035 * n_full * 28)

    stages_spec = [
        ("dual_preprocess", pre_read_bytes, pre_write_bytes, float(stage_timings_ms["dual_preprocess_total_ms"])),
        ("coarse_sgm", coarse_read_bytes, coarse_write_bytes, float(stage_timings_ms["coarse_total_ms"])),
        ("refinement_census", census_read_bytes, census_write_bytes, float(stage_timings_ms["refinement_census_ms"])),
        ("refinement_guided_match", match_read_bytes, match_write_bytes, float(stage_timings_ms["refinement_guided_match_ms"])),
        ("refinement_lr", lr_read_bytes, lr_write_bytes, float(stage_timings_ms["refinement_lr_ms"])),
        ("refinement_metric", metric_read_bytes, metric_write_bytes, float(stage_timings_ms["refinement_metric_ms"])),
        ("refinement_planar_recovery", planar_read_bytes, planar_write_bytes, float(stage_timings_ms["refinement_planar_recovery_ms"])),
    ]

    stage_rows = []
    total_read = 0
    total_write = 0
    total_roofline_ms = 0.0
    for name, r_b, w_b, t_ms in stages_spec:
        tot_b = r_b + w_b
        total_read += r_b
        total_write += w_b
        bw_floor_ms = (tot_b / (effective_peak_bw_gbps * 1e9)) * 1e3 + 0.012
        total_roofline_ms += bw_floor_ms
        achieved_gbps = (tot_b / max(t_ms * 1e-3, 1e-9)) * 1e-9
        stage_rows.append(
            {
                "stage": name,
                "read_mb": round(r_b * 1e-6, 3),
                "write_mb": round(w_b * 1e-6, 3),
                "total_traffic_mb": round(tot_b * 1e-6, 3),
                "measured_gpu_ms": round(t_ms, 4),
                "roofline_bandwidth_floor_ms": round(bw_floor_ms, 4),
                "achieved_effective_bw_gbps": round(achieved_gbps, 2),
            }
        )

    # Quantify the exact v10.2 memory-traffic savings from architectural fusion & write elimination:
    # 1. Skip 8 diagnostic planes in guided_match: -58.98 MB
    # 2. Fuse lr + metric (eliminating intermediate disp/conf/val global write+read): -44.24 MB
    # 3. Left-eye-only completed buffer initialization in metric: -25.80 MB
    eliminable_bytes_v102 = (
        match_diag_write_bytes
        + (lr_write_bytes + metric_read_bytes - n_two_eyes * 4)
        + (metric_completed_two_eye_write_bytes // 2)
    )

    return {
        "effective_reference_bandwidth_gbps": effective_peak_bw_gbps,
        "per_stage_breakdown": stage_rows,
        "total_pipeline_read_mb": round(total_read * 1e-6, 2),
        "total_pipeline_write_mb": round(total_write * 1e-6, 2),
        "total_pipeline_traffic_mb": round((total_read + total_write) * 1e-6, 2),
        "total_measured_stage_sum_ms": round(sum(r["measured_gpu_ms"] for r in stage_rows), 4),
        "total_bandwidth_roofline_floor_ms": round(total_roofline_ms, 4),
        "v10_2_architectural_traffic_reduction_plan": {
            "guided_match_diagnostic_plane_writes_mb": round(match_diag_write_bytes * 1e-6, 2),
            "lr_metric_fusion_intermediate_traffic_mb": round((lr_write_bytes + metric_read_bytes - n_two_eyes * 4) * 1e-6, 2),
            "right_eye_completed_buffer_init_writes_mb": round((metric_completed_two_eye_write_bytes // 2) * 1e-6, 2),
            "total_eliminable_global_memory_traffic_mb": round(eliminable_bytes_v102 * 1e-6, 2),
        },
    }


def measure_joint_3metric_tuple(work_dir: Path) -> dict[str, Any]:
    """Measure the joint 3-metric tuple (T_frame, E_frame, Delta_T_competing_workload)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    env = gpu_env()
    wait_gpu_idle_hysteresis()

    # 1. Measure baseline idle GPU power over 2.0 s after a 2.0 s settle period
    time.sleep(2.0)
    idle_samples = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        idle_samples.append(sample_gpu_telemetry())
        time.sleep(0.05)
    idle_power_w = float(np.percentile([s["power_w"] for s in idle_samples], 25))
    idle_temp_c = float(np.median([s["temp_c"] for s in idle_samples]))

    # 2. Measure T_frame and E_frame during a sustained 240-frame GPU run
    active_samples: list[dict[str, float]] = []
    stop_flag = False

    def _poller() -> None:
        while not stop_flag:
            active_samples.append(sample_gpu_telemetry())
            time.sleep(0.015)

    out_json = work_dir / "v10_0_sustained_240.json"
    out_json.unlink(missing_ok=True)
    poller_thread = threading.Thread(target=_poller, daemon=True)
    t_start = time.monotonic()
    poller_thread.start()
    subprocess.run(
        [
            str(DUAL_BENCH),
            "--coarse-manifest", str(COARSE_4X_MANIFEST),
            "--target-manifest", str(DEFAULT_MANIFEST),
            "--coarse-disparities", "32",
            "--descriptor-mode", "ortho-hybrid",
            "--fast-mode", "1",
            "--selective-repair", "1",
            "--planar-recovery", "1",
            "--planar-min-anchors", "10",
            "--planar-coarse-affine", "1",
            "--planar-ray-crossing-guard", "1",
            "--planar-coarse-lr-guard", "1",
            "--radius", "2", "--subpixel", "1",
            "--coarse-schedule", "shared-census-sequential",
            "--coarse-right-reference-mode", "left-aggregated-selection",
            "--frames", "240", "--warmup", "20",
            "--output", str(out_json),
        ],
        env=env,
        check=True,
    )
    t_elapsed = time.monotonic() - t_start
    stop_flag = True
    poller_thread.join(timeout=1.0)

    bench_doc = json.loads(out_json.read_text())
    t_frame_gpu_p50 = float(bench_doc["gpu_combined_queue_span_ms"]["p50"])
    t_frame_gpu_p95 = float(bench_doc["gpu_combined_queue_span_ms"]["p95"])
    t_frame_gpu_p99 = float(bench_doc["gpu_combined_queue_span_ms"]["p99"])
    t_frame_wall_p50 = float(bench_doc["host_wall_ms"]["p50"])

    # Compute active GPU power & energy per frame (260 total frames = 20 warmup + 240 measured)
    total_frames = 260
    active_powers = [s["power_w"] for s in active_samples if s["gpu_busy_pct"] >= 10.0] or [
        s["power_w"] for s in active_samples
    ]
    mean_active_power_w = float(np.mean(active_powers)) if active_powers else idle_power_w
    peak_active_power_w = float(np.max(active_powers)) if active_powers else idle_power_w
    gross_energy_mj_per_frame = mean_active_power_w * t_frame_gpu_p50
    net_dynamic_power_w = max(0.0, mean_active_power_w - idle_power_w)
    net_dynamic_energy_mj_per_frame = net_dynamic_power_w * t_frame_gpu_p50

    # 3. Measure Delta_T_competing_workload:
    # Run a repeatable reference GPU compute workload (`REFINE_BENCH` 40 frames) standalone vs
    # concurrently with background StereoForge GPU load.
    wait_gpu_idle_hysteresis()
    probe_json = work_dir / "competing_probe_solo.json"
    probe_json.unlink(missing_ok=True)
    probe_cmd = [
        str(REFINE_BENCH),
        "--coarse-manifest", str(COARSE_4X_MANIFEST),
        "--target-manifest", str(DEFAULT_MANIFEST),
        "--variant", "raw_borrow",
        "--frames", "40", "--warmup", "5",
        "--coarse-disparities", "32",
        "--coarse-schedule", "shared-census-sequential",
        "--coarse-right-reference-mode", "left-aggregated-selection",
        "--descriptor-mode", "ortho-hybrid",
        "--fast-mode", "1",
        "--selective-repair", "0",
        "--planar-recovery", "0",
        "--radius", "2", "--subpixel", "1",
        "--output", str(probe_json),
    ]
    t_p0 = time.monotonic()
    subprocess.run(probe_cmd, env=env, check=True, stdout=subprocess.DEVNULL)
    solo_wall_ms = (time.monotonic() - t_p0) * 1000.0
    solo_doc = json.loads(probe_json.read_text())
    solo_gpu_p50_ms = float(solo_doc["gpu_combined_queue_span_ms"]["p50"])

    # Now run the same competing probe while StereoForge v9.4 runs continuously in the background
    bg_json = work_dir / "bg_load.json"
    bg_json.unlink(missing_ok=True)
    probe_cont_json = work_dir / "competing_probe_contended.json"
    probe_cont_json.unlink(missing_ok=True)
    bg_proc = subprocess.Popen(
        [
            str(DUAL_BENCH),
            "--coarse-manifest", str(COARSE_4X_MANIFEST),
            "--target-manifest", str(DEFAULT_MANIFEST),
            "--coarse-disparities", "32",
            "--descriptor-mode", "ortho-hybrid",
            "--fast-mode", "1",
            "--selective-repair", "1",
            "--planar-recovery", "1",
            "--radius", "2", "--subpixel", "1",
            "--coarse-schedule", "shared-census-sequential",
            "--coarse-right-reference-mode", "left-aggregated-selection",
            "--frames", "240", "--warmup", "20",
            "--output", str(bg_json),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
    )
    time.sleep(0.35)
    probe_cont_json = work_dir / "competing_probe_contended.json"
    probe_cmd_cont = list(probe_cmd)
    probe_cmd_cont[-1] = str(probe_cont_json)
    t_p1 = time.monotonic()
    subprocess.run(probe_cmd_cont, env=env, check=True, stdout=subprocess.DEVNULL)
    contended_wall_ms = (time.monotonic() - t_p1) * 1000.0
    bg_proc.wait(timeout=30.0)
    cont_doc = json.loads(probe_cont_json.read_text())
    contended_gpu_p50_ms = float(cont_doc["gpu_combined_queue_span_ms"]["p50"])

    delta_gpu_ms = max(0.0, contended_gpu_p50_ms - solo_gpu_p50_ms)
    delta_gpu_pct = 100.0 * delta_gpu_ms / max(solo_gpu_p50_ms, 1e-6)

    # Duty-cycle GPU time fractions at 30 Hz and 60 Hz
    duty_30hz_pct = 100.0 * t_frame_gpu_p50 / (1000.0 / 30.0)
    duty_60hz_pct = 100.0 * t_frame_gpu_p50 / (1000.0 / 60.0)

    roofline = compute_stage_memory_traffic_and_roofline(bench_doc["gpu_stage_timings_ms"])

    return {
        "schema": "stereoforge.v10_0_roofline_and_energy_baseline.v1",
        "provenance": current_provenance_manifest(),
        "joint_3_metric_tuple": {
            "T_frame_ms": {
                "gpu_combined_p50_ms": round(t_frame_gpu_p50, 4),
                "gpu_combined_p95_ms": round(t_frame_gpu_p95, 4),
                "gpu_combined_p99_ms": round(t_frame_gpu_p99, 4),
                "host_wall_p50_ms": round(t_frame_wall_p50, 4),
                "serialized_gpu_time_fraction_at_30hz_pct": round(duty_30hz_pct, 2),
                "serialized_gpu_time_fraction_at_60hz_pct": round(duty_60hz_pct, 2),
            },
            "E_frame_mj": {
                "idle_gpu_power_w": round(idle_power_w, 2),
                "idle_gpu_temp_c": round(idle_temp_c, 1),
                "active_gpu_power_mean_w": round(mean_active_power_w, 2),
                "active_gpu_power_peak_w": round(peak_active_power_w, 2),
                "net_dynamic_power_w": round(net_dynamic_power_w, 2),
                "gross_energy_mj_per_frame": round(gross_energy_mj_per_frame, 3),
                "net_dynamic_energy_mj_per_frame": round(net_dynamic_energy_mj_per_frame, 3),
                "sustained_run_elapsed_s": round(t_elapsed, 3),
                "sustained_run_frames": total_frames,
            },
            "Delta_T_competing_workload": {
                "competing_probe_solo_gpu_p50_ms": round(solo_gpu_p50_ms, 4),
                "competing_probe_contended_gpu_p50_ms": round(contended_gpu_p50_ms, 4),
                "delta_competing_gpu_p50_ms": round(delta_gpu_ms, 4),
                "delta_competing_gpu_p50_pct": round(delta_gpu_pct, 2),
                "competing_probe_solo_wall_ms": round(solo_wall_ms, 2),
                "competing_probe_contended_wall_ms": round(contended_wall_ms, 2),
                "estimated_60hz_duty_scaled_slowdown_pct": round(delta_gpu_pct * (duty_60hz_pct / 100.0), 2),
                "estimated_30hz_duty_scaled_slowdown_pct": round(delta_gpu_pct * (duty_30hz_pct / 100.0), 2),
            },
        },
        "roofline_and_memory_traffic_model": roofline,
        "stage_timings_ms": bench_doc["gpu_stage_timings_ms"],
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "benchmarks/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/sf_v10_0_roofline")
    report = measure_joint_3metric_tuple(work)
    out_path = out_dir / "v10_0_roofline_and_energy_baseline.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path}")
