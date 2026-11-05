#!/usr/bin/env python3
"""Baseline physical-quality numbers of the current estimator modes on recorded raw ZED 2 frames (CPU Slang backend).

Usage: PYTHONPATH=python:benchmarks python benchmarks/physical_depth_v14/baseline_cpu.py FRAMES.npy [--out report.json]
Measures coverage, pixel-locking index and temporal std over the (static) clip; no ground truth is needed for these.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stereoforge.depth_metrics import pixel_locking_index, temporal_std_mm
from stereoforge.slang_cpu_backend import SlangCpuRefinementBackend

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json"
COARSE = ROOT / "artifacts/depth-video/v8-review-milestone/fixtures/zed2_320x180/manifest.json"
MODES = {
    "fast_live_default": dict(adaptive_tiles=True, fast_mode=True, ultra_fast=True, continuous_subpixel=False, planar_recovery=True, selective_repair=True),
    "full_continuous_subpixel": dict(adaptive_tiles=False, fast_mode=False, ultra_fast=False, continuous_subpixel=True, planar_recovery=False, selective_repair=True),
}


def maps(manifest: Path) -> list[np.ndarray]:
    m = json.loads(manifest.read_text())
    return [np.fromfile(manifest.parent / m["maps"][k]["file"], dtype="<f4") for k in ("left_x", "left_y", "right_x", "right_y")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--out")
    args = ap.parse_args()
    frames = np.load(args.frames)
    cal = json.loads(FULL.read_text())["calibration"]
    p1, p2 = np.array(cal["P1"]).reshape(3, 4), np.array(cal["P2"]).reshape(3, 4)
    fx, baseline = float(p1[0, 0]), float(-p2[0, 3] / p2[0, 0])
    backend, t_maps, c_maps = SlangCpuRefinementBackend(), maps(FULL), maps(COARSE)
    report = {}
    for name, mode in MODES.items():
        z_stack, d_stack = [], []
        for raw in frames:
            r = backend.run_full_camera_to_depth_pipeline(
                raw_yuyv=np.ascontiguousarray(raw).reshape(-1), target_maps=t_maps, coarse_maps=c_maps,
                eye_width=1280, source_height=720, stride_bytes=5120, width=1280, height=720, coarse_width=320, coarse_height=180,
                fx=fx, baseline=baseline, principal_offset=0.0, disparities=128, radius=2, subpixel=True, ortho_hybrid=True, **mode)
            n = 1280 * 720
            valid = r["validity"][:n].reshape(720, 1280) != 0
            z_stack.append(np.where(valid, r["depth"][:n].reshape(720, 1280), np.nan))
            d_stack.append(np.where(valid, r["disparity"][:n].reshape(720, 1280), np.nan))
        z_stack, d_stack = np.stack(z_stack), np.stack(d_stack)
        report[name] = {"coverage_pct": float(np.isfinite(z_stack).mean() * 100),
                        "pixel_locking_index": float(np.nanmean([pixel_locking_index(d) for d in d_stack])),
                        "temporal": temporal_std_mm(z_stack, min_frames=max(3, len(frames) // 2))}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
