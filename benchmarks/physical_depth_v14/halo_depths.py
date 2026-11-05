#!/usr/bin/env python3
"""Dump left image + depth maps of the live-fast, full-continuous and reference estimators for halo scoring.

Step 1 (repo venv): python halo_depths.py FRAMES.npy OUT.npz [--frames 0 5]
Step 2 (any venv with onnxruntime + the U2-Net human-seg model): halo_score.py OUT.npz MODEL.onnx
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from benchmarks.physical_depth_v14 import reference_sgm as R
from benchmarks.physical_depth_v14.audit_rectification import MANIFEST, load_maps, rectified_gray_pair
from benchmarks.physical_depth_v14.baseline_cpu import COARSE, MODES, maps as manifest_maps
from benchmarks.physical_depth_v14.calibrate_sigma import sigma_d
from stereoforge.slang_cpu_backend import SlangCpuRefinementBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--frames-idx", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--sigma-model")
    args = ap.parse_args()
    raws = np.load(args.frames, mmap_mode="r")
    maps = load_maps()
    cal = json.loads(MANIFEST.read_text())["calibration"]
    p1, p2 = np.array(cal["P1"]).reshape(3, 4), np.array(cal["P2"]).reshape(3, 4)
    fx, baseline = float(p1[0, 0]), float(-p2[0, 3] / p2[0, 0])
    fxb = fx * baseline
    coef = np.array(json.load(open(args.sigma_model))["coef"]) if args.sigma_model else None
    backend, tm, cm = SlangCpuRefinementBackend(), manifest_maps(MANIFEST), manifest_maps(COARSE)
    out = {}
    for i in args.frames_idx:
        raw = np.ascontiguousarray(raws[i])
        left, right = rectified_gray_pair(raw, maps)
        out[f"left_{i}"] = left
        for name, mode in MODES.items():
            r = backend.run_full_camera_to_depth_pipeline(
                raw_yuyv=raw.reshape(-1), target_maps=tm, coarse_maps=cm, eye_width=1280, source_height=720, stride_bytes=5120, width=1280, height=720,
                coarse_width=320, coarse_height=180, fx=fx, baseline=baseline, principal_offset=0.0, disparities=128, radius=2, subpixel=True, ortho_hybrid=True, **mode)
            n = 1280 * 720
            valid = r["validity"][:n].reshape(720, 1280) != 0
            out[f"{name}_{i}"] = np.where(valid, r["depth"][:n].reshape(720, 1280), np.nan).astype(np.float32)
        o = R.estimate(left, right)
        d = np.where(o["valid"], o["disparity"], np.nan)
        out[f"reference_{i}"] = (fxb / d).astype(np.float32)
        if coef is not None:
            sig = sigma_d(o["uniqueness"], o["curvature"], coef)
            out[f"reference_trusted_{i}"] = np.where(o["valid"] & (sig <= 0.01 * o["disparity"]), fxb / o["disparity"], np.nan).astype(np.float32)
    np.savez(args.out, **out)
    print("saved", sorted(out))


if __name__ == "__main__":
    main()
