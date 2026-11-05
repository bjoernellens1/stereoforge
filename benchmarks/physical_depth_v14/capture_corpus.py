#!/usr/bin/env python3
"""Record one scene of the v14 physical-depth corpus: raw ZED 2 YUYV pairs (+ optional RealSense D435i depth/colour).

  python benchmarks/physical_depth_v14/capture_corpus.py --scene S1_target_1.0m --frames 150 --out ~/stereoforge_corpus [--realsense]

Output (per scene): zed_raw.npy (N,720,2560,2) uint8, zed_t_ns.npy, and with --realsense rs_depth.npy (N,480,640) float32 m
(aligned to colour), rs_color.npy, rs_t_ns.npy, rs_intrinsics.json. Add `--tape-m 1.0` to store the measured distance.
The RealSense half runs under REALSENSE_PYTHON (default miniforge3 python, which has pyrealsense2).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def zed_capture(out: Path, frames: int, device: str) -> None:
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    for _ in range(20):  # let auto-exposure settle
        cap.read()
    raw = np.empty((frames, 720, 2560, 2), np.uint8)
    stamps = np.empty(frames, np.int64)
    for i in range(frames):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("ZED read failed")
        raw[i] = frame.reshape(720, 2560, 2)
        stamps[i] = time.time_ns()
    cap.release()
    np.save(out / "zed_raw.npy", raw)
    np.save(out / "zed_t_ns.npy", stamps)


def realsense_worker(out: Path, frames: int) -> None:
    import numpy as np
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # D435i is on USB 2.1 here: 848x480@30 does not resolve
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipe.start(cfg)
    sensor = profile.get_device().first_depth_sensor()
    if sensor.supports(rs.option.emitter_enabled):
        sensor.set_option(rs.option.emitter_enabled, 1.0)
    if sensor.supports(rs.option.laser_power):
        sensor.set_option(rs.option.laser_power, sensor.get_option_range(rs.option.laser_power).max)
    scale = sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    depth, color, stamps = [], [], []
    for _ in range(30):
        pipe.wait_for_frames()
    while len(depth) < frames:
        fs = align.process(pipe.wait_for_frames())
        d, c = fs.get_depth_frame(), fs.get_color_frame()
        if d and c:
            depth.append(np.asarray(d.get_data()).astype(np.float32) * scale)
            color.append(np.asarray(c.get_data()).copy())
            stamps.append(time.time_ns())
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pipe.stop()
    np.save(out / "rs_depth.npy", np.stack(depth))
    np.save(out / "rs_color.npy", np.stack(color))
    np.save(out / "rs_t_ns.npy", np.asarray(stamps, np.int64))
    (out / "rs_intrinsics.json").write_text(json.dumps({"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy, "width": intr.width, "height": intr.height, "depth_scale": scale}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--out", default=str(Path.home() / "stereoforge_corpus"))
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--tape-m", type=float)
    ap.add_argument("--realsense", action="store_true")
    ap.add_argument("--realsense-worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.realsense_worker:
        realsense_worker(Path(args.realsense_worker), args.frames)
        return
    out = Path(args.out) / args.scene
    out.mkdir(parents=True, exist_ok=True)
    proc = None
    if args.realsense:
        py = os.environ.get("REALSENSE_PYTHON", "/home/bjoern/miniforge3/bin/python3")
        proc = subprocess.Popen([py, __file__, "--realsense-worker", str(out), "--frames", str(args.frames)])
    zed_capture(out, args.frames, args.device)
    if proc is not None and proc.wait() != 0:
        sys.exit("RealSense worker failed")
    (out / "scene.json").write_text(json.dumps({"scene": args.scene, "frames": args.frames, "tape_m": args.tape_m, "realsense": args.realsense, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
