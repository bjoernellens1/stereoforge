#!/usr/bin/env python3
"""Phase 0/1: measure vertical disparity of rectified ZED 2 pairs from raw YUYV frames.

Usage: PYTHONPATH=python:. python benchmarks/physical_depth_v14/audit_rectification.py FRAMES.npy [--out report.json]
FRAMES.npy: uint8 array (N,720,2560,2) of raw packed YUYV pairs (left|right).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from stereoforge.epipolar_audit import audit_epipolar

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/cameras/zed2/fixtures/SN26041639-fixture/manifest.json"


def load_maps() -> list[np.ndarray]:
    m = json.loads(MANIFEST.read_text())
    return [np.fromfile(MANIFEST.parent / m["maps"][k]["file"], dtype="<f4").reshape(720, 1280) for k in ("left_x", "left_y", "right_x", "right_y")]


def rectified_gray_pair(raw: np.ndarray, maps: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_YUYV)
    left = cv2.remap(rgb[:, :1280], maps[0], maps[1], cv2.INTER_LINEAR)
    right = cv2.remap(rgb[:, 1280:], maps[2], maps[3], cv2.INTER_LINEAR)
    return cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--out")
    args = ap.parse_args()
    frames = np.load(args.frames)
    maps = load_maps()
    rows = []
    for i, raw in enumerate(frames):
        r = audit_epipolar(*rectified_gray_pair(raw, maps))
        rows.append({"frame": i, "matches": r.matches, "median_dy_px": r.median_dy_px, "robust_rms_dy_px": r.robust_rms_dy_px,
                     "tilt_px": r.tilt_px, "dy_grid_median_px": np.round(r.dy_grid_median_px, 3).tolist(), "median_dx_px": r.median_dx_px})
    ok = [x for x in rows if x["matches"] >= 30]
    summary = {"frames": len(rows), "usable": len(ok),
               "median_dy_px": float(np.median([x["median_dy_px"] for x in ok])) if ok else None,
               "robust_rms_dy_px": float(np.median([x["robust_rms_dy_px"] for x in ok])) if ok else None,
               "tilt_px": float(np.median([x["tilt_px"] for x in ok])) if ok else None}
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "frames": rows}, indent=2))


if __name__ == "__main__":
    main()
