#!/usr/bin/env python3
"""Evaluate the reference SGM prototype on a static clip: coverage, pixel locking, temporal stability.

Usage: PYTHONPATH=python:. python benchmarks/physical_depth_v14/eval_reference.py FRAMES.npy [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.physical_depth_v14 import reference_sgm as R
from benchmarks.physical_depth_v14.audit_rectification import MANIFEST, load_maps, rectified_gray_pair
from stereoforge.depth_metrics import pixel_locking_index, temporal_std_mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--out")
    args = ap.parse_args()
    frames, maps = np.load(args.frames), load_maps()
    cal = json.loads(MANIFEST.read_text())["calibration"]
    p1, p2 = np.array(cal["P1"]).reshape(3, 4), np.array(cal["P2"]).reshape(3, 4)
    fxb = float(p1[0, 0]) * float(-p2[0, 3] / p2[0, 0])
    z_stack, d_stack = [], []
    for raw in frames:
        o = R.estimate(*rectified_gray_pair(raw, maps))
        d = np.where(o["valid"], o["disparity"], np.nan)
        d_stack.append(d)
        z_stack.append(fxb / d)
    z_stack = np.stack(z_stack)
    report = {"coverage_pct": float(np.isfinite(z_stack).mean() * 100),
              "pixel_locking_index": float(np.nanmean([pixel_locking_index(d) for d in d_stack])),
              "temporal": temporal_std_mm(z_stack, min_frames=max(3, len(frames) // 2))}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
