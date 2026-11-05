#!/usr/bin/env python3
"""Phase 1: self-calibrate the residual vertical disparity of the factory rectification from natural scenes.

Fits dy(x, y) = a + b*u + c*v (u,v in [-0.5,0.5]) from matched features and writes `calibration_overlay.json`
(a dy correction applied to the right-eye rectification map). Validates on held-out frames by re-auditing with the
corrected maps. Usage: PYTHONPATH=python:. python benchmarks/physical_depth_v14/refine_calibration.py FRAMES.npy --out overlay.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.physical_depth_v14.audit_rectification import load_maps
from stereoforge.epipolar_audit import epipolar_samples, fit_dy_plane, summarize_samples

SHAPE = (720, 1280)


def rect_pair(raw: np.ndarray, maps: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_YUYV)
    left = cv2.remap(rgb[:, :1280], maps[0], maps[1], cv2.INTER_LINEAR)
    right = cv2.remap(rgb[:, 1280:], maps[2], maps[3], cv2.INTER_LINEAR)
    return cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)


def apply_overlay(maps: list[np.ndarray], coef: np.ndarray) -> list[np.ndarray]:
    """Shift the right-eye y map so that the measured vertical disparity plane becomes zero."""
    h, w = SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = coef[0] + coef[1] * (xx - w / 2) / w + coef[2] * (yy - h / 2) / h
    return [maps[0], maps[1], maps[2], (maps[3] + dy).astype(np.float32)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--out")
    ap.add_argument("--holdout", type=int, default=3, help="number of trailing frames held out for validation")
    args = ap.parse_args()
    frames = np.load(args.frames)
    maps = load_maps()
    train, test = frames[: len(frames) - args.holdout], frames[len(frames) - args.holdout:]
    pl, dy = [], []
    for raw in train:
        p, d, _ = epipolar_samples(*rect_pair(raw, maps))
        pl.append(p)
        dy.append(d)
    pl, dy = np.concatenate(pl), np.concatenate(dy)
    coef = fit_dy_plane(pl, dy, SHAPE)
    corrected = apply_overlay(maps, coef)
    result = {"coef_px": coef.tolist(), "train_samples": int(len(dy))}
    for name, m in (("factory", maps), ("overlay", corrected)):
        reps = [summarize_samples(*epipolar_samples(*rect_pair(raw, m)), SHAPE) for raw in test]
        result[f"heldout_{name}"] = {"median_dy_px": float(np.median([r.median_dy_px for r in reps])),
                                    "robust_rms_dy_px": float(np.median([r.robust_rms_dy_px for r in reps])),
                                    "tilt_px": float(np.nanmedian([r.tilt_px for r in reps]))}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"schema": "stereoforge.calibration_overlay.v1", "right_dy_plane_px": coef.tolist(), "validation": result}, indent=2))


if __name__ == "__main__":
    main()
