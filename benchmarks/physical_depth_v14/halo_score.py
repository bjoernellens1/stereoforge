#!/usr/bin/env python3
"""Score foreground growth (halo) of depth maps against an independent person segmentation (validation only).

python halo_score.py DEPTHS.npz u2net_human_seg.onnx   (needs numpy, opencv, onnxruntime)
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import onnxruntime as ort

from stereoforge.depth_metrics import halo_band_fraction

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def person_mask(session: ort.InferenceSession, gray: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    small = (cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 - MEAN) / STD
    out = session.run(None, {session.get_inputs()[0].name: small.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return cv2.resize(out, (gray.shape[1], gray.shape[0])) > 0.5


def main() -> None:
    data = np.load(sys.argv[1])
    session = ort.InferenceSession(sys.argv[2], providers=["CPUExecutionProvider"])
    frames = sorted({k.split("_")[-1] for k in data.files if k.startswith("left_")})
    result = {}
    for fi in frames:
        mask = person_mask(session, data[f"left_{fi}"])
        for k in data.files:
            if k.endswith(f"_{fi}") and not k.startswith("left_"):
                z = data[k]
                h = halo_band_fraction(z, mask)
                result.setdefault(k.rsplit("_", 1)[0], []).append({"frame": fi, "mask_pct": float(mask.mean() * 100), "valid_pct": float(np.isfinite(z).mean() * 100), **h})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
