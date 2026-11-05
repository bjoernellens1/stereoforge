#!/usr/bin/env python3
"""Calibrate the per-pixel disparity uncertainty model sigma_d(uniqueness, curvature) from temporal scatter.

On a static clip the per-pixel temporal std of disparity is an unbiased estimate of random error. We regress
log(std) on log(uniqueness) and log(curvature) of the *same* pixels (fit on even pixels, evaluated on odd pixels)
and write `sigma_model_v1.json`. Systematic bias (calibration) is a separate term measured against physical references.
Usage: python benchmarks/physical_depth_v14/calibrate_sigma.py STACK.npz --out sigma_model_v1.json
STACK.npz has D (N,H,W) disparity, V valid, C curvature, U uniqueness from reference_sgm.estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEATURES = ("log_uniq", "log_curv")


def features(uniq: np.ndarray, curv: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(uniq), np.log(np.maximum(uniq, 0.0) + 0.02), np.log(np.maximum(curv, 0.0) + 0.5)], axis=-1)


def sigma_d(uniq: np.ndarray, curv: np.ndarray, coef: np.ndarray, floor: float = 0.03, ceil: float = 8.0) -> np.ndarray:
    return np.clip(np.exp(features(uniq, curv) @ coef), floor, ceil)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack")
    ap.add_argument("--out")
    args = ap.parse_args()
    s = np.load(args.stack)
    D, V, C, U = s["D"], s["V"], s["C"], s["U"]
    enough = V.sum(0) >= max(3, len(D) * 6 // 10)
    Dm = np.where(V, D, np.nan)
    mean, std = np.nanmean(Dm, 0), np.nanstd(Dm, 0)
    uniq = np.nanmean(np.where(V, U, np.nan), 0)
    curv = np.nanmean(np.where(V, C, np.nan), 0)
    ok = enough & np.isfinite(std) & np.isfinite(uniq) & np.isfinite(curv) & (std < 20)
    ys, xs = np.nonzero(ok)
    fit = (ys + xs) % 2 == 0
    X = features(uniq[ys, xs], curv[ys, xs])
    y = np.log(np.maximum(std[ys, xs], 0.02))
    coef = np.linalg.lstsq(X[fit], y[fit], rcond=None)[0]
    sig = sigma_d(uniq[ys, xs], curv[ys, xs], coef)
    held = ~fit
    # coverage of the model on held-out pixels using the individual frame residuals around the clip mean
    resid = np.abs(D[:, ys, xs] - mean[ys, xs]) / sig
    valid = V[:, ys, xs]
    within1 = float((resid[:, held][valid[:, held]] < 1).mean())
    within2 = float((resid[:, held][valid[:, held]] < 2).mean())
    corr = float(np.corrcoef(np.log(sig[held]), np.log(np.maximum(std[ys, xs][held], 0.02)))[0, 1])
    report = {"schema": "stereoforge.sigma_model.v1", "features": ["1", "log(uniq+0.02)", "log(curv+0.5)"], "coef": coef.tolist(),
              "heldout": {"n_pixels": int(held.sum()), "log_corr": corr, "within_1sigma": within1, "within_2sigma": within2,
                          "median_sigma_px": float(np.median(sig[held])), "median_measured_std_px": float(np.median(std[ys, xs][held]))},
              "note": "random error only (temporal scatter of a static scene); bias is validated separately against targets/RealSense"}
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
