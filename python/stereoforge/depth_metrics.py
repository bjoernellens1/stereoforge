"""Physical depth-quality metrics (numpy/OpenCV only). Depth in metres, NaN or <=0 means invalid."""
from __future__ import annotations

import cv2
import numpy as np


def _valid(z: np.ndarray) -> np.ndarray:
    return np.isfinite(z) & (z > 0)


def range_error_stats(z: np.ndarray, z_ref: np.ndarray, edges=(0.5, 1.0, 1.5, 2.0, 3.0)) -> list[dict]:
    """Bias, MAE, RMSE (mm) and relative RMSE per reference-range bin over pixels valid in both."""
    ok = _valid(z) & _valid(z_ref)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = ok & (z_ref >= lo) & (z_ref < hi)
        if sel.sum() < 50:
            out.append({"range_m": [lo, hi], "n": int(sel.sum())})
            continue
        err = (z[sel] - z_ref[sel]).astype(np.float64)
        out.append({"range_m": [lo, hi], "n": int(sel.sum()), "bias_mm": float(err.mean() * 1e3), "mae_mm": float(np.abs(err).mean() * 1e3),
                    "rmse_mm": float(np.sqrt((err ** 2).mean()) * 1e3), "rel_rmse_pct": float(np.sqrt(((err / z_ref[sel]) ** 2).mean()) * 100)})
    return out


def plane_rmse_mm(z: np.ndarray, mask: np.ndarray, fx: float, fy: float, cx: float, cy: float, iters: int = 200, tol_m: float = 0.01, seed: int = 0) -> dict:
    """RANSAC plane fit to back-projected valid pixels in `mask`; returns inlier orthogonal RMSE (mm) and inlier fraction."""
    ys, xs = np.nonzero(mask & _valid(z))
    if len(xs) < 100:
        return {"n": int(len(xs))}
    d = z[ys, xs].astype(np.float64)
    pts = np.stack([(xs - cx) * d / fx, (ys - cy) * d / fy, d], axis=1)
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inl = np.abs((pts - p[0]) @ n) < tol_m
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    centred = pts[best] - pts[best].mean(0)
    n = np.linalg.svd(centred, full_matrices=False)[2][-1]
    res = centred @ n
    return {"n": int(len(pts)), "inlier_fraction": float(best_n / len(pts)), "rmse_mm": float(np.sqrt((res ** 2).mean()) * 1e3)}


def halo_band_fraction(z: np.ndarray, object_mask: np.ndarray, band_px=(5, 40), rel_tol: float = 0.12) -> dict:
    """Valid pixels in a band *outside* the object whose depth equals the object's median depth (foreground growth)."""
    m8 = object_mask.astype(np.uint8)
    ell = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    band = (cv2.dilate(m8, ell(band_px[1])) > 0) & ~(cv2.dilate(m8, ell(band_px[0])) > 0)
    inner = (cv2.erode(m8, ell(5)) > 0) & _valid(z)
    if inner.sum() < 200:
        return {"n_band_valid": 0}
    ref = float(np.median(z[inner]))
    valid_band = band & _valid(z)
    halo = valid_band & (np.abs(z - ref) < rel_tol * ref)
    return {"object_depth_m": ref, "n_band_valid": int(valid_band.sum()), "n_halo": int(halo.sum()),
            "halo_fraction": float(halo.sum() / max(valid_band.sum(), 1))}


def pixel_locking_index(disparity: np.ndarray) -> float:
    """Max deviation of the fractional-disparity histogram from uniform (0 = no pixel locking, ~1 = fully locked)."""
    d = disparity[np.isfinite(disparity) & (disparity > 0)]
    if d.size < 1000:
        return float("nan")
    hist, _ = np.histogram(d - np.floor(d), bins=10, range=(0.0, 1.0))
    p = hist / hist.sum()
    return float(np.abs(p - 0.1).max() / 0.9)


def temporal_std_mm(stack: np.ndarray, min_frames: int = 20) -> dict:
    """Per-pixel temporal std (mm) over a static (N,H,W) stack, for pixels valid in at least `min_frames` frames."""
    v = _valid(stack)
    n = v.sum(0)
    keep = n >= min_frames
    zm = np.where(v, stack, np.nan)
    std = np.nanstd(zm, axis=0)[keep] * 1e3
    return {"n_pixels": int(keep.sum()), "median_std_mm": float(np.median(std)) if std.size else float("nan"),
            "p90_std_mm": float(np.percentile(std, 90)) if std.size else float("nan")}


def zscore_calibration(err: np.ndarray, sigma: np.ndarray) -> dict:
    """How well sigma predicts error: expected 68.3 % within 1 sigma and 95.4 % within 2 sigma."""
    ok = np.isfinite(err) & np.isfinite(sigma) & (sigma > 0)
    z = np.abs(err[ok] / sigma[ok])
    if z.size < 100:
        return {"n": int(z.size)}
    return {"n": int(z.size), "within_1sigma": float((z < 1).mean()), "within_2sigma": float((z < 2).mean()), "median_abs_z": float(np.median(z))}


def false_valid_rate(z: np.ndarray, z_ref: np.ndarray, rel_tol: float = 0.03) -> dict:
    """Among pixels valid in both, the fraction wrong by more than rel_tol of range (valid-but-wrong)."""
    ok = _valid(z) & _valid(z_ref)
    if ok.sum() < 100:
        return {"n": int(ok.sum())}
    wrong = np.abs(z[ok] - z_ref[ok]) > rel_tol * z_ref[ok]
    return {"n": int(ok.sum()), "false_valid_rate": float(wrong.mean()), "coverage_of_reference": float(ok.sum() / max(_valid(z_ref).sum(), 1))}
