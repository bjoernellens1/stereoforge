"""Epipolar self-check for rectified stereo pairs (pure OpenCV/numpy).

Rectification is correct when matching features share a row: the vertical disparity (dy) of true matches must be
~0 and uniform over the image. A median dy offset or a spatial dy gradient reveals calibration error
(rotation/tilt, cy offset); it is measured from the images themselves, so it also works as a live health check.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class EpipolarReport:
    matches: int
    median_dy_px: float
    robust_rms_dy_px: float
    dy_grid_median_px: np.ndarray  # 3x3 median dy per image region (NaN when too few matches)
    median_dx_px: float

    @property
    def tilt_px(self) -> float:
        """Peak-to-peak spatial variation of the vertical disparity across the 3x3 grid."""
        g = self.dy_grid_median_px[np.isfinite(self.dy_grid_median_px)]
        return float(g.max() - g.min()) if g.size else float("nan")


def _subpixel_dy(left_gray: np.ndarray, right_gray: np.ndarray, pl: np.ndarray, dy: np.ndarray, dx: np.ndarray, half: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Refine integer keypoint offsets by ZNCC template matching with a parabolic peak; returns (dy, valid)."""
    h, w = left_gray.shape
    out = np.full(len(pl), np.nan)
    margin = 4
    for i, ((x, y), ddy, ddx) in enumerate(zip(pl, dy, dx)):
        xl, yl = int(round(x)), int(round(y))
        xr, yr = int(round(x - ddx)), int(round(y + ddy))
        if min(xl, yl, xr, yr) < half + margin or xl + half + margin > w or xr + half + margin > w or yl + half + margin > h or yr + half + margin > h:
            continue
        tmpl = left_gray[yl - half:yl + half + 1, xl - half:xl + half + 1].astype(np.float32)
        area = right_gray[yr - half - margin:yr + half + margin + 1, xr - half - margin:xr + half + margin + 1].astype(np.float32)
        # Reject near one-dimensional texture (stripes): the vertical offset is unobservable along them.
        gy_, gx_ = np.gradient(tmpl)
        ex, ey = float((gx_ ** 2).sum()), float((gy_ ** 2).sum())
        if min(ex, ey) < 0.25 * max(ex, ey):
            continue
        res = cv2.matchTemplate(area, tmpl, cv2.TM_CCOEFF_NORMED)
        _, peak, _, (px, py) = cv2.minMaxLoc(res)
        if peak < 0.8 or not (1 <= px < res.shape[1] - 1 and 1 <= py < res.shape[0] - 1):
            continue
        c, up, dn = res[py, px], res[py - 1, px], res[py + 1, px]
        denom = up - 2 * c + dn
        sub = 0.5 * (up - dn) / denom if denom < -1e-6 else 0.0
        out[i] = (yr - margin + py + sub) - (yr - margin + margin) + (yr - yl)  # = (py - margin) + sub + (yr - yl)
    return out, np.isfinite(out)


def audit_epipolar(left_gray: np.ndarray, right_gray: np.ndarray, max_features: int = 4000, max_abs_dy: float = 3.0) -> EpipolarReport:
    """Match features between rectified grey images and summarise vertical disparity."""
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=10)
    kl, dl = orb.detectAndCompute(left_gray, None)
    kr, dr = orb.detectAndCompute(right_gray, None)
    if dl is None or dr is None:
        return EpipolarReport(0, float("nan"), float("nan"), np.full((3, 3), np.nan), float("nan"))
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dl, dr, k=2)
    pts_l, pts_r = [], []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            pts_l.append(kl[pair[0].queryIdx].pt)
            pts_r.append(kr[pair[0].trainIdx].pt)
    pl, pr = np.asarray(pts_l, np.float64).reshape(-1, 2), np.asarray(pts_r, np.float64).reshape(-1, 2)
    dy = pr[:, 1] - pl[:, 1]
    dx = pl[:, 0] - pr[:, 0]  # disparity (left x minus right x)
    keep = (np.abs(dy) < max_abs_dy) & (dx > 0)
    pl, dy, dx = pl[keep], dy[keep], dx[keep]
    dy_sub, ok = _subpixel_dy(left_gray, right_gray, pl, dy, dx)
    pl, dy, dx = pl[ok], dy_sub[ok], dx[ok]
    if len(dy) < 30:
        return EpipolarReport(int(len(dy)), float("nan"), float("nan"), np.full((3, 3), np.nan), float("nan"))
    med = float(np.median(dy))
    mad = float(np.median(np.abs(dy - med)))
    h, w = left_gray.shape
    grid = np.full((3, 3), np.nan)
    gx = np.minimum((pl[:, 0] * 3 / w).astype(int), 2)
    gy = np.minimum((pl[:, 1] * 3 / h).astype(int), 2)
    for iy in range(3):
        for ix in range(3):
            sel = (gx == ix) & (gy == iy)
            if sel.sum() >= 10:
                grid[iy, ix] = np.median(dy[sel])
    return EpipolarReport(int(len(dy)), med, 1.4826 * mad, grid, float(np.median(dx)))
