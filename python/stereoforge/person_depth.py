"""Person-mask guided depth refinement (pure numpy/OpenCV; no ROS or model dependencies)."""
from __future__ import annotations

import cv2
import numpy as np


def refine_depth(z: np.ndarray, mask: np.ndarray, halo_px: int = 81, rel_halo: float = 0.12, rel_anchor: float = 0.35) -> np.ndarray:
    """Return a copy of depth `z` (metres, NaN invalid) refined with the person `mask`."""
    h, w = z.shape
    small = (w // 4, h // 4)
    valid = np.isfinite(z) & (z > 0)
    # Morphology on the mask runs at 1/4 resolution; the silhouette itself stays full resolution.
    m_small = cv2.resize(mask.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST)
    ellipse = lambda px: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    inner = cv2.resize(cv2.erode(m_small, ellipse(3)), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    if np.count_nonzero(valid & inner) < 200:
        return z.copy()
    person_depth = float(np.median(z[valid & inner]))
    out = z.copy()
    far = cv2.resize(cv2.dilate(m_small, ellipse(halo_px // 4 | 1)), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    near = cv2.resize(cv2.dilate(m_small, ellipse(3)), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    out[(far & ~near) & valid & (np.abs(z - person_depth) < rel_halo * person_depth)] = np.nan
    anchors = valid & mask & (np.abs(z - person_depth) < rel_anchor * person_depth)
    a_small = cv2.resize(anchors.astype(np.float32), small, interpolation=cv2.INTER_AREA)
    z_small = cv2.resize(np.where(anchors, z, 0.0).astype(np.float32), small, interpolation=cv2.INTER_AREA)
    fill = np.zeros_like(z)
    filled = np.zeros(z.shape, dtype=bool)
    for sigma in (3.0, 12.0):  # fine scale first; the wider scale only fills what is still unsupported
        num = cv2.GaussianBlur(z_small, (0, 0), sigma)
        den = cv2.GaussianBlur(a_small, (0, 0), sigma)
        value = cv2.resize(num / np.maximum(den, 1e-6), (w, h), interpolation=cv2.INTER_LINEAR)
        support = (cv2.resize(den, (w, h), interpolation=cv2.INTER_LINEAR) > 0.02) & ~filled
        fill[support] = value[support]
        filled |= support
    replace = mask & ~anchors & filled
    out[replace] = fill[replace]
    return out


def colorize(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scale = 254.0 / (vmax - vmin)
    u8 = cv2.addWeighted(depth_m, scale, depth_m, 0.0, 1.0 - vmin * scale, dtype=cv2.CV_8U)
    ramp = np.arange(255, -1, -1, dtype=np.uint8).reshape(256, 1)
    lut = np.ascontiguousarray(cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)[:, :, ::-1])
    lut[0] = (20, 20, 20)
    return cv2.applyColorMap(u8, lut)
