"""Native RealSense reference geometry, used only by comparison tools.

Use native uint16 depth and its own recorded factory intrinsics. SDK alignment
retains native Z at color pixels; that image is not color-camera axial depth.
Deprojection delegates distortion to the installed SDK rather than approximating
it. Source semantics: librealsense v2.58.1 src/rs.cpp, rs2_deproject_pixel_to_point.
No rig transform, clock association or absolute accuracy is qualified here.
"""
from __future__ import annotations

import numpy as np


def native_points(depth: np.ndarray, factory: dict, pixels: np.ndarray, rs) -> np.ndarray:
    """Return Nx3 metres in the depth optical frame for integer native (x,y).

    Missing depth and out-of-image samples stay NaN; there is no clipping,
    interpolation or spatial completion. ``rs`` is the installed pyrealsense2.
    """
    depth, pixels = np.asarray(depth), np.asarray(pixels)
    if factory.get('schema') != 'stereoforge.native_realsense_factory.v1':
        raise ValueError('native factory schema required')
    camera, scale = factory['depth'], factory['depth_scale_m']
    if depth.dtype != np.uint16 or depth.shape != (camera['height'], camera['width']):
        raise ValueError('native uint16 depth must match its factory dimensions')
    values = [scale, camera['fx'], camera['fy'], camera['ppx'], camera['ppy']]
    coeffs = np.asarray(camera['coeffs'], dtype=float)
    if not np.isfinite(values).all() or min(values[:3]) <= 0 or coeffs.shape != (5,) or not np.isfinite(coeffs).all():
        raise ValueError('invalid native scale or intrinsics')
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.issubdtype(pixels.dtype, np.integer):
        raise ValueError('integer Nx2 native pixel coordinates required')
    model = camera['model']
    supported = ('none', 'brown_conrady', 'inverse_brown_conrady', 'kannala_brandt4', 'ftheta')
    if model not in tuple('distortion.' + name for name in supported):
        raise ValueError('unsupported deprojection distortion model')
    intr = rs.intrinsics()
    for key in ('width', 'height', 'fx', 'fy', 'ppx', 'ppy', 'coeffs'):
        setattr(intr, key, camera[key])
    intr.model = getattr(rs.distortion, model.removeprefix('distortion.'))
    points = np.full((len(pixels), 3), np.nan, dtype=np.float64)
    for i, (x, y) in enumerate(pixels):
        if 0 <= x < camera['width'] and 0 <= y < camera['height'] and depth[y, x] != 0:
            points[i] = rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], float(depth[y, x]) * scale)
    points[~np.isfinite(points).all(axis=1)] = np.nan
    return points


def color_points(points: np.ndarray, factory: dict) -> np.ndarray:
    """Map native 3D points into color optical coordinates using factory SE(3)."""
    transform = np.asarray(factory['T_color_from_depth'], dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError('factory mapping must be a finite rigid transform')
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Nx3 native points required')
    return points @ transform[:3, :3].T + transform[:3, 3]
