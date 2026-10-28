"""Offline diagnostic hypotheses only, never calibration/support qualification.

An interior ray fit cannot establish SDK border/interpolation support or physical
calibration. No outlier removal, admission thresholds, or automatic winner exist.
"""
from functools import wraps

import cv2
import numpy as np

RIG_ROTATION_REPRESENTATION_BUDGET = 1e-6

UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)


def _numeric(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError, cv2.error) as exc:
            raise ValueError('geometry arithmetic failed') from exc
    return checked


def _array(value, shape=None):
    a = np.asarray(value, dtype=np.float64)
    # Leave headroom for products/norms; reject extreme finite input before BLAS/CV.
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError('invalid shape or nonfinite geometry')
    if a.size and np.max(np.abs(a)) > np.sqrt(np.finfo(np.float64).max) / 16:
        raise ValueError('geometry exceeds safe arithmetic range')
    return a


def _camera(value):
    K = _array(value, (3, 3))
    if (K[0, 0] <= 0 or K[1, 1] <= 0 or K[0, 1] != 0 or K[1, 0] != 0
            or not np.array_equal(K[2], [0, 0, 1])):
        raise ValueError('expected positive-focal, zero-skew pinhole K')
    return K


def _rotation(value):
    Q = _array(value, (3, 3))
    if not np.allclose(Q.T @ Q, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(
            np.linalg.det(Q), 1., atol=1e-10, rtol=0):
        raise ValueError('expected proper SO(3) rotation')
    return Q


def _rigid(value):
    T = _array(value, (4, 4))
    R = T[:3, :3]
    orthogonality = float(np.max(np.abs(R.T @ R - np.eye(3))))
    determinant = float(np.linalg.det(R))
    budget = RIG_ROTATION_REPRESENTATION_BUDGET
    if orthogonality > budget or abs(determinant - 1.) > budget:
        raise ValueError('rig rotation exceeds proper SO(3) representation budget')
    U, _, Vt = np.linalg.svd(R)
    projected = _rotation(U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt)
    stats = dict(orthogonality_max=orthogonality, input_determinant=determinant,
                 projection_delta_max=float(np.max(np.abs(projected - R))),
                 representation_budget=budget)
    if not np.array_equal(T[3], [0, 0, 0, 1]) or np.linalg.norm(T[:3, 3]) <= 0:
        raise ValueError('expected rigid transform with finite nonzero baseline')
    normalized = T.copy()
    normalized[:3, :3] = projected
    return normalized, stats


def _bearings(xy):
    rays = np.column_stack((xy, np.ones(len(xy))))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return _array(rays)


def _summary(residuals):
    return dict(count=int(len(residuals)), median=float(np.median(residuals)),
                p95=float(np.percentile(residuals, 95)), max=float(np.max(residuals)))


@_numeric
def fit_ray_rotation(raw_xy, rectified_xy, raw_K, raw_disto, rectified_K, train_mask):
    """Fit training bearings by Wahba; report every held-out pixel residual.

    Distortion follows OpenCV RAD_TAN order (0/4/5/8/12 coefficients).
    undistortPointsIter uses 100 iterations and epsilon 1e-14. All arrays are
    float64; a boolean mask selects training data before the SVD is formed.
    """
    raw, rect = _array(raw_xy), _array(rectified_xy)
    if raw.ndim != 2 or raw.shape[1:] != (2,) or len(raw) < 6 or rect.shape != raw.shape:
        raise ValueError('expected at least six paired Nx2 points')
    mask = np.asarray(train_mask)
    if mask.dtype != np.bool_ or mask.shape != (len(raw),) or min(mask.sum(), (~mask).sum()) < 3:
        raise ValueError('boolean mask must select at least three train and validation pairs')
    K, Kr, d = _camera(raw_K), _camera(rectified_K), _array(raw_disto)
    if d.ndim != 1 or len(d) not in (0, 4, 5, 8, 12):
        raise ValueError('expected OpenCV RAD_TAN coefficients of length 0/4/5/8/12')
    undistorted = cv2.undistortPointsIter(raw[:, None, :], K, d if len(d) else None,
                                        None, None, UNDISTORT_CRITERIA).reshape(-1, 2)
    src = _bearings(_array(undistorted))
    # Explicit inverse-K homogeneous projection; pinhole structure avoids inversion.
    dst = _bearings((rect - Kr[:2, 2]) / [Kr[0, 0], Kr[1, 1]])
    for rays in (src, dst):
        for subset in (mask, ~mask):
            if np.linalg.matrix_rank(rays[subset]) < 3:
                raise ValueError('degenerate bearing directions')
    U, _, Vt = np.linalg.svd(dst[mask].T @ src[mask])
    Q = _rotation(U @ np.diag([1., 1., np.linalg.det(U @ Vt)]) @ Vt)
    rotated = src @ Q.T
    if not np.isfinite(rotated).all() or np.any(rotated[:, 2] <= 0):
        raise ValueError('fit produces nonpositive depth or nonfinite bearings')
    projected = rotated @ Kr.T
    pixels = projected[:, :2] / projected[:, 2:3]
    residuals = _array(np.linalg.norm(pixels - rect, axis=1))
    return dict(status='diagnostic_unqualified', Q=Q, residuals_px=residuals,
                train=_summary(residuals[mask]), validation=_summary(residuals[~mask]),
                undistort_criteria=UNDISTORT_CRITERIA)


@_numeric
def compare_rig_hypotheses(left_Q, right_Q, sdk_transform, rectified_transform):
    """Report both SDK directions against R_rect=Q_L R Q_R.T, t_rect=Q_L t.

    Translation errors retain the input transform's units; alignment is a cosine.
    Rigid input rotations may differ from SO(3) by at most 1e-6 in max-entry
    orthogonality and absolute determinant error (float32 metadata budget).
    Closest-SO(3) polar projections are reported and used for rotation/inverse
    arithmetic; translations and source metadata remain unchanged. Supplied Q
    stays strict at 1e-10. This is numerical normalization, not physical evidence.
    The full supplied rectified rotation is used. Never picks a winner.
    """
    Ql, Qr = _rotation(left_Q), _rotation(right_Q)
    sdk, sdk_stats = _rigid(sdk_transform)
    rect, rect_stats = _rigid(rectified_transform)
    R, t, reference = sdk[:3, :3], sdk[:3, 3], rect[:3, 3]
    candidates = {'direct_right_pose': (R, t), 'inverse_mapping': (R.T, -R.T @ t)}
    result = dict(status='diagnostic_unqualified', sdk_rotation_normalization=sdk_stats,
                  rectified_rotation_normalization=rect_stats)
    for name, (candidate_R, candidate_t) in candidates.items():
        expected_R, expected_t = Ql @ candidate_R @ Qr.T, Ql @ candidate_t
        angle = np.arccos(np.clip((np.trace(rect[:3, :3].T @ expected_R) - 1.) / 2., -1., 1.))
        alignment = np.dot(expected_t / np.linalg.norm(expected_t),
                           reference / np.linalg.norm(reference))
        result[name] = dict(orientation_error_rad=float(angle),
                            translation_error=float(np.linalg.norm(expected_t - reference)),
                            translation_alignment=float(np.clip(alignment, -1., 1.)))
    return result
