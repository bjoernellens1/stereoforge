import json

import cv2
import numpy as np
import pytest

from benchmarks.physical_depth_v14 import reference_sgm as R
from benchmarks.physical_depth_v14.calibrate_sigma import sigma_d


def _pair(shift: float):
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((180, 320)) * 255).astype(np.float32), (0, 0), 1.5)
    right = cv2.warpAffine(img, np.float32([[1, 0, -shift], [0, 1, 0]]), (320, 180), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(img, 0, 255).astype(np.uint8), np.clip(right, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("shift", [12.0, 12.3, 12.5])
def test_recovers_known_subpixel_shift_without_pixel_locking(shift):
    out = R.estimate(*_pair(shift), R.Params(disparities=32))
    sel = out["valid"].copy()
    sel[:, :40] = False
    assert sel.mean() > 0.7
    err = out["disparity"][sel] - shift
    assert abs(np.median(err)) < 0.15
    assert np.sqrt(np.mean(err ** 2)) < 0.25


def test_flat_region_gets_low_confidence_and_textured_region_high():
    left, right = _pair(10.0)
    left[:, 160:] = 128
    right[:, 160:] = 128
    out = R.estimate(left, right, R.Params(disparities=32))
    assert out["uniqueness"][:, 60:140].mean() > 4 * max(out["uniqueness"][:, 190:300].mean(), 1e-3)


def test_sigma_model_is_monotone_in_uniqueness():
    coef = np.array([-1.9, -0.78, -0.01])
    u = np.array([0.05, 0.2, 1.0, 5.0])
    s = sigma_d(u, np.ones_like(u), coef)
    assert np.all(np.diff(s) < 0) and s[-1] >= 0.03
