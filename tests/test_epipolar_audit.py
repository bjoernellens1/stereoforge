import cv2
import numpy as np
import pytest

from stereoforge.epipolar_audit import audit_epipolar


def _pair(dy: float, tilt: float = 0.0):
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur((rng.random((480, 960)) * 255).astype(np.float32), (0, 0), 1.5)
    shear = np.float32([[1, 0, -20], [tilt, 1, dy]])  # right = left shifted 20 px in x, dy in y (+ optional x-dependent tilt)
    right = cv2.warpAffine(img, shear, (960, 480), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(img, 0, 255).astype(np.uint8), np.clip(right, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("dy", [0.0, 0.3, -0.4])
def test_vertical_offset_is_recovered_to_a_few_hundredths_of_a_pixel(dy):
    report = audit_epipolar(*_pair(dy))
    assert report.matches > 500
    assert report.median_dy_px == pytest.approx(dy, abs=0.05)
    assert report.robust_rms_dy_px < 0.05
    assert report.median_dx_px == pytest.approx(20.0, abs=0.5)


def test_tilt_across_the_image_is_reported():
    report = audit_epipolar(*_pair(0.0, tilt=0.0008))  # 0.77 px of vertical shear over 960 px
    assert report.tilt_px > 0.3
