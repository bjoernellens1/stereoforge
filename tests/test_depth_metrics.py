import numpy as np
import pytest

from stereoforge.depth_metrics import (false_valid_rate, halo_band_fraction, pixel_locking_index, plane_rmse_mm,
                                       range_error_stats, temporal_std_mm, zscore_calibration)

FX = FY = 500.0
CX, CY = 320.0, 180.0


def test_range_error_reports_bias_per_bin():
    ref = np.full((100, 100), 1.2, np.float32)
    est = ref + 0.01
    stats = range_error_stats(est, ref)
    bin1 = next(s for s in stats if s["range_m"] == [1.0, 1.5])
    assert bin1["bias_mm"] == pytest.approx(10.0, abs=1e-3) and bin1["rmse_mm"] == pytest.approx(10.0, abs=1e-3)


def test_plane_rmse_recovers_noise_level_and_ignores_outliers():
    h, w = 360, 640
    rng = np.random.default_rng(0)
    z = np.full((h, w), 1.5, np.float32) + rng.normal(0, 0.003, (h, w)).astype(np.float32)
    z[::17, ::13] = 0.7  # outliers
    r = plane_rmse_mm(z, np.ones((h, w), bool), FX, FY, CX, CY)
    assert 2.0 < r["rmse_mm"] < 4.5 and r["inlier_fraction"] > 0.95


def test_halo_fraction_counts_foreground_depth_outside_the_object():
    z = np.full((300, 300), 2.0, np.float32)
    obj = np.zeros((300, 300), bool)
    obj[100:200, 100:200] = True
    z[obj] = 0.6
    clean = halo_band_fraction(z, obj)
    z2 = z.copy()
    z2[80:100, 100:200] = 0.62  # 20 px of grown foreground above the object
    grown = halo_band_fraction(z2, obj)
    assert clean["halo_fraction"] == 0.0 and grown["halo_fraction"] > 0.02


def test_pixel_locking_index_separates_integer_from_uniform_disparity():
    rng = np.random.default_rng(0)
    uniform = rng.uniform(10, 60, 50000)
    locked = np.round(uniform)
    assert pixel_locking_index(uniform) < 0.1 and pixel_locking_index(locked) > 0.9


def test_temporal_std_and_zscore_and_false_valid():
    rng = np.random.default_rng(1)
    stack = 1.0 + rng.normal(0, 0.005, (40, 20, 20)).astype(np.float32)
    assert temporal_std_mm(stack)["median_std_mm"] == pytest.approx(5.0, abs=0.8)
    err = rng.normal(0, 0.01, 20000)
    z = zscore_calibration(err, np.full_like(err, 0.01))
    assert z["within_1sigma"] == pytest.approx(0.683, abs=0.02) and z["within_2sigma"] == pytest.approx(0.954, abs=0.02)
    ref = np.full((50, 50), 2.0, np.float32)
    est = ref.copy()
    est[:5] = 2.5
    assert false_valid_rate(est, ref)["false_valid_rate"] == pytest.approx(0.1, abs=1e-6)
