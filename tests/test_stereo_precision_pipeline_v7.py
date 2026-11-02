import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from stereo_precision_pipeline_v7 import (
    census_24bit_legacy,
    census_32bit_multiscale,
    census_winner_audit,
    lr_consistency_interpolated,
    local_disparity_support_filter,
    speckle_filter_depth,
)


def test_32bit_eps_census_suppresses_shot_noise():
    rng = np.random.default_rng(42)
    # Almost uniform low-texture patch with +/-1 LSB sensor shot noise
    base = np.full((32, 64), 120, dtype=np.uint32)
    noisy_left = (base + rng.integers(-1, 2, size=base.shape)).astype(np.uint32)
    noisy_right = (base + rng.integers(-1, 2, size=base.shape)).astype(np.uint32)

    legacy_l = census_24bit_legacy(noisy_left)
    legacy_r = census_24bit_legacy(noisy_right)
    multi_l = census_32bit_multiscale(noisy_left, epsilon=1)
    multi_r = census_32bit_multiscale(noisy_right, epsilon=1)

    legacy_flip_rate = np.bitwise_count(legacy_l[4:-4, 4:-4] ^ legacy_r[4:-4, 4:-4]).mean() / 24.0
    multi_flip_rate = np.bitwise_count(multi_l[4:-4, 4:-4] ^ multi_r[4:-4, 4:-4]).mean() / 32.0
    # Deadband epsilon=1 reduces per-bit spurious flips in +/-1 LSB shot noise by >50%
    assert multi_flip_rate < legacy_flip_rate * 0.45


def test_32bit_census_known_shift_and_box3x3():
    rng = np.random.default_rng(7)
    left = rng.integers(20, 220, size=(48, 96), dtype=np.uint32)
    shift = 11
    right = np.zeros_like(left)
    right[:, :-shift] = left[:, shift:]

    dl = census_32bit_multiscale(left, epsilon=1)
    dr = census_32bit_multiscale(right, epsilon=1)
    ys = np.array([16, 20, 28], dtype=np.int32)
    xs = np.array([40, 52, 64], dtype=np.int32)
    disp, _, _, valid = census_winner_audit(dl, dr, ys, xs, 32, "left", max_bits=32, box_radius=1)
    assert np.all(valid)
    assert np.all(disp == shift)


def test_interpolated_lr_consistency_on_slanted_plane():
    sy, sx = np.indices((32, 120), dtype=np.float32)
    slope = 0.12
    d_left = 10.25 + slope * (sx - 60.0)
    x_from_xr = (sx + (10.25 - slope * 60.0)) / (1.0 - slope)
    d_right = 10.25 + slope * (x_from_xr - 60.0)
    valid = np.ones_like(d_left, dtype=np.uint32)

    _, res_round = lr_consistency_interpolated(d_left, d_right, valid, valid, tolerance=0.05, interpolate=False)
    acc_interp, res_interp = lr_consistency_interpolated(d_left, d_right, valid, valid, tolerance=0.05, interpolate=True)

    # Interior pixels (where x - d_left is inside [1, w-2]) have near-zero residual with linear interpolation
    interior = slice(4, 28), slice(25, 110)
    assert np.nanmax(res_interp[interior]) < 1e-4
    assert np.nanpercentile(res_round[interior], 90) > 0.03
    assert acc_interp[interior].mean() == 1.0


def test_disparity_domain_speckle_preserves_far_slanted_surface_and_removes_speckles():
    fx_b = 63.041385  # ZED 2 HD fx * baseline (px * m)
    h, w = 40, 60
    yy, xx = np.indices((h, w), dtype=np.float32)
    # Continuous slanted wall from Z=2.5m to Z=4.8m (0.6 px disparity step per column)
    disp_wall = 25.0 - 0.22 * xx
    z_wall = fx_b / disp_wall
    depth = np.full((h, w), np.nan, dtype=np.float32)
    depth[8:32, 5:55] = z_wall[8:32, 5:55]
    # Add isolated 6-pixel speckle at Z=1.2m
    depth[2:4, 10:13] = 1.2

    # Metric 0.05m filter wrongly fragments the far wall because dZ/dx > 0.05m at Z > 3.4m
    metric_filtered, metric_stats = speckle_filter_depth(
        depth, min_component_pixels=50, max_neighbor_delta_m=0.05, eight_connected=False
    )
    # Disparity-domain 1.25 px filter keeps the entire wall intact while removing the 6-pixel speckle
    disp_filtered, disp_stats = speckle_filter_depth(
        depth,
        min_component_pixels=50,
        fx_baseline_px_m=fx_b,
        max_neighbor_disparity_px=1.25,
        eight_connected=True,
    )
    wall_pixels = 24 * 50
    assert disp_stats["output_valid_pixels"] == wall_pixels
    assert disp_stats["invalidated_pixels"] == 6
    assert np.all(np.isnan(disp_filtered[2:4, 10:13]))
    assert metric_stats["output_valid_pixels"] < wall_pixels


def test_local_disparity_support_removes_single_pixel_spike():
    fx_b = 63.04
    depth = np.full((16, 16), 2.0, dtype=np.float32)
    depth[8, 8] = 0.7  # Isolated single-pixel spike
    out = local_disparity_support_filter(depth, fx_baseline=fx_b, max_disp_step_px=1.25, min_neighbors=2)
    assert np.isnan(out[8, 8])
    assert np.isfinite(out[4, 4]) and abs(float(out[4, 4]) - 2.0) < 1e-4
