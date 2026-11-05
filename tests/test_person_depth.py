import numpy as np

from stereoforge.person_depth import colorize, refine_depth


def _scene():
    h, w = 360, 640
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - 320) ** 2 / 90**2 + (yy - 180) ** 2 / 120**2 <= 1.0
    z = np.full((h, w), 2.0, dtype=np.float32)
    z[mask] = 0.6
    return z, mask


def test_holes_inside_mask_are_filled_with_person_depth():
    z, mask = _scene()
    rng = np.random.default_rng(1)
    z[mask & (rng.random(z.shape) < 0.7)] = np.nan
    out = refine_depth(z, mask)
    inside = out[mask]
    assert np.isfinite(inside).mean() > 0.95
    assert np.allclose(inside[np.isfinite(inside)], 0.6, atol=0.02)


def test_person_depth_halo_outside_mask_is_removed_but_background_is_kept():
    z, mask = _scene()
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    ring = (xx - 320) ** 2 / 115**2 + (yy - 180) ** 2 / 145**2 <= 1.0
    halo = ring & ~mask
    z[halo] = 0.62
    out = refine_depth(z, mask)
    assert np.isnan(out[halo & ~(((xx - 320) ** 2 / 102**2 + (yy - 180) ** 2 / 132**2) <= 1.0)]).all()
    far = ~ring
    assert np.array_equal(out[far], z[far])


def test_no_person_leaves_depth_unchanged():
    z, _ = _scene()
    out = refine_depth(z, np.zeros(z.shape, dtype=bool))
    assert np.array_equal(out, z)


def test_colorize_marks_invalid_pixels_dark():
    z = np.full((8, 8), np.nan, dtype=np.float32)
    z[0, 0] = 1.0
    col = colorize(z, 0.4, 2.5)
    assert col.shape == (8, 8, 3)
    assert tuple(col[4, 4]) == (20, 20, 20)
    assert tuple(col[0, 0]) != (20, 20, 20)
