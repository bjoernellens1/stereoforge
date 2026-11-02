from pathlib import Path
import numpy as np
import pytest

from benchmarks.matching_cost_inspector_v8 import (
    aggregate_sgm_4path_vectorized,
    build_cost_volumes,
    build_ortho_hybrid_u32,
    geometric_qualification_metrics,
    inspect_cost_curves,
    ortho_hybrid_cost_slice,
    select_from_aggregated,
)


def make_synthetic_stereo(h: int = 48, w: int = 96, shift: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    yy, xx = np.indices((h, w), dtype=np.float32)
    base = 40.0 + 25.0 * np.sin(xx * 0.22) + 15.0 * np.cos(yy * 0.18) + rng.normal(0.0, 1.0, (h, w))
    left = np.clip(np.round(base), 0, 255).astype(np.uint32)
    right = np.zeros_like(left)
    right[:, : w - shift] = left[:, shift:]
    return left, right


def test_ortho_hybrid_u32_recovers_exact_shift():
    left, right = make_synthetic_stereo(h=48, w=96, shift=10)
    dl = build_ortho_hybrid_u32(left)
    dr = build_ortho_hybrid_u32(right)
    # Border of 2 pixels must be zero
    assert np.all(dl[:2, :] == 0) and np.all(dl[-2:, :] == 0)
    assert np.all(dl[:, :2] == 0) and np.all(dl[:, -2:] == 0)
    # At true shift=10 in interior, cost must be 0
    c_true = ortho_hybrid_cost_slice(dl[6:-6, 16:-6], dr[6:-6, 6:-16])
    assert np.median(c_true) == 0


def test_cost_volumes_and_4path_sgm_selection():
    left, right = make_synthetic_stereo(h=40, w=88, shift=8)
    cl, cr = build_cost_volumes(left, right, disparities=24, mode="ortho_hybrid_u32")
    agg_l = aggregate_sgm_4path_vectorized(cl, p1=8, p2=32)
    agg_r = aggregate_sgm_4path_vectorized(cr, p1=8, p2=32)
    disp_l, conf_l, val_l = select_from_aggregated(
        agg_l, non_adjacent_runner_up=True, uniqueness_ratio=0.05, min_margin=2, denom_floor=24
    )
    disp_r, conf_r, val_r = select_from_aggregated(
        agg_r, non_adjacent_runner_up=True, uniqueness_ratio=0.05, min_margin=2, denom_floor=24
    )
    interior = val_l[6:-6, 14:-6]
    assert interior.mean() > 0.85
    assert np.median(disp_l[6:-6, 14:-6][interior]) == pytest.approx(8.0, abs=0.25)
    assert np.all(conf_l[val_l] > 0.0)


def test_non_adjacent_runner_up_handles_subpixel_plateau():
    agg = np.full((8, 8, 16), 200, dtype=np.int32)
    # Subpixel peak split across d=6 (cost 20) and d=7 (cost 21), all other disparities >= 120
    agg[:, :, 6] = 20
    agg[:, :, 7] = 21
    _, conf_all, val_all = select_from_aggregated(
        agg, non_adjacent_runner_up=False, uniqueness_ratio=0.10, min_margin=2, denom_floor=24
    )
    _, conf_non_adj, val_non_adj = select_from_aggregated(
        agg, non_adjacent_runner_up=True, uniqueness_ratio=0.10, min_margin=2, denom_floor=24
    )
    assert not np.any(val_all)
    assert np.all(val_non_adj)
    assert np.all(conf_non_adj > 0.7)


def test_inspect_cost_curves_and_geometric_qualification():
    left, right = make_synthetic_stereo(h=64, w=128, shift=12)
    nat_d = np.full((64, 128), 12.0, dtype=np.float32)
    nat_v = np.ones((64, 128), dtype=np.uint32)
    # Introduce a wrong native patch
    nat_d[16:32, 40:72] = 3.0
    cv_d = np.full((64, 128), 12.0, dtype=np.float32)
    rep = inspect_cost_curves(left, right, nat_d, nat_d, nat_v, nat_v, cv_d, cv_d, disparities=32)
    assert rep["cv_consistent_native_failed_breakdown"]["count"] > 0
    lr_ok = np.ones((64, 128), dtype=bool)
    geom = geometric_qualification_metrics(cv_d, lr_ok, left)
    assert geom["coherent_surface_fraction_all_pixels"] > 0.9
    assert geom["speckle_survival_rate_of_lr_accepted"] > 0.95
