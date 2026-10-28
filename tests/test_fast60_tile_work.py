import numpy as np
import pytest
from benchmarks.fast60_tile_work import estimate, selection


def test_nondivisible_tiles_and_halo_overlap_count_repeated_work():
    mask = np.zeros((5, 5), bool)
    mask[0, 0] = mask[4, 4] = True
    row = estimate(mask, tile=4, halo=1)
    assert row['total_risk_pixels'] == 2
    assert row['selected_risk_pixels'] == 2 and 'selected_pixels' not in row
    assert row['dispatch_tiles'] == 2
    assert row['expanded_pixel_visits'] == 29  # 25 + 4, overlapping area counted twice
    assert row['full_d128_candidates'] == 29 * 128
    assert row['four_path_state_updates'] == 29 * 128 * 4
    assert estimate(mask, tile=4, halo=0)['expanded_pixel_visits'] == 17


def test_empty_and_full_edge_tile():
    assert estimate(np.zeros((3, 7), bool), 4, 2)['expanded_pixel_visits'] == 0
    assert estimate(np.ones((3, 7), bool), 4, 0)['expanded_pixel_visits'] == 21
    with pytest.raises(ValueError): estimate(np.zeros((0, 2), bool), 4, 0)


def test_observed_selection_uses_no_truth():
    prior = np.array([[1, 0, 1, 1]], bool)
    raw = np.array([[1, 1, 0, 1]], bool)
    provisional = np.array([[1, 1, 1, 0]], bool)
    assert selection(prior, raw, provisional).tolist() == [[False, True, True, True]]
    with pytest.raises(ValueError): selection(prior, raw[:, :2], provisional)


def test_observed_census_gating_removes_unsupported_risks():
    prior = np.array([[0, 0, 1, 1]], bool)
    raw = np.array([[0, 0, 1, 1]], bool)
    lr = np.array([[0, 0, 1, 1]], bool)
    support = np.array([[0, 1, 1, 1]], bool)
    assert selection(prior, raw, lr, support).tolist() == [[False, True, False, False]]
    with pytest.raises(ValueError): selection(prior, raw, lr, support[:, :2])


def test_risk_concentration_thresholds_and_partial_tile_area():
    mask = np.zeros((3, 7), bool)
    mask[0, :4] = True       # 4/12 exactly one third
    mask[0, 4] = True        # 1/9 in partial tile
    row = estimate(mask, 4, 0, minimum_fraction=.25)
    assert row['total_risk_pixels'] == 5
    assert row['selected_risk_pixels'] == 4 and row['unselected_risk_pixels'] == 1
    assert row['risk_coverage'] == pytest.approx(.8)
    assert row['dispatch_tiles'] == 1 and row['expanded_pixel_visits'] == 12
    assert estimate(mask, 4, 0, minimum_fraction=1/3)['dispatch_tiles'] == 1
    assert estimate(mask, 4, 0, minimum_fraction=.5)['selected_risk_pixels'] == 0
    assert estimate(mask, 4, 0, minimum_fraction=.1)['selected_risk_pixels'] == 5
    assert estimate(np.zeros((3, 7), bool), 4, 0, minimum_fraction=.1)['risk_coverage'] is None
