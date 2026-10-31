import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from fast60_anchor_evaluate import evaluate
from fast60_anchor_hypotheses import AnchorConfig


def scene(disparity=5):
    rng = np.random.default_rng(42)
    left = rng.integers(0, 256, (48, 96), dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-disparity] = left[:, disparity:]
    truth = np.full(left.shape, disparity, dtype=np.float32)
    return left, right, truth


def test_correct_anchors_and_fixed_support():
    left,right,truth=scene()
    out=evaluate(left,right,truth, config=AnchorConfig(stride=3,disparities=16))
    assert out['selected_anchor_count'] > 0
    assert out['correct_anchor_count'] > 0
    assert out['correct_fraction_of_selected_known'] > 0.95
    assert out['known_truth_pixels'] == truth.size
    assert out['correct_anchor_fraction_of_fixed_truth_support'] < 1


def test_rescues_missing_coarse_and_risk():
    left,right,truth=scene()
    coarse=np.full_like(truth,np.nan)
    risk=np.zeros_like(left,dtype=bool)
    risk[:, 12:65]=True
    out=evaluate(left,right,truth,coarse=coarse,risk=risk,config=AnchorConfig(stride=3,disparities=16))
    assert out['coarse_comparison']['coarse_correct_on_fixed_support']==0
    assert out['coarse_comparison']['anchors_correct_where_coarse_incorrect_or_missing']==out['correct_anchor_count']
    assert out['risk']['correct_risk_anchors']>0
    assert out['risk']['correct_risk_anchor_coverage']<1


def test_coarse_already_correct_not_counted_as_rescue():
    left,right,truth=scene()
    out=evaluate(left,right,truth,coarse=truth.copy(),config=AnchorConfig(stride=4,disparities=16))
    assert out['coarse_comparison']['anchors_correct_where_coarse_incorrect_or_missing']==0


@pytest.mark.parametrize('mutate', [
    lambda l,r,t: (l,r,t[:12]),
    lambda l,r,t: (l,r,t.astype(str)),
    lambda l,r,t: (l,r,t),
])
def test_rejection_or_valid_case(mutate):
    l,r,t=mutate(*scene())
    if t.dtype.kind in 'US' or t.shape != l.shape:
        with pytest.raises(ValueError):evaluate(l,r,t)
    else:
        with pytest.raises(ValueError):evaluate(l,r,t,risk=np.zeros_like(l,dtype=np.uint8))
