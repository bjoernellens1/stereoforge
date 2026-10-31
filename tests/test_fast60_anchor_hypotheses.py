import importlib.util
from pathlib import Path
import numpy as np
import pytest

P=Path(__file__).resolve().parents[1]/"benchmarks/fast60_anchor_hypotheses.py"
spec=importlib.util.spec_from_file_location("fast60_anchor_hypotheses",P)
mod=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=mod
spec.loader.exec_module(mod)


def pair(disparity=7, seed=5):
    rng=np.random.default_rng(seed)
    left=rng.integers(0,256,(48,96),dtype=np.uint8)
    right=np.zeros_like(left)
    right[:,:-disparity]=left[:,disparity:]
    right[:,-disparity:]=rng.integers(0,256,(48,disparity),dtype=np.uint8)
    return left,right


def test_recovers_known_shift():
    left,right=pair()
    a=mod.match_anchors(left,right,mod.AnchorConfig(disparities=16,stride=2,min_gradient=10))
    assert a.count>25
    assert a.evaluated>0
    assert np.mean(a.disparity==7) > .99
    truth=np.full(left.shape,7,dtype=np.float32)
    result=mod.candidate_recall(a,truth)
    assert result["correct_fraction_on_known"] > .99


def test_rejects_constant_ambiguous_texture():
    image=np.full((32,64),110,dtype=np.uint8)
    a=mod.match_anchors(image,image)
    assert a.count==0


def test_reproducible_and_independent_of_truth():
    left,right=pair(seed=20)
    c=mod.AnchorConfig(disparities=24,stride=3)
    a=mod.match_anchors(left,right,c)
    b=mod.match_anchors(left.copy(),right.copy(),c)
    assert np.array_equal(a.disparity,b.disparity)
    assert np.array_equal(a.left_x,b.left_x)
    assert np.array_equal(a.y,b.y)
    assert a.evaluated==b.evaluated
    assert mod.candidate_recall(a,np.full(left.shape,np.nan))["correct_fraction_on_known"] is None


def test_invalid_inputs_and_configs():
    img=np.zeros((16,32),dtype=np.uint8)
    with pytest.raises(ValueError):mod.match_anchors(img,img[:,:-1])
    with pytest.raises(ValueError):mod.match_anchors(img.astype(np.float32),img)
    with pytest.raises(ValueError):mod.AnchorConfig(disparities=0)
    with pytest.raises(ValueError):mod.AnchorConfig(stride=0)
    with pytest.raises(ValueError):mod.candidate_recall(mod.match_anchors(img,img),img,tolerance=-1)
