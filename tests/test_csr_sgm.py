"""Exact sparse-vs-dense recurrence tests and thin-object candidate rescue fixture."""
import numpy as np
import pytest
from benchmarks.compact_sgm_reference import dense_masked_sgm
from benchmarks.csr_sgm import (
    candidate_plan, pack_csr, validate_csr, csr_sgm, normalized_work, CsrVolume
)


@pytest.mark.parametrize("seed", range(12))
def test_sparse_csr_matches_independent_dense_sgm(seed):
    rng=np.random.default_rng(seed)
    h,w,d=4,7,11
    cost=rng.integers(0,100,size=(h,w,d)).astype(float)
    selected=rng.random((h,w,d)) < (0.15 + seed/50)
    selected[0,0,:]=False  # Force empty predecessor reset path.
    selected[1,1,:]=True   # Also test dense local candidates.
    packed=pack_csr(cost,selected)
    result=csr_sgm(packed)
    dense=dense_masked_sgm(np.where(selected,cost,np.inf),
                            np.zeros((h,w),dtype=int),
                            np.full((h,w),d,dtype=int))
    for y in range(h):
        for x in range(w):
            p=y*w+x
            for k in range(int(packed.offsets[p]),int(packed.offsets[p+1])):
                disp=int(packed.hypotheses[k])
                np.testing.assert_array_equal(result.paths[:,k],dense.paths[:,y,x,disp])
                assert result.sums[k] == dense.sums[y,x,disp]
            assert result.valid[y,x]==bool(selected[y,x].any())
    assert result.candidate_state_updates==4*int(selected.sum())
    assert result.dense_state_slots==4*h*w*d
    assert 0 <= normalized_work(result) <= 1


def test_independent_high_resolution_anchor_preserves_thin_candidate():
    h,w,d=9,15,16
    coarse=np.full((h,w),3.,dtype=float)
    valid=np.ones((h,w),dtype=bool)
    anchor=np.full((h,w),np.nan,dtype=float)
    anchors_valid=np.zeros((h,w),dtype=bool)
    anchor[4,7]=8.0
    anchors_valid[4,7]=True
    plan=candidate_plan(coarse,valid,anchor,anchors_valid,d,coarse_radius=1,anchor_radius=1)
    assert plan.mask[4,7,8]
    assert plan.mask[4,7,3]
    assert plan.disagreement_pixels==1
    assert plan.candidate_slots < plan.dense_slots/3
    # Cost cube is independent of which hypotheses are selected; all other
    # pixels have target disparity 3, and a one-pixel thin foreground has d=8.
    desired=np.full((h,w),3,dtype=int)
    desired[4,7]=8
    costs=np.full((h,w,d),100.,dtype=float)
    for y in range(h):
        for x in range(w):
            costs[y,x,desired[y,x]]=0
    out=csr_sgm(pack_csr(costs,plan.mask))
    assert out.disparity[4,7]==8
    without_anchor=candidate_plan(coarse,valid,anchor,np.zeros_like(anchors_valid),d,1,1)
    assert not without_anchor.mask[4,7,8]
    assert csr_sgm(pack_csr(costs,without_anchor.mask)).disparity[4,7]!=8


def test_missing_both_priors_uses_full_search_and_is_not_claimed_as_savings():
    z=np.full((3,4),np.nan)
    valid=np.zeros((3,4),bool)
    plan=candidate_plan(z,valid,z,valid,8)
    assert plan.fallback_pixels==12
    assert plan.candidate_slots==plan.dense_slots
    assert np.all(plan.mask)


def test_rejects_invalid_prior_and_unsorted_or_malformed_csr():
    coarse=np.zeros((3,3),float); valid=np.ones((3,3),bool)
    anchor=coarse.copy(); no=np.zeros_like(valid)
    coarse[1,1]=np.nan
    with pytest.raises(ValueError):
        candidate_plan(coarse,valid,anchor,no,8)
    with pytest.raises(ValueError):
        candidate_plan(np.ones((3,3)),valid.astype(np.uint8),anchor,no,8)
    cost=np.ones((3,3,8),float)
    good=pack_csr(cost,np.ones_like(cost,bool))
    validate_csr(good)
    duplicate=good.hypotheses.copy();duplicate[1]=duplicate[0]
    with pytest.raises(ValueError):
        validate_csr(CsrVolume(good.width,good.height,good.disparities,good.offsets,duplicate,good.costs))
    with pytest.raises(ValueError):
        pack_csr(np.full((3,3,8),np.inf),np.ones((3,3,8),bool))


def test_empty_candidate_state_never_fabricates_disparity():
    x=pack_csr(np.ones((4,5,7)),np.zeros((4,5,7),bool))
    out=csr_sgm(x)
    assert not out.valid.any()
    assert np.all(out.disparity==-1)
    assert out.candidate_state_updates==0



def test_symmetric_left_right_consistency_rejects_occluded_and_wrong_matches():
    from benchmarks.csr_sgm import CsrResult, csr_left_right_consistency
    def result(d,v):
        d=np.asarray(d,dtype=np.int32)
        v=np.asarray(v,dtype=bool)
        return CsrResult(np.zeros((4,0),np.uint32),np.zeros(0,np.uint64),
                         d,v,0,0)
    left=result([[0,1,1,1,1,1,1]],[[1,1,1,1,1,1,1]])
    right=result([[0,1,1,1,1,1,1]],[[1,1,1,1,1,1,1]])
    masks=csr_left_right_consistency(left,right,0)
    assert masks.left_valid[0].tolist()==[True,False,False,False,False,False,False]
    # All left d=1 matches with a right d=1 except the first out-of-image pair.
    right=result([[0,1,1,1,1,1,1]],[[1,1,1,1,1,1,1]])
    left=result([[1,1,1,1,1,1,1]],[[1,1,1,1,1,1,1]])
    masks=csr_left_right_consistency(left,right,0)
    assert masks.left_valid[0].tolist()==[False,False,True,True,True,True,True]
    assert masks.right_valid[0].tolist()==[False,True,True,True,True,True,False]
    # Introduce invalid and wrong correspondence in one right-image location.
    right.valid[0,3]=False
    masks=csr_left_right_consistency(left,right,0)
    assert not masks.left_valid[0,4]
    assert not masks.right_valid[0,3]
    with pytest.raises(ValueError):
        csr_left_right_consistency(left,right,-1)
    with pytest.raises(ValueError):
        csr_left_right_consistency(left,result([[0]],[[1]]))
