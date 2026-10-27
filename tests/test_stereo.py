import numpy as np
import pytest
from stereoforge.reference import census, matching_cost, aggregate_sgm, select_disparity

def test_census_has_known_24_bit_center_descriptor_and_invalid_border():
    image=np.arange(49,dtype=np.uint32).reshape(7,7)
    desc,valid=census(image)
    assert desc[3,3]==4095  # twelve preceding neighbours are darker
    assert valid[3,3] and not valid[0,0]

def test_cost_invalidates_out_of_bounds_and_recovers_shift():
    rng=np.random.default_rng(7)
    l=rng.integers(0,256,(12,24),dtype=np.uint32)
    r=np.zeros_like(l); r[:,:-3]=l[:,3:]
    a,av=census(l); b,bv=census(r)
    cost=matching_cost(a,b,av,bv,8)
    assert cost[5,10,3]==0
    assert cost[5,0,3]>=10000
    for paths in (4,8):
        disp,conf,valid=select_disparity(aggregate_sgm(cost,paths),subpixel=False)
        assert np.median(disp[3:-3,8:-3])==3
        assert valid[5,10] and conf[5,10]>0

def test_aggregation_has_hand_checked_recurrence():
    c=np.array([[[0,5],[4,0],[0,5]]],np.uint32)
    # Every pixel has vertical singleton paths; horizontal recurrences add P1.
    out=aggregate_sgm(c,4,p1=1,p2=3)
    np.testing.assert_array_equal(out,[[[1,20],[16,2],[1,20]]])

@pytest.mark.gpu
def test_vulkan_stage_parity():
    from stereoforge.gpu import VulkanStereo
    rng=np.random.default_rng(11)
    l=rng.integers(0,256,(12,24),dtype=np.uint32)
    r=np.zeros_like(l); r[:,:-3]=l[:,3:]
    for paths in (4,8):
        gpu=VulkanStereo(8,paths=paths)
        got=gpu.debug_stages(l,r)
        a,av=census(l); b,bv=census(r)
        np.testing.assert_array_equal(got['left_census'],a)
        cost=matching_cost(a,b,av,bv,8)
        np.testing.assert_array_equal(got['cost'],cost)
        expected=aggregate_sgm(cost,paths)
        np.testing.assert_array_equal(got['aggregate'],expected)
        disp,confidence,valid=select_disparity(expected)
        np.testing.assert_allclose(got['disparity'],disp,atol=1e-5,equal_nan=True)
        np.testing.assert_array_equal(got['valid'],valid)
