import subprocess
import numpy as np
import pytest
from benchmarks.compact_sgm_reference import dense_masked_sgm
from benchmarks.fp2_banded import run_cost, run_image, buckets
from stereoforge.refinement import census

@pytest.fixture(scope='module')
def runner(tmp_path_factory):
    out=tmp_path_factory.mktemp('fp2')/'runner';subprocess.run(['c++','-std=c++20','-O2','benchmarks/fp2_banded_runner.cpp','-o',str(out)],check=True);return out

def test_buckets_and_out_of_range():
    assert buckets(np.array([0,1,16,17,32,33,64,65,128])).tolist()==[0,16,16,32,32,64,64,128,128]
    with pytest.raises(ValueError):buckets(np.array([129]))

@pytest.mark.parametrize('seed',[11,19,27])
def test_packed_all_paths_against_dense_oracle(runner,tmp_path,seed):
    rng=np.random.default_rng(seed);h,w,d=5,8,128;cost=rng.integers(0,25,(h,w,d)).astype(float);cost[rng.random(cost.shape)<.2]=np.inf
    lo=rng.integers(0,128,(h,w));hi=np.array([[rng.integers(v,129) for v in row] for row in lo]);lo[0,0]=hi[0,0]=17;lo[0,1]=127;hi[0,1]=128
    expected=dense_masked_sgm(cost,lo,hi,8,32);result=run_cost(runner,tmp_path,cost,lo,hi)
    np.testing.assert_array_equal(result['paths'],expected.paths);np.testing.assert_array_equal(result['sums'],expected.sums)
    win=np.where(np.isfinite(expected.sums).any(axis=2),np.argmin(expected.sums,axis=2),-1);np.testing.assert_array_equal(result['winner'],win)
    assert result['stats']['active_slots']==int((hi-lo).sum());assert result['stats']['winner_visits']==int((hi-lo).sum())
    assert result['stats']['aggregate_bytes']==4*int(buckets(hi-lo).sum())

@pytest.mark.parametrize('eye',[0,1])
def test_independent_census_image_mode(runner,tmp_path,eye):
    rng=np.random.default_rng(7);images=rng.integers(0,256,(2,8,12),dtype=np.uint8);support=np.ones_like(images,bool);support[1,3,3]=False;a,b=(census(images[i],support[i]) for i in (eye,1-eye));h,w,d=8,12,6;lo=np.zeros((h,w),int);hi=np.full((h,w),d,int);cost=np.full((h,w,d),np.inf)
    for y,x,q in np.ndindex(h,w,d):
        ox=x+q if eye else x-q
        if a.support[y,x] and 0<=ox<w and b.support[y,ox]:cost[y,x,q]=(int(a.descriptor[y,x])^int(b.descriptor[y,ox])).bit_count()
    expected=dense_masked_sgm(cost,lo,hi);got=run_image(runner,tmp_path,a,b,lo,hi,eye,d,dump=True)
    np.testing.assert_array_equal(got['paths'],expected.paths);np.testing.assert_array_equal(got['sums'],expected.sums)

def test_source_guard_no_dense_pixel_disparity_arrays():
    from pathlib import Path
    s=Path('benchmarks/fp2_banded_runner.cpp').read_text()
    assert 'n*d' not in s and 'n * d' not in s
    assert 'for(int q=0;q<d' not in s and 'for (int q=0;q<d' not in s

def test_shift_disjoint_empty_holes_and_all_buckets(runner,tmp_path):
    h,w,d=1,8,128;lo=np.array([[0,0,30,90,70,4,0,0]]);hi=np.array([[0,16,62,126,128,69,128,1]])
    cost=np.zeros((h,w,d),float);cost[0,2,30:62]=np.inf;cost[0,4,100]=np.inf
    expected=dense_masked_sgm(cost,lo,hi);got=run_cost(runner,tmp_path,cost,lo,hi)
    np.testing.assert_array_equal(got['paths'],expected.paths);np.testing.assert_array_equal(got['sums'],expected.sums)
    assert got['stats']['bucket_histogram']==[1,2,1,2,2]
    assert got['stats']['padding_slots']==got['stats']['bucket_slots']-int((hi-lo).sum())


def test_full_band_and_invalid_cost_fails(runner,tmp_path):
    lo=np.zeros((1,3),int);hi=np.full((1,3),128);cost=np.ones((1,3,128),float);expected=dense_masked_sgm(cost,lo,hi)
    got=run_cost(runner,tmp_path,cost,lo,hi);np.testing.assert_array_equal(got['paths'],expected.paths)
    cost[0,0,0]=np.nan
    with pytest.raises(ValueError):run_cost(runner,tmp_path/'bad',cost,lo,hi)
