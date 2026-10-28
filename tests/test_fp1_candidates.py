import numpy as np
import pytest
from benchmarks.fp1_candidates import tile_points,local_minima,point_search,construct_sets,verify_hashes

def test_points_deterministic_and_separated():
    img=np.zeros((8,8),np.uint8);img[:,4:]=200;support=np.ones((8,8),bool)
    a=tile_points(img,support,0,0);assert a==tile_points(img,support,0,0);assert a[0]==(4,4)
    assert len(a)<=4 and all(max(abs(y-v),abs(x-u))>=2 for i,(y,x) in enumerate(a) for v,u in a[:i])
    support[:]=False;assert tile_points(img,support,0,0)==[]

def test_plateaus_separation_and_boundaries():
    costs=np.array([0,0,4,3,4,0,0,5,np.inf])
    assert local_minima(costs,2,4)==[(0,0,True),(5,0,True)]
    assert local_minima(np.full(5,np.inf),2,4)==[]
    assert local_minima(np.array([np.inf,4,0,4,1]),2,4)==[(2,0,False)]

def test_search_eye_support_and_disparity_extremes():
    a=np.array([[0,1,2,3]],np.uint32);b=a.copy();support=np.ones_like(a,bool)
    left=point_search(a,b,support,support,0,2,0,4);right=point_search(a,b,support,support,0,2,1,4)
    assert left['attempts']==4 and left['supported']==3 and right['supported']==2
    support[:]=False;assert point_search(a,b,support,support,0,2,0,4)['peaks']==[]

def test_construct_dedup_support_and_no_coarse_independence():
    shape=(2,3);support=np.ones(shape,bool);other=support.copy();lo=np.zeros(shape,int);hi=np.full(shape,2,int);prior=np.ones(shape,bool)
    tiles={(0,0):[(1,False),(1,True),(3,False)]}
    first=construct_sets(shape,support,other,lo,hi,prior,tiles,1,8,window=1)
    changed=construct_sets(shape,support,other,np.ones(shape,int)*6,np.ones(shape,int)*8,~prior,tiles,1,8,window=1)
    np.testing.assert_array_equal(first['local'],changed['local'])
    assert np.all(first['coarse'][:,2]==1)
    assert np.all(first['union']==(first['coarse']|first['local']))
    assert np.all(first['union']<=first['full'])

def test_hash_verification_fails_closed(tmp_path):
    f=tmp_path/'a';f.write_bytes(b'a');import hashlib
    expected={str(f):hashlib.sha256(b'a').hexdigest()};verify_hashes(expected);f.write_bytes(b'b')
    with pytest.raises(ValueError):verify_hashes(expected)

def test_recall_and_work_partitions_exclude_out_of_range():
    from benchmarks.fp1_candidates import contains_near,summarize
    h,w=2,3;shape=(h,w);support=np.ones(shape,bool);bits=np.zeros((h,w,4),np.uint32);bits[:,:,0]=1<<2
    truth={k:np.ones(shape,bool) for k in ('valid','visible','remap_support','region_thin','region_near','region_edges')};truth.update({k:np.zeros(shape,bool) for k in ('off_image','out_of_range','mixed_bilinear')});truth['out_of_range'][0,0]=True;truth['disparity']=np.full(shape,2.);truth['surface_id']=np.array([[1,1,1],[2,2,2]])
    old=np.zeros(shape,bool);out=summarize(bits,truth,support,old)
    assert out['thin']['eligible']==5 and out['thin']['represented']==5 and out['thin']['out_of_range_visible']==1
    assert out['thin']['recovered_vs_old']==5 and sum(v['eligible'] for v in out['thin_surfaces'].values())==5
    assert out['provisional_screen']['recover_half_old_missing'] and out['provisional_screen']['nonempty_and_at_most_32_for_90_percent']
    assert contains_near(bits,truth['disparity']).all()

def test_disparity_bit_boundary_overflow_and_no_coarse_on_mutated_bounds():
    from benchmarks.fp1_candidates import count_bits,contains_near
    h,w=1,130;s=np.ones((h,w),bool);lo=np.zeros((h,w),int);hi=np.full((h,w),128,int);prior=np.ones((h,w),bool)
    tiles={(0,0):[(0,False),(127,True)]}
    got=construct_sets((h,w),s,s,lo,hi,prior,tiles,1,128,window=0)
    assert int(got['local'][0,0,0])==1 and int(got['local'][0,0,3])==(1<<31)
    assert int(count_bits(got['local'])[0,0])==2
    truth=np.full((h,w),127.);assert contains_near(got['local'],truth)[0,0]
    with pytest.raises(ValueError):construct_sets((h,w),s,s,lo,hi,prior,{(0,0):[(0,False)]*9},1)


def test_empty_candidates_do_not_pass_budget_screen():
    from benchmarks.fp1_candidates import summarize
    shape=(2,2);s=np.ones(shape,bool);truth={k:s.copy() for k in ('valid','visible','remap_support','region_thin','region_near','region_edges')};truth.update({k:~s for k in ('off_image','out_of_range','mixed_bilinear')});truth['disparity']=np.ones(shape);truth['surface_id']=np.ones(shape,int)
    out=summarize(np.zeros((2,2,4),np.uint32),truth,s,np.zeros(shape,bool))
    assert out['work_distribution']['nonempty']==0 and not out['provisional_screen']['nonempty_and_at_most_32_for_90_percent']

def test_equal_disjoint_minima_are_not_plateau_ties():
    assert local_minima(np.array([0,3,3,3,0]),2,4)==[(0,0,False),(4,0,False)]
