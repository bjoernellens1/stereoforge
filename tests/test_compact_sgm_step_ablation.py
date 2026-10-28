import subprocess
import numpy as np
import pytest
from benchmarks.compact_sgm_step_ablation import step_paths,run_compiled
from benchmarks.compact_sgm_reference import compact_sgm

@pytest.fixture(scope='module')
def runner(tmp_path_factory):
    out=tmp_path_factory.mktemp('step')/'runner'
    subprocess.run(['c++','-std=c++20','-O2','benchmarks/compact_sgm_step_runner.cpp','-o',str(out)],check=True)
    return out

def test_independent_scalar_paths_and_disconnected_intervals(runner,tmp_path):
    rng=np.random.default_rng(497);cost=rng.integers(0,25,(5,8,8)).astype(float);cost[rng.random(cost.shape)<.2]=np.inf
    lo=rng.integers(0,5,(5,8));hi=np.minimum(8,lo+rng.integers(0,5,(5,8)))
    image=np.tile(np.array([10,10,10,130,130,130,10,10],np.uint8),(5,1));support=rng.random((5,8))>.1
    expected=step_paths(cost,lo,hi,image,support);paths,sums,winners=run_compiled(runner,tmp_path,cost,lo,hi,image,support)
    np.testing.assert_array_equal(paths,expected);np.testing.assert_array_equal(sums,expected.sum(axis=0))
    expected_w=np.argmin(sums,axis=2).astype(np.int32);expected_w[np.isinf(sums).all(axis=2)]=-1;np.testing.assert_array_equal(winners,expected_w)

def test_steps_are_directional_and_disabled_at_support_or_image_boundary():
    cost=np.full((1,6,3),100.);cost[0,1,0]=0;cost[0,2,2]=0
    lo=np.zeros((1,6),int);hi=np.full((1,6),3);image=np.array([[0,0,100,100,100,100]],np.uint8);support=np.ones((1,6),bool)
    p=step_paths(cost,lo,hi,image,support);assert p[0,0,2,2] == 8
    support[0,0]=False;p=step_paths(cost,lo,hi,image,support);assert p[0,0,2,2] == 32
    # Isolated bright pixel lacks the same-side support and is kept fixed.
    image=np.array([[0,0,100,0,0,0]],np.uint8);support[:]=True;p=step_paths(cost,lo,hi,image,support);assert p[0,0,2,2] == 32

def test_flat_matches_fixed(runner,tmp_path):
    rng=np.random.default_rng(9);cost=rng.integers(0,20,(4,7,6)).astype(float);lo=np.zeros((4,7),int);hi=np.full((4,7),6)
    image=np.zeros((4,7),np.uint8);support=np.ones_like(image,bool)
    paths,_,_=run_compiled(runner,tmp_path,cost,lo,hi,image,support)
    np.testing.assert_array_equal(paths,compact_sgm(cost,lo,hi).paths)

@pytest.mark.parametrize('eye',[0,1])
def test_image_mode_uses_independent_eye_descriptors_and_support(runner,tmp_path,eye):
    from stereoforge.refinement import census
    from benchmarks.compact_sgm_step_ablation import run_eye
    rng=np.random.default_rng(417);images=rng.integers(0,256,(2,8,12),dtype=np.uint8)
    support=np.ones_like(images,bool);support[1,3,4]=False
    a,b=(census(images[i],support[i]) for i in (eye,1-eye));h,w,d=8,12,6
    lo=rng.integers(0,d+1,(h,w));hi=np.array([[rng.integers(v,d+1) for v in row] for row in lo]);cost=np.full((h,w,d),np.inf)
    for y,x,q in np.ndindex(h,w,d):
        ox=x+q if eye else x-q
        if a.support[y,x] and 0<=ox<w and b.support[y,ox]:cost[y,x,q]=(int(a.descriptor[y,x])^int(b.descriptor[y,ox])).bit_count()
    scores=step_paths(cost,lo,hi,images[eye],a.support).sum(axis=0)
    expected=np.where(np.isfinite(scores).any(axis=2),np.argmin(scores,axis=2),-1)
    result=run_eye(runner,tmp_path,a,b,lo,hi,images[eye].astype(np.uint32),eye,d)
    np.testing.assert_array_equal(result['winners'],expected)
    for y,x in np.ndindex(h,w):
        ordered=np.sort(scores[y,x][np.isfinite(scores[y,x])]);inf=np.iinfo(np.uint32).max
        assert result['count'][y,x]==len(ordered)
        assert result['best'][y,x]==(int(ordered[0]) if len(ordered) else inf)
        assert result['second'][y,x]==(int(ordered[1]) if len(ordered)>1 else inf)

@pytest.mark.parametrize('policy',['fixed','gradient'])
def test_matched_controls_against_independent_recurrences(runner,tmp_path,monkeypatch,policy):
    from benchmarks.compact_sgm_gradient_ablation import adaptive_paths
    rng=np.random.default_rng(717);cost=rng.integers(0,25,(4,8,7)).astype(float);cost[rng.random(cost.shape)<.1]=np.inf
    lo=rng.integers(0,4,(4,8));hi=rng.integers(4,8,(4,8));image=rng.integers(0,256,(4,8),dtype=np.uint8);support=np.ones_like(image,bool)
    monkeypatch.setenv('SGM_STEP_POLICY',policy)
    paths,_,_=run_compiled(runner,tmp_path,cost,lo,hi,image,support)
    ref=compact_sgm(cost,lo,hi).paths if policy=='fixed' else adaptive_paths(cost,lo,hi,image)
    np.testing.assert_array_equal(paths,ref)

def test_exact_step_thresholds_and_reversed_direction():
    cost=np.full((1,6,3),100.);cost[0,1,0]=0;cost[0,2,2]=0;lo=np.zeros((1,6),int);hi=np.full((1,6),3);support=np.ones((1,6),bool)
    for delta,side,expected in ((63,0,32),(64,16,8),(64,17,32)):
        image=np.array([[side,0,delta,delta,delta,delta]],np.uint8)
        assert step_paths(cost,lo,hi,image,support)[0,0,2,2]==expected
    mirror_cost=cost[:,::-1,:].copy();mirror_image=np.array([[80,80,80,80,16,16]],np.uint8)
    assert step_paths(mirror_cost,lo,hi,mirror_image,support)[1,0,3,2]==8

@pytest.mark.parametrize('missing',[0,1,2,3])
def test_all_four_step_support_pixels_required(missing):
    cost=np.full((1,6,3),100.);cost[0,1,0]=0;cost[0,2,2]=0;lo=np.zeros((1,6),int);hi=np.full((1,6),3)
    image=np.array([[0,0,100,100,100,100]],np.uint8);support=np.ones((1,6),bool);support[0,missing]=False
    assert step_paths(cost,lo,hi,image,support)[0,0,2,2]==32
