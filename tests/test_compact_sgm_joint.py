import subprocess
import numpy as np
import pytest
from benchmarks.compact_sgm_reference import compact_sgm
from benchmarks.compact_sgm_joint import run_cost,run_eye
from benchmarks.compact_sgm_local_evidence import candidate,features
from stereoforge.refinement import census

@pytest.fixture(scope='module')
def runner(tmp_path_factory):
    out=tmp_path_factory.mktemp('joint')/'runner'
    subprocess.run(['c++','-std=c++20','-O2','benchmarks/compact_sgm_joint_runner.cpp','-o',str(out)],check=True)
    return out

def test_scaled_recurrence_and_masks(runner,tmp_path):
    rng=np.random.default_rng(18);cost=rng.integers(0,25,(4,7,8)).astype(float);cost[rng.random(cost.shape)<.2]=np.inf
    lo=rng.integers(0,4,(4,7));hi=rng.integers(4,9,(4,7));expected=compact_sgm(cost,lo,hi,8,32).paths
    actual=run_cost(runner,tmp_path,2*cost,lo,hi)
    np.testing.assert_array_equal(actual,expected*2)

@pytest.mark.parametrize('eye',[0,1])
@pytest.mark.parametrize('mode',['baseline','joint'])
def test_independent_image_costs_and_paths(runner,tmp_path,eye,mode):
    rng=np.random.default_rng(459);images=rng.integers(0,256,(2,8,12),dtype=np.uint8);support=np.ones_like(images,bool);support[1,3,3]=False
    desc=[census(images[i],support[i]) for i in (0,1)];h,w,d=8,12,6;lo=rng.integers(0,3,(h,w));hi=rng.integers(3,d+1,(h,w));cost=[]
    for q in range(d):
        c=candidate(features(images[eye]),features(images[1-eye]),desc[eye].descriptor,desc[1-eye].descriptor,desc[eye].support,desc[1-eye].support,lo,hi,eye,q,'census' if mode=='baseline' else 'joint').astype(float)
        c[c==np.iinfo(np.uint32).max]=np.inf
        if mode=='baseline':c*=2
        cost.append(c)
    ref=compact_sgm(np.stack(cost,axis=2),lo,hi,16,64);got=run_eye(runner,tmp_path,desc[eye],desc[1-eye],images[eye],images[1-eye],lo,hi,eye,d,mode)
    expected=ref.paths.sum(axis=0);winner=np.where(np.isfinite(expected).any(axis=2),np.argmin(expected,axis=2),-1)
    np.testing.assert_array_equal(got['winners'],winner)
    for y,x in np.ndindex(h,w):
        ordered=np.sort(expected[y,x][np.isfinite(expected[y,x])]);inf=np.iinfo(np.uint32).max
        assert got['count'][y,x]==len(ordered)
        assert got['best'][y,x]==(ordered[0] if len(ordered) else inf)
        assert got['second'][y,x]==(ordered[1] if len(ordered)>1 else inf)

def test_exploratory_gates_and_empty_support():
    from benchmarks.compact_sgm_joint import evaluate_gates
    b={r:{'eligible':100,'correct_metric_count':50,'lr_false_accepted':10} for r in ('all','near','edges','thin')}
    c={r:dict(v) for r,v in b.items()};c['thin']['correct_metric_count']=55
    surface={'6':{'eligible':10,'false_accepted':4}}
    assert evaluate_gates(c,b,surface,surface)['pass']
    c['thin']['lr_false_accepted']=11
    assert not evaluate_gates(c,b,surface,surface)['pass']
    c['thin']['lr_false_accepted']=10;c['near']['eligible']=0
    assert not evaluate_gates(c,b,surface,surface)['pass']
    assert not evaluate_gates(c,b,{},surface)['pass']

def test_invalid_policy_rejected(runner,tmp_path):
    rng=np.random.default_rng(1);images=rng.integers(0,256,(2,8,12),dtype=np.uint8);support=np.ones_like(images,bool);desc=[census(images[i],support[i]) for i in (0,1)];lo=np.zeros((8,12),int);hi=np.full((8,12),6)
    with pytest.raises(ValueError):run_eye(runner,tmp_path,desc[0],desc[1],images[0],images[1],lo,hi,0,6,'invented')


def test_gate_exact_percentage_boundaries():
    from benchmarks.compact_sgm_joint import evaluate_gates
    b={r:{'eligible':100,'correct_metric_count':50,'lr_false_accepted':10} for r in ('all','near','edges','thin')}
    c={r:dict(v) for r,v in b.items()};c['thin']['correct_metric_count']=55;c['all']['correct_metric_count']=49
    surface={'6':{'eligible':1,'false_accepted':1}}
    assert evaluate_gates(c,b,surface,surface)['pass']
    c['thin']['correct_metric_count']=54
    assert not evaluate_gates(c,b,surface,surface)['pass']
    c['thin']['correct_metric_count']=55;c['all']['correct_metric_count']=48
    assert not evaluate_gates(c,b,surface,surface)['pass']
    c['all']['correct_metric_count']=None
    assert not evaluate_gates(c,b,surface,surface)['pass']
