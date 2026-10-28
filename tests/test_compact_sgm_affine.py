import numpy as np
import pytest
from benchmarks.compact_sgm_affine import estimate_affine,apply_affine,sample_anchors

def grid():
    y,x=np.indices((512,512));right=((x*13+y*7)%180+30).astype(np.uint8);left=np.clip(1.1*right+7,0,255).round().astype(np.uint8);mask=np.ones_like(right,bool);disp=np.zeros_like(right,np.float32)
    return left,right,mask,disp

def test_known_affine_and_spatial_split_deterministic():
    left,right,s,d=grid();a=sample_anchors(left,right,s,s,d,s)
    fit=estimate_affine(a);assert fit['accepted'];assert abs(fit['gain']-1.1)<.01;assert abs(fit['offset']-7)<1
    assert not set(a['train_tiles']) & set(a['validation_tiles']);assert fit==estimate_affine(sample_anchors(left,right,s,s,d,s))

def test_outliers_are_rejected_and_validation_is_independent():
    left,right,s,d=grid();a=sample_anchors(left,right,s,s,d,s);a['train_left'][:100]=230
    fit=estimate_affine(a);assert fit['accepted'];assert fit['inliers']<fit['train_count']
    a['validation_left'][:]=255;assert not estimate_affine(a)['accepted']

def test_unsaturated_only_and_cross_view_support():
    left,right,s,d=grid();d[:]=1.;other=s.copy();other[:,1::2]=False;right[:,::4]=255
    valid=np.zeros_like(s);valid[:,1::2]=True
    a=sample_anchors(left,right,s,other,d,valid)
    assert np.all(a['train_right']>0) and np.all(a['train_right']<255)
    assert np.all(a['train_x']>=1) and np.all(other[a['train_y'],a['train_x']-1])

def test_refuse_too_few_flat_and_invalid_reprojection():
    left,right,s,d=grid();assert not estimate_affine(sample_anchors(left,right,s,s,d,np.zeros_like(s)))['accepted']
    right[:]=80;assert not estimate_affine(sample_anchors(left,right,s,s,d,s))['accepted']
    d[:]=.5
    with pytest.raises(ValueError):sample_anchors(left,right,s,s,d,s)

def test_correction_clipping_mask_and_rounding():
    image=np.array([[0,20,100,250,255]],np.uint8);mapped,safe=apply_affine(image,1.1,7.)
    assert mapped.tolist()==[[7,29,117,255,255]];assert safe.tolist()==[[False,True,True,False,False]]

@pytest.mark.parametrize('eye',[0,1])
def test_masked_cost_matches_independent_scalar_paths(tmp_path,eye):
    import subprocess
    from stereoforge.refinement import census
    from benchmarks.compact_sgm_reference import compact_sgm
    from benchmarks.compact_sgm_local_evidence import candidate,features
    from benchmarks.compact_sgm_affine import run_masked
    runner=tmp_path/'runner';subprocess.run(['c++','-std=c++20','-O2','benchmarks/compact_sgm_affine_runner.cpp','-o',str(runner)],check=True)
    rng=np.random.default_rng(5);images=rng.integers(1,255,(2,8,12),dtype=np.uint8);support=np.ones_like(images,bool);dsc=[census(images[i],support[i]) for i in (0,1)];safe=np.ones_like(images,bool);safe[1,3,4]=False;h,w=8,12;lo=np.zeros((h,w),int);hi=np.full((h,w),6);cost=[]
    for q in range(6):
        a,b=dsc[eye],dsc[1-eye];kwargs=(features(images[eye]),features(images[1-eye]),a.descriptor,b.descriptor,a.support,b.support,lo,hi,eye,q)
        base=candidate(*kwargs,'census').astype(float);base[base==np.iinfo(np.uint32).max]=np.inf;base*=2
        joint=candidate(*kwargs,'joint').astype(float);joint[joint==np.iinfo(np.uint32).max]=np.inf
        for y,x in np.ndindex(h,w):
            ox=x+q if eye else x-q
            if 0<x<w-1 and 0<ox<w-1 and all(safe[eye,y,j] for j in (x-1,x,x+1)) and all(safe[1-eye,y,j] for j in (ox-1,ox,ox+1)):base[y,x]=joint[y,x]
        cost.append(base)
    scores=compact_sgm(np.stack(cost,axis=2),lo,hi,16,64).paths.sum(axis=0);expected=np.where(np.isfinite(scores).any(axis=2),np.argmin(scores,axis=2),-1)
    got=run_masked(runner,tmp_path/'run',dsc[eye],dsc[1-eye],images[eye],images[1-eye],safe[eye],safe[1-eye],lo,hi,eye,6)
    np.testing.assert_array_equal(got['winners'],expected)
    for y,x in np.ndindex(h,w):
        ordered=np.sort(scores[y,x][np.isfinite(scores[y,x])]);inf=np.iinfo(np.uint32).max
        assert got['best'][y,x]==(ordered[0] if len(ordered) else inf)
        assert got['second'][y,x]==(ordered[1] if len(ordered)>1 else inf)

def test_unsupported_lr_anchor_fails_closed():
    left,right,s,d=grid();bad=s.copy();bad[0,0]=False
    with pytest.raises(ValueError,match='unsupported target anchor'):sample_anchors(left,right,s,bad,d,s)

def test_gain_and_offset_bounds_fail_closed():
    left,right,s,d=grid();a=sample_anchors(left,right,s,s,d,s)
    a['train_left']=a['train_right']*1.4;a['validation_left']=a['validation_right']*1.4
    assert not estimate_affine(a)['accepted']
    a['train_left']=a['train_right']+41;a['validation_left']=a['validation_right']+41
    assert not estimate_affine(a)['accepted']

def test_source_mutation_prior_rejected(tmp_path,monkeypatch):
    import sys
    from benchmarks.compact_sgm_affine_experiment import main
    (tmp_path/'result.json').write_text('{}')
    argv=['affine']
    for name in ('runner','build-manifest','artifact-root','previous-root','family-root','native-report','dump-root','output-root'):argv.extend(['--'+name,str(tmp_path)])
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='prior report hash mismatch'):main()

def test_rejected_fit_drives_exact_all_plane_baseline_fallback():
    from benchmarks.compact_sgm_affine import fallback_selection
    left,right,s,d=grid();right[:]=80;fit=estimate_affine(sample_anchors(left,right,s,s,d,s));assert not fit['accepted']
    baseline={'raw_disparity.f32':np.array([[np.nan,5]],np.float32),'raw_validity.u32':np.array([[False,True]]),'validity.u32':np.array([[False,True]]),'metric_validity.u32':np.array([[False,True]]),'depth.f32':np.array([[np.nan,2]],np.float32)}
    got=fallback_selection(fit,baseline)
    assert set(got)==set(baseline)
    for key in baseline:
        np.testing.assert_array_equal(got[key],baseline[key]);assert not np.shares_memory(got[key],baseline[key])
    with pytest.raises(ValueError):fallback_selection({'accepted':True},baseline)
