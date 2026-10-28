import numpy as np
import pytest
from benchmarks.compact_sgm_local_evidence import features,candidate,enumerate_min,transform,INVALID,MODES

def scalar(a,b,da,db,x,y,q,eye,mode):
    ox=x+q if eye else x-q;h=(int(da[y,x])^int(db[y,ox])).bit_count();center=abs(int(a[y,x])-int(b[y,ox]));g=abs((int(a[y,x+1])-int(a[y,x-1]))-(int(b[y,ox+1])-int(b[y,ox-1])))
    v=sum(abs((int(a[y+i,x])-int(a[y,x]))-(int(b[y+i,ox])-int(b[y,ox]))) for i in range(-2,3))
    return {'census':h,'center':h*256+center,'vertical':h*4096+v,'joint':2*h+min(center,32)//4+min(g,64)//8}[mode]

@pytest.mark.parametrize('eye',[0,1])
@pytest.mark.parametrize('mode',MODES)
def test_costs_match_independent_scalar_and_supported_geometry(eye,mode):
    rng=np.random.default_rng(13);a=rng.integers(10,200,(7,10),dtype=np.uint8);b=rng.integers(10,200,(7,10),dtype=np.uint8);da=rng.integers(0,2**24,a.shape,dtype=np.uint32);db=rng.integers(0,2**24,a.shape,dtype=np.uint32)
    sa=np.zeros(a.shape,bool);sa[2:-2,2:-2]=True;sb=sa.copy();sb[3,4]=False;lo=np.zeros(a.shape,int);hi=np.full(a.shape,3)
    actual=candidate(features(a),features(b),da,db,sa,sb,lo,hi,eye,1,mode)
    for y,x in np.ndindex(a.shape):
        ox=x+1 if eye else x-1
        valid=sa[y,x] and 0<=ox<a.shape[1] and sb[y,ox]
        assert actual[y,x]==(scalar(a,b,da,db,x,y,1,eye,mode) if valid else INVALID)

def test_additive_invariance_and_saturation():
    x=np.arange(81,dtype=np.uint8).reshape(9,9);shift=transform(x,'offset25');np.testing.assert_array_equal(features(x)[1][0],features(shift)[1][0]);np.testing.assert_array_equal(features(x)[2],features(shift)[2]);assert transform(np.array([[240]],np.uint8),'offset25')[0,0]==255
    assert transform(np.array([[100]],np.uint8),'gain08_bias20')[0,0]==100

def test_ties_and_no_support():
    a=np.ones((7,9),np.uint8);d=np.zeros(a.shape,np.uint32);support=np.zeros(a.shape,bool);support[2:-2,2:-2]=True;lo=np.zeros(a.shape,int);hi=np.full(a.shape,3)
    out=enumerate_min(features(a),features(a),d,d,support,support,lo,hi,1,3,'census')
    assert out['winner'][3,3]==0 and out['ties'][3,3]==3
    assert out['winner'][0,0]==-1 and out['ties'][0,0]==0

def test_frozen_mismatch_fails_before_output(tmp_path,monkeypatch):
    import sys
    from benchmarks.compact_sgm_local_evidence import main
    for suffix in ('.npz','.npz.selection.npz','.json'):(tmp_path/('compact-fixed'+suffix)).write_bytes(b'forged')
    monkeypatch.setattr(sys,'argv',['evidence','--artifact-root',str(tmp_path),'--family-root',str(tmp_path),'--native-report',str(tmp_path/'native'),'--dump-root',str(tmp_path),'--output-root',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='frozen artifact mismatch'):main()
    assert not list((tmp_path/'out').iterdir())

def test_scoring_reconciles_ambiguity_and_unique_wrong():
    from benchmarks.compact_sgm_local_evidence import score
    a=np.zeros((7,9),np.uint8);desc=np.zeros(a.shape,np.uint32);support=np.zeros(a.shape,bool);support[2:-2,2:-2]=True;lo=np.zeros(a.shape,int);hi=np.full(a.shape,3)
    result=enumerate_min(features(a),features(a),desc,desc,support,support,lo,hi,1,3,'census');result['mode']='census'
    t={k:np.ones(a.shape,bool) for k in ('valid','visible','remap_support','region_thin')};t.update({k:np.zeros(a.shape,bool) for k in ('out_of_range','off_image','mixed_bilinear')});t['disparity']=np.ones(a.shape);t['surface_id']=np.ones(a.shape,int)
    r=score(features(a),features(a),desc,desc,support,support,lo,hi,1,t,result)
    assert r['all']['eligible']==15
    assert sum(r['all']['local'].values())==15 and sum(r['all']['near'].values())==15
    assert r['all']['local']['ambiguous']>0
    assert sum(v['eligible'] for v in r['surfaces'].values())==15

@pytest.mark.parametrize('eye',[0,1])
@pytest.mark.parametrize('mode',MODES)
def test_enumerated_minimum_against_independent_scalar(eye,mode):
    rng=np.random.default_rng(71);a=rng.integers(0,256,(7,10),np.uint8);b=rng.integers(0,256,(7,10),np.uint8);da=rng.integers(0,2**24,a.shape,dtype=np.uint32);db=rng.integers(0,2**24,a.shape,dtype=np.uint32)
    sa=np.zeros(a.shape,bool);sa[2:-2,2:-2]=True;sb=sa.copy();lo=np.zeros(a.shape,int);hi=np.full(a.shape,3)
    out=enumerate_min(features(a),features(b),da,db,sa,sb,lo,hi,eye,3,mode)
    for y,x in np.ndindex(a.shape):
        vals=[]
        for q in range(3):
            ox=x+q if eye else x-q
            if sa[y,x] and 0<=ox<a.shape[1] and sb[y,ox]:vals.append((scalar(a,b,da,db,x,y,q,eye,mode),q))
        if vals:
            m=min(v for v,q in vals)
            assert out['best'][y,x]==m and out['winner'][y,x]==min(q for v,q in vals if v==m)
            assert out['ties'][y,x]==sum(v==m for v,q in vals) and out['candidate_count'][y,x]==len(vals)
        else:assert out['best'][y,x]==INVALID and out['winner'][y,x]==-1 and out['candidate_count'][y,x]==0
