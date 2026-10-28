import numpy as np
import pytest
from benchmarks.compact_sgm_population import local_stats,summarize,INF

def brute(a,b,sa,sb,lo,hi,eye):
    h,w=a.shape;out=[]
    for y in range(h):
        row=[]
        for x in range(w):
            c=[]
            for d in range(int(lo[y,x]),int(hi[y,x])):
                ox=x+d if eye else x-d
                if sa[y,x] and 0<=ox<w and sb[y,ox]:c.append(((int(a[y,x])^int(b[y,ox])).bit_count(),d))
            row.append(c)
        out.append(row)
    return out

@pytest.mark.parametrize('eye',[0,1])
def test_local_stats_against_independent_enumeration(eye):
    rng=np.random.default_rng(713);a=rng.integers(0,2**24,(4,9),dtype=np.uint32);b=rng.integers(0,2**24,(4,9),dtype=np.uint32);sa=rng.random(a.shape)>.2;sb=rng.random(a.shape)>.2
    lo=rng.integers(0,4,a.shape);hi=rng.integers(4,7,a.shape);raw=np.full(a.shape,np.nan,np.float32);valid=np.zeros(a.shape,bool)
    got=local_stats(a,b,sa,sb,lo,hi,eye,raw,valid,6);ref=brute(a,b,sa,sb,lo,hi,eye)
    for y,x in np.ndindex(a.shape):
        costs=ref[y][x]
        assert got['local_best'][y,x]==(min(c for c,d in costs) if costs else INF)
        assert got['local_winner'][y,x]==(min(d for c,d in costs if c==min(z for z,_ in costs)) if costs else -1)
        assert got['local_ties'][y,x]==(sum(c==min(z for z,_ in costs) for c,d in costs) if costs else 0)

def test_ties_unsupported_and_raw_invalid_reconciled():
    a=np.zeros((1,6),np.uint32);b=a.copy();support=np.ones_like(a,bool);other=support.copy();other[0,2]=False
    lo=np.zeros_like(a,int);hi=np.full_like(a,3,int);raw=np.array([[np.nan,np.nan,1,1,1,np.nan]],np.float32);valid=np.isfinite(raw);lr=valid.copy();metric=lr.copy()
    stats=local_stats(a,b,support,other,lo,hi,1,raw,valid,3)
    truth={k:np.ones_like(a,bool) for k in ['valid','visible','remap_support','region_thin']};truth.update({k:np.zeros_like(a,bool) for k in ['out_of_range','off_image','mixed_bilinear']});truth['disparity']=np.ones_like(a,float);truth['surface_id']=np.array([[1,1,1,2,2,2]])
    out=summarize(a,b,support,other,lo,hi,1,raw,valid,lr,metric,truth,stats)
    assert out['all']['eligible']==6
    assert sum(x['eligible'] for x in out['surfaces'].values())==6
    assert sum(out['all']['cross'].values())==6
    assert out['all']['partitions']['raw']['raw_invalid']==3
    assert out['all']['partitions']['local']['near_tied_min']>0
    assert out['all']['partitions']['raw_local']['different_tied_min']>0

def test_bad_selection_is_rejected():
    a=np.zeros((1,3),np.uint32);s=np.ones_like(a,bool);lo=np.zeros_like(a,int);hi=np.full_like(a,2,int)
    with pytest.raises(ValueError):local_stats(a,a,s,s,lo,hi,0,np.full(a.shape,2.,np.float32),s,2)

def test_truth_near_support_and_lr_partition():
    a=np.zeros((1,5),np.uint32);support=np.ones_like(a,bool);other=np.zeros_like(a,bool);other[0,0]=True
    lo=np.zeros_like(a,int);hi=np.full_like(a,3,int);raw=np.full(a.shape,np.nan,np.float32);valid=np.zeros(a.shape,bool);lr=valid.copy();metric=valid.copy()
    stats=local_stats(a,a,support,other,lo,hi,0,raw,valid,3)
    truth={k:np.ones_like(a,bool) for k in ['valid','visible','remap_support','region_thin']};truth.update({k:np.zeros_like(a,bool) for k in ['out_of_range','off_image','mixed_bilinear']});truth['disparity']=np.full(a.shape,1.);truth['surface_id']=np.ones_like(a)
    got=summarize(a,a,support,other,lo,hi,0,raw,valid,lr,metric,truth,stats)['all']
    assert got['partitions']['prior']['inside_supported']==3
    assert got['partitions']['prior']['inside_unsupported']==2
    assert got['partitions']['final']=={'raw_invalid':5}
    assert sum(got['cross'].values())==5


def test_wrong_frozen_hash_rejected(tmp_path,monkeypatch):
    import sys
    from benchmarks.compact_sgm_population import main
    for suffix in ('.npz','.npz.selection.npz','.json'):(tmp_path/('compact-fixed'+suffix)).write_bytes(b'forged')
    monkeypatch.setattr(sys,'argv',['population','--mode','compact-fixed','--artifact-root',str(tmp_path),'--family-root',str(tmp_path),'--native-report',str(tmp_path/'native'),'--dump-root',str(tmp_path),'--truthfree',str(tmp_path/'truthfree'),'--output',str(tmp_path/'output')])
    with pytest.raises(ValueError,match='frozen artifact mismatch'):main()
    assert not (tmp_path/'output').exists()
