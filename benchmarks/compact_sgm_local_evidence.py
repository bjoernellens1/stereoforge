"""Offline local stereo costs on pinned inputs; does not change matching."""
import hashlib,json
from pathlib import Path
import numpy as np
from benchmarks.compact_sgm_population import ARTIFACTS, BYTE_COUNTS, sha
import benchmarks.compact_sgm_population as population

MODES=('census','center','vertical','joint')
EXPOSURES=('identity','offset25','gain08_bias20')
INVALID=np.iinfo(np.uint32).max
POLICY={'candidate_interval':'frozen per-eye [lo,hi), D128','support':'both 5x5 Census footprints; implies 1x5 and horizontal neighbours', 'truth_tolerance':1.,'tie':'smallest disparity, report all ties','costs':{'census':'Hamming', 'center':'256*Hamming + abs(centerL-centerR)', 'vertical':'4096*Hamming + sum(abs((L_i-L_center)-(R_i-R_center)), i=-2..2)', 'joint':'2*Hamming+floor(min(abs(centerL-centerR),32)/4)+floor(min(abs((L[x+1]-L[x-1])-(R[x+1]-R[x-1])),64)/8)'},'exposures':'identity; clip(right+25); clip(rint(0.8*right+20)); 8-bit synthetic transforms', 'source_eye':'right image transformed; each eye matches own reference independently'}


def transform(image,name):
    if image.dtype!=np.uint8:raise ValueError('expected uint8')
    a=image.astype(np.float64)
    if name=='identity':return image.copy()
    if name=='offset25':a=a+25
    elif name=='gain08_bias20':a=np.rint(.8*a+20)
    else:raise ValueError('unknown exposure')
    return np.clip(a,0,255).astype(np.uint8)


def features(image):
    """Signed stencil arrays; sampling only where Census support is valid."""
    im=image.astype(np.int16);return im,tuple(np.roll(im,-dy,axis=0)-im for dy in (-2,-1,0,1,2)),np.roll(im,-1,axis=1)-np.roll(im,1,axis=1)


def candidate(own,other,desc_a,desc_b,support_a,support_b,lo,hi,eye,q,mode):
    if eye not in (0,1) or mode not in MODES:raise ValueError('bad mode or eye')
    h,w=desc_a.shape;q=np.broadcast_to(q,(h,w));x=np.broadcast_to(np.arange(w),(h,w));y=np.broadcast_to(np.arange(h)[:,None],(h,w));ox=x+q if eye else x-q;target=np.clip(ox,0,w-1).astype(np.intp)
    valid=(q>=0)&(q>=lo)&(q<hi)&(ox>=0)&(ox<w)&support_a&support_b[y,target]
    bits=np.bitwise_xor(desc_a,desc_b[y,target]);ham=(BYTE_COUNTS[bits&255]+BYTE_COUNTS[(bits>>8)&255]+BYTE_COUNTS[(bits>>16)&255]+BYTE_COUNTS[(bits>>24)&255]).astype(np.uint32)
    ai,av,ag=own;bi,bv,bg=other;center=np.abs(ai-bi[y,target]).astype(np.uint32)
    if mode=='census':cost=ham
    elif mode=='center':cost=ham*256+center
    elif mode=='vertical':cost=ham*4096+sum(np.abs(a-b[y,target]).astype(np.uint32) for a,b in zip(av,bv))
    else:cost=2*ham+np.minimum(center,32)//4+np.minimum(np.abs(ag-bg[y,target]).astype(np.uint32),64)//8
    return np.where(valid,cost,INVALID).astype(np.uint32)


def enumerate_min(own,other,desc_a,desc_b,support_a,support_b,lo,hi,eye,disparities,mode):
    shape=desc_a.shape;best=np.full(shape,INVALID,np.uint32);winner=np.full(shape,-1,np.int16);ties=np.zeros(shape,np.uint16);count=np.zeros(shape,np.uint16)
    for q in range(disparities):
        c=candidate(own,other,desc_a,desc_b,support_a,support_b,lo,hi,eye,q,mode);good=c!=INVALID;better=c<best;equal=(c==best)&good
        count+=good;winner[better]=q;ties=np.where(better,1,ties+equal).astype(np.uint16);best=np.minimum(best,c)
    return {'best':best,'winner':winner,'ties':ties,'candidate_count':count}


def score(own,other,desc_a,desc_b,support_a,support_b,lo,hi,eye,truth,result):
    gt=truth['disparity'];finite=np.isfinite(gt);valid=np.asarray(truth['valid'],bool)&np.asarray(truth['visible'],bool)&~np.asarray(truth['out_of_range'],bool)&~np.asarray(truth['off_image'],bool)&np.asarray(truth['remap_support'],bool)&~np.asarray(truth['mixed_bilinear'],bool)&np.asarray(truth['region_thin'],bool)&support_a&finite
    base=np.floor(np.where(finite,gt,0)).astype(np.int32);near_best=np.full(gt.shape,INVALID,np.uint32)
    for offset in (-1,0,1,2):
        q=base+offset;near=finite&(np.abs(q-gt)<=1)&(q>=0)&(q<128)
        c=candidate(own,other,desc_a,desc_b,support_a,support_b,lo,hi,eye,q,result['mode']);near_best=np.minimum(near_best,np.where(near,c,INVALID).astype(np.uint32))
    best,winner,ties=result['best'],result['winner'],result['ties'];near_supported=near_best!=INVALID
    # Three disjoint possible local outcomes; tie category does not silently choose a match.
    local=np.where(best==INVALID,'no_candidate',np.where(ties>1,'ambiguous',np.where(np.abs(winner-gt)<=1,'unique_correct','unique_wrong')))
    near=np.where(~near_supported,'no_supported_near',np.where(near_best>best,'near_cost_higher',np.where(ties>1,'near_tied_min','near_unique_min')))
    def counts(mask):
        out={'eligible':int(mask.sum())}
        for name,arr in (('local',local),('near',near)):
            vals,nums=np.unique(arr[mask],return_counts=True);out[name]={str(k):int(v) for k,v in zip(vals,nums)}
            if sum(out[name].values())!=out['eligible']:raise ValueError('partition failure')
        out['candidate_evaluations']=int(result['candidate_count'][mask].sum(dtype=np.uint64));return out
    surfaces=np.asarray(truth['surface_id']);out={'all':counts(valid),'surfaces':{str(int(s)):counts(valid&(surfaces==s)) for s in np.unique(surfaces[valid])}}
    if sum(v['eligible'] for v in out['surfaces'].values())!=out['all']['eligible']:raise ValueError('surface partition failure')
    return out


def main():
    import argparse,sys
    sys.path.insert(0,str(Path(__file__).parent))
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    import stereoforge.refinement as refinement
    from stereoforge.refinement import census
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact-root','family-root','native-report','dump-root','output-root'):p.add_argument('--'+name,required=True)
    args=p.parse_args();artifact=Path(args.artifact_root);family=Path(args.family_root);out=Path(args.output_root);out.mkdir(parents=True,exist_ok=True)
    coarse,target,family_file=family/'half/manifest.json',family/'full/manifest.json',family/'family.json';native=Path(args.native_report)
    frozen={};source_hash=sha(Path(__file__));population_hash=sha(Path(population.__file__))
    for mode,expected in ARTIFACTS.items():
        f=artifact/(mode+'.npz');s=artifact/(mode+'.npz.selection.npz');r=artifact/(mode+'.json')
        if tuple(map(sha,(f,s,r)))!=expected:raise ValueError('frozen artifact mismatch')
        report=cmp.strict_json(r.read_text())
        if report['hashes']['frozen_intervals']!=sha(f) or report['hashes']['frozen_selection']!=sha(s):raise ValueError('report mismatch')
        with np.load(f,allow_pickle=False) as a:
            lo,hi=a['lo'].copy(),a['hi'].copy();recorded=json.loads(str(a['input_hashes']));policy=str(a['policy']);config=json.loads(str(a['config']))
        if policy!='P2=32' or config['disparities']!=128:raise ValueError('unexpected matcher')
        for key,path in (('coarse',coarse),('target',target),('family',family_file),('report',native)):
            if sha(path)!=recorded[key]:raise ValueError('frozen input mismatch '+key)
        for name,digest in recorded['dependencies'].items():
            if sha(Path(__file__).parent/name)!=digest:raise ValueError('dependency mismatch '+name)
        if sha(Path(__file__).parent/'compact_sgm_step_experiment.py')!=recorded['source'] or sha(Path(refinement.__file__))!=recorded['refinement_source']:raise ValueError('source mismatch')
        frozen[mode]=(lo,hi,report,expected)
    ctx=cmp.load_context(coarse,target);nr=cmp.strict_json(native.read_text());diag.validate_report(cmp,nr,ctx,cmp.oracle.Settings(radius=2,subpixel=False))
    h,w=ctx['images'].shape[1:];planes=cmp._read_planes(nr['files'],cmp.PLANES,(2,h,w),args.dump_root);support=diag.verified_census_support(ctx['images'],ctx['support'],planes['census_validity.u32'],cmp.oracle)
    if any(lo.shape!=(2,h,w) or hi.shape!=(2,h,w) for lo,hi,_,_ in frozen.values()):raise ValueError('bounds shape mismatch')
    if np.any(ctx['images']>255):raise ValueError('invalid luma')
    base=ctx['images'].astype(np.uint8);paths={};saturation={}
    def pair(exp):
        images=[base[0],transform(base[1],exp)];desc=[census(images[i],ctx['support'][i]) for i in (0,1)]
        if any(not np.array_equal(desc[i].support,support[i]) for i in (0,1)):raise ValueError('support mismatch')
        return images,desc,[features(x) for x in images]
    for exp in EXPOSURES:
        images,desc,feature=pair(exp);saturation[exp]={'zero':int((images[1]==0).sum()),'max':int((images[1]==255).sum()),'pixels':h*w}
        for mode,(lo,hi,_,_) in frozen.items():
            for costmode in MODES:
                arrays={}
                for eye in (0,1):
                    a,b=desc[eye],desc[1-eye]
                    result=enumerate_min(feature[eye],feature[1-eye],a.descriptor,b.descriptor,a.support,b.support,lo[eye],hi[eye],eye,128,costmode)
                    for key,value in result.items():arrays[f'{eye}_{key}']=value
                path=out/f'{mode}-{exp}-{costmode}.npz'
                with open(path,'xb') as fh:np.savez_compressed(fh,**arrays)
                paths[(mode,exp,costmode)]=(path,sha(path))
    # All 24 configuration files, each with both eye selections, exist before truth is loaded.
    if len(paths)!=len(frozen)*len(EXPOSURES)*len(MODES) or sha(Path(__file__))!=source_hash or sha(Path(population.__file__))!=population_hash:raise ValueError('incomplete freeze')
    for (mode,exp,costmode),(path,digest) in paths.items():
        if sha(path)!=digest:raise ValueError('frozen local evidence changed')
    truth=cmp.load_truth(family_file,ctx)
    result={'scope':'local-cost-only post-hoc synthetic development ablation; no GPU or stereo output claims','policy':POLICY,'source_sha256':source_hash,'population_source_sha256':population_hash,'source_id':ctx['source_id'],'saturation':saturation,'inputs':{m:list(v[3]) for m,v in frozen.items()},'truthfree':{f'{m}-{e}-{c}':digest for (m,e,c),(_,digest) in paths.items()},'results':{}}
    for exp in EXPOSURES:
        images,desc,feature=pair(exp)
        for mode,(lo,hi,report,_) in frozen.items():
            for costmode in MODES:
                path,digest=paths[(mode,exp,costmode)]
                if sha(path)!=digest:raise ValueError('local freeze changed')
                with np.load(path,allow_pickle=False) as a:
                    eyes={}
                    for eye,name in enumerate(('left','right')):
                        local={key:a[f'{eye}_{key}'].copy() for key in ('best','winner','ties','candidate_count')};local['mode']=costmode
                        aa,bb=desc[eye],desc[1-eye]
                        scored=score(feature[eye],feature[1-eye],aa.descriptor,bb.descriptor,aa.support,bb.support,lo[eye],hi[eye],eye,truth[name],local)
                        if scored['all']['eligible']!=report['quality'][name]['thin']['eligible']:raise ValueError('denominator mismatch')
                        eyes[name]=scored
                result['results'][f'{mode}-{exp}-{costmode}']=eyes
    if sha(Path(__file__))!=source_hash or sha(Path(population.__file__))!=population_hash:raise ValueError('source changed')
    with open(out/'result.json','x') as fh:json.dump(result,fh,indent=2,sort_keys=True,allow_nan=False);fh.write('\n')

if __name__=='__main__':main()
