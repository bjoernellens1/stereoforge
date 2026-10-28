"""Post-hoc thin-scene census diagnostics; no truth enters matching decisions."""
import hashlib
import json
from pathlib import Path
import numpy as np

BYTE_COUNTS = np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
INF=255
POLICY={'disparities':128,'truth_tolerance':1.0,'local_tie':'lowest disparity',
        'truth_domain':'visible in-range unmixed thin truth with own Census and remap support',
        'categories':'cross-product, exact integer counts; raw invalid has no retained winner'}
ARTIFACTS={
 'compact-fixed':('c36edc76f2e8554dccd273510d2b7af8ecd43556f63fa70fba53668d2b95ed40','a48ce724042ca3598f39fcf4544114ced53b49130566bf234637315600d566a0','3884c9772b8027d345e4b36b64dd34738f0253c10fd4d2b799084fe78cb64a52'),
 'full-fixed':('3f4b85e65dda18dfdf08e87c7f3ca16b4200e0b02928c2a9aad3d39ebb3a97da','88ba7d9a01a4bdd030e92f8c28357b3fb38ca34d5b8127fbf3c5e6b91bf4b56a','129b54718c04ef7b05f841319455cb70fd8e0445ad0b5f2d9d7b3a5df22c6319')}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cost_plane(own,other,own_support,other_support,q,eye,lo,hi):
    """Local integer Census costs; q scalar or HxW; no truth required."""
    h,w=own.shape; q=np.broadcast_to(q,(h,w));x=np.broadcast_to(np.arange(w),(h,w));y=np.broadcast_to(np.arange(h)[:,None],(h,w))
    ox=x+q if eye else x-q
    valid=(q>=lo)&(q<hi)&(ox>=0)&(ox<w)&own_support
    target=np.clip(ox,0,w-1).astype(np.intp);valid &= other_support[y,target]
    v=np.bitwise_xor(own,other[y,target]).astype(np.uint32)
    pop=BYTE_COUNTS[v & 255]+BYTE_COUNTS[(v>>8)&255]+BYTE_COUNTS[(v>>16)&255]+BYTE_COUNTS[(v>>24)&255]
    return np.where(valid,pop,INF).astype(np.uint8)


def local_stats(own,other,own_support,other_support,lo,hi,eye,raw,raw_valid,disparities):
    if eye not in (0,1) or own.shape!=other.shape or any(x.shape!=own.shape for x in (own_support,other_support,lo,hi,raw,raw_valid)):
        raise ValueError('shape or eye mismatch')
    if np.any(lo<0) or np.any(hi>disparities) or np.any(lo>hi): raise ValueError('bad bounds')
    best=np.full(own.shape,INF,np.uint8);winner=np.full(own.shape,-1,np.int16);ties=np.zeros(own.shape,np.uint16)
    for q in range(disparities):
        c=cost_plane(own,other,own_support,other_support,q,eye,lo,hi);better=c<best;equal=(c==best)&(c<INF)
        winner[better]=q;ties=np.where(better,1,ties+equal).astype(np.uint16);best=np.minimum(best,c)
    raw_q=np.where(raw_valid,raw,0).astype(np.int32)
    raw_cost=cost_plane(own,other,own_support,other_support,raw_q,eye,lo,hi)
    if np.any(raw_valid & ((~np.isfinite(raw))|(raw!=raw_q)|(raw_cost==INF))): raise ValueError('invalid raw selection')
    return {'local_best':best,'local_winner':winner,'local_ties':ties,'raw_cost':raw_cost}


def summarize(own,other,own_support,other_support,lo,hi,eye,raw,raw_valid,lr,metric,truth,stats):
    shape=own.shape
    if any(np.asarray(truth[k]).shape!=shape for k in ('valid','visible','out_of_range','off_image','remap_support','mixed_bilinear','region_thin','disparity','surface_id')): raise ValueError('truth shape mismatch')
    if np.any(lr & ~raw_valid) or np.any(metric & ~lr):raise ValueError('invalid masks')
    gt=np.asarray(truth['disparity']);finite=np.isfinite(gt)
    eligible=(np.asarray(truth['valid'],bool)&np.asarray(truth['visible'],bool)&~np.asarray(truth['out_of_range'],bool)&~np.asarray(truth['off_image'],bool)&np.asarray(truth['remap_support'],bool)&own_support&~np.asarray(truth['mixed_bilinear'],bool)&np.asarray(truth['region_thin'],bool)&finite)
    base=np.floor(np.where(finite,gt,0)).astype(np.int32);near_best=np.full(shape,INF,np.uint8);interval=np.zeros(shape,bool)
    for offset in (-1,0,1,2):
        q=base+offset;near=finite & (np.abs(q-gt)<=1)&(q>=0)&(q<128)
        interval |= near & (q>=lo)&(q<hi)
        c=cost_plane(own,other,own_support,other_support,q,eye,lo,hi);near_best=np.minimum(near_best,np.where(near,c,INF).astype(np.uint8))
    supported=near_best<INF;best=stats['local_best'];ties=stats['local_ties'];localwinner=stats['local_winner']
    # Encode disjoint categories; include all categories even for absent support.
    prior=np.where(~interval,'outside_interval',np.where(~supported,'inside_unsupported','inside_supported'))
    local=np.where(~supported,'no_supported_near',np.where(near_best>best,'near_cost_higher',np.where(ties==1,'near_unique_min','near_tied_min')))
    raw_state=np.where(~raw_valid,'raw_invalid',np.where(np.abs(raw-gt)<=1,'raw_correct','raw_wrong'))
    raw_local=np.where(~raw_valid,'raw_invalid',np.where(raw==localwinner,'same_tiebroken_winner',np.where(stats['raw_cost']==best,'different_tied_min','higher_cost')))
    final=np.where(~raw_valid,'raw_invalid',np.where(~lr,'lr_rejected',np.where(metric,'metric_accepted','lr_accepted_metric_invalid')))
    surface=np.asarray(truth['surface_id']);result={}
    def counts(mask):
        rows={}
        for key,arr in [('prior',prior),('local',local),('raw',raw_state),('raw_local',raw_local),('final',final)]:
            values,nums=np.unique(arr[mask],return_counts=True); rows[key]={str(v):int(n) for v,n in zip(values,nums)}
            if sum(rows[key].values()) != int(mask.sum()): raise ValueError('partition mismatch')
        pairs={}
        for i in np.flatnonzero(mask):
            key='|'.join((str(prior.flat[i]),str(local.flat[i]),str(raw_state.flat[i]),str(raw_local.flat[i]),str(final.flat[i])))
            pairs[key]=pairs.get(key,0)+1
        if sum(pairs.values()) != int(mask.sum()): raise ValueError('cross partition mismatch')
        return {'eligible':int(mask.sum()),'partitions':rows,'cross':pairs}
    result['all']=counts(eligible);result['surfaces']={str(int(s)):counts(eligible&(surface==s)) for s in np.unique(surface[eligible])}
    if sum(row['eligible'] for row in result['surfaces'].values())!=result['all']['eligible']:raise ValueError('surface mismatch')
    return result


def main():
    import argparse
    import sys
    sys.path.insert(0,str(Path(__file__).parent))
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    import stereoforge.refinement as refinement
    from stereoforge.refinement import census
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=tuple(ARTIFACTS),required=True)
    for name in ('artifact-root','family-root','native-report','dump-root','truthfree','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();artifact=Path(args.artifact_root);family=Path(args.family_root)
    f=artifact/(args.mode+'.npz');s=artifact/(args.mode+'.npz.selection.npz');r=artifact/(args.mode+'.json')
    if tuple(map(sha,(f,s,r)))!=ARTIFACTS[args.mode]:raise ValueError('frozen artifact mismatch')
    report=cmp.strict_json(r.read_text());
    if report['hashes']['frozen_intervals']!=sha(f) or report['hashes']['frozen_selection']!=sha(s):raise ValueError('report mismatch')
    coarse=family/'half/manifest.json';target=family/'full/manifest.json';family_file=family/'family.json';native=Path(args.native_report)
    with np.load(f,allow_pickle=False) as archive:
        lo=archive['lo'].copy();hi=archive['hi'].copy();recorded=json.loads(str(archive['input_hashes']));policy=str(archive['policy']);config=json.loads(str(archive['config']))
    if policy!='P2=32' or config['disparities']!=128:raise ValueError('unexpected matcher')
    for key,path in (('coarse',coarse),('target',target),('family',family_file),('report',native)):
        if sha(path)!=recorded[key]:raise ValueError('frozen input mismatch: '+key)
    root=Path(__file__).parent
    for name,digest in recorded['dependencies'].items():
        if sha(root/name)!=digest:raise ValueError('dependency mismatch: '+name)
    if sha(root/'compact_sgm_step_experiment.py')!=recorded['source'] or sha(Path(refinement.__file__))!=recorded['refinement_source']:raise ValueError('source mismatch')
    ctx=cmp.load_context(coarse,target);report_native=cmp.strict_json(native.read_text());settings=cmp.oracle.Settings(radius=2,subpixel=False)
    diag.validate_report(cmp,report_native,ctx,settings)
    h,w=ctx['images'].shape[1:];native_planes=cmp._read_planes(report_native['files'],cmp.PLANES,(2,h,w),args.dump_root)
    support=diag.verified_census_support(ctx['images'],ctx['support'],native_planes['census_validity.u32'],cmp.oracle)
    if lo.shape!=(2,h,w) or hi.shape!=(2,h,w):raise ValueError('bounds shape mismatch')
    with np.load(s,allow_pickle=False) as a:
        selected={k:a[k].copy() for k in ('raw_validity.u32','raw_disparity.f32','validity.u32','metric_validity.u32')}
    if any(v.shape!=(2,h,w) for v in selected.values()):raise ValueError('selection shape mismatch')
    descriptors=[census(ctx['images'][i],ctx['support'][i]) for i in (0,1)]
    if any(not np.array_equal(descriptors[i].support,support[i]) for i in (0,1)):raise ValueError('Census support mismatch')
    stats=[]
    for eye in (0,1):
        a,b=descriptors[eye],descriptors[1-eye]
        stats.append(local_stats(a.descriptor,b.descriptor,a.support,b.support,lo[eye],hi[eye],eye,selected['raw_disparity.f32'][eye],selected['raw_validity.u32'][eye],128))
    provenance={'freeze':sha(f),'selection':sha(s),'result':sha(r),'coarse':sha(coarse),'target':sha(target),'family':sha(family_file),'native_report':sha(native),'diagnostic_source':sha(Path(__file__)),'policy':POLICY}
    with open(args.truthfree,'xb') as output:
        np.savez_compressed(output,**{f'{eye}_{key}':value for eye,v in enumerate(stats) for key,value in v.items()},provenance=np.array(json.dumps(provenance,sort_keys=True)))
    # Truth is opened only after truth-independent local statistics are persisted.
    frozen_stats=sha(args.truthfree);truth=cmp.load_truth(family_file,ctx)
    results={}
    for eye,name in enumerate(('left','right')):
        a,b=descriptors[eye],descriptors[1-eye]
        results[name]=summarize(a.descriptor,b.descriptor,a.support,b.support,lo[eye],hi[eye],eye,selected['raw_disparity.f32'][eye],selected['raw_validity.u32'][eye],selected['validity.u32'][eye],selected['metric_validity.u32'][eye],truth[name],stats[eye])
        expected=report['quality'][name]['thin']['eligible']
        if results[name]['all']['eligible']!=expected:raise ValueError('truth domain disagrees with frozen report')
    if sha(args.truthfree)!=frozen_stats or tuple(map(sha,(f,s,r)))!=ARTIFACTS[args.mode] or sha(Path(__file__))!=provenance['diagnostic_source']:raise ValueError('input changed during analysis')
    output={'scope':'post-hoc inspected analytic development scene; no GPU, physical, held-out, or causal claim', 'policy':POLICY,'provenance':provenance,'truthfree_sha256':frozen_stats,'results':results}
    with open(args.output,'x') as fh:json.dump(output,fh,sort_keys=True,indent=2,allow_nan=False);fh.write('\n')

if __name__=='__main__':main()
