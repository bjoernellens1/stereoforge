"""CPU-only FP1 tile-hypothesis recall; candidates are not depth."""
import hashlib
from pathlib import Path
import numpy as np

POLICY={'tile':8,'max_points':4,'gradient_points':3,'point_separation_chebyshev':2,'minima_per_point':2,'minimum_separation':4,'max_hypotheses_per_tile':8,'window_radius':4,'disparities':128,'truth_tolerance':1.0,'fallback':'no fabricated candidate; invalid when missing', 'ties':'only contiguous equal-cost minimum plateaus flagged; separate equal-cost minima are not plateau ties','support':'both-eye Census footprint at each candidate'}

def tile_points(image,support,ty,tx):
    h,w=image.shape; y0,x0=ty*8,tx*8; y1,x1=min(y0+8,h),min(x0+8,w)
    if y0>=h or x0>=w:return []
    selected=[];cy,cx=y0+4,x0+4
    if cy<y1 and cx<x1 and support[cy,cx]:selected.append((cy,cx))
    max_selected=4 if selected else 3
    options=[]
    for y in range(y0,y1):
        for x in range(x0,x1):
            if not support[y,x] or x==0 or x+1>=w or y==0 or y+1>=h:continue
            grad=abs(int(image[y,x+1])-int(image[y,x-1]))+abs(int(image[y+1,x])-int(image[y-1,x]))
            if grad:options.append((-grad,y,x))
    for _,y,x in sorted(options):
        if all(max(abs(y-v),abs(x-u))>=2 for v,u in selected):selected.append((y,x))
        if len(selected)>=max_selected:break
    return selected

def local_minima(costs,limit=2,separation=4):
    a=np.asarray(costs);n=len(a);peaks=[];i=0
    while i<n:
        if not np.isfinite(a[i]):i+=1;continue
        j=i+1
        while j<n and a[j]==a[i]:j+=1
        left=a[i-1] if i else np.inf;right=a[j] if j<n else np.inf
        if a[i]<left and a[i]<right:peaks.append((i,int(a[i]),j-i>1))
        i=j
    selected=[]
    for p in sorted(peaks,key=lambda x:(x[1],x[0])):
        if all(abs(p[0]-q[0])>=separation for q in selected):selected.append(p)
        if len(selected)>=limit:break
    return selected

def point_search(own,other,support,other_support,y,x,eye,disparities):
    if eye not in (0,1):raise ValueError('bad eye')
    h,w=own.shape;cost=np.full(disparities,np.inf);attempts=0;supported=0
    if not (0<=y<h and 0<=x<w) or not support[y,x]:return {'peaks':[],'attempts':0,'supported':0}
    for d in range(disparities):
        attempts+=1;ox=x+d if eye else x-d
        if 0<=ox<w and other_support[y,ox]:cost[d]=(int(own[y,x])^int(other[y,ox])).bit_count();supported+=1
    return {'peaks':local_minima(cost),'attempts':attempts,'supported':supported}

def construct_sets(shape,support,other_support,lo,hi,prior,tiles,eye,disparities=128,window=4):
    h,w=shape
    if eye not in (0,1) or disparities<1 or disparities>128 or any(x.shape!=shape for x in (support,other_support,lo,hi,prior)) or np.any(lo<0) or np.any(hi>disparities) or np.any(lo>hi):raise ValueError('invalid candidate input')
    tile_h=(h+7)//8;tile_w=(w+7)//8;bits=np.zeros((tile_h,tile_w,disparities),bool)
    for (ty,tx),peaks in tiles.items():
        if not (0<=ty<tile_h and 0<=tx<tile_w) or len(peaks)>8:raise ValueError('invalid tile hypotheses')
        for d,_ in peaks:
            if not 0<=d<disparities:raise ValueError('invalid hypothesis')
            bits[ty,tx,max(0,d-window):min(disparities,d+window+1)]=True
    blocks=(disparities+31)//32;result={k:np.zeros((h,w,blocks),np.uint32) for k in ('coarse','local','union','full')}
    for q in range(disparities):
        target=np.zeros(shape,bool)
        if q<w:
            if eye:target[:,:w-q]=other_support[:,q:]
            else:target[:,q:]=other_support[:,:w-q]
        available=support&target;coarse=available&prior&(q>=lo)&(q<hi);local=available&np.repeat(np.repeat(bits[:,:,q],8,axis=0),8,axis=1)[:h,:w]
        flag=np.uint32(1<<q%32);b=q//32
        result['coarse'][:,:,b] |= np.where(coarse,flag,0).astype(np.uint32)
        result['local'][:,:,b] |= np.where(local,flag,0).astype(np.uint32)
        result['union'][:,:,b] |= np.where(coarse|local,flag,0).astype(np.uint32)
        result['full'][:,:,b] |= np.where(available,flag,0).astype(np.uint32)
    return result

def verify_hashes(expected):
    for path,digest in expected.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('input changed: '+path)

def contains_near(bits,truth):
    h,w=truth.shape;y,x=np.indices((h,w));finite=np.isfinite(truth);base=np.floor(np.where(finite,truth,0)).astype(np.int32);out=np.zeros((h,w),bool)
    for delta in (-1,0,1,2):
        q=base+delta;good=finite&(q>=0)&(q<128)&(np.abs(q-truth)<=1);safe=np.clip(q,0,127);block=safe//32;flag=np.left_shift(np.uint32(1),(safe%32).astype(np.uint32));out|=good&((bits[y,x,block]&flag)!=0)
    return out

def count_bits(bits):
    table=np.array([i.bit_count() for i in range(256)],np.uint8)
    h,w,_=bits.shape;return table[bits.view(np.uint8).reshape(h,w,-1)].sum(axis=2,dtype=np.uint16)

def summarize(bits,truth,support,old_recall):
    t=truth;finite=np.isfinite(t['disparity']);visible=np.asarray(t['valid'],bool)&np.asarray(t['visible'],bool)&~np.asarray(t['off_image'],bool)&np.asarray(t['remap_support'],bool)&support&~np.asarray(t['mixed_bilinear'],bool)&finite
    eligible=visible&~np.asarray(t['out_of_range'],bool);near=contains_near(bits,t['disparity']);counts=count_bits(bits)
    def region(mask):
        domain=eligible&mask;missing=domain&~old_recall;recovered=missing&near
        return {'eligible':int(domain.sum()),'represented':int((domain&near).sum()),'missing':int((domain&~near).sum()),'old_missing':int(missing.sum()),'recovered_vs_old':int(recovered.sum()),'out_of_range_visible':int((visible&mask&np.asarray(t['out_of_range'],bool)).sum())}
    result={name:region(np.ones(support.shape,bool) if name=='all' else np.asarray(t['region_'+name],bool)) for name in ('all','thin','near','edges')}
    result['thin_surfaces']={str(int(sid)):region(np.asarray(t['region_thin'],bool)&(t['surface_id']==sid)) for sid in np.unique(t['surface_id'][eligible&np.asarray(t['region_thin'],bool)])}
    result['work_distribution']={'own_supported_pixels':int(support.sum()),'at_most_32':int((support&(counts<=32)).sum()),'nonempty':int((support&(counts>0)).sum()),'nonempty_at_most_32':int((support&(counts>0)&(counts<=32)).sum()),'candidate_sum':int(counts.sum(dtype=np.uint64)),'candidate_histogram':{str(int(k)):int(v) for k,v in zip(*np.unique(counts[support],return_counts=True))}}
    thin=result['thin'];work=result['work_distribution'];result['provisional_screen']={'recover_half_old_missing':bool(thin['old_missing'] and 2*thin['recovered_vs_old']>=thin['old_missing']),'nonempty_and_at_most_32_for_90_percent':bool(work['own_supported_pixels'] and 10*work['nonempty_at_most_32']>=9*work['own_supported_pixels'])}
    if result['all']['represented']+result['all']['missing']!=result['all']['eligible'] or sum(x['eligible'] for x in result['thin_surfaces'].values())!=result['thin']['eligible']:raise ValueError('partition failure')
    return result


def main():
    import argparse,json,sys
    sys.path.insert(0,str(Path(__file__).parent))
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    import stereoforge.refinement as refinement
    from stereoforge.refinement import census
    from benchmarks.compact_sgm_population import ARTIFACTS,sha
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('artifact-root','family-root','native-report','dump-root','output-root'):p.add_argument('--'+name,required=True)
    args=p.parse_args();artifact=Path(args.artifact_root);family=Path(args.family_root);out=Path(args.output_root);out.mkdir(parents=True,exist_ok=True)
    f=artifact/'compact-fixed.npz';selection=artifact/'compact-fixed.npz.selection.npz';report=artifact/'compact-fixed.json'
    if tuple(map(sha,(f,selection,report)))!=ARTIFACTS['compact-fixed']:raise ValueError('pinned artifact mismatch')
    r=cmp.strict_json(report.read_text())
    if r['hashes']['frozen_intervals']!=sha(f) or r['hashes']['frozen_selection']!=sha(selection):raise ValueError('report mismatch')
    coarse,target,family_file=family/'half/manifest.json',family/'full/manifest.json',family/'family.json';native=Path(args.native_report)
    with np.load(f,allow_pickle=False) as a:
        old_lo=a['lo'].copy();old_hi=a['hi'].copy();recorded=json.loads(str(a['input_hashes']))
    for key,path in (('coarse',coarse),('target',target),('family',family_file),('report',native)):
        if sha(path)!=recorded[key]:raise ValueError('input mismatch '+key)
    root=Path(__file__).parent
    for name,digest in recorded['dependencies'].items():
        if sha(root/name)!=digest:raise ValueError('dependency mismatch '+name)
    if sha(root/'compact_sgm_step_experiment.py')!=recorded['source'] or sha(Path(refinement.__file__))!=recorded['refinement_source']:raise ValueError('source mismatch')
    src_paths=[Path(__file__),root/'compact_sgm_population.py',root/'compare_daxa_refinement.py',root/'fast60_prior_diagnostic.py',Path(refinement.__file__),coarse,target,family_file,native,f,selection,report]
    hashes={str(path):sha(path) for path in src_paths}
    ctx=cmp.load_context(coarse,target);native_report=cmp.strict_json(native.read_text());diag.validate_report(cmp,native_report,ctx,cmp.oracle.Settings(radius=2,subpixel=False))
    h,w=ctx['images'].shape[1:];planes=cmp._read_planes(native_report['files'],cmp.PLANES,(2,h,w),args.dump_root)
    diag.validate_wire_masks({k:planes[k] for k in ('prediction_validity.u32','raw_validity.u32','validity.u32','census_validity.u32')})
    diag.validate_wire_values(planes,128,2);support=diag.verified_census_support(ctx['images'],ctx['support'],planes['census_validity.u32'],cmp.oracle)
    desc=[census(ctx['images'][i],ctx['support'][i]) for i in (0,1)]
    if any(not np.array_equal(desc[i].support,support[i]) for i in (0,1)):raise ValueError('support mismatch')
    if old_lo.shape!=(2,h,w) or old_hi.shape!=(2,h,w):raise ValueError('old interval shape mismatch')
    all_sets={k:[] for k in ('coarse','local','union','full','old_compact')};tile_meta=[];work=[]
    for eye in (0,1):
        tymax,txmax=(h+7)//8,(w+7)//8;hyp=np.full((tymax,txmax,8),-1,np.int16);tie=np.zeros((tymax,txmax,8),bool);source=np.full((tymax,txmax,8,2),-1,np.int16);points=np.full((tymax,txmax,4,2),-1,np.int16);point_counts=np.zeros((tymax,txmax,4,3),np.uint16)
        tiles={};stats={'tiles':tymax*txmax,'selected_points':0,'full_search_attempts':0,'supported_hamming_operations':0,'local_minima_retained':0,'plateau_ties':0,'tile_hypothesis_overflow':0,'census_descriptor_comparisons_own_eye':int(24*desc[eye].support.sum()),'candidate_filter_slots_per_config':int(h*w*128),'gradient_evaluations':int(desc[eye].support[1:-1,1:-1].sum()),'grid_point_checks':int(tymax*txmax)}
        for ty in range(tymax):
            for tx in range(txmax):
                peaks=[]
                for pi,(y,x) in enumerate(tile_points(ctx['images'][eye],desc[eye].support,ty,tx)):
                    points[ty,tx,pi]=[y,x]
                    stats['selected_points']+=1;found=point_search(desc[eye].descriptor,desc[1-eye].descriptor,desc[eye].support,desc[1-eye].support,y,x,eye,128)
                    point_counts[ty,tx,pi]=[found['attempts'],found['supported'],len(found['peaks'])]
                    stats['full_search_attempts']+=found['attempts'];stats['supported_hamming_operations']+=found['supported']
                    for disparity,cost,is_tie in found['peaks']:peaks.append((disparity,is_tie,y,x));stats['plateau_ties']+=int(is_tie)
                stats['tile_hypothesis_overflow']+=max(0,len(peaks)-8);peaks=peaks[:8];stats['local_minima_retained']+=len(peaks)
                if peaks:
                    tiles[(ty,tx)]=[(d,t) for d,t,_,_ in peaks]
                    for i,(d,t,y,x) in enumerate(peaks):hyp[ty,tx,i]=d;tie[ty,tx,i]=t;source[ty,tx,i]=[y,x]
        prior=np.asarray(planes['prediction_validity.u32'][eye],bool)&support[eye]
        lo=np.where(prior,planes['lower.u32'][eye],0).astype(np.int32);hi=np.where(prior,planes['upper.u32'][eye]+1,0).astype(np.int32)
        if np.any(lo<0) or np.any(hi>128) or np.any(lo>hi):raise ValueError('invalid prediction intervals')
        result=construct_sets((h,w),support[eye],support[1-eye],lo,hi,prior,tiles,eye)
        old=construct_sets((h,w),support[eye],support[1-eye],old_lo[eye],old_hi[eye],np.ones((h,w),bool),{},eye)['coarse']
        for name in ('coarse','local','union','full'):all_sets[name].append(result[name])
        all_sets['old_compact'].append(old);tile_meta.append({'hyp':hyp,'tie':tie,'source':source,'points':points,'point_counts':point_counts});work.append(stats)
    outputs={}
    # Candidate and accounting decisions for every configuration precede truth.
    for name,values in all_sets.items():
        path=out/(name+'.npz')
        with open(path,'xb') as fh:np.savez_compressed(fh,candidates=np.stack(values))
        outputs[name]=sha(path)
    meta=out/'tiles.npz'
    with open(meta,'xb') as fh:np.savez_compressed(fh,**{f'{eye}_{key}':value for eye,row in enumerate(tile_meta) for key,value in row.items()})
    meta_sha=sha(meta);manifest=out/'pretruth.json'
    pre={'policy':POLICY,'sources':hashes,'source_id':ctx['source_id'],'candidate_hashes':outputs,'tile_hash':meta_sha,'work':work,'tile_meta_bytes':sum(v.nbytes for row in tile_meta for v in row.values()),'retained_candidate_bytes_per_config':int(sum(a.nbytes for a in all_sets['coarse'])),'note':'tile hypotheses are shared proposal regions, not stereo observations or depth; sorting, feature materialization, prior generation and downstream recurrence are not measured'}
    with open(manifest,'x') as fh:json.dump(pre,fh,indent=2,sort_keys=True,allow_nan=False);fh.write('\n')
    manifest_sha=sha(manifest)
    verify_hashes(hashes)
    for name,digest in outputs.items():
        if sha(out/(name+'.npz'))!=digest:raise ValueError('candidate mutation')
    if sha(meta)!=meta_sha:raise ValueError('tile mutation')
    truth=cmp.load_truth(family_file,ctx);result={'scope':'post-hoc analytic candidate recall only; no depth, speed or GPU claim','pretruth_sha256':manifest_sha,'results':{}}
    for eye,name in enumerate(('left','right')):
        old_recall=contains_near(all_sets['old_compact'][eye],truth[name]['disparity']);result['results'][name]={}
        for config,values in all_sets.items():result['results'][name][config]=summarize(values[eye],truth[name],support[eye],old_recall)
    if sha(manifest)!=manifest_sha or sha(meta)!=meta_sha:raise ValueError('pretruth mutation')
    for name,digest in outputs.items():
        if sha(out/(name+'.npz'))!=digest:raise ValueError('candidate mutation during scoring')
    with open(out/'result.json','x') as fh:json.dump(result,fh,indent=2,sort_keys=True,allow_nan=False);fh.write('\n')

if __name__=='__main__':main()
