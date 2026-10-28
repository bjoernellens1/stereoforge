"""CPU-only joint-Census scanline SGM runner interface."""
from pathlib import Path
import subprocess
import numpy as np
from benchmarks.compact_sgm_reference import _inputs

def _write(root,name,a):np.asarray(a,dtype='<u4').tofile(root/name)
def _load(root,name,shape,dtype='<u4'):return np.fromfile(root/name,dtype=dtype).reshape(shape)

def run_cost(runner,directory,cost,lo,hi):
    cost,lo,hi=_inputs(cost,lo,hi,16,64);h,w,d=cost.shape;root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    wire=np.full(cost.shape,np.iinfo(np.uint32).max,dtype='<u4');finite=np.isfinite(cost);wire[finite]=cost[finite].astype(np.uint32);wire.tofile(root/'cost')
    for name,a in [('lo',lo),('hi',hi),('luma',np.zeros((h,w),np.uint32)),('own_support',np.ones((h,w),np.uint32))]:_write(root,name,a)
    subprocess.run([str(runner),str(root),str(h),str(w),str(d),'16','64'],check=True)
    paths=_load(root,'paths',(4,h,w,d)).astype(float);paths[paths==np.iinfo(np.uint32).max]=np.inf;return paths

def run_eye(runner,directory,own,other,image,other_image,lo,hi,eye,disparities,mode):
    image,other_image=np.asarray(image),np.asarray(other_image)
    if eye not in (0,1) or mode not in ('baseline','joint') or image.dtype!=np.uint8 or other_image.dtype!=np.uint8 or any(a.shape!=image.shape for a in (other_image,own.descriptor,other.descriptor,own.support,other.support,lo,hi)):raise ValueError('invalid input')
    root=Path(directory);root.mkdir(parents=True,exist_ok=True);h,w=image.shape
    for name,a in [('own',own.descriptor),('other',other.descriptor),('own_support',own.support),('other_support',other.support),('luma',image),('other_luma',other_image),('lo',lo),('hi',hi)]:_write(root,name,a)
    subprocess.run([str(runner),str(root),str(h),str(w),str(disparities),'16','64',str(eye),mode],check=True)
    result={name:_load(root,name,(h,w),dtype) for name,dtype in [('winners','<i4'),('best','<u4'),('second','<u4'),('count','<u4')]}
    result['candidate_updates'],result['interval_slots']=map(int,(root/'stats').read_text().split());return result

GATES={'thin_gain_pp':5.,'max_loss_pp':1.,'max_thin_false_increase':0,'max_surface_false_increase':0,'scope':'exploratory; every eye/exposure/interval must pass; empty support fails'}

def evaluate_gates(candidate,baseline,candidate_surface,baseline_surface):
    if not candidate['thin']['eligible'] or not baseline['thin']['eligible']:return {'pass':False,'reason':'empty thin support'}
    def compare(a,b,pp):
        an,bn=a['eligible'],b['eligible'];ac,bc=a.get('correct_metric_count'),b.get('correct_metric_count')
        if not an or not bn or type(ac) is not int or type(bc) is not int or not 0<=ac<=an or not 0<=bc<=bn:return False
        return 100*(ac*bn-bc*an)>=pp*an*bn
    checks={'thin_gain':compare(candidate['thin'],baseline['thin'],5),
            'thin_false':candidate['thin']['lr_false_accepted']<=baseline['thin']['lr_false_accepted']}
    for region in ('all','near','edges'):
        a,b=candidate[region],baseline[region]
        checks[region]=compare(a,b,-1)
    # Surface IDs with eligible thin support are supplied from independently validated truth.
    if set(candidate_surface)!=set(baseline_surface) or not candidate_surface:return {'pass':False,'reason':'missing surface support'}
    for sid,a in candidate_surface.items():
        b=baseline_surface[sid]
        checks['surface_'+sid]=bool(a['eligible'] and b['eligible'] and a['false_accepted']<=b['false_accepted'])
    return {'pass':all(checks.values()),'checks':checks}


def main():
    import argparse,hashlib,json,sys,tempfile
    sys.path.insert(0,str(Path(__file__).parent))
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    import stereoforge.refinement as refinement
    from stereoforge.refinement import census,lr_consistency
    from benchmarks.compact_sgm_population import ARTIFACTS,sha
    from benchmarks.compact_sgm_local_evidence import transform,EXPOSURES
    from benchmarks.compact_sgm_quality import select_eye
    from benchmarks.compact_sgm_step_experiment import surface_counts
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runner','build-manifest','artifact-root','family-root','native-report','dump-root','output-root'):p.add_argument('--'+name,required=True)
    args=p.parse_args();artifact=Path(args.artifact_root);family=Path(args.family_root);out=Path(args.output_root);out.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(__file__).with_name('compact_sgm_joint_runner.cpp'),Path(__file__).with_name('compact_sgm_population.py'),Path(__file__).with_name('compact_sgm_local_evidence.py'),Path(__file__).with_name('compact_sgm_quality.py'),Path(__file__).with_name('compact_sgm_step_experiment.py'),Path(__file__).with_name('compare_daxa_refinement.py'),Path(__file__).with_name('fast60_prior_diagnostic.py'),Path(refinement.__file__)]
    source_hashes={str(x):sha(x) for x in sources};runner_hash=sha(args.runner);build_hash=sha(args.build_manifest);build=cmp.strict_json(Path(args.build_manifest).read_text())
    if build['source_sha256']!=sha(sources[1]) or build['binary_sha256']!=runner_hash or not isinstance(build.get('command'),list) or not isinstance(build.get('compiler_version'),str):raise ValueError('runner provenance mismatch')
    coarse,target,family_file=family/'half/manifest.json',family/'full/manifest.json',family/'family.json';native=Path(args.native_report);frozen={}
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
        if sha(Path(__file__).with_name('compact_sgm_step_experiment.py'))!=recorded['source'] or sha(Path(refinement.__file__))!=recorded['refinement_source']:raise ValueError('source mismatch')
        frozen[mode]=(lo,hi,report,expected,s)
    ctx=cmp.load_context(coarse,target);nr=cmp.strict_json(native.read_text());diag.validate_report(cmp,nr,ctx,cmp.oracle.Settings(radius=2,subpixel=False))
    h,w=ctx['images'].shape[1:];planes=cmp._read_planes(nr['files'],cmp.PLANES,(2,h,w),args.dump_root);support=diag.verified_census_support(ctx['images'],ctx['support'],planes['census_validity.u32'],cmp.oracle)
    if np.any(ctx['images']>255):raise ValueError('invalid luma')
    base=ctx['images'].astype(np.uint8);geom=ctx['geometry'];outputs={};work={};saturation={}
    with tempfile.TemporaryDirectory(prefix='joint-sgm-') as temp:
        for exp in EXPOSURES:
            images=[base[0],transform(base[1],exp)];descs=[census(images[i],ctx['support'][i]) for i in (0,1)]
            if any(not np.array_equal(descs[i].support,support[i]) for i in (0,1)):raise ValueError('support mismatch')
            saturation[exp]={'zero':int((images[1]==0).sum()),'max':int((images[1]==255).sum()),'pixels':h*w}
            for interval,(lo,hi,_,_,selection) in frozen.items():
                for policy in ('baseline','joint'):
                    raw=[];valid=[];counts=[]
                    for eye in (0,1):
                        row=run_eye(args.runner,Path(temp)/str(eye),descs[eye],descs[1-eye],images[eye],images[1-eye],lo[eye],hi[eye],eye,128,policy)
                        disparity,_,mask=select_eye(row,lo[eye],hi[eye],128);raw.append(disparity);valid.append(mask);counts.append({k:row[k] for k in ('candidate_updates','interval_slots')})
                    raw,valid=np.stack(raw),np.stack(valid);lr=lr_consistency(raw,valid,1.)
                    with np.errstate(divide='ignore',invalid='ignore'):
                        denom=raw-np.float32(geom.cx_offset);z=np.float32(geom.fx*geom.baseline_m)/denom
                    metric=lr&np.asarray(ctx['support'],bool)&np.isfinite(z)&(z>0)&(denom>0)&(raw>0)
                    actual={'raw_validity.u32':valid,'raw_disparity.f32':raw,'validity.u32':lr,'metric_validity.u32':metric,'depth.f32':np.where(metric,z,np.nan).astype(np.float32)}
                    if exp=='identity' and policy=='baseline':
                        with np.load(selection,allow_pickle=False) as reference:
                            for key,value in actual.items():
                                if not np.array_equal(value,reference[key],equal_nan=True):raise ValueError('scaled identity baseline mismatch '+interval+' '+key)
                    key=f'{interval}-{exp}-{policy}';path=out/(key+'.npz')
                    with open(path,'xb') as fh:np.savez_compressed(fh,**actual)
                    outputs[key]=(path,sha(path));work[key]=counts
    if len(outputs)!=12:raise ValueError('incomplete pre-truth selection set')
    for path,digest in outputs.values():
        if sha(path)!=digest:raise ValueError('selection changed')
    if any(sha(path)!=digest for path,digest in source_hashes.items()) or sha(args.runner)!=runner_hash or sha(args.build_manifest)!=build_hash:raise ValueError('source changed')
    # Only open truth after all twelve selection files have been written and checked.
    truth=cmp.load_truth(family_file,ctx);metrics={};surface={};thin_ids={}
    for eye,name in enumerate(('left','right')):
        t=truth[name];eligible=np.asarray(t['valid'],bool)&np.asarray(t['visible'],bool)&~np.asarray(t['out_of_range'],bool)&~np.asarray(t['off_image'],bool)&np.asarray(t['remap_support'],bool)&support[eye]&~np.asarray(t['mixed_bilinear'],bool)&np.asarray(t['region_thin'],bool)
        thin_ids[name]={str(int(sid)):int((eligible & (t['surface_id']==sid)).sum()) for sid in np.unique(t['surface_id'][eligible])}
    for key,(path,digest) in outputs.items():
        if sha(path)!=digest:raise ValueError('selection changed during scoring')
        with np.load(path,allow_pickle=False) as a:actual={k:a[k].copy() for k in a.files}
        metrics[key]=cmp.evaluate_quality(actual,truth,support);surface[key]=surface_counts(actual,truth)
        for eye,name in enumerate(('left','right')):
            t=truth[name];domain=np.asarray(t['valid'],bool)&np.asarray(t['visible'],bool)&~np.asarray(t['out_of_range'],bool)&~np.asarray(t['off_image'],bool)&np.asarray(t['remap_support'],bool)&support[eye]&~np.asarray(t['mixed_bilinear'],bool)
            correct=np.asarray(actual['metric_validity.u32'][eye],bool)&(np.abs(actual['raw_disparity.f32'][eye].astype(np.float64)-t['disparity'])<=1)
            for region in ('all','near','edges','thin'):
                mask=np.ones(domain.shape,bool) if region=='all' else np.asarray(t['region_'+region],bool)
                metrics[key][name][region]['correct_metric_count']=int((domain&mask&correct).sum())
    gates={}
    for interval in frozen:
        for exp in EXPOSURES:
            a=f'{interval}-{exp}-joint';b=f'{interval}-{exp}-baseline';gates[a]={}
            for eye in ('left','right'):
                sa={sid:{'eligible':n,'false_accepted':surface[a][eye][sid]['false_accepted']} for sid,n in thin_ids[eye].items()}
                sb={sid:{'eligible':n,'false_accepted':surface[b][eye][sid]['false_accepted']} for sid,n in thin_ids[eye].items()}
                gates[a][eye]=evaluate_gates(metrics[a][eye],metrics[b][eye],sa,sb)
    result={'scope':'post-hoc synthetic development; fixed intervals under synthetic brightness perturbation; CPU only, not live or GPU quality','policy':{'p1':16,'p2':64,'costs':['2*Census','joint'], 'gates':GATES,'exposure':list(EXPOSURES)},'source_id':ctx['source_id'],'sources':source_hashes,'runner_sha256':runner_hash,'build_sha256':build_hash,'frozen_inputs':{k:list(v[3]) for k,v in frozen.items()},'selections':{k:v[1] for k,v in outputs.items()},'work':work,'saturation':saturation,'metrics':metrics,'surface_counts':surface,'thin_surface_eligible':thin_ids,'gates':gates}
    with open(out/'result.json','x') as fh:json.dump(result,fh,indent=2,sort_keys=True,allow_nan=False);fh.write('\n')

if __name__=='__main__':main()
