"""Frozen Census-anchor affine correction experiment; CPU-only development."""
import hashlib,json,tempfile
from pathlib import Path
import numpy as np
from benchmarks.compact_sgm_affine import PARAMETERS,apply_affine,estimate_affine,run_masked,sample_anchors,fallback_selection
from benchmarks.compact_sgm_population import ARTIFACTS,sha
from benchmarks.compact_sgm_joint import evaluate_gates,GATES
from benchmarks.compact_sgm_local_evidence import transform,EXPOSURES

PRIOR_RESULT='3cd89a0e5236e41702316a88458465deb3cf3bd69b3f586259444a8ec8f5e981'

def main():
    import argparse,sys
    sys.path.insert(0,str(Path(__file__).parent))
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    import stereoforge.refinement as refinement
    from stereoforge.refinement import census,lr_consistency
    from benchmarks.compact_sgm_quality import select_eye
    from benchmarks.compact_sgm_step_experiment import surface_counts
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runner','build-manifest','artifact-root','previous-root','family-root','native-report','dump-root','output-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output_root);out.mkdir(parents=True,exist_ok=True);previous=Path(a.previous_root);prior=previous/'result.json'
    if sha(prior)!=PRIOR_RESULT:raise ValueError('prior report hash mismatch')
    prev=cmp.strict_json(prior.read_text())
    for key,digest in prev['selections'].items():
        if sha(previous/(key+'.npz'))!=digest:raise ValueError('prior selection mismatch: '+key)
    # Pin the previous inputs and sources, not only the JSON report.
    if prev['frozen_inputs']!={k:list(v) for k,v in ARTIFACTS.items()}:raise ValueError('unexpected prior artifacts')
    for path,digest in prev['sources'].items():
        if sha(path)!=digest:raise ValueError('prior source mismatch: '+path)
    root=Path(__file__).parent;sources=[Path(__file__),root/'compact_sgm_affine.py',root/'compact_sgm_affine_runner.cpp',root/'compact_sgm_joint.py',root/'compact_sgm_local_evidence.py',root/'compact_sgm_population.py',root/'compact_sgm_quality.py',root/'compact_sgm_step_experiment.py',root/'compare_daxa_refinement.py',root/'fast60_prior_diagnostic.py',Path(refinement.__file__)]
    source_hashes={str(s):sha(s) for s in sources};build=cmp.strict_json(Path(a.build_manifest).read_text());runner_sha=sha(a.runner);build_sha=sha(a.build_manifest)
    if build['source_sha256']!=sha(root/'compact_sgm_affine_runner.cpp') or build['binary_sha256']!=runner_sha or not isinstance(build.get('command'),list) or not isinstance(build.get('compiler_version'),str):raise ValueError('build mismatch')
    family=Path(a.family_root);coarse,target,family_file=family/'half/manifest.json',family/'full/manifest.json',family/'family.json';native=Path(a.native_report);artifact=Path(a.artifact_root);frozen={}
    for mode,expected in ARTIFACTS.items():
        f=artifact/(mode+'.npz');s=artifact/(mode+'.npz.selection.npz');r=artifact/(mode+'.json')
        if tuple(map(sha,(f,s,r)))!=expected:raise ValueError('frozen interval mismatch')
        with np.load(f,allow_pickle=False) as n:lo=n['lo'].copy();hi=n['hi'].copy();recorded=json.loads(str(n['input_hashes']))
        for k,pth in [('coarse',coarse),('target',target),('family',family_file),('report',native)]:
            if sha(pth)!=recorded[k]:raise ValueError('fixture mismatch '+k)
        frozen[mode]=(lo,hi)
    ctx=cmp.load_context(coarse,target);nr=cmp.strict_json(native.read_text());diag.validate_report(cmp,nr,ctx,cmp.oracle.Settings(radius=2,subpixel=False));h,w=ctx['images'].shape[1:]
    planes=cmp._read_planes(nr['files'],cmp.PLANES,(2,h,w),a.dump_root);support=diag.verified_census_support(ctx['images'],ctx['support'],planes['census_validity.u32'],cmp.oracle)
    if np.any(ctx['images']>255):raise ValueError('bad luma')
    base=ctx['images'].astype(np.uint8);outputs={};fits={};work={};geom=ctx['geometry']
    with tempfile.TemporaryDirectory(prefix='affine-sgm-') as temp:
        for exp in EXPOSURES:
            images=[base[0],transform(base[1],exp)];descs=[census(images[i],ctx['support'][i]) for i in (0,1)]
            if any(not np.array_equal(descs[i].support,support[i]) for i in (0,1)):raise ValueError('Census support mismatch')
            for interval,(lo,hi) in frozen.items():
                key=f'{interval}-{exp}';bp=previous/(key+'-baseline.npz');jp=previous/(key+'-joint.npz')
                with np.load(bp,allow_pickle=False) as n:baseline={name:n[name].copy() for name in n.files}
                anchors=sample_anchors(images[0],images[1],descs[0].support,descs[1].support,baseline['raw_disparity.f32'][0],baseline['validity.u32'][0]);fit=estimate_affine(anchors)
                ah=hashlib.sha256()
                for name in ('train_x','train_y','train_left','train_right','validation_x','validation_y','validation_left','validation_right'):ah.update(anchors[name].tobytes())
                fit['anchors_sha256']=ah.hexdigest();fit['input_left_sha256']=hashlib.sha256(images[0].tobytes()).hexdigest();fit['input_right_sha256']=hashlib.sha256(images[1].tobytes()).hexdigest();fit['tile_policy']=PARAMETERS
                if fit['accepted']:
                    corrected,right_safe=apply_affine(images[1],fit['gain'],fit['offset']);left_safe=(images[0]>0)&(images[0]<255);corrected_images=[images[0],corrected];safes=[left_safe,right_safe]
                    fit['corrected_right_sha256']=hashlib.sha256(corrected.tobytes()).hexdigest();fit['right_safe_count']=int(right_safe.sum());raw=[];valid=[];counts=[]
                    for eye in (0,1):
                        row=run_masked(a.runner,Path(temp)/str(eye),descs[eye],descs[1-eye],corrected_images[eye],corrected_images[1-eye],safes[eye],safes[1-eye],lo[eye],hi[eye],eye,128)
                        disparity,_,mask=select_eye(row,lo[eye],hi[eye],128);raw.append(disparity);valid.append(mask);counts.append({k:row[k] for k in ('candidate_updates','interval_slots')})
                    raw,valid=np.stack(raw),np.stack(valid);lr=lr_consistency(raw,valid,1.)
                    with np.errstate(divide='ignore',invalid='ignore'):
                        denom=raw-np.float32(geom.cx_offset);z=np.float32(geom.fx*geom.baseline_m)/denom
                    metric=lr&np.asarray(ctx['support'],bool)&np.isfinite(z)&(z>0)&(denom>0)&(raw>0)
                    actual={'raw_validity.u32':valid,'raw_disparity.f32':raw,'validity.u32':lr,'metric_validity.u32':metric,'depth.f32':np.where(metric,z,np.nan).astype(np.float32)};work[key]=counts
                else:actual=fallback_selection(fit,baseline);work[key]='exact frozen Census fallback'
                # Bind pre-existing uncorrected branch to the pinned report explicitly.
                if sha(bp)!=prev['selections'][key+'-baseline'] or sha(jp)!=prev['selections'][key+'-joint']:raise ValueError('prior changed')
                fit_path=out/(key+'.fit.json')
                with open(fit_path,'x') as fh:json.dump(fit,fh,sort_keys=True,indent=2,allow_nan=False);fh.write('\n')
                path=out/(key+'.npz')
                with open(path,'xb') as fh:np.savez_compressed(fh,**actual)
                outputs[key]=(path,sha(path),fit_path,sha(fit_path));fits[key]=fit
    if len(outputs)!=6:raise ValueError('incomplete pretruth outputs')
    for path,digest,fp,fd in outputs.values():
        if sha(path)!=digest or sha(fp)!=fd:raise ValueError('outputs changed')
    if sha(prior)!=PRIOR_RESULT or any(sha(path)!=digest for path,digest in source_hashes.items()) or sha(a.runner)!=runner_sha or sha(a.build_manifest)!=build_sha:raise ValueError('source changed')
    # Both branches of every prior configuration were frozen in the prior experiment;
    # all six fit records and corrected/fallback selections are now frozen before truth.
    truth=cmp.load_truth(family_file,ctx);quality={};surface={};thin_ids={}
    for eye,name in enumerate(('left','right')):
        t=truth[name];eligible=np.asarray(t['valid'],bool)&np.asarray(t['visible'],bool)&~np.asarray(t['out_of_range'],bool)&~np.asarray(t['off_image'],bool)&np.asarray(t['remap_support'],bool)&support[eye]&~np.asarray(t['mixed_bilinear'],bool)&np.asarray(t['region_thin'],bool)
        thin_ids[name]={str(int(sid)):int((eligible & (t['surface_id']==sid)).sum()) for sid in np.unique(t['surface_id'][eligible])}
    for key,(path,digest,fp,fd) in outputs.items():
        if sha(path)!=digest or sha(fp)!=fd:raise ValueError('output changed')
        with np.load(path,allow_pickle=False) as n:actual={k:n[k].copy() for k in n.files}
        quality[key]=cmp.evaluate_quality(actual,truth,support);surface[key]=surface_counts(actual,truth)
        for eye,name in enumerate(('left','right')):
            t=truth[name];domain=np.asarray(t['valid'],bool)&np.asarray(t['visible'],bool)&~np.asarray(t['out_of_range'],bool)&~np.asarray(t['off_image'],bool)&np.asarray(t['remap_support'],bool)&support[eye]&~np.asarray(t['mixed_bilinear'],bool)
            correct=np.asarray(actual['metric_validity.u32'][eye],bool)&(np.abs(actual['raw_disparity.f32'][eye].astype(np.float64)-t['disparity'])<=1)
            for region in ('all','near','edges','thin'):
                mask=np.ones(domain.shape,bool) if region=='all' else np.asarray(t['region_'+region],bool)
                quality[key][name][region]['correct_metric_count']=int((domain&mask&correct).sum())
    gates={}
    for key in outputs:
        baseline=key+'-baseline';gates[key]={}
        for eye in ('left','right'):
            sa={sid:{'eligible':n,'false_accepted':surface[key][eye][sid]['false_accepted']} for sid,n in thin_ids[eye].items()}
            sb={sid:{'eligible':n,'false_accepted':prev['surface_counts'][baseline][eye][sid]['false_accepted']} for sid,n in thin_ids[eye].items()}
            gates[key][eye]=evaluate_gates(quality[key][eye],prev['metrics'][baseline][eye],sa,sb)
    result={'scope':'post-hoc synthetic development; Census-derived anchors, same-frame correlated data; no GPU/live/independent calibration claim','parameters':PARAMETERS,'gates_policy':GATES,'source_id':ctx['source_id'],'sources':source_hashes,'runner_sha256':runner_sha,'build_sha256':build_sha,'prior_sha256':PRIOR_RESULT,'outputs':{k:{'selection':v[1],'fit':v[3]} for k,v in outputs.items()},'fits':fits,'work':work,'quality':quality,'surface_counts':surface,'thin_surface_eligible':thin_ids,'gates':gates}
    with open(out/'result.json','x') as fh:json.dump(result,fh,indent=2,sort_keys=True,allow_nan=False);fh.write('\n')

if __name__=='__main__':main()
