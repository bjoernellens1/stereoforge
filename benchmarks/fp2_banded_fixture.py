"""Matched frozen 720p CPU selections for packed FP2 bands; no truth is loaded."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from benchmarks.fp2_banded import run_image
from benchmarks.compact_sgm_population import ARTIFACTS,sha

def main():
    import sys
    sys.path.insert(0,str(Path(__file__).parent))
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    import stereoforge.refinement as refinement
    from stereoforge.refinement import census,lr_consistency
    from benchmarks.compact_sgm_quality import select_eye
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runner','build-manifest','artifact-root','family-root','native-report','dump-root','output-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(__file__).parent;out=Path(a.output_root);out.mkdir(parents=True,exist_ok=True);artifact=Path(a.artifact_root);family=Path(a.family_root)
    sources=[Path(__file__),root/'fp2_banded.py',root/'fp2_banded_runner.cpp',root/'compact_sgm_reference.py',root/'compact_sgm_population.py',root/'compact_sgm_quality.py',root/'compare_daxa_refinement.py',root/'fast60_prior_diagnostic.py',Path(refinement.__file__)]
    hashes={str(p):sha(p) for p in sources};runner_sha=sha(a.runner);build_sha=sha(a.build_manifest);build=cmp.strict_json(Path(a.build_manifest).read_text())
    if build['source_sha256']!=sha(root/'fp2_banded_runner.cpp') or build['binary_sha256']!=runner_sha:raise ValueError('build mismatch')
    coarse,target,family_file=family/'half/manifest.json',family/'full/manifest.json',family/'family.json';native=Path(a.native_report);modes={}
    for mode,expected in ARTIFACTS.items():
        f=artifact/(mode+'.npz');s=artifact/(mode+'.npz.selection.npz');r=artifact/(mode+'.json')
        if tuple(map(sha,(f,s,r)))!=expected:raise ValueError('frozen artifact mismatch')
        report=cmp.strict_json(r.read_text())
        if report['hashes']['frozen_intervals']!=sha(f) or report['hashes']['frozen_selection']!=sha(s):raise ValueError('report mismatch')
        with np.load(f,allow_pickle=False) as n:lo=n['lo'].copy();hi=n['hi'].copy();recorded=json.loads(str(n['input_hashes']))
        for key,path in (('coarse',coarse),('target',target),('family',family_file),('report',native)):
            if sha(path)!=recorded[key]:raise ValueError('input mismatch '+key)
        for name,digest in recorded['dependencies'].items():
            if sha(root/name)!=digest:raise ValueError('dependency mismatch '+name)
        if sha(root/'compact_sgm_step_experiment.py')!=recorded['source'] or sha(Path(refinement.__file__))!=recorded['refinement_source']:raise ValueError('source mismatch')
        modes[mode]=(lo,hi,s,expected)
    ctx=cmp.load_context(coarse,target);nr=cmp.strict_json(native.read_text());diag.validate_report(cmp,nr,ctx,cmp.oracle.Settings(radius=2,subpixel=False));h,w=ctx['images'].shape[1:]
    planes=cmp._read_planes(nr['files'],cmp.PLANES,(2,h,w),a.dump_root);support=diag.verified_census_support(ctx['images'],ctx['support'],planes['census_validity.u32'],cmp.oracle)
    desc=[census(ctx['images'][i],ctx['support'][i]) for i in (0,1)]
    if any(not np.array_equal(desc[i].support,support[i]) for i in (0,1)):raise ValueError('support mismatch')
    result={'scope':'frozen 720p synthetic CPU output parity, no truth, no GPU or performance claim','source_id':ctx['source_id'],'sources':hashes,'runner_sha256':runner_sha,'build_sha256':build_sha,'modes':{}};geom=ctx['geometry']
    with tempfile.TemporaryDirectory(prefix='fp2-banded-') as tmp:
        for mode,(lo,hi,selection,expected) in modes.items():
            raw=[];valid=[];stats=[]
            for eye in (0,1):
                row=run_image(a.runner,Path(tmp)/str(eye),desc[eye],desc[1-eye],lo[eye],hi[eye],eye,128)
                row['winners']=row.pop('winner');disparity,_,mask=select_eye(row,lo[eye],hi[eye],128)
                raw.append(disparity);valid.append(mask);stats.append(row['stats'])
            raw,valid=np.stack(raw),np.stack(valid);lr=lr_consistency(raw,valid,1.)
            with np.errstate(divide='ignore',invalid='ignore'):
                denom=raw-np.float32(geom.cx_offset);z=np.float32(geom.fx*geom.baseline_m)/denom
            metric=lr&np.asarray(ctx['support'],bool)&np.isfinite(z)&(z>0)&(denom>0)&(raw>0)
            actual={'raw_validity.u32':valid,'raw_disparity.f32':raw,'validity.u32':lr,'metric_validity.u32':metric,'depth.f32':np.where(metric,z,np.nan).astype(np.float32)}
            with np.load(selection,allow_pickle=False) as ref:
                for key,value in actual.items():
                    if not np.array_equal(value,ref[key],equal_nan=True):raise ValueError('selection mismatch '+mode+' '+key)
            path=out/(mode+'.npz')
            with open(path,'xb') as fh:np.savez_compressed(fh,**actual)
            result['modes'][mode]={'reference_hashes':list(expected),'output_sha256':sha(path),'stats_by_eye':stats}
    if any(sha(path)!=digest for path,digest in hashes.items()) or sha(a.runner)!=runner_sha or sha(a.build_manifest)!=build_sha:raise ValueError('source changed')
    with open(out/'result.json','x') as fh:json.dump(result,fh,indent=2,sort_keys=True,allow_nan=False);fh.write('\n')

if __name__=='__main__':main()
