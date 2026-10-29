"""Optional resident FP2 lane: parity before repeated-input timing."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from benchmarks.compact_sgm_population import ARTIFACTS
from benchmarks.compact_sgm_quality import select_eye
from benchmarks.fp2_720p_prepare import exact_file, sha
from benchmarks.fp2_720p_diagnostic import verify_prepared, selections

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (2, 720, 1280)


def verify_binary(binary):
    binary = Path(binary)
    info = json.loads(subprocess.check_output([str(binary), '--identity'], text=True))
    d = Path(info['shader_dir'])
    paths = {
        'host_sha256': ROOT/'apps/fp2_fused_fixture.cpp',
        'no_path_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_banded_fused.slang',
        'sum_reset_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_sum_reset.slang',
        'no_path_spirv_sha256': d/'fp2-fused.spv',
        'sum_reset_spirv_sha256': d/'fp2-resident-sum_reset.spv',
        'included_host_sha256': ROOT/'apps/fp2_prototype_fixture.cpp',
        'lut_sha256': ROOT/'apps/fp2_metric_lut.hpp',
        'reset_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_integrated_reset.slang',
        'reset_spirv_sha256': Path(info['reset_spirv_path']),
        'census_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_census.slang',
        'census_helper_sha256': ROOT/'python/stereoforge/shaders/ops/refinement.slang',
        'census_spirv_sha256': Path(info['census_spirv_path']),
        'cost_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_image_cost.slang',
        'cost_spirv_sha256': Path(info['cost_spirv_path']),
        'recurrence_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_banded.slang',
        'recurrence_spirv_sha256': Path(info['recurrence_spirv_path']),
        'selection_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_selection.slang',
        'workspace_spirv_sha256': d/'fp2-select-reset_workspace.spv',
        'select_spirv_sha256': d/'fp2-select-select_packed.spv',
        'lr_spirv_sha256': d/'fp2-select-consistency.spv',
        'metric_spirv_sha256': d/'fp2-select-metric.spv',
        'compiler_sha256': Path(info['compiler_path']),
    }
    for key, path in paths.items():
        if sha(path) != info[key]:
            raise ValueError('binary identity mismatch '+key)
    return {'binary_sha256': sha(binary), 'embedded': info}


def equal_bits(actual, expected, label):
    if not np.array_equal(actual, expected):
        raise ValueError('output mismatch '+label)


def require_nonempty(raw, lr, metric):
    for name, values in (('raw',raw),('LR',lr),('metric',metric)):
        if len(values)!=2 or any(int(value)<=0 for value in values):
            raise ValueError('empty '+name+' eye population')


def compare_outputs(out, prepared, artifact, source_id):
    """Compare GPU output words to CPU runner and fixed five-plane selections."""
    n = int(np.prod(SHAPE[1:]))
    raw = exact_file(Path(out)/'raw.bin', '<u4', 2*n*7).reshape(*SHAPE, 7)
    obs = exact_file(Path(out)/'obs.bin', '<u4', 2*n*9).reshape(*SHAPE, 9)
    path = Path(artifact)
    files = (path/'compact-fixed.npz', path/'compact-fixed.npz.selection.npz', path/'compact-fixed.json')
    if tuple(map(sha, files)) != ARTIFACTS['compact-fixed']:
        raise ValueError('pinned selection identity mismatch')
    with np.load(files[1], allow_pickle=False) as frozen:
        for eye in (0, 1):
            d = Path(prepared)/('eye'+str(eye)); shape = SHAPE[1:]
            data = {'winners': exact_file(d/'winner','<i4',n).reshape(shape)}
            data.update({name: exact_file(d/name,'<u4',n).reshape(shape) for name in ('best','second','count')})
            lo, hi = (exact_file(d/name,'<u4',n).reshape(shape) for name in ('lo','hi'))
            disparity, confidence, valid = select_eye(data, lo, hi, 128)
            equal_bits(raw[eye,...,0], data['winners'].view(np.uint32), 'winner')
            for col, name in ((1,'best'),(2,'second'),(3,'count')):
                equal_bits(raw[eye,...,col], data[name], name)
            equal_bits(raw[eye,...,4], disparity.view(np.uint32), 'raw disparity')
            equal_bits(raw[eye,...,5], confidence.view(np.uint32), 'confidence')
            equal_bits(raw[eye,...,6], valid.astype(np.uint32), 'raw valid')
        equal_bits(raw[...,4], np.asarray(frozen['raw_disparity.f32'],np.float32).view(np.uint32), 'pinned raw disparity')
        equal_bits(raw[...,6], np.asarray(frozen['raw_validity.u32'],np.uint32), 'pinned raw validity')
        lr = np.asarray(frozen['validity.u32'],np.uint32)
        metric = np.asarray(frozen['metric_validity.u32'],np.uint32)
        equal_bits(obs[...,0], lr, 'pinned LR validity')
        equal_bits(obs[...,3], metric, 'pinned metric validity')
        depth = np.asarray(frozen['depth.f32'],np.float32)
        finite = metric.astype(bool)
        equal_bits(obs[...,1][finite],depth.view(np.uint32)[finite], 'pinned finite depth')
        if np.any(obs[...,1][~finite] != np.uint32(0x7fc00000)) or np.any(~np.isnan(depth[~finite])):
            raise ValueError('invalid depth encoding mismatch')
        equal_bits(obs[...,2],np.where(finite,raw[...,5],0).astype(np.uint32),'metric confidence')
        equal_bits(obs[...,4],metric,'provenance')
        equal_bits(obs[...,5],np.where(finite,source_id & 0xffffffff,0xffffffff).astype(np.uint32),'source low')
        equal_bits(obs[...,6],np.where(finite,source_id >> 32,0xffffffff).astype(np.uint32),'source high')
        if np.any(obs[...,7:] != 0):
            raise ValueError('nonzero observation age')
    counts = {'raw_valid': [int(raw[e,...,6].sum()) for e in (0,1)],
              'lr_valid': [int(obs[e,...,0].sum()) for e in (0,1)],
              'metric_valid': [int(obs[e,...,3].sum()) for e in (0,1)]}
    require_nonempty(counts['raw_valid'],counts['lr_valid'],counts['metric_valid'])
    return {'raw_sha256': sha(Path(out)/'raw.bin'), 'obs_sha256': sha(Path(out)/'obs.bin'), **counts}


def verify_tiny(binary):
    binary=Path(binary)
    info=json.loads(subprocess.check_output([str(binary),'--identity'],text=True))
    directory=Path(info['shader_dir'])
    paths={'host_sha256':ROOT/'apps/fp2_fused_tiny.cpp',
           'no_path_shader_sha256':ROOT/'python/stereoforge/shaders/daxa/fp2_banded_fused.slang',
           'no_path_spirv_sha256':directory/'fp2-fused.spv',
           'sum_reset_shader_sha256':ROOT/'python/stereoforge/shaders/daxa/fp2_sum_reset.slang',
           'reset_spirv_sha256':directory/'fp2-resident-sum_reset.spv',
           'selection_shader_sha256':ROOT/'python/stereoforge/shaders/daxa/fp2_selection.slang',
           'select_spirv_sha256':directory/'fp2-select-select_packed.spv',
           'lr_spirv_sha256':directory/'fp2-select-consistency.spv',
           'metric_spirv_sha256':directory/'fp2-select-metric.spv',
           'lut_host_sha256':ROOT/'apps/fp2_metric_lut.hpp',
           'included_host_sha256':ROOT/'apps/fp2_prototype_fixture.cpp',
           'compiler_sha256':Path(info['compiler_path'])}
    for key,path in paths.items():
        if sha(path)!=info[key]:raise ValueError('tiny binary identity mismatch '+key)
    return {'binary_sha256':sha(binary),'embedded':info}


def check_proof(path, expected_hash, manifest_sha, host, tiny, harness_sha, baseline_sha, baseline):
    if not path or not expected_hash or sha(path)!=expected_hash:
        raise ValueError('parity proof missing or changed')
    proof=json.loads(Path(path).read_text())
    if (proof.get('scope')!='fused parity' or proof.get('manifest_sha256')!=manifest_sha or
            proof.get('host')!=host or proof.get('tiny')!=tiny or
            proof.get('harness_sha256')!=harness_sha or proof.get('baseline_sha256')!=baseline_sha or
            'contracts passed' not in proof.get('tiny_output','')):
        raise ValueError('parity proof identity mismatch')
    for key in ('raw_sha256','obs_sha256'):
        if proof['gpu_outputs'][key]!=baseline['gpu_outputs'][key]:
            raise ValueError('parity proof output mismatch')
    return proof


def cost_work_for_eye(lo,hi,own,other,eye,disparities=128):
    lo=np.asarray(lo);hi=np.asarray(hi);own=np.asarray(own);other=np.asarray(other)
    if lo.ndim!=2 or hi.shape!=lo.shape or own.shape!=lo.shape or other.shape!=lo.shape or eye not in (0,1) or disparities<=0 or np.any(lo<0) or np.any(hi<lo) or np.any(hi>disparities) or np.any(own<0) or np.any(other<0) or np.any(own>1) or np.any(other>1):raise ValueError('invalid work inputs')
    height,width=lo.shape;x=np.arange(width)[None,:];active=bounds=owned=valid=0
    for d in range(disparities):
        band=(lo<=d)&(d<hi);active+=int(band.sum());target=x-d if eye==0 else x+d
        inbounds=(target>=0)&(target<width);bounds+=int((band&inbounds).sum())
        usable=band&inbounds&(own!=0);owned+=int(usable.sum())
        mapped=np.clip(target,0,width-1);valid+=int((usable&(other[:,mapped[0]]!=0)).sum())
    return {'active_visits_four_paths':4*active,'in_bounds_four_paths':4*bounds,
            'own_supported_four_paths':4*owned,'descriptor_pairs_popcount_four_paths':4*valid}


def requested_cost_work(prepared):
    """Count predicate-qualified candidate visits from pinned supports, not GPU traffic."""
    result=[];width,height=1280,720;n=width*height
    for eye in range(2):
        root=Path(prepared)/('eye'+str(eye))
        planes=[exact_file(root/name,'<u4',n).reshape(height,width) for name in ('lo','hi','own_support','other_support')]
        result.append(cost_work_for_eye(*planes,eye))
    return result


def main():
    import csv
    import sys
    sys.path.insert(0,str(ROOT/'benchmarks'))
    import compare_daxa_refinement as cmp
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','manifest-sha256','host','tiny','artifact-root','family-root','output-root','baseline-result'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--parity',action='store_true');p.add_argument('--time',action='store_true')
    p.add_argument('--proof');p.add_argument('--proof-sha256')
    a=p.parse_args();prepared=Path(a.prepared);output=Path(a.output_root)
    if a.parity and a.time:raise ValueError('choose one execution mode')
    if output.exists():raise ValueError('output root already exists')
    manifest=verify_prepared(prepared,a.manifest_sha256);binary=verify_binary(a.host);tiny=verify_tiny(a.tiny);source_hash=sha(Path(__file__))
    if sha(a.baseline_result)!='e690b1895b63cf2c9c158419bf0662e7e316426d1d3c164f8c078d3700090041':raise ValueError('baseline result mismatch')
    baseline=json.loads(Path(a.baseline_result).read_text())
    if baseline['manifest_sha256']!=a.manifest_sha256:raise ValueError('baseline input mismatch')
    family=Path(a.family_root);pinned={Path(name).resolve():digest for name,digest in manifest['sources'].items()}
    for path in (family/'half/manifest.json',family/'full/manifest.json',family/'family.json'):
        if path.resolve() not in pinned or sha(path)!=pinned[path.resolve()]:raise ValueError('calibration family mismatch')
    ctx=cmp.load_context(family/'half/manifest.json',family/'full/manifest.json')
    if ctx['source_id']!=manifest['source_id']:raise ValueError('source identity mismatch')
    selections(prepared,a.artifact_root,ctx)
    geom=ctx['geometry'];numerator=np.float32(geom.fx*geom.baseline_m);cx=np.float32(geom.cx_offset)
    work=requested_cost_work(prepared)
    num_bits=int(numerator.view(np.uint32));cx_bits=int(cx.view(np.uint32));source_id=int(ctx['source_id'])
    if not 0<=source_id<2**63:raise ValueError('source ID outside signed domain')
    if a.time:
        check_proof(a.proof,a.proof_sha256,a.manifest_sha256,binary,tiny,source_hash,sha(a.baseline_result),baseline)
    output.mkdir(parents=True,exist_ok=False)
    tiny_log=''
    if a.parity:
        tiny_log=subprocess.check_output([str(Path(a.tiny).resolve()),'--gpu'],text=True,timeout=120)
        if 'contracts passed' not in tiny_log:raise ValueError('tiny parity failed')
    mode='--gpu' if a.parity else '--time' if a.time else '--check'
    subprocess.run([str(Path(a.host).resolve()),mode,str(prepared.resolve()),str(output.resolve()),str(num_bits),str(cx_bits),str(source_id)],check=True,timeout=600)
    result={'scope':'fused parity' if a.parity else 'fused timing' if a.time else 'CPU preflight only',
            'source_id':source_id,'numerator_bits':num_bits,'offset_bits':cx_bits,'manifest_sha256':a.manifest_sha256,
            'host':binary,'tiny':tiny,'tiny_output':tiny_log,'requested_cost_work':work,'harness_sha256':source_hash,'baseline_sha256':sha(a.baseline_result)}
    if a.parity:
        result['gpu_outputs']=compare_outputs(output,prepared,a.artifact_root,source_id)
        for key in ('raw_sha256','obs_sha256'):
            if result['gpu_outputs'][key]!=baseline['gpu_outputs'][key]:raise ValueError('baseline output mismatch '+key)
    if a.time:
        with (output/'samples.csv').open() as stream:rows=list(csv.DictReader(stream))
        if len(rows)!=100 or any(set(row)!={'gpu_ms','record_ms','submit_ms','wait_ms'} for row in rows):raise ValueError('invalid sample shape')
        values=np.array([[float(row[k]) for k in ('gpu_ms','record_ms','submit_ms','wait_ms')] for row in rows],dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values<0):raise ValueError('invalid sample')
        result['samples_sha256']=sha(output/'samples.csv')
        result['sample_count']=len(rows);result['warmup']=10;result['quantile_policy']='NumPy linear interpolation';result['input_scope']='one repeated preuploaded analytic frame; one in-flight slot'
        result['summary_ms']={name:{'p50':float(np.quantile(values[:,i],.5)),'p95':float(np.quantile(values[:,i],.95)),'p99':float(np.quantile(values[:,i],.99)),'max':float(np.max(values[:,i]))} for i,name in enumerate(('gpu','record','submit','wait'))}
    if verify_prepared(prepared,a.manifest_sha256)!=manifest or verify_binary(a.host)!=binary or verify_tiny(a.tiny)!=tiny or sha(Path(__file__))!=source_hash:raise ValueError('identity changed during run')
    with (output/'result.json').open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
