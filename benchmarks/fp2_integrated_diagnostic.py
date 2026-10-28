"""Isolated 720p integrated FP2 parity; no live or performance claim."""
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
        'host_sha256': ROOT/'apps/fp2_integrated_fixture.cpp',
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


def main():
    import sys
    sys.path.insert(0,str(ROOT/'benchmarks'))
    import compare_daxa_refinement as cmp
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','manifest-sha256','host','artifact-root','family-root','output-root'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--gpu',action='store_true')
    a=p.parse_args();prepared=Path(a.prepared);output=Path(a.output_root)
    if output.exists():
        raise ValueError('output root already exists')
    manifest=verify_prepared(prepared,a.manifest_sha256);binary=verify_binary(a.host);source_hash=sha(Path(__file__))
    family=Path(a.family_root)
    pinned_sources={Path(name).resolve():digest for name,digest in manifest['sources'].items()}
    for path in (family/'half/manifest.json',family/'full/manifest.json',family/'family.json'):
        if path.resolve() not in pinned_sources or sha(path)!=pinned_sources[path.resolve()]:
            raise ValueError('calibration family identity mismatch')
    ctx=cmp.load_context(family/'half/manifest.json',family/'full/manifest.json')
    if ctx['source_id'] != manifest['source_id']:
        raise ValueError('source identity mismatch')
    selections(prepared,a.artifact_root,ctx)
    geometry=ctx['geometry']; numerator=np.float32(geometry.fx*geometry.baseline_m);cx=np.float32(geometry.cx_offset)
    num_bits=int(numerator.view(np.uint32));cx_bits=int(cx.view(np.uint32));source_id=int(ctx['source_id'])
    if not 0<=source_id<2**63:
        raise ValueError('source ID outside signed 63-bit domain')
    output.mkdir(parents=True,exist_ok=False)
    mode='--gpu' if a.gpu else '--check'
    command=[str(Path(a.host).resolve()),mode,str(prepared.resolve()),str(output.resolve()),str(num_bits),str(cx_bits),str(source_id)]
    run=subprocess.run(command,check=True,timeout=600)
    del run
    result={'scope':'720p analytic GPU integrated parity' if a.gpu else 'CPU preflight only',
            'source_id':source_id,'numerator_bits':num_bits,'offset_bits':cx_bits,'manifest_sha256':a.manifest_sha256,
            'host':binary,'harness_sha256':source_hash}
    if a.gpu:
        result['gpu_outputs']=compare_outputs(output,prepared,a.artifact_root,source_id)
    if verify_prepared(prepared,a.manifest_sha256)!=manifest or verify_binary(a.host)!=binary or sha(Path(__file__))!=source_hash:
        raise ValueError('identity changed during run')
    with (output/'result.json').open('x') as file:
        json.dump(result,file,sort_keys=True,indent=2,allow_nan=False);file.write('\n')


if __name__=='__main__':
    main()
