"""Identity-checked 720p compact GPU parity diagnostic, not a performance test."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from benchmarks.compact_sgm_population import ARTIFACTS
from benchmarks.compact_sgm_quality import select_eye
from benchmarks.fp2_720p_prepare import exact_file, sha
from stereoforge.refinement import lr_consistency

ROOT = Path(__file__).resolve().parents[1]
FILES = {'lo', 'hi', 'own_luma', 'other_luma', 'own_rect_support', 'other_rect_support',
         'own', 'other', 'own_support', 'other_support', 'offset', 'winner', 'best',
         'second', 'count', 'paths', 'sums', 'stats.json'}


def verify_prepared(root, expected_sha):
    root = Path(root)
    manifest_path = root/'manifest.json'
    if sha(manifest_path) != expected_sha:
        raise ValueError('prepared manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    if set(manifest['eyes']) != {'0', '1'}:
        raise ValueError('eye manifest mismatch')
    for name, digest in manifest['sources'].items():
        if sha(name) != digest:
            raise ValueError('source mismatch ' + name)
    for eye in ('0', '1'):
        files = manifest['eyes'][eye]['files']
        if set(files) != FILES:
            raise ValueError('file manifest mismatch')
        for name, info in files.items():
            path = root/('eye'+eye)/name
            if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
                raise ValueError('prepared file mismatch ' + str(path))
    return manifest


def verify_binary(binary):
    binary = Path(binary)
    info = json.loads(subprocess.check_output([str(binary), '--identity'], text=True))
    paths = {
        'host_sha256': ROOT/'apps/fp2_720p_fixture.cpp',
        'included_host_sha256': ROOT/'apps/fp2_prototype_fixture.cpp',
        'census_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_census.slang',
        'census_helper_sha256': ROOT/'python/stereoforge/shaders/ops/refinement.slang',
        'census_spirv_sha256': Path(info['census_spirv_path']),
        'cost_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_image_cost.slang',
        'cost_spirv_sha256': Path(info['cost_spirv_path']),
        'recurrence_shader_sha256': ROOT/'python/stereoforge/shaders/daxa/fp2_banded.slang',
        'recurrence_spirv_sha256': Path(info['recurrence_spirv_path']),
        'compiler_sha256': Path(info['compiler_path']),
    }
    for key, path in paths.items():
        if sha(path) != info[key]:
            raise ValueError('binary identity mismatch ' + key)
    return {'binary_sha256': sha(binary), 'embedded': info}


def selections(prepared, artifact, context):
    artifact = Path(artifact)
    paths = (artifact/'compact-fixed.npz', artifact/'compact-fixed.npz.selection.npz', artifact/'compact-fixed.json')
    if tuple(map(sha, paths)) != ARTIFACTS['compact-fixed']:
        raise ValueError('pinned selection mismatch')
    with np.load(paths[0], allow_pickle=False) as ref:
        frozen_lo, frozen_hi = ref['lo'], ref['hi']
    raw, valid = [], []
    for eye in (0, 1):
        d = Path(prepared)/('eye'+str(eye)); shape = (720, 1280); n = 720*1280
        lo, hi = (exact_file(d/name, '<u4', n).reshape(shape) for name in ('lo', 'hi'))
        if not np.array_equal(lo, frozen_lo[eye]) or not np.array_equal(hi, frozen_hi[eye]):
            raise ValueError('prepared interval mismatch')
        result = {'winners': exact_file(d/'winner', '<i4', n).reshape(shape)}
        result.update({name: exact_file(d/name, '<u4', n).reshape(shape) for name in ('best', 'second', 'count')})
        disparity, _, mask = select_eye(result, lo, hi, 128)
        raw.append(disparity); valid.append(mask)
    raw, valid = np.stack(raw), np.stack(valid)
    lr = lr_consistency(raw, valid, 1.)
    geom = context['geometry']
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = raw-np.float32(geom.cx_offset)
        z = np.float32(geom.fx*geom.baseline_m)/denom
    metric = lr & np.asarray(context['support'], bool) & np.isfinite(z) & (z > 0) & (denom > 0) & (raw > 0)
    actual = {'raw_validity.u32': valid, 'raw_disparity.f32': raw, 'validity.u32': lr,
              'metric_validity.u32': metric, 'depth.f32': np.where(metric, z, np.nan).astype(np.float32)}
    with np.load(paths[1], allow_pickle=False) as ref:
        for key, value in actual.items():
            if not np.array_equal(value, ref[key], equal_nan=True):
                raise ValueError('selection mismatch ' + key)
    return sorted(actual)


def invoke_host(binary, root, mode, eye):
    try:
        result = subprocess.run([str(Path(binary).resolve()), mode, str((Path(root)/('eye'+str(eye))).resolve()), str(eye), '1280', '720'], check=True, timeout=600, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        sys.stdout.write(error.stdout or '')
        sys.stderr.write(error.stderr or '')
        raise
    sys.stderr.write(result.stderr)
    return result.stdout


def main():
    import sys
    sys.path.insert(0, str(ROOT/'benchmarks'))
    import compare_daxa_refinement as cmp
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'manifest-sha256', 'host', 'artifact-root', 'family-root', 'result'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--gpu', action='store_true')
    a = p.parse_args(); root = Path(a.prepared); result = Path(a.result)
    if result.exists():
        raise ValueError('result already exists')
    source_identity = sha(Path(__file__))
    manifest = verify_prepared(root, a.manifest_sha256)
    binary = verify_binary(a.host)
    family = Path(a.family_root)
    context = cmp.load_context(family/'half/manifest.json', family/'full/manifest.json')
    if context['source_id'] != manifest['source_id']:
        raise ValueError('source identity mismatch')
    five = selections(root, a.artifact_root, context)
    logs = []
    for eye in (0, 1):
        logs.append(invoke_host(a.host, root, '--gpu' if a.gpu else '--check', eye))
    if verify_prepared(root, a.manifest_sha256) != manifest or verify_binary(a.host) != binary or sha(Path(__file__)) != source_identity:
        raise ValueError('identity changed during run')
    with result.open('x') as stream:
        json.dump({'scope': '720p analytic compact GPU parity' if a.gpu else 'CPU preflight only',
                   'manifest_sha256': a.manifest_sha256, 'host': binary, 'source_sha256': source_identity,
                   'five_planes_cpu_selection': five, 'gpu_host_output': logs}, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    main()
