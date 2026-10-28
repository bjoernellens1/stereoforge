"""CPU-only, fail-closed preparation for the compact FP2 720p parity experiment.

This tool does not execute Vulkan or load truth. It freezes per-eye inputs and
packed CPU oracle files; it is not a live-camera or performance benchmark.
"""
import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
from pathlib import Path

import numpy as np
from benchmarks.compact_sgm_population import ARTIFACTS
from benchmarks.fp2_banded import buckets

CHUNK_BYTES = 8 * 1024 * 1024
MAX_ORACLE_BYTES = 2_000_000_000
MAX_CHILD_ADDRESS_BYTES = 2_000_000_000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stream_equal(left, right, chunk_bytes=CHUNK_BYTES):
    if chunk_bytes <= 0 or chunk_bytes > CHUNK_BYTES:
        raise ValueError('invalid chunk bound')
    left, right = Path(left), Path(right)
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(chunk_bytes), b.read(chunk_bytes)
            if x != y:
                return False
            if not x:
                return True


def checked_offsets(lo, hi, shape):
    lo, hi = np.asarray(lo), np.asarray(hi)
    if lo.shape != shape or hi.shape != shape or lo.dtype.kind not in 'iu' or hi.dtype.kind not in 'iu':
        raise ValueError('invalid interval shape or type')
    if np.any(lo < 0) or np.any(lo > hi) or np.any(hi > 128):
        raise ValueError('invalid interval bounds')
    capacities = buckets(hi - lo)
    offset = np.empty(lo.size + 1, dtype='<u8')
    offset[0] = 0
    np.cumsum(capacities.ravel(), dtype=np.uint64, out=offset[1:])
    if int(offset[-1]) > shape[0] * shape[1] * 128:
        raise ValueError('invalid allocation extent')
    return offset


def exact_file(path, dtype, count):
    path = Path(path)
    expected = np.dtype(dtype).itemsize * count
    if path.stat().st_size != expected:
        raise ValueError('incorrect file extent: ' + str(path))
    return np.fromfile(path, dtype=dtype, count=count)


def write_plane(path, a):
    with Path(path).open('xb') as stream:
        np.asarray(a, dtype='<u4').tofile(stream)


def prepare(args):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    import stereoforge.refinement as refinement
    from stereoforge.refinement import census

    root = Path(__file__).parent
    artifact, family = Path(args.artifact_root), Path(args.family_root)
    frozen, selection, report = (artifact / 'compact-fixed.npz', artifact / 'compact-fixed.npz.selection.npz', artifact / 'compact-fixed.json')
    expected = ARTIFACTS['compact-fixed']
    if tuple(map(sha, (frozen, selection, report))) != expected:
        raise ValueError('frozen artifact mismatch')
    recorded_report = cmp.strict_json(report.read_text())
    if recorded_report['hashes']['frozen_intervals'] != sha(frozen) or recorded_report['hashes']['frozen_selection'] != sha(selection):
        raise ValueError('report mismatch')
    with np.load(frozen, allow_pickle=False) as a:
        lo, hi = a['lo'].copy(), a['hi'].copy()
        recorded = json.loads(str(a['input_hashes']))
    coarse, target, family_file = family / 'half/manifest.json', family / 'full/manifest.json', family / 'family.json'
    native = Path(args.native_report)
    tracked = [frozen, selection, report, coarse, target, family_file, native, Path(args.runner), Path(args.build_manifest), Path(__file__), root/'fp2_banded.py', root/'fp2_banded_runner.cpp', root/'compact_sgm_reference.py', root/'compact_sgm_population.py', root/'compact_sgm_quality.py', root/'compare_daxa_refinement.py', root/'fast60_prior_diagnostic.py', Path(refinement.__file__)]
    for key, path in (('coarse', coarse), ('target', target), ('family', family_file), ('report', native)):
        if sha(path) != recorded[key]:
            raise ValueError('input mismatch ' + key)
    for name, digest in recorded['dependencies'].items():
        path = root / name
        if sha(path) != digest:
            raise ValueError('dependency mismatch ' + name)
        tracked.append(path)
    source = root / 'compact_sgm_step_experiment.py'
    if sha(source) != recorded['source'] or sha(Path(refinement.__file__)) != recorded['refinement_source']:
        raise ValueError('source mismatch')
    tracked.append(source)
    build = cmp.strict_json(Path(args.build_manifest).read_text())
    if build['source_sha256'] != sha(root/'fp2_banded_runner.cpp') or build['binary_sha256'] != sha(args.runner):
        raise ValueError('CPU runner identity mismatch')
    before = {str(path): sha(path) for path in tracked}
    ctx = cmp.load_context(coarse, target)
    if ctx['images'].shape != (2, 720, 1280) or lo.shape != (2, 720, 1280):
        raise ValueError('expected two 720p eyes')
    nr = cmp.strict_json(native.read_text())
    diag.validate_report(cmp, nr, ctx, cmp.oracle.Settings(radius=2, subpixel=False))
    planes = cmp._read_planes(nr['files'], cmp.PLANES, (2, 720, 1280), args.dump_root)
    support = diag.verified_census_support(ctx['images'], ctx['support'], planes['census_validity.u32'], cmp.oracle)
    descriptors = [census(ctx['images'][i], ctx['support'][i]) for i in (0, 1)]
    if any(not np.array_equal(descriptors[i].support, support[i]) for i in (0, 1)):
        raise ValueError('Census support mismatch')
    if any(np.count_nonzero(d.support) == 0 for d in descriptors):
        raise ValueError('empty supported descriptors')
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=False)
    result = {'scope': 'CPU preparation, analytic 720p compact intervals, no truth or GPU', 'source_id': ctx['source_id'], 'sources': before, 'eyes': {}}
    for eye in (0, 1):
        path = out / ('eye' + str(eye)); path.mkdir()
        offsets = checked_offsets(lo[eye], hi[eye], (720, 1280))
        total = int(offsets[-1])
        oracle_bytes = total * 20
        if oracle_bytes > MAX_ORACLE_BYTES or shutil.disk_usage(path).free < oracle_bytes + 512_000_000:
            raise ValueError('insufficient bounded oracle disk budget')
        for name, data in (('lo', lo[eye]), ('hi', hi[eye]), ('own_luma', ctx['images'][eye]), ('other_luma', ctx['images'][1-eye]), ('own_rect_support', ctx['support'][eye]), ('other_rect_support', ctx['support'][1-eye]), ('own', descriptors[eye].descriptor), ('other', descriptors[1-eye].descriptor), ('own_support', descriptors[eye].support), ('other_support', descriptors[1-eye].support)):
            write_plane(path/name, data)
        env = os.environ.copy(); env['FP2_DUMP_PATHS'] = '1'
        def bound_child():
            resource.setrlimit(resource.RLIMIT_AS, (MAX_CHILD_ADDRESS_BYTES, MAX_CHILD_ADDRESS_BYTES))
        subprocess.run([str(Path(args.runner).resolve()), str(path.resolve()), '720', '1280', '128', '8', '32', str(eye)], env=env, check=True, timeout=1200, preexec_fn=bound_child)
        cpu_offsets = exact_file(path/'offset', '<u8', 720*1280+1)
        if not np.array_equal(cpu_offsets, offsets):
            raise ValueError('oracle offsets mismatch')
        for name, byte_count in (('paths', total*16), ('sums', total*4), ('winner', 720*1280*4), ('best', 720*1280*4), ('second', 720*1280*4), ('count', 720*1280*4)):
            if (path/name).stat().st_size != byte_count:
                raise ValueError('oracle output size mismatch ' + name)
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in path.iterdir() if p.is_file()}
        result['eyes'][str(eye)] = {'bucket_words': total, 'oracle_path_sum_bytes': oracle_bytes, 'files': files}
    if any(sha(Path(path)) != digest for path, digest in before.items()):
        raise ValueError('source changed during preparation')
    # Revalidate referenced capture/calibration contents and decoded planes at end.
    after_ctx = cmp.load_context(coarse, target)
    if (after_ctx['source_id'] != ctx['source_id'] or
            not np.array_equal(after_ctx['images'], ctx['images']) or
            not np.array_equal(after_ctx['support'], ctx['support'])):
        raise ValueError('decoded input changed during preparation')
    after_planes = cmp._read_planes(nr['files'], cmp.PLANES, (2, 720, 1280), args.dump_root)
    if any(not np.array_equal(after_planes[k], planes[k], equal_nan=True) for k in planes):
        raise ValueError('native input planes changed during preparation')
    with (out/'manifest.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runner', 'build-manifest', 'artifact-root', 'family-root', 'native-report', 'dump-root', 'output-root'):
        parser.add_argument('--' + name, required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
