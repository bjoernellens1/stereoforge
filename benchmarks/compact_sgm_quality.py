"""Small-input compiled recurrence parity harness; not a quality benchmark.

Eye is a test label only: this harness accepts already constructed costs and
therefore makes no assertion about left/right image geometry. Infinity means
unsupported, and ties select the lowest global disparity; no uniqueness or LR
filter is applied. This must not be used as metric depth output.
"""
import subprocess
from pathlib import Path

import numpy as np

INF = np.iinfo(np.uint32).max


def run_costs(runner, directory, cost, lo, hi, p1, p2, eye):
    if eye not in ('left', 'right'):
        raise ValueError('invalid eye')
    from benchmarks.compact_sgm_reference import _inputs
    cost, lo, hi = _inputs(cost, lo, hi, p1, p2)
    h, w, d = cost.shape
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    converted = np.full(cost.shape, INF, dtype='<u4')
    finite = np.isfinite(cost)
    converted[finite] = cost[finite].astype(np.uint32)
    converted.tofile(root / 'cost')
    np.asarray(lo, dtype='<u4').tofile(root / 'lo')
    np.asarray(hi, dtype='<u4').tofile(root / 'hi')
    subprocess.run([str(runner), str(root), str(h), str(w), str(d), str(p1), str(p2)], check=True)
    def load(name, shape):
        values = np.fromfile(root / name, dtype='<u4').reshape(shape).astype(np.float64)
        values[values == INF] = np.inf
        return values
    return (load('paths', (4, h, w, d)), load('sums', (h, w, d)),
            np.fromfile(root / 'winners', dtype='<i4').reshape(h, w))


def run_eye(runner, directory, own, other, lo, hi, eye, disparities, dump_scores=False):
    """Four-path on-demand Census Hamming, with one score volume per invocation."""
    import os
    if eye not in (0, 1):
        raise ValueError('eye must be 0 or 1')
    shape = own.descriptor.shape
    if len(shape) != 2 or any(a.shape != shape for a in (other.descriptor, own.support, other.support, lo, hi)):
        raise ValueError('plane shapes differ')
    if np.any(lo < 0) or np.any(hi > disparities) or np.any(lo > hi):
        raise ValueError('invalid intervals')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, value in [('own', own.descriptor), ('other', other.descriptor),
                        ('own_support', own.support), ('other_support', other.support), ('lo', lo), ('hi', hi)]:
        np.asarray(value, dtype='<u4').tofile(root / name)
    env = os.environ.copy()
    env.pop('COMPACT_DUMP_SCORES', None)
    if dump_scores:
        env['COMPACT_DUMP_SCORES'] = '1'
    h, w = shape
    subprocess.run([str(runner), str(root), str(h), str(w), str(disparities), '8', '32', str(eye)], check=True, env=env)
    result = {name: np.fromfile(root / name, dtype=dtype).reshape(shape)
              for name, dtype in [('winners', '<i4'), ('best', '<u4'), ('second', '<u4'), ('count', '<u4')]}
    result['candidate_updates'], result['interval_slots'] = map(int, (root / 'stats').read_text().split())
    if dump_scores:
        scores = np.fromfile(root / 'sums', dtype='<u4').reshape(h, w, disparities).astype(np.float64)
        scores[scores == INF] = np.inf
        result['scores'] = scores
    return result


def select_eye(result, lo, hi, disparities):
    """Unique best, reject ties and restricted endpoints; no subpixel fit."""
    winner, best, second, count = (result[k] for k in ('winners', 'best', 'second', 'count'))
    restricted = ((winner == lo) & (lo > 0)) | ((winner == hi - 1) & (hi < disparities))
    valid = (count >= 2) & (winner > 0) & (second > best) & ~restricted
    confidence = np.zeros(winner.shape, dtype=np.float32)
    np.divide((second.astype(np.float64) - best).astype(np.float32),
              np.maximum(second, 1).astype(np.float32), out=confidence, where=valid)
    disparity = np.where(valid, winner, np.nan).astype(np.float32)
    return disparity, confidence, valid


def main():
    import argparse
    import hashlib
    import json
    import resource
    import tempfile
    import time
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    import compact_sgm_work as work
    from stereoforge.refinement import census, lr_consistency

    parser = argparse.ArgumentParser(description='Offline compact SGM development quality; not a GPU performance test')
    for name in ('runner', 'coarse-manifest', 'target-manifest', 'family', 'report', 'dump-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--build-manifest', required=True, help='compiler invocation and binary/source hashes')
    parser.add_argument('--unrestricted', action='store_true', help='full D infinity-semantics control')
    args = parser.parse_args()
    build = cmp.strict_json(Path(args.build_manifest).read_text())
    if build['source_sha256'] != cmp.sha256(Path(__file__).with_name('compact_sgm_runner.cpp')) or build['binary_sha256'] != cmp.sha256(Path(args.runner)):
        raise ValueError('runner build manifest hash mismatch')
    if not isinstance(build.get('command'), list) or not isinstance(build.get('compiler_version'), str):
        raise ValueError('missing compiler provenance')
    context = cmp.load_context(args.coarse_manifest, args.target_manifest)
    settings = cmp.oracle.Settings(radius=2, subpixel=False)
    report_path = Path(args.report).resolve()
    report = cmp.strict_json(report_path.read_text())
    diag.validate_report(cmp, report, context, settings)
    h, w = context['images'].shape[1:]
    native = cmp._read_planes(report['files'], cmp.PLANES, (2, h, w), args.dump_root)
    diag.validate_wire_masks({k: native[k] for k in ('prediction_validity.u32', 'raw_validity.u32', 'validity.u32', 'census_validity.u32')})
    diag.validate_wire_values(native, settings.disparities, settings.radius)
    support = diag.verified_census_support(context['images'], context['support'], native['census_validity.u32'], cmp.oracle)
    descriptors = [census(context['images'][i], context['support'][i]) for i in (0, 1)]
    intervals = []
    for i in (0, 1):
        lo, hi, _ = work.active_intervals(native['prediction_validity.u32'][i], support[i], native['lower.u32'][i], native['upper.u32'][i], settings.disparities)
        if args.unrestricted:
            lo = np.zeros((h, w), dtype=np.int32)
            hi = np.full((h, w), settings.disparities, dtype=np.int32)
        intervals.append((lo, hi))
    # Freeze intervals before any truth is loaded.
    interval_hashes = [hashlib.sha256(np.asarray(p, dtype='<u4').tobytes()).hexdigest() for pair in intervals for p in pair]
    raw, confidence, valid, timing, cpu_timing, counts = [], [], [], [], [], []
    with tempfile.TemporaryDirectory(prefix='compact-sgm-') as temp:
        for i in (0, 1):
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            start = time.perf_counter()
            eye = run_eye(args.runner, Path(temp) / str(i), descriptors[i], descriptors[1-i], *intervals[i], i, settings.disparities)
            timing.append(time.perf_counter() - start)
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_timing.append((after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime))
            counts.append({k: eye[k] for k in ('candidate_updates', 'interval_slots')})
            d, c, v = select_eye(eye, *intervals[i], settings.disparities)
            raw.append(d); confidence.append(c); valid.append(v)
    raw = np.stack(raw); valid = np.stack(valid)
    lr = lr_consistency(raw, valid, 1.)
    geometry = context['geometry']
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = raw - np.float32(geometry.cx_offset)
        z = np.float32(geometry.fx * geometry.baseline_m) / denom
    metric = lr & np.asarray(context['support'], bool) & np.isfinite(z) & (z > 0) & (denom > 0) & (raw > 0)
    actual = {'raw_validity.u32': valid, 'raw_disparity.f32': raw, 'validity.u32': lr,
              'metric_validity.u32': metric, 'depth.f32': np.where(metric, z, np.nan).astype(np.float32)}
    truth = cmp.load_truth(args.family, context)
    result = {'scope': 'offline analytic development fixture; no GPU, physical, or 60 FPS claim',
              'configuration': {'disparities': settings.disparities, 'p1': 8, 'p2': 32, 'subpixel': False,
                                'intervals': 'full D control' if args.unrestricted else 'frozen radius 2 with full D for missing priors',
                                'selection': 'lowest tie winner; zero participates in competition but a zero winner is invalid; at least 2 finite candidates, positive best-second margin, reject restricted interval endpoints',
                                'lr_tolerance_pixels': 1.0},
              'hashes': {'coarse_manifest': cmp.sha256(context['coarse_path']), 'target_manifest': cmp.sha256(context['target_path']),
                         'family': cmp.sha256(Path(args.family)), 'native_report': cmp.sha256(report_path),
                         'runner': cmp.sha256(Path(args.runner)), 'runner_source': cmp.sha256(Path(__file__).with_name('compact_sgm_runner.cpp')),
                         'harness_source': cmp.sha256(Path(__file__)), 'native_planes': {k: v['sha256'] for k, v in report['files'].items()},
                         'interval_lo_hi_by_eye': interval_hashes},
              'build': build, 'output_plane_hashes': {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in actual.items()},
              'quality_caveat': 'LR accepted counts include pixels outside clean truth support; false accepted means not visible or disparity error >1 px, not all outside-domain pixels',
              'counts_by_eye': counts, 'wall_seconds_by_eye_including_file_io': timing,
              'child_cpu_seconds_by_eye': cpu_timing, 'children_max_rss_kib_process_lifetime_high_water_not_per_eye': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              'quality': cmp.evaluate_quality(actual, truth, support)}
    with Path(args.output).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
