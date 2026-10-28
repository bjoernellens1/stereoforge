#!/usr/bin/env python3
"""Development-only correlated LR-repair ablation; freezes outputs before truth."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import compare_daxa_refinement as cmp
import fast60_prior_diagnostic as diag
from compact_sgm_lr_repair import repair
from compact_sgm_gradient_experiment import surface_quality, surface_counts
import stereoforge.refinement as refinement
from stereoforge.refinement import census
from compact_sgm_gradient_experiment import CONFIG


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transitions(base, candidate, truth, support):
    output = {}
    for eye, name in enumerate(('left', 'right')):
        t = truth[name]
        domain = (t['valid'].astype(bool) & t['visible'].astype(bool) & ~t['out_of_range'].astype(bool)
                  & ~t['off_image'].astype(bool) & t['remap_support'].astype(bool) & support[eye]
                  & ~t['mixed_bilinear'].astype(bool))
        def labels(result):
            valid = result['metric_validity.u32'][eye].astype(bool)
            correct = np.abs(result['raw_disparity.f32'][eye] - t['disparity']) <= 1
            return np.where(~valid, 0, np.where(correct, 1, 2))
        a, b = labels(base), labels(candidate)
        output[name] = {region: [[int(np.count_nonzero(domain & mask & (a == i) & (b == j)))
                                   for j in range(3)] for i in range(3)]
                        for region, mask in [('all', np.ones(domain.shape, bool)),
                                             ('thin', t['region_thin'].astype(bool)),
                                             ('edges', t['region_edges'].astype(bool))]}
    return output


def main():
    p = argparse.ArgumentParser()
    for flag in ('coarse-manifest', 'target-manifest', 'family', 'report', 'dump-root', 'fixed-freeze', 'adaptive-freeze', 'fixed-report', 'adaptive-report', 'output-dir'):
        p.add_argument('--'+flag, required=True)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    context = cmp.load_context(args.coarse_manifest, args.target_manifest)
    settings = cmp.oracle.Settings(radius=2, subpixel=False)
    report = cmp.strict_json(Path(args.report).read_text())
    diag.validate_report(cmp, report, context, settings)
    h, w = context['images'].shape[1:]
    native = cmp._read_planes(report['files'], cmp.PLANES, (2, h, w), args.dump_root)
    support = diag.verified_census_support(context['images'], context['support'], native['census_validity.u32'], cmp.oracle)
    desc = np.stack([census(context['images'][i], context['support'][i]).descriptor for i in (0, 1)])
    paths = [Path(args.fixed_freeze), Path(args.adaptive_freeze)]
    deps = [Path(__file__), Path(__file__).with_name('compact_sgm_lr_repair.py'), Path(args.family),
            Path(args.report), Path(refinement.__file__), Path(cmp.__file__), Path(diag.__file__), Path(__file__).with_name('compact_sgm_gradient_experiment.py'), context['coarse_path'], context['target_path'], *paths,
            *[Path(str(path)+'.selection.npz') for path in paths], Path(args.fixed_report), Path(args.adaptive_report)]
    for path in paths:
        with np.load(path, allow_pickle=False) as frozen:
            prior = json.loads(str(frozen['input_hashes']))
            for name, expected in prior['dependencies'].items():
                dependency = Path(__file__).parent / name
                if digest(dependency) != expected:
                    raise ValueError('baseline dependency changed: ' + name)
                deps.append(dependency)
            if digest(Path(refinement.__file__)) != prior['refinement_source']:
                raise ValueError('baseline refinement source changed')
    hashes = {str(path): digest(path) for path in deps}
    records = {}
    for label, path, result_path in zip(('fixed', 'adaptive'), paths, (args.fixed_report, args.adaptive_report)):
        source_report = cmp.strict_json(Path(result_path).read_text())
        if (source_report['hashes']['frozen_intervals'] != digest(path)
                or source_report['hashes']['frozen_selection'] != digest(str(path)+'.selection.npz')):
            raise ValueError('baseline selection or interval hash mismatch')
        with np.load(path, allow_pickle=False) as frozen:
            ih = json.loads(str(frozen['input_hashes']))
            if (ih['family'] != hashes[str(Path(args.family))] or ih['report'] != hashes[str(Path(args.report))]
                    or ih['coarse'] != digest(context['coarse_path']) or ih['target'] != digest(context['target_path'])
                    or json.loads(str(frozen['config'])) != CONFIG
                    or (label == 'adaptive' and bool(frozen['full_d']))
                    or (label == 'adaptive' and ih['baseline_freeze'] != digest(paths[0]))):
                raise ValueError('baseline inputs do not match')
        with np.load(str(path)+'.selection.npz', allow_pickle=False) as f:
            base = {k: f[k] for k in f.files}
        result = repair(base['raw_disparity.f32'], base['raw_validity.u32'], desc, support)
        geom = context['geometry']
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = result.raw - np.float32(geom.cx_offset)
            z = np.float32(geom.fx * geom.baseline_m) / denom
        metric = result.lr & context['support'].astype(bool) & np.isfinite(z) & (z > 0) & (denom > 0) & (result.raw > 0)
        actual = {'raw_validity.u32': result.valid, 'raw_disparity.f32': result.raw,
                  'validity.u32': result.lr, 'metric_validity.u32': metric,
                  'depth.f32': np.where(metric, z, np.nan).astype(np.float32)}
        records[label] = (base, actual, result)
    # Both candidate decisions and policy are materialized before truth is read.
    pretruth = out/'selection.npz'
    with pretruth.open('xb') as f:
        np.savez_compressed(f, **{label+'__'+key: value for label, (_, actual, _) in records.items() for key, value in actual.items()},
                            hashes=np.array(json.dumps(hashes, sort_keys=True)),
                            policy=np.array('D128 full opposite-reference Census search, unique best, >=2 supported candidates, strict LR 1px; reject multiple source claims; preserve valid original'))
    pretruth_hash = digest(pretruth)
    if any(digest(p) != hashes[str(p)] for p in deps):
        raise ValueError('inputs changed before truth')
    truth = cmp.load_truth(args.family, context)
    if digest(pretruth) != pretruth_hash or any(digest(p) != hashes[str(p)] for p in deps):
        raise ValueError('frozen inputs changed during truth loading')
    output = {'scope': 'previously inspected synthetic development fixture; CPU-only, no runtime, camera, held-out or release claim',
              'policy': 'opposite-reference full D128 unique Census search, correlated with request and same image pair; not independent photometric or SGM evidence',
              'selection_sha256': pretruth_hash, 'input_sha256': hashes, 'configs': {}}
    for label, (base, actual, result) in records.items():
        output['configs'][label] = {'work': {k: getattr(result, k) for k in ('requests', 'targets', 'candidates', 'ties', 'unsupported', 'conflicts', 'zero_winners')},
            'base_quality': cmp.evaluate_quality(base, truth, support), 'quality': cmp.evaluate_quality(actual, truth, support),
            'base_surface_counts': surface_counts(base, truth), 'surface_counts': surface_counts(actual, truth),
            'base_surface_quality': {n: surface_quality(truth[n], {k:v[e] for k,v in base.items()}, support[e]) for e,n in enumerate(('left','right'))},
            'surface_quality': {n: surface_quality(truth[n], {k:v[e] for k,v in actual.items()}, support[e]) for e,n in enumerate(('left','right'))},
            'transitions': transitions(base, actual, truth, support), 'transition_order': ['invalid','correct','false']}
    with (out/'result.json').open('x') as f:
        json.dump(output, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')

if __name__ == '__main__':
    main()
