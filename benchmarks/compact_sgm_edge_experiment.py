#!/usr/bin/env python3
"""Offline development-only fixed edge-expanded independent-eye compact SGM.

Configuration and provisional gates are fixed here before truth is loaded.
This is a CPU reference experiment, not a production matcher or GPU benchmark.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from benchmarks.compact_sgm_work import active_intervals
from benchmarks.compact_sgm_quality import run_eye, select_eye

CONFIG = {'gradient_threshold': 64, 'edge_dilation_radius': 2,
          'extra_disparity_radius': 8, 'disparities': 128,
          'p1': 8, 'p2': 32, 'lr_tolerance_pixels': 1.0}
# Provisional development screen only, not a scientific or release criterion.
GATES = {'max_candidate_fraction_of_dense': 0.50,
         'max_correct_metric_completeness_loss_pp_vs_full_sgm': 2.0,
         'max_thin_correct_metric_completeness_loss_pp_vs_full_sgm': 2.0,
         'max_near_correct_metric_completeness_loss_pp_vs_full_sgm': 2.0,
         'max_edges_correct_metric_completeness_loss_pp_vs_full_sgm': 2.0,
         'max_surface_false_accepted_increase': 0}


def intervals(image, prediction_valid, support, lower, upper, disparities):
    image = np.asarray(image)
    if image.ndim != 2 or not np.issubdtype(image.dtype, np.unsignedinteger) or np.any(image > 255):
        raise ValueError('expected unsigned 8-bit-range 2D luma')
    lo, hi, guided = active_intervals(prediction_valid, support, lower, upper, disparities)
    if lo.shape != image.shape:
        raise ValueError('image/interval shape mismatch')
    # Integer luminance differences, with each adjacent pair attributed to both pixels.
    v = image.astype(np.int16)
    edge = np.zeros(image.shape, bool)
    x = np.abs(v[:, 1:] - v[:, :-1]) >= CONFIG['gradient_threshold']
    y = np.abs(v[1:, :] - v[:-1, :]) >= CONFIG['gradient_threshold']
    edge[:, 1:] |= x; edge[:, :-1] |= x
    edge[1:, :] |= y; edge[:-1, :] |= y
    radius = CONFIG['edge_dilation_radius']
    padded = np.pad(edge, radius)
    expanded = np.zeros_like(edge)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            expanded |= padded[dy:dy + image.shape[0], dx:dx + image.shape[1]]
    changed = expanded & guided
    lo[changed] = np.maximum(0, lo[changed] - CONFIG['extra_disparity_radius'])
    hi[changed] = np.minimum(disparities, hi[changed] + CONFIG['extra_disparity_radius'])
    return lo, hi, expanded


def surface_counts(actual, truth):
    output = {}
    for eye, name in enumerate(('left', 'right')):
        t = truth[name]
        ids = t['surface_id']
        accepted = np.asarray(actual['validity.u32'][eye], bool)
        visible = np.asarray(t['valid'], bool) & np.asarray(t['visible'], bool)
        error = np.abs(actual['raw_disparity.f32'][eye] - t['disparity'])
        false = accepted & (~visible | ~np.isfinite(error) | (error > 1))
        output[name] = {}
        for sid in np.unique(ids):
            mask = ids == sid
            output[name][str(int(sid))] = {'pixels': int(mask.sum()), 'accepted': int((mask & accepted).sum()),
                                           'false_accepted': int((mask & false).sum())}
    return output


def surface_quality(truth, actual, support):
    t = truth
    visible = np.asarray(t['valid'], bool) & np.asarray(t['visible'], bool)
    domain = (visible & ~np.asarray(t['out_of_range'], bool) & ~np.asarray(t['off_image'], bool)
              & np.asarray(t['remap_support'], bool) & np.asarray(support, bool) & ~np.asarray(t['mixed_bilinear'], bool))
    error = np.abs(actual['raw_disparity.f32'] - t['disparity'])
    metric, lr = np.asarray(actual['metric_validity.u32'], bool), np.asarray(actual['validity.u32'], bool)
    result = {}
    for sid in np.unique(t['surface_id']):
        eligible = domain & (t['surface_id'] == sid)
        result[str(int(sid))] = {'eligible': int(eligible.sum()), 'metric_accepted': int((eligible & metric).sum()),
                                 'correct_metric': int((eligible & metric & (error <= 1)).sum()),
                                 'lr_false_accepted': int((eligible & lr & (~np.isfinite(error) | (error > 1))).sum())}
    return result


def thin_representability(truth, actual, support, lo, hi):
    """Count disjoint outcomes on fixed clean, visible, in-range thin truth support."""
    t = truth
    eligible = (np.asarray(t['valid'], bool) & np.asarray(t['visible'], bool)
                & ~np.asarray(t['out_of_range'], bool) & ~np.asarray(t['off_image'], bool)
                & np.asarray(t['remap_support'], bool) & np.asarray(support, bool)
                & ~np.asarray(t['mixed_bilinear'], bool) & np.asarray(t['region_thin'], bool))
    disparity = np.asarray(t['disparity'])
    # An integer candidate within one pixel exists iff these integer bounds overlap.
    representable = eligible & np.isfinite(disparity) & (np.ceil(disparity - 1) < hi) & (np.floor(disparity + 1) >= lo)
    metric = np.asarray(actual['metric_validity.u32'], bool)
    correct = metric & (np.abs(actual['raw_disparity.f32'] - disparity) <= 1)
    return {'eligible': int(eligible.sum()), 'representable': int(representable.sum()),
            'not_representable': int((eligible & ~representable).sum()),
            'representable_metric_invalid': int((representable & ~metric).sum()),
            'representable_metric_incorrect': int((representable & metric & ~correct).sum()),
            'representable_metric_correct': int((representable & correct).sum())}


def main():
    import argparse
    import sys
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    import stereoforge.refinement as refinement
    from stereoforge.refinement import census, lr_consistency

    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runner', 'build-manifest', 'coarse-manifest', 'target-manifest', 'family', 'report', 'dump-root', 'frozen', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    build = cmp.strict_json(Path(args.build_manifest).read_text())
    if (build['source_sha256'] != cmp.sha256(Path(__file__).with_name('compact_sgm_runner.cpp'))
            or build['binary_sha256'] != cmp.sha256(Path(args.runner))
            or not isinstance(build.get('command'), list) or not isinstance(build.get('compiler_version'), str)):
        raise ValueError('invalid runner build provenance')
    context = cmp.load_context(args.coarse_manifest, args.target_manifest)
    settings = cmp.oracle.Settings(radius=2, subpixel=False)
    report = cmp.strict_json(Path(args.report).read_text())
    diag.validate_report(cmp, report, context, settings)
    h, w = context['images'].shape[1:]
    native = cmp._read_planes(report['files'], cmp.PLANES, (2, h, w), args.dump_root)
    diag.validate_wire_masks({k: native[k] for k in ('prediction_validity.u32', 'raw_validity.u32', 'validity.u32', 'census_validity.u32')})
    diag.validate_wire_values(native, settings.disparities, settings.radius)
    support = diag.verified_census_support(context['images'], context['support'], native['census_validity.u32'], cmp.oracle)
    descriptors = [census(context['images'][i], context['support'][i]) for i in (0, 1)]
    bounds = [intervals(context['images'][i], native['prediction_validity.u32'][i], support[i],
                        native['lower.u32'][i], native['upper.u32'][i], settings.disparities) for i in (0, 1)]
    input_hashes = {'coarse': cmp.sha256(context['coarse_path']), 'target': cmp.sha256(context['target_path']),
                    'report': cmp.sha256(Path(args.report)), 'runner': cmp.sha256(Path(args.runner)),
                    'build_manifest': cmp.sha256(Path(args.build_manifest)),
                    'family': cmp.sha256(Path(args.family)),
                    'refinement_source': cmp.sha256(Path(refinement.__file__)),
                    'source': cmp.sha256(Path(__file__)),
                    'dependencies': {p: cmp.sha256(Path(__file__).with_name(p)) for p in
                        ('compact_sgm_work.py', 'compact_sgm_quality.py', 'compare_daxa_refinement.py',
                         'fast60_prior_diagnostic.py', 'compact_sgm_runner.cpp')},
                    'native_planes': {k: v['sha256'] for k, v in report['files'].items()}}
    # Exclusive creation guarantees the actual intervals and decisions are persisted before truth access.
    with open(args.frozen, 'xb') as f:
        np.savez_compressed(f, lo=np.stack([b[0] for b in bounds]), hi=np.stack([b[1] for b in bounds]),
                            edge=np.stack([b[2] for b in bounds]),
                            config=np.array(json.dumps(CONFIG, sort_keys=True)), gates=np.array(json.dumps(GATES, sort_keys=True)),
                            input_hashes=np.array(json.dumps(input_hashes, sort_keys=True)))
    frozen_hash = cmp.sha256(Path(args.frozen))
    raw, valid, stats, seconds = [], [], [], []
    with tempfile.TemporaryDirectory(prefix='compact-edge-') as temp:
        for eye in (0, 1):
            start = time.perf_counter()
            result = run_eye(args.runner, Path(temp) / str(eye), descriptors[eye], descriptors[1-eye],
                             bounds[eye][0], bounds[eye][1], eye, settings.disparities)
            seconds.append(time.perf_counter() - start)
            stats.append({k: result[k] for k in ('candidate_updates', 'interval_slots')})
            d, _, v = select_eye(result, bounds[eye][0], bounds[eye][1], settings.disparities)
            raw.append(d); valid.append(v)
    raw, valid = np.stack(raw), np.stack(valid)
    lr = lr_consistency(raw, valid, CONFIG['lr_tolerance_pixels'])
    geom = context['geometry']
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = raw - np.float32(geom.cx_offset)
        z = np.float32(geom.fx * geom.baseline_m) / denom
    metric = lr & np.asarray(context['support'], bool) & np.isfinite(z) & (z > 0) & (denom > 0) & (raw > 0)
    actual = {'raw_validity.u32': valid, 'raw_disparity.f32': raw, 'validity.u32': lr,
              'metric_validity.u32': metric, 'depth.f32': np.where(metric, z, np.nan).astype(np.float32)}
    # Persist output selections prior to opening ground truth, too.
    selection_path = str(args.frozen) + '.selection.npz'
    with open(selection_path, 'xb') as f:
        np.savez_compressed(f, **actual)
    selection_hash = cmp.sha256(Path(selection_path))
    if cmp.sha256(Path(args.frozen)) != frozen_hash:
        raise ValueError('frozen intervals changed before truth')
    for name, expected in input_hashes['dependencies'].items():
        if cmp.sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError('experiment dependency changed before truth')
    cmp._read_planes(report['files'], cmp.PLANES, (2, h, w), args.dump_root)
    if (cmp.sha256(Path(args.family)) != input_hashes['family']
            or cmp.sha256(Path(args.build_manifest)) != input_hashes['build_manifest']
            or cmp.sha256(Path(refinement.__file__)) != input_hashes['refinement_source']
            or cmp.sha256(context['coarse_path']) != input_hashes['coarse'] or cmp.sha256(context['target_path']) != input_hashes['target']
            or cmp.sha256(Path(args.report)) != input_hashes['report'] or cmp.sha256(Path(args.runner)) != input_hashes['runner']
            or cmp.sha256(Path(__file__)) != input_hashes['source']):
        raise ValueError('experiment input changed before truth')
    truth = cmp.load_truth(args.family, context)
    if cmp.sha256(Path(selection_path)) != selection_hash:
        raise ValueError('frozen selection changed during truth loading')
    if cmp.sha256(Path(args.family)) != input_hashes['family'] or cmp.sha256(Path(args.frozen)) != frozen_hash:
        raise ValueError('frozen input changed during truth loading')
    dense = 4 * 2 * h * w * settings.disparities
    output = {'scope': 'exploratory synthetic development; settings chosen after scene inspection, not blind held-out, no live or GPU performance claim',
              'limitations': 'four-path interval-slot fraction includes unsupported candidates; actual supported updates reported separately; runner allocates dense aggregate scores; per-eye timing includes runner IO but excludes preprocessing; D128 truth excludes disparities above 127 and does not qualify near field',
              'configuration': CONFIG, 'provisional_gates': GATES,
              'gate_status': 'not evaluated: requires matched independent full-SGM baseline',
              'work': {'interval_slots_by_eye': [s['interval_slots'] for s in stats],
                       'candidate_updates_by_eye': [s['candidate_updates'] for s in stats],
                       'four_path_dense_slots_both_eyes': dense,
                       'interval_fraction': sum(s['interval_slots'] for s in stats) / dense},
              'runner_wall_seconds_including_io_by_eye': seconds,
              'hashes': {'config_source': cmp.sha256(Path(__file__)), 'runner': cmp.sha256(Path(args.runner)),
                         'build_manifest': cmp.sha256(Path(args.build_manifest)),
                    'family': cmp.sha256(Path(args.family)), 'target_manifest': cmp.sha256(context['target_path']),
                         'coarse_manifest': cmp.sha256(context['coarse_path']), 'native_report': cmp.sha256(Path(args.report)),
                         'family': cmp.sha256(Path(args.family)), 'frozen_intervals': cmp.sha256(Path(args.frozen)),
                         'frozen_selection': selection_hash},
              'quality': cmp.evaluate_quality(actual, truth, support), 'surface_counts': surface_counts(actual, truth),
              'surface_counts_domain': 'all pixels bearing each surface ID; LR false accepts include occluded and disparity error above one pixel, including outside clean truth support',
              'surface_quality_domain': 'visible in-range, unmixed truth with valid census and remap support; no outside-domain pixels',
              'surface_quality_on_clean_fixed_support': {name: surface_quality(truth[name], {k: v[eye] for k, v in actual.items()}, support[eye])
                                                 for eye, name in enumerate(('left', 'right'))},
              'thin_representability': {name: thin_representability(truth[name], {k: v[eye] for k, v in actual.items()},
                                        support[eye], bounds[eye][0], bounds[eye][1])
                                        for eye, name in enumerate(('left', 'right'))},
              'thin_denominator': 'visible in-range clean thin truth with census and remap support; interval contains integer within 1 disparity pixel'}
    with open(args.output, 'x') as f:
        json.dump(output, f, indent=2, sort_keys=True, allow_nan=False); f.write('\n')


if __name__ == '__main__':
    main()
