#!/usr/bin/env python3
"""Offline first-pass interval work estimate; no truth or match-based selection.

Wave estimates assume one independent pixel's interval occupies whole waves,
rounded upward separately for that pixel; this is not actual GPU scheduling,
traffic, elapsed time or a claim about a viable compact SGM implementation.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def rounded_lanes(length, wave):
    length = np.asarray(length)
    if wave <= 0 or np.any(length < 0):
        raise ValueError('invalid wave or length')
    return ((length + wave - 1) // wave) * wave


def active_intervals(prediction_valid, census_support, lower, upper, disparities):
    valid, support, lo, hi = map(np.asarray, (prediction_valid, census_support, lower, upper))
    if valid.ndim != 2 or not valid.size or any(a.shape != valid.shape for a in (support, lo, hi)) or disparities < 1:
        raise ValueError('invalid input shape or disparities')
    if not np.all((valid == 0) | (valid == 1)) or not np.all((support == 0) | (support == 1)):
        raise ValueError('nonbinary masks')
    guided = valid.astype(bool) & support.astype(bool)
    if not np.issubdtype(lo.dtype, np.integer) or not np.issubdtype(hi.dtype, np.integer):
        raise ValueError('bounds must be integer')
    if np.any((lo[guided] < 0) | (lo[guided] > hi[guided]) | (hi[guided] >= disparities)):
        raise ValueError('invalid guided inclusive bounds')
    low = np.zeros(valid.shape, dtype=np.int64)
    high = np.full(valid.shape, disparities, dtype=np.int64)
    low[guided] = lo[guided]
    high[guided] = hi[guided].astype(np.int64) + 1
    return low, high, guided


def estimate_eye(prediction_valid, census_support, lower, upper, disparities):
    low, high, guided = active_intervals(prediction_valid, census_support, lower, upper, disparities)
    valid, support = np.asarray(prediction_valid), np.asarray(census_support)
    lengths = high - low
    pixels = valid.size
    fallback = pixels - int(guided.sum())
    active = int(lengths.sum())
    dense = pixels * disparities
    return {'pixels': pixels, 'guided_pixels': int(guided.sum()), 'fallback_pixels': fallback,
            'fallback_fraction': fallback / pixels, 'unsupported_pixels': int((~support.astype(bool)).sum()),
            'active_candidates': active, 'dense_candidates': dense, 'active_fraction': active / dense,
            'four_path_candidate_updates': 4 * active, 'four_path_dense_updates': 4 * dense,
            'wave32_lanes': int(rounded_lanes(lengths, 32).sum()),
            'wave64_lanes': int(rounded_lanes(lengths, 64).sum())}


def truth_coverage(lo, hi, truth, support, regions, tolerance=1.0):
    """Count truth with an integer candidate within tolerance in [lo, hi)."""
    lo, hi, truth, support = map(np.asarray, (lo, hi, truth, support))
    if any(a.shape != lo.shape for a in (hi, truth, support)) or tolerance < 0:
        raise ValueError('invalid truth coverage inputs')
    support = support.astype(bool)
    if np.any(~np.isfinite(truth[support])) or np.any(lo >= hi):
        raise ValueError('invalid supported truth or interval')
    with np.errstate(invalid='ignore'):
        represented = np.ceil(np.maximum(lo, truth - tolerance)) <= np.floor(np.minimum(hi - 1, truth + tolerance))
    rows = {}
    for name, region in regions.items():
        region = np.asarray(region, dtype=bool)
        if region.shape != lo.shape:
            raise ValueError('invalid region shape')
        eligible = region & support
        denominator = int(eligible.sum())
        missed = int((eligible & ~represented).sum())
        rows[name] = {'region_pixels': int(region.sum()), 'eligible': denominator,
                      'excluded': int((region & ~support).sum()), 'representable': denominator - missed,
                      'unrepresentable': missed, 'unrepresentable_fraction': missed / denominator if denominator else None}
    return rows


def main():
    import compare_daxa_refinement as cmp
    import fast60_prior_diagnostic as diag
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coarse-manifest', 'target-manifest', 'report', 'dump-root', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--family', help='optional fixed truth family, evaluated after interval selection')
    parser.add_argument('--radius', type=int, choices=(2, 4, 8), default=2)
    parser.add_argument('--subpixel', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    context = cmp.load_context(args.coarse_manifest, args.target_manifest)
    report_path = Path(args.report).resolve()
    report = cmp.strict_json(report_path.read_text())
    settings = cmp.oracle.Settings(radius=args.radius, subpixel=bool(args.subpixel))
    diag.validate_report(cmp, report, context, settings)
    h, w = context['images'].shape[1:]
    actual = cmp._read_planes(report['files'], cmp.PLANES, (2, h, w), args.dump_root)
    diag.validate_wire_masks({name: actual[name] for name in
        ('prediction_validity.u32', 'raw_validity.u32', 'validity.u32', 'census_validity.u32')})
    diag.validate_wire_values(actual, settings.disparities, settings.radius)
    support = diag.verified_census_support(context['images'], context['support'], actual['census_validity.u32'], cmp.oracle)
    eyes = {name: estimate_eye(actual['prediction_validity.u32'][i], support[i], actual['lower.u32'][i],
                              actual['upper.u32'][i], settings.disparities)
            for i, name in enumerate(('left', 'right'))}
    # Freeze both eyes' selected windows before optionally reading truth.
    intervals = [active_intervals(actual['prediction_validity.u32'][i], support[i],
                 actual['lower.u32'][i], actual['upper.u32'][i], settings.disparities)[:2] for i in (0, 1)]
    totals = {k: sum(e[k] for e in eyes.values()) for k in ('pixels', 'guided_pixels', 'fallback_pixels',
              'unsupported_pixels', 'active_candidates', 'dense_candidates', 'four_path_candidate_updates',
              'four_path_dense_updates', 'wave32_lanes', 'wave64_lanes')}
    totals['fallback_fraction'] = totals['fallback_pixels'] / totals['pixels']
    totals['active_fraction'] = totals['active_candidates'] / totals['dense_candidates']
    result = {'schema': 'stereoforge.compact_sgm_work.v1',
              'scope': 'offline first-pass work counts; no truth, raw match or LR selection',
              'model': 'one pixel per independent wave, rounded per interval; not actual GPU scheduling or timing',
              'configuration': {key: report.get(key) for key in ('width', 'height', 'radius', 'subpixel', 'disparities', 'source_id')},
              'input_hashes': {'report': cmp.sha256(report_path), 'script': cmp.sha256(Path(__file__)),
                               'coarse_manifest': cmp.sha256(context['coarse_path']), 'target_manifest': cmp.sha256(context['target_path']),
                               'native_planes': {k: v['sha256'] for k, v in report['files'].items()}},
              'eyes': eyes, 'total': totals}
    if args.family:
        truth = cmp.load_truth(args.family, context)
        coverage = {}
        for i, name in enumerate(('left', 'right')):
            t = truth[name]
            domain = (t['valid'].astype(bool) & t['visible'].astype(bool) & ~t['out_of_range'].astype(bool) &
                      ~t['off_image'].astype(bool) & t['remap_support'].astype(bool) & support[i] &
                      ~t['mixed_bilinear'].astype(bool))
            regions = {'all': np.ones_like(domain), **{r: t['region_' + r] for r in ('near', 'thin', 'edges')}}
            coverage[name] = truth_coverage(*intervals[i], t['disparity'], domain, regions)
        result['truth_coverage'] = {'scope': 'post-hoc analytic candidate representability, not recovered accuracy',
                                    'tolerance_disparity_pixels': 1.0, 'eyes': coverage}
        result['input_hashes']['family'] = cmp.sha256(Path(args.family))
    with Path(args.output).open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')


if __name__ == '__main__':
    main()
