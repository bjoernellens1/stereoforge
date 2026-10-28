#!/usr/bin/env python3
"""Offline, truth-labelled attribution of guided matching failures; never drives dispatch."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

CATEGORIES = ('no_prediction', 'window_misses_truth', 'raw_invalid',
              'raw_error', 'correct_lr_invalid', 'correct_lr_valid')


def validate_wire_masks(masks):
    for name, mask in masks.items():
        if np.any((mask != 0) & (mask != 1)):
            raise ValueError('nonbinary ' + name)


def validate_wire_values(planes, disparities, radius=2):
    valid = planes['prediction_validity.u32'] != 0
    prediction = planes['prediction.f32']
    lower, upper = planes['lower.u32'], planes['upper.u32']
    if np.any(valid & (~np.isfinite(prediction) | (lower > upper) | (upper >= disparities) |
                       (prediction < lower) | (prediction > upper))):
        raise ValueError('invalid valid prediction or window')
    # Match the specified binary32 addition and floor used by the oracle.
    centre = np.floor(prediction[valid].astype(np.float32) + np.float32(.5)).astype(np.int64)
    if np.any(lower[valid] != np.maximum(0, centre - radius)) or np.any(
            upper[valid] != np.minimum(disparities - 1, centre + radius)):
        raise ValueError('window differs from configured prediction radius')
    raw_valid = planes['raw_validity.u32'] != 0
    raw = planes['raw_disparity.f32']
    if np.any(raw_valid & ~valid):
        raise ValueError('guided raw validity without prediction')
    if np.any(raw_valid & (~np.isfinite(raw) | (raw < 0) | (raw >= disparities))):
        raise ValueError('invalid valid raw disparity')
    lr_valid = planes['validity.u32'] != 0
    disparity = planes['disparity.f32']
    if np.any(lr_valid & (~raw_valid | ~np.isfinite(disparity) | (disparity != raw))):
        raise ValueError('invalid LR disparity')


def validate_report(comparator, report, context, settings):
    validation = report.get('validation_requested')
    if type(validation) is not bool:
        raise ValueError('invalid validation_requested')
    expected = comparator.expected_report(context, settings, validation)
    for key, value in expected.items():
        if key not in report:
            raise ValueError('missing report field: ' + key)
        comparator._same(report[key], value, key)


def verified_census_support(images, support, native, oracle):
    expected = np.stack([oracle.census(images[i], support[i]).support for i in (0, 1)])
    if not np.array_equal(expected, np.asarray(native, dtype=bool)):
        raise ValueError('native Census support differs from CPU fixture support')
    return expected


def classify(truth, support, prediction, prediction_valid, lower, upper, raw, raw_valid, lr_valid):
    arrays = [np.asarray(a) for a in (truth, support, prediction, prediction_valid, lower, upper, raw, raw_valid, lr_valid)]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('plane shapes differ')
    truth, support, prediction, prediction_valid, lower, upper, raw, raw_valid, lr_valid = arrays
    support = support.astype(bool)
    if not np.isfinite(truth[support]).all():
        raise ValueError('nonfinite truth on supported pixels')
    prediction_ok = prediction_valid.astype(bool) & np.isfinite(prediction)
    with np.errstate(invalid='ignore'):
        # A qualifying integer exists iff the inclusive intervals intersect.
        window_ok = (np.isfinite(lower) & np.isfinite(upper) & (lower <= upper) &
                     (np.ceil(np.maximum(lower, truth - 1)) <= np.floor(np.minimum(upper, truth + 1))))
        raw_correct = np.isfinite(raw) & (np.abs(raw.astype(np.float64) - truth) <= 1)
    available = support & prediction_ok & window_ok
    valid_raw = raw_valid.astype(bool)
    good = available & valid_raw & raw_correct
    return dict(zip(CATEGORIES, (support & ~prediction_ok,
        support & prediction_ok & ~window_ok, available & ~valid_raw,
        available & valid_raw & ~raw_correct, good & ~lr_valid.astype(bool),
        good & lr_valid.astype(bool))))


def region_counts(masks, regions):
    if tuple(masks) != CATEGORIES:
        raise ValueError('unexpected categories')
    domain = np.zeros_like(next(iter(masks.values())), dtype=np.uint8)
    for mask in masks.values():
        domain += np.asarray(mask, dtype=np.uint8)
    if np.any(domain > 1):
        raise ValueError('overlapping categories')
    rows = {}
    for name, region in regions.items():
        region = np.asarray(region, dtype=bool)
        if region.shape != domain.shape:
            raise ValueError('region shape differs')
        counts = {k: int(np.count_nonzero(v & region)) for k, v in masks.items()}
        denominator = int(np.count_nonzero((domain != 0) & region))
        if sum(counts.values()) != denominator:
            raise ValueError('partition does not conserve support')
        rows[name] = {'denominator': denominator, 'counts': counts}
    return rows


def main():
    import compare_daxa_refinement as cmp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coarse-manifest', required=True)
    parser.add_argument('--target-manifest', required=True)
    parser.add_argument('--family', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--dump-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--radius', type=int, choices=(2, 4, 8), default=2)
    parser.add_argument('--subpixel', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    context = cmp.load_context(args.coarse_manifest, args.target_manifest)
    truth = cmp.load_truth(args.family, context)
    report_path = Path(args.report).resolve()
    report = cmp.strict_json(report_path.read_text())
    h, w = context['images'].shape[1:]
    settings = cmp.oracle.Settings(radius=args.radius, subpixel=bool(args.subpixel))
    validate_report(cmp, report, context, settings)
    actual = cmp._read_planes(report['files'], cmp.PLANES, (2, h, w), args.dump_root)
    validate_wire_masks({name: actual[name] for name in
                         ('prediction_validity.u32', 'raw_validity.u32', 'validity.u32', 'census_validity.u32')})
    validate_wire_values(actual, settings.disparities, settings.radius)
    census_support = verified_census_support(context['images'], context['support'],
                                             actual['census_validity.u32'], cmp.oracle)
    result = {'schema': 'stereoforge.fast60_prior_diagnostic.v1',
              'scope': 'offline analytic development truth labeling only; never used for GPU dispatch',
              'denominator': 'visible in-range centre-ray truth with full Census/remap support and no mixed raw footprint',
              'threshold_disparity_pixels': 1.0,
              'configuration': {key: report.get(key) for key in ('width', 'height', 'coarse_width', 'coarse_height', 'radius', 'subpixel', 'disparities', 'coarse_disparities', 'min_margin', 'prior_confidence_floor', 'max_coarse_span', 'lr_tolerance', 'source_id')},
              'input_hashes': {'report': cmp.sha256(report_path), 'family': cmp.sha256(Path(args.family)),
                  'diagnostic_source': cmp.sha256(Path(__file__)),
                  'coarse_manifest': cmp.sha256(context['coarse_path']), 'target_manifest': cmp.sha256(context['target_path']),
                  'native_planes': {k: v['sha256'] for k, v in report['files'].items()}}, 'eyes': {}}
    for eye, name in enumerate(('left', 'right')):
        t = truth[name]
        domain = (t['valid'].astype(bool) & t['visible'].astype(bool) & ~t['out_of_range'].astype(bool) &
                  ~t['off_image'].astype(bool) & t['remap_support'].astype(bool) &
                  census_support[eye] & ~t['mixed_bilinear'].astype(bool))
        masks = classify(t['disparity'], domain, actual['prediction.f32'][eye], actual['prediction_validity.u32'][eye],
                         actual['lower.u32'][eye], actual['upper.u32'][eye], actual['raw_disparity.f32'][eye],
                         actual['raw_validity.u32'][eye], actual['validity.u32'][eye])
        result['eyes'][name] = region_counts(masks, {'all': np.ones_like(domain), **{r: t['region_'+r] for r in ('near', 'thin', 'edges')}})
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
