#!/usr/bin/env python3
"""Offline estimate of dense tile recovery work from observed matching masks."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def selection(prediction_valid, raw_valid, provisional_lr_valid, census_support=None):
    arrays = [np.asarray(a, dtype=bool) for a in (prediction_valid, raw_valid, provisional_lr_valid)]
    if arrays[0].ndim != 2 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('selection planes must share two-dimensional shape')
    risks = ~arrays[0] | ~arrays[1] | ~arrays[2]
    if census_support is not None:
        support = np.asarray(census_support, dtype=bool)
        if support.shape != risks.shape:
            raise ValueError('Census support shape differs')
        risks &= support
    return risks


def estimate(mask, tile, halo, minimum_fraction=0):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not all(mask.shape) or tile <= 0 or halo < 0 or not 0 <= minimum_fraction <= 1:
        raise ValueError('invalid mask or tile geometry')
    h, w = mask.shape
    tiles = visits = selected_risk = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            risky = int(mask[y:y2, x:x2].sum())
            if risky and risky >= minimum_fraction * ((y2 - y) * (x2 - x)):
                tiles += 1
                selected_risk += risky
                visits += (min(h, y2 + halo) - max(0, y - halo)) * (min(w, x2 + halo) - max(0, x - halo))
    total_risk = int(mask.sum())
    return dict(total_risk_pixels=total_risk,
                selected_risk_pixels=selected_risk, unselected_risk_pixels=total_risk - selected_risk,
                risk_coverage=selected_risk / total_risk if total_risk else None,
                dispatch_tiles=tiles,
                expanded_pixel_visits=visits, full_d128_candidates=visits * 128,
                four_path_state_updates=visits * 128 * 4)


def main():
    import compare_daxa_refinement as cmp
    from fast60_prior_diagnostic import validate_report, validate_wire_masks, validate_wire_values, verified_census_support
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coarse-manifest', 'target-manifest', 'report', 'dump-root', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    context = cmp.load_context(args.coarse_manifest, args.target_manifest)
    report_path = Path(args.report)
    report = cmp.strict_json(report_path.read_text())
    settings = cmp.oracle.Settings(radius=2, subpixel=False, recovery=False)
    validate_report(cmp, report, context, settings)
    h, w = context['images'].shape[1:]
    planes = cmp._read_planes(report['files'], cmp.PLANES, (2, h, w), args.dump_root)
    validate_wire_masks({name: planes[name] for name in ('prediction_validity.u32', 'raw_validity.u32', 'validity.u32', 'census_validity.u32')})
    validate_wire_values(planes, settings.disparities, settings.radius)
    raw_valid = planes['raw_validity.u32'] != 0
    provisional = cmp.oracle.lr_consistency(planes['raw_disparity.f32'], raw_valid, settings.lr_tolerance)
    support = verified_census_support(context['images'], context['support'],
                                     planes['census_validity.u32'], cmp.oracle)
    policies = {}
    for name, gated in [('unrestricted', False), ('census_supported', True)]:
        policies[name] = [selection(planes['prediction_validity.u32'][eye], raw_valid[eye], provisional[eye],
                                    support[eye] if gated else None) for eye in (0, 1)]
    full_pixels = 2 * h * w
    policy_results = {}
    for name, masks in policies.items():
        rows = []
        thresholds = (('at_least_one', 0),) if name == 'unrestricted' else (
            ('at_least_one', 0), ('10_percent', .10), ('25_percent', .25), ('50_percent', .50))
        for label, fraction in thresholds:
            for tile in (16, 32, 64):
                for halo in (0, 4, 8):
                    left, right = [estimate(mask, tile, halo, fraction) for mask in masks]
                    combined = {k: left[k] + right[k] for k in left if k != 'risk_coverage'}
                    total = combined['total_risk_pixels']
                    combined['risk_coverage'] = combined['selected_risk_pixels'] / total if total else None
                    combined['fraction_of_full_frame_two_eye_d128_p4'] = combined['four_path_state_updates'] / (full_pixels * 128 * 4)
                    rows.append(dict(threshold=label, minimum_fraction=fraction, tile=tile, halo=halo, left=left, right=right, combined=combined))
        policy_results[name] = rows
    result = dict(schema='stereoforge.fast60_tile_work.v1',
                  scope='offline diagnostic work/risk concentration estimate; thresholds are not accepted fallback policies; no truth selection, GPU timing, accuracy, or qualified performance',
                  selection='missing prior OR invalid raw match OR provisional LR failure; independent eye masks; supported policy additionally requires independent full-resolution Census support',
                  provisional_lr='recomputed from both native raw disparity and validity planes',
                  accounting='expanded tile rectangles clipped to image, overlap counted per tile; full preprocessing and LR excluded',
                  width=w, height=h, full_frame_two_eye_pixels=full_pixels,
                  full_frame_two_eye_d128_candidates=full_pixels * 128,
                  full_frame_two_eye_d128_p4_state_updates=full_pixels * 128 * 4,
                  input_hashes=dict(report=cmp.sha256(report_path), coarse_manifest=cmp.sha256(context['coarse_path']),
                                    target_manifest=cmp.sha256(context['target_path']),
                                    script=cmp.sha256(Path(__file__)),
                                    native_planes={k: v['sha256'] for k, v in report['files'].items()}),
                  policies=policy_results)
    with Path(args.output).open('x') as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


if __name__ == '__main__':
    main()
