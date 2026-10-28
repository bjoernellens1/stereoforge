#!/usr/bin/env python3
"""CPU-only partial coarse-prior experiment; truth labels evaluation only."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from stereoforge import refinement as oracle


def partial_prior(disparity, confidence, validity):
    """At least two reliable taps, half the weight, and no border extrapolation."""
    d = np.asarray(disparity, np.float32)
    c = np.asarray(confidence, np.float32)
    v = np.asarray(validity, bool)
    if d.ndim != 2 or c.shape != d.shape or v.shape != d.shape:
        raise ValueError('coarse plane shape mismatch')
    ch, cw = d.shape
    h, w = 2 * ch, 2 * cw
    yy, xx = np.indices((h, w))
    x0, y0 = (2 * xx - 1) // 4, (2 * yy - 1) // 4
    wx = ((2 * xx - 1) % 4).astype(np.float32) * np.float32(.25)
    wy = ((2 * yy - 1) % 4).astype(np.float32) * np.float32(.25)
    total = np.zeros((h, w), np.float32)
    weights = np.zeros_like(total)
    count = np.zeros((h, w), np.uint8)
    low = np.full((h, w), np.inf, np.float32)
    high = np.full((h, w), -np.inf, np.float32)
    inside_all = np.ones((h, w), bool)
    for dy in (0, 1):
        for dx in (0, 1):
            x, y = x0 + dx, y0 + dy
            inside = (x >= 0) & (x < cw) & (y >= 0) & (y < ch)
            inside_all &= inside
            sx, sy = np.clip(x, 0, cw - 1), np.clip(y, 0, ch - 1)
            value, conf = d[sy, sx], c[sy, sx]
            good = inside & v[sy, sx] & np.isfinite(value) & (value >= 0) & (value <= 63.5) & np.isfinite(conf) & (conf >= .05) & (conf <= 1)
            weight = (wx if dx else 1 - wx) * (wy if dy else 1 - wy)
            weights += np.where(good, weight, 0)
            total += np.where(good, value, 0) * weight
            count += good
            low = np.minimum(low, np.where(good, value, np.inf))
            high = np.maximum(high, np.where(good, value, -np.inf))
    ok = inside_all & (count >= 2) & (weights >= .5) & ((high - low) <= 2)
    estimate = np.zeros_like(total)
    np.divide(total, weights, out=estimate, where=ok)
    estimate *= np.float32(2)
    center = np.floor(estimate + np.float32(.5)).astype(np.int32)
    lower = np.where(ok, np.maximum(0, center - 2), 0).astype(np.int32)
    upper = np.where(ok, np.minimum(127, center + 2), -1).astype(np.int32)
    return oracle.Prediction(np.where(ok, estimate, oracle.CANONICAL_NAN).astype(np.float32), ok, np.zeros((h, w), np.uint32), lower, upper)


def merge_matches(base_d, base_v, base_prior, candidate_d, candidate_v):
    take = ~np.asarray(base_prior, bool)
    return np.where(take, candidate_d, base_d).astype(np.float32), np.where(take, candidate_v, base_v).astype(bool)


def validate_controls(actual, cpu_support):
    for name in ('census_validity.u32', 'raw_validity.u32', 'validity.u32'):
        if np.any(actual[name] > 1):
            raise ValueError('nonbinary native mask: ' + name)
    if not np.array_equal(actual['census_validity.u32'].astype(bool), cpu_support):
        raise ValueError('CPU/native Census support mismatch')


def validate_profile(report, expected, frozen):
    from compare_daxa_refinement import _same
    for key, value in expected.items():
        _same(report.get(key), value, 'native profile ' + key)
    for key, value in frozen['report_identity'].items():
        _same(report.get(key), value, 'native frozen identity ' + key)
    for key, value in frozen['device_identity'].items():
        _same(report.get('device', {}).get(key), value, 'native frozen device ' + key)


def _hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    import compare_daxa_refinement as cmp
    context = cmp.load_context(args.coarse_manifest, args.target_manifest)
    report_path = Path(args.report)
    report = cmp.strict_json(report_path.read_text())
    h, w = context['images'].shape[1:]
    shape = (2, h, w)
    if _hash(args.freeze) != args.freeze_sha256:
        raise ValueError('freeze hash mismatch')
    frozen = cmp.strict_json(Path(args.freeze).read_text())
    comparison = cmp.strict_json(Path(args.comparison).read_text())
    if (frozen.get('schema') != 'stereoforge.refinement_freeze.v1'
            or comparison.get('status') != 'conditional_implementation_agreement'
            or comparison.get('freeze_sha256') != args.freeze_sha256
            or comparison.get('native_report_sha256') != _hash(report_path)):
        raise ValueError('native frozen comparison binding mismatch')
    settings = oracle.Settings(radius=2, subpixel=False, recovery=False)
    validate_profile(report, cmp.expected_report(context, settings, True), frozen)
    actual = cmp._read_planes(report['files'], cmp.PLANES, shape, args.dump_root)
    coarse = cmp._read_planes(report['coarse_files'], cmp.COARSE_PLANES, (2, h // 2, w // 2), args.dump_root, True)
    census = [oracle.census(context['images'][i], context['support'][i]) for i in (0, 1)]
    validate_controls(actual, np.stack([c.support for c in census]))
    raw, validity, work, selected = [], [], [], []
    for eye in (0, 1):
        prior = oracle.predict_prior(coarse['disparity.f32'][eye], coarse['confidence.f32'][eye], coarse['validity.u32'][eye], (h, w), settings)
        match = oracle.match_eye(census[eye], census[1-eye], prior, eye, settings)
        native_prior = actual['prediction_validity.u32'][eye].astype(bool)
        for label, a, b in [('prior validity', prior.valid, native_prior), ('raw validity', match.valid, actual['raw_validity.u32'][eye] != 0), ('winner', match.integer_winner.astype(np.uint32), actual['winner.u32'][eye]), ('work', match.candidates, actual['evaluated_candidates.u32'][eye])]:
            if not np.array_equal(a, b):
                raise ValueError('CPU/native control mismatch: ' + label)
        if not np.array_equal(match.disparity[match.valid], actual['raw_disparity.f32'][eye][match.valid]):
            raise ValueError('CPU/native raw disparity mismatch')
        del match
        proposal = partial_prior(coarse['disparity.f32'][eye], coarse['confidence.f32'][eye], coarse['validity.u32'][eye])
        region = ~native_prior & proposal.valid
        match = oracle.match_eye(census[eye], census[1-eye], proposal, eye, settings, region=region)
        merged, valid = merge_matches(actual['raw_disparity.f32'][eye], actual['raw_validity.u32'][eye] != 0, native_prior, match.disparity, match.valid)
        raw.append(merged)
        validity.append(valid)
        work.append(int(match.candidates.sum(dtype=np.uint64)))
        selected.append(int(region.sum()))
        del match
    raw, validity = np.stack(raw), np.stack(validity)
    lr = oracle.lr_consistency(raw, validity, settings.lr_tolerance)
    control_lr = oracle.lr_consistency(actual['raw_disparity.f32'], actual['raw_validity.u32'] != 0, settings.lr_tolerance)
    if not np.array_equal(control_lr, actual['validity.u32'] != 0):
        raise ValueError('CPU/native control LR mismatch')
    geometry = context['geometry']
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        z = np.float32(geometry.fx * geometry.baseline_m) / (raw - geometry.cx_offset)
    metric = lr & np.isfinite(z) & (z > 0) & (raw > 0) & ((raw - geometry.cx_offset) > 0)
    candidate = {'raw_validity.u32': validity, 'metric_validity.u32': metric, 'validity.u32': lr, 'raw_disparity.f32': raw, 'depth.f32': np.where(metric, z, np.nan)}
    # Loading truth here ensures that it cannot influence candidate selection.
    truth = cmp.load_truth(args.family, context)
    support = actual['census_validity.u32'] != 0
    baseline = cmp.evaluate_quality(actual, truth, support)
    result = cmp.evaluate_quality(candidate, truth, support)
    checks = {}
    for eye in ('left', 'right'):
        checks[eye] = {}
        for region, minimum in [('thin', .05), ('all', -.01), ('near', -.01)]:
            before = baseline[eye][region]['correct_metric_completeness']
            after = result[eye][region]['correct_metric_completeness']
            checks[eye][region] = bool(before is not None and after is not None and after - before >= minimum)
        before = baseline[eye]['all']['lr_false_accepted']
        after = result[eye]['all']['lr_false_accepted']
        checks[eye]['false_accepts'] = bool(after - before <= 200 and after <= before * 1.10)
    data = {'schema': 'stereoforge.fast60_partial_prior_experiment.v1', 'scope': 'offline development only; no GPU or physical qualification',
            'limitations': ['no predefined region-specific false-geometry or edge gates; this harness does not permit promotion',
                            'freeze and comparison provide self-consistency, not independent attestation of provenance',
                            'candidate planes are not persisted; output contains aggregate results only'],
            'settings': {'radius': 2, 'min_confidence': .05, 'min_taps': 2, 'min_weight': .5, 'max_span': 2, 'lr_tolerance': 1, 'subpixel': False},
            'input_hashes': {'report': _hash(report_path), 'freeze': _hash(args.freeze), 'comparison': _hash(args.comparison), 'family': _hash(args.family), 'coarse_manifest': _hash(args.coarse_manifest), 'target_manifest': _hash(args.target_manifest), 'script': _hash(__file__), 'native_planes': {k: v['sha256'] for k, v in report['files'].items()}, 'coarse_planes': {k: v['sha256'] for k, v in report['coarse_files'].items()}},
            'false_accept_definition': 'lr_false_accepted counts all LR accepted pixels in the region, including outside clean truth support; lr_accepted_total is its denominator, not a rate on clean support',
            'cpu_native_control': 'exact prior and raw validity, integer winner, candidate work, raw integer disparity and LR masks',
            'candidate_selected_pixels': selected, 'candidate_supported_work': work, 'baseline': baseline, 'candidate': result, 'checks': checks,
            'accepted': all(all(v.values()) for v in checks.values())}
    with Path(args.output).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'checks': checks, 'accepted': data['accepted'], 'selected': selected, 'work': work}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('coarse-manifest', 'target-manifest', 'family', 'report', 'dump-root', 'freeze', 'freeze-sha256', 'comparison', 'output'):
        parser.add_argument('--' + arg, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
