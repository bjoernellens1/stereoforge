#!/usr/bin/env python3
"""Frozen-decision analytic evaluation for the experimental CPU anchor matcher.

Development-only, sparse-point metrics: these are not dense completeness,
physical qualification, GPU timings, or production acceptance.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

import compare_daxa_refinement as fixture
import sparse_stereo_reference as matcher
from stereoforge.refinement import census

ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    'cell_size': 16, 'min_margin': 1, 'disparity_counts': [128, 224],
    'selectors': ['gradient', 'descriptor'],
    'correct_error_px': 1.0, 'min_correct_fraction_all_accepted': 0.95,
    'thin_surface_min_correct': 1, 'near_min_correct': 1, 'edge_min_correct': 1,
    'scope': 'provisional analytic development gates; not promotion thresholds',
}


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _ratio(num, den):
    return num / den if den else None


def evaluate(decisions, truth, census_support, disparity_count):
    """Score frozen decisions with nearest-integer opposite-eye Census support.

    Ground-truth cross coordinates round as floor(value + 0.5), including
    half-pixel ties; the resulting y must equal the rectified reference row.
    This checks one discrete descriptor footprint, not bilinear support.
    """
    output = {}
    for eye_index, eye in enumerate(('left', 'right')):
        t = truth[eye]
        d = np.asarray(t['disparity'])
        valid = t['valid'].astype(bool) & np.isfinite(d)
        in_range = valid & (d >= 0) & (d <= disparity_count - 1)
        visible = t['visible'].astype(bool)
        occluded = t['occluded'].astype(bool)
        off = t['off_image'].astype(bool)
        mixed = t['mixed_bilinear'].astype(bool)
        supported = t['remap_support'].astype(bool) & census_support[eye_index]
        geometric_eligible = valid & visible & ~occluded & ~off & ~mixed & supported & in_range
        u = np.asarray(t['cross_u'], dtype=np.float64)
        v = np.asarray(t['cross_v'], dtype=np.float64)
        rounded_u = np.floor(u + 0.5)
        rounded_v = np.floor(v + 0.5)
        height, width = d.shape
        rows_index = np.arange(height)[:, None]
        counterpart_in_grid = (np.isfinite(rounded_u) & np.isfinite(rounded_v) &
                               (rounded_u >= 0) & (rounded_u < width) &
                               (rounded_v == rows_index))
        opposite_supported = np.zeros(d.shape, bool)
        cy, cx = np.nonzero(counterpart_in_grid)
        opposite_supported[cy, cx] = census_support[1 - eye_index, cy, rounded_u[cy, cx].astype(np.intp)]
        eligible = geometric_eligible & counterpart_in_grid & opposite_supported
        selected = np.zeros(d.shape, bool)
        accepted = np.zeros(d.shape, bool)
        estimate = np.full(d.shape, np.nan)
        for m in decisions:
            if m['eye'] != eye:
                continue
            y, x = m['y'], m['x']
            if selected[y, x]:
                raise ValueError('duplicate selected point')
            selected[y, x] = True
            if m['valid']:
                accepted[y, x] = True
                estimate[y, x] = m['disparity']
        correct = accepted & eligible & (np.abs(estimate - d) <= POLICY['correct_error_px'])
        false_eligible = accepted & eligible & ~correct
        regions = {'all': np.ones(d.shape, bool)}
        regions.update({name: t['region_' + name].astype(bool) for name in ('thin', 'edges', 'near', 'far')})
        rows = {}
        def row(mask):
            e = eligible & mask
            a = accepted & mask
            c = correct & mask
            n = int(e.sum()); sel = int((selected & e).sum()); acc = int((a & e).sum())
            return {'geometric_eligible_pixels': int((geometric_eligible & mask).sum()),
                    'eligible_pixels': n, 'selected_eligible': sel,
                    'accepted_eligible': acc, 'correct': int(c.sum()),
                    'false_accepted_eligible': int((false_eligible & mask).sum()),
                    'accepted_total': int(a.sum()), 'accepted_outside_eligible': int((a & ~eligible).sum()),
                    'correct_over_all_accepted': _ratio(int(c.sum()), int(a.sum())),
                    'correct_over_eligible_pixels': _ratio(int(c.sum()), n),
                    'selected_over_eligible_pixels': _ratio(sel, n)}
        for name, mask in regions.items():
            rows[name] = row(mask)
        surfaces = {}
        for sid in np.unique(t['surface_id'][valid]):
            if sid > 0:
                surface = t['surface_id'] == sid
                surfaces[str(int(sid))] = row(surface)
                surfaces[str(int(sid))]['thin_correct'] = int((correct & surface & regions['thin']).sum())
        rows['surfaces'] = surfaces
        rows['excluded'] = {
            'invalid_truth': int((selected & ~valid).sum()),
            'unsupported': int((selected & ~supported).sum()),
            'mixed': int((selected & mixed).sum()),
            'occluded': int((selected & occluded).sum()),
            'off_image': int((selected & off).sum()),
            'out_of_range': int((selected & valid & ~in_range).sum()),
            'counterpart_outside_integer_grid': int((selected & ~counterpart_in_grid).sum()),
            'opposite_unsupported': int((selected & counterpart_in_grid & ~opposite_supported).sum()),
        }
        # Reasons may overlap; these counters are not additive.
        rows['accepted_excluded'] = {
            'outside_eligible': int((accepted & ~eligible).sum()),
            'supported_but_outside_eligible': int((accepted & supported & ~eligible).sum()),
            'invalid_truth': int((accepted & ~valid).sum()),
            'unsupported': int((accepted & ~supported).sum()),
            'mixed': int((accepted & mixed).sum()),
            'occluded': int((accepted & occluded).sum()),
            'off_image': int((accepted & off).sum()),
            'out_of_range': int((accepted & valid & ~in_range).sum()),
            'counterpart_outside_integer_grid': int((accepted & ~counterpart_in_grid).sum()),
            'opposite_unsupported': int((accepted & counterpart_in_grid & ~opposite_supported).sum()),
        }
        thin_ids = [str(int(i)) for i in np.unique(t['surface_id'][eligible & regions['thin']]) if i > 0]
        accuracy = rows['all']['correct_over_all_accepted']
        rows['provisional_gates'] = {
            'accuracy_all_accepted': accuracy is not None and accuracy >= POLICY['min_correct_fraction_all_accepted'],
            'thin_each_eligible_surface': bool(thin_ids) and all(surfaces[i]['thin_correct'] >= POLICY['thin_surface_min_correct'] for i in thin_ids),
            'near': rows['near']['correct'] >= POLICY['near_min_correct'],
            'edge': rows['edges']['correct'] >= POLICY['edge_min_correct'],
        }
        output[eye] = rows
    return output


def run(coarse, target, family, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Persist immutable settings before accessing either matcher output or truth.
    _write(output / 'policy.json', POLICY)
    context = fixture.load_context(coarse, target)
    image = context['images'].astype(np.uint8)
    support = context['support'].astype(bool)
    source = str(context['source_id'])
    hashes = {'coarse_manifest': fixture.sha256(Path(coarse)),
              'target_manifest': fixture.sha256(Path(target)),
              'family_manifest': fixture.sha256(Path(family)),
              'matcher_source': fixture.sha256(ROOT / 'benchmarks/sparse_stereo_reference.py'),
              'evaluator_source': fixture.sha256(Path(__file__)),
              'refinement_source': fixture.sha256(ROOT / 'python/stereoforge/refinement.py'),
              'luma': _hash_bytes(image.tobytes()), 'support': _hash_bytes(support.tobytes())}
    runs = {}
    for selection in POLICY['selectors']:
        runs[selection] = {}
        for count in POLICY['disparity_counts']:
            result = matcher.sparse_stereo(image[0], image[1], support[0], support[1], source_id=source,
                                           cell_size=POLICY['cell_size'], disparities=count,
                                           min_margin=POLICY['min_margin'], selection=selection)
            runs[selection][str(count)] = {'matches': [asdict(m) for m in result.matches],
                                           'forward_candidates': result.forward_candidates,
                                           'reverse_candidates': result.reverse_candidates}
    frozen = {'policy': POLICY, 'hashes': hashes, 'source_id': source, 'runs': runs}
    _write(output / 'decisions.json', frozen)
    decision_hash = fixture.sha256(output / 'decisions.json')
    # Analytic truth is deliberately loaded only after the decision file is closed.
    truth = fixture.load_truth(family, context)
    cs = np.stack([census(image[i], support[i]).support for i in range(2)])
    report = {'schema': 'stereoforge.sparse_anchor_evaluation.v1',
              'scope': 'analytic synthetic development only; sparse points, not dense completeness',
              'counterpart_support': 'nearest integer floor(coordinate + 0.5), same rectified row, opposite Census support; not bilinear support',
              'decision_sha256': decision_hash, 'policy_sha256': fixture.sha256(output / 'policy.json'),
              'hashes': hashes, 'source_id': source, 'policy': POLICY, 'profiles': {}}
    for selection in POLICY['selectors']:
        report['profiles'][selection] = {}
        for count in POLICY['disparity_counts']:
            r = runs[selection][str(count)]
            report['profiles'][selection][str(count)] = {
                'selection': selection, 'forward_candidates': r['forward_candidates'],
                'reverse_candidates': r['reverse_candidates'],
                'eyes': evaluate(r['matches'], truth, cs, count)}
    _write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('coarse', 'target', 'family', 'output'):
        parser.add_argument('--' + arg, required=True)
    a = parser.parse_args()
    run(a.coarse, a.target, a.family, a.output)


if __name__ == '__main__':
    main()
