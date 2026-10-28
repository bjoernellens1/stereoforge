#!/usr/bin/env python3
"""Paired, frozen-decision comparison of sparse Census costs on development data.

CPU exploratory evidence only; excludes selection work and does not measure GPU time.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import compare_daxa_refinement as fixture
import evaluate_sparse_anchors as base
import sparse_stereo_reference as single
import sparse_vertical_reference as vertical
from stereoforge.refinement import census

ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    'cell_size': 16, 'disparity_count': 224, 'min_margin': 1,
    'selectors': ['combined', 'combined_edge'],
    'matchers': ['single', 'vertical3'],
    'correct_error_px': 1.0,
    'min_correct_all_accepted_numerator': 95,
    'min_correct_all_accepted_denominator': 100,
    'thin_surface_min_correct': 2,
    'thin_surface_accuracy_numerator': 90,
    'thin_surface_accuracy_denominator': 100,
    'scope': 'exploratory previously inspected development scene; not held-out qualification',
}


def validate_policy():
    if (POLICY['correct_error_px'] != base.POLICY['correct_error_px'] or
            POLICY['min_correct_all_accepted_numerator'] / POLICY['min_correct_all_accepted_denominator'] != base.POLICY['min_correct_fraction_all_accepted']):
        raise ValueError('scoring policy differs from base evaluator')
    if POLICY['selectors'] != ['combined', 'combined_edge'] or POLICY['matchers'] != ['single', 'vertical3']:
        raise ValueError('matcher or selector policy differs')
    if (POLICY['cell_size'], POLICY['disparity_count'], POLICY['min_margin']) != (16, 224, 1):
        raise ValueError('matcher settings differ from frozen experiment')


def _passes(correct, accepted, num, den):
    return accepted > 0 and correct * den >= accepted * num


def thin_surface_scores(matches, truth, support, disparity_count):
    """Score each thin portion separately with the base eligibility rules."""
    result = {}
    for eye in ('left', 'right'):
        t = truth[eye]
        ids = np.unique(np.asarray(t['surface_id'])[np.asarray(t['valid'], bool) & np.asarray(t['region_thin'], bool)])
        rows = {}
        for sid in ids:
            if sid <= 0:
                continue
            mask = (np.asarray(t['surface_id']) == sid) & np.asarray(t['region_thin'], bool)
            restricted = {name: dict(value) for name, value in truth.items()}
            restricted[eye]['valid'] = np.asarray(t['valid'], bool) & mask
            subset = [m for m in matches if m['eye'] == eye and mask[m['y'], m['x']]]
            score = base.evaluate(subset, restricted, support, disparity_count)[eye]['all']
            if not score['eligible_pixels']:
                continue
            correct, accepted = score['correct'], score['accepted_total']
            rows[str(int(sid))] = {
                'eligible_pixels': score['eligible_pixels'], 'accepted': accepted,
                'correct': correct, 'false_accepted_eligible': score['false_accepted_eligible'],
                'accepted_outside_eligible': score['accepted_outside_eligible'],
                'correct_over_all_accepted': base._ratio(correct, accepted),
                'passes': correct >= POLICY['thin_surface_min_correct'] and _passes(
                    correct, accepted, POLICY['thin_surface_accuracy_numerator'], POLICY['thin_surface_accuracy_denominator']),
            }
        result[eye] = {'surfaces': rows, 'passes': bool(rows) and all(row['passes'] for row in rows.values())}
    return result


def run(coarse, target, family, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    validate_policy()
    base._write(output / 'policy.json', POLICY)
    context = fixture.load_context(coarse, target)
    image = context['images'].astype(np.uint8)
    support = context['support'].astype(bool)
    source = str(context['source_id'])
    hashes = {name: fixture.sha256(path) for name, path in {
        'coarse_manifest': Path(coarse), 'target_manifest': Path(target),
        'family_manifest': Path(family), 'fixture_loader_source': ROOT / 'benchmarks/compare_daxa_refinement.py',
        'single_source': ROOT / 'benchmarks/sparse_stereo_reference.py',
        'vertical_source': ROOT / 'benchmarks/sparse_vertical_reference.py',
        'scoring_source': ROOT / 'benchmarks/evaluate_sparse_anchors.py',
        'evaluator_source': Path(__file__), 'refinement_source': ROOT / 'python/stereoforge/refinement.py',
    }.items()}
    hashes.update(luma=base._hash_bytes(image.tobytes()), support=base._hash_bytes(support.tobytes()))
    runs = {}
    for selection in POLICY['selectors']:
        runs[selection] = {}
        for name, engine in [('single', single), ('vertical3', vertical)]:
            result = engine.sparse_stereo(image[0], image[1], support[0], support[1], source_id=source,
                                          cell_size=POLICY['cell_size'], disparities=POLICY['disparity_count'],
                                          min_margin=POLICY['min_margin'], selection=selection)
            matches = [asdict(m) for m in result.matches]
            fw, rev = result.forward_candidates, result.reverse_candidates
            fops = fw if name == 'single' else result.forward_hamming_ops
            rops = rev if name == 'single' else result.reverse_hamming_ops
            if (fops, rops) != ((fw, rev) if name == 'single' else (3 * fw, 3 * rev)):
                raise ValueError('descriptor comparison count differs from policy')
            runs[selection][name] = {
                'matches': matches, 'forward_candidates': fw, 'reverse_candidates': rev,
                'forward_hamming_upper_bound': fops, 'reverse_hamming_upper_bound': rops,
                'selected_by_eye': {eye: sum(m['eye'] == eye for m in matches) for eye in ('left', 'right')},
                'accepted_by_eye': {eye: sum(m['eye'] == eye and m['valid'] for m in matches) for eye in ('left', 'right')},
                'reasons_by_eye': {eye: dict(sorted(Counter(m['reason'] for m in matches if m['eye'] == eye).items())) for eye in ('left', 'right')},
            }
        points = lambda name: [(m['eye'], m['y'], m['x']) for m in runs[selection][name]['matches']]
        if points('single') != points('vertical3'):
            raise ValueError('paired matchers did not use identical selected points')
    base._write(output / 'decisions.json', {'policy': POLICY, 'hashes': hashes, 'source_id': source,
                                          'runs': runs, 'numpy_version': np.__version__})
    decision_hash = fixture.sha256(output / 'decisions.json')
    policy_hash = fixture.sha256(output / 'policy.json')
    validate_policy()
    if json.loads((output / 'policy.json').read_text()) != POLICY:
        raise ValueError('persisted policy differs')
    truth = fixture.load_truth(family, context)
    cs = np.stack([census(image[i], support[i]).support for i in range(2)])
    profiles = {}
    for selection, pair in runs.items():
        profiles[selection] = {}
        for name, r in pair.items():
            eyes = base.evaluate(r['matches'], truth, cs, POLICY['disparity_count'])
            thin = thin_surface_scores(r['matches'], truth, cs, POLICY['disparity_count'])
            for eye in ('left', 'right'):
                all_row = eyes[eye]['all']
                eyes[eye]['paired_provisional_gates'] = {
                    'accuracy_all_accepted': _passes(all_row['correct'], all_row['accepted_total'],
                                                     POLICY['min_correct_all_accepted_numerator'], POLICY['min_correct_all_accepted_denominator']),
                    'thin_each_eligible_surface': thin[eye]['passes'],
                }
                eyes[eye]['thin_surface_scores'] = thin[eye]['surfaces']
            profiles[selection][name] = {**{k: v for k, v in r.items() if k != 'matches'}, 'eyes': eyes}
    report = {'schema': 'stereoforge.sparse_vertical_evaluation.v1', 'scope': POLICY['scope'],
              'work_scope': 'Hamming upper bound: single skips Hamming when fewer than two candidates; vertical3 currently evaluates exactly three per candidate; selection and other work excluded',
              'counterpart_support': 'nearest integer floor(coordinate + 0.5), opposite Census support',
              'decision_sha256': decision_hash, 'policy_sha256': policy_hash, 'hashes': hashes,
              'source_id': source, 'numpy_version': np.__version__, 'policy': POLICY, 'profiles': profiles}
    base._write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('coarse', 'target', 'family', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.coarse, args.target, args.family, args.output)


if __name__ == '__main__':
    main()
