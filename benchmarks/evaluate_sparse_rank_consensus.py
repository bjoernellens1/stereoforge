#!/usr/bin/env python3
"""Paired, frozen-decision comparison of sparse Census and rank-consensus costs on development data.

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
import sparse_rank_consensus as rank
import evaluate_sparse_vertical as vertical_scores
from stereoforge.refinement import census

ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    'cell_size': 16, 'disparity_count': 224, 'min_margin': 1,
    'selectors': ['combined', 'combined_edge'],
    'matchers': ['single', 'rank'],
    'correct_error_px': 1.0,
    'min_correct_all_accepted_numerator': 95,
    'min_correct_all_accepted_denominator': 100,
    'thin_surface_min_correct': 2, 'near_min_correct': 1, 'edge_min_correct': 1,
    'thin_false_numerator': 5, 'thin_false_denominator': 100,
    'thin_surface_accuracy_numerator': 90,
    'thin_surface_accuracy_denominator': 100,
    'scope': 'exploratory previously inspected development scene; not held-out qualification',
}


def validate_policy():
    if (POLICY['correct_error_px'] != base.POLICY['correct_error_px'] or
            POLICY['min_correct_all_accepted_numerator'] / POLICY['min_correct_all_accepted_denominator'] != base.POLICY['min_correct_fraction_all_accepted']):
        raise ValueError('scoring policy differs from base evaluator')
    for key in ('correct_error_px', 'min_correct_all_accepted_numerator', 'min_correct_all_accepted_denominator',
                'thin_surface_min_correct', 'thin_surface_accuracy_numerator', 'thin_surface_accuracy_denominator'):
        if POLICY[key] != vertical_scores.POLICY[key]:
            raise ValueError('thin scorer policy differs')
    if any(POLICY[k] != base.POLICY[k] for k in ('near_min_correct', 'edge_min_correct')):
        raise ValueError('scoring policy differs')
    if (POLICY['thin_false_numerator'], POLICY['thin_false_denominator']) != (5, 100):
        raise ValueError('thin false-accept policy differs')
    if POLICY['selectors'] != ['combined', 'combined_edge'] or POLICY['matchers'] != ['single', 'rank']:
        raise ValueError('matcher or selector policy differs')
    if (POLICY['cell_size'], POLICY['disparity_count'], POLICY['min_margin']) != (16, 224, 1):
        raise ValueError('matcher settings differ from frozen experiment')


def _passes(correct, accepted, num, den):
    return accepted > 0 and correct * den >= accepted * num


def _thin_passes(row):
    # Treat unscorable accepts as failures for the gate; they are not known false.
    return (row['accepted_total'] > 0 and
            (row['accepted_total'] - row['correct']) * POLICY['thin_false_denominator'] <=
            row['accepted_total'] * POLICY['thin_false_numerator'])


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
        'rank_source': ROOT / 'benchmarks/sparse_rank_consensus.py',
        'thin_scorer_source': ROOT / 'benchmarks/evaluate_sparse_vertical.py',
        'scoring_source': ROOT / 'benchmarks/evaluate_sparse_anchors.py',
        'evaluator_source': Path(__file__), 'refinement_source': ROOT / 'python/stereoforge/refinement.py',
    }.items()}
    hashes.update(luma=base._hash_bytes(image.tobytes()), support=base._hash_bytes(support.tobytes()))
    association = {'metadata': context.get('metadata'),
                   'capture': context.get('target', {}).get('capture'),
                   'calibration': context.get('target', {}).get('calibration'),
                   'rectified': context.get('target', {}).get('rectified')}
    hashes['association'] = base._hash_bytes(json.dumps(association, sort_keys=True, default=str).encode())
    hashes['source_id'] = base._hash_bytes(source.encode())
    runs = {}
    for selection in POLICY['selectors']:
        runs[selection] = {}
        common = dict(source_id=source, cell_size=POLICY['cell_size'],
                      disparities=POLICY['disparity_count'], min_margin=POLICY['min_margin'],
                      selection=selection)
        baseline = single.sparse_stereo(image[0], image[1], support[0], support[1], **common)
        consensus = rank.sparse_stereo(image[0], image[1], support[0], support[1],
                                       baseline=baseline, **common)
        for name, result in [('single', baseline), ('rank', consensus)]:
            if result.source_id != source:
                raise ValueError('matcher source identity differs')
            matches = [asdict(m) for m in result.matches]
            fw, rev = result.forward_candidates, result.reverse_candidates
            if min(fw, rev) < 0:
                raise ValueError('negative candidate count')
            row = {'matches': matches, 'forward_candidates': fw, 'reverse_candidates': rev,
                   'selected_by_eye': {eye: sum(m['eye'] == eye for m in matches) for eye in ('left', 'right')},
                   'accepted_by_eye': {eye: sum(m['eye'] == eye and m['valid'] for m in matches) for eye in ('left', 'right')},
                   'reasons_by_eye': {eye: dict(sorted(Counter(m['reason'] for m in matches if m['eye'] == eye).items())) for eye in ('left', 'right')}}
            if name == 'rank':
                if (fw, rev) != (baseline.forward_candidates, baseline.reverse_candidates):
                    raise ValueError('baseline candidate accounting differs')
                rf, rr = result.rank_forward_candidates, result.rank_reverse_candidates
                attempts = result.rank_attempted_points
                pixels, samples = result.rank_descriptor_pixels, result.rank_descriptor_samples
                if min(rf, rr, attempts, pixels, samples) < 0 or samples != 8 * pixels:
                    raise ValueError('invalid rank work accounting')
                if attempts != sum(m.valid for m in baseline.matches):
                    raise ValueError('rank attempted count differs from baseline valid points')
                row.update(rank_forward_candidates=rf, rank_reverse_candidates=rr,
                           rank_attempted_points=attempts, rank_descriptor_pixels=pixels,
                           rank_descriptor_samples=samples,
                           total_forward_candidates=fw + rf, total_reverse_candidates=rev + rr)
            runs[selection][name] = row
        points = lambda name: [(m['eye'], m['y'], m['x']) for m in runs[selection][name]['matches']]
        if points('single') != points('rank'):
            raise ValueError('paired matchers did not use identical selected points')
    base._write(output / 'decisions.json', {'policy': POLICY, 'hashes': hashes, 'source_id': source,
                                          'runs': runs, 'numpy_version': np.__version__, 'association': association})
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
            thin = vertical_scores.thin_surface_scores(r['matches'], truth, cs, POLICY['disparity_count'])
            for eye in ('left', 'right'):
                all_row = eyes[eye]['all']
                eyes[eye]['paired_provisional_gates'] = {
                    'accuracy_all_accepted': _passes(all_row['correct'], all_row['accepted_total'],
                                                     POLICY['min_correct_all_accepted_numerator'], POLICY['min_correct_all_accepted_denominator']),
                    'thin_each_eligible_surface': thin[eye]['passes'],
                    'thin_unconfirmed_or_false_rate': _thin_passes(eyes[eye]['thin']),
                    'near': eyes[eye]['near']['correct'] >= POLICY['near_min_correct'],
                    'edge': eyes[eye]['edges']['correct'] >= POLICY['edge_min_correct'],
                }
                eyes[eye]['thin_surface_scores'] = thin[eye]['surfaces']
            profiles[selection][name] = {**{k: v for k, v in r.items() if k != 'matches'}, 'eyes': eyes}
    report = {'schema': 'stereoforge.sparse_rank_consensus_evaluation.v1', 'scope': POLICY['scope'],
              'work_scope': 'baseline counts supported candidate upper bounds; rank counts actual supported Hamming candidate evaluations; total counts baseline once plus rank; selection and rank preprocessing are excluded from candidate totals and reported separately, no GPU timings',
              'counterpart_support': 'nearest integer floor(coordinate + 0.5), opposite Census support',
              'thin_gate_scope': 'all accepted minus confirmed correct over all accepted; includes outside-eligibility unknowns, which are not confirmed false; known false_accepted_eligible reported separately',
              'decision_sha256': decision_hash, 'policy_sha256': policy_hash, 'hashes': hashes,
              'source_id': source, 'numpy_version': np.__version__, 'association': association, 'policy': POLICY, 'profiles': profiles}
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
