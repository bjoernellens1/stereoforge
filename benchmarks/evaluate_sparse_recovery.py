#!/usr/bin/env python3
"""Compare sparse-anchor selectors on previously inspected development truth.

Offline CPU exploration only. Counts describe sparse selections, not dense
completeness, GPU latency, held-out qualification, or physical performance.
"""
import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

import compare_daxa_refinement as fixture
import evaluate_sparse_anchors as base
import sparse_stereo_reference as matcher
from stereoforge.refinement import census

ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    'cell_size': 16,
    'edge_subdivisions': 4,
    'edge_subcell_size_for_full_cells': 4,
    'min_margin': 1,
    'disparity_count': 224,
    'selectors': ['gradient', 'descriptor', 'combined', 'combined_edge'],
    'correct_error_px': base.POLICY['correct_error_px'],
    'min_correct_fraction_all_accepted': base.POLICY['min_correct_fraction_all_accepted'],
    'thin_surface_min_correct': base.POLICY['thin_surface_min_correct'],
    'near_min_correct': base.POLICY['near_min_correct'],
    'edge_min_correct': base.POLICY['edge_min_correct'],
    'scope': 'exploratory development gates on previously inspected data, not blind preregistration or promotion',
}


def validate_policy():
    """Reject scorer or hardcoded selector settings that differ from policy."""
    for key in ('correct_error_px', 'min_correct_fraction_all_accepted',
                'thin_surface_min_correct', 'near_min_correct', 'edge_min_correct'):
        if POLICY[key] != base.POLICY[key]:
            raise ValueError('scoring policy differs: ' + key)
    if (POLICY['edge_subdivisions'] != 4 or
            POLICY['edge_subcell_size_for_full_cells'] != POLICY['cell_size'] // 4 or
            POLICY['cell_size'] % 4):
        raise ValueError('edge subdivision policy differs from matcher')


def run(coarse, target, family, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    validate_policy()
    # Fix every selector and threshold before loading images or running a matcher.
    base._write(output / 'policy.json', POLICY)
    context = fixture.load_context(coarse, target)
    image = context['images'].astype(np.uint8)
    support = context['support'].astype(bool)
    source = str(context['source_id'])
    hashes = {
        'coarse_manifest': fixture.sha256(Path(coarse)),
        'target_manifest': fixture.sha256(Path(target)),
        'family_manifest': fixture.sha256(Path(family)),
        'fixture_loader_source': fixture.sha256(ROOT / 'benchmarks/compare_daxa_refinement.py'),
        'matcher_source': fixture.sha256(ROOT / 'benchmarks/sparse_stereo_reference.py'),
        'scoring_source': fixture.sha256(ROOT / 'benchmarks/evaluate_sparse_anchors.py'),
        'evaluator_source': fixture.sha256(Path(__file__)),
        'refinement_source': fixture.sha256(ROOT / 'python/stereoforge/refinement.py'),
        'luma': base._hash_bytes(image.tobytes()),
        'support': base._hash_bytes(support.tobytes()),
    }
    runs = {}
    for selection in POLICY['selectors']:
        result = matcher.sparse_stereo(
            image[0], image[1], support[0], support[1], source_id=source,
            cell_size=POLICY['cell_size'], disparities=POLICY['disparity_count'],
            min_margin=POLICY['min_margin'], selection=selection)
        matches = [asdict(m) for m in result.matches]
        runs[selection] = {
            'matches': matches,
            'forward_candidates': result.forward_candidates,
            'reverse_candidates': result.reverse_candidates,
            'selected_by_eye': {eye: sum(m['eye'] == eye for m in matches) for eye in ('left', 'right')},
            'accepted_by_eye': {eye: sum(m['eye'] == eye and m['valid'] for m in matches) for eye in ('left', 'right')},
            'reasons_by_eye': {eye: dict(sorted(Counter(m['reason'] for m in matches if m['eye'] == eye).items()))
                               for eye in ('left', 'right')},
        }
    base._write(output / 'decisions.json', {
        'policy': POLICY, 'hashes': hashes, 'source_id': source, 'runs': runs,
        'numpy_version': np.__version__,
    })
    decision_hash = fixture.sha256(output / 'decisions.json')
    policy_hash = fixture.sha256(output / 'policy.json')
    # Freeze/close all selections before any truth values are read.
    validate_policy()
    if json.loads((output / 'policy.json').read_text()) != POLICY:
        raise ValueError('persisted recovery policy differs')
    truth = fixture.load_truth(family, context)
    cs = np.stack([census(image[i], support[i]).support for i in range(2)])
    profiles = {}
    for selection in POLICY['selectors']:
        r = runs[selection]
        profiles[selection] = {key: value for key, value in r.items() if key != 'matches'}
        profiles[selection]['eyes'] = base.evaluate(r['matches'], truth, cs, POLICY['disparity_count'])
    report = {
        'schema': 'stereoforge.sparse_recovery_evaluation.v1',
        'scope': 'previously inspected analytic development scene; sparse points, not dense completeness or held-out qualification',
        'counterpart_support': 'nearest integer floor(coordinate + 0.5), same rectified row, opposite Census support; not bilinear support',
        'decision_sha256': decision_hash, 'policy_sha256': policy_hash,
        'hashes': hashes, 'source_id': source, 'policy': POLICY,
        'numpy_version': np.__version__, 'profiles': profiles,
    }
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
