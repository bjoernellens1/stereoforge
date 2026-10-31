"""Truth-independent anchor matching and fixed-support stereo candidate evaluation.

Input NPZ: left/right uint8 rectified images, truth disparity (negative or NaN
means unknown); optional coarse full-resolution disparity and a Boolean risk mask.
Truth and masks never enter match_anchors. No claim of dense depth recovery.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from fast60_anchor_hypotheses import AnchorConfig, match_anchors


def evaluate(left, right, truth, *, coarse=None, risk=None, config=AnchorConfig(), tolerance=1.0):
    left, right, truth = map(np.asarray, (left, right, truth))
    if left.shape != right.shape or truth.shape != left.shape or truth.ndim != 2:
        raise ValueError('left/right/truth must have identical HxW dimensions')
    if not np.issubdtype(truth.dtype, np.number):
        raise ValueError('truth disparity must be numeric')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid tolerance')
    if coarse is not None:
        coarse = np.asarray(coarse)
        if coarse.shape != truth.shape or not np.issubdtype(coarse.dtype, np.number):
            raise ValueError('coarse disparity shape or dtype mismatch')
    if risk is not None:
        risk = np.asarray(risk)
        if risk.shape != truth.shape or risk.dtype != np.bool_:
            raise ValueError('risk mask must be a Boolean HxW array')
    anchors = match_anchors(left, right, config)
    known_all = np.isfinite(truth) & (truth >= 0)
    anchor_mask = np.zeros(truth.shape, dtype=bool)
    anchor_mask[anchors.y, anchors.left_x] = True
    selected_known = anchor_mask & known_all
    prediction = np.full(truth.shape, np.nan, dtype=np.float64)
    prediction[anchors.y, anchors.left_x] = anchors.disparity
    correct = selected_known & (np.abs(prediction - truth) <= tolerance)
    selected_error = selected_known & ~correct
    result = {
        'schema': 'stereoforge.fast60_anchor_evaluation.v1',
        'input_shape': list(left.shape),
        'selected_anchor_count': anchors.count,
        'evaluated_candidate_count': anchors.evaluated,
        'known_truth_pixels': int(known_all.sum()),
        'known_selected_anchor_count': int(selected_known.sum()),
        'correct_anchor_count': int(correct.sum()),
        'incorrect_anchor_count': int(selected_error.sum()),
        'correct_fraction_of_selected_known': float(correct.sum()/selected_known.sum()) if selected_known.any() else None,
        'correct_anchor_fraction_of_fixed_truth_support': float(correct.sum()/known_all.sum()) if known_all.any() else None,
        'config': dict(disparities=config.disparities, stride=config.stride, min_gradient=config.min_gradient,
                       uniqueness_margin=config.uniqueness_margin, max_hamming=config.max_hamming,
                       lr_tolerance=config.lr_tolerance, tolerance=tolerance),
    }
    if coarse is not None:
        coarse_known = known_all & np.isfinite(coarse) & (coarse >= 0)
        coarse_correct = coarse_known & (np.abs(coarse-truth) <= tolerance)
        rescued = correct & ~coarse_correct
        result['coarse_comparison'] = {
            'coarse_correct_on_fixed_support': int(coarse_correct.sum()),
            'anchors_correct_where_coarse_incorrect_or_missing': int(rescued.sum()),
            'rescued_fraction_of_fixed_truth_support': float(rescued.sum()/known_all.sum()) if known_all.any() else None,
            'correct_anchor_already_coarse_correct': int((correct & coarse_correct).sum()),
            'incorrect_anchor_where_coarse_correct': int((selected_error & coarse_correct).sum()),
        }
    if risk is not None:
        risk_support = risk & known_all
        result['risk'] = {
            'fixed_truth_risk_pixels': int(risk_support.sum()),
            'correct_risk_anchors': int((correct & risk).sum()),
            'incorrect_risk_anchors': int((selected_error & risk).sum()),
            'correct_risk_anchor_coverage': float((correct & risk).sum()/risk_support.sum()) if risk_support.any() else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_npz', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--disparities', type=int, default=128)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--min-gradient', type=int, default=16)
    parser.add_argument('--uniqueness-margin', type=int, default=3)
    parser.add_argument('--max-hamming', type=int, default=8)
    parser.add_argument('--lr-tolerance', type=int, default=0)
    parser.add_argument('--truth-tolerance', type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; specify a new path')
    with np.load(args.input_npz, allow_pickle=False) as data:
        mandatory = {'left', 'right', 'truth_disparity'}
        if not mandatory.issubset(data.files):
            parser.error('NPZ must contain left, right, truth_disparity')
        result = evaluate(data['left'], data['right'], data['truth_disparity'],
            coarse=data['coarse_disparity'] if 'coarse_disparity' in data.files else None,
            risk=data['risk_mask'] if 'risk_mask' in data.files else None,
            config=AnchorConfig(disparities=args.disparities, stride=args.stride,
                min_gradient=args.min_gradient, uniqueness_margin=args.uniqueness_margin,
                max_hamming=args.max_hamming, lr_tolerance=args.lr_tolerance),
            tolerance=args.truth_tolerance)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
