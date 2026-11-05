"""Frozen campaign metrics on independent, fixed support; depth is axial Z in metres.

Provenance follows DaxaProvenance: 1/5 stereo, 2/3 temporal, 4 neural, 6/7 spatial.
Temporal-planar (7) is counted in both inference totals, never measured.
Scores alone never establish reference independence, registration, or timing.
"""
from __future__ import annotations

import numpy as np

TARGETS = {
    'range_m': [0.5, 3.0], 'tau_floor_m': .010, 'tau_relative': .01,
    'reference_uncertainty_fraction': 1 / 3,
    'max_bad_2px_regression_percentage_points': .25,
    'live_unique_fps_min': 30, 'live_gpu_p95_ms_max': 33.3,
    'live_ingress_publication_p95_ms_max': 100, 'microbenchmark_gpu_ms_max': 4.167,
    'promotion': 'paired scene-level confidence intervals; no supported catastrophic/halo/false-fill increase',
}


def score(depth, reference, support, provenance, *, reference_uncertainty_m):
    """Unconditional correct *measured* completeness, never valid-only completeness.

    Qualification here is numerical only; the caller must also qualify reference
    independence, geometry and clock. Uncertainty is a total reference error bound.
    """
    depth, reference = np.asarray(depth), np.asarray(reference)
    support, provenance = np.asarray(support, dtype=bool), np.asarray(provenance)
    if not (depth.shape == reference.shape == support.shape == provenance.shape):
        raise ValueError('depth, reference, fixed support and provenance must have equal shapes')
    if np.any(support & (~np.isfinite(reference) | (reference <= 0))):
        raise ValueError('reference must be positive and finite on all fixed support')
    uncertainty = np.broadcast_to(np.asarray(reference_uncertainty_m), depth.shape)
    if np.any(support & (~np.isfinite(uncertainty) | (uncertainty < 0))):
        raise ValueError('reference uncertainty must be finite and nonnegative')
    if not np.isin(provenance, list(range(8))).all():
        raise ValueError('unknown provenance')
    finite = np.isfinite(depth) & (depth > 0)
    measured = support & finite & np.isin(provenance, [1, 5])
    n, count = int(support.sum()), int(measured.sum())
    tau = np.maximum(.010, .01 * reference)
    err = depth[measured] - reference[measured]
    absolute = np.abs(err)
    correct = int(np.count_nonzero(absolute <= tau[measured]))
    qualified = support & (uncertainty <= tau / 3)
    pct = lambda num, den: float(100 * num / den) if den else None
    percentile = lambda values, q: float(np.percentile(values, q)) if values.size else None
    result = dict(support_pixels=n, measured_pixels=count, correct_measured_pixels=correct,
                  correct_completeness_pct=pct(correct, n), measured_coverage_pct=pct(count, n),
                  temporal_pixels=int(np.count_nonzero(support & finite & np.isin(provenance, [2, 3, 7]))),
                  spatial_pixels=int(np.count_nonzero(support & finite & np.isin(provenance, [6, 7]))),
                  neural_pixels=int(np.count_nonzero(support & finite & (provenance == 4))),
                  reference_qualified_pixels=int(qualified.sum()),
                  bias_mm=float(np.median(err) * 1000) if count else None,
                  rmse_mm=float(np.sqrt(np.mean(err ** 2)) * 1000) if count else None,
                  p95_absolute_error_mm=percentile(absolute * 1000, 95),
                  p99_absolute_error_mm=percentile(absolute * 1000, 99),
                  median_absolute_error_mm=percentile(absolute * 1000, 50))
    normalized = err / tau[measured]
    result['numerical_accuracy_gate_passed'] = bool(n and count and qualified.sum() == n
                                        and abs(np.median(normalized)) <= 1
                                        and np.percentile(np.abs(normalized), 95) <= 2)
    curves = {}
    for mm in (5, 10, 20, 50):
        good = np.count_nonzero(absolute <= mm / 1000)
        curves[f'{mm}mm'] = pct(good, n)
        result[f'bad_{mm}mm_pct_of_measured'] = pct(count - good, count)
    for percent in (.5, 1, 2):
        curves[f'{percent:g}percent'] = pct(np.count_nonzero(absolute <= reference[measured] * percent / 100), n)
    result['correct_completeness_curve_pct'] = curves
    return result


def failure_accounting(masks):
    """Ordered observed predicates: full overlapping bits and disjoint first reason.

    A missing predicate must be omitted, never silently treated as passing.
    `unattributed` means no supplied predicate, not proof of correct output.
    """
    if not masks or len(masks) > 31:
        raise ValueError('supply 1..31 observed reason masks')
    arrays = [np.asarray(m, dtype=bool) for m in masks.values()]
    if any(m.shape != arrays[0].shape for m in arrays):
        raise ValueError('reason masks must have equal shapes')
    bits = np.zeros(arrays[0].shape, np.uint32)
    first = np.zeros_like(bits)
    all_counts, first_counts = {}, {}
    for i, (name, mask) in enumerate(zip(masks, arrays)):
        bit = np.uint32(1 << i)
        selected = mask & (first == 0)
        first[selected] = bit
        bits[mask] |= bit
        all_counts[name], first_counts[name] = int(mask.sum()), int(selected.sum())
    first_counts['unattributed'] = int(np.count_nonzero(first == 0))
    return dict(all_reasons=all_counts, first_failure=first_counts,
                reason_bits={name: 1 << i for i, name in enumerate(masks)}), bits, first


def require_equivalent(actual, diagnostic):
    """Refuse counters from a diagnostic rerun that changes any exported output."""
    if actual.keys() != diagnostic.keys() or not actual:
        raise ValueError('diagnostic equivalence requires identical nonempty output sets')
    for key in actual:
        left, right = np.ascontiguousarray(actual[key]), np.ascontiguousarray(diagnostic[key])
        if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
            raise ValueError(f'diagnostic equivalence failed: {key}')


def dynamic_reference_available(speed_m_s, timing_uncertainty_s, admissible_error_m):
    if timing_uncertainty_s is None:
        return False
    values = np.asarray([speed_m_s, timing_uncertainty_s, admissible_error_m])
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('motion and timing budgets must be finite and nonnegative')
    return bool(speed_m_s * timing_uncertainty_s <= admissible_error_m)
