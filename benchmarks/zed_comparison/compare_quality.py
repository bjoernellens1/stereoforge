"""Cold independent-truth comparison; no SDK invocation or native runtime work."""
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from itertools import combinations
import math
import json
from pathlib import Path

import numpy as np

from . import schema as s
from .truth import load_truth
from .quality_metrics import evaluate_frame, paired_decision

DIGEST_ALGORITHM = 'sha256_canonical_sorted_unique_scene_sequence_partition_input_v1'


def _digest(value):
    return sha256(s._canonical(value).encode()).hexdigest()


def _method(record):
    return {k: record['implementation'][k] for k in ('name', 'version', 'commit', 'device', 'settings')}


def _descriptors(value):
    if type(value) is dict:
        if s.FILE_KEYS <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from _descriptors(child)
    elif type(value) is list:
        for child in value:
            yield from _descriptors(child)


def _ratio(n, d):
    if n is None or not d:
        return None
    v = n / d
    return v if math.isfinite(v) else None


def _sum(rows, key):
    values = [r.get(key) for r in rows]
    if any(v is None for v in values):
        return None
    value = sum(values)
    return value if math.isfinite(value) else None


def _population(rows):
    if not rows or any(r is None for r in rows):
        return None
    out = {k: _sum(rows, k) for k in ('truth_count', 'accepted_count', 'error_count', 'correct_count',
           'accepted_incorrect_count', 'abs_error_sum_m', 'relative_error_sum', 'squared_error_sum_m2')}
    n, k, c = out['truth_count'], out['error_count'], out['correct_count']
    out.update(mae_m=_ratio(out['abs_error_sum_m'], k), relative_mae=_ratio(out['relative_error_sum'], k),
               completeness=_ratio(out['accepted_count'], n), correct_completeness=_ratio(c, n),
               false_accept_rate=_ratio(out['accepted_incorrect_count'], n),
               conditional_false_accept_rate=_ratio(out['accepted_incorrect_count'], out['accepted_count']),
               p95_abs_error_m=None, p95_unavailable_reason='pooled_distribution_not_retained')
    out['rmse_m'] = math.sqrt(out['squared_error_sum_m2'] / k) if k and out['squared_error_sum_m2'] is not None else None
    maxima = [r['max_abs_error_m'] for r in rows if r.get('max_abs_error_m') is not None]
    out['max_abs_error_m'] = max(maxima) if maxima else None
    return out


def _acceptance(rows):
    if not rows or any(r is None for r in rows):
        return None
    out = {k: _sum(rows, k) for k in ('unknown_accept_count', 'unsupported_accept_count',
                                     'known_negative_count', 'known_negative_accept_count')}
    out['known_negative_false_accept_rate'] = _ratio(out['known_negative_accept_count'], out['known_negative_count'])
    return out


def _pool(rows):
    """Pool additive support/error statistics, never frame percentages/quantiles."""
    if not rows:
        return None
    lanes = set().union(*(r['lanes'] for r in rows))
    regions = set().union(*(r['regions'] for r in rows))
    bins = set().union(*(r['distance_bins'] for r in rows))
    out = dict(lanes={k: _population([r['lanes'].get(k) for r in rows]) for k in sorted(lanes)},
               regions={}, distance_bins={k: _population([r['distance_bins'].get(k) for r in rows]) for k in sorted(bins)},
               **_acceptance(rows))
    out['all_prediction_accept_count'] = _sum(rows, 'all_prediction_accept_count')
    out['region_acceptance'] = {}
    for name in sorted(regions):
        values = [r['regions'].get(name) for r in rows]
        out['regions'][name] = ({k: _population([v.get(k) for v in values]) for k in sorted(lanes)}
                                if all(v is not None for v in values) else None)
        out['region_acceptance'][name] = _acceptance([r['region_acceptance'].get(name) for r in rows])
    boundary = out['regions'].get('boundary')
    out['boundary_false_accept_rate'] = boundary['requested']['false_accept_rate'] if boundary and boundary.get('requested') else None
    thin = out['regions'].get('thin')
    out['thin_pixel_recall'] = (thin['requested']['correct_completeness'] if thin and thin.get('requested') and
                              all(r['thin_pixel_recall'] is not None for r in rows) else None)
    out['boundary_f1'] = dict(value=None, reason='pooled_boundary_matches_not_retained')
    out['confidence_curve'] = None
    out['confidence_curve_unavailable_reason'] = 'frame_specific_score_thresholds_not_pooled'
    out['surface_geometry'] = dict(value=None, reason='independent_surface_geometry_unavailable')
    return out


@contextmanager
def _observation(record, root, cache):
    hashes, fingerprints = cache
    paths = {d['path'] for d in _descriptors(record)}
    with ExitStack() as stack:
        files = s._Files(root, stack, {p: hashes[p] for p in paths}, {p: fingerprints[p] for p in paths})
        files.recheck()
        obs = record['observation']['value']
        arrays = {k: files.plane(obs[k]) for k in ('depth', 'validity')}
        arrays['confidence'] = (files.plane(obs['confidence']['value']) if obs['confidence']['status'] == 'available' else None)
        arrays['confidence_direction'] = obs['confidence_direction']
        try:
            yield arrays
        finally:
            files.recheck()


def _summary(metrics, region):
    lanes = metrics['regions'].get(region)
    population = lanes.get('requested') if lanes else None
    return dict(mae_m=population['mae_m'] if population else None,
                completeness=population['completeness'] if population else None,
                boundary_false_accept_rate=metrics['boundary_false_accept_rate'])


def compare_quality(records, truth, policy, *, root=None, truth_root=None):
    """Evaluate exact portable input groups against declared independent truth.

    An available ZED SDK observation declares execution in supplied evidence;
    this evaluator never invokes or independently attests a vendor implementation.
    """
    s._require(type(records) is list, 'records must be a list')
    s.validate_partitions(records)
    if type(policy) is dict and policy == {'status': 'unqualified', 'reason': 'missing_policy'}:
        policy = None
    elif type(policy) is dict and 'status' in policy:
        expected = 'policy_ready' if policy.get('frozen') is True else 'unqualified'
        s._require(policy['status'] == expected, 'policy loader status disagrees with frozen state')
        policy = {k: v for k, v in policy.items() if k != 'status'}
    missing_policy = policy is None
    if not missing_policy:
        s.validate_policy(policy)
    metric_policy = policy if policy is not None else dict(requested_range_m=[.3, 8.], required_regions=[], freeze_parameters={})
    result = dict(schema='stereoforge.zed_comparison.quality.v1', status='unqualified', parity_qualified=False,
                  reasons=[], methods=[], frames=[], strata=[], intersections=[], comparisons=[])
    reasons = result['reasons']
    if missing_policy:
        reasons.append('missing_policy')
        result['requested_range_source'] = 'project_default_0.3_to_8_m_without_correctness_thresholds'
    else:
        result['requested_range_source'] = 'supplied_policy'
        if not policy['frozen']:
            reasons.append('provisional_policy')
    entries = [{k: r[k] for k in ('scene_id', 'sequence_id', 'partition', 'input_identity')} for r in records]
    canonical_entries = sorted({s._canonical(e) for e in entries})
    # Sorting canonical rows defines the exact manifest ordering.
    result['partition_digest'] = _digest([json.loads(e) for e in canonical_entries])
    result['partition_digest_algorithm'] = DIGEST_ALGORITHM
    if root is None:
        reasons.append('missing_record_root')
    if truth_root is None:
        reasons.append('missing_truth_root')
    if root is None or truth_root is None:
        return result
    if type(root) is list:
        s._require(len(root) == len(records) and all(p is not None for p in root), 'root list must match record order')
        roots = [Path(p).resolve() for p in root]
    else:
        roots = [Path(root).resolve()] * len(records)
    caches = {}
    for record, directory in zip(records, roots):
        cache = caches.setdefault(directory, ({}, {}))
        with ExitStack() as stack:
            files = s._Files(directory, stack, *cache)
            s._validate(record, files)
    bundle = load_truth(truth, records, root=truth_root)
    truth_ids = {f['input_identity'] for f in bundle.metadata['frames']}
    frames_by_id = {f['input_identity']: f for f in bundle.metadata['frames']}
    methods, groups, stratum_methods = {}, {}, {}
    for index, record in enumerate(records):
        method = _digest(_method(record))
        stratum = _digest(s._calibration_identity(record['calibration']))
        impl = record['implementation']
        label = impl['name'] + ' / ' + str(impl['settings'].get('variant', impl['settings'].get('depth_mode', impl['version'])))
        methods[method] = dict(method_key=method, label=label, implementation=_method(record))
        row = dict(input_identity=record['input_identity'], method_key=method, stratum_key=stratum,
                   scene_id=record['scene_id'], sequence_id=record['sequence_id'], partition=record['partition'])
        result['frames'].append(row)
        groups.setdefault((stratum, record['input_identity']), []).append((index, record, row))
        stratum_methods.setdefault(stratum, set()).add(method)
    result['methods'] = [methods[k] for k in sorted(methods)]
    if not records:
        reasons.append('no_records')
    for (stratum, identity), members in sorted(groups.items()):
        available = []
        for index, record, row in members:
            if identity not in truth_ids or record['observation']['status'] != 'available':
                row.update(status='unavailable', reason='missing_truth' if identity not in truth_ids else 'missing_observation')
            else:
                available.append((index, record, row))
        if not available:
            continue
        with ExitStack() as stack:
            frame = stack.enter_context(bundle.frame(identity))
            arrays = {}
            for index, record, row in available:
                obs = stack.enter_context(_observation(record, roots[index], caches[roots[index]]))
                arrays[row['method_key']] = obs
                bounds = record['implementation']['settings'].get('search_bounds')
                if type(bounds) is dict and bounds.get('status') == 'available':
                    bounds = bounds['value']
                elif type(bounds) is dict and bounds.get('status') == 'unavailable':
                    bounds = None
                row.update(status='available', metrics=evaluate_frame(obs['depth'], obs['validity'], obs['confidence'],
                           obs['confidence_direction'], frame, metric_policy, search_bounds=bounds))
            for left, right in combinations(sorted(arrays), 2):
                a, b = arrays[left], arrays[right]
                mask = (a['validity'] != 0) & (b['validity'] != 0)
                mask &= np.isfinite(a['depth']) & (a['depth'] > 0) & np.isfinite(b['depth']) & (b['depth'] > 0)
                ma = evaluate_frame(a['depth'], mask, None, 'unavailable', frame, metric_policy)
                mb = evaluate_frame(b['depth'], mask, None, 'unavailable', frame, metric_policy)
                result['intersections'].append(dict(stratum_key=stratum, input_identity=identity,
                    candidate_method=left, reference_method=right,
                    intersection_count=ma['lanes']['requested']['accepted_count'],
                    candidate=ma['lanes']['requested'], reference=mb['lanes']['requested']))
    for stratum, method_keys in sorted(stratum_methods.items()):
        rows = [f for f in result['frames'] if f['stratum_key'] == stratum]
        result['strata'].append(dict(stratum_key=stratum, methods=[dict(method_key=m,
            frame_count=sum(f['method_key'] == m for f in rows),
            available_frame_count=sum(f['method_key'] == m and f['status'] == 'available' for f in rows),
            metrics=_pool([f['metrics'] for f in rows if f['method_key'] == m and f['status'] == 'available']))
            for m in sorted(method_keys)]))
        references = {f['method_key'] for f in rows if f['status'] == 'available' and
            methods[f['method_key']]['implementation']['name'] == 'ZED SDK' and
            methods[f['method_key']]['implementation']['settings'].get('depth_mode') == metric_policy.get('primary_reference')}
        candidates = method_keys - references
        if not references:
            reasons.append('missing_executed_reference')
        if not candidates:
            reasons.append('no_candidate_methods')
        for candidate in sorted(candidates):
            for reference in sorted(references):
                comparison = _comparison(rows, candidate, reference, policy, metric_policy,
                                         frames_by_id, result['partition_digest'])
                comparison.update(stratum_key=stratum, candidate_method=candidate, reference_method=reference)
                result['comparisons'].append(comparison)
                reasons.extend(comparison['reasons'])
    # Recheck all admitted dependencies once, including records without observations.
    for directory, cache in caches.items():
        with ExitStack() as stack:
            s._Files(directory, stack, *cache).recheck()
    bundle.recheck()
    reasons[:] = sorted(set(reasons))
    result['parity_qualified'] = bool(result['comparisons']) and not reasons and all(c['parity_qualified'] for c in result['comparisons'])
    result['status'] = 'qualified' if result['parity_qualified'] else 'unqualified'
    return result


def _comparison(rows, candidate, reference, policy, metric_policy, frames, digest):
    reasons = []
    crows = [r for r in rows if r['method_key'] == candidate]
    rrows = {r['input_identity']: r for r in rows if r['method_key'] == reference}
    paired = []
    if {r['input_identity'] for r in crows} != set(rrows):
        reasons.append('candidate_reference_input_set_mismatch')
    for c in crows:
        r = rrows.get(c['input_identity'])
        if r is None or c['status'] != 'available' or r['status'] != 'available':
            reasons.append('missing_matching_reference_or_observation')
            continue
        paired.append((c, r))
    if not paired:
        reasons.append('no_paired_observations')
    if any(c['partition'] != 'held_out' for c, _ in paired):
        reasons.append('development_data_in_comparison')
    if not any(c['partition'] == 'held_out' for c, _ in paired):
        reasons.append('missing_held_out_data')
    freeze = (policy or {}).get('freeze_parameters') or {}
    if policy is None or not policy['frozen']:
        reasons.append('missing_policy' if policy is None else 'provisional_policy')
    if freeze.get('partition_digest') != digest:
        reasons.append('partition_digest_mismatch_or_missing')
    uncertainty = freeze.get('uncertainty')
    required = metric_policy.get('required_regions', [])
    recipes = freeze.get('region_definitions') or {}
    for c, r in paired:
        frame = frames[c['input_identity']]
        if 'boundary' in required:
            definition = recipes.get('boundary') or {}
            radius = definition.get('tolerance_pixels')
            valid_definition = (type(radius) is int and 0 <= radius <= 8 and
                all(type(definition.get(k)) in (int, float) and math.isfinite(definition[k]) and definition[k] >= 0
                    for k in ('absolute_jump_m', 'relative_jump')))
            evidence = [row['metrics']['boundary_f1'] for row in (c, r)]
            if (not valid_definition or frame['masks']['boundary_edges']['status'] != 'available' or
                any(e.get('value') is None or not e.get('truth_count') for e in evidence)):
                reasons.append('missing_boundary_evidence')
        for name in required:
            if not recipes.get(name) or (name != 'all' and frame['regions'].get(name, {}).get('status') != 'available'):
                reasons.append('missing_required_region_or_recipe:' + name)
        registration, timestamp = frame['registration'], frame['timestamp_uncertainty']
        if not uncertainty or registration['status'] != 'available' or timestamp['status'] != 'available':
            reasons.append('missing_registered_uncertainty_bounds')
        elif (registration['value']['max_error_m'] > uncertainty['registration_max_m'] or
              timestamp['value']['max_error_ns'] > uncertainty['timestamp_max_ns']):
            reasons.append('truth_uncertainty_exceeds_policy')
    # Pool exact matched frame sets into independent scene/sequence units.
    sequence_pairs = {}
    for c, r in paired:
        sequence_pairs.setdefault((c['scene_id'], c['sequence_id']), ([], []))
        a, b = sequence_pairs[(c['scene_id'], c['sequence_id'])]
        a.append(c['metrics']); b.append(r['metrics'])
    units = []
    for (scene, sequence), (a, b) in sorted(sequence_pairs.items()):
        ca, rb = _pool(a), _pool(b)
        for region in required:
            units.append(dict(scene_id=scene, sequence_id=sequence, region=region,
                              candidate=_summary(ca, region), reference=_summary(rb, region)))
    statistics = paired_decision(units, metric_policy)
    reasons.extend(statistics['reasons'])
    if not statistics['statistics_pass']:
        reasons.append('paired_statistics_not_passing')
    return dict(parity_qualified=not reasons and statistics['statistics_pass'],
                reasons=sorted(set(reasons)), paired_units=units, statistics=statistics)
