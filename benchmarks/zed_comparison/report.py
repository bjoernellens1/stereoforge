"""Compose raw benchmark tracks without treating external summary flags as proof."""
from . import schema as s
from .compare_quality import compare_quality, _digest, _method
from .evaluate_temporal import evaluate_temporal
from .native_profile import validate_full_sgm_profile
from .performance import summarize_timing
from .ros2_delivery import summarize_delivery, validate_delivery_policy

SCHEMA = 'stereoforge.zed_comparison.report.v1'
POLICY_SCHEMA = 'stereoforge.zed_comparison.report_policy.v1'
TRACK_FIELDS = {
    'quality': {'records', 'truth', 'root', 'truth_root'},
    'temporal': {'records', 'truth', 'root', 'truth_root', 'protocol', 'protocol_root'},
    'timing': {'records', 'root'},
    'delivery': {'events'},
}


def _same_estimator(quality_impl, live_impl, live_profile):
    """Compare the executed algorithm while allowing distinct ingestion lanes."""
    try:
        if quality_impl['name'] != 'StereoForge' or live_impl['name'] != 'StereoForge':
            return False
        for key in ('commit', 'device'):
            if quality_impl[key]['status'] != 'available' or live_impl[key]['status'] != 'available':
                return False
        commit = quality_impl['commit']['value']
        s._hex(commit, 40, 'commit')
        if commit != live_impl['commit']['value']:
            return False
        quality_device, live_device = quality_impl['device']['value'], live_impl['device']['value']
        for key in ('device_uuid', 'driver_uuid'):
            s._hex(quality_device[key], 32, key)
            if quality_device[key] == '0' * 32 or quality_device[key] != live_device[key]:
                return False
        for key in ('vendor_id', 'device_id', 'driver_version'):
            s._integer(quality_device[key], key, 0)
            if type(live_device[key]) is not int or quality_device[key] != live_device[key]:
                return False
        qprofile = validate_full_sgm_profile(quality_impl['settings']['effective_estimator'])
        lprofile = validate_full_sgm_profile(live_profile)
        return qprofile == lprofile == live_impl['settings']['effective_estimator']
    except (KeyError, TypeError, ValueError):
        return False


def _matching_strata(quality, quality_result, run):
    """Find qualified quality method/strata on the actual live calibration."""
    try:
        impl = run['implementation']
        if impl['status'] != 'available':
            return set()
        live = impl['value']
        settings = live['settings']
        raw = settings['original_calibration_identity']
        rectified = settings['rectified_calibration_identity']
        s._hex(raw, 64, 'raw calibration')
        s._hex(rectified, 64, 'rectified calibration')
        if rectified != run['calibration_identity']:
            return set()
        profile = validate_full_sgm_profile(run['effective_estimator'])
        depth_output = run['output_profiles']['depth']
        if (depth_output['width'], depth_output['height'], depth_output['encoding']) != (
                profile['width'], profile['height'], '32FC1'):
            return set()
        qualified = {(c['candidate_method'], c['stratum_key']) for c in quality_result['comparisons']
                     if c.get('parity_qualified') is True}
        available = {(f['method_key'], f['stratum_key']) for f in quality_result['frames']
                     if f.get('status') == 'available'}
        matches = set()
        for record in quality['records']:
            calibration = record['calibration']
            method = _method(record)
            pair = (_digest(method), _digest(s._calibration_identity(calibration)))
            geometry = calibration['geometry']
            if (pair in qualified and pair in available and calibration['raw']['sha256'] == raw and
                calibration['rectified']['sha256'] == rectified and
                (geometry['width'], geometry['height']) == (profile['width'], profile['height']) and
                _same_estimator(method, live, profile)):
                matches.add(pair)
        return matches
    except (KeyError, TypeError, ValueError):
        return set()


def build_report(records, policy):
    """Recompute track results and require evidence-bound cross-track identity.

    Track envelopes contain raw evaluator inputs; external summary documents
    are deliberately not accepted. A result is a software assessment of the
    supplied artifacts, not independent attestation of their provenance.
    Temporal evaluation alone measures sequence behavior; it does not yet bind
    an enabled reconstruction to its base estimator and live delivery. Thus a
    policy requiring temporal reconstruction cannot currently qualify.
    Timing comparisons also lack a frozen per-profile processing budget and
    matched-candidate limit evaluation, so they cannot qualify the report yet.
    """
    s._require(type(records) is list and len(records) <= 4, 'invalid report tracks')
    s._keys(policy, {'schema','frozen','quality','delivery','require_temporal','require_view'}, 'report policy')
    s._require(policy['schema'] == POLICY_SCHEMA and all(type(policy[k]) is bool for k in
               ('frozen','require_temporal','require_view')), 'invalid report policy')
    if policy['quality'] is not None: s.validate_policy(policy['quality'])
    if policy['delivery'] is not None: validate_delivery_policy(policy['delivery'])
    tracks={}
    for envelope in records:
        s._keys(envelope, {'kind','payload'}, 'track envelope')
        kind=envelope['kind']; s._require(type(kind) is str and kind in TRACK_FIELDS and kind not in tracks, 'invalid or repeated track')
        s._keys(envelope['payload'], TRACK_FIELDS[kind], kind + ' payload')
        tracks[kind]=envelope['payload']
    reasons=['processing_budget_not_verified']; results={}
    if not policy['frozen']: reasons.append('provisional_report_policy')
    quality=tracks.get('quality')
    if quality is None:
        reasons.append('missing_quality')
    else:
        results['quality']=compare_quality(quality['records'], quality['truth'], policy['quality'],
                                           root=quality['root'], truth_root=quality['truth_root'])
        if results['quality']['parity_qualified'] is not True: reasons.append('quality_unqualified')
    temporal=tracks.get('temporal')
    if policy['require_temporal']:
        reasons.append('temporal_reconstruction_binding_not_verified')
    if temporal is None:
        if policy['require_temporal']: reasons.append('missing_temporal')
    else:
        results['temporal']=evaluate_temporal(temporal['records'],temporal['truth'],policy['quality'],
            root=temporal['root'],truth_root=temporal['truth_root'],protocol=temporal['protocol'],protocol_root=temporal['protocol_root'])
        if policy['require_temporal'] and results['temporal']['quality_qualified'] is not True: reasons.append('temporal_unqualified')
    timing=tracks.get('timing')
    if timing is None:
        reasons.append('missing_timing')
    else:
        results['timing']=summarize_timing(timing['records'],root=timing['root'])
        if not any(x['qualified'] is True for x in results['timing']['comparisons']): reasons.append('timing_unqualified')
    delivery=tracks.get('delivery')
    if delivery is None:
        reasons.append('missing_delivery')
    else:
        results['delivery']=summarize_delivery(delivery['events'],policy=policy['delivery'])
        if results['delivery']['qualified'] is not True: reasons.append('delivery_unqualified')
        if policy['require_view']:
            # A profile label does not establish execution or incremental cost.
            reasons.append('view_execution_and_cost_not_verified')
    if quality is not None and timing is not None and delivery is not None:
        q=results['quality']; t=results['timing']; d=results['delivery']
        for run in d['runs']:
            impl=run.get('implementation',{})
            if impl.get('status') != 'available': reasons.append('missing_delivery_implementation'); continue
            matching = _matching_strata(quality, q, run)
            if not matching:
                reasons.append('quality_delivery_binding_mismatch'); continue
            if not any((row['method_key'], row['calibration_identity']) in matching and
                       row['session_id'] == comp['session_id'] and row['phase'] == comp['phase'] and
                       row['profile_id'] in comp['profiles']
                       for comp in t['comparisons'] if comp.get('qualified') is True
                       for row in timing['records']):
                reasons.append('timing_delivery_binding_mismatch')
        if not d['runs']: reasons.append('missing_delivery_runs')
    return dict(schema=SCHEMA,qualified=not reasons,reasons=sorted(set(reasons)),tracks=results,
                scope='artifact-backed software assessment; provenance is not independently attested')
