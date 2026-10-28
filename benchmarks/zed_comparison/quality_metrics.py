"""Cold, aligned NumPy depth metrics with fixed truth denominators.

Rates are fractions (multiply by 100 for percentage points). Counts and sums
are additive; quantiles are frame-local and must never be averaged as quantiles.
"""
from __future__ import annotations

import math
import numpy as np


METHOD = 'hierarchical_scene_sequence_bootstrap_v1'


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _ratio(num, den):
    return _finite(num / den) if den else None


def _mask(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or not np.all((a == 0) | (a == 1)):
        raise ValueError(f'{name} must be aligned and binary')
    return a.astype(bool, copy=False)


def _population(support, accepted, correct, error, relative):
    selected = support & accepted
    e = error[selected]
    r = relative[selected]
    n, k = int(np.count_nonzero(support)), int(e.size)
    with np.errstate(over='ignore', invalid='ignore'):
        total, rel, square = (_finite(np.sum(e)), _finite(np.sum(r)), _finite(np.sum(e * e)))
        # Scaling keeps RMSE meaningful even when its additive square sum overflows.
        maximum = float(np.max(e)) if k else None
        rmse = _finite(maximum * np.sqrt(np.mean((e / maximum) ** 2))) if k and maximum else (0. if k else None)
    count_correct = int(np.count_nonzero(selected & correct)) if correct is not None else None
    return dict(truth_count=n, accepted_count=k, error_count=k, correct_count=count_correct,
                accepted_incorrect_count=k-count_correct if count_correct is not None else None,
                abs_error_sum_m=total, relative_error_sum=rel, squared_error_sum_m2=square,
                mae_m=_ratio(total,k) if total is not None else None,
                relative_mae=_ratio(rel,k) if rel is not None else None,
                p95_abs_error_m=_finite(np.partition(e, math.ceil(.95*k)-1)[math.ceil(.95*k)-1]) if k else None,
                max_abs_error_m=_finite(maximum) if k else None, rmse_m=rmse,
                false_accept_rate=_ratio(k-count_correct,n) if count_correct is not None else None,
                conditional_false_accept_rate=_ratio(k-count_correct,k) if count_correct is not None else None,
                completeness=_ratio(k,n), correct_completeness=_ratio(count_correct,n) if count_correct is not None else None)


def _acceptance(accepted, sample, negative, geometric, region, *, negative_available=True):
    unknown = ~sample & ~negative
    neg_n = int(np.count_nonzero(negative & region))
    neg_k = int(np.count_nonzero(accepted & negative & region))
    return dict(unknown_accept_count=int(np.count_nonzero(accepted & unknown & region)),
                unsupported_accept_count=int(np.count_nonzero(accepted & ~geometric & region)) if geometric is not None else None,
                known_negative_count=neg_n if negative_available else None,
                known_negative_accept_count=neg_k if negative_available else None,
                known_negative_false_accept_rate=_ratio(neg_k, neg_n) if negative_available else None)


def _dilate(mask, radius):
    """Bounded square pixel neighborhood without wraparound or pixel loops."""
    h, w = mask.shape
    padded = np.pad(mask, radius)
    out = np.zeros_like(mask)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            out |= padded[dy:dy+h, dx:dx+w]
    return out


def _boundary(depth, accepted, truth, sample, region, policy):
    params = policy.get('freeze_parameters', {}).get('region_definitions') or {}
    definition = params.get('boundary') or {}
    edges = truth.get('boundary_edges')
    fields = ('absolute_jump_m','relative_jump','tolerance_pixels')
    unavailable = dict(value=None, reason='missing_independent_boundary_edges_or_frozen_definition')
    if edges is None or any(k not in definition for k in fields):
        return unavailable
    absolute, relative, radius = (definition[k] for k in fields)
    if (not isinstance(radius,int) or isinstance(radius,bool) or not 0 <= radius <= 8 or
            not all(isinstance(x,(int,float)) and math.isfinite(x) and x >= 0 for x in (absolute,relative))):
        raise ValueError('invalid boundary definition')
    edges = _mask(edges, sample.shape, 'boundary_edges')
    # Both endpoints of every tested pair require independent truth. Unknown
    # neighborhoods are excluded from both precision and recall populations.
    eligible = sample & region
    if radius:
        eligible &= ~_dilate(~sample, radius)
    predicted = np.zeros_like(sample)
    pair_eligible = np.zeros_like(sample)
    for axis in (0,1):
        first = (slice(None,-1),slice(None)) if axis == 0 else (slice(None),slice(None,-1))
        second = (slice(1,None),slice(None)) if axis == 0 else (slice(None),slice(1,None))
        pair = eligible[first] & eligible[second]
        pair_eligible[first] |= pair; pair_eligible[second] |= pair
        with np.errstate(over='ignore',invalid='ignore'):
            jump = np.abs(depth[first]-depth[second])
            limit = np.maximum(absolute, relative*np.minimum(depth[first],depth[second]))
        detected = pair & accepted[first] & accepted[second] & (jump > limit)
        predicted[first] |= detected; predicted[second] |= detected
    expected = edges & pair_eligible
    predicted &= pair_eligible
    tp_precision = int(np.count_nonzero(predicted & _dilate(expected,radius)))
    tp_recall = int(np.count_nonzero(expected & _dilate(predicted,radius)))
    npred, ntruth = int(predicted.sum()), int(expected.sum())
    precision, recall = _ratio(tp_precision,npred), _ratio(tp_recall,ntruth)
    score = (2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else
             0. if ntruth and (npred == 0 or precision == recall == 0) else None)
    return dict(value=score, reason=None if score is not None else 'empty_boundary_denominator',
                predicted_count=npred, truth_count=ntruth, matched_prediction_count=tp_precision,
                matched_truth_count=tp_recall, precision=precision, recall=recall)


def evaluate_frame(depth, validity, confidence, confidence_direction, truth_frame, policy, *, search_bounds=None):
    """Evaluate one exact aligned grid; no I/O, interpolation or GPU access."""
    depth = np.asarray(depth, dtype=np.float64)
    truth = np.asarray(truth_frame['depth_m'], dtype=np.float64)
    if depth.ndim != 2 or depth.shape != truth.shape:
        raise ValueError('prediction and truth must share an exact 2D grid')
    sample = _mask(truth_frame['sample'], depth.shape, 'sample')
    if np.any(sample & (~np.isfinite(truth) | (truth <= 0))):
        raise ValueError('sample truth must be finite positive axial depth')
    valid = _mask(validity, depth.shape, 'validity')
    accepted = valid & np.isfinite(depth) & (depth > 0)
    negative_value = truth_frame.get('known_negative')
    negative = np.zeros_like(sample) if negative_value is None else _mask(negative_value,depth.shape,'known_negative')
    if np.any(negative & sample):
        raise ValueError('sample and known-negative populations overlap')
    geometric_value = truth_frame.get('geometric_support')
    geometric = None if geometric_value is None else _mask(geometric_value,depth.shape,'geometric_support')
    lower, upper = policy.get('requested_range_m',[.3,8.])
    requested = sample & (truth >= lower) & (truth <= upper)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        error = np.abs(depth-truth)
        relative = error/truth
    freeze = policy.get('freeze_parameters') or {}
    correctness = freeze.get('correctness')
    correct = None
    g = truth_frame.get('geometry') or truth_frame.get('metadata',{}).get('geometry',{})
    if correctness is not None:
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            if correctness['mode'] == 'depth_tolerance':
                correct = error <= np.maximum(correctness['absolute_m'],correctness['relative']*truth)
            elif correctness['mode'] == 'disparity_pixels':
                fb = g['fx']*g['baseline_m']
                correct = np.abs(fb/depth-fb/truth) <= correctness['max_error_pixels']
            else:
                raise ValueError('unknown correctness definition')
    search = None
    if search_bounds is not None:
        lo, hi = search_bounds['min_disparity'], search_bounds['max_disparity_exclusive']
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError('invalid disparity search bounds')
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            disparity = g['fx']*g['baseline_m']/truth + g['cx_left']-g['cx_right']
        search = requested & (disparity >= lo) & (disparity < hi)
    supports = dict(requested=requested, full_sampled=sample,
                    common_support=None if geometric is None else requested & geometric,
                    search_supported=search)
    def populations(region):
        return {name: (_population(support & region,accepted,correct,error,relative) if support is not None else None)
                for name,support in supports.items()}
    all_pixels = np.ones_like(sample)
    regions = {'all':all_pixels}
    for name,value in (truth_frame.get('regions') or {}).items():
        if name == 'all':
            if value is not None and not np.all(_mask(value,depth.shape,'region all')):
                raise ValueError('declared all region cannot shrink implicit full-grid support')
            continue
        regions[name] = None if value is None else _mask(value,depth.shape,'region '+name)
    for name in policy.get('required_regions',[]):
        regions.setdefault(name,None)
    result = dict(lanes=populations(all_pixels),
                  regions={name:populations(mask) if mask is not None else None for name,mask in regions.items()},
                  **_acceptance(accepted,sample,negative,geometric,all_pixels,negative_available=negative_value is not None))
    result['all_prediction_accept_count'] = int(np.count_nonzero(accepted))
    result['region_acceptance'] = {name: _acceptance(accepted,sample,negative,geometric,mask,negative_available=negative_value is not None) if mask is not None else None
                                   for name,mask in regions.items()}
    boundary_region = regions.get('boundary')
    result['boundary_false_accept_rate'] = (result['regions']['boundary']['requested']['false_accept_rate']
                                            if boundary_region is not None else None)
    result['thin_pixel_recall'] = (result['regions']['thin']['requested']['correct_completeness']
                                   if regions.get('thin') is not None and (freeze.get('region_definitions') or {}).get('thin') else None)
    result['boundary_f1'] = (_boundary(depth,accepted,truth_frame,sample,boundary_region,policy)
                             if boundary_region is not None else dict(value=None,reason='missing_boundary_region'))
    result['surface_geometry'] = dict(value=None,reason='independent_surface_geometry_unavailable')
    bins = [0.3,0.5,1.,2.,3.,5.,8.]
    masks = {'<0.3':sample & (truth < .3), '>8':sample & (truth > 8)}
    for i,(lo,hi) in enumerate(zip(bins[:-1],bins[1:])):
        masks[f'[{lo:g},{hi:g}{"]" if i == len(bins)-2 else ")"}'] = sample & (truth >= lo) & ((truth <= hi) if i == len(bins)-2 else (truth < hi))
    result['distance_bins'] = {name:_population(mask,accepted,correct,error,relative) for name,mask in masks.items()}
    result['confidence_curve'] = []
    result['confidence_kind'] = 'uncalibrated_score'
    if confidence is not None:
        confidence = np.asarray(confidence,dtype=np.float64)
        if confidence.shape != depth.shape:
            raise ValueError('confidence must share exact prediction grid')
        aliases = {'high_is_good':True,'low_is_good':False,'higher_is_better':True,'higher_is_good':True,'lower_is_better':False,'lower_is_good':False}
        if confidence_direction not in aliases:
            raise ValueError('unknown confidence direction')
        eligible = accepted & np.isfinite(confidence)
        thresholds = np.unique(confidence[eligible])
        if aliases[confidence_direction]: thresholds = thresholds[::-1]
        if len(thresholds)>101:
            thresholds = thresholds[np.unique(np.linspace(0,len(thresholds)-1,101,dtype=int))]
        for threshold in thresholds:
            keep = eligible & ((confidence >= threshold) if aliases[confidence_direction] else (confidence <= threshold))
            result['confidence_curve'].append(dict(threshold=float(threshold),
                **_population(requested,keep,correct,error,relative), **_acceptance(keep,sample,negative,geometric,all_pixels,negative_available=negative_value is not None)))
    return result


def paired_decision(units, policy):
    """Bootstrap already aggregated scene/sequence pairs, with equal scene weight.

    The returned statistics gate cannot certify registration, held-out provenance,
    method identity or scientific parity; those remain the coordinator's gates.
    """
    result = dict(parity_qualified=False, statistics_pass=False, method=METHOD,
                  draws=2000, seed=0, interval_definition='percentile one-sided upper confidence bound; nearest rank',
                  gates={}, reasons=[])
    freeze = policy.get('freeze_parameters') or {}
    stats, uncertainty = freeze.get('statistics'), freeze.get('uncertainty')
    seen=set()
    for u in units:
        key=tuple(u.get(k) for k in ('scene_id','sequence_id','region'))
        if not all(isinstance(v,str) and v for v in key) or key in seen:
            raise ValueError('missing or duplicate scene/sequence/region unit')
        seen.add(key)
        for side in ('candidate','reference'):
            summary=u.get(side)
            if not isinstance(summary,dict): raise ValueError('missing paired summary')
            for name in ('mae_m','completeness','boundary_false_accept_rate'):
                value=summary.get(name)
                if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):
                    raise ValueError('nonfinite or malformed paired summary')
                if value is not None and (value < 0 or (name != 'mae_m' and value > 1)):
                    raise ValueError('paired summary outside physical metric range')
    result['scene_count']=len({u['scene_id'] for u in units})
    result['sequence_count']=len({(u['scene_id'],u['sequence_id']) for u in units})
    if not policy.get('frozen'):
        result['reasons'].append('provisional_policy')
    if not stats or stats.get('method') != METHOD or stats.get('unit') != 'scene_sequence':
        result['reasons'].append('unsupported_or_missing_statistics')
    if not uncertainty or not 0 < uncertainty.get('confidence_level',0) < 1:
        result['reasons'].append('missing_uncertainty')
    if stats and (result['scene_count'] < stats.get('min_scenes',1) or result['sequence_count'] < stats.get('min_sequences',1)):
        result['reasons'].append('insufficient_independent_units')
    if not freeze.get('zero_reference_error') or not freeze.get('correctness') or not freeze.get('region_definitions') or not freeze.get('partition_digest'):
        result['reasons'].append('missing_frozen_parameters')
    regions = set(policy.get('required_regions',[]))
    if not regions <= {u['region'] for u in units}:
        result['reasons'].append('missing_required_region_units')
    if not units: result['reasons'].append('no_paired_units')
    if result['reasons']: return result
    rng=np.random.default_rng(0)
    confidence=uncertainty['confidence_level']
    margins=policy['margins']
    for region in sorted(regions | {u['region'] for u in units}):
        selected=[u for u in units if u['region']==region]
        scenes=sorted({u['scene_id'] for u in selected})
        sequences={(u['scene_id'],u['sequence_id']) for u in selected}
        if len(scenes)<stats['min_scenes'] or len(sequences)<stats['min_sequences']:
            result['reasons'].append('insufficient_region_units:'+region); continue
        values=[]
        missing=False
        for scene in scenes:
            comparisons=[]
            for u in sorted((u for u in selected if u['scene_id']==scene),key=lambda u:u['sequence_id']):
                c,r=u['candidate'],u['reference']
                if any(c.get(k) is None or r.get(k) is None for k in ('mae_m','completeness','boundary_false_accept_rate')):
                    missing=True;break
                mae=(c['mae_m']-freeze['zero_reference_error']['absolute_mae_margin_m'] if r['mae_m']==0 else
                     c['mae_m']-margins['relative_mae_ratio']*r['mae_m'])
                if not math.isfinite(mae):
                    raise ValueError('nonfinite paired comparison arithmetic')
                comparisons.append([(r['completeness']-c['completeness'])*100,mae,
                                    (c['boundary_false_accept_rate']-r['boundary_false_accept_rate'])*100])
            values.append(np.asarray(comparisons,dtype=float))
        if missing:
            result['reasons'].append('unavailable_region_summary:'+region);continue
        # Vectorized over draws/sequences; a small scene loop preserves variable
        # sequence populations and samples fresh sequences for repeated scenes.
        scene_indices=rng.integers(0,len(scenes),size=(2000,len(scenes)))
        samples=np.zeros((2000,3),dtype=float)
        for column in range(len(scenes)):
            for index,array in enumerate(values):
                rows=np.flatnonzero(scene_indices[:,column]==index)
                if not len(rows):continue
                picks=rng.integers(0,len(array),size=(len(rows),len(array)))
                samples[rows]+=np.mean(array[picks],axis=1)/len(scenes)
        rank=math.ceil(confidence*2000)-1
        upper=np.sort(samples,axis=0)[rank]
        point=np.mean([np.mean(array,axis=0) for array in values],axis=0)
        gates={}
        for i,(name,margin) in enumerate((('completeness_loss',margins['completeness_loss_pp']),('mae',0.),('boundary_false_accept',margins['false_boundary_excess_pp']))):
            gates[name]=dict(estimate=float(point[i]),upper_bound=float(upper[i]),margin=float(margin),passed=bool(upper[i]<=margin))
        result['gates'][region]=gates
    result['statistics_pass']=not result['reasons'] and bool(result['gates']) and all(g['passed'] for gates in result['gates'].values() for g in gates.values())
    return result
