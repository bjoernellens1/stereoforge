"""Offline controlled temporal evidence with bounded vectorized track state.

Stability of an output is distinct from a fresh independent observation. This
module does not run a camera, SDK, tracker, registration, or VIO estimator.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import schema as s
from .compare_quality import _digest, _method, _descriptors, DIGEST_ALGORITHM
from .truth import load_truth, _rigid, MAX_FRAME_BYTES

SCHEMA = 'stereoforge.zed_comparison.temporal_protocol.v1'


def temporal_settings(record):
    """Only explicit effective settings carry temporal semantics."""
    value = record['implementation']['settings'].get('temporal', {})
    s._require(type(value) is dict and set(value) <= {'stabilization', 'tracking', 'filling'}, 'unknown temporal settings')
    out = {}
    for key in ('stabilization', 'tracking', 'filling'):
        item = deepcopy(value.get(key, dict(status='unavailable', reason='effective setting not supplied')))
        def check(x):
            if key == 'stabilization':
                s._number(x, key, minimum=0)
                s._require(x <= 100, 'stabilization exceeds 100')
            else:
                s._require(type(x) is bool, key + ' must be boolean')
        s._availability(item, check, key)
        out[key] = item
    complete = all(v['status'] == 'available' for v in out.values())
    on = ((out['stabilization']['status'] == 'available' and out['stabilization']['value'] > 0) or
          (out['filling']['status'] == 'available' and out['filling']['value']))
    out['state'] = 'temporal_on' if on else 'temporal_off' if complete else 'unknown'
    return out


def _content_tree(value):
    if type(value) is dict:
        return {k: _content_tree(v) for k, v in value.items() if not (k == 'path' and s.FILE_KEYS <= value.keys())}
    if type(value) is list:
        return [_content_tree(v) for v in value]
    return value


def _protocol(document, records, root, files):
    if isinstance(document, (str, Path)):
        document = s._read_json(document)
    document = deepcopy(document)
    s._json_value(document)
    s._require(len(s._canonical(document).encode()) <= s.MAX_JSON_BYTES, 'protocol exceeds bounded JSON size')
    s._keys(document, {'schema','frozen','partition_digest','provenance','limits','sequences'}, 'temporal protocol')
    s._require(document['schema'] == SCHEMA and type(document['frozen']) is bool, 'invalid temporal protocol schema/frozen')
    s._hex(document['partition_digest'], 64, 'partition_digest')
    p = document['provenance']
    s._keys(p, {'independent','kind','artifact','generator'}, 'protocol provenance')
    s._require(p['independent'] is True and p['kind'] in ('analytic','measured'), 'independent protocol required')
    def file(d):
        s._descriptor(d); files.check(d)
    file(p['artifact'])
    generator = s._availability(p['generator'], file, 'protocol generator')
    s._require(p['kind'] != 'analytic' or generator, 'analytic motion/bounds require bound generator artifact')
    limits = document['limits']
    s._keys(limits, {'max_gap_ns','pose_timestamp_tolerance_ns','track_position_tolerance_m','max_tracks','ghost_change_absolute_m','ghost_change_relative'}, 'protocol limits')
    s._integer(limits['max_gap_ns'], 'max_gap_ns', 1)
    s._integer(limits['pose_timestamp_tolerance_ns'], 'pose_timestamp_tolerance_ns')
    s._integer(limits['max_tracks'], 'max_tracks', 1, 2097152)
    for key in ('track_position_tolerance_m','ghost_change_relative'):
        s._number(limits[key], key, minimum=0)
    s._number(limits['ghost_change_absolute_m'], 'ghost_change_absolute_m', positive=True)
    strata = {}
    for r in records:
        key = (r['scene_id'],r['sequence_id'],r['partition'],_digest(s._calibration_identity(r['calibration'])))
        strata.setdefault(key, {})[r['input_identity']] = r
    s._require(type(document['sequences']) is list, 'sequences must be list')
    admitted = {}
    for seq in document['sequences']:
        s._keys(seq, {'scene_id','sequence_id','partition','rig_motion','scene_motion','calibration_profile','frames','recovery_events'}, 'protocol sequence')
        for key in ('scene_id','sequence_id'):s._text(seq[key],key)
        s._require(seq['partition'] in ('development','held_out') and seq['rig_motion'] in ('stationary','known','estimated','unavailable') and seq['scene_motion'] in ('static','dynamic') and seq['calibration_profile'] in ('factory','effective','analytic'), 'unsupported sequence declaration')
        s._require(type(seq['frames']) is list and bool(seq['frames']), 'sequence frames must be nonempty list')
        ids = [f.get('input_identity') for f in seq['frames'] if type(f) is dict]
        s._require(len(ids) == len(seq['frames']) and len(ids) == len(set(ids)), 'duplicate protocol frame')
        matches = [k for k,v in strata.items() if k[:3] == (seq['scene_id'],seq['sequence_id'],seq['partition']) and set(v) == set(ids)]
        s._require(len(matches) == 1, 'protocol frames must exactly bind unique sequence/calibration input set')
        key = matches[0]; s._require(key not in admitted, 'duplicate protocol sequence stratum')
        inputs = strata[key]
        for f in seq['frames']:
            s._keys(f, {'input_identity','world_from_left','track_ids'}, 'protocol frame')
            s._hex(f['input_identity'],64,'input_identity');r=inputs[f['input_identity']]
            def pose(v):
                s._keys(v, {'matrix','capture'}, 'pose');_rigid(v['matrix']);s._timestamp(v['capture'])
                capture=r['timestamps']['capture']
                s._require(v['capture']['clock_domain']==capture['clock_domain'], 'pose capture clock mismatch')
                s._require(abs(v['capture']['value_ns']-capture['value_ns']) <= limits['pose_timestamp_tolerance_ns'], 'pose timestamp outside tolerance')
            has_pose=s._availability(f['world_from_left'],pose,'world_from_left')
            s._require(seq['rig_motion']!='known' or has_pose, 'known rig motion requires all poses')
            def tracks(d):
                s._descriptor(d,True)
                g=r['calibration']['geometry']
                s._require(d['format']=='i64_le' and (d['width'],d['height'])==(g['width'],g['height']), 'track plane format/geometry mismatch')
                s._require(d['size_bytes']<=MAX_FRAME_BYTES,'track frame exceeds 512 MiB plane extent')
                files.check(d)
            s._availability(f['track_ids'],tracks,'track_ids')
        s._require(type(seq['recovery_events']) is list,'recovery_events must be list')
        seen=set()
        for e in seq['recovery_events']:
            s._keys(e,{'event_id','kind','start_input_identity','target_track_ids','justification'},'recovery event')
            s._text(e['event_id'],'event_id');s._text(e['justification'],'justification')
            s._require(e['event_id'] not in seen and e['kind'] in ('disocclusion','motion_reset','capture_gap','appearance') and e['start_input_identity'] in inputs, 'invalid recovery event')
            seen.add(e['event_id']);targets=e['target_track_ids']
            s._require(type(targets) is list and bool(targets) and len(targets)==len(set(targets)), 'unique recovery targets required')
            s._require(len(targets)<=limits['max_tracks'],'recovery targets exceed max_tracks')
            for target in targets:s._integer(target,'target_track_id',0,s.INT64_MAX)
        if seq['rig_motion']=='stationary':
            poses=[np.asarray(f['world_from_left']['value']['matrix']) for f in seq['frames'] if f['world_from_left']['status']=='available']
            s._require(not poses or all(bool(np.allclose(pose,poses[0],rtol=0,atol=1e-12)) for pose in poses[1:]),'stationary rig has moving supplied poses')
        admitted[key]=seq
    s._require(set(admitted)==set(strata),'missing protocol sequence stratum')
    return document, admitted


class _Moments:
    """Sorted ID association; one vector per statistic, never a pixel loop."""
    def __init__(self, maximum):
        self.maximum=maximum;self.ids=np.empty(0,dtype=np.int64)
        self.n=np.empty(0,dtype=np.int64);self.mean=np.empty(0);self.m2=np.empty(0)
        self.finished=dict(track_count=0,observation_count=0,bias_sum_m=0.,std_sum_m=0.)
    def update(self, ids, values):
        s._require(bool(np.all(np.isfinite(values))),'nonfinite track errors')
        if np.array_equal(ids,self.ids):
            # Stable dense tracks already share state order: update contiguous
            # vectors directly instead of sorting and gathering every frame.
            s._require(len(ids)<=self.maximum,'track state exceeds max_tracks')
            self.n+=1
            delta=values-self.mean
            self.mean+=delta/self.n
            self.m2+=delta*(values-self.mean)
            s._require(bool(np.all(np.isfinite(self.mean))) and bool(np.all(np.isfinite(self.m2))),'track moment numerical overflow')
            return
        union=np.union1d(self.ids,ids)
        s._require(len(union)<=self.maximum,'track state exceeds max_tracks')
        if len(union)!=len(self.ids):
            where=np.searchsorted(union,self.ids)
            for name in ('n','mean','m2'):
                old=getattr(self,name);new=np.zeros(len(union),dtype=old.dtype);new[where]=old;setattr(self,name,new)
            self.ids=union
        at=np.searchsorted(self.ids,ids);n=self.n[at]+1;delta=values-self.mean[at]
        self.mean[at]+=delta/n;self.m2[at]+=delta*(values-self.mean[at]);self.n[at]=n
        s._require(bool(np.all(np.isfinite(self.mean[at]))) and bool(np.all(np.isfinite(self.m2[at]))),'track moment numerical overflow')
    def reset(self):
        summary=self._summary()
        for key in self.finished:self.finished[key]+=summary[key]
        self.n[:]=0;self.mean[:]=0;self.m2[:]=0
    def _summary(self):
        mask=self.n>=2
        return dict(track_count=int(mask.sum()),observation_count=int(self.n[mask].sum()),bias_sum_m=float(self.mean[mask].sum()),std_sum_m=float(np.sqrt(np.maximum(0,self.m2[mask]/np.maximum(1,self.n[mask]-1))).sum()))
    def report(self):
        current=self._summary();out={k:self.finished[k]+current[k] for k in current};n=out['track_count']
        out.update(mean_track_bias_m=out.pop('bias_sum_m')/n if n else None,mean_track_error_std_m=out.pop('std_sum_m')/n if n else None,p95_error_std_m=None,p95_unavailable_reason='pooled_track_distribution_not_retained',population='track_segments_with_at_least_two_unique_captures',std_ddof=1)
        return out


def _correct(depth, truth, definition, g):
    if definition is None:return None
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        if definition['mode']=='depth_tolerance':return np.abs(depth-truth)<=np.maximum(definition['absolute_m'],definition['relative']*truth)
        return np.abs(g['fx']*g['baseline_m']/depth-g['fx']*g['baseline_m']/truth)<=definition['max_error_pixels']


def evaluate_temporal(records, truth, policy, *, root=None, truth_root=None, protocol=None, protocol_root=None):
    """Score controlled sequences with explicit export, truth and protocol roots."""
    start=perf_counter();s._require(type(records) is list,'records must be a list')
    result=dict(schema='stereoforge.zed_comparison.temporal.v1',status='unqualified',quality_qualified=False,reasons=[],repeat_count=0,groups=[])
    reasons=result['reasons']
    for r in records:s._validate(r);temporal_settings(r)
    if policy is not None and policy.get('reason')=='missing_policy':policy=None
    if policy is not None:
        policy=deepcopy(policy);status=policy.pop('status',None)
        s._require(status is None or status==('policy_ready' if policy.get('frozen') else 'unqualified'),'policy loader status disagrees with frozen state');s.validate_policy(policy)
    for name,value in [('record_root',root),('truth_root',truth_root),('protocol',protocol),('protocol_root',protocol_root)]:
        if value is None:reasons.append('missing_'+name)
    if reasons:return result
    if type(root) is list:
        s._require(len(root)==len(records) and all(p is not None for p in root),'root list must match record order');roots=[Path(p).resolve() for p in root]
    else:roots=[Path(root).resolve()]*len(records)
    caches={};unique={};rows=[]
    for r,directory in zip(records,roots):
        cache=caches.setdefault(directory,({},{}))
        with ExitStack() as stack:s._validate(r,s._Files(directory,stack,*cache))
        key=(_digest(_method(r)),r['input_identity'])
        content=_digest(_content_tree(r))
        if key in unique:
            s._require(unique[key]==content,'conflicting repeated output for same method/input');result['repeat_count']+=1
        else:unique[key]=content;rows.append((r,directory))
    records=[r for r,_ in rows];s.validate_partitions(records)
    manifest=sorted({s._canonical({k:r[k] for k in ('scene_id','sequence_id','partition','input_identity')}) for r in records})
    digest=_digest([json.loads(x) for x in manifest]);result.update(partition_digest=digest,partition_digest_algorithm=DIGEST_ALGORITHM)
    # Capture chronology is checked before protocol binding so clock errors remain explicit.
    sequences={}
    for r in records:sequences.setdefault((r['scene_id'],r['sequence_id'],_digest(s._calibration_identity(r['calibration']))),{})[r['input_identity']]=r
    for frames in sequences.values():
        ordered=sorted(frames.values(),key=lambda r:r['ordinal']);clocks={r['timestamps']['capture']['clock_domain'] for r in ordered}
        s._require(len(clocks)==1 and not (clocks & {'unknown','unavailable'}),'capture clock mismatch or unknown clock')
        s._require(all(b['source_id']>a['source_id'] and b['timestamps']['capture']['value_ns']>a['timestamps']['capture']['value_ns'] for a,b in zip(ordered,ordered[1:])),'capture timestamps/source IDs must strictly increase')
    bundle=load_truth(truth,records,root=truth_root);pcache=({}, {})
    with ExitStack() as stack:
        pfiles=s._Files(protocol_root,stack,*pcache);doc,protocols=_protocol(protocol,records,protocol_root,pfiles)
    if not records:reasons.append('no_records')
    if not policy or not policy['frozen']:reasons.append('missing_or_provisional_policy')
    if not doc['frozen']:reasons.append('provisional_protocol')
    freeze=(policy or {}).get('freeze_parameters') or {};definition=freeze.get('correctness')
    if definition is None:reasons.append('missing_correctness_threshold')
    if doc['partition_digest']!=digest or freeze.get('partition_digest')!=digest:reasons.append('partition_digest_mismatch_or_missing')
    grouped={}
    for r,directory in rows:
        key=(r['scene_id'],r['sequence_id'],r['partition'],_digest(s._calibration_identity(r['calibration'])))
        grouped.setdefault((key,_digest(_method(r))),[]).append((r,directory))
    for (key,method),members in sorted(grouped.items()):
        seq=protocols[key];settings=temporal_settings(members[0][0]);local=[]
        if not all(settings[k]['status']=='available' for k in ('stabilization','tracking','filling')):local.append('incomplete_effective_temporal_settings')
        if key[2]!='held_out':local.append('development_data')
        if {r['input_identity'] for r,_ in members}!={f['input_identity'] for f in seq['frames']}:local.append('method_input_set_mismatch')
        if seq['rig_motion'] in ('estimated','unavailable'):local.append('motion_not_independently_known')
        group=_evaluate_group(members,seq,doc['limits'],bundle,protocol_root,pcache,caches,definition,freeze,local)
        group.update(scene_id=key[0],sequence_id=key[1],partition=key[2],stratum_key=key[3],method_key=method,implementation=_method(members[0][0]),temporal_settings=settings,rig_motion=seq['rig_motion'],calibration_profile=seq['calibration_profile'])
        result['groups'].append(group);reasons.extend(local)
    for directory,cache in caches.items():
        with ExitStack() as stack:s._Files(directory,stack,*cache).recheck()
    with ExitStack() as stack:s._Files(protocol_root,stack,*pcache).recheck()
    bundle.recheck();result['reasons']=sorted(set(reasons));result['quality_qualified']=bool(result['groups']) and not reasons
    result['status']='qualified' if result['quality_qualified'] else 'unqualified'
    result['offline_evaluation_seconds']=perf_counter()-start
    result['qualification_scope']='controlled temporal evaluator evidence; no SDK parity or independent VIO qualification'
    return result


def _evaluate_group(members, seq, limits, bundle, protocol_root, pcache, caches, definition, freeze, reasons):
    maximum=limits['max_tracks'];output=_Moments(maximum);fresh_moments=_Moments(maximum);raw=_Moments(maximum)
    frames={f['input_identity']:f for f in seq['frames']};members=sorted(members,key=lambda x:x[0]['ordinal'])
    captures={r['source_id']:r['timestamps']['capture']['value_ns'] for r,_ in members}
    capture_keys=np.asarray(sorted(captures),dtype=np.int64)
    capture_times=np.asarray([captures[int(k)] for k in capture_keys],dtype=np.int64)
    events=[];active=[];previous=None;previous_time=None;world_ids=np.empty(0,dtype=np.int64);world_points=np.empty((0,3));seen_tracks=np.empty(0,dtype=np.int64);implicit_ids=np.empty(0,dtype=np.int64);implicit_z=np.empty(0)
    counts=dict(fresh_observation_count=0,derived_current_source_count=0,historical_count=0,unknown_freshness_count=0,valid_output_count=0)
    gaps=[];ghost_count=0;ghost_population=0;max_state=0
    stationary_grid=seq['rig_motion']=='stationary';static_grid=stationary_grid and seq['scene_motion']=='static'
    for r,directory in members:
        identity=r['input_identity'];f=frames[identity];time=r['timestamps']['capture']['value_ns'];g=r['calibration']['geometry']
        if previous_time is not None and time-previous_time>limits['max_gap_ns']:
            gaps.append(dict(previous_capture_ns=previous_time,capture_ns=time,gap_ns=time-previous_time));output.reset();fresh_moments.reset();raw.reset();previous=None
        previous_time=time
        with ExitStack() as stack:
            # Only this frame's record dependencies are rechecked in this context.
            cache=caches[directory];paths={d['path'] for d in _descriptors(r)}
            files=s._Files(directory,stack,{p:cache[0][p] for p in paths},{p:cache[1][p] for p in paths});files.recheck()
            pf=s._Files(protocol_root,stack,*pcache)
            frame=stack.enter_context(bundle.frame(identity));truth=frame['depth_m'];sample=frame['sample']
            if bundle.metadata['provenance']['kind']=='measured':
                reference=getattr(bundle,'physical_references',{}).get(identity,{})
                if reference.get('status')!='metadata_ready':reasons.append('missing_physical_reference_metadata')
            uncertainty=freeze.get('uncertainty')
            registration=frame['registration'];timestamp=frame['timestamp_uncertainty']
            if not uncertainty or registration['status']!='available' or timestamp['status']!='available':reasons.append('missing_registered_uncertainty_bounds')
            elif registration['value']['max_error_m']>uncertainty['registration_max_m'] or timestamp['value']['max_error_ns']>uncertainty['timestamp_max_ns']:reasons.append('truth_uncertainty_exceeds_policy')
            if f['track_ids']['status']=='available':
                ids=pf.plane(f['track_ids']['value']);s._require(bool(np.all(ids>=-1)),'track IDs below -1')
                supported=ids>=0;track_ids=ids[supported]
                s._require(len(np.unique(track_ids))==len(track_ids),'duplicate track IDs within frame')
                s._require(bool(np.all(sample[supported])),'track IDs require independent sampled truth')
            elif static_grid:
                ids=np.arange(truth.size,dtype=np.int64).reshape(truth.shape);supported=sample;track_ids=ids[supported]
            else:
                reasons.append('missing_independent_correspondence');ids=np.full(truth.shape,-1,dtype=np.int64);supported=ids>=0;track_ids=ids[supported]
            seen_tracks=np.union1d(seen_tracks,track_ids)
            s._require(len(seen_tracks)<=maximum,'track population/state exceeds max_tracks')
            # Independent geometric consistency is checked before output visibility.
            order=np.argsort(track_ids);sorted_ids=track_ids[order];z=truth[supported][order]
            if static_grid and f['track_ids']['status']!='available':
                common,old,new=np.intersect1d(implicit_ids,sorted_ids,return_indices=True)
                s._require(bool(np.all(np.abs(implicit_z[old]-z[new])<=limits['track_position_tolerance_m'])),'implicit pixel tracks require constant sampled truth')
                union=np.union1d(implicit_ids,sorted_ids);s._require(len(union)<=maximum,'track state exceeds max_tracks')
                merged=np.zeros(len(union));merged[np.searchsorted(union,implicit_ids)]=implicit_z;merged[np.searchsorted(union,sorted_ids)]=z
                implicit_ids,implicit_z=union,merged
            if seq['scene_motion']=='static' and (stationary_grid or f['world_from_left']['status']=='available'):
                # Perspective coordinates from the independently bound calibration snapshot.
                cal=bundle._calibrations[identity]['left'];v,u=np.nonzero(supported);u=u[order];v=v[order]
                xyz=np.column_stack(((u-cal['cx'])*z/cal['fx'],(v-cal['cy'])*z/cal['fy'],z));pose=np.asarray(f['world_from_left']['value']['matrix']) if f['world_from_left']['status']=='available' else np.eye(4);points=xyz@pose[:3,:3].T+pose[:3,3]
                common,old,new=np.intersect1d(world_ids,sorted_ids,return_indices=True)
                s._require(bool(np.all(np.linalg.norm(world_points[old]-points[new],axis=1)<=limits['track_position_tolerance_m'])),'tracked static world positions inconsistent with independent pose/truth')
                union=np.union1d(world_ids,sorted_ids);s._require(len(union)<=maximum,'track state exceeds max_tracks')
                merged=np.zeros((len(union),3));merged[np.searchsorted(union,world_ids)]=world_points;merged[np.searchsorted(union,sorted_ids)]=points;world_ids,world_points=union,merged
            for event in seq['recovery_events']:
                if event['start_input_identity']==identity:
                    targets=np.asarray(event['target_track_ids'],dtype=np.int64)
                    s._require(bool(np.all(np.isin(targets,track_ids))),'recovery targets unsupported at event start')
                    entry=dict(event_id=event['event_id'],kind=event['kind'],target_track_ids=event['target_track_ids'],start_capture_ns=time,first_correct_geometry_ns=[None]*len(targets),first_fresh_correct_measurement_ns=[None]*len(targets))
                    events.append(entry);active.append((entry,targets,np.full(len(targets),-1,dtype=np.int64),np.full(len(targets),-1,dtype=np.int64)))
            if r['observation']['status']!='available':
                reasons.append('missing_observation');files.recheck();previous=None;continue
            o=r['observation']['value'];depth=files.plane(o['depth']);valid=files.plane(o['validity'])!=0
            fresh=np.zeros(valid.shape,dtype=bool);counts['valid_output_count']+=int(valid.sum())
            metadata={k:files.plane(o[k]['value']) if o[k]['status']=='available' else None for k in ('provenance','source_ids','age_ns')}
            if any(a is None for a in metadata.values()):
                counts['unknown_freshness_count']+=int(valid.sum());reasons.append('missing_freshness_metadata')
            else:
                provenance,source,age=(metadata[k] for k in ('provenance','source_ids','age_ns'))
                supported_source=np.isin(source,capture_keys);future=source>r['source_id']
                if bool(np.any(valid & (~supported_source | future))):reasons.append('unsupported_or_future_source_identity')
                # Validate exact elapsed capture age for every ancestry we can see.
                at=np.searchsorted(capture_keys,source);safe=np.minimum(at,len(capture_keys)-1);expected=time-capture_times[safe]
                if bool(np.any(valid & supported_source & (age!=expected))):reasons.append('inconsistent_source_age')
                fresh=valid & (age==0) & (source==r['source_id']) & np.isin(provenance,[1,4,5])
                counts['derived_current_source_count']+=int(np.sum(fresh & (provenance==5)))
                counts['fresh_observation_count']+=int(fresh.sum());counts['historical_count']+=int(np.sum(valid & ~fresh))
                if bool(np.any(valid & (provenance==3))):reasons.append('fused_ancestry_incomplete')
            visible=supported & valid
            output.update(ids[visible],depth[visible].astype(np.float64)-truth[visible]);observed=visible & fresh
            fresh_moments.update(ids[observed],depth[observed].astype(np.float64)-truth[observed])
            if static_grid:raw.update(ids[visible],depth[visible].astype(np.float64))
            correct=_correct(depth,truth,definition,g)
            if correct is not None:
                for entry,targets,geometry_times,fresh_times in active:
                    correct_ids=ids[visible & correct];fresh_ids=ids[observed & correct]
                    geometry_times[(geometry_times<0)&np.isin(targets,correct_ids)]=time-entry['start_capture_ns']
                    fresh_times[(fresh_times<0)&np.isin(targets,fresh_ids)]=time-entry['start_capture_ns']
            if stationary_grid and definition is not None and previous is not None:
                old_depth,old_sample=previous;negative=frame['known_negative'];known=sample | (negative if negative is not None else False)
                changed=old_sample & known & ((sample & (np.abs(truth-old_depth)>np.maximum(limits['ghost_change_absolute_m'],limits['ghost_change_relative']*np.abs(old_depth)))) | (negative if negative is not None else False))
                old_correct=_correct(depth,old_depth,definition,g);current_bad=(sample & ~correct) | (negative if negative is not None else False)
                ghost_population+=int(changed.sum());ghost_count+=int(np.sum(changed & valid & old_correct & current_bad))
            previous=(truth.copy(),sample.copy()) if stationary_grid else None
            max_state=max(max_state,len(seen_tracks))
            files.recheck()
            if f['track_ids']['status']=='available':
                d=f['track_ids']['value'];pf._unchanged(d['path'],pcache[1][d['path']])
    if len(members)<2:reasons.append('insufficient_unique_captures')
    for entry,targets,geometry_times,fresh_times in active:
        entry['status']='available' if definition is not None else 'unavailable'
        entry['reason']=None if definition is not None else 'missing_correctness_threshold'
        entry['first_correct_geometry_ns']=[int(t) if t>=0 else None for t in geometry_times]
        entry['first_fresh_correct_measurement_ns']=[int(t) if t>=0 else None for t in fresh_times]
        entry['geometry_censored_count']=int(np.sum(geometry_times<0));entry['fresh_censored_count']=int(np.sum(fresh_times<0))
    ghost=dict(status='available',ghost_compatible_count=ghost_count,changed_known_truth_count=ghost_population) if stationary_grid and definition is not None else dict(status='unavailable',reason='missing_correctness_threshold' if definition is None else 'independent_dense_visibility_reprojection_not_provided_for_moving_rig')
    return dict(**counts,unique_capture_count=len(members),gaps=gaps,output_error_jitter=output.report(),fresh_error_jitter=fresh_moments.report(),raw_depth_std=raw.report() if static_grid else dict(status='unavailable',reason='raw_depth_std_requires_static_rig_and_scene'),recoveries=events,ghosting=ghost,max_track_state=max_state,reasons=sorted(set(reasons)))
