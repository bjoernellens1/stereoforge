"""Controlled temporal evidence distinguishes output stability from acquisition."""
from copy import deepcopy
from hashlib import sha256
import numpy as np
import pytest
from tests.test_zed_comparison_schema import record, available, unavailable, policy
from tests.test_zed_truth import make_truth
from benchmarks.zed_comparison import schema as s
from benchmarks.zed_comparison.evaluate_temporal import evaluate_temporal, temporal_settings


def plane(root, name, values, fmt='i64_le'):
    dtype={'i64_le':'<i8','f32_le_m':'<f4','u8_mask':'u1','u8_provenance':'u1'}[fmt]
    a=np.asarray(values,dtype=dtype).reshape(2,2); data=a.tobytes(); (root/name).write_bytes(data)
    return dict(path=name,sha256=sha256(data).hexdigest(),size_bytes=len(data),width=2,height=2,stride_bytes=2*a.dtype.itemsize,format=fmt)


def sequence(record, root, *, depths=(1,1,1), motion='stationary', explicit=False):
    records=[]; truth=None
    for i,z in enumerate(depths):
        r=deepcopy(record); r.update(ordinal=i,source_id=i+1,partition='held_out')
        r['timestamps']['capture']['value_ns']=100+i*100
        r['implementation']['settings']['temporal']={k:available(v) for k,v in dict(stabilization=0,tracking=False,filling=False).items()}
        o=r['observation']['value']; o['depth']=plane(root,f'depth-{i}',[z]*4,'f32_le_m');o['validity']=plane(root,f'valid-{i}',[1]*4,'u8_mask')
        for key,vals,fmt in [('provenance',[1]*4,'u8_provenance'),('source_ids',[i+1]*4,'i64_le'),('age_ns',[0]*4,'i64_le')]:o[key]=available(plane(root,f'{key}-{i}',vals,fmt))
        r['input_identity']=s.input_identity(r);records.append(r)
        t=make_truth(r,root,depth=[[1,1],[1,1]])
        desc=t['frames'][0]['depth']; data=(root/desc['path']).read_bytes(); name=f'truth-{i}';(root/name).write_bytes(data);desc['path']=name
        if truth is None:truth=t
        else:truth['frames'].append(t['frames'][0])
    digest=sha256(s._canonical([__import__('json').loads(v) for v in sorted({s._canonical({k:r[k] for k in ('scene_id','sequence_id','partition','input_identity')}) for r in records})]).encode()).hexdigest()
    frames=[dict(input_identity=r['input_identity'],world_from_left=unavailable(),track_ids=available(plane(root,f'tracks-{i}',[0,1,2,3])) if explicit else unavailable()) for i,r in enumerate(records)]
    protocol=dict(schema='stereoforge.zed_comparison.temporal_protocol.v1',frozen=True,partition_digest=digest,provenance=dict(independent=True,**truth['provenance']),limits=dict(max_gap_ns=150,pose_timestamp_tolerance_ns=0,track_position_tolerance_m=.001,max_tracks=8,ghost_change_absolute_m=.1,ghost_change_relative=0),sequences=[dict(scene_id='scene-a',sequence_id='sequence-a',partition='held_out',rig_motion=motion,scene_motion='static',calibration_profile='analytic',frames=frames,recovery_events=[])])
    p=policy();p['freeze_parameters']['correctness']=dict(mode='depth_tolerance',absolute_m=.01,relative=0,combine='max')
    return records,truth,p,protocol


def run(data,root):
    r,t,p,protocol=data
    return evaluate_temporal(r,t,p,root=root,truth_root=root,protocol=protocol,protocol_root=root)


def test_duplicate_frame_not_fresh(record,tmp_path):
    data=sequence(record,tmp_path);data[0].append(deepcopy(data[0][0]))
    out=run(data,tmp_path)
    assert out['repeat_count']==1 and out['groups'][0]['fresh_observation_count']==12
    assert out['groups'][0]['output_error_jitter']['track_count']==4


def test_clock_mismatch_rejected(record,tmp_path):
    data=sequence(record,tmp_path); r=data[0][1];r['timestamps']['capture']['clock_domain']='other';r['input_identity']=s.input_identity(r)
    with pytest.raises(ValueError,match='clock'):run(data,tmp_path)


def test_historical_depth_not_observed(record,tmp_path):
    data=sequence(record,tmp_path)
    for i,r in enumerate(data[0]):r['observation']['value']['provenance']=available(plane(tmp_path,f'history-{i}',[3]*4,'u8_provenance'))
    out=run(data,tmp_path)['groups'][0]
    assert out['fresh_observation_count']==0 and out['historical_count']==12
    assert out['fresh_error_jitter']['track_count']==0


def test_motion_occlusion_recovery(record,tmp_path):
    data=sequence(record,tmp_path,depths=(2,2,1),motion='known',explicit=True)
    seq=data[3]['sequences'][0];seq['scene_motion']='dynamic'
    for r,f in zip(data[0],seq['frames']):f['world_from_left']=available(dict(matrix=np.eye(4).tolist(),capture=r['timestamps']['capture']))
    seq['recovery_events']=[dict(event_id='reveal',kind='disocclusion',start_input_identity=data[0][0]['input_identity'],target_track_ids=[0,1],justification='independently controlled reveal')]
    out=run(data,tmp_path)['groups'][0]
    assert out['recoveries'][0]['first_correct_geometry_ns']==[200,200]
    assert out['recoveries'][0]['first_fresh_correct_measurement_ns']==[200,200]
    assert out['ghosting']['status']=='unavailable'


def test_gap_resets_moments(record,tmp_path):
    data=sequence(record,tmp_path,depths=(1,2,3));data[3]['limits']['max_gap_ns']=50
    out=run(data,tmp_path)['groups'][0]
    assert len(out['gaps'])==2 and out['output_error_jitter']['track_count']==0


def test_settings_missing_and_validation(record):
    assert temporal_settings(record)['state']=='unknown'
    record['implementation']['settings']['temporal']={'stabilization':available(True)}
    with pytest.raises(ValueError):temporal_settings(record)


@pytest.mark.parametrize('case',['duplicate_tracks','bounded','missing_pose','unobserved_target','conflict','mutated'])
def test_protocol_rejections(record,tmp_path,case):
    data=sequence(record,tmp_path,motion='known',explicit=True);seq=data[3]['sequences'][0]
    for r,f in zip(data[0],seq['frames']):f['world_from_left']=available(dict(matrix=np.eye(4).tolist(),capture=r['timestamps']['capture']))
    if case=='duplicate_tracks':seq['frames'][0]['track_ids']=available(plane(tmp_path,'duplicate',[0,0,2,3]))
    elif case=='bounded':data[3]['limits']['max_tracks']=2
    elif case=='missing_pose':seq['frames'][0]['world_from_left']=unavailable()
    elif case=='unobserved_target':seq['recovery_events']=[dict(event_id='x',kind='appearance',start_input_identity=data[0][0]['input_identity'],target_track_ids=[9],justification='x')]
    elif case=='conflict':
        other=deepcopy(data[0][0]);other['observation']['value']['depth']=plane(tmp_path,'conflict',[2]*4,'f32_le_m');data[0].append(other)
    else:(tmp_path/'tracks-0').write_bytes(bytes(32))
    with pytest.raises(ValueError):run(data,tmp_path)


def test_old_source_reset_age_blocks_qualification(record,tmp_path):
    data=sequence(record,tmp_path)
    data[0][1]['observation']['value']['source_ids']=available(plane(tmp_path,'old-source',[1]*4))
    out=run(data,tmp_path)
    assert 'inconsistent_source_age' in out['reasons']
    assert out['groups'][0]['historical_count']==4


def test_moving_tracks_keep_independent_alignment(record,tmp_path):
    data=sequence(record,tmp_path,motion='known',explicit=True);seq=data[3]['sequences'][0]
    # One physical plane translated one pixel relative to the optical frame.
    for i,(r,f) in enumerate(zip(data[0],seq['frames'])):
        transform=np.eye(4);transform[0,3]=i*.2
        f['world_from_left']=available(dict(matrix=transform.tolist(),capture=r['timestamps']['capture']))
        f['track_ids']=available(plane(tmp_path,f'move-{i}',[i, i+1, -1,-1]))
    # Repeated IDs move from x=1 to x=0 while the pose translates by one pixel.
    out=run(data,tmp_path)
    assert out['groups'][0]['output_error_jitter']['track_count']==2


def test_static_world_position_inconsistency_rejected(record,tmp_path):
    data=sequence(record,tmp_path,motion='known',explicit=True)
    for i,(r,f) in enumerate(zip(data[0],data[3]['sequences'][0]['frames'])):
        transform=np.eye(4);transform[0,3]=i
        f['world_from_left']=available(dict(matrix=transform.tolist(),capture=r['timestamps']['capture']))
    with pytest.raises(ValueError,match='world positions'):run(data,tmp_path)


def test_unrecovered_targets_are_censored(record,tmp_path):
    data=sequence(record,tmp_path,depths=(2,2,2))
    data[3]['sequences'][0]['recovery_events']=[dict(event_id='reset',kind='motion_reset',start_input_identity=data[0][0]['input_identity'],target_track_ids=[0],justification='controlled reset')]
    row=run(data,tmp_path)['groups'][0]['recoveries'][0]
    assert row['first_correct_geometry_ns']==[None] and row['geometry_censored_count']==1


def test_missing_threshold_and_motion_correspondence_are_unavailable(record,tmp_path):
    data=sequence(record,tmp_path,motion='estimated');data[2]['freeze_parameters']['correctness']=None
    out=run(data,tmp_path)
    assert 'missing_independent_correspondence' in out['reasons']
    assert out['groups'][0]['ghosting']['status']=='unavailable'


def test_stationary_dynamic_ghost_against_known_current_truth(record,tmp_path):
    data=sequence(record,tmp_path,explicit=True);data[3]['sequences'][0]['scene_motion']='dynamic'
    f=data[1]['frames'][1];f['depth']=plane(tmp_path,'changed-truth',[2]*4,'f32_le_m')
    # Prior depth remains numerically stable while independent current truth changes.
    out=run(data,tmp_path)['groups'][0]
    assert out['ghosting']['ghost_compatible_count']==4
    assert out['ghosting']['changed_known_truth_count']==8


def test_protocol_file_mutation_and_track_bound(record,tmp_path,monkeypatch):
    import benchmarks.zed_comparison.evaluate_temporal as module
    data=sequence(record,tmp_path,explicit=True);original=module._correct;called=[]
    def mutate(*args):
        if not called:
            path=tmp_path/'tracks-0';path.write_bytes(path.read_bytes());called.append(1)
        return original(*args)
    monkeypatch.setattr(module,'_correct',mutate)
    with pytest.raises(ValueError,match='changed'):run(data,tmp_path)


def test_complete_frozen_controlled_lane_can_qualify(record,tmp_path):
    data=sequence(record,tmp_path);p=data[2];p['frozen']=True
    p['freeze_parameters'].update(region_definitions={r:dict(description='declared region') for r in p['required_regions']},zero_reference_error=dict(method='absolute_margin',absolute_mae_margin_m=.01),uncertainty=dict(method='analytic bounds',confidence_level=.95,registration_max_m=.01,timestamp_max_ns=10),statistics=dict(method='controlled',unit='scene_sequence',min_scenes=1,min_sequences=1),partition_digest=data[3]['partition_digest'])
    out=run(data,tmp_path)
    assert out['quality_qualified'],out['reasons']
    assert out['groups'][0]['max_track_state']==4


def test_stationary_declaration_rejects_moving_supplied_pose(record,tmp_path):
    data=sequence(record,tmp_path,explicit=True);seq=data[3]['sequences'][0];seq['scene_motion']='dynamic'
    for i,(r,f) in enumerate(zip(data[0],seq['frames'])):
        pose=np.eye(4);pose[0,3]=i*.1
        f['world_from_left']=available(dict(matrix=pose.tolist(),capture=r['timestamps']['capture']))
    with pytest.raises(ValueError,match='stationary rig'):run(data,tmp_path)


def test_method_cannot_qualify_selected_subset(record,tmp_path):
    data=sequence(record,tmp_path);other=deepcopy(data[0][0]);other['implementation']['name']='subset';data[0].append(other)
    out=run(data,tmp_path)
    assert 'method_input_set_mismatch' in out['reasons']


def test_measured_without_physical_metadata_is_scoreable_unqualified(record,tmp_path):
    data=sequence(record,tmp_path);data[1]['provenance']['kind']='measured'
    out=run(data,tmp_path)
    assert 'missing_physical_reference_metadata' in out['reasons']
    assert out['groups'][0]['output_error_jitter']['track_count']==4


def test_duplicate_delivery_with_descriptor_alias_counts_once(record,tmp_path):
    data=sequence(record,tmp_path);other=deepcopy(data[0][0]);desc=other['observation']['value']['depth'];name='depth-alias'
    (tmp_path/name).write_bytes((tmp_path/desc['path']).read_bytes());desc['path']=name;data[0].append(other)
    assert run(data,tmp_path)['repeat_count']==1


def test_historical_geometry_can_recover_before_fresh_measurement(record,tmp_path):
    data=sequence(record,tmp_path)
    for i in (0,1):data[0][i]['observation']['value']['provenance']=available(plane(tmp_path,f'reprojected-{i}',[2]*4,'u8_provenance'))
    data[3]['sequences'][0]['recovery_events']=[dict(event_id='reveal',kind='disocclusion',start_input_identity=data[0][0]['input_identity'],target_track_ids=[0],justification='controlled reveal')]
    event=run(data,tmp_path)['groups'][0]['recoveries'][0]
    assert event['first_correct_geometry_ns']==[0]
    assert event['first_fresh_correct_measurement_ns']==[200]


@pytest.mark.parametrize('case',['root_list','protocol_extra','pose_clock','unsafe_track'])
def test_strict_admission(record,tmp_path,case):
    data=sequence(record,tmp_path,explicit=True)
    if case=='root_list':
        with pytest.raises(ValueError,match='root list'):
            evaluate_temporal(*data[:3],root=[tmp_path],truth_root=tmp_path,protocol=data[3],protocol_root=tmp_path)
        return
    if case=='protocol_extra':data[3]['extra']=1
    elif case=='unsafe_track':data[3]['sequences'][0]['frames'][0]['track_ids']['value']['path']='../escape'
    else:data[3]['sequences'][0]['frames'][0]['world_from_left']=available(dict(matrix=np.eye(4).tolist(),capture=dict(value_ns=100,clock_domain='other')))
    with pytest.raises(ValueError):run(data,tmp_path)


def test_moments_identical_ids_avoid_reassociation(record,monkeypatch):
    from benchmarks.zed_comparison.evaluate_temporal import _Moments
    moments=_Moments(4);ids=np.arange(4,dtype=np.int64)
    moments.update(ids,np.array([1.,2.,3.,4.]))
    def forbidden(*args,**kwargs):
        raise AssertionError('unchanged track IDs must avoid reassociation')
    monkeypatch.setattr(np,'union1d',forbidden);monkeypatch.setattr(np,'searchsorted',forbidden)
    moments.update(ids,np.array([3.,4.,5.,6.]))
    np.testing.assert_array_equal(moments.n,[2]*4)
    np.testing.assert_allclose(moments.mean,[2,3,4,5])
    np.testing.assert_allclose(moments.m2,[2]*4)
    assert moments.report()['mean_track_error_std_m']==pytest.approx(np.sqrt(2))


@pytest.mark.parametrize('reorder',[False,True])
def test_moments_reassociation_gap_and_fastpath_parity(reorder):
    from benchmarks.zed_comparison.evaluate_temporal import _Moments
    moments=_Moments(4)
    moments.update(np.array([1,3]),np.array([2.,8.]))
    ids=np.array([3,1]) if reorder else np.array([1,3])
    moments.update(ids,np.array([10.,4.]) if reorder else np.array([4.,10.]))
    np.testing.assert_allclose(moments.mean,[3.,9.]);np.testing.assert_allclose(moments.m2,[2.,2.])
    moments.update(np.array([0,3]),np.array([5.,12.]))
    np.testing.assert_array_equal(moments.ids,[0,1,3]);np.testing.assert_array_equal(moments.n,[1,2,3])
    np.testing.assert_allclose(moments.mean,[5,3,10]);np.testing.assert_allclose(moments.m2,[0,2,8])
    moments.reset();moments.update(np.array([0,1,3]),np.array([1.,2.,3.]))
    moments.update(np.array([0,1,3]),np.array([3.,4.,5.]))
    report=moments.report();assert report['track_count']==5 and report['observation_count']==11
    assert report['std_ddof']==1


def test_moments_fastpath_empty_bounds_and_overflow():
    from benchmarks.zed_comparison.evaluate_temporal import _Moments
    moments=_Moments(2);moments.update(np.array([],dtype=np.int64),np.array([]))
    assert moments.report()['track_count']==0
    with pytest.raises(ValueError,match='max_tracks'):moments.update(np.arange(3),np.ones(3))
    moments.update(np.array([1]),np.array([1e308]))
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(ValueError,match='overflow'):moments.update(np.array([1]),np.array([-1e308]))
    for value in (np.inf,np.nan):
        with pytest.raises(ValueError,match='nonfinite'):moments.update(np.array([1]),np.array([value]))
