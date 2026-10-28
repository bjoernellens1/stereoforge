import copy
import pytest
import benchmarks.zed_comparison.report as report
from benchmarks.zed_comparison.report import build_report

POLICY = dict(schema='stereoforge.zed_comparison.report_policy.v1', frozen=True,
              quality=None, delivery=None, require_temporal=False, require_view=False)


def test_missing_quality_blocks_promotion():
    result=build_report([], POLICY)
    assert not result['qualified']
    assert 'missing_quality' in result['reasons']
    assert 'missing_timing' in result['reasons']
    assert 'missing_delivery' in result['reasons']


def test_summary_flags_not_accepted_as_raw_evidence():
    forged=[dict(kind='quality', payload={'parity_qualified':True,'status':'qualified'})]
    with pytest.raises(ValueError): build_report(forged,POLICY)


def test_duplicate_tracks_and_unknown_keys_rejected():
    delivery=dict(kind='delivery',payload={'events':[]})
    with pytest.raises(ValueError): build_report([delivery,delivery],POLICY)
    with pytest.raises(ValueError): build_report([dict(delivery,qualified=True)],POLICY)


def test_provisional_policy_cannot_promote():
    policy=copy.deepcopy(POLICY);policy['frozen']=False
    assert 'provisional_report_policy' in build_report([],policy)['reasons']


def test_real_empty_evaluators_remain_unqualified():
    quality=dict(kind='quality',payload=dict(records=[],truth=None,root=None,truth_root=None))
    delivery=dict(kind='delivery',payload=dict(events=[]))
    result=build_report([quality,delivery],POLICY)
    assert not result['qualified']
    assert 'quality_unqualified' in result['reasons']
    assert 'delivery_unqualified' in result['reasons']
    assert result['tracks']['quality']['parity_qualified'] is False


def test_view_label_alone_is_not_execution_evidence():
    policy=dict(POLICY,require_view=True)
    result=build_report([dict(kind='delivery',payload=dict(events=[]))],policy)
    assert 'view_execution_and_cost_not_verified' in result['reasons']


def cross_track_fixture():
    """A binding fixture; evaluators are stubbed, so this is not qualification evidence."""
    temporal={k:dict(status='available', value=v) for k,v in
              [('stabilization',0),('tracking',False),('filling',False)]}
    profile=dict(algorithm='independent_sgm',width=1280,height=720,disparities=128,paths=4,
                 p1=8,p2=32,subpixel=True,lr_tolerance=1.0,right_reference_mode='independent_sgm',
                 depth_units='axial_Z_metres_rectified_rays',temporal=temporal,compiler_sha256='a'*64,
                 stereo_spirv_sha256={'daxa-stereo-census.spv':'b'*64},
                 depth_spirv_sha256={'daxa-depth-metric.spv':'c'*64})
    device=dict(device_uuid='1'*32,driver_uuid='2'*32,vendor_id=4098,device_id=123,driver_version=10)
    common=dict(name='StereoForge',commit=dict(status='available',value='d'*40),
                device=dict(status='available',value=device))
    quality_impl=dict(common,version='canonical_rgb',settings=dict(effective_estimator=copy.deepcopy(profile)))
    raw='e'*64; rect='f'*64
    live_impl=dict(common,version='native_ros_yuyv_v1',settings=dict(effective_estimator=copy.deepcopy(profile),
                   original_calibration_identity=raw,rectified_calibration_identity=rect))
    geometry=dict(width=1280,height=720,fx=528,fy=528,baseline_m=.119,cx_left=640,cx_right=640)
    calibration=dict(raw=dict(path='raw.json',sha256=raw,size_bytes=10),
                     rectified=dict(path='rect.json',sha256=rect,size_bytes=10),
                     resolved_file=dict(status='unavailable',reason='none'),geometry=geometry)
    record=dict(implementation=quality_impl,calibration=calibration)
    method=report._digest(report._method(record)); stratum=report._digest(report.s._calibration_identity(calibration))
    quality_result=dict(parity_qualified=True,comparisons=[dict(candidate_method=method,stratum_key=stratum,parity_qualified=True)],
                        methods=[dict(method_key=method,implementation=quality_impl)],
                        frames=[dict(method_key=method,stratum_key=stratum,status='available')])
    timing_record=dict(method_key=method,calibration_identity=stratum,session_id='s',phase='warm',profile_id='p')
    timing_result=dict(comparisons=[dict(qualified=True,session_id='s',phase='warm',profiles=['p','q'])])
    run=dict(implementation=dict(status='available',value=live_impl),calibration_identity=rect,
             effective_estimator=copy.deepcopy(profile),
             output_profiles=dict(depth=dict(width=1280,height=720,encoding='32FC1')))
    tracks=[dict(kind='quality',payload=dict(records=[record],truth=None,root=None,truth_root=None)),
            dict(kind='timing',payload=dict(records=[timing_record],root=None)),
            dict(kind='delivery',payload=dict(events=[]))]
    return tracks,quality_result,timing_result,dict(qualified=True,runs=[run])


def _run_binding(monkeypatch, fixture):
    tracks,q,t,d=fixture
    monkeypatch.setattr(report,'compare_quality',lambda *a,**k:q)
    monkeypatch.setattr(report,'summarize_timing',lambda *a,**k:t)
    monkeypatch.setattr(report,'summarize_delivery',lambda *a,**k:d)
    return build_report(tracks,POLICY)


def test_matching_estimator_and_composite_calibration_bind_distinct_ingestion(monkeypatch):
    fixture=cross_track_fixture()
    assert fixture[0][1]['payload']['records'][0]['calibration_identity'] != fixture[3]['runs'][0]['calibration_identity']
    result = _run_binding(monkeypatch,fixture)
    assert not result['qualified']
    assert result['reasons'] == ['processing_budget_not_verified']


def test_qualified_timing_comparison_does_not_establish_processing_budget(monkeypatch):
    result = _run_binding(monkeypatch, cross_track_fixture())
    assert result['tracks']['timing']['comparisons'][0]['qualified'] is True
    assert not result['qualified']
    assert 'processing_budget_not_verified' in result['reasons']


def test_temporal_off_sequence_cannot_qualify_reconstruction(monkeypatch):
    tracks, q, t, d = cross_track_fixture()
    temporal = dict(kind='temporal', payload=dict(records=[], truth=None, root=None,
                    truth_root=None, protocol=None, protocol_root=None))
    monkeypatch.setattr(report, 'compare_quality', lambda *a, **k: q)
    monkeypatch.setattr(report, 'summarize_timing', lambda *a, **k: t)
    monkeypatch.setattr(report, 'summarize_delivery', lambda *a, **k: d)
    # Even a qualified sequence with the identical temporal-off method is not
    # evidence that temporal reconstruction was enabled or bound to delivery.
    monkeypatch.setattr(report, 'evaluate_temporal', lambda *a, **k: dict(
        quality_qualified=True, partition_digest=q.get('partition_digest'),
        groups=[dict(method_key=q['methods'][0]['method_key'])]))
    result = build_report(tracks + [temporal], dict(POLICY, require_temporal=True))
    assert not result['qualified']
    assert 'temporal_reconstruction_binding_not_verified' in result['reasons']


def test_temporal_requirement_fails_closed_without_track():
    result = build_report([], dict(POLICY, require_temporal=True))
    assert 'missing_temporal' in result['reasons']
    assert 'temporal_reconstruction_binding_not_verified' in result['reasons']


@pytest.mark.parametrize('mutation', ['commit','commit_unknown','device','driver','estimator','raw','rectified',
                                      'dimensions','missing_profile','quality_stratum','quality_unavailable','no_depth'])
def test_cross_track_binding_rejects_changed_or_missing_evidence(monkeypatch, mutation):
    fixture=cross_track_fixture();tracks,q,_,d=fixture
    run=d['runs'][0]; live=run['implementation']['value'];record=tracks[0]['payload']['records'][0]
    if mutation == 'commit': live['commit']=dict(status='available',value='0'*40)
    elif mutation == 'commit_unknown': live['commit']=dict(status='unavailable',reason='unknown')
    elif mutation == 'device': live['device']['value']=dict(live['device']['value'],device_uuid='3'*32)
    elif mutation == 'driver': live['device']['value']=dict(live['device']['value'],driver_version=11)
    elif mutation == 'estimator': run['effective_estimator']['p2']=33
    elif mutation == 'raw': live['settings']['original_calibration_identity']='0'*64
    elif mutation == 'rectified': run['calibration_identity']='0'*64
    elif mutation == 'dimensions': record['calibration']['geometry']['width']=1279
    elif mutation == 'missing_profile': del live['settings']['effective_estimator']
    elif mutation == 'quality_stratum': q['comparisons'][0]['stratum_key']='0'*64
    elif mutation == 'quality_unavailable': q['frames'][0]['status']='unavailable'
    elif mutation == 'no_depth': run['output_profiles']={}
    result=_run_binding(monkeypatch,fixture)
    assert not result['qualified'] and 'quality_delivery_binding_mismatch' in result['reasons']


@pytest.mark.parametrize('mutation',['stratum','session','phase','profile','comparison'])
def test_timing_must_be_in_qualified_comparison_for_same_method_and_stratum(monkeypatch,mutation):
    fixture=cross_track_fixture();tracks,_,timing,_=fixture
    row=tracks[1]['payload']['records'][0]
    if mutation == 'stratum': row['calibration_identity']='0'*64
    elif mutation == 'session': row['session_id']='different'
    elif mutation == 'phase': row['phase']='cold'
    elif mutation == 'profile': row['profile_id']='other'
    elif mutation == 'comparison': timing['comparisons'][0]['qualified']=False
    result=_run_binding(monkeypatch,fixture)
    assert not result['qualified'] and 'timing_delivery_binding_mismatch' in result['reasons']
