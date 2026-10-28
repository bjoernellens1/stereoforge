"""Independent-truth evaluation keeps missing evidence separate from a pass."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from benchmarks.zed_comparison.schema import load_policy, validate_policy

ROOT = Path(__file__).resolve().parents[1]


def test_policy_dict_and_path_validation_share_semantics_without_mutation():
    path = ROOT/'configs/benchmarks/zed_quality_policy.json'
    policy = json.loads(path.read_text()); original = deepcopy(policy)
    assert validate_policy(policy) == load_policy(path)
    assert policy == original
    assert validate_policy(policy)['status'] == 'unqualified'


def test_policy_dict_rejects_unknown_or_nonfinite_values():
    policy = json.loads((ROOT/'configs/benchmarks/zed_quality_policy.json').read_text())
    for change in ({**policy, 'invented': True}, {**policy, 'margins': {**policy['margins'], 'relative_mae_ratio': float('nan')}}):
        with pytest.raises(ValueError): validate_policy(change)

from tests.test_zed_comparison_schema import record, available, unavailable
from tests.test_zed_truth import make_truth
from benchmarks.zed_comparison import schema


def policy():
    return json.loads((ROOT/'configs/benchmarks/zed_quality_policy.json').read_text())


def compare(records, truth, p, **kwargs):
    from benchmarks.zed_comparison.compare_quality import compare_quality
    return compare_quality(records, truth, p, **kwargs)


def test_unexecuted_sdk_never_parity_pass(record,tmp_path):
    t=make_truth(record,tmp_path)
    sdk=deepcopy(record); sdk['implementation']['name']='ZED SDK'; sdk['implementation']['settings']['depth_mode']='NEURAL_LIGHT'; sdk['observation']=unavailable('SDK not executed')
    result=compare([record,sdk],t,policy(),root=tmp_path,truth_root=tmp_path)
    assert result['parity_qualified'] is False
    assert 'missing_executed_reference' in result['reasons']
    assert any(f['status']=='unavailable' for f in result['frames'])


def test_missing_roots_unqualified(record,tmp_path):
    t=make_truth(record,tmp_path)
    result=compare([record],t,policy())
    assert result['parity_qualified'] is False
    assert 'missing_record_root' in result['reasons']
    assert 'missing_truth_root' in result['reasons']


def test_changed_bytes_rejected(record,tmp_path):
    t=make_truth(record,tmp_path); (tmp_path/'depth').write_bytes(b'x'*16)
    with pytest.raises(ValueError): compare([record],t,policy(),root=tmp_path,truth_root=tmp_path)


def test_duplicate_and_cross_partition_methods_rejected(record,tmp_path):
    t=make_truth(record,tmp_path)
    for other in (deepcopy(record),{**deepcopy(record),'partition':'held_out'}):
        with pytest.raises(ValueError): compare([record,other],t,policy(),root=tmp_path,truth_root=tmp_path)


def test_same_input_metrics_and_intersection(record,tmp_path):
    t=make_truth(record,tmp_path,depth=[[1,1],[1,2]])
    other=deepcopy(record); other['implementation']['name']='reference fixture'
    result=compare([record,other],t,policy(),root=[tmp_path,tmp_path],truth_root=tmp_path)
    assert len(result['methods'])==2 and len(result['frames'])==2
    assert result['frames'][0]['metrics']['lanes']['requested']['truth_count']==4
    assert result['frames'][0]['metrics']['lanes']['requested']['accepted_count']==2
    pair=result['intersections'][0]
    assert pair['intersection_count']==2
    assert pair['candidate']['mae_m']==0
    assert pair['candidate']['truth_count']==4
    aggregate=result['strata'][0]['methods'][0]['metrics']['lanes']['requested']
    assert aggregate['p95_abs_error_m'] is None
    assert aggregate['p95_unavailable_reason']
    json.dumps(result,allow_nan=False)


def test_missing_truth_and_policy_are_explicit(record,tmp_path):
    t=make_truth(record,tmp_path);t['frames']=[]
    result=compare([record],t,None,root=tmp_path,truth_root=tmp_path)
    assert 'missing_policy' in result['reasons']
    assert result['frames'][0]['reason']=='missing_truth'
    assert not result['parity_qualified']


def qualified_fixture(record,tmp_path):
    record['partition']='held_out'
    record['implementation']['settings']['temporal'] = dict(
        stabilization=available(0), tracking=available(False), filling=available(False))
    from hashlib import sha256
    import struct
    for name, data in [('depth',struct.pack('<4f',1,1,2,2)),('validity',bytes([1]*4))]:
        desc=record['observation']['value'][name]
        (tmp_path/desc['path']).write_bytes(data)
        desc.update(sha256=sha256(data).hexdigest(),size_bytes=len(data))
    t=make_truth(record,tmp_path,depth=[[1,1],[2,2]])
    f=t['frames'][0]
    f['masks']['boundary_edges']=available(deepcopy(f['masks']['sample']))
    for name in ('thin','boundary','near'):
        f['regions'][name]=available(deepcopy(f['masks']['sample']))
    sdk=deepcopy(record);sdk['implementation']['name']='ZED SDK';sdk['implementation']['settings']['depth_mode']='NEURAL_LIGHT'
    p=policy(); p['frozen']=True
    digest=compare([record,sdk],t,None,root=tmp_path,truth_root=tmp_path)['partition_digest']
    p['freeze_parameters']=dict(correctness=dict(mode='depth_tolerance',absolute_m=.01,relative=0,combine='max'),
        region_definitions={r:dict(description='independent fixture support') for r in p['required_regions']},
        zero_reference_error=dict(method='absolute_margin',absolute_mae_margin_m=.01),
        uncertainty=dict(method='declared bounded registration',confidence_level=.95,registration_max_m=.01,timestamp_max_ns=10),
        statistics=dict(method='hierarchical_scene_sequence_bootstrap_v1',unit='scene_sequence',min_scenes=1,min_sequences=1),partition_digest=digest)
    p['freeze_parameters']['region_definitions']['boundary'].update(absolute_jump_m=.1,relative_jump=0.,tolerance_pixels=0)
    return [record,sdk],t,p


def test_frozen_held_out_declared_executed_pair_can_pass(record,tmp_path):
    records,t,p=qualified_fixture(record,tmp_path)
    before=deepcopy((records,t,p))
    out=compare(records,t,p,root=tmp_path,truth_root=tmp_path)
    assert out['parity_qualified'] is True
    assert out['comparisons'][0]['statistics']['statistics_pass'] is True
    assert len(out['comparisons'][0]['paired_units'])==4
    assert (records,t,p)==before


@pytest.mark.parametrize('gate', ['digest','region','registration','timestamp','bound','units','statistics','heldout','candidate'])
def test_external_missing_gates_cannot_pass(record,tmp_path,gate):
    records,t,p=qualified_fixture(record,tmp_path)
    if gate=='digest':p['freeze_parameters']['partition_digest']='0'*64
    elif gate=='region':t['frames'][0]['regions']['thin']=unavailable('missing thin support')
    elif gate=='registration':t['frames'][0]['registration']=unavailable('not registered')
    elif gate=='timestamp':t['frames'][0]['timestamp_uncertainty']=unavailable('no bound')
    elif gate=='bound':t['frames'][0]['registration']['value']['max_error_m']=.1
    elif gate=='units':p['freeze_parameters']['statistics']['min_scenes']=2
    elif gate=='statistics':p['freeze_parameters']['statistics']['method']='unknown'
    elif gate=='candidate':records=records[1:]
    else:
        for r in records:r['partition']='development'
        t['frames'][0]['partition']='development'
    out=compare(records,t,p,root=tmp_path,truth_root=tmp_path)
    assert not out['parity_qualified']
    assert out['reasons']


def test_per_calibration_strata_are_separate(record,tmp_path):
    from hashlib import sha256
    t=make_truth(record,tmp_path)
    other=deepcopy(record); other['implementation']['name']='other calibration'
    cal=json.loads((tmp_path/'rect.json').read_text())
    data=json.dumps(cal,indent=1).encode(); (tmp_path/'rect-other.json').write_bytes(data)
    other['calibration']['rectified']=dict(path='rect-other.json',sha256=sha256(data).hexdigest(),size_bytes=len(data))
    other['input_identity']=schema.input_identity(other)
    f=deepcopy(t['frames'][0]); f['input_identity']=other['input_identity'];f['calibration']=deepcopy(other['calibration']['rectified']);t['frames'].append(f)
    out=compare([record,other],t,policy(),root=tmp_path,truth_root=tmp_path)
    assert len(out['strata'])==2
    assert out['intersections']==[]


def test_observed_file_mutation_during_metrics_rejected(record,tmp_path,monkeypatch):
    import benchmarks.zed_comparison.compare_quality as module
    t=make_truth(record,tmp_path); original=module.evaluate_frame
    def mutate(*args,**kwargs):
        out=original(*args,**kwargs)
        (tmp_path/'depth').write_bytes((tmp_path/'depth').read_bytes())
        return out
    monkeypatch.setattr(module,'evaluate_frame',mutate)
    with pytest.raises(ValueError): compare([record],t,policy(),root=tmp_path,truth_root=tmp_path)


def test_aggregate_uses_sums_and_fixed_denominators(record,tmp_path):
    t=make_truth(record,tmp_path,depth=[[1,1],[1,2]])
    other=deepcopy(record);other['ordinal']=1;other['source_id']-=1
    other['input_identity']=schema.input_identity(other)
    f=deepcopy(t['frames'][0]);f['input_identity']=other['input_identity'];f['source_id']=other['source_id'];t['frames'].append(f)
    out=compare([record,other],t,policy(),root=tmp_path,truth_root=tmp_path)
    pop=out['strata'][0]['methods'][0]['metrics']['lanes']['requested']
    assert pop['truth_count']==8 and pop['accepted_count']==4
    assert pop['completeness']==.5 and pop['mae_m']==0


def test_loader_policy_status_roundtrip(record,tmp_path):
    t=make_truth(record,tmp_path)
    enriched=load_policy(ROOT/'configs/benchmarks/zed_quality_policy.json')
    out=compare([record],t,enriched,root=tmp_path,truth_root=tmp_path)
    assert 'provisional_policy' in out['reasons']
    enriched['status']='policy_ready'
    with pytest.raises(ValueError):compare([record],t,enriched,root=tmp_path,truth_root=tmp_path)
    out=compare([record],t,{'status':'unqualified','reason':'missing_policy'},root=tmp_path,truth_root=tmp_path)
    assert 'missing_policy' in out['reasons']


def test_candidate_missing_reference_inputs_cannot_select_easy_subset(record,tmp_path):
    records,t,p=qualified_fixture(record,tmp_path)
    sdk=deepcopy(records[1]);sdk['ordinal']=1;sdk['source_id']-=1;sdk['input_identity']=schema.input_identity(sdk)
    f=deepcopy(t['frames'][0]);f['input_identity']=sdk['input_identity'];f['source_id']=sdk['source_id'];t['frames'].append(f)
    records.append(sdk)
    digest=compare(records,t,None,root=tmp_path,truth_root=tmp_path)['partition_digest']
    p['freeze_parameters']['partition_digest']=digest
    out=compare(records,t,p,root=tmp_path,truth_root=tmp_path)
    assert not out['parity_qualified']
    assert 'candidate_reference_input_set_mismatch' in out['reasons']

from tests.test_zed_native_observation import executed
from tests.test_zed_native_input import native_record


def test_actual_native_left_depth_integration(executed,tmp_path):
    from benchmarks.zed_comparison.native_observation import export_native_observation
    source,source_path,report,report_path,dump=executed
    path=export_native_observation(report_path,source_path,dump,tmp_path/'portable')
    native=schema.load_record(path)
    t=make_truth(source,tmp_path,depth=[[2]*4]*4)
    out=compare([native],t,policy(),root=[path.parent],truth_root=tmp_path)
    pop=out['frames'][0]['metrics']['lanes']['requested']
    assert pop['truth_count']==16 and pop['accepted_count']==15
    assert pop['abs_error_sum_m']==0 and pop['mae_m']==0
    assert pop['completeness']==15/16
    assert not out['parity_qualified']


@pytest.mark.parametrize('roots',[[],[None],['one','two']])
def test_root_list_must_match_record_order(record,tmp_path,roots):
    t=make_truth(record,tmp_path)
    with pytest.raises(ValueError):compare([record],t,policy(),root=roots,truth_root=tmp_path)


def test_method_identity_includes_device_and_settings(record,tmp_path):
    t=make_truth(record,tmp_path);other=deepcopy(record)
    other['implementation']['device']=available({'device':'other'})
    changed=deepcopy(record);changed['implementation']['settings']['variant']='second'
    out=compare([record,other,changed],t,policy(),root=tmp_path,truth_root=tmp_path)
    assert len(out['methods'])==3
    assert len(out['intersections'])==3


def test_empty_input_cannot_vacuously_qualify(tmp_path):
    from hashlib import sha256
    (tmp_path/'artifact').write_bytes(b'a')
    d=dict(path='artifact',size_bytes=1,sha256=sha256(b'a').hexdigest())
    t=dict(schema='stereoforge.zed_comparison.truth.v1',independent=True,provenance=dict(kind='measured',artifact=d,generator=unavailable()),frames=[])
    out=compare([],t,policy(),root=tmp_path,truth_root=tmp_path)
    assert not out['parity_qualified']
    assert 'no_records' in out['reasons']


@pytest.mark.parametrize('case',['definition','edges','empty'])
def test_required_boundary_evidence_cannot_be_unavailable(record,tmp_path,case):
    records,t,p=qualified_fixture(record,tmp_path)
    if case=='definition':del p['freeze_parameters']['region_definitions']['boundary']['absolute_jump_m']
    elif case=='edges':t['frames'][0]['masks']['boundary_edges']=unavailable('not supplied')
    else:
        from hashlib import sha256
        data=bytes(4);(tmp_path/'empty-edges').write_bytes(data)
        desc=deepcopy(t['frames'][0]['masks']['sample']);desc.update(path='empty-edges',sha256=sha256(data).hexdigest())
        t['frames'][0]['masks']['boundary_edges']=available(desc)
    out=compare(records,t,p,root=tmp_path,truth_root=tmp_path)
    assert not out['parity_qualified']
    assert 'missing_boundary_evidence' in out['reasons']


def test_earlier_truth_mutated_during_later_frame_rejected(record,tmp_path,monkeypatch):
    import benchmarks.zed_comparison.compare_quality as module
    t=make_truth(record,tmp_path)
    other=deepcopy(record);other['ordinal']=1;other['source_id']-=1;other['input_identity']=schema.input_identity(other)
    f=deepcopy(t['frames'][0]);f['input_identity']=other['input_identity'];f['source_id']=other['source_id']
    data=(tmp_path/'truth-depth').read_bytes();(tmp_path/'later-truth').write_bytes(data)
    f['depth']['path']='later-truth';t['frames'].append(f)
    original=module.evaluate_frame;seen=[]
    def mutate(depth,validity,confidence,direction,frame,*args,**kwargs):
        out=original(depth,validity,confidence,direction,frame,*args,**kwargs)
        if seen:
            path=tmp_path/seen[0];path.write_bytes(path.read_bytes())
        seen.append(frame['metadata']['depth']['path'])
        return out
    monkeypatch.setattr(module,'evaluate_frame',mutate)
    with pytest.raises(ValueError):compare([record,other],t,policy(),root=tmp_path,truth_root=tmp_path)
    assert len(seen)==2


@pytest.mark.parametrize('candidate_on,reference_on', [(True, False), (False, True), (True, True)])
def test_temporal_on_cannot_qualify_instantaneous_comparison(record, tmp_path, candidate_on, reference_on):
    records, t, p = qualified_fixture(record, tmp_path)
    for item, enabled in zip(records, (candidate_on, reference_on)):
        item['implementation']['settings']['temporal']['stabilization'] = available(50 if enabled else 0)
    out = compare(records, t, p, root=tmp_path, truth_root=tmp_path)
    assert out['parity_qualified'] is False
    assert 'instantaneous_comparison_requires_explicit_temporal_off' in out['comparisons'][0]['reasons']


@pytest.mark.parametrize('index', [0, 1])
def test_unknown_temporal_settings_cannot_qualify_instantaneous_comparison(record, tmp_path, index):
    records, t, p = qualified_fixture(record, tmp_path)
    del records[index]['implementation']['settings']['temporal']
    out = compare(records, t, p, root=tmp_path, truth_root=tmp_path)
    assert out['parity_qualified'] is False
    assert 'instantaneous_comparison_requires_explicit_temporal_off' in out['comparisons'][0]['reasons']


def test_epipolar_residual_identity_and_vertical_offset():
    import numpy as np
    from benchmarks.zed_comparison.compare_quality import evaluate_epipolar
    from tests.test_zed_epipolar import envelope
    identity = evaluate_epipolar(np.array([[2., 3., 2., 3.]]), envelope())
    offset = evaluate_epipolar(np.array([[2., 3., 2., 1.]]), envelope())
    assert identity['vertical']['absolute_median_px'] == 0
    assert offset['vertical']['signed_median_px'] == 2
    assert offset['vertical']['absolute_p95_px'] == 2
    assert offset['metric_truth'] is False
    assert offset['quality_qualified'] is False


def test_synthetic_measured_truth_without_reference_keeps_metrics_unqualified(record, tmp_path):
    """Synthetic declarations exercise admission gates, not measured accuracy."""
    records, t, p = qualified_fixture(record, tmp_path)
    t['provenance']['kind'] = 'measured'
    out = compare(records, t, p, root=tmp_path, truth_root=tmp_path)
    assert all(frame['status'] == 'available' and frame['metrics'] for frame in out['frames'])
    assert out['parity_qualified'] is False
    assert 'missing_physical_reference_evidence' in out['comparisons'][0]['reasons']


@pytest.mark.parametrize('dense', [False, True])
def test_synthetic_physical_reference_dense_label_qualification_gate(record, tmp_path, dense):
    """Dummy artifacts validate structural gates only; no survey was performed."""
    from tests.test_zed_truth import add_physical_reference, write_physical_manifest
    records, t, p = qualified_fixture(record, tmp_path)
    metadata = add_physical_reference(t, tmp_path)
    # Keep frame bounds conservative relative to the reference and within policy.
    for frame in t['frames']:
        frame['registration']['value']['max_error_m'] = .001
        frame['timestamp_uncertainty']['value']['max_error_ns'] = 10
    if dense:
        metadata['sampling'] = 'dense'
        metadata['dense_geometry_labels'] = True
        metadata['labels'] = available(deepcopy(metadata['artifact']))
        write_physical_manifest(t, tmp_path, metadata)
    admitted = schema.validate_physical_reference(metadata, root=tmp_path)
    assert admitted['status'] == 'metadata_ready'
    assert admitted['dense_geometry_labels_allowed'] is dense
    out = compare(records, t, p, root=tmp_path, truth_root=tmp_path)
    assert all(frame['status'] == 'available' and frame['metrics'] for frame in out['frames'])
    assert out['parity_qualified'] is dense
    assert ('missing_dense_physical_geometry_labels' in out['comparisons'][0]['reasons']) is (not dense)
