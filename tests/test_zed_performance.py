from copy import deepcopy
from hashlib import sha256
import json
import pytest
from benchmarks.zed_comparison import performance as p


def available(value):
    return dict(status='available', value=value)


def unavailable():
    return dict(status='unavailable', reason='not measured')


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def record(run='r', position=0, phase='warm', name='A'):
    impl = dict(name=name, version='1', commit=unavailable(), device=unavailable(), settings={})
    desc = dict(path='report.json', sha256='0'*64, size_bytes=0)
    r = dict(schema=p.TIMING_SCHEMA, session_id='session', run_id=run, profile_id='', method_key=digest(impl),
             implementation=impl, platform=dict(host_id='host', device_uuid=available('uuid'), vendor='v', driver='d', os='o'),
             input_identities=['1'*64], calibration_identity='2'*64, phase=phase, warmup_count=3 if phase=='warm' else 0,
             lane='gpu_resident', completion_boundary='final_submit_wait_and_query', scope='compute', requested_outputs=['depth'],
             isolation=dict(file_io_in_span=False, diagnostic_retrieval_in_span=False),
             cache={k:unavailable() for k in ('temperature','allocation_activity','host_output_properties')},
             sources=dict(timing=desc, freeze=desc, quality=unavailable(), protocol=unavailable()),
             order=dict(block=0,position=position), sample_count=3,
             samples={k:unavailable() for k in p.METRICS}, stages={})
    r['samples']['gpu_total_ms']=available([1,2,3]); r['samples']['complete_wall_ms']=available([2,3,4])
    if phase=='cold': r['samples']['cold_prepare_ms']=available([5,6,7])
    r['profile_id']=p.profile_identity(r)
    return r


def test_reject_mixed_completion_boundaries():
    a=record(); b=record('b',1,name='B'); b['lane']='opaque_api'; b['completion_boundary']='opaque_api_return'
    with pytest.raises(ValueError, match='mixed'): p.summarize_timing([a,b])


def test_cold_separate_from_warm():
    a=record(); b=record('cold', phase='cold'); b['order']['block']=1
    result=p.summarize_timing([a,b])
    assert len(result['populations'])==2


def test_cross_platform_not_speedup():
    a=record(); b=record('b',1,name='B'); b['platform']['host_id']='other'
    result=p.summarize_timing([a,b]); assert result['comparisons'][0]['ratio'] is None


def test_incomplete_interleaving_unqualified():
    result=p.summarize_timing([record(),record('b',1,name='B')])
    assert not result['comparisons'][0]['qualified']
    assert 'incomplete_interleaving' in result['comparisons'][0]['reasons']


def native_report(r):
    raw=dict(enqueue_cpu_ms=[.1,.2,.3],wait_cpu_ms=[1,2,3],query_cpu_ms=[.1,.1,.1],
             host_wall_ms=r['samples']['complete_wall_ms']['value'],gpu_combined_queue_span_ms=[1,2,3], gpu_upload_ms=[.1,.1,.1])
    return dict(variant='full_sgm', slots=2, radius=2, sgm_directions=4, subpixel=True, sgm_subpixel=True,
        sgm_disparities=128, right_reference_mode='independent_sgm', lr_dependency='independent_references',
        sampling_profile='canonical_rgb8_box2x2_halfup_gray15_support_and4_halfpixel_v1',
        scope='static_repeated_canonical_record; no_live_quality_or_pipeline_FPS_claim',
        validation_errors_after_teardown=0, validation_warnings_after_teardown=0, final_estimator='independent_sgm',
        schema='stereoforge.daxa_depth_rectified_benchmark.v1', input_identity=r['input_identities'][0],
        frames=3,warmup=3,raw_samples_ms=raw,timing_metadata=dict(schema='stereoforge.native_timing_metadata.v1',
        units='ms',sample_population='measured_iterations_only',sample_order='chronological',measured_count=3,
        excluded_warmup_count=3,phase='warm',source_reuse='static_retained_record',
        completion_boundary='final_submit_wait_and_query',readback_in_span=False,file_io_in_span=False,
        diagnostic_retrieval_in_span=False,legacy_summary_policy='nearest_rank',columns={k:available('scope') for k in raw},
        cold_prepare=available(45),cold_prepare_scope='construction_only',cache_temperature=unavailable(),
        allocation_activity=unavailable(),host_output_properties=unavailable()))


def adapt(r, raw):
    return p.adapt_native_timing(raw, {k:v for k,v in r.items() if k not in ('schema','profile_id','sample_count','samples','stages')})


def write_json(root, name, value):
    data=json.dumps(value).encode(); (root/name).write_bytes(data)
    return dict(path=name,sha256=sha256(data).hexdigest(),size_bytes=len(data))


def qualified_abba(root):
    retained_artifacts=[write_json(root,'compiler.json',{'compiler':'fake'}),
        write_json(root,'shader.json',{'spirv':'fake'}),write_json(root,'input.json',{'input':'fake'})]
    records=[record(str(i),i%4,name='A' if i%4 in (0,3) else 'B') for i in range(8)]
    for i,r in enumerate(records):
        r['order']['block']=i//4
        raw=native_report(r)
        raw['admitted_files']=[{**retained_artifacts[2], 'path':'/private/original/input.json'}]
        raw['shader_compiler']={'compiler_sha256':retained_artifacts[0]['sha256']}
        for key in ('producer_spirv_sha256','stereo_spirv_sha256','depth_spirv_sha256'):
            raw[key]={'fake.spv':retained_artifacts[1]['sha256']}
        from benchmarks.zed_comparison.native_observation import _settings
        raw['sgm_subpixel']=raw['subpixel']=r['implementation']['name']=='A'
        raw['temporal']=dict(stabilization=available(0),tracking=available(False),filling=available(False))
        r['implementation']=dict(name='StereoForge',version='canonical_native_v1',
            commit=dict(status='unavailable',reason='native report did not record a StereoForge commit'),
            device=dict(status='unavailable',reason='native report did not record a device'),
            settings={**_settings(raw), 'temporal':raw['temporal']})
        r['method_key']=digest(r['implementation']); r['profile_id']=p.profile_identity(r)
        r['sources']['timing']=write_json(root,f'raw{i}.json',raw)
        records[i]=adapt(r,raw)
    a,b=records[:2]
    artifact=write_json(root,'artifact.json',dict(retained='compiler/source/input manifest'))
    freeze=dict(schema='stereoforge.zed_comparison.timing_freeze.v1',frozen=True,
        profiles=[{**{k:r[k] for k in ('profile_id','method_key','input_identities','calibration_identity','requested_outputs')},
                   'artifacts':[artifact,*retained_artifacts]} for r in (a,b)])
    frozen=write_json(root,'freeze.json',freeze)
    contention=write_json(root,'contention.json',dict(schema='stereoforge.zed_comparison.contention.v1',status='clear',platforms=[a['platform']],method='declared CPU fixture'))
    protocol=write_json(root,'protocol.json',dict(schema='stereoforge.zed_comparison.timing_protocol.v1',frozen=True,
        session_id='session',order=[a['profile_id'],b['profile_id'],b['profile_id'],a['profile_id']],min_blocks=2,
        min_samples_per_run=3,contention_before=available(contention),contention_after=available(contention)))
    quality=write_json(root,'quality.json',dict(schema='stereoforge.zed_comparison.quality.v1',status='qualified',
        parity_qualified=True,reasons=[],methods=[dict(method_key=r['method_key'],implementation=r['implementation']) for r in (a,b)],
        frames=[dict(status='available',method_key=r['method_key'],input_identity=r['input_identities'][0],partition='held_out',stratum_key=r['calibration_identity']) for r in (a,b)],
        comparisons=[dict(candidate_method=a['method_key'],reference_method=b['method_key'],parity_qualified=True,reasons=[])],
        strata=[],intersections=[],requested_range_source='supplied_policy',partition_digest='a'*64,partition_digest_algorithm='sha256'))
    for r in records:
        r['sources'].update(freeze=frozen,quality=available(quality),protocol=available(protocol))
    return records


def test_synthetic_same_device_abba_qualified_and_setup_separate(tmp_path):
    records=qualified_abba(tmp_path)
    result=p.summarize_timing(records,root=tmp_path)
    assert result['comparisons'][0]['qualified']
    assert result['comparisons'][0]['ratio']==1
    population=result['populations'][0]
    assert population['cold_prepare_setup'][0]['value']==available(45)
    assert 'cold_prepare_ms' not in population['samples']
    assert population['statistics']['samples']['complete_wall_ms']['p95_ms']==4


def test_nearest_rank_uses_raw_chronology():
    a=record(); a['samples']['complete_wall_ms']=available([9,1,5])
    result=p.summarize_timing([a])['populations'][0]
    assert result['samples']['complete_wall_ms']==[9,1,5]
    assert result['statistics']['samples']['complete_wall_ms']['p50_ms']==5
    assert result['statistics']['samples']['complete_wall_ms']['p99_ms']==9


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(extra=True),lambda r:r.update(sample_count=True),
    lambda r:r['samples'].update(gpu_total_ms=available([1,float('nan'),2])),
    lambda r:r['samples'].update(gpu_total_ms=available([1,2])),
    lambda r:r['platform'].update(device_uuid='friendly name'),
    lambda r:r.update(input_identities=['1'*64,'1'*64]),
    lambda r:r.update(warmup_count=0),lambda r:r['cache'].update(temperature=None),
    lambda r:r['implementation'].update(settings=[]),
])
def test_reject_malformed_record(mutate):
    r=record(); mutate(r)
    with pytest.raises(ValueError): p.validate_timing_lane(r)


def test_duplicate_runs_and_slots():
    with pytest.raises(ValueError,match='repeated run'): p.summarize_timing([record(),record(position=1)])
    with pytest.raises(ValueError,match='duplicate interleaving'): p.summarize_timing([record(),record('b')])


def test_adapter_rejects_missing_metadata_wrong_arrays_and_context():
    r=record(); raw=native_report(r)
    with pytest.raises(ValueError,match='missing timing_metadata'): adapt(r,{k:v for k,v in raw.items() if k!='timing_metadata'})
    raw['raw_samples_ms']['host_wall_ms']=[1]
    with pytest.raises(ValueError,match='dimensions'): adapt(r,raw)
    raw=native_report(r); r['phase']='cold'
    with pytest.raises(ValueError,match='contradicts'): adapt(r,raw)


def test_adapter_null_and_legacy_zero_unavailable_and_stage_groups():
    r=record(); raw=native_report(r)
    raw['raw_samples_ms']['gpu_inactive_ms']=[0,0,0]
    raw['timing_metadata']['columns']['gpu_inactive_ms']=unavailable()
    raw['raw_samples_ms']['gpu_null_ms']=None;raw['timing_metadata']['columns']['gpu_null_ms']=unavailable()
    raw['stereo_stage_raw_samples_ms']={'left_0':[1,2,3]}
    raw['timing_metadata']['columns']['stereo_stage_raw_samples_ms.left_0']=available('stage')
    result=adapt(r,raw)
    assert result['stages']['gpu_inactive_ms']['status']=='unavailable'
    assert result['stages']['gpu_null_ms']['status']=='unavailable'
    assert result['stages']['stereo_stage_raw_samples_ms.left_0']==available([1,2,3])


def test_wrong_raw_array_rejected(tmp_path):
    records=qualified_abba(tmp_path);records[0]['samples']['gpu_total_ms']=available([99,2,3])
    with pytest.raises(ValueError,match='raw native'):p.summarize_timing(records,root=tmp_path)


def test_unknown_raw_schema_rejected(tmp_path):
    records=qualified_abba(tmp_path)
    records[0]['sources']['timing']=write_json(tmp_path,'arbitrary.json',{'schema':'anything','samples':[1,2,3]})
    with pytest.raises(ValueError,match='unrecognized raw'):p.summarize_timing(records,root=tmp_path)


def rewrite_source(tmp_path,records,kind,mutate):
    descriptor=records[0]['sources'][kind]
    if kind in ('quality','protocol'):descriptor=descriptor['value']
    value=json.loads((tmp_path/descriptor['path']).read_text());mutate(value)
    desc=write_json(tmp_path,descriptor['path'],value)
    for r in records:r['sources'][kind]=available(desc) if kind in ('quality','protocol') else desc


@pytest.mark.parametrize('mutate',[
    lambda q:q.update(status='unqualified'),lambda q:q.update(parity_qualified=False),
    lambda q:q.update(reasons=['failed']),lambda q:q['frames'][0].update(partition='development'),
    lambda q:q['frames'][0].update(stratum_key='9'*64),lambda q:q['frames'][0].update(input_identity='9'*64),
    lambda q:q['comparisons'][0].update(reasons=['failed']),
])
def test_unqualified_quality_cannot_promote(tmp_path,mutate):
    records=qualified_abba(tmp_path);rewrite_source(tmp_path,records,'quality',mutate)
    assert not p.summarize_timing(records,root=tmp_path)['comparisons'][0]['qualified']


def test_quality_implementation_binding_rejected(tmp_path):
    records=qualified_abba(tmp_path);rewrite_source(tmp_path,records,'quality',lambda q:q['methods'][0]['implementation'].update(settings={'unknown':True}))
    with pytest.raises(ValueError,match='method binding'):p.summarize_timing(records,root=tmp_path)


def test_unfrozen_or_reordered_abba_unqualified(tmp_path):
    records=qualified_abba(tmp_path);rewrite_source(tmp_path,records,'freeze',lambda f:f.update(frozen=False))
    assert 'missing_or_unfrozen_profile' in p.summarize_timing(records,root=tmp_path)['comparisons'][0]['reasons']
    records=qualified_abba(tmp_path);records[0],records[1]=records[1],records[0]
    assert not p.summarize_timing(records,root=tmp_path)['comparisons'][0]['qualified']


def test_raw_input_binding_rejected(tmp_path):
    records=qualified_abba(tmp_path)
    descriptor=records[0]['sources']['timing'];raw=json.loads((tmp_path/descriptor['path']).read_text())
    raw['input_identity']='9'*64
    records[0]['sources']['timing']=write_json(tmp_path,descriptor['path'],raw)
    with pytest.raises(ValueError,match='input identity'):p.summarize_timing(records,root=tmp_path)


def test_shared_file_hashed_once_and_final_fingerprint(tmp_path,monkeypatch):
    records=qualified_abba(tmp_path)
    original=p.s._Files._hash;counts={}
    def tracked(files,path):
        if path not in files.hashes:counts[path]=counts.get(path,0)+1
        return original(files,path)
    monkeypatch.setattr(p.s._Files,'_hash',tracked)
    p.summarize_timing(records,root=[tmp_path]*8)
    assert counts['quality.json']==counts['freeze.json']==counts['contention.json']==counts['artifact.json']==1
    original_quality=p._quality
    def mutate(document,r):
        result=original_quality(document,r)
        (tmp_path/'artifact.json').write_text('changed after checking')
        return result
    monkeypatch.setattr(p,'_quality',mutate)
    with pytest.raises(ValueError,match='changed'):p.summarize_timing(records,root=tmp_path)


def test_raw_settings_cannot_be_supplied_from_context(tmp_path):
    records=qualified_abba(tmp_path)
    records[0]['implementation']['settings']['unknown_effective_setting']='claimed'
    records[0]['method_key']=digest(records[0]['implementation']);records[0]['profile_id']=p.profile_identity(records[0])
    with pytest.raises(ValueError,match='settings binding'):p.summarize_timing(records,root=tmp_path)


def test_absent_actual_temporal_metadata_blocks_promotion(tmp_path):
    records=qualified_abba(tmp_path)
    from benchmarks.zed_comparison.native_observation import _settings
    # Keep the context implementation exactly consistent with the actual report;
    # unknown temporal state must still block despite a claimed qualified report.
    raw=json.loads((tmp_path/'raw0.json').read_text());raw.pop('temporal')
    r=records[0];r['implementation']['settings']=_settings(raw)
    r['method_key']=digest(r['implementation']);r['profile_id']=p.profile_identity(r)
    r['sources']['timing']=write_json(tmp_path,'raw0.json',raw)
    # Exercise raw binding directly, before changed freeze/quality bindings fail.
    assert p._raw_binding(raw,r)==['raw_temporal_settings_not_bound']


def test_adapter_rejects_invented_cache_properties():
    r=record();r['cache']['temperature']=available('warm')
    with pytest.raises(ValueError,match='cache'):adapt(r,native_report(r))


def test_contended_or_missing_contention_unqualified(tmp_path):
    records=qualified_abba(tmp_path)
    rewrite_source(tmp_path,records,'protocol',lambda q:q.update(contention_after=unavailable()))
    result=p.summarize_timing(records,root=tmp_path)
    assert 'unavailable_or_contended_platform' in result['comparisons'][0]['reasons']


def test_changed_raw_file_hash_and_root_list_rejected(tmp_path):
    records=qualified_abba(tmp_path);(tmp_path/'raw0.json').write_text('{}')
    with pytest.raises(ValueError,match='hash/size'):p.summarize_timing(records,root=tmp_path)
    with pytest.raises(ValueError,match='root list'):p.summarize_timing(records,root=[tmp_path])


def test_freeze_cannot_qualify_generic_artifacts_without_native_coverage(tmp_path):
    records=qualified_abba(tmp_path)
    rewrite_source(tmp_path,records,'freeze',lambda doc:[profile.update(artifacts=profile['artifacts'][:1]) for profile in doc['profiles']])
    result=p.summarize_timing(records,root=tmp_path)
    assert 'freeze_artifact_coverage_missing' in result['comparisons'][0]['reasons']


def test_unknown_reported_input_extent_not_covered(tmp_path):
    records=qualified_abba(tmp_path)
    desc=records[0]['sources']['timing'];raw=json.loads((tmp_path/desc['path']).read_text())
    raw['admitted_files'][0]['size_bytes']+=1
    records[0]['sources']['timing']=write_json(tmp_path,desc['path'],raw)
    assert 'freeze_artifact_coverage_missing' in p.summarize_timing(records,root=tmp_path)['comparisons'][0]['reasons']


def test_sdk_opaque_return_cannot_claim_gpu_completion():
    r=record();r.update(lane='opaque_api',completion_boundary='opaque_api_return',requested_outputs=['sdk_depth_api'])
    r['samples']={key:unavailable() for key in p.METRICS}
    r['samples']['opaque_api_ms']=r['samples']['complete_wall_ms']=available([1,2,3]);r['profile_id']=p.profile_identity(r)
    raw=dict(schema='stereoforge.zed_comparison.sdk_timing.v1',operation='benchmark',lane='opaque_api',timing_lane='opaque-api',
        units='ms',sample_order='chronological',sample_population='successful_measured_grabs_only',
        file_io_in_span=False,diagnostic_retrieval_in_span=False,playback_pacing=False,
        completion_boundary='opaque_api_return',sample_count=3,phase='warm',excluded_warmup_count=3,
        requested_outputs=['sdk_depth_api'],opaque_grab_api_ms=[1,2,3],complete_wall_ms=[1,2,3])
    assert p._raw_binding(raw,r)==['raw_input_identity_not_bound']
    r['samples']['gpu_total_ms']=available([1,2,3])
    with pytest.raises(ValueError,match='internal timings'):p._raw_binding(raw,r)


def test_sdk_wrong_metadata_or_wall_array_rejected():
    r=record();r.update(lane='opaque_api',completion_boundary='opaque_api_return')
    raw=dict(schema='stereoforge.zed_comparison.sdk_timing.v1',completion_boundary='opaque_api_return')
    with pytest.raises(ValueError,match='metadata'):p._raw_binding(raw,r)


def test_large_finite_samples_have_finite_summary_mean():
    r=record();r['samples']['complete_wall_ms']=available([1e308]*3)
    value=p.summarize_timing([r])['populations'][0]['statistics']['samples']['complete_wall_ms']['mean_ms']
    assert value==1e308


@pytest.mark.parametrize('status', ['unavailable', None])
def test_quality_frames_require_explicit_available_status(tmp_path, status):
    records=qualified_abba(tmp_path)
    def mutate(document):
        for frame in document['frames']:
            if status is None:
                frame.pop('status')
            else:
                frame.update(status=status, reason='missing_observation')
    rewrite_source(tmp_path,records,'quality',mutate)
    result=p.summarize_timing(records,root=tmp_path)
    assert not result['comparisons'][0]['qualified']
    assert 'unqualified_quality' in result['comparisons'][0]['reasons']
