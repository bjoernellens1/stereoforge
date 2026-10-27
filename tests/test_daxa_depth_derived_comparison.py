"""Derived experiment CPU/protocol tests; never start GPU children."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import compare_daxa_depth_derived as derived
import compare_daxa_depth as legacy
from test_daxa_depth_comparison import fixture,identity,profile,native_report,raw_eye


def dense(width=8,D=4):
    return np.full((1,width,D),100,dtype='<u4'),np.ones((1,width),'<u4')


def values(result):
    return tuple(np.frombuffer(result[p],'<u4' if p.endswith('u32') else '<f4') for p in derived.RAW_PLANES)


def test_shift_each_candidate_and_subpixel_neighbor_independently():
    s,m=dense();s[0,0,1]=0 # Unshifted same row must never enter Right x0.
    s[0,1,1]=5;s[0,0,0]=10;s[0,2,2]=20
    d,c,v=values(derived.scalar_derived_selection(s,m,m))
    assert d[0]==pytest.approx(1-.25) and c[0]==.5 and v[0]==1


def test_subpixel_missing_neighbor_stays_integer():
    s,m=dense();s[0,2,2]=5;s[0,1,1]=10;s[0,3,3]=20
    left=m.copy();left[0,1]=0
    d,c,v=values(derived.scalar_derived_selection(s,left,m))
    assert d[0]==2 and v[0]==1


@pytest.mark.parametrize('kind',['tie','zero','threshold','single','right_invalid','off_edge'])
def test_invalid_candidates(kind):
    s,m=dense();l=m.copy();r=m.copy();s[0,1,1]=5
    if kind=='tie':s[0,2,2]=5
    if kind=='zero':s[0,0,0]=1
    if kind=='threshold':s[:]=10000
    if kind=='single':l[:]=0;l[0,1]=1
    if kind=='right_invalid':r[0,0]=0
    if kind=='off_edge':s[0,-1,1]=0;r[0,:-1]=0
    d,c,v=values(derived.scalar_derived_selection(s,l,r))
    pixel=-1 if kind=='off_edge' else 0
    assert v[pixel]==0 and c[pixel]==0 and np.isnan(d[pixel])


def test_supported_last_disparity_and_boundary_integer():
    s,m=dense(width=4);s[0,3,3]=1
    d,c,v=values(derived.scalar_derived_selection(s,m,m))
    assert v[0]==1 and d[0]==3


def test_census_mask_independent_geometry():
    actual=np.frombuffer(derived.census_validity(8,6),'<u4').reshape(6,8)
    expected=np.zeros((6,8),'<u4');expected[2:4,2:6]=1
    assert np.array_equal(actual,expected)


def test_both_metric_layers_are_explicitly_correlated_provenance():
    out=derived.expected_observations([raw_eye(5,1)]*2,[np.ones(5,'<u4').tobytes()]*2,5,1,7)
    assert np.frombuffer(out['provenance.u32'],'<u4').tolist()==[0]+[5]*8+[0]
    assert np.frombuffer(out['source_ids.i64'],'<i8').tolist()==[-1]+[7]*8+[-1]
    assert not any(np.frombuffer(out['age_ns.i64'],'<i8'))


@pytest.mark.parametrize('paths,p2,bits',[(4,32,8),(8,32,16),(4,64,16)])
def test_memory_uses_actual_packing_width(paths,p2,bits):
    m={'source':{'eye_width':1280,'height':720,'stride_bytes':5120}}
    p=dict(width=1280,height=720,disparities=128,paths=paths,p2=p2)
    obs,total=legacy.requested_bytes_for_schedule(m,p,2,'shared-census-sequential')
    assert derived.requested_bytes_for_schedule(m,p,2)==(obs,total-2*1280*720*128*bits//8)
    if paths==4 and p2==32:assert derived.requested_bytes_for_schedule(m,p,2)[1]==575079728


def bound_identity():
    return dict(identity(),stereo_derived_shader_source_sha256_triple='selector:parameters:sgm',depth_derived_shader_source_sha256_triple='derived:depth:ops')


def derived_report(m,metadata,metric,paths=4):
    r=native_report(m,metadata,metric,'shared-census-sequential',paths)
    p=dict(profile(),paths=paths)
    r.update(schema='stereoforge.daxa_depth.derived.v1',scope=derived.SCOPE,right_reference_mode=derived.RIGHT_REFERENCE_MODE,left_estimator='independent-sgm',right_estimator=derived.RIGHT_REFERENCE_MODE,lr_dependency='left-aggregated-correlated',lr_references_independent=False,metric_provenance_valid_value=5,metric_provenance_invalid_value=0,metric_provenance_scope='both_left_and_right_layers_share_correlated_LR_dependency',right_recurrence_allocated=False,stereo_phase_count=paths+5,
        stereo_dispatch_form='shared_census_once_left_aggregation_then_derived_right_selection',stereo_timing_scope='shared_census_left_aggregation_derived_right_selection_phase_intervals; no_isolated_per_reference_times',
        both_stereo_integer_reference_checked=False,both_stereo_packed_penalties_reference_checked=False,left_stereo_integer_reference_checked=True,left_stereo_packed_penalties_reference_checked=True,derived_right_mask_reference_checked=True,right_stereo_integer_winner_reference_checked=False,independent_reference_penalty_buffers=False,stereo_derived_shader_source_sha256_triple='selector:parameters:sgm',depth_derived_shader_source_sha256_triple='derived:depth:ops')
    r['gpu_paired_stereo_phases_ms']={k:v for k,v in r['gpu_paired_stereo_phases_ms'].items() if not k.startswith('right_')}
    r['gpu_paired_stereo_phases_ms']['derived_right_select']=dict(p50=1.,p95=2.,mean=1.2)
    r['requested_observation_buffer_bytes'],r['requested_combined_buffer_bytes']=derived.requested_bytes_for_schedule(m,p,1)
    r['right_penalty_buffer_savings_bytes']=legacy.requested_bytes_for_schedule(m,p,1,'shared-census-sequential')[1]-r['requested_combined_buffer_bytes']
    return r


@pytest.mark.parametrize('paths',[4,8])
def test_derived_schema_and_9_or_13_phases(tmp_path,paths):
    _,m,metadata,metric=fixture(tmp_path);r=derived_report(m,metadata,metric,paths)
    assert len(r['gpu_paired_stereo_phases_ms'])==(9 if paths==4 else 13)
    derived.validate_report(r,m,metadata,metric,bound_identity(),dict(profile(),paths=paths),1,0,1,32,False)
    with pytest.raises(ValueError):legacy.validate_report(r,m,metadata,metric,bound_identity(),dict(profile(),paths=paths),1,0,1,32,False,'shared-census-sequential')


@pytest.mark.parametrize('field,value',[
 ('schema','stereoforge.daxa_depth.v1'),('right_reference_mode','independent-sgm'),('scope','independent'),
 ('left_stereo_integer_reference_checked',False),('left_stereo_packed_penalties_reference_checked',False),('derived_right_mask_reference_checked',False),
 ('both_stereo_integer_reference_checked',True),('both_stereo_packed_penalties_reference_checked',True),('right_stereo_integer_winner_reference_checked',True),
 ('metric_compound_float_reference_checked',True),('stereo_compound_float_reference_checked',True),('gpu_left_stereo_ms',{}),('gpu_right_stereo_ms',{}),
 ('requested_combined_buffer_bytes',1),('independent_reference_penalty_buffers',True),('stereo_shared_shader_source_sha256_pair','tampered'),('stereo_layered_shader_source_sha256','tampered'),
 ('gpu_paired_stereo_phases_ms',{}),('metric_calibration',{}),('gpu_combined_queue_span_ms',{'p50':0.,'p95':0.,'mean':0.}),
])
def test_derived_contract_rejects_tampering(tmp_path,field,value):
    _,m,metadata,metric=fixture(tmp_path);r=derived_report(m,metadata,metric);r[field]=value
    with pytest.raises(ValueError):derived.validate_report(r,m,metadata,metric,bound_identity(),profile(),1,0,1,32,False)


def test_quality_difference_does_not_promote_or_invent_groundtruth():
    a=raw_eye(4,1);b=raw_eye(4,1,2.)
    result=derived.quality_difference(a,b,4,1)
    assert result['promoted'] is False and result['mean_absolute_disparity_difference']==1
    assert 'not_physical_truth' in result['scope']


def test_reference_source_is_independent_dense_per_pixel_loop():
    shader=(derived.base.ROOT/'shaders/reference/derived_right.slang').read_text()
    assert 'sums[(pixel+d)*disparity_count+d]' in shader
    assert 'for(uint d=0;d<disparity_count;d++)' in shader
    assert '#include' not in shader and 'packed' not in shader.replace('No packed production selector imports.','')


def test_derived_compiled_getter_identities_are_required(tmp_path):
    _,m,metadata,metric=fixture(tmp_path);r=derived_report(m,metadata,metric)
    ident=identity();ident.update(stereo_derived_shader_source_sha256_triple='selector:parameters:sgm',depth_derived_shader_source_sha256_triple='derived:depth:ops')
    del r['stereo_derived_shader_source_sha256_triple']
    # Missing actual compiled getters must fail even with correct old controls.
    with pytest.raises(ValueError,match='derived'):
        derived.validate_report(r,m,metadata,metric,ident,profile(),1,0,1,32,False)


@pytest.mark.parametrize('mutation',[None,'derived_byte','left_control_byte','prov1','identity','source_tamper','manifest_setup','native_schema','raw_extent','reference_profile','fixture_tamper','input_tamper','report_tamper'])
def test_mocked_independent_children_and_postbinding(tmp_path,monkeypatch,mutation):
    from types import SimpleNamespace
    path,m,metadata,metric=fixture(tmp_path);ident=bound_identity();ident['manifest_sha256']=derived.base.sha256(path);ident['fixture_file_sha256']={str(path):derived.base.sha256(path)}
    report=derived_report(m,metadata,metric);report['fixture_manifest_sha256']=ident['manifest_sha256']
    raw=[raw_eye(),raw_eye()];supports=[np.ones(40,'<u4').tobytes()]*2
    monkeypatch.setattr(derived,'fixture_planes',lambda _: (m,metadata,(b'\0'*160,b'\0'*160),supports,metric,{}))
    identities=[]
    def identities_fn(*args):
        identities.append(1);result=copy.deepcopy(ident)
        if mutation=='identity' and len(identities)>1:result['reference_executable_sha256']='changed'
        if mutation=='source_tamper' and len(identities)>1:result['source_sha256']['actual_reference']='changed'
        if mutation=='manifest_setup':result['manifest_sha256']='changed'
        return result
    monkeypatch.setattr(derived,'native_identities',identities_fn)
    calls=[];out=tmp_path/'result'
    def write_raw(prefix,layer):
        for name,data in layer.items():Path(str(prefix)+'.'+name).write_bytes(data)
    def run(cmd,**kwargs):
        calls.append(cmd);name=cmd[0]
        if name=='native':
            assert cmd[cmd.index('--right-reference-mode')+1]=='left-aggregated-selection'
            assert cmd[cmd.index('--stereo-schedule')+1]=='shared-census-sequential'
            r=copy.deepcopy(report)
            if mutation=='native_schema':r['schema']='stereoforge.daxa_depth.v1'
            (out/'native.json').write_text(__import__('json').dumps(r))
            for eye,layer in zip(('left','right'),raw):write_raw(out/('gpu.'+eye),{k:v for k,v in layer.items() if k in derived.RAW_PLANES})
            expected=derived.expected_observations(raw,supports,8,5,7)
            if mutation=='prov1':expected['provenance.u32']=legacy.expected_observations(raw,supports,8,5,7)['provenance.u32']
            for plane,data in expected.items():(out/('gpu.'+plane)).write_bytes(data)
            if mutation=='fixture_tamper':path.write_text('changed')
            if mutation=='input_tamper':(out/'left.u32').chmod(0o644);(out/'left.u32').write_bytes(b'changed')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        if name=='reference':
            assert Path(cmd[cmd.index('--left-valid')+1]).read_bytes()==derived.census_validity(8,5)
            for eye,layer in zip(('left','right'),raw):write_raw(out/('reference.'+eye),layer)
            if mutation=='derived_byte':(out/'reference.right.disparity.f32').write_bytes(np.full(40,2.,'<f4').tobytes())
            if mutation=='raw_extent':(out/'reference.right.disparity.f32').write_bytes(b'bad')
            result=dict(schema='stereoforge.rhi_derived_right.v1',adapter='mock',vulkan_validation=False,width=8,height=5,disparities=16,paths=4,metric_depth=True)
            if mutation=='reference_profile':result['paths']=8
        else:
            prefix=Path(cmd[cmd.index('--output')+1]);eye=prefix.name.split('-')[-1]
            row=copy.deepcopy(raw[0 if eye=='left' else 1])
            if eye=='right':row={k:derived.mirror_rows(v,8,5) for k,v in row.items()}
            if mutation=='left_control_byte' and eye=='left':row['disparity.f32']=np.zeros(40,'<f4').tobytes()
            write_raw(prefix,row)
            if mutation=='report_tamper' and eye=='right':(out/'native.json').write_text('{}')
            result=dict(width=8,height=5,disparities=16,paths=4,iterations=1,adapter='mock',metric_depth=True,vulkan_validation=False)
        return SimpleNamespace(returncode=0,stdout=__import__('json').dumps(result),stderr='')
    monkeypatch.setattr(derived.subprocess,'run',run)
    result=derived.run_comparison(path,'rhi','native','compiler',out,'reference',disparities=16,frames=1,warmup=0,slots=1,validation_only=True)
    assert result['passed'] is (mutation is None)
    assert result['promoted'] is False and result['compound_cpu_float_reference_checked'] is False
    if mutation is None:
        assert len(calls)==4 and result['reference_left_control_byteexact'] is True
        assert len(result['comparison'])==9 and len(result['raw_stereo_comparison'])==6
    if mutation=='manifest_setup':assert not calls




@pytest.mark.parametrize('tamper',[False,True])
def test_identity_preserves_all_33_old_spvs_and_binds_three_new_and_getters(tmp_path,monkeypatch,tamper):
    from types import SimpleNamespace
    import compare_daxa_view as view
    native=tmp_path/'reference';native.write_text('binary')
    old={f'old-control-{i}.spv':str(i) for i in range(33)}
    ident=dict(offline_shader_directory=str(tmp_path),offline_spirv_sha256=old,source_sha256={},compiler_input_sha256={})
    monkeypatch.setattr(derived.legacy,'native_identities',lambda *a:copy.deepcopy(ident))
    monkeypatch.setattr(derived.base,'sha256',lambda p:'sha:'+str(p))
    monkeypatch.setattr(view,'dependencies',lambda *a:dict(sha256='dependencies'))
    monkeypatch.setattr(derived,'compiler_header_identities',lambda *a:{'header':'sha'})
    expected={key:':'.join('sha:'+str(derived.base.ROOT/name) for name in names) for key,names in derived.DERIVED_SHADER_GROUPS.items()}
    if tamper:expected['depth_derived_shader_source_sha256_triple']='tampered'
    monkeypatch.setattr(derived.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=__import__('json').dumps(expected),stderr=''))
    if tamper:
        with pytest.raises(ValueError,match='compiled derived'):derived.native_identities('rhi','native','compiler','manifest',native)
    else:
        actual=derived.native_identities('rhi','native','compiler','manifest',native)
        assert len(actual['offline_spirv_sha256'])==36
        assert all(actual['offline_spirv_sha256'][key]==value for key,value in old.items())
        assert actual['derived_compiled_shader_getters']==expected
        assert 'derived_reference' in actual['compiler_input_sha256']
