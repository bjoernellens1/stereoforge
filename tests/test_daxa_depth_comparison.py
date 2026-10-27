"""Independent CPU oracle and mocked child protocol; never starts native GPU work."""
import copy
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import compare_daxa_depth as depth


def raw_eye(width=8,height=5,disparity=1.,z=12.5):
    n=width*height
    return {'disparity.f32':np.full(n,disparity,'<f4').tobytes(),'confidence.f32':np.full(n,.75,'<f4').tobytes(),
            'validity.u32':np.ones(n,'<u4').tobytes(),'depth.f32':np.full(n,z,'<f4').tobytes(),'depth_validity.u32':np.ones(n,'<u4').tobytes()}


def test_oracle_masks_actual_gpu_depth_bytes_and_retains_signed64_identity():
    raw=[raw_eye(5,1),raw_eye(5,1)];supports=[np.ones(5,'<u4').tobytes()]*2
    # A distinctive GPU result is copied unchanged; no scalar depth recomputation.
    z=np.array([1.25,7.125,9.5,123.25,19.],'<f4');raw[0]['depth.f32']=z.tobytes()
    out=depth.expected_observations(raw,supports,5,1,(1<<63)-1)
    assert np.frombuffer(out['lr_validity.u32'],'<u4').tolist()==[0,1,1,1,1,1,1,1,1,0]
    assert out['depth.f32'][4:20]==z.tobytes()[4:20]
    assert np.frombuffer(out['source_ids.i64'],'<i8').tolist()==[-1]+[(1<<63)-1]*8+[-1]
    assert set(np.frombuffer(out['age_ns.i64'],'<i8'))=={0}
    assert struct.unpack('<I',out['depth.f32'][:4])[0]==0x7fc00000


def test_strict_support_and_metric_rejection_leave_lr_planes_unchanged():
    raw=[raw_eye(5,1),raw_eye(5,1)];support=np.array([1,1,0,1,1],'<u4');supports=[support.tobytes(),np.ones(5,'<u4').tobytes()]
    mask=np.ones(5,'<u4');mask[3]=0;raw[0]['depth_validity.u32']=mask.tobytes()
    out=depth.expected_observations(raw,supports,5,1,7)
    assert np.frombuffer(out['lr_validity.u32'],'<u4').tolist()[:5]==[0,1,1,1,1]
    assert np.frombuffer(out['metric_validity.u32'],'<u4').tolist()[:5]==[0,1,0,0,1]
    # Source support x2 also rejects its right-eye correspondence x1.
    assert np.frombuffer(out['metric_validity.u32'],'<u4').tolist()[5:]==[1,0,1,1,0]
    assert np.frombuffer(out['lr_confidence.f32'],'<f4')[2]==.75
    assert np.frombuffer(out['metric_confidence.f32'],'<f4')[2]==0
    assert np.frombuffer(out['source_ids.i64'],'<i8')[2]==-1
    assert np.frombuffer(out['provenance.u32'],'<u4')[2]==0


def test_double_coordinate_half_rounding_and_tolerance_nextafter():
    # Left x3,d1.5 selects right x2, not x1 (floor(x-d+.5)).
    raw=[raw_eye(6,1,1.5),raw_eye(6,1,1.5)];d=np.full(6,1.5,'<f4');d[1]=8;raw[1]['disparity.f32']=d.tobytes()
    out=depth.expected_observations(raw,[np.ones(6,'<u4').tobytes()]*2,6,1,0)
    assert np.frombuffer(out['lr_validity.u32'],'<u4')[3]==1
    for other_d,accepted in ((np.float32(2.5),True),(np.nextafter(np.float32(2.5),np.float32(3)),False)):
        raw=[raw_eye(6,1,1.5),raw_eye(6,1,1.5)];other=np.full(6,1.5,'<f4');other[2]=other_d;raw[1]['disparity.f32']=other.tobytes()
        mask=np.frombuffer(depth.expected_observations(raw,[np.ones(6,'<u4').tobytes()]*2,6,1,7)['lr_validity.u32'],'<u4')
        assert bool(mask[3]) is accepted


@pytest.mark.parametrize('value',[np.nan,np.inf,-1.,0.])
def test_invalid_disparities_never_index_or_make_observations(value):
    raw=[raw_eye(5,1),raw_eye(5,1)];d=np.ones(5,'<f4');d[2]=value;raw[0]['disparity.f32']=d.tobytes()
    if value==0:
        valid=np.ones(5,'<u4');valid[2]=0;raw[0]['validity.u32']=valid.tobytes() # Frozen SGM rejects zero disparity.
    out=depth.expected_observations(raw,[np.ones(5,'<u4').tobytes()]*2,5,1,7)
    assert np.frombuffer(out['lr_validity.u32'],'<u4')[2]==0
    assert struct.unpack_from('<I',out['lr_disparity.f32'],8)[0]==0x7fc00000


@pytest.mark.parametrize('source_id',[-1,1<<63,True,1.5])
def test_oracle_rejects_invalid_source_ids(source_id):
    with pytest.raises(ValueError):depth.expected_observations([raw_eye(5,1)]*2,[np.ones(5,'<u4').tobytes()]*2,5,1,source_id)


def fixture(tmp_path):
    root=tmp_path/'fixture';root.mkdir();m=dict(rectified=dict(width=8,height=5),source=dict(eye_width=8,height=5,stride_bytes=32),capture=dict(record_ordinal=0,sha256='capture'),calibration=dict(original_identity='original',rectified_identity='rectified'))
    metadata=dict(frame_id=7,camera_timestamp_ns=100,host_ingress_timestamp_ns=0,camera_clock_domain='recorded',host_clock_domain='unknown',controls_json=' {"gain":2} ',calibration_identity='original')
    path=root/'manifest.json';path.write_text(json.dumps(m));metric=dict(fx=100.,baseline_m=.125,cx_offset=0.)
    bits={k+'_bits':struct.unpack('<I',struct.pack('<f',v))[0] for k,v in metric.items()};metric.update(bits)
    return path,m,metadata,metric


def identity():
    return dict(compiler_sha256='compiler',manifest_sha256='manifest',capture_sha256='capture',fixture_file_sha256={},source_sha256={},
                shader_source_sha256_pairs=dict(depth='dw:do',preprocess='pw:po',stereo='sw:so'))


def profile():return dict(width=8,height=5,disparities=16,paths=4,p1=8,p2=32,subpixel=True)


def native_report(m,metadata,metric):
    stats=dict(p50=1.,p95=2.,mean=1.2)
    return dict(profile(),schema='stereoforge.daxa_depth.v1',scope=depth.SCOPE,frames=1,warmup=0,slots=1,required_wave_size=32,lr_tolerance=1.,
        metric_support_policy=depth.SUPPORT_POLICY,layers='left_then_right',depth_units='axial_Z_metres_rectified_rays',source_metadata=metadata,source_id=metadata['frame_id'],
        camera_timestamp_ns=100,host_ingress_timestamp_ns=0,camera_clock_domain='recorded',host_clock_domain='unknown',controls_json=metadata['controls_json'],
        original_identity='original',rectified_identity='rectified',fixture_manifest_sha256='manifest',capture_sha256='capture',record_ordinal=0,
        metric_calibration=metric,shader_source_sha256_pair='dw:do',preprocessing_shader_source_sha256_pair='pw:po',stereo_shader_source_sha256_pair='sw:so',
        gpu_combined_scope=depth.GPU_SCOPE,requested_memory_scope=depth.MEMORY_SCOPE,admission_completion_policy=depth.ADMISSION_POLICY,initialization_ms=1.,diagnostic_after_timing_ms=1.,
        lr_mask_metadata_reference_checked=True,both_stereo_integer_reference_checked=True,both_stereo_packed_penalties_reference_checked=True,expected_preprocess_planes_checked=True,
        metric_compound_float_reference_checked=False,stereo_compound_float_reference_checked=False,completed_output_retirements=1,
        requested_observation_buffer_bytes=depth.requested_bytes(m,profile(),1)[0],requested_combined_buffer_bytes=depth.requested_bytes(m,profile(),1)[1],
        capture_drops='not_measured; repeated_fixture',camera_age='unknown; no_clock_mapping',validation_requested=False,
        device=dict(device='mock',slangc_sha256='compiler',validation_errors=0,validation_warnings=0),validation_errors_after_teardown=0,validation_warnings_after_teardown=0,
        **{key:dict(stats) for key in depth.TIMINGS})


@pytest.mark.parametrize('mutation',[None,'scope','support_policy','layers','units','metric_bits','metric_float','compiler','shader','integer_false','packed_false','compound_true','preprocess_false','lr_false','source_id','metadata','retirement','memory','nonfinite','diagnostic','missing_plane','changed_plane','changed_raw','missing_raw','changed_input','changed_source','changed_report','wrong_adapter','rhi_metric','rhi_profile','admission_missing','admission_changed','changed_setup_manifest'])
def test_mocked_whole_combined_protocol(tmp_path,monkeypatch,mutation):
    manifest,m,metadata,metric=fixture(tmp_path);ident=identity();luma=(np.arange(40,dtype='<u4').tobytes(),np.arange(40,80,dtype='<u4').tobytes());supports=(np.ones(40,'<u4').tobytes(),)*2
    def planes(path):
        if mutation=='changed_setup_manifest':path.write_text(json.dumps(dict(m,rebound_during_cpu_derivation=True)))
        return copy.deepcopy(m),copy.deepcopy(metadata),luma,supports,copy.deepcopy(metric),dict(binary_sha256='cv')
    monkeypatch.setattr(depth,'fixture_planes',planes)
    calls=[];probes=0
    def identities(*args):
        nonlocal probes
        probes+=1
        value=dict(copy.deepcopy(ident),manifest_sha256=depth.base.sha256(manifest))
        return dict(value,changed=True) if mutation=='changed_source' and probes>1 else value
    monkeypatch.setattr(depth,'native_identities',identities)
    raw=[raw_eye(),raw_eye()]
    def child(command,**kwargs):
        calls.append(command);out=tmp_path/'result'
        if '--manifest' in command:
            assert '--check-reference' in command
            r=native_report(m,metadata,metric);r['fixture_manifest_sha256']=depth.base.sha256(manifest)
            changes={'scope':('scope','wrong'),'support_policy':('metric_support_policy','wrong'),'layers':('layers','right_then_left'),'units':('depth_units','range'),'integer_false':('both_stereo_integer_reference_checked',False),'packed_false':('both_stereo_packed_penalties_reference_checked',False),'compound_true':('metric_compound_float_reference_checked',True),'preprocess_false':('expected_preprocess_planes_checked',False),'lr_false':('lr_mask_metadata_reference_checked',False),'source_id':('source_id',8),'retirement':('completed_output_retirements',0),'memory':('requested_observation_buffer_bytes',1)}
            if mutation in changes:r[changes[mutation][0]]=changes[mutation][1]
            if mutation=='metadata':r['source_metadata']['controls_json']='{}'
            if mutation=='admission_missing':del r['admission_completion_policy']
            if mutation=='admission_changed':r['admission_completion_policy']='one_query_only'
            if mutation=='metric_bits':r['metric_calibration']['fx_bits']+=1
            if mutation=='metric_float':r['metric_calibration']['fx']=101.
            if mutation=='compiler':r['device']['slangc_sha256']='wrong'
            if mutation=='shader':r['shader_source_sha256_pair']='wrong'
            if mutation=='nonfinite':r['host_wall_ms']['mean']=float('nan')
            (out/'native.json').write_text(json.dumps(r))
            expected=depth.expected_observations(raw,supports,8,5,7)
            for plane,data in expected.items():
                if mutation=='missing_plane' and plane=='age_ns.i64':continue
                if mutation=='changed_plane' and plane=='depth.f32':data=bytes([data[0]^1])+data[1:]
                Path(str(out/'gpu')+'.'+plane).write_bytes(data)
            for eye,name in enumerate(('left','right')):
                for plane in depth.RAW_PLANES:
                    if mutation=='missing_raw' and name=='right' and plane=='validity.u32':continue
                    data=raw[eye][plane]
                    if mutation=='changed_raw' and name=='right' and plane=='confidence.f32':data=bytes([data[0]^1])+data[1:]
                    Path(str(out/'gpu')+'.'+name+'.'+plane).write_bytes(data)
            return SimpleNamespace(returncode=0,stdout='',stderr='WARNING: invalid' if mutation=='diagnostic' else '')
        eye=0 if Path(command[command.index('--output')+1]).name=='rhi-left' else 1
        for key,field in (('--fx','fx'),('--baseline-m','baseline_m'),('--principal-offset','cx_offset')):
            assert struct.pack('<f',float(command[command.index(key)+1]))==struct.pack('<I',metric[field+'_bits'])
        for plane,data in raw[eye].items():
            Path(command[command.index('--output')+1]+'.'+plane).write_bytes(depth.mirror_rows(data,8,5) if eye else data)
        if mutation=='changed_input':(out/'left.u32').write_bytes(bytes(160))
        if mutation=='changed_report' and eye:
            (out/'native.json').write_text('{}')
        r=dict(width=8,height=5,disparities=16,paths=4,iterations=1,vulkan_validation=False,metric_depth=mutation!='rhi_metric',transfer_inclusive_total_ms=1.,adapter='other' if mutation=='wrong_adapter' else 'mock')
        if mutation=='rhi_profile':r['paths']=8
        return SimpleNamespace(returncode=0,stdout=json.dumps(r),stderr='')
    monkeypatch.setattr(depth.subprocess,'run',child)
    result=depth.run_comparison(manifest,Path('/rhi'),Path('/daxa'),Path('/compiler'),tmp_path/'result',disparities=16,frames=1,warmup=0,slots=1,validation_only=True)
    assert result['passed'] is (mutation is None)
    if mutation=='changed_setup_manifest':assert not calls and 'CPU oracle setup' in result['error']
    if mutation is None:
        assert len(calls)==3 and set(result['comparison'])==set(depth.PLANES)
        assert all(row['exact'] for row in result['comparison'].values())
        assert result['compound_cpu_float_reference_checked'] is False
        assert result['frozen_gpu_output_bytes_checked'] is True
        assert result['selected'] is None


def test_reject_bounds_and_output_collision_before_child(tmp_path,monkeypatch):
    monkeypatch.setattr(depth.subprocess,'run',lambda *args,**kwargs:pytest.fail('child forbidden'))
    for settings in ({'frames':0},{'warmup':31},{'slots':True},{'wave_size':16},{'disparities':256},{'paths':3},{'p1':40,'p2':32}):
        with pytest.raises(ValueError):depth.run_comparison(Path('/fixture'),Path('/rhi'),Path('/daxa'),Path('/compiler'),tmp_path/'out',validation_only=True,**settings)
    path,m,metadata,metric=fixture(tmp_path);monkeypatch.setattr(depth,'fixture_planes',lambda path:(m,metadata,(bytes(160),)*2,(bytes(160),)*2,metric,{}));out=tmp_path/'existing';out.mkdir()
    with pytest.raises(ValueError):depth.run_comparison(path,Path('/rhi'),Path('/daxa'),Path('/compiler'),out,disparities=16,validation_only=True)


def test_large_coordinate_nextafter_half_uses_double_coordinate_arithmetic():
    width=1005;raw=[raw_eye(width,1),raw_eye(width,1)]
    for eye in range(2):raw[eye]['validity.u32']=np.zeros(width,'<u4').tobytes()
    d=np.nextafter(np.float32(4.5),np.float32(5));a=np.ones(width,'<f4');b=a.copy();av=np.zeros(width,'<u4');bv=av.copy();a[1000]=b[995]=d;av[1000]=bv[995]=1
    raw[0]['disparity.f32']=a.tobytes();raw[1]['disparity.f32']=b.tobytes();raw[0]['validity.u32']=av.tobytes();raw[1]['validity.u32']=bv.tobytes()
    out=depth.expected_observations(raw,[np.ones(width,'<u4').tobytes()]*2,width,1,7)
    assert np.frombuffer(out['lr_validity.u32'],'<u4')[1000]==1


@pytest.mark.parametrize('z',[np.nan,np.inf,0.,-1.])
def test_nonfinite_or_nonpositive_rhi_metric_depth_cannot_be_promoted(z):
    raw=[raw_eye(5,1,z=z),raw_eye(5,1,z=z)]
    out=depth.expected_observations(raw,[np.ones(5,'<u4').tobytes()]*2,5,1,7)
    assert np.frombuffer(out['lr_validity.u32'],'<u4').sum()==8
    assert not np.frombuffer(out['metric_validity.u32'],'<u4').any()
    assert set(np.frombuffer(out['source_ids.i64'],'<i8'))=={-1}


def test_actual_compiler_dependency_headers_are_hashed_fail_closed(tmp_path,monkeypatch):
    header=tmp_path/'actual.hpp';header.write_text('actual header bytes')
    calls=[]
    def deps(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=0,stdout='target: #deps1\n    '+str(header)+'\n',stderr='')
    monkeypatch.setattr(depth.subprocess,'run',deps)
    assert depth.compiler_header_identities(tmp_path)=={str(header):depth.base.sha256(header)}
    assert len(calls)==7
    header.unlink()
    with pytest.raises(OSError):depth.compiler_header_identities(tmp_path)
