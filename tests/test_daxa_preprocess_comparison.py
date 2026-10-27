"""CPU-only preprocessing fixture and strict native child protocol."""
from pathlib import Path
import copy
import json
import struct
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import compare_daxa_preprocess as compare
from stereoforge.types import Calibration,CameraModel


def calibration(width=8,height=5):
    pose=np.eye(4);pose[0,3]=.12
    return Calibration(CameraModel(width,height,width*1.5,width*1.5,(width-1)/2,(height-1)/2),CameraModel(width,height,width*1.5,width*1.5,(width-1)/2,(height-1)/2),pose)


def record_file(tmp_path,records=2,width=8,height=5,vary_pixels=False):
    """Encode existing capture.sfp v1 for CPU protocol fixtures, not a new format."""
    c=calibration(width,height);chunks=[]
    def text(s):
        b=s.encode();return struct.pack('<I',len(b))+b
    for record_index in range(records):
        pixels=np.random.default_rng(20261027+record_index).integers(0,256,width*height*4,dtype=np.uint8) if vary_pixels else np.arange(width*height*4,dtype=np.uint8)
        payload=pixels.tobytes()
        data=struct.pack('<IIIQQQ',width*2,height,width*4,7,100,0)
        for s in ('unknown_camera','unknown_host',c.identity,'synthetic-device','{"gain": 2, "raw":true}','unverified'):data+=text(s)
        data+=struct.pack('<Q',0)+text('unknown_controls')
        for s in ('raw','unknown_imu','unknown_raw_not_pose'):data+=text(s)
        data+=struct.pack('<QI',3,1)
        data+=struct.pack('<Q7IB',89,0x7fc01234,0x80000000,0x3f800000,0,1,2,0x7f800000,1)
        data+=struct.pack('<Q',len(payload))+payload
        value=14695981039346656037
        for b in data:value=((value^b)*1099511628211)&0xffffffffffffffff
        chunks.append(data+struct.pack('<Q',value))
    path=tmp_path/'capture.sfp';path.write_bytes(b'SFYUYV1\0'+struct.pack('<IQ',1,records)+b''.join(chunks))
    return path,c


def test_existing_sfp_record_ordinal_preserves_duplicate_and_raw_metadata(tmp_path):
    path,c=record_file(tmp_path)
    one,payload=compare.read_capture_record(path,0);two,again=compare.read_capture_record(path,1)
    assert one==two and payload==again
    assert one['frame_id']==7 and one['host_ingress_timestamp_ns']==0
    assert one['camera_clock_domain']=='unknown_camera' and one['calibration_identity']==c.identity
    assert one['controls_json']=='{"gain": 2, "raw":true}'
    assert one['imu_samples'][0]['acceleration_bits']==[0x7fc01234,0x80000000,0x3f800000]
    assert one['imu_samples'][0]['temperature_bits']==0x7f800000
    with pytest.raises(ValueError):compare.read_capture_record(path,2)
    changed=bytearray(path.read_bytes());changed[-9]^=1;path.write_bytes(changed)
    with pytest.raises(ValueError):compare.read_capture_record(path,1)


def test_exported_fixture_binds_canonical_calibration_and_eight_expected_planes(tmp_path):
    path,c=record_file(tmp_path);output=tmp_path/'fixture'
    manifest=compare.export_fixture(path,1,c,output)
    assert manifest['schema']==compare.SCHEMA and manifest['profile']==compare.PROFILE
    assert manifest['calibration']['original_identity']==c.identity
    assert manifest['capture']['record_ordinal']==1 and manifest['capture']['sha256']==compare.sha256(path)
    assert manifest['calibration']['original_json']['sha256']==c.identity
    assert set(manifest['expected'])==set(compare.PLANES)
    compare.validate_fixture(output/'manifest.json')
    plane=output/manifest['maps']['left_x']['file'];plane.write_bytes(b'bad')
    with pytest.raises(ValueError):compare.validate_fixture(output/'manifest.json')


def test_support_ignores_zero_weight_taps_but_flags_partial_border():
    mx=np.array([[0.,7.,7.5,-.5,-1/64,7+1/64]],np.float32);my=np.zeros_like(mx)
    result=compare.support_mask(mx,my,8,5)
    assert result.tolist()==[[1,1,0,0,1,1]]


@pytest.mark.parametrize('mutation',['calibration','map_nan','missing_plane','unknown_profile','zero_id','image_convention'])
def test_fixture_rejects_bad_identity_layout_and_profile(tmp_path,mutation):
    path,c=record_file(tmp_path);output=tmp_path/'fixture';m=compare.export_fixture(path,0,c,output)
    if mutation=='calibration':m['calibration']['original_identity']='0'*64
    if mutation=='map_nan':
        p=output/m['maps']['left_x']['file'];p.write_bytes(struct.pack('<I',0x7fc00000)*40);m['maps']['left_x']['sha256']=compare.sha256(p)
    if mutation=='missing_plane':del m['expected']['right_support']
    if mutation=='unknown_profile':m['profile']='approximate_yuv'
    if mutation=='zero_id':m['calibration']['rectified_identity']=''
    if mutation=='image_convention':m['provenance']['image_convention']='maps_original_to_rectified'
    (output/'manifest.json').write_bytes(compare.canonical(m))
    with pytest.raises(ValueError):compare.validate_fixture(output/'manifest.json')


def test_export_refuses_missing_calibration_and_existing_output(tmp_path):
    path,c=record_file(tmp_path);output=tmp_path/'fixture';compare.export_fixture(path,0,c,output)
    before={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):compare.export_fixture(path,0,c,output)
    assert before=={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    wrong=calibration();object.__setattr__(wrong,'serial','other')
    with pytest.raises(ValueError):compare.export_fixture(path,0,wrong,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_arithmetic_conversion_profile_matches_installed_cv2():
    import cv2
    assert cv2.__version__=='4.14.0'
    rng=np.random.default_rng(20261027);packed=rng.integers(0,256,(17,34,2),dtype=np.uint8)
    pairs=packed.reshape(17,17,4).astype(np.int32);u=pairs[...,1]-128;v=pairs[...,3]-128;out=[]
    for i in (0,2):
        y=np.maximum(0,pairs[...,i]-16)*1220542
        out.append(np.clip((np.stack([y+1673527*v,y-409993*u-852492*v,y+2116026*u],axis=-1)+524288)>>20,0,255).astype(np.uint8))
    expected=np.stack(out,axis=2).reshape(17,34,3);rgb=cv2.cvtColor(packed,cv2.COLOR_YUV2RGB_YUY2)
    assert np.array_equal(expected,rgb)
    a=rgb.astype(np.uint32);gray=((9798*a[...,0]+19235*a[...,1]+3735*a[...,2]+16384)>>15).astype(np.uint8)
    assert np.array_equal(gray,cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))


def test_support_matches_independent_cv2_quantized_coordinates_and_rejects_nonfinite():
    import cv2
    x=np.array([[base+(phase+.5)/32 for base in (-2,-1,0,7,8) for phase in range(32)]],np.float32)
    y=np.zeros_like(x);xy,alpha=cv2.convertMaps(x,y,cv2.CV_16SC2)
    expected=[]
    for (ix,iy),fraction in zip(xy.reshape(-1,2),alpha.flat):
        fx=int(fraction)&31;fy=int(fraction)>>5
        weights=((32-fx)*(32-fy),fx*(32-fy),(32-fx)*fy,fx*fy)
        expected.append(all(weight==0 or 0<=int(ix)+dx<8 and 0<=int(iy)+dy<5 for (dx,dy),weight in zip(((0,0),(1,0),(0,1),(1,1)),weights)))
    assert compare.support_mask(x,y,8,5).tolist()==[list(map(int,expected))]
    for value in (np.nan,np.inf,-np.inf,32769):
        with pytest.raises(ValueError):compare.support_mask(np.array([[value]],np.float32),np.zeros((1,1),np.float32),8,5)


def report(m,metadata):
    stats=dict(p50=1.,p95=2.,mean=1.1)
    return dict(schema='stereoforge.daxa_preprocess.v1',profile=compare.PROFILE,source=m['source'],rectified=m['rectified'],scope='repeated_retained_record_staged_two_pass_no_lr',source_metadata=metadata,calibration={key:m['calibration'][key] for key in ('original_identity','rectified_identity')},
        frames=1,warmup=0,slots=1,validation_requested=False,expected_planes_checked=True,stereo=None,
        fixture_manifest_sha256='manifest',shader_source_sha256_pair='wrapper:ops',
        device=dict(slangc_sha256='compiler',validation_errors=0,validation_warnings=0),
        validation_errors_after_teardown=0,validation_warnings_after_teardown=0,completed_input_retirement_count=1,
        preprocess_gpu_ms={key:stats for key in compare.GPU_STAGES},**{key:stats for key in compare.TIMINGS})


@pytest.mark.parametrize('mutation',[None,'metadata','missing_imu','false_oracle','false_validation','shape','profile','nonfinite','retirement','stereo','compiler','gpu_missing','gpu_nonfinite','extra_retirement','typed_metadata','rectified_identity'])
def test_strict_native_protocol_gates(tmp_path,mutation):
    path,c=record_file(tmp_path);m=compare.export_fixture(path,0,c,tmp_path/'fixture');metadata,_=compare.read_capture_record(path,0)
    r=report(m,metadata)
    identity=dict(manifest_sha256='manifest',compiler_sha256='compiler',shader_source_sha256_pair='wrapper:ops')
    compare.validate_report(r,m,metadata,identity,1,0,1,False,None)
    if mutation is None:return
    if mutation=='metadata':r['source_metadata']=dict(metadata,host_ingress_timestamp_ns=1)
    if mutation=='missing_imu':r['source_metadata']=dict(metadata);del r['source_metadata']['imu_samples']
    if mutation=='false_oracle':r['expected_planes_checked']=False
    if mutation=='false_validation':r['validation_requested']=True
    if mutation=='shape':r['source']=dict(m['source'],eye_width=16)
    if mutation=='profile':r['profile']='unknown'
    if mutation=='nonfinite':r['upload_cpu_ms']=dict(p50=float('inf'),p95=1,mean=1)
    if mutation=='retirement':r['completed_input_retirement_count']=0
    if mutation=='stereo':r['stereo']={}
    if mutation=='compiler':r['device']=dict(r['device'],slangc_sha256='other')
    if mutation=='gpu_missing':r['preprocess_gpu_ms']=dict(r['preprocess_gpu_ms']);del r['preprocess_gpu_ms']['decode']
    if mutation=='gpu_nonfinite':r['preprocess_gpu_ms']=dict(r['preprocess_gpu_ms'],rectify=dict(p50=float('nan'),p95=1,mean=1))
    if mutation=='extra_retirement':r['completed_input_retirement_count']=2
    if mutation=='typed_metadata':r['source_metadata']=dict(metadata,frame_id=7.0)
    if mutation=='rectified_identity':r['calibration']=dict(r['calibration'],rectified_identity='bad')
    with pytest.raises(ValueError):compare.validate_report(r,m,metadata,identity,1,0,1,False,None)


@pytest.mark.parametrize('mode',['validation','resident','changed_plane','missing_plane','diagnostic','metadata','changed_input','report_mutation'])
def test_fake_child_orchestration_is_exact_and_fail_closed(tmp_path,monkeypatch,mode):
    from types import SimpleNamespace
    path,c=record_file(tmp_path);fixture=tmp_path/'fixture';m=compare.export_fixture(path,0,c,fixture);manifest=fixture/'manifest.json'
    metadata,_=compare.read_capture_record(path,0);identities=dict(manifest_sha256=compare.sha256(manifest),compiler_sha256='compiler',shader_source_sha256_pair='wrapper:ops')
    calls=[];probes=0
    def identity(*args):
        nonlocal probes
        probes+=1
        if probes>1 and mode=='changed_input':return dict(identities,manifest_sha256='changed')
        if probes>1 and mode=='report_mutation':(tmp_path/'result/native.json').write_text('{}')
        return copy.deepcopy(identities)
    monkeypatch.setattr(compare,'native_identities',identity)
    def child(command,**kwargs):
        calls.append(command);out=tmp_path/'result'
        r=report(m,metadata);r['fixture_manifest_sha256']=identities['manifest_sha256']
        if mode=='metadata':r['source_metadata']=dict(metadata,frame_id=8)
        (out/'native.json').write_text(json.dumps(r))
        if '--dump-prefix' in command:
            prefix=command[command.index('--dump-prefix')+1]
            for key,suffix in compare.PLANES.items():
                raw=(fixture/m['expected'][key]['file']).read_bytes()
                if mode=='missing_plane' and key=='right_luma':continue
                if mode=='changed_plane' and key=='original_left_rgb':raw=bytes([raw[0]^1])+raw[1:]
                Path(prefix+'.'+suffix).write_bytes(raw)
        return SimpleNamespace(returncode=0,stdout='',stderr='validation error' if mode=='diagnostic' else '')
    monkeypatch.setattr(compare.subprocess,'run',child)
    evidence=compare.run_comparison(manifest,Path('/native'),Path('/compiler'),tmp_path/'result',frames=1,warmup=0,validation_only=True,resident=mode=='resident')
    assert evidence['passed']==(mode in ('validation','resident'))
    assert len(calls)==1 and '--check-reference' in calls[0]
    assert ('--dump-prefix' in calls[0])==(mode!='resident')
    assert evidence['selected'] is None and not evidence['performance_qualified_counts']
    if mode=='validation':assert set(evidence['planes'])==set(compare.PLANES)
    if mode=='resident':assert 'planes' not in evidence


def test_compare_preflight_counts_and_output_collisions(tmp_path,monkeypatch):
    path,c=record_file(tmp_path);fixture=tmp_path/'fixture';compare.export_fixture(path,0,c,fixture)
    monkeypatch.setattr(compare.subprocess,'run',lambda *a,**k:pytest.fail('preflight must not run native'))
    with pytest.raises(ValueError):compare.run_comparison(fixture/'manifest.json',Path('/native'),Path('/compiler'),tmp_path/'result',frames=1,warmup=0)
    link=tmp_path/'link';link.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError):compare.run_comparison(fixture/'manifest.json',Path('/native'),Path('/compiler'),link)


def test_native_identities_require_exact_getter_spirv_and_bind_all_bytes(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import compare_daxa_view
    path,c=record_file(tmp_path);fixture=tmp_path/'fixture';compare.export_fixture(path,0,c,fixture)
    source=tmp_path/'repo';shader=tmp_path/'spirv';shader.mkdir();binary=tmp_path/'binary';compiler=tmp_path/'compiler';library=tmp_path/'library'
    for file in (binary,compiler,library):file.write_bytes(b'original')
    files=('python/stereoforge/shaders/daxa/preprocess.slang','python/stereoforge/shaders/ops/preprocess.slang',
        'src/daxa_preprocess.cpp','include/stereoforge/daxa_preprocess.hpp','apps/daxa_preprocess_benchmark.cpp','cmake/DaxaPreprocess.cmake',
        'src/capture.cpp','include/stereoforge/capture.hpp','src/daxa_stereo.cpp','include/stereoforge/daxa_stereo.hpp',
        'src/daxa_frame.cpp','include/stereoforge/daxa_frame.hpp','src/daxa_foundation.cpp','include/stereoforge/daxa_foundation.hpp',
        'include/stereoforge/contracts.hpp','CMakeLists.txt','cmake/Daxa.cmake','cmake/DaxaStereo.cmake')
    for name in files:
        file=source/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'source')
    monkeypatch.setattr(compare,'ROOT',source)
    monkeypatch.setattr(compare,'optional_dependency_identity',lambda *a:{'json.hpp':'pinned'})
    monkeypatch.setattr(compare.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=str(shader),stderr=''))
    monkeypatch.setattr(compare_daxa_view,'dependencies',lambda *a:dict(sha256={str(library):compare.sha256(library)}))
    with pytest.raises(OSError):compare.native_identities(binary,compiler,fixture/'manifest.json')
    for name in ('decode_yuyv','rectify_gray'):(shader/('daxa-preprocess-'+name+'.spv')).write_bytes(b'SPIRV')
    first=compare.native_identities(binary,compiler,fixture/'manifest.json')
    assert first==compare.native_identities(binary,compiler,fixture/'manifest.json')
    for file in (binary,compiler,library,shader/'daxa-preprocess-rectify_gray.spv',source/'src/daxa_preprocess.cpp'):
        original=file.read_bytes();file.write_bytes(b'changed')
        assert first!=compare.native_identities(binary,compiler,fixture/'manifest.json')
        file.write_bytes(original)
    monkeypatch.setattr(compare.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=str(shader),stderr='validation warning'))
    with pytest.raises(ValueError):compare.native_identities(binary,compiler,fixture/'manifest.json')


def test_bridge_cli_profile_and_resident_scope(tmp_path):
    stereo=dict(variant='packed-specialized',disparities=128,paths=4,p1=8,p2=32,required_wave_size=32,scope='single_reference_no_lr')
    command=compare.commands(Path('/native'),Path('/manifest'),tmp_path,30,3,2,True,stereo,True)
    assert '--dump-prefix' not in command and '--check-reference' in command and '--stereo' in command and '--validation' in command
    for key,value in (('disparities',128),('paths',4),('p1',8),('p2',32),('wave-size',32)):
        assert command[command.index('--'+key)+1]==str(value)


@pytest.mark.parametrize('mutation',['magic','version','nonfinalized','truncated','trailing','negative_ordinal','bool_ordinal'])
def test_sfp_rejects_corruption_before_export(tmp_path,mutation):
    path,_=record_file(tmp_path,1);data=bytearray(path.read_bytes());ordinal=0
    if mutation=='magic':data[0]^=1
    if mutation=='version':struct.pack_into('<I',data,8,2)
    if mutation=='nonfinalized':struct.pack_into('<Q',data,12,0xffffffffffffffff)
    if mutation=='truncated':data=data[:-1]
    if mutation=='trailing':data+=b'not-a-record'
    if mutation=='negative_ordinal':ordinal=-1
    if mutation=='bool_ordinal':ordinal=False
    path.write_bytes(data)
    with pytest.raises(ValueError):compare.read_capture_record(path,ordinal)


def test_fixture_paths_and_provenance_are_bound(tmp_path):
    path,c=record_file(tmp_path);fixture=tmp_path/'fixture';m=compare.export_fixture(path,0,c,fixture)
    for value in ('../capture.sfp',str(path.resolve())):
        changed=copy.deepcopy(m);changed['capture']['file']=value
        (fixture/'manifest.json').write_bytes(compare.canonical(changed))
        with pytest.raises(ValueError):compare.validate_fixture(fixture/'manifest.json')
    m['provenance']['cv2_version']='4.13.0';(fixture/'manifest.json').write_bytes(compare.canonical(m))
    with pytest.raises(ValueError):compare.validate_fixture(fixture/'manifest.json')


def test_two_different_records_with_repeated_identity_produce_distinct_luma_fixtures(tmp_path):
    path,c=record_file(tmp_path,width=64,height=32,vary_pixels=True)
    first,pixels0=compare.read_capture_record(path,0);second,pixels1=compare.read_capture_record(path,1)
    assert first==second and pixels0!=pixels1
    a=compare.export_fixture(path,0,c,tmp_path/'first');b=compare.export_fixture(path,1,c,tmp_path/'second')
    assert a['capture']['sha256']==b['capture']['sha256']
    assert a['maps']==b['maps'] and a['calibration']==b['calibration']
    for key in ('left_luma','right_luma','original_left_rgb','rect_right_rgb'):
        assert a['expected'][key]['sha256']!=b['expected'][key]['sha256']


def test_optional_dependency_headers_are_actual_recorded_compiler_inputs(tmp_path,monkeypatch):
    from types import SimpleNamespace
    header=tmp_path/'json.hpp';header.write_bytes(b'synthetic pinned header')
    crypto=tmp_path/'openssl/evp.h';crypto.parent.mkdir();crypto.write_bytes(b'EVP API')
    stdout='object: #deps 2\n    '+str(header)+'\n    '+str(crypto)+'\n'
    monkeypatch.setattr(compare.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=stdout,stderr=''))
    with pytest.raises(ValueError):compare.optional_dependency_identity(tmp_path)
    monkeypatch.setattr(compare,'JSON_HEADER_SHA256',compare.sha256(header))
    hashes=compare.optional_dependency_identity(tmp_path)
    assert hashes[str(header)]==compare.sha256(header) and hashes[str(crypto)]==compare.sha256(crypto)
    header.write_bytes(b'wrong version')
    with pytest.raises(ValueError):compare.optional_dependency_identity(tmp_path)


@pytest.mark.parametrize('resident',[False,True])
@pytest.mark.parametrize('mutation',[None,'missing_integer','missing_packed','missing_float','false_integer','false_packed','true_float','numeric_bool','missing_scope','wrong_scope'])
def test_stereo_diagnostic_reference_flags_fail_closed(tmp_path,monkeypatch,resident,mutation):
    from types import SimpleNamespace
    path,c=record_file(tmp_path);fixture=tmp_path/'fixture';m=compare.export_fixture(path,0,c,fixture);manifest=fixture/'manifest.json'
    metadata,_=compare.read_capture_record(path,0)
    identity=dict(manifest_sha256=compare.sha256(manifest),compiler_sha256='compiler',shader_source_sha256_pair='wrapper:ops')
    monkeypatch.setattr(compare,'native_identities',lambda *args:copy.deepcopy(identity))
    stereo=dict(variant='packed-specialized',disparities=16,paths=4,p1=8,p2=32,required_wave_size=32,scope='single_reference_no_lr')
    def child(command,**kwargs):
        r=report(m,metadata);r['fixture_manifest_sha256']=identity['manifest_sha256']
        flags=dict(integer_reference_checked=not resident,packed_penalties_reference_checked=not resident,compound_float_reference_checked=False)
        if mutation in ('missing_integer','missing_packed','missing_float'):del flags[{'missing_integer':'integer_reference_checked','missing_packed':'packed_penalties_reference_checked','missing_float':'compound_float_reference_checked'}[mutation]]
        if mutation in ('false_integer','false_packed'):flags['integer_reference_checked' if mutation=='false_integer' else 'packed_penalties_reference_checked']=resident
        if mutation=='true_float':flags['compound_float_reference_checked']=True
        if mutation=='numeric_bool':flags['integer_reference_checked']=int(not resident)
        r['stereo']=dict(stereo,gpu_ms=dict(p50=1.,p95=2.,mean=1.1),reference_scope='integer_and_packed_checks_require_check_reference_and_dump_prefix',**flags)
        if mutation=='missing_scope':del r['stereo']['reference_scope']
        if mutation=='wrong_scope':r['stereo']['reference_scope']='all_float_checked'
        (tmp_path/'result/native.json').write_text(json.dumps(r))
        if not resident:
            prefix=command[command.index('--dump-prefix')+1]
            for key,suffix in compare.PLANES.items():Path(prefix+'.'+suffix).write_bytes((fixture/m['expected'][key]['file']).read_bytes())
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(compare.subprocess,'run',child)
    result=compare.run_comparison(manifest,Path('/native'),Path('/compiler'),tmp_path/'result',frames=1,warmup=0,validation_only=True,stereo=stereo,resident=resident)
    assert result['passed']==(mutation is None)
