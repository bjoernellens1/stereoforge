"""CPU-only independent native temporal sequence protocol tests."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import compare_daxa_temporal as temporal


@pytest.mark.parametrize('case',['identity','translation','rotation','missing-pose','limits','provenance5','ancestry-cap','shared-id','host-rejection','boundaries','age-overflow','unsupported','ties','visibility','explicit-reset','unsupported-subnormal'])
def test_literal_sequence_expected_planes_and_roundtrip(tmp_path,case):
    path=temporal.export_sequence(tmp_path/'input',case)
    sequence=temporal.load_sequence(path);expected=temporal.expected_sequence(sequence)
    assert sequence['schema']=='stereoforge.temporal.sequence.v1'
    assert expected['native_executed'] is False and expected['general_float_qualified'] is False
    assert len(expected['frames'])==len(sequence['frames'])
    for row in expected['frames']:
        if row['accepted']:assert set(row['planes'])==set(temporal.PLANES)


@pytest.mark.parametrize('mutation',['missing_plane','shape','bad_invalid','nan','origin','path','dtype','bad_pose','config_bool'])
def test_input_corruption_fails_closed(tmp_path,mutation):
    path=temporal.export_sequence(tmp_path/'input','identity');data=json.loads(path.read_text())
    if mutation=='missing_plane':del data['frames'][0]['fresh']['confidence']
    if mutation=='shape':data['frames'][0]['fresh']['validity']=[1]
    if mutation=='bad_invalid':data['frames'][0]['fresh']['source_ids'][0][0][0]=4
    if mutation=='nan':data['frames'][0]['fresh']['confidence'][0][0][0]=float('nan')
    if mutation=='origin':data['frames'][0]['fresh']['provenance'][0][2][2]=3
    if mutation=='path':data['producer_manifest']={'file':'../outside','sha256':'0'*64}
    if mutation=='dtype':data['plane_layout']['depth']['dtype']='float64'
    if mutation=='bad_pose':data['frames'][0]['pose']['world_from_original_left'][15]=2
    if mutation=='config_bool':data['config']['max_reuse']=True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):temporal.load_sequence(path)


def mocked_native(tmp_path,monkeypatch,mutation=None,profile='exact_rational'):
    path=temporal.export_sequence(tmp_path/'input','identity',float_profile=profile)
    expected=temporal.expected_sequence(temporal.load_sequence(path));calls=[]
    identity={'source_sha256':{name:'frozen' for name in temporal.NATIVE_SOURCE_FILES},'compiler_sha256':'compiler','binary_sha256':'binary','spirv_sha256':{'temporal':'spv'}}
    reads=[]
    def identify(*args):
        reads.append(1);value=copy.deepcopy(identity)
        if mutation=='identity' and len(reads)>1:value['source_sha256']['oracle']='changed'
        return value
    monkeypatch.setattr(temporal,'collect_identities',identify)
    def child(command,**kwargs):
        calls.append(command)
        out=Path(command[command.index('--output')+1]);dump=Path(command[command.index('--dump-directory')+1]);dump.mkdir()
        rows=[]
        for e in expected['frames']:
            row=dict(index=e['index'],accepted=e['accepted'],qualification_scope=e['qualification_scope'],native_executed=True)
            if e['accepted']:
                row['planes']={};row['domain_counters']=dict(e['domain_counters']);row['work_counters']={key:e['work'][value] for key,value in temporal.WORK_KEYS.items()};row['work_counters']['history_eligible']+=e['domain_counters']['projected_domain_rejected']
                for name,raw in e['planes'].items():
                    file=f'frame-{e["index"]:04d}.{name}';payload=raw
                    if mutation=='float_ulp' and name=='depth.f32':
                        v=np.frombuffer(raw,'<u4').copy();v[12]^=1;payload=v.tobytes()
                    if mutation=='metadata' and name=='oldest_age_ns.i64':payload=b'\x01'+raw[1:]
                    p=dump/file;p.write_bytes(payload);row['planes'][name]={'file':file,'sha256':temporal.sha256(p)}
                    if mutation=='path':row['planes'][name]['file']='../'+file
                if mutation=='missing':del row['planes']['origin_mask.u32']
            rows.append(row)
        report=dict(identities=copy.deepcopy(identity),producer_binding=dict(kind='literal_controlled_planes_after_actual_Depth',geometry_models_checked=True,calibration_map_association_checked=True),schema='stereoforge.daxa_temporal.fixture.v1',sequence_sha256=temporal.sha256(path),observations_overridden_for_fixture=True,measured_stereo_output=False,performance_qualified=False,validation_requested='--validation' in command,validation_errors=0,validation_warnings=0,frames=rows)
        if mutation=='performance':report['performance_qualified']=True
        out.write_text(json.dumps(report))
        if mutation=='input':path.write_text(path.read_text()+' ')
        return SimpleNamespace(returncode=0,stdout='Validation Error: VUID-temporal-test' if mutation=='diagnostic' else '',stderr='')
    monkeypatch.setattr(temporal.subprocess,'run',child)
    return temporal.run_comparison(path,'native','compiler',tmp_path/'output',validation=True),calls


@pytest.mark.parametrize('mutation',[None,'float_ulp','metadata','missing','identity','input','path','performance','diagnostic'])
def test_native_protocol_exact_outputs_and_hashes_fail_closed(tmp_path,monkeypatch,mutation):
    result,calls=mocked_native(tmp_path,monkeypatch,mutation)
    assert result['passed'] is (mutation is None)
    assert len(calls)==1 and result['physical_qualified'] is False
    if mutation is None:assert result['native_executed'] is True and result['exact_float_profile_checked'] is True


def test_general_float_reports_ulp_difference_without_qualification(tmp_path,monkeypatch):
    result,_=mocked_native(tmp_path,monkeypatch,'float_ulp','general_float')
    assert result['passed'] is False and result['general_float_qualified'] is False
    assert any(row['float_word_mismatches'] for row in result['comparisons'])


def test_absent_native_is_not_reported_as_cpu_qualification(tmp_path):
    path=temporal.export_sequence(tmp_path/'input','identity')
    result=temporal.run_comparison(path,tmp_path/'missing-native',tmp_path/'missing-compiler',tmp_path/'out')
    assert result['passed'] is False and result['native_executed'] is False


def test_normal_domain_adapter_rejects_fresh_subnormal_without_changing_frozen_oracle(tmp_path):
    path=temporal.export_sequence(tmp_path/'input','identity');data=json.loads(path.read_text());data['frames'][0]['fresh']['depth'][0][2][2]=float(np.nextafter(np.float32(0),np.float32(1)))
    path.write_text(json.dumps(data));expected=temporal.expected_sequence(temporal.load_sequence(path))
    row=expected['frames'][0]
    assert row['domain_counters']['fresh_domain_rejected']==1
    assert np.frombuffer(row['planes']['validity.u32'],'<u4')[12]==0
    assert np.frombuffer(row['planes']['depth.f32'],'<u4')[12]==0x7fc00000
    # Frozen float64 authority keeps its broader semantics.
    frame=temporal.fresh_frame(data['frames'][0],(2,5,5))
    assert frame.valid[0,2,2]


def test_projected_subnormal_rejected_before_visibility_allows_normal_winner():
    geometry=temporal.oracle.Geometry(np.array([[1.,1.,0.,0.]]*2),np.array([np.eye(4)]*2),np.ones((2,1,3),bool))
    tiny=float(np.finfo(np.float32).tiny)
    z=np.array([[[tiny,2*tiny,np.nan]]]*2)
    history=temporal.oracle.fresh_frame(z,timestamp_ns=0,capture_id=1)
    motion=np.array([[0.,0.,-1.,tiny],[0.,1.,0.,0.],[1.,0.,0.,tiny/2],[0.,0.,0.,1.]])
    filtered,counters=temporal.normal_history(history,geometry,np.array([motion]*2),1,temporal.oracle.Config())
    assert counters['projected_domain_rejected']==2
    projected,_=temporal.oracle.warp(filtered,geometry,np.array([motion]*2),1,temporal.oracle.Config())
    assert projected.valid[:,0,0].all() and not projected.valid[:,0,1:].any()
    assert projected.depth[:,0,0].tolist()==[2.5*tiny]*2
    unfiltered,_=temporal.oracle.warp(history,geometry,np.array([motion]*2),1,temporal.oracle.Config())
    assert unfiltered.depth[:,0,0].tolist()==[tiny/2]*2



def test_rotation_fixture_contains_literal_moving_witness(tmp_path):
    path=temporal.export_sequence(tmp_path/'input','rotation');expected=temporal.expected_sequence(temporal.load_sequence(path))
    before=np.frombuffer(expected['frames'][0]['planes']['validity.u32'],'<u4').reshape(2,5,5)
    after=np.frombuffer(expected['frames'][1]['planes']['validity.u32'],'<u4').reshape(2,5,5)
    assert before[:,2,3].tolist()==[1,1]
    assert after[:,1,2].tolist()==[1,1] and after.sum()==2


def test_diagnostic_only_frames_never_claim_native_validation(tmp_path,monkeypatch):
    path=temporal.export_sequence(tmp_path/'input','identity');data=json.loads(path.read_text())
    for row in data['frames']:row['qualification_scope']='diagnostic_reference_only'
    data['geometry']['profile']='diagnostic_only';path.write_text(json.dumps(data))
    def forbidden(*args,**kwargs):raise AssertionError('diagnostic-only must not launch native')
    monkeypatch.setattr(temporal.subprocess,'run',forbidden)
    result=temporal.run_comparison(path,'missing-native','missing-compiler',tmp_path/'out')
    assert result['native_executed'] is False and result['passed'] is False


def test_boundary_and_reuse_fixture_literal_outcomes(tmp_path):
    path=temporal.export_sequence(tmp_path/'bounds','boundaries');rows=temporal.expected_sequence(temporal.load_sequence(path))['frames']
    assert np.frombuffer(rows[1]['planes']['age_ns.i64'],'<i8').max()==20
    assert not np.frombuffer(rows[2]['planes']['validity.u32'],'<u4').any()
    path=temporal.export_sequence(tmp_path/'ties','ties');rows=temporal.expected_sequence(temporal.load_sequence(path))['frames']
    assert np.frombuffer(rows[1]['planes']['confidence.f32'],'<f4').max()==.5

@pytest.mark.parametrize('malformation',['fresh','pose'])
def test_malformed_diagnostic_frame_is_skipped_without_state_access(tmp_path,malformation):
    path=temporal.export_sequence(tmp_path/'input','identity');data=json.loads(path.read_text())
    row=data['frames'][0];row['qualification_scope']='diagnostic_reference_only'
    if malformation=='fresh':del row['fresh']['confidence']
    else:row['pose']['world_from_original_left'][15]=2
    path.write_text(json.dumps(data))
    loaded=temporal.load_sequence(path)
    expected=temporal.expected_sequence(loaded)
    assert expected['frames'][0]['accepted'] is False
    assert expected['frames'][0]['qualification_scope']=='diagnostic_reference_only'
    assert expected['frames'][1]['accepted'] is True


def test_actual_identity_collector_binds_dependencies_and_reaches_launch(tmp_path,monkeypatch):
    build=tmp_path/'build';build.mkdir();native=build/'daxa_temporal_fixture';native.write_bytes(b'diagnostic-not-executed')
    compiler=tmp_path/'slangc';compiler.write_bytes(b'compiler')
    spv=build/'temporal_clear.spv';spv.write_bytes(b'spv')
    dependency=tmp_path/'libvulkan.so';dependency.write_bytes(b'dependency')
    for name in ('CMakeCache.txt','build.ninja'):(build/name).write_text(name)
    ldd_calls=[];header_calls=[];launches=[]
    def ldd(command,**kwargs):
        assert command==['ldd',str(native.resolve())];ldd_calls.append(command)
        return f'libvulkan.so => {dependency} (0x1000)\n'
    def headers(directory,targets):
        header_calls.extend(targets)
        return {str(temporal.ROOT/'include/stereoforge/daxa_temporal.hpp'):temporal.sha256(temporal.ROOT/'include/stereoforge/daxa_temporal.hpp')}
    monkeypatch.setattr(temporal.legacy.subprocess,'check_output',ldd)
    monkeypatch.setattr(temporal.depth,'compiler_header_identities',headers)
    identity=temporal.collect_identities(native,compiler)
    assert identity['dependency_sha256'][str(dependency)]==temporal.sha256(dependency)
    assert identity['build_configuration_sha256'][str(build/'build.ninja')]==temporal.sha256(build/'build.ninja')
    assert identity['source_sha256']['cmake/DaxaTemporal.cmake']==temporal.sha256(temporal.ROOT/'cmake/DaxaTemporal.cmake')
    assert ('stereoforge_daxa_temporal','src/daxa_temporal.cpp') in header_calls
    def child(command,**kwargs):
        launches.append(command);return SimpleNamespace(returncode=1,stdout='intentional CPU mock stop before native work',stderr='')
    monkeypatch.setattr(temporal.subprocess,'run',child)
    path=temporal.export_sequence(tmp_path/'input','identity')
    result=temporal.run_comparison(path,native,compiler,tmp_path/'output')
    assert result['native_executed'] is True and result['passed'] is False
    assert len(launches)==1 and launches[0][0]==str(native)
    assert len(ldd_calls)==2


def test_explicit_reset_adapter_accepts_lower_order_then_recovers_from_new_seed(tmp_path):
    path=temporal.export_sequence(tmp_path/'input','identity');data=json.loads(path.read_text())
    first,reset,recovered=data['frames']
    first['capture_id']=10;first['host_ingress_timestamp_ns']=100;first['pose']['timestamp_ns']=100
    reset['capture_id']=1;reset['host_ingress_timestamp_ns']=0;reset['pose']['timestamp_ns']=0;reset['reset']=True
    recovered['capture_id']=2;recovered['host_ingress_timestamp_ns']=10;recovered['pose']['timestamp_ns']=10
    for eye in range(2):
        recovered['fresh']['depth'][eye][2][2]=None
        for name in ('confidence','validity','provenance','age_ns'):recovered['fresh'][name][eye][2][2]=0
        recovered['fresh']['source_ids'][eye][2][2]=-1
    path.write_text(json.dumps(data));rows=temporal.expected_sequence(temporal.load_sequence(path))['frames']
    assert all(row['accepted'] for row in rows)
    assert np.frombuffer(rows[1]['planes']['contributor_count.u32'],'<u4').max()==1
    assert np.frombuffer(rows[2]['planes']['age_ns.i64'],'<i8').max()==10
    assert np.frombuffer(rows[2]['planes']['reuse.u32'],'<u4').max()==1


def test_unsupported_subnormal_is_canonicalized_before_domain_counter(tmp_path):
    path=temporal.export_sequence(tmp_path/'input','identity');data=json.loads(path.read_text())
    tiny=float(np.nextafter(np.float32(0),np.float32(1)))
    for eye in (0,1):data['frames'][0]['fresh']['depth'][eye][2][2]=tiny
    data['geometry']['support'][0][2][2]=0
    path.write_text(json.dumps(data));row=temporal.expected_sequence(temporal.load_sequence(path))['frames'][0]
    assert row['domain_counters']['fresh_domain_rejected']==1
    assert not np.frombuffer(row['planes']['validity.u32'],'<u4').any()
    assert np.all(np.frombuffer(row['planes']['source_ids.i64'],'<i8')==-1)
    assert np.all(np.frombuffer(row['planes']['depth.f32'],'<u4')==0x7fc00000)


@pytest.mark.parametrize('scope,history_survives',[('native_host_rejection',False),('diagnostic_reference_only',True)])
def test_reset_command_precedes_malformed_pose_but_diagnostic_row_is_skipped(tmp_path,scope,history_survives):
    path=temporal.export_sequence(tmp_path/'input','identity');data=json.loads(path.read_text())
    rejected=data['frames'][1];rejected['reset']=True;rejected['qualification_scope']=scope
    rejected['pose']['world_from_original_left'][0]=2.  # finite but not rigid
    recovered=data['frames'][2]
    for eye in (0,1):
        recovered['fresh']['depth'][eye][2][2]=None
        for name in ('confidence','validity','provenance','age_ns'):recovered['fresh'][name][eye][2][2]=0
        recovered['fresh']['source_ids'][eye][2][2]=-1
    path.write_text(json.dumps(data));rows=temporal.expected_sequence(temporal.load_sequence(path))['frames']
    assert rows[0]['accepted'] is True and rows[1]['accepted'] is False and rows[2]['accepted'] is True
    valid=np.frombuffer(rows[2]['planes']['validity.u32'],'<u4')
    assert bool(valid.any()) is history_survives
    assert rows[2]['work']['source_pixels_scanned']==(50 if history_survives else 0)
    if history_survives:
        assert np.frombuffer(rows[2]['planes']['age_ns.i64'],'<i8').max()==20
    else:
        assert np.all(np.frombuffer(rows[2]['planes']['source_ids.i64'],'<i8')==-1)
        assert not np.frombuffer(rows[2]['planes']['contributor_count.u32'],'<u4').any()


def test_actual_collector_ignores_ldd_aslr_but_binds_library_bytes_and_paths(tmp_path,monkeypatch):
    build=tmp_path/'build';build.mkdir();native=build/'daxa_temporal_fixture';native.write_bytes(b'not-executed')
    compiler=tmp_path/'slangc';compiler.write_bytes(b'compiler')
    (build/'temporal_clear.spv').write_bytes(b'spv')
    for name in ('CMakeCache.txt','build.ninja'):(build/name).write_text(name)
    library=tmp_path/'libvulkan.so';library.write_bytes(b'original-library')
    moved=tmp_path/'elsewhere/libvulkan.so';moved.parent.mkdir();moved.write_bytes(b'original-library')
    state={'address':'0x1000','path':library};calls=[]
    def ldd(command,**kwargs):
        assert command==['ldd',str(native.resolve())];calls.append(command)
        return f'linux-vdso.so.1 ({state["address"]})\nlibvulkan.so => {state["path"]} ({state["address"]})\n'
    monkeypatch.setattr(temporal.legacy.subprocess,'check_output',ldd)
    monkeypatch.setattr(temporal.depth,'compiler_header_identities',lambda *args:{'fixture-header':'same'})
    first=temporal.collect_identities(native,compiler)
    state['address']='0x9999'
    second=temporal.collect_identities(native,compiler)
    assert first==second
    library.write_bytes(b'changed-library')
    changed=temporal.collect_identities(native,compiler)
    assert changed!=first
    state['path']=moved
    relocated=temporal.collect_identities(native,compiler)
    assert relocated!=first
    assert len(calls)==4
    assert relocated['dependency_sha256']=={str(moved):temporal.sha256(moved)}
