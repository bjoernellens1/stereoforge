"""CPU-only independent known-point and retained-view protocol contracts."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import compare_daxa_view_depth as view


def test_nonidentity_rectification_known_point_rotation_once_and_right_translation():
    fixture=view.geometry_fixture();witness=fixture['witnesses'][0]
    assert fixture['raw_size']!=fixture['rect_size']
    assert fixture['rectified'][0]['cx']!=fixture['rectified'][1]['cx']
    for eye in (0,1):
        pixel,z=view.rectified_observation(fixture,witness['original_left_point'],eye)
        rebuilt=view.reconstruct_world(fixture,pixel,z,eye)
        assert np.allclose(rebuilt,witness['original_left_point'],atol=1e-14,rtol=0)
    assert view.project_known_point(witness['original_left_point'],fixture['view'])[:2]==(6,7)


def test_geometry_association_requires_actual_quantized_map_model_relationship():
    fixture=view.geometry_fixture();maps=view.geometry_maps(fixture)
    evidence=view.validate_map_association(fixture,maps)
    assert evidence['producer']=='independent_calibrated_pinhole_distortion_geometry'
    assert evidence['quantized_sampling']=='OpenCV_INTER_LINEAR_5bit_nearest_even'
    wrong={key:value.copy() for key,value in maps.items()};wrong['left_x'][2,3]+=1
    with pytest.raises(ValueError,match='association'):view.validate_map_association(fixture,wrong)


def test_perspective_axial_and_cylindrical_range_are_distinct():
    fixture=view.geometry_fixture();point=[.75,0.,1.]
    p=view.project_known_point(point,dict(fixture['view'],projection='perspective'))
    c=view.project_known_point(point,dict(fixture['view'],projection='cylindrical'))
    assert p[2]==1. and c[2]==1.25


def test_missing_metric_defaults_and_preview_namespace_never_certify_geometry():
    output=view.empty_output(3)
    view.assign_preview(output,1,0x030201,eye=1)
    assert output['rgb.u32'][1]==0x030201 and output['rgb_validity.u32'][1]==1
    assert output['validity.u32'][1]==0 and np.isnan(output['depth.f32'][1])
    assert output['owner.u32'][1]==0xffffffff and output['source_ids.i64'][1]==-1
    assert output['age_ns.i64'][1]==0 and output['provenance.u32'][1]==0 and output['approximate.u32'][1]==1


@pytest.mark.parametrize('case',['tie','conflict','missing','unsupported','preview'])
def test_generated_fixture_known_geometry_and_synthetic_domains(tmp_path,case):
    geometry=view.export_fixture(tmp_path/'fixture',case)
    manifest,metadata,actual,planes,images=view.load_fixture(tmp_path/'fixture')
    assert actual==geometry and metadata['device_identity']=='synthetic-known-geometry'
    assert manifest['source']['eye_width']!=manifest['rectified']['width']
    gain=view.forced_gain(actual,planes,images)
    out=view.expected_known_output(actual['geometry'],actual['geometry']['view'],planes,actual['witnesses'],[images['rect_left'],images['rect_right']],[images['original_left'],images['original_right']],exposure_gain=gain)
    if case in ('tie','conflict'):
        t=7*13+6
        assert out['validity.u32'].sum()==1 and out['depth.f32'][t]==1
        assert out['owner.u32'][t]==17 and out['source_ids.i64'][t]==7
        if case=='tie':assert out['source.u32'][t]==3 and out['rgb.u32'][t]==0xb5b5b5
    else:assert not out['validity.u32'].any()


def test_packed_frozen_control_preserves_unique_metric_owners_and_preview_bounds(tmp_path):
    view.export_fixture(tmp_path/'fixture');_,_,g,p,images=view.load_fixture(tmp_path/'fixture')
    m=view.export_rhi_control(tmp_path/'control',g,p,images)
    assert m['source_width']==35+16*12 and m['source_height']==1
    rays=np.fromfile(tmp_path/'control/rays.f32','<f4').reshape(-1,3);depth=np.fromfile(tmp_path/'control/source_depth.f32','<f4')
    assert (rays[17]*depth[17]).tolist()==[0.,.75,1.]
    assert (rays[35+192+16]*depth[35+192+16]).tolist()==[0.,.75,1.]
    previews=np.fromfile(tmp_path/'control/preview_indices.u32','<u4').reshape(2,-1)
    for eye in (0,1):assert np.all((previews[eye]==view.UINT_MAX)|((previews[eye]>=eye*227+35)&(previews[eye]<(eye+1)*227)))


def test_payload_tampering_and_unsupported_metric_rejected_before_children(tmp_path):
    import json
    directory=tmp_path/'fixture';view.export_fixture(directory)
    data=directory/'observations.depth.f32';data.write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash/size'):view.load_fixture(directory)


def test_map_association_does_not_accept_self_asserted_hash(tmp_path):
    import json
    directory=tmp_path/'fixture';view.export_fixture(directory);manifest=json.loads((directory/'manifest.json').read_text());geometry=json.loads((directory/'geometry.json').read_text());entry=manifest['maps']['left_x'];path=directory/entry['file'];values=np.frombuffer(path.read_bytes(),'<f4').copy();values[17]+=1;path.write_bytes(values.tobytes());entry['sha256']=view.base.sha256(path);(directory/'manifest.json').write_bytes(view.preprocess.canonical(manifest));geometry['manifest_sha256']=view.base.sha256(directory/'manifest.json');(directory/'geometry.json').write_bytes(view.preprocess.canonical(geometry))
    with pytest.raises(ValueError,match='association'):view.load_fixture(directory)


def report_fixture(manifest,metadata,g,identity,mode='strict'):
    f=g['geometry'];v=f['view'];w,h=f['rect_size'];rw,rh=f['raw_size'];poses=[]
    for eye in (0,1):
        R=np.eye(4);R[:3,:3]=np.asarray(f['R2' if eye else 'R1']).T;poses.append((np.asarray(v['virtual_from_original_left'])@(np.asarray(f['right_to_left']) if eye else np.eye(4))@R).astype('<f4').ravel().astype(float).tolist())
    result=dict(schema='stereoforge.daxa_view_depth.v1',scope='synthetic_known_observation_typed_compositor_fixture_no_stereo_accuracy_or_performance',observations_overridden_for_fixture=True,measured_stereo_output=False,metric_source='synthetic_known_geometry',performance_qualified=False,rectified_width=w,rectified_height=h,original_width=rw,original_height=rh,target_width=v['width'],target_height=v['height'],projection=v['projection'],depth_convention='virtual_axial_Z' if v['projection']=='perspective' else 'virtual_ray_range',mode=mode,map_geometry_association='opencv4.14-effective-quantized-map-model-validation',map_geometry_truth_scope='effective_OpenCV4.14_sample_association_of_supplied_models_not_physical_calibration_truth',preview_valid_count=int((view.original_preview_indices(f,v)!=view.UINT_MAX).sum()),metric_owner_count=2*w*h,preview_namespace_count=2*rw*rh,preview_only_owner=view.UINT_MAX,fixture_manifest_sha256=identity['manifest_sha256'],capture_sha256=manifest['capture']['sha256'],original_identity=manifest['calibration']['original_identity'],rectified_identity=manifest['calibration']['rectified_identity'],source_id=metadata['frame_id'],shader_source_identities=identity['view_shader_source_identities'],compound_cpu_float_reference_checked=False,physical_geometry_qualified=False,live_60hz_qualified=False,synthetic_capture=True,replacement_transfer_completion_waited=True,validation_requested=False,diagnostic_observation_sha256=[g['observations'][n]['sha256'] for n in view.depth.PLANES],source_metadata={k:metadata[k] for k in ('frame_id','camera_timestamp_ns','host_ingress_timestamp_ns','camera_clock_domain','host_clock_domain','calibration_identity','device_identity')},effective_source_to_virtual_left=poses[0],effective_source_to_virtual_right=poses[1],device=dict(device='mock',slangc_sha256='compiler',validation_errors=0,validation_warnings=0),validation_errors_after_teardown=0,validation_warnings_after_teardown=0)
    import math
    result.update(virtual_camera_pose_policy='explicit_virtual_from_original_left',virtual_from_original_left=np.asarray(v['virtual_from_original_left']).astype('<f4').ravel().astype(float).tolist(),virtual_intrinsics=[float(np.float32(v[k])) for k in ('fx','fy','cx','cy')],hfov_radians=float(np.float32(v['hfov_radians'])),requested_horizontal_fov_radians=float(np.float32(v['hfov_radians'])) if v['projection']=='cylindrical' else math.atan((v['width']-.5-v['cx'])/v['fx'])-math.atan((-.5-v['cx'])/v['fx']),virtual_camera_canvas_policy='calibration_left_canvas_unless_explicit_intrinsics_and_dimensions',coverage_scope='explicit_posttiming_export_target_canvas_pixel_fractions_not_observed_FOV_or_physical_truth',rgb_valid_coverage=0.,metric_valid_coverage=0.)
    return result


@pytest.mark.parametrize('field,value',[
 ('performance_qualified',True),('physical_geometry_qualified',True),('live_60hz_qualified',True),('compound_cpu_float_reference_checked',True),
 ('observations_overridden_for_fixture',False),('measured_stereo_output',True),('metric_source','actual_stereo'),('replacement_transfer_completion_waited',False),
 ('metric_owner_count',384),('preview_namespace_count',70),('preview_only_owner',0),('map_geometry_association','hash_only'),('source_id',8),
 ('depth_convention','virtual_ray_range'),('diagnostic_observation_sha256',[]),('shader_source_identities',{}),('effective_source_to_virtual_left',list(np.eye(4).ravel())),
 ('validation_errors_after_teardown',1),('source_metadata',{}),('preview_valid_count',0),
])
def test_report_rejects_diagnostic_promotion_namespace_or_association_tampering(tmp_path,field,value):
    view.export_fixture(tmp_path/'fixture');m,metadata,g,_,_=view.load_fixture(tmp_path/'fixture');identity=dict(manifest_sha256=view.base.sha256(tmp_path/'fixture/manifest.json'),compiler_sha256='compiler',view_shader_source_identities={'view':'sha'})
    report=report_fixture(m,metadata,g,identity);report[field]=value
    with pytest.raises(ValueError):view.validate_report(report,m,metadata,g,identity,'strict',False)


@pytest.mark.parametrize('case,projection,rgb_pattern', [('tie','perspective','constant'),('conflict','perspective','constant'),('preview','perspective','periphery'),('tie','cylindrical','constant'),('missing','cylindrical','constant')])
@pytest.mark.parametrize('mutation',[None,'float_ulp','owner','metadata','identity','input','report'])
def test_mocked_full_eleven_plane_chain_preserves_exact_controls(tmp_path,monkeypatch,case,projection,rgb_pattern,mutation,validation_mode="off"):
    import copy,json
    from types import SimpleNamespace
    directory=tmp_path/'fixture';view.export_fixture(directory,case,projection,rgb_pattern);m,metadata,g,p,images=view.load_fixture(directory);identity=dict(manifest_sha256=view.base.sha256(directory/'manifest.json'),compiler_sha256='compiler',view_shader_source_identities={'view':'sha'});calls=[];identities=[];output=tmp_path/'out';f=g['geometry'];camera=f['view'];n=np.prod(f['rect_size']);M=n+np.prod(f['raw_size'])
    def identify(*args):
        identities.append(1);actual=copy.deepcopy(identity)
        if mutation=='identity' and len(identities)>1:actual['source']='changed'
        return actual
    monkeypatch.setattr(view,'native_identities',identify)
    expected={mode:view.expected_known_output(f,camera,p,g['witnesses'],[images['rect_left'],images['rect_right']],[images['original_left'],images['original_right']],mode,view.forced_gain(g,p,images)) for mode in ('strict','preview')}
    def child(cmd,**kwargs):
        calls.append(cmd);mode=cmd[cmd.index('--mode')+1];native=cmd[0]=='native';prefix=Path(cmd[cmd.index('--dump-prefix')+1]);data={k:v.copy() for k,v in expected[mode].items()}
        if native:
            assert cmd[cmd.index('--observations-prefix')+1]==str(directory/'observations')
            report=report_fixture(m,metadata,g,identity,mode)
            report['validation_requested']=validation_mode!='off'
            assert ('--validation' in cmd)==(validation_mode!='off')
            assert view.legacy.validation_requested(False,kwargs['env'])==(validation_mode!='off')
            report['rgb_valid_coverage']=float(data['rgb_validity.u32'].sum())/len(data['rgb_validity.u32']);report['metric_valid_coverage']=float(data['validity.u32'].sum())/len(data['validity.u32'])
            if mutation=='report':report['performance_qualified']=True
            if mutation=='float_ulp':data['depth.f32'].view('<u4')[0]^=1
            if mutation=='owner':data['owner.u32'][0]=999
            if mutation=='metadata':data['source_ids.i64'][0]=99
        else:
            assert view.legacy.validation_requested(False,kwargs['env'])==(validation_mode=='both')
            valid=data['validity.u32']!=0;owners=data['owner.u32'];owners[valid]=(owners[valid]//n)*M+owners[valid]%n
            preview=data['approximate.u32']!=0;indices=view.original_preview_indices(f,camera);chosen=np.where(indices[0]!=view.UINT_MAX,indices[0],indices[1]);raw_n=np.prod(f['raw_size']);owners[preview]=(chosen[preview]//raw_n)*M+n+chosen[preview]%raw_n
            report=dict(mode=mode,input='resident',readback='gpu',frames=1,warmup=0,source_width=int(M),source_height=1,target_width=camera['width'],target_height=camera['height'],python_interpreter_used=False,timestamp_ns=100,calibration_id='synthetic-independent-known3D-control',device='mock')
        if mutation=='control_validation_claim' and not native:report['validation_requested']=True
        if mutation=='native_validation_claim' and native:report['validation_requested']=False
        if mutation=='control_validation_qualified' and not native:report['validation_qualified']=True
        Path(cmd[cmd.index('--output')+1]).write_text(json.dumps(report))
        for name,array in data.items():Path(str(prefix)+'.'+name).write_bytes(array.tobytes())
        if mutation=='input':(directory/'observations.depth.f32').write_bytes(b'changed')
        stream=mutation.split('_')[-1] if mutation and mutation.startswith(('native_diag_','control_diag_')) else None
        diagnostic=bool(stream and native==mutation.startswith('native_'))
        return SimpleNamespace(returncode=0,stdout='Validation Error: VUID-mock-failure' if diagnostic and stream=='stdout' else '',stderr='Warning: mock' if diagnostic and stream=='stderr' else '')
    monkeypatch.setattr(view.subprocess,'run',child)
    result=view.run_comparison(directory,'native','rhi','compiler',output,validation_mode!='off',validation_mode=='native-only')
    assert result['passed'] is (mutation is None)
    assert result['performance_qualified'] is False and result['compound_cpu_float_reference_checked'] is False
    if mutation is None:
        assert len(calls)==(4 if projection=='perspective' else 2)
        assert all(len(rows)==11 for rows in result['comparisons'].values())
        assert result['frozen_compositor_control_checked'] is (projection=='perspective')
        assert result['frozen_compositor_control_available'] is (projection=='perspective')
        assert result['validation']['native_required_layer_report_checked'] is (validation_mode!='off')
        assert result['validation']['control_required_layer_checked'] is False
        assert result['validation']['overall_validation_qualified'] is False
        assert result['validation']['control_activation_requested'] is (validation_mode=='both')
    if mutation and '_diag_' in mutation:
        name=('strict-daxa' if mutation.startswith('native_') else 'strict-rhi')
        assert result['children'][name]['diagnostics'][mutation.split('_')[-1]]['line_count']==1
        assert result['report_sha256'][name+'.'+mutation.split('_')[-1]]==view.base.sha256(output/(name+'.'+mutation.split('_')[-1]))
        assert len(calls)==(1 if mutation.startswith('native_') else 2)


def test_shared_raw_cell_observations_keep_distinct_known_virtual_points(tmp_path):
    directory=tmp_path/'fixture';view.export_fixture(directory,'shared-raw-cell');_,_,g,p,images=view.load_fixture(directory)
    witnesses=[w for w in g['witnesses'] if w['eye']==0];maps=view.geometry_maps(g['geometry'])
    raw_cells=[tuple(np.floor([maps['left_x'][w['pixel'][1],w['pixel'][0]]+.5,maps['left_y'][w['pixel'][1],w['pixel'][0]]+.5]).astype(int)) for w in witnesses]
    assert len(witnesses)==2 and raw_cells[0]==raw_cells[1]
    targets=[view.project_known_point(w['original_left_point'],g['geometry']['view'])[:2] for w in witnesses]
    assert targets==[(6,7),(7,7)]


def test_known_virtual_pose_is_applied_once(tmp_path):
    directory=tmp_path/'fixture';view.export_fixture(directory,virtual_pose='rotate-z90');_,_,g,p,images=view.load_fixture(directory)
    out=view.expected_known_output(g['geometry'],g['geometry']['view'],p,g['witnesses'],[images['rect_left'],images['rect_right']],[images['original_left'],images['original_right']],exposure_gain=view.forced_gain(g,p,images))
    assert out['validity.u32'][4*13+3]==1 and out['depth.f32'][4*13+3]==1


def test_unsupported_fixture_really_uses_unsupported_remap_sample(tmp_path):
    directory=tmp_path/'fixture';view.export_fixture(directory,'unsupported');m,_,g,p,_=view.load_fixture(directory)
    active=np.flatnonzero(p['lr_validity.u32']);assert len(active)==1
    index=int(active[0]);n=np.prod(g['geometry']['rect_size']);name='right' if index>=n else 'left'
    support=np.fromfile(directory/m['expected'][name+'_support']['file'],'<u4')
    assert support[index%n]==0 and p['metric_validity.u32'][index]==0


def test_unsupported_metric_attack_rejected_even_if_all_hashes_updated(tmp_path):
    import json
    directory=tmp_path/'fixture';view.export_fixture(directory,'unsupported');m,_,g,p,_=view.load_fixture(directory);i=int(np.flatnonzero(p['lr_validity.u32'])[0])
    for name,value in [('depth.f32',1.25),('metric_confidence.f32',.5),('metric_validity.u32',1),('provenance.u32',1),('source_ids.i64',7)]:
        data=p[name];data[i]=value;entry=g['observations'][name];path=directory/entry['file'];path.write_bytes(data.tobytes());entry['sha256']=view.base.sha256(path)
    (directory/'geometry.json').write_bytes(view.preprocess.canonical(g))
    with pytest.raises(ValueError,match='unsupported geometry'):view.load_fixture(directory)


@pytest.mark.parametrize('tamper',[False,True])
def test_identity_preserves_old_spvs_and_binds_all_eighteen_view_shaders(tmp_path,monkeypatch,tamper):
    from types import SimpleNamespace
    old={f'old-{i}':'sha'+str(i) for i in range(33)}
    monkeypatch.setattr(view.depth,'native_identities',lambda *args:dict(offline_shader_directory=str(tmp_path),source_sha256={},offline_spirv_sha256=dict(old),compiler_input_sha256={}))
    monkeypatch.setattr(view.legacy,'dependencies',lambda *args:dict(sha256='deps'))
    monkeypatch.setattr(view.base,'sha256',lambda path:'sha:'+str(path))
    monkeypatch.setattr(view.depth,'compiler_header_identities',lambda *args:{'header':'sha'})
    pairs=dict(view_depth_shader_source_sha256_pair=('python/stereoforge/shaders/daxa/view_depth.slang','python/stereoforge/shaders/ops/projection.slang'),depth_shader_source_sha256_pair=('python/stereoforge/shaders/daxa/depth.slang','python/stereoforge/shaders/ops/stereo_observation.slang'),preprocess_shader_source_sha256_pair=('python/stereoforge/shaders/daxa/preprocess.slang','python/stereoforge/shaders/ops/preprocess.slang'))
    expected={key:':'.join('sha:'+str(view.ROOT/path) for path in paths) for key,paths in pairs.items()}
    if tamper:expected['view_depth_shader_source_sha256_pair']='tampered'
    monkeypatch.setattr(view.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=__import__('json').dumps(expected),stderr=''))
    if tamper:
        with pytest.raises(ValueError,match='compiled view'):view.native_identities(tmp_path/'rhi',tmp_path/'native','compiler','manifest')
    else:
        identity=view.native_identities(tmp_path/'rhi',tmp_path/'native','compiler','manifest')
        assert len(identity['offline_spirv_sha256'])==51 and all(identity['offline_spirv_sha256'][k]==v for k,v in old.items())
        assert identity['view_shader_source_identities']==expected
        assert set(identity['compiler_input_sha256'])=={'view_daxa','view_rhi'}


def test_owner_normalization_cannot_hide_control_strict_hole_owner_corruption():
    g=dict(geometry=view.geometry_fixture());raw={k:v.tobytes() for k,v in view.empty_output(1).items()};raw['owner.u32']=np.array([99],'<u4').tobytes()
    with pytest.raises(ValueError,match='strict hole'):view.normalize_control_owners(raw,g)


def test_owner_normalization_requires_real_preview_namespace_and_preserves_other_planes():
    g=dict(geometry=view.geometry_fixture());raw=view.empty_output(1);view.assign_preview(raw,0,0x123456,0);raw['owner.u32'][0]=35+9
    packed={k:v.tobytes() for k,v in raw.items()};actual=view.normalize_control_owners(packed,g)
    assert np.frombuffer(actual['owner.u32'],'<u4')[0]==view.UINT_MAX
    assert all(actual[name]==packed[name] for name in packed if name!='owner.u32')
    packed['owner.u32']=np.array([9],'<u4').tobytes()
    with pytest.raises(ValueError,match='preview owner'):view.normalize_control_owners(packed,g)


def test_cylindrical_declared_hfov_limit_behind_z_and_vertical_horizontal_radius():
    import math
    camera=view.geometry_fixture()['view'];wide=dict(camera,projection='cylindrical',hfov_radians=float(np.nextafter(np.float32(2*math.pi),np.float32(0))))
    assert view.project_known_point([.25,0,-1],wide) is not None
    assert view.project_known_point([0,0,-1],wide) is None # excluded atan2 seam at finite HFOV<2pi
    assert view.project_known_point([.25,0,-1],dict(camera,projection='perspective')) is None
    assert view.project_known_point([.75,1.25,1],dict(camera,projection='cylindrical'))[1]==8
    with pytest.raises(ValueError,match='HFOV'):view.project_known_point([0,0,1],dict(wide,hfov_radians=2*math.pi))


def test_self_rehashed_expected_rgb_cannot_replace_actual_capture_preprocessing(tmp_path):
    import json
    directory=tmp_path/'fixture';view.export_fixture(directory);m=json.loads((directory/'manifest.json').read_text());g=json.loads((directory/'geometry.json').read_text());entry=m['expected']['original_left_rgb'];path=directory/entry['file'];data=bytearray(path.read_bytes());data[0]^=1;path.write_bytes(data);entry['sha256']=view.base.sha256(path);(directory/'manifest.json').write_bytes(view.preprocess.canonical(m));g['manifest_sha256']=view.base.sha256(directory/'manifest.json');(directory/'geometry.json').write_bytes(view.preprocess.canonical(g))
    with pytest.raises(ValueError,match='actual CPU OpenCV preprocessing'):view.load_fixture(directory)


def test_metric_correspondence_support_rejected_after_self_rehash(tmp_path):
    import json
    directory=tmp_path/'fixture';view.export_fixture(directory);_,_,g,p,_=view.load_fixture(directory);p['lr_disparity.f32'][17]=10
    entry=g['observations']['lr_disparity.f32'];path=directory/entry['file'];path.write_bytes(p['lr_disparity.f32'].tobytes());entry['sha256']=view.base.sha256(path);(directory/'geometry.json').write_bytes(view.preprocess.canonical(g))
    with pytest.raises(ValueError,match='correspondence support'):view.load_fixture(directory)


@pytest.mark.parametrize('validation_mode',['both','native-only'])
def test_explicit_validation_split_preserves_outputs_and_reports_scope(tmp_path,monkeypatch,validation_mode):
    for name in view.legacy.VALIDATION_ENVIRONMENT:monkeypatch.delenv(name,raising=False)
    if validation_mode=='native-only':
        for name in view.legacy.VALIDATION_ENVIRONMENT:monkeypatch.setenv(name,'validation')
    test_mocked_full_eleven_plane_chain_preserves_exact_controls(tmp_path,monkeypatch,'tie','perspective','constant',None,validation_mode)


@pytest.mark.parametrize('mutation',['native_diag_stdout','native_diag_stderr','control_diag_stdout','control_diag_stderr'])
def test_explicit_split_never_suppresses_diagnostics_or_retries(tmp_path,monkeypatch,mutation):
    for name in view.legacy.VALIDATION_ENVIRONMENT:monkeypatch.delenv(name,raising=False)
    test_mocked_full_eleven_plane_chain_preserves_exact_controls(tmp_path,monkeypatch,'tie','perspective','constant',mutation,'native-only')


def test_native_only_validation_requires_explicit_native_validation_before_fixture(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('must reject before fixture/output/child')
    monkeypatch.setattr(view,'load_fixture',forbidden);monkeypatch.setattr(view.base,'create_output',forbidden);monkeypatch.setattr(view.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='requires.*validation'):
        view.run_comparison('missing','native','rhi','compiler',tmp_path/'out',native_only_validation=True)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('mutation',['control_validation_claim','native_validation_claim','control_validation_qualified'])
def test_explicit_split_rejects_false_child_validation_claims(tmp_path,monkeypatch,mutation):
    for name in view.legacy.VALIDATION_ENVIRONMENT:monkeypatch.delenv(name,raising=False)
    test_mocked_full_eleven_plane_chain_preserves_exact_controls(tmp_path,monkeypatch,'tie','perspective','constant',mutation,'native-only')
