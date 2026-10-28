"""Complete CPU wire/parser contracts; fake executables never use a GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))


def harness():
    assert importlib.util.find_spec('compare_daxa_refinement'), 'refinement comparator is missing'
    import compare_daxa_refinement
    return compare_daxa_refinement


@pytest.fixture
def wire(tmp_path):
    h = harness()
    import refinement_quality_fixtures as quality
    from stereoforge import refinement as r
    family = quality.export_scene_family(tmp_path/'family', width=64, height=36)
    root = family.parent
    context = h.load_context(root/'half/manifest.json', root/'full/manifest.json')
    settings = r.Settings(radius=4, subpixel=False)
    prior = np.full((2, 18, 32), 2, np.float32)
    confidence = np.ones_like(prior); validity = np.ones_like(prior, bool)
    result = r.refine_pair(context['images'], context['support'], prior, confidence, validity,
                           context['geometry'], context['source_id'], settings)
    planes = h.oracle_planes(result, context)
    dump = tmp_path/'dump'; dump.mkdir()
    def entry(name, a, coarse=False):
        a = np.asarray(a, dtype='<u4' if name.endswith('.u32') else '<i8' if name.endswith('.i64') else '<f4')
        p = dump/(('coarse_' if coarse else '')+name); p.write_bytes(a.tobytes())
        e = dict(path=str(p), sha256=h.sha256(p))
        if not coarse: e['elements'] = a.size
        return e
    compiler = dict(version='2026.19',commit='0fb6b7587a94fa33d88141ec9930766eedc6f59a',package_sha256='bd6cfc47b7353b2cffa36a866b1584749f76f4b2dbebf6dda26bc3fb40dc3c0e', compiler_sha256='a'*64, floating_point_mode='precise', spirv_sha256={e:'b'*64 for e in h.ENTRIES})
    frozen = dict(report_identity=dict(refinement_shader_compiler=compiler, coarse_shader_compiler=dict(version='2026.12',compiler_sha256='c'*64), shader_identity='d'*64+':'+'e'*64), device_identity={'device':'CPU fake protocol fixture'})
    report = h.expected_report(context, settings, False)
    report.update(copy.deepcopy(frozen['report_identity']), device=copy.deepcopy(frozen['device_identity']), files={name:entry(name,a) for name,a in planes.items()}, coarse_files={name:entry(name,a,True) for name,a in [('disparity.f32',prior),('confidence.f32',confidence),('validity.u32',validity)]}, requested_refinement_buffer_bytes=1234)
    return h, report, context, frozen, dump, settings, family


def test_complete_wire_compares_and_never_claims_global_float_qualification(wire):
    h, report, context, frozen, dump, settings, _ = wire
    result = h.compare_report(report, context, frozen, dump, settings)
    assert result['integer_masks_identity_exact']
    assert result['operation_bounds_checked']
    assert result['gpu_global_arithmetic_qualified'] is False
    assert result['physical_qualified'] is False


@pytest.mark.parametrize('field,value', [('source_id',True),('source_id',-1),('radius',2),('recovery',True),('diagnostic_injection',True),('coarse_manifest_sha256','0'*64),('target_rectified_identity','0'*64)])
def test_report_binding_tamper_fails(wire,field,value):
    h, report, context, frozen, dump, settings, _ = wire
    report[field] = value
    with pytest.raises(ValueError): h.compare_report(report,context,frozen,dump,settings)


@pytest.mark.parametrize('plane', ['prediction_validity.u32','census.u32','winner.u32','supported_candidates.u32','evaluated_candidates.u32','reason.u32','metric_validity.u32','source_ids.i64'])
def test_exact_plane_tamper_fails_even_with_updated_file_hash(wire,plane):
    h, report, context, frozen, dump, settings, _ = wire
    path = Path(report['files'][plane]['path']); b = bytearray(path.read_bytes()); b[0] ^= 1; path.write_bytes(b)
    report['files'][plane]['sha256'] = h.sha256(path)
    with pytest.raises(ValueError): h.compare_report(report,context,frozen,dump,settings)


def test_noncanonical_invalid_nan_is_rejected(wire):
    h, report, context, frozen, dump, settings, _ = wire
    entry = report['files']['depth.f32']; path = Path(entry['path']); a = np.frombuffer(path.read_bytes(),'<u4').copy(); a[0]=0x7fc00001;path.write_bytes(a.tobytes());entry['sha256']=h.sha256(path)
    with pytest.raises(ValueError): h.compare_report(report,context,frozen,dump,settings)


def test_dump_parent_symlink_and_size_and_compiler_tamper_rejected(wire,tmp_path):
    h, report, context, frozen, dump, settings, _ = wire
    original = copy.deepcopy(report)
    report['refinement_shader_compiler']['compiler_sha256']='0'*64
    with pytest.raises(ValueError): h.compare_report(report,context,frozen,dump,settings)
    report=original;entry=report['files']['luma.u32'];entry['elements']+=1
    with pytest.raises(ValueError): h.compare_report(report,context,frozen,dump,settings)
    entry['elements']-=1; alias=tmp_path/'alias';alias.symlink_to(dump,target_is_directory=True);entry['path']=str(alias/'luma.u32')
    with pytest.raises(ValueError): h.compare_report(report,context,frozen,dump,settings)


def test_truth_quality_denominator_includes_missing_estimates_and_regions(wire):
    h, report, context, frozen, dump, settings, family = wire
    truth = h.load_truth(family, context)
    result = h.compare_report(report,context,frozen,dump,settings,truth=truth)
    q=result['world_quality']
    assert q['scope']=='analytic synthetic development; not physical or heldout'
    for eye in ['left','right']:
        assert q[eye]['all']['eligible'] >= q[eye]['all']['metric_accepted']
        assert 0 <= q[eye]['all']['metric_completeness'] <= 1
        assert {'thin','edges','near','far'} <= q[eye].keys()


def test_comparator_module_exists():
    assert importlib.util.find_spec('compare_daxa_refinement'), 'refinement comparator is missing'


def fake_dispatch_freeze(wire, tmp_path):
    """An explicitly frozen CPU executable implementing the real CLI/wire."""
    h,report,context,frozen,dump,settings,family=wire
    spirv=tmp_path/'spirv';spirv.mkdir()
    import struct
    # Valid instruction stream with no floating ops, for parser-only fake native.
    data=struct.pack('<5I',0x07230203,0x00010500,0,1,0)
    for e in h.ENTRIES:(spirv/f'daxa-refinement-{e}.spv').write_bytes(data)
    compilers=[]
    for version in ['2026.12','2026.19']:
        p=tmp_path/('compiler-'+version)
        import shutil
        shutil.copyfile('/bin/true',p);p.chmod(0o755);compilers.append(p)
    template=tmp_path/'template.json';native=tmp_path/'fake-native'
    native.write_text('#!'+sys.executable+'\n'+'''import json,shutil,sys\nfrom pathlib import Path\nargs=sys.argv[1:]\nif args==['--shader-directory']:\n print(SHADERS);raise SystemExit(0)\nopts=dict(zip(args[::2],args[1::2]))\nr=json.loads(Path(TEMPLATE).read_text())\nfor group in ['files','coarse_files']:\n for name,e in r[group].items():\n  p=opts['--dump-prefix']+'.'+('coarse_' if group=='coarse_files' else '')+name\n  shutil.copyfile(e['path'],p);e['path']=p\nPath(opts['--output']).write_text(json.dumps(r))\n'''.replace('SHADERS',repr(str(spirv))).replace('TEMPLATE',repr(str(template))))
    native.chmod(0o755)
    frozen=copy.deepcopy(frozen)
    identity=frozen['report_identity'];identity['coarse_shader_compiler']['compiler_sha256']=h.sha256(compilers[0]);identity['refinement_shader_compiler']['compiler_sha256']=h.sha256(compilers[1]);identity['refinement_shader_compiler']['spirv_sha256']={e:h.sha256(spirv/f'daxa-refinement-{e}.spv') for e in h.ENTRIES};identity['shader_identity']=h.sha256(h.ROOT/'python/stereoforge/shaders/daxa/refinement.slang')+':'+h.sha256(h.ROOT/'python/stereoforge/shaders/ops/refinement.slang')
    report=copy.deepcopy(report);report.update(copy.deepcopy(identity))
    frozen['device_identity']={k:report['device'].get(k,'CPU-fake') for k in h.DEVICE_KEYS};report['device'].update(frozen['device_identity'])
    template.write_text(json.dumps(report))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(modules={f'daxa-refinement-{e}.spv':h.audit_spirv(spirv/f'daxa-refinement-{e}.spv') for e in h.ENTRIES})))
    checkpoint=tmp_path/'checkpoint.json';checkpoint.write_text(json.dumps({str(native):h.sha256(native)}))
    paths=h._source_paths()+h._fixture_paths(context['coarse_path'],context['coarse'])+h._fixture_paths(context['target_path'],context['target'])+[native,*compilers,audit,checkpoint,template,*spirv.iterdir(),*dump.iterdir()]
    descriptor=lambda p:dict(file=str(p.resolve()),sha256=h.sha256(p))
    runtime={label:h.runtime_dependencies(exe) for label,exe in [('native',native),('coarse_compiler',compilers[0]),('refinement_compiler',compilers[1])]}
    paths += [Path(p) for closure in runtime.values() for p in closure]
    inventory={str(p.resolve()):h.sha256(p) for p in paths}
    aliases={str(p.absolute()):str(p.resolve()) for p in paths if p.absolute()!=p.resolve()}
    frozen.update(schema='stereoforge.refinement_freeze.v1',native=descriptor(native),coarse_compiler=descriptor(compilers[0]),refinement_compiler=descriptor(compilers[1]),audit=descriptor(audit),checkpoint=descriptor(checkpoint),shader_directory=str(spirv),file_sha256=inventory,runtime_alias_targets=aliases,coarse_manifest=descriptor(context['coarse_path']),target_manifest=descriptor(context['target_path']),truth_family=None,runtime_dependency_sha256=runtime)
    path=tmp_path/'freeze.json';path.write_bytes(h.fixture.canonical(frozen))
    return path,native,frozen,context


def test_real_cli_dispatch_matches_direct_api_without_parser_mock(wire,tmp_path):
    import subprocess
    h=wire[0];path,native,frozen,context=fake_dispatch_freeze(wire,tmp_path)
    direct=h.run_comparison(context['coarse_path'],context['target_path'],native,path,h.sha256(path),tmp_path/'direct')
    command=[sys.executable,str(h.ROOT/'benchmarks/compare_daxa_refinement.py'),'compare','--coarse-manifest',str(context['coarse_path']),'--target-manifest',str(context['target_path']),'--native',str(native),'--freeze',str(path),'--freeze-sha256',h.sha256(path),'--output',str(tmp_path/'cli')]
    child=subprocess.run(command,capture_output=True,text=True)
    assert child.returncode==0,child.stderr
    cli=json.loads(child.stdout)
    assert cli['float_residuals']==direct['float_residuals']
    assert cli['integer_masks_identity_exact'] is True and cli['gpu_global_arithmetic_qualified'] is False
    assert h.native_command(native,context,tmp_path/'direct',h.oracle.Settings(radius=4,subpixel=False),False)==direct['command']
    with pytest.raises(ValueError):h.run_comparison(context['coarse_path'],context['target_path'],native,path,h.sha256(path),tmp_path/'direct')


@pytest.mark.parametrize('kind',['compiler','spirv','source','wrong-freeze-hash','fixture'])
def test_explicit_freeze_tamper_fails(wire,tmp_path,kind):
    h=wire[0];path,native,frozen,context=fake_dispatch_freeze(wire,tmp_path)
    digest=h.sha256(path)
    if kind=='wrong-freeze-hash':digest='0'*64
    elif kind=='compiler':Path(frozen['refinement_compiler']['file']).write_text('changed')
    elif kind=='spirv':next(Path(frozen['shader_directory']).glob('*.spv')).write_bytes(b'changed')
    elif kind=='source':
        # A copied bound source stand-in tests closure hashing without modifying
        # any shared repository source or weakening actual required-source checks.
        source=tmp_path/'source.cpp';source.write_text('before');frozen['file_sha256'][str(source)]=h.sha256(source);path.write_bytes(h.fixture.canonical(frozen));digest=h.sha256(path);source.write_text('after')
    else:Path(context['target_path']).write_bytes(b'changed')
    with pytest.raises(ValueError):h.verify_freeze(path,digest,native,context)


def test_spirv_no_contraction_relaxed_fma_and_malformed_fail_closed(tmp_path):
    h=harness();import struct
    def check(words):
        p=tmp_path/'test.spv';p.write_bytes(struct.pack('<'+'I'*(len(words)+5),0x07230203,0x10500,0,20,0,*words));return h.audit_spirv(p)
    # OpFAdd resultid2 with a NoContraction decoration on2.
    add=[(5<<16)|129,1,2,3,4]
    precise=[(3<<16)|71,2,42]
    assert check(precise+add)['floating_arithmetic']==1
    for instructions in [add,precise+add+[(3<<16)|71,2,0],precise+add+[(3<<16)|71,2,40],[(7<<16)|12,1,2,3,50,4,5],[(20<<16)|129]]:
        with pytest.raises(ValueError):check(instructions)


def test_bounded_confidence_residual_allowed_but_exact_mask_difference_never(wire):
    h,report,context,frozen,dump,settings,_=wire
    entry=report['files']['raw_confidence.f32'];p=Path(entry['path']);a=np.frombuffer(p.read_bytes(),'<f4').copy()
    ix=np.flatnonzero((a>0)&(a<1))[0];a[ix]=np.nextafter(a[ix],np.float32(np.inf));p.write_bytes(a.tobytes());entry['sha256']=h.sha256(p)
    for name,validity in [('confidence.f32','validity.u32'),('metric_confidence.f32','metric_validity.u32')]:
        valid=np.frombuffer(Path(report['files'][validity]['path']).read_bytes(),'<u4')
        if valid[ix]:
            e=report['files'][name];p=Path(e['path']);c=np.frombuffer(p.read_bytes(),'<f4').copy();c[ix]=a[ix];p.write_bytes(c.tobytes());e['sha256']=h.sha256(p)
    result=h.compare_report(report,context,frozen,dump,settings)
    assert result['float_residuals']['raw_confidence.f32']['different_nominal_bits']==1
    e=report['files']['raw_validity.u32'];p=Path(e['path']);v=np.frombuffer(p.read_bytes(),'<u4').copy();v[ix]=0;p.write_bytes(v.tobytes());e['sha256']=h.sha256(p)
    with pytest.raises(ValueError):h.compare_report(report,context,frozen,dump,settings)


def test_quality_rejected_domains_are_explicit_and_missing_outputs_stay_in_denominator():
    h=harness();shape=(1,6)
    t={k:np.zeros(shape,bool) for k in ['occluded','off_image','out_of_range','mixed_bilinear','region_thin','region_edges','region_near','region_far']}
    t.update(valid=np.ones(shape,bool),visible=np.ones(shape,bool),remap_support=np.ones(shape,bool),disparity=np.full(shape,4.),axial_z_m=np.full(shape,2.))
    t['mixed_bilinear'][0,0]=True;t['out_of_range'][0,1]=True;t['remap_support'][0,2]=False
    raw=np.array([[[4,4,4,4,8,np.nan]]]*2,np.float32);valid=np.array([[[1,1,1,1,1,0]]]*2,np.uint32)
    actual={'raw_disparity.f32':raw,'raw_validity.u32':valid,'metric_validity.u32':valid,'validity.u32':valid,'depth.f32':np.full((2,1,6),2.,np.float32)}
    quality=h.evaluate_quality(actual,{'left':t,'right':copy.deepcopy(t)},np.ones((2,1,6),bool))
    for eye in ['left','right']:
        row=quality[eye]['all']
        assert row['eligible']==3 and row['metric_accepted']==2
        assert row['metric_completeness']==2/3 and row['correct_metric_completeness']==1/3
        assert row['lr_accepted_total']==5 and row['lr_false_accepted']==1
        assert row['lr_accepted_out_of_range']==1 and row['lr_accepted_unsupported']==1
        assert row['lr_accepted_mixed']==1
        assert quality[eye]['mixed']['eligible']==0
        assert quality[eye]['mixed']['measurement_domain']=='ineligible_mixed_footprint_diagnostic'
        assert quality[eye]['mixed']['region_pixels']==1


@pytest.mark.parametrize('label',['coarse_compiler','refinement_compiler'])
def test_each_compiler_runtime_dependency_map_is_rechecked(wire,tmp_path,label):
    h=wire[0];path,native,frozen,context=fake_dispatch_freeze(wire,tmp_path)
    closure=frozen['runtime_dependency_sha256'][label]
    shared=next(p for p in closure if p!=frozen[label]['file'])
    closure[shared]='0'*64;path.write_bytes(h.fixture.canonical(frozen))
    with pytest.raises(ValueError,match='runtime dependency resolution '+label):h.verify_freeze(path,h.sha256(path),native,context)


def test_runtime_alias_same_byte_retarget_is_rejected(wire,tmp_path):
    h=wire[0];path,native,frozen,context=fake_dispatch_freeze(wire,tmp_path)
    first=tmp_path/'first.so';second=tmp_path/'second.so';alias=tmp_path/'soname.so'
    first.write_bytes(b'identical bytes');second.write_bytes(first.read_bytes());alias.symlink_to(first)
    frozen['file_sha256'].update({str(first):h.sha256(first),str(second):h.sha256(second)})
    frozen['runtime_alias_targets'][str(alias)]=str(first)
    path.write_bytes(h.fixture.canonical(frozen));digest=h.sha256(path)
    h.verify_freeze(path,digest,native,context)
    alias.unlink();alias.symlink_to(second)
    with pytest.raises(ValueError,match='runtime alias retargeted'):h.verify_freeze(path,digest,native,context)


def test_unsupported_arithmetic_domain_is_error_not_waived_metric_check(wire):
    h,report,context,frozen,dump,settings,_=wire
    from stereoforge import refinement as r
    context=copy.copy(context);context['geometry']=r.Geometry(1e-20,1e-20)
    coarse={name:np.frombuffer(Path(entry['path']).read_bytes(),'<u4' if name.endswith('.u32') else '<f4').reshape(2,18,32) for name,entry in report['coarse_files'].items()}
    result=r.refine_pair(context['images'],context['support'],coarse['disparity.f32'],coarse['confidence.f32'],coarse['validity.u32'],context['geometry'],context['source_id'],settings)
    for name,array in h.oracle_planes(result,context).items():
        e=report['files'][name];p=Path(e['path']);dtype='<i8' if name.endswith('.i64') else '<u4' if name.endswith('.u32') else '<f4';p.write_bytes(np.asarray(array,dtype).tobytes());e['sha256']=h.sha256(p)
    report['metric_calibration']=h.expected_report(context,settings,False)['metric_calibration']
    with pytest.raises(ValueError,match='outside finite normal'):h.compare_report(report,context,frozen,dump,settings)


def test_strict_json_duplicate_nonfinite_nonobject_and_missing_wire_rejected(wire):
    h,report,context,frozen,dump,settings,_=wire
    for text in ['{"source_id":7,"source_id":8}','{"x":NaN}','[]','{"x":1e999}']:
        with pytest.raises(ValueError):h.strict_json(text)
    report.pop('coarse_files')
    with pytest.raises((ValueError,KeyError)):h.compare_report(report,context,frozen,dump,settings)
