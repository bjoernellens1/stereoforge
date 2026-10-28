"""CPU-only right-reference mirror protocol; no native/GPU execution."""
import copy
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import compare_daxa_stereo_right as right


def profile():return dict(width=8,height=5,disparities=16,paths=4,p1=8,p2=32,subpixel=True)


def test_mirror_reverses_elements_within_rows_not_bytes():
    raw=struct.pack('<6I',0x12345678,2,3,4,5,6)
    mirrored=right.mirror_rows(raw,3,2)
    assert struct.unpack('<6I',mirrored)==(3,2,0x12345678,6,5,4)
    assert right.mirror_rows(mirrored,3,2)==raw
    with pytest.raises(ValueError):right.mirror_rows(raw[:-1],3,2)


def report(p,identity):
    stats=dict(p50=1.,p95=2.,mean=1.1)
    return dict(p,variant='packed-specialized',reference='right',frames=1,warmup=0,slots=1,metric_depth=False,required_wave_size=32,full_subgroups_required=True,
        scope='single_reference_retained_luma_no_lr_consistency',integer_reference_checked=True,packed_penalties_reference_checked=True,compound_float_reference_checked=False,
        validation_errors_after_teardown=0,validation_warnings_after_teardown=0,device=dict(device='mock',slangc_sha256='compiler',validation_errors=0,validation_warnings=0),
        shader_source_sha256_pair='wrapper:ops',stages=dict(census=stats),**{key:stats for key in ('gpu_graph_ms','cpu_submit_ms','cpu_wait_ms','cpu_timestamp_query_ms','submit_wait_and_query_wall_ms')},
        **right.base.packed_contract(p),diagnostic_intermediates_reconstructed=True)


@pytest.mark.parametrize('mutation',[None,'reference','variant','integer_missing','integer_false','packed_false','compound_true','metric','full_subgroups','wave','compiler','shader','packed_layout','packed_bits','plane','missing_plane','diagnostic','nonfinite','changed_source','changed_original','changed_mirror','changed_report','missing_words'])
def test_fake_children_exact_right_protocol(tmp_path,monkeypatch,mutation):
    p=profile();identity=dict(daxa=dict(compiler_sha256='compiler',shader_source_sha256={'wrapper':'wrapper','ops':'ops'}),rhi={},source_sha256={})
    probes=0
    def identities(*args):
        nonlocal probes
        probes+=1
        return dict(identity,changed=True) if mutation=='changed_source' and probes>1 else copy.deepcopy(identity)
    monkeypatch.setattr(right,'native_identities',identities)
    calls=[]
    def child(command,**kwargs):
        calls.append(command);out=tmp_path/'result'
        if '--reference' not in command:
            for eye,original in (('left','right'),('right','left')):
                path=Path(command[command.index('--'+eye)+1])
                assert path.read_bytes()==right.mirror_rows((out/(original+'.u32')).read_bytes(),p['width'],p['height'])
            for plane in right.base.OUTPUT_PLANES:
                Path(str(out/'rhi-mirrored')+'.'+plane).write_bytes(struct.pack('<40I',*range(40)))
            r=dict(width=p['width'],height=p['height'],disparities=p['disparities'],paths=p['paths'],iterations=1,vulkan_validation=False,metric_depth=False,transfer_inclusive_total_ms=1.,adapter='mock')
            return SimpleNamespace(returncode=0,stdout=json.dumps(r),stderr='')
        assert command[command.index('--reference')+1]=='right'
        assert '--check-integer-reference' in command and '--check-reference' not in command
        r=report(p,identity)
        changes={'reference':('reference','left'),'variant':('variant','packed'),'integer_false':('integer_reference_checked',False),'packed_false':('packed_penalties_reference_checked',False),'compound_true':('compound_float_reference_checked',True),'metric':('metric_depth',True),'full_subgroups':('full_subgroups_required',False),'wave':('required_wave_size',64),'packed_layout':('packed_penalty_layout','unknown'),'packed_bits':('packing_bits',16)}
        if mutation in changes:r[changes[mutation][0]]=changes[mutation][1]
        if mutation=='integer_missing':del r['integer_reference_checked']
        if mutation=='compiler':r['device']['slangc_sha256']='changed'
        if mutation=='shader':r['shader_source_sha256_pair']='changed'
        if mutation=='nonfinite':r['gpu_graph_ms']['mean']=float('nan')
        (out/'right.json').write_text(json.dumps(r))
        for plane in right.base.OUTPUT_PLANES:
            if mutation=='missing_plane' and plane=='validity.u32':continue
            raw=right.mirror_rows(struct.pack('<40I',*range(40)),p['width'],p['height'])
            if mutation=='plane' and plane=='disparity.f32':raw=bytes([raw[0]^1])+raw[1:]
            Path(str(out/'right')+'.'+plane).write_bytes(raw)
        if mutation!='missing_words':Path(str(out/'right')+'.packed_penalties.u32').write_bytes(bytes(right.base.packed_contract(p)['packed_penalty_word_count']*4))
        if mutation in ('changed_original','changed_mirror'):
            path=out/('left.u32' if mutation=='changed_original' else 'mirror-right.u32');path.chmod(0o644);path.write_bytes(bytes(160))
        if mutation=='changed_report':
            original=right.base.compare_planes
            def change(*args,**kwargs):
                result=original(*args,**kwargs);(out/'right.json').write_text('{}');return result
            monkeypatch.setattr(right.base,'compare_planes',change)
        return SimpleNamespace(returncode=0,stdout='',stderr='WARNING: loader' if mutation=='diagnostic' else '')
    monkeypatch.setattr(right.subprocess,'run',child)
    result=right.run_comparison(Path('/rhi'),Path('/daxa'),Path('/compiler'),tmp_path/'result',p,frames=1,warmup=0,validation_only=True,intermediates=True)
    assert result['passed']==(mutation is None)
    if mutation is None:
        assert len(calls)==2 and result['selected'] is None
        assert result['reference']=='right' and not result['timing_contract']['comparable_end_to_end']
        assert set(result['comparison']['planes'])==set(right.base.OUTPUT_PLANES)
        assert set(result['fixture']['derived_sha256'])=={'mirror-right.u32','mirror-left.u32'}
        assert set(result['rhi_unmirrored_sha256'])==set(right.base.OUTPUT_PLANES)


def test_bounds_and_collision_reject_before_children(tmp_path,monkeypatch):
    monkeypatch.setattr(right.subprocess,'run',lambda *args,**kwargs:pytest.fail('no child permitted'))
    for changes in ({'width':0},{'height':100000},{'disparities':256},{'paths':3},{'p2':20000},{'subpixel':1}):
        with pytest.raises(ValueError):right.run_comparison(Path('/r'),Path('/d'),Path('/c'),tmp_path/'result',dict(profile(),**changes),validation_only=True)
    output=tmp_path/'exists';output.mkdir()
    with pytest.raises(ValueError):right.run_comparison(Path('/r'),Path('/d'),Path('/c'),output,profile(),validation_only=True)


def test_identity_requires_all_four_actual_right_spirv_files(tmp_path,monkeypatch):
    old={str(i):'old-control' for i in range(21)}
    monkeypatch.setattr(right.base,'native_identities',lambda *args:dict(daxa=dict(offline_shader_directory=str(tmp_path),offline_spirv_sha256=dict(old)),source_sha256={}))
    for entry in right.RIGHT_ENTRIES:(tmp_path/('daxa-stereo-'+entry+'.spv')).write_bytes(entry.encode())
    identity=right.native_identities(Path('/r'),Path('/d'),Path('/c'))
    assert len(identity['daxa']['offline_spirv_sha256'])==25
    for entry in right.RIGHT_ENTRIES:assert identity['daxa']['offline_spirv_sha256'][entry]==right.base.sha256(tmp_path/('daxa-stereo-'+entry+'.spv'))
    (tmp_path/('daxa-stereo-'+right.RIGHT_ENTRIES[-1]+'.spv')).unlink()
    with pytest.raises(OSError):right.native_identities(Path('/r'),Path('/d'),Path('/c'))


def test_real_cli_small_and_720p_commands_without_children(tmp_path):
    for p,frames,warmup in ((profile(),1,0),(dict(profile(),width=1280,height=720,disparities=128),30,3)):
        right.validate_bounds(p,frames,warmup,2,64,1234567,frames==1)
        calls=right.commands(Path('/rhi'),Path('/daxa'),tmp_path,p,frames,warmup,2,64,True,False)
        assert set(calls)=={'rhi','right'}
        assert '--reference' not in calls['rhi'] and '--warmup' not in calls['rhi']
        assert calls['right'][calls['right'].index('--warmup')+1]==str(warmup)
        assert calls['right'][calls['right'].index('--wave-size')+1]=='64'
        assert '--dump-intermediates' not in calls['right']
        for key in ('rhi','right'):assert '--validation' in calls[key]


@pytest.mark.parametrize('validation_only',[False,True])
def test_d96_right_profile_admission_preserves_independent_integer_gate(tmp_path,validation_only):
    p=dict(profile(),width=13,height=7,disparities=96)
    right.validate_bounds(p,1 if validation_only else 30,0 if validation_only else 3,2,32,1234567,validation_only)
    commands=right.commands(Path('/rhi'),Path('/daxa'),tmp_path,p,1,0,2,32,True,True)
    assert all(c[c.index('--disparities')+1]=='96' for c in commands.values())
    assert '--check-integer-reference' in commands['right'] and '--reference' in commands['right']
    assert commands['right'][commands['right'].index('--reference')+1]=='right'
