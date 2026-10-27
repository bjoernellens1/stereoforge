"""CPU-only comparison orchestration: real byte inputs, strict output contracts."""
import copy
import json
from pathlib import Path
import struct
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import compare_daxa_stereo as compare


def profile():
    return dict(width=32,height=16,disparities=32,paths=4,p1=8,p2=32,subpixel=True)


def test_shared_fixture_is_explicit_exact_little_endian_uint32(tmp_path):
    manifest=compare.write_fixture(tmp_path,profile(),1234567)
    left=(tmp_path/'left.u32').read_bytes();right=(tmp_path/'right.u32').read_bytes()
    assert len(left)==len(right)==32*16*4
    a=struct.unpack('<512I',left);b=struct.unpack('<512I',right)
    assert all(0<=value<=255 for value in a)
    for y in range(16):
        assert b[y*32:y*32+28]==a[y*32+4:y*32+32]
        assert b[y*32+28:y*32+32]==(0,0,0,0)
    assert manifest['left_sha256']==compare.sha256(tmp_path/'left.u32')


@pytest.mark.parametrize('text',['{"x":NaN}','{"x":Infinity}','{"x":1e999}','{"x":1,"x":2}','{} garbage'])
def test_nonfinite_duplicate_and_malformed_json_rejected(text):
    with pytest.raises(ValueError):compare.strict_json(text)


@pytest.mark.parametrize('field',list(profile()))
def test_changed_daxa_parameters_rejected(field):
    p=profile();r=dict(p,variant='wave',frames=1,warmup=0,slots=1,metric_depth=False,required_wave_size=64,full_subgroups_required=True,device=dict(subgroups=dict(default_size=64)))
    compare.validate_profile(r,p,'wave',1,0,1)
    changed=copy.deepcopy(r);changed[field]=not r[field] if field=='subpixel' else r[field]+1
    with pytest.raises(ValueError):compare.validate_profile(changed,p,'wave',1,0,1)


def test_frozen_rhi_missing_parameter_echo_is_command_bound_not_invented():
    p=profile();r={key:p[key] for key in ('width','height','disparities','paths')};r['iterations']=30
    compare.validate_rhi_profile(r,p,30)
    assert all(key not in r for key in ('p1','p2','subpixel'))
    with pytest.raises(ValueError):compare.validate_rhi_profile(dict(r,width=33),p,30)


def test_missing_or_different_planes_rejected(tmp_path):
    p=profile();a=tmp_path/'a';b=tmp_path/'b'
    for prefix in (a,b):
        for plane in compare.OUTPUT_PLANES:
            Path(str(prefix)+'.'+plane).write_bytes(bytes(32*16*4))
    assert compare.compare_planes(a,b,p)['passed']
    path=Path(str(b)+'.validity.u32');path.write_bytes(b'changed')
    with pytest.raises(ValueError):compare.compare_planes(a,b,p)
    path.unlink()
    with pytest.raises(ValueError):compare.compare_planes(a,b,p)


@pytest.mark.parametrize('message',['Validation Error: VUID-test-broken','SYNC-HAZARD-READ-AFTER-WRITE','Daxa assertion failed','WARNING: loader'])
def test_diagnostics_veto_zero_child_exit(message):
    with pytest.raises(ValueError):compare.require_child_clean(0,'{}',message)


def test_existing_or_symlink_output_directory_is_not_reused(tmp_path):
    path=tmp_path/'output';compare.create_output(path)
    with pytest.raises(ValueError):compare.create_output(path)
    link=tmp_path/'link';link.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError):compare.create_output(link)


def test_commands_pass_same_actual_luma_paths_and_complete_profile(tmp_path):
    p=profile();paths={'left':tmp_path/'left.u32','right':tmp_path/'right.u32'}
    commands=compare.commands(Path('/native_stereo'),Path('/daxa_stereo'),tmp_path,p,paths,30,3,1,False,True)
    assert set(commands)=={'rhi','baseline','wave'}
    for command in commands.values():
        for key,value in dict(p,left=paths['left'],right=paths['right']).items():
            expected='1' if value is True else str(value)
            assert command[command.index('--'+key)+1]==expected
    assert '--iterations' in commands['rhi'] and '--warmup' not in commands['rhi']
    assert '--dump-intermediates' in commands['baseline']


def test_identity_stability_ignores_ldd_load_addresses_but_binds_dependency_bytes(tmp_path,monkeypatch):
    import shlex
    import compare_daxa_view
    directory=tmp_path/'spirv';directory.mkdir()
    for entry in ('census','cost','aggregate_baseline','aggregate_wave','select_output','depth_convert','inspect'):
        (directory/('daxa-stereo-'+entry+'.spv')).write_bytes(b'identity bytes')
    binary=tmp_path/'native';binary.write_text('#!/bin/sh\nprintf "%s\\n" '+shlex.quote(str(directory))+'\n');binary.chmod(0o755)
    compiler=tmp_path/'slangc';compiler.write_bytes(b'compiler bytes')
    address=iter(range(100))
    monkeypatch.setattr(compare_daxa_view,'dependencies',lambda *args:dict(ldd=[str(next(address))],sha256={'library.so':'a'*64}))
    first=compare.native_identities(binary,binary,compiler)
    second=compare.native_identities(binary,binary,compiler)
    assert first==second
    with pytest.raises(OSError):compare.native_identities(binary,binary,compiler,True)
    for entry in ('clear_packed','aggregate_packed8','aggregate_packed16','select_packed'):
        (directory/('daxa-stereo-'+entry+'.spv')).write_bytes(b'packed identity bytes')
    packed=compare.native_identities(binary,binary,compiler,True)
    assert len(first['daxa']['offline_spirv_sha256'])==7 and len(packed['daxa']['offline_spirv_sha256'])==11
    (directory/'daxa-stereo-select_packed.spv').write_bytes(b'changed packed code')
    assert packed!=compare.native_identities(binary,binary,compiler,True)
    assert first==compare.native_identities(binary,binary,compiler)
    with pytest.raises(OSError):compare.native_identities(binary,binary,compiler,include_specialized=True)
    for entry in ('aggregate_specialized8','aggregate_specialized16','select_specialized8','select_specialized16'):
        (directory/('daxa-stereo-'+entry+'.spv')).write_bytes(b'specialized code')
    specialized=compare.native_identities(binary,binary,compiler,include_specialized=True)
    assert len(specialized['daxa']['offline_spirv_sha256'])==15
    (directory/'daxa-stereo-select_specialized8.spv').write_bytes(b'stale specialized code')
    assert specialized!=compare.native_identities(binary,binary,compiler,include_specialized=True)
    with pytest.raises(OSError):compare.native_identities(binary,binary,compiler,include_d128=True)
    for entry in ('aggregate_d1288','aggregate_d12816'):
        (directory/('daxa-stereo-'+entry+'.spv')).write_bytes(b'd128 code')
    d128=compare.native_identities(binary,binary,compiler,include_d128=True)
    assert len(d128['daxa']['offline_spirv_sha256'])==17
    (directory/'daxa-stereo-aggregate_d1288.spv').write_bytes(b'changed d128 code')
    assert d128!=compare.native_identities(binary,binary,compiler,include_d128=True)
    monkeypatch.setattr(compare_daxa_view,'dependencies',lambda *args:dict(ldd=['different address'],sha256={'library.so':'b'*64}))
    assert compare.native_identities(binary,binary,compiler)!=first


def test_shared_fixture_mutation_is_detected_before_next_child(tmp_path):
    p=profile();manifest=compare.write_fixture(tmp_path,p,1234567)
    paths={eye:tmp_path/(eye+'.u32') for eye in ('left','right')}
    compare.verify_fixture(paths,manifest)
    paths['right'].chmod(0o644);paths['right'].write_bytes(b'changed fixture')
    with pytest.raises(ValueError,match='luma bytes changed'):compare.verify_fixture(paths,manifest)


def test_plane_comparison_is_bit_exact_including_invalid_nan_payloads(tmp_path):
    p=profile();a=tmp_path/'a';b=tmp_path/'b'
    for prefix in (a,b):
        for plane in compare.OUTPUT_PLANES:Path(str(prefix)+'.'+plane).write_bytes(struct.pack('<I',0x7fc00000 if plane=='disparity.f32' else 0)*(32*16))
    assert compare.compare_planes(a,b,p)['passed']
    path=Path(str(b)+'.disparity.f32');data=bytearray(path.read_bytes());data[0]=1;path.write_bytes(data)
    assert not compare.compare_planes(a,b,p)['passed']


@pytest.mark.parametrize('mode',['default','packed','changed_plane','unchecked_words','wrong_layout','unmarked_reconstruction','specialized','specialized_changed_plane','specialized_unchecked_words','specialized_wrong_layout','specialized_unmarked_reconstruction','specialized_missing_integer_flag','d128','d128_changed_plane','d128_unchecked_words','d128_missing_integer_flag','d128_wrong_variant'])
def test_cpu_only_full_orchestration_records_shared_bytes_and_distinct_timing_scopes(tmp_path,monkeypatch,mode):
    from types import SimpleNamespace
    directory=tmp_path/'comparison'
    identities=dict(daxa=dict(compiler_sha256='compiler',shader_source_sha256={'wrapper':'wrapper','ops':'ops'}))
    monkeypatch.setattr(compare,'native_identities',lambda *args:copy.deepcopy(identities))
    calls=[]
    d128=mode.startswith('d128');specialized=mode.startswith('specialized') or d128
    target='packed-d128' if d128 else 'packed-specialized' if specialized else 'packed'
    failure=mode.removeprefix('specialized_').removeprefix('d128_')
    def child(command,**kwargs):
        calls.append(command)
        flags={'--dump-intermediates','--check-integer-reference'};opt={};i=1
        while i<len(command):
            key=command[i]
            if key in flags:opt[key]=True;i+=1
            else:opt[key]=command[i+1];i+=2
        p={key:int(opt['--'+key]) for key in ('width','height','disparities','paths','p1','p2')};p['subpixel']=bool(int(opt['--subpixel']))
        n=p['width']*p['height'];is_rhi=command[0].endswith('rhi')
        if is_rhi:
            report=dict(p,adapter='CPU protocol fixture',iterations=int(opt['--iterations']),vulkan_validation=False,metric_depth=False,transfer_inclusive_total_ms=10)
            prefix=opt['--output']
        else:
            stats=dict(p50=1,p95=2,mean=1.1)
            report=dict(p,variant=opt['--variant'],frames=int(opt['--frames']),warmup=int(opt['--warmup']),slots=int(opt['--slots']),metric_depth=False,
                scope='single_reference_retained_luma_no_lr_consistency',gpu_graph_ms=stats,cpu_submit_ms=stats,cpu_wait_ms=stats,
                cpu_timestamp_query_ms=stats,submit_wait_and_query_wall_ms=stats,stages={'census':stats},
                device=dict(device='CPU protocol fixture',validation_errors=0,validation_warnings=0,slangc_sha256='compiler',subgroups=dict(default_size=64)),shader_source_sha256_pair='wrapper:ops',
                required_wave_size=64 if opt['--variant']!='baseline' else 0,full_subgroups_required=opt['--variant']!='baseline',
                integer_reference_checked='--check-integer-reference' in opt,
                validation_errors_after_teardown=0,validation_warnings_after_teardown=0)
            prefix=opt['--dump-prefix']
            if opt['--variant'] in compare.PACKED_VARIANTS:
                report.update(packed_report(p,True),packed_penalties_reference_checked=True,scalar_reference_checked=False,compound_float_reference_checked=False)
                if opt['--variant']==target:
                    if failure=='unchecked_words':report['packed_penalties_reference_checked']=False
                    if failure=='wrong_layout':report['packed_penalty_layout']='unknown'
                    if failure=='unmarked_reconstruction':report['diagnostic_intermediates_reconstructed']=False
                    if failure=='missing_integer_flag':del report['integer_reference_checked']
                    if failure=='wrong_variant':report['variant']='packed-specialized'
                Path(prefix+'.packed_penalties.u32').write_bytes(bytes(report['packed_penalty_word_count']*4))
            Path(opt['--output']).write_text(json.dumps(report))
            if '--dump-intermediates' in opt:
                for plane in compare.INTERMEDIATE_PLANES:Path(prefix+'.'+plane).write_bytes(bytes(n*p['disparities']*4))
        for plane in compare.OUTPUT_PLANES:Path(prefix+'.'+plane).write_bytes(bytes(n*4))
        if not is_rhi and opt['--variant']==target and failure=='changed_plane':
            path=Path(prefix+'.disparity.f32');path.write_bytes(bytes([1])+bytes(n*4-1))
        return SimpleNamespace(returncode=0,stdout=json.dumps(report) if is_rhi else '',stderr='')
    monkeypatch.setattr(compare.subprocess,'run',child)
    argv=['compare','--output',str(directory),'--compiler','/unused',
        '--rhi-binary','/rhi','--daxa-binary','/daxa','--width','32','--height','16','--disparities','32','--validation-only']
    if mode!='default':argv+=['--include-packed','--dump-intermediates','--check-integer-reference']
    if specialized:argv+=['--include-specialized']
    if d128:
        argv+=['--include-d128'];argv[argv.index('--disparities')+1]='128'
    monkeypatch.setattr(sys,'argv',argv)
    if mode in ('default','packed','specialized','d128'):compare.main()
    else:
        with pytest.raises(SystemExit):compare.main()
    evidence=json.loads((directory/'comparison.json').read_text())
    assert evidence['passed']==(mode in ('default','packed','specialized','d128'))
    assert len(calls)==(3 if mode=='default' else 6 if d128 else 5 if specialized else 4)
    assert evidence['performance_qualified_counts'] is False and evidence['frames']==1 and evidence['warmup']==0
    assert evidence['timing_contract']['comparable_end_to_end'] is False
    assert evidence['selected'] is None
    if mode=='default':assert len(evidence['comparisons'])==3
    if mode=='packed':
        assert evidence['comparisons']['rhi_vs_packed']['passed']
        assert evidence['comparisons']['baseline_vs_packed']['passed']
        assert evidence['comparisons']['baseline_vs_packed_reconstructed']['packed_representation'].startswith('RECONSTRUCTED')
        assert evidence['packed_word_reference_checked'] is True
        assert evidence['packed_word_dump']['representation']=='actual GPU packed penalty words'
        assert evidence['children']['packed']['report']['compound_float_reference_checked'] is False
    if mode=='specialized':
        assert evidence['include_packed'] and evidence['include_specialized']
        for other in ('rhi','baseline','packed'):assert evidence['comparisons'][other+'_vs_packed_specialized']['passed']
        assert evidence['comparisons']['baseline_vs_packed_specialized_reconstructed']['packed_representation'].startswith('RECONSTRUCTED')
        assert evidence['packed_specialized_word_reference_checked'] is True
        assert evidence['packed_specialized_word_dump']['representation']=='actual GPU packed penalty words'
    if mode=='d128':
        assert evidence['include_d128'] and evidence['include_specialized'] and evidence['include_packed']
        for other in ('rhi','baseline','packed','packed-specialized'):assert evidence['comparisons'][other+'_vs_packed_d128']['passed']
        assert evidence['packed_d128_word_reference_checked'] is True
        assert evidence['packed_d128_word_dump']['representation']=='actual GPU packed penalty words'
    assert all(command[command.index('--left')+1]==str(directory/'left.u32') for command in calls)


def test_wave_profile_requires_actual_full_subgroup_selection():
    p=profile();report=dict(p,variant='wave',frames=1,warmup=0,slots=1,metric_depth=False,
        required_wave_size=64,full_subgroups_required=True,device=dict(subgroups=dict(default_size=64)))
    compare.validate_profile(report,p,'wave',1,0,1)
    with pytest.raises(ValueError):compare.validate_profile(dict(report,full_subgroups_required=False),p,'wave',1,0,1)
    with pytest.raises(ValueError):compare.validate_profile(dict(report,required_wave_size=32),p,'wave',1,0,1)


def test_packed_is_opt_in_and_uses_same_small_and_720p_profile_files(tmp_path):
    for p in (dict(profile(),width=64,height=32,disparities=16),dict(profile(),width=1280,height=720,disparities=128)):
        paths={'left':tmp_path/'left.u32','right':tmp_path/'right.u32'}
        base=compare.commands(Path('/rhi'),Path('/daxa'),tmp_path,p,paths,30,3,1,False,False)
        packed=compare.commands(Path('/rhi'),Path('/daxa'),tmp_path,p,paths,30,3,1,False,False,include_packed=True)
        assert set(base)=={'rhi','baseline','wave'}
        assert set(packed)=={'rhi','baseline','wave','packed'}
        assert all(packed[name]==base[name] for name in base)
        for key,value in dict(p,left=paths['left'],right=paths['right'],frames=30,warmup=3).items():
            command=packed['packed'];expected=str(int(value)) if type(value) is bool else str(value)
            assert command[command.index('--'+key)+1]==expected


def packed_report(p, reconstructed=False):
    bits=8 if p['paths']*p['p2']<=255 else 16
    words=(p['disparities']+32//bits-1)//(32//bits)
    return dict(packing_bits=bits,packed_penalty_layout='pixel_major_disparity_lsb_u32',
        packed_penalty_bound=p['paths']*p['p2'],packed_penalty_words_per_pixel=words,
        packed_penalty_word_count=p['width']*p['height']*words,
        diagnostic_intermediates_reconstructed=reconstructed)


@pytest.mark.parametrize('paths,p2,bits',[(4,63,8),(4,64,16),(8,31,8),(8,32,16),(8,8191,16)])
def test_packed_width_bound_and_actual_schema(paths,p2,bits):
    p=dict(profile(),paths=paths,p2=p2);r=packed_report(p)
    assert compare.packed_contract(p)['packing_bits']==bits
    assert compare.validate_packed_report(r,p,False)==compare.packed_contract(p)
    for key in r:
        bad=dict(r);bad[key]='unknown' if type(r[key]) is str else not r[key] if type(r[key]) is bool else r[key]+1
        with pytest.raises(ValueError):compare.validate_packed_report(bad,p,False)
    with pytest.raises(ValueError):compare.packed_contract(dict(p,paths=8,p2=8192))


def test_packed_reconstructed_annotation_and_actual_word_dump_are_separate(tmp_path):
    p=profile();r=packed_report(p,True)
    contract=compare.validate_packed_report(r,p,True)
    with pytest.raises(ValueError):compare.validate_packed_report(dict(r,diagnostic_intermediates_reconstructed=False),p,True)
    prefix=tmp_path/'packed';path=Path(str(prefix)+'.packed_penalties.u32')
    with pytest.raises(ValueError):compare.packed_word_identity(prefix,contract)
    path.write_bytes(bytes(contract['packed_penalty_word_count']*4))
    identity=compare.packed_word_identity(prefix,contract)
    assert identity['representation']=='actual GPU packed penalty words'
    assert identity['bytes']==path.stat().st_size and identity['sha256']==compare.sha256(path)
    assert 'parity' not in identity
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):compare.packed_word_identity(prefix,contract)


def test_packed_profile_and_explicit_integer_reference_command():
    p=profile();r=dict(p,variant='packed',frames=1,warmup=0,slots=1,metric_depth=False,
        required_wave_size=32,full_subgroups_required=True,device=dict(subgroups=dict(default_size=64)))
    compare.validate_profile(r,p,'packed',1,0,1,32)
    with pytest.raises(ValueError):compare.validate_profile(dict(r,full_subgroups_required=False),p,'packed',1,0,1,32)
    calls=compare.commands(Path('/rhi'),Path('/daxa'),Path('/output'),p,{'left':'L','right':'R'},1,0,1,False,True,
        include_packed=True,check_integer_reference=True,wave_size=32)
    for case in ('baseline','wave','packed'):assert '--check-integer-reference' in calls[case]
    assert '--check-integer-reference' not in calls['rhi']
    assert calls['packed'][-3:]==['32','--dump-intermediates','--check-integer-reference']


def test_specialized_opt_in_implies_packed_and_preserves_existing_commands(tmp_path):
    p=profile();paths={'left':'L','right':'R'}
    old=compare.commands(Path('/rhi'),Path('/daxa'),tmp_path,p,paths,30,3,1,False,True,include_packed=True)
    new=compare.commands(Path('/rhi'),Path('/daxa'),tmp_path,p,paths,30,3,1,False,True,
        include_specialized=True,check_integer_reference=True,wave_size=32)
    assert set(new)==set(old)|{'packed-specialized'}
    for case in old:
        stripped=[x for x in new[case] if x!='--check-integer-reference']
        if case in ('wave','packed'):stripped[stripped.index('--wave-size')+1]='0'
        assert stripped==old[case]
    c=new['packed-specialized']
    assert c[c.index('--variant')+1]=='packed-specialized'
    assert c[c.index('--wave-size')+1]=='32' and '--check-integer-reference' in c
    r=dict(p,variant='packed-specialized',frames=30,warmup=3,slots=1,metric_depth=False,
        required_wave_size=32,full_subgroups_required=True)
    compare.validate_profile(r,p,'packed-specialized',30,3,1,32)
    for key,value in [('variant','packed'),('full_subgroups_required',False),('required_wave_size',64),('p2',33)]:
        with pytest.raises(ValueError):compare.validate_profile(dict(r,**{key:value}),p,'packed-specialized',30,3,1,32)


def test_d128_opt_in_implies_previous_candidates_and_rejects_wrong_shape(tmp_path,monkeypatch):
    paths={'left':'L','right':'R'};p=dict(profile(),disparities=128)
    calls=compare.commands(Path('/rhi'),Path('/daxa'),tmp_path,p,paths,30,3,1,False,False,include_d128=True)
    assert set(calls)=={'rhi','baseline','wave','packed','packed-specialized','packed-d128'}
    assert calls['packed-d128'][calls['packed-d128'].index('--variant')+1]=='packed-d128'
    with pytest.raises(ValueError):compare.commands(Path('/rhi'),Path('/daxa'),tmp_path,profile(),paths,30,3,1,False,False,include_d128=True)
    monkeypatch.setattr(compare.subprocess,'run',lambda *a,**k:pytest.fail('wrong-shape must not execute children'))
    directory=tmp_path/'wrong-shape'
    monkeypatch.setattr(sys,'argv',['compare','--output',str(directory),'--compiler','/unused','--disparities','64','--include-d128'])
    with pytest.raises(SystemExit):compare.main()
    assert not directory.exists()
