"""CPU contracts for the explicit, bounded static GPU matrix driver."""
import copy
from pathlib import Path
import sys
import math

import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
import compare_multiresolution as matrix


def family():
    p1 = np.array([[500., 0., 600., 0.], [0., 510., 350., 0.], [0., 0., 1., 0.]])
    p2 = p1.copy(); p2[0, 2] += .75; p2[0, 3] = -60.
    result = []
    for name, width, height, d, scale in [('full',1280,720,128,1.),('medium',960,540,96,.75),('half',640,360,64,.5)]:
        a = np.array([[scale,0,(scale-1)/2],[0,scale,(scale-1)/2],[0,0,1.]])
        m = dict(rectified=dict(width=width,height=height),source={'eye_width':1280,'height':720},capture={'sha256':'a'*64,'record_ordinal':0},
                 calibration=dict(original_identity='original',original_json={'sha256':'b'*64},R1=np.eye(3).tolist(),R2=np.eye(3).tolist(),P1=(a@p1).tolist(),P2=(a@p2).tolist()),
                 expected={key:{'sha256':'c'*64} for key in ('original_left_rgb','original_right_rgb')},
                 provenance={'multiresolution_profile':dict(schema='stereoforge.multiresolution_profile.v1',donor_manifest={'sha256':'d'*64},donor_files_sha256={'manifest.json':'d'*64},donor_manifest_sha256_before='d'*64,donor_manifest_sha256_after='d'*64,donor_rectified={'width':1280,'height':720},target_rectified={'width':width,'height':height},scale={'numerator':int(scale*4),'denominator':4},pixel_center_convention=matrix.PIXEL_CENTER_CONVENTION,exporter_sha256='e'*64)})
        result.append(dict(name=name,manifest=m,metadata={'frame_id':5,'controls_json':'{}'},disparities=d))
    return result


def test_family_and_schedule():
    rows=matrix.validate_profile_family(family())
    assert [r['disparities'] for r in rows]==[128,96,64]
    assert matrix.interleaved_schedule(['A','B','C'])==[(0,'A'),(0,'B'),(0,'C'),(1,'B'),(1,'C'),(1,'A'),(2,'C'),(2,'A'),(2,'B')]


@pytest.mark.parametrize('mutate',[
    lambda r:r[1]['manifest']['capture'].update(sha256='f'*64),
    lambda r:r[1]['manifest']['calibration'].update(original_identity='other'),
    lambda r:r[1]['manifest']['calibration']['R1'][0].__setitem__(0,.9),
    lambda r:r[1]['manifest']['expected']['original_left_rgb'].update(sha256='f'*64),
    lambda r:r[1]['manifest']['provenance']['multiresolution_profile']['donor_manifest'].update(sha256='f'*64),
    lambda r:r[1]['manifest']['calibration']['P2'][0].__setitem__(3,-60.),
    lambda r:r[1]['manifest']['rectified'].update(width=961),
    lambda r:r[1].update(disparities=128),
    lambda r:r[1].update(paths=8),
    lambda r:r[1].update(stereo_schedule='paired-per-path'),
    lambda r:r[1].update(subpixel=False),
    lambda r:r[1]['metadata'].update(frame_id=6),
])
def test_changed_family_rejected(mutate):
    rows=family();mutate(rows)
    with pytest.raises(ValueError):matrix.validate_profile_family(rows)


@pytest.mark.parametrize('samples',[[],[True],[float('nan')],[float('inf')],[-1],None,'123'])
def test_bad_samples(samples):
    with pytest.raises(ValueError):matrix.summarize_samples(samples)


def test_nearest_rank_and_chronological_preservation():
    values=list(range(99,-1,-1)); before=values.copy()
    s=matrix.summarize_samples(values)
    assert s['p99']==98 and s['max']==99 and s['p50']==49 and s['mean']==49.5
    assert values==before
    assert matrix.summarize_samples([3])['p99']==3


def test_quality_denominators_and_unknown():
    predicted=np.array([[1.,2.,3.,9.,100.]])
    validity=np.array([[1,0,1,1,1]],dtype=np.uint32)
    truth=np.array([[1.,2.,2.,0.,100.]])
    support=np.array([[1,1,1,1,0]],dtype=np.uint32)
    declaration={'independent':True,'grid_shape':[1,5],'source':'measured range target'}
    q=matrix.evaluate_quality(predicted,validity,truth,np.array([[1,1,1,0,1]]),support,truth_declaration=declaration,error_threshold_m=.1)
    assert q['truth_valid_supported']==3 and q['compared']==2
    assert q['completeness']==2/3 and q['correct_completeness']==1/3
    assert q['false_accepts']==1 and q['unscored_accepts']==1 and q['mae_m']==.5
    assert matrix.evaluate_quality(predicted,validity)['status']=='unknown_no_independent_truth'
    with pytest.raises(ValueError):matrix.evaluate_quality(predicted,validity,truth,np.ones((1,5)),support,truth_declaration={'independent':False})


@pytest.mark.parametrize('kwargs',[
    {'error_threshold_m':float('nan')}, {'error_threshold_m':True},
    {'truth_declaration':{'independent':True,'grid_shape':[2,2],'source':'range'}},
])
def test_quality_bad_contract(kwargs):
    args=dict(truth_declaration={'independent':True,'grid_shape':[1,1],'source':'range'},error_threshold_m=.1);args.update(kwargs)
    with pytest.raises(ValueError):matrix.evaluate_quality([[1]],[[1]],[[1]],[[1]],[[1]],**args)


def driver_inputs(tmp_path, monkeypatch):
    """Actual freeze verifier and Depth report validator, only process mocked."""
    import json
    import struct
    import test_daxa_depth_comparison as reference
    rows=family()
    native=tmp_path/'native'; native.write_bytes(b'native')
    compiler=tmp_path/'compiler'; compiler.write_bytes(b'compiler')
    dependency=tmp_path/'dependency'; dependency.write_bytes(b'library/header/source/spv')
    profiles=tmp_path/'profiles.json'; profiles.write_text('[]')
    files={str(p.resolve()):matrix.sha256(p) for p in (native,compiler,dependency,profiles,Path(matrix.__file__))}
    for module in (matrix.depth,matrix.depth.preprocess,matrix.depth.base,matrix.depth.right,matrix.fixture_exporter):
        files[str(Path(module.__file__).resolve())]=matrix.sha256(module.__file__)
    identities={}
    device={k:'mock-'+k for k in matrix.DEVICE_KEYS}
    device['slangc_sha256']=matrix.sha256(compiler)
    for row in rows:
        row['manifest']['source']['stride_bytes']=5120
        row['manifest']['calibration']['rectified_identity']='rectified-'+row['name']
        row['manifest']['provenance']['multiresolution_profile']['nominal_near_depth_m_by_disparities']={str(row['disparities']):.5}
        row['metadata'].update(camera_timestamp_ns=100,host_ingress_timestamp_ns=0,camera_clock_domain='recorded',host_clock_domain='unknown')
        row['metric']=dict(fx=500.,baseline_m=.125,cx_offset=0.)
        row['metric'].update({k+'_bits':struct.unpack('<I',struct.pack('<f',v))[0] for k,v in list(row['metric'].items())})
        path=tmp_path/(row['name']+'.json'); path.write_text(json.dumps(row['manifest']))
        row['manifest_path']=path; row['inventory']={str(path):matrix.sha256(path)}; files.update(row['inventory'])
        ident=reference.identity()
        ident.update(binary_sha256=matrix.sha256(native),compiler_sha256=matrix.sha256(compiler),manifest_sha256=matrix.sha256(path),
                     offline_shader_directory=str(tmp_path),offline_spirv_sha256={'dependency':matrix.sha256(dependency)})
        for key in ('source_sha256','shader_source_sha256','dependency_sha256','build_configuration_sha256','compiler_input_sha256','optional_dependency_header_sha256'):
            ident[key]={str(dependency):matrix.sha256(dependency)}
        ident['fixture_file_sha256']=row['inventory'].copy()
        identities[row['name']]=ident
    freeze=tmp_path/'freeze.json'
    freeze.write_text(json.dumps(dict(schema='stereoforge.multiresolution_freeze.v1',native=dict(file=str(native),sha256=matrix.sha256(native)),
                     compiler=dict(file=str(compiler),sha256=matrix.sha256(compiler)),file_sha256=files,report_identity=identities,device_identity=device)))
    monkeypatch.setattr(matrix,'load_profiles',lambda _:rows)
    for key in ('VK_INSTANCE_LAYERS','VK_LAYER_VALIDATE_SYNC','VK_LAYER_ENABLES','VK_LOADER_DEBUG','SF_DAXA_VALIDATION'):
        monkeypatch.delenv(key,raising=False)
    def report(row,frames=100,warmup=20):
        m,meta,metric=row['manifest'],row['metadata'],row['metric']
        r=reference.native_report(m,meta,metric)
        r.update(width=m['rectified']['width'],height=m['rectified']['height'],disparities=row['disparities'],frames=frames,warmup=warmup,slots=2,
                 source_id=meta['frame_id'],source_metadata=meta,capture_sha256=m['capture']['sha256'],
                 original_identity=m['calibration']['original_identity'],rectified_identity=m['calibration']['rectified_identity'],
                 fixture_manifest_sha256=matrix.sha256(row['manifest_path']),completed_output_retirements=frames+warmup)
        r['device'].update(device)
        profile={k:r[k] for k in ('width','height','disparities','paths','p1','p2','subpixel')}
        r['requested_observation_buffer_bytes'],r['requested_combined_buffer_bytes']=matrix.depth.requested_bytes(m,profile,2)
        r['raw_samples_ms']={k:[float(i+1) for i in range(frames)] for k in matrix.depth.TIMINGS}
        for k,values in r['raw_samples_ms'].items():
            r[k]=dict(mean=sum(values)/frames,p50=values[frames//2],p95=values[min(frames-1,frames*95//100)])
        return r
    return rows,native,profiles,freeze,dependency,report


def test_nine_ordered_native_commands_and_reports(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    rows,native,profiles,freeze,_,report=driver_inputs(tmp_path,monkeypatch)
    calls=[]
    def child(command,**kwargs):
        calls.append(command)
        assert kwargs==dict(capture_output=True,text=True,errors='replace')
        def value(k):return command[command.index(k)+1]
        assert '--check-reference' in command and '--dump-prefix' not in command and '--validation' not in command
        assert all(word not in command for word in ('--derived','--view','--temporal'))
        for k,v in {'--frames':'100','--warmup':'20','--slots':'2','--wave-size':'32','--paths':'4','--p1':'8','--p2':'32','--stereo-schedule':'sequential','--lr-tolerance':'1','--right-reference-mode':'independent-sgm'}.items():assert value(k)==v
        row=next(r for r in rows if str(r['manifest_path'])==value('--manifest'))
        assert int(value('--disparities'))==row['disparities']
        Path(value('--output')).write_text(json.dumps(report(row)))
        return SimpleNamespace(returncode=0,stdout='done',stderr='')
    monkeypatch.setattr(matrix.subprocess,'run',child)
    result=matrix.run_comparison(profiles,native,freeze,matrix.sha256(freeze),tmp_path/'results')
    assert result['passed'], result
    assert [r['profile'] for r in result['runs']]==['full','medium','half','medium','half','full','half','full','medium']
    assert len(calls)==9 and result['inputs_unchanged_after']
    assert result['runs'][0]['statistics_ms']['host_wall_ms']['p99']==99
    assert result['runs'][0]['candidate_hypotheses']==2*1280*720*128
    assert result['quality']['status']=='unknown_no_independent_truth'
    assert result['unique_throughput']=='unmeasured_static_fixture'
    with pytest.raises(ValueError):matrix.run_comparison(profiles,native,freeze,matrix.sha256(freeze),tmp_path/'results')


@pytest.mark.parametrize('failure',['exit','report','input','old_output'])
def test_child_failures_retain_artifacts_and_recheck_inputs(tmp_path,monkeypatch,failure):
    import json
    from types import SimpleNamespace
    rows,native,profiles,freeze,dependency,report=driver_inputs(tmp_path,monkeypatch)
    calls=[]
    def child(command,**kwargs):
        calls.append(command)
        if failure=='input':dependency.write_bytes(b'mutated')
        output=Path(command[command.index('--output')+1])
        row=next(r for r in rows if str(r['manifest_path'])==command[command.index('--manifest')+1])
        r=report(row)
        if failure=='report':r['raw_samples_ms']['gpu_left_stereo_ms']=None
        output.write_text(json.dumps(r))
        if failure=='old_output' and len(calls)==2:
            (tmp_path/'results/r1-full/native.json').write_bytes(b'mutation')
        return SimpleNamespace(returncode=1 if failure=='exit' else 0,stdout='partial evidence',stderr='failure' if failure=='exit' else '')
    monkeypatch.setattr(matrix.subprocess,'run',child)
    result=matrix.run_comparison(profiles,native,freeze,matrix.sha256(freeze),tmp_path/'results')
    assert not result['passed'] and len(calls)==(2 if failure=='old_output' else 1)
    assert (tmp_path/'results/r1-full/stdout').read_text()=='partial evidence'
    assert (tmp_path/'results/comparison.json').exists()
    if failure in ('input','old_output'):assert result['inputs_unchanged_after'] is False


def test_freeze_hash_and_missing_bound_dependency_rejected_before_dispatch(tmp_path,monkeypatch):
    import json
    _,native,profiles,freeze,dependency,_=driver_inputs(tmp_path,monkeypatch)
    monkeypatch.setattr(matrix.subprocess,'run',lambda *_a,**_k:pytest.fail('must not dispatch'))
    with pytest.raises(ValueError,match='freeze SHA256'):
        matrix.run_comparison(profiles,native,freeze,'0'*64,tmp_path/'bad')
    f=json.loads(freeze.read_text()); del f['file_sha256'][str(dependency)]
    freeze.write_text(json.dumps(f))
    with pytest.raises(ValueError,match='absent from reviewed freeze'):
        matrix.run_comparison(profiles,native,freeze,matrix.sha256(freeze),tmp_path/'bad')


@pytest.mark.parametrize('mutation',['summary','missing','length','nonfinite','boolean'])
def test_raw_samples_reject_malformed_and_preserve_old_policy(tmp_path,monkeypatch,mutation):
    rows,_,_,_,_,report=driver_inputs(tmp_path,monkeypatch)
    r=report(rows[0]);key=matrix.depth.TIMINGS[0]
    if mutation=='summary':r[key]['p50']=50 # old policy uses index50 ->51, not nearest rank50
    if mutation=='missing':del r['raw_samples_ms'][key]
    if mutation=='length':r['raw_samples_ms'][key].pop()
    if mutation=='nonfinite':r['raw_samples_ms'][key][0]=math.nan
    if mutation=='boolean':r['raw_samples_ms'][key][0]=True
    with pytest.raises(ValueError):matrix._validate_samples(r,100)


def test_no_implicit_dispatch_on_cli_missing_subcommand(tmp_path):
    import subprocess
    result=subprocess.run([sys.executable,matrix.__file__],capture_output=True,text=True)
    assert result.returncode==2 and 'command' in result.stderr


@pytest.mark.parametrize('repeats',[0,1,2,4,10,True,3.0])
def test_unbalanced_or_unbounded_repetitions_rejected(repeats):
    with pytest.raises(ValueError):matrix.interleaved_schedule(['A','B','C'],repeats)


@pytest.mark.parametrize('values',[([[float('nan')]],[[1]]), ([[1]],[[2]]), ([[0]],[[1]]), ([[1,2]],[[1,1]])])
def test_quality_rejects_nonphysical_valid_depth_and_wrong_shape(values):
    p,v=values
    with pytest.raises(ValueError):matrix.evaluate_quality(p,v,[[1]],[[1]],[[1]],truth_declaration={'independent':True,'grid_shape':[1,1],'source':'physical fixture'})


def test_quality_empty_truth_and_large_finite_errors():
    declaration={'independent':True,'grid_shape':[1,2],'source':'physical fixture'}
    empty=matrix.evaluate_quality([[1,1]],[[1,1]],[[0,0]],[[0,0]],[[1,1]],truth_declaration=declaration)
    assert empty['completeness'] is None and empty['mae_m'] is None and empty['unscored_accepts']==2
    huge=matrix.evaluate_quality([[1.e300,1.e300]],[[1,1]],[[1,1]],[[1,1]],[[1,1]],truth_declaration=declaration)
    assert math.isfinite(huge['rmse_m']) and huge['rmse_m']==1.e300


def test_legacy_summary_matches_actual_cpp_sorted_accumulation(tmp_path,monkeypatch):
    import json
    import shutil
    import subprocess
    rows,_,_,_,_,report=driver_inputs(tmp_path,monkeypatch)
    r=report(rows[0])
    # These positive fractional samples distinguish sorted accumulation, raw
    # chronology accumulation, and Python3.12 compensated sum even after /N.
    values=[8.]*15+[2.]*22+[1.]*8+[.3]*10+[.2]*13+[.1]*15+[.01]*17
    source=(matrix.ROOT/'apps/daxa_depth_benchmark.cpp').read_text()
    summary=source[source.index('json summary('):source.index('// Post-timing only:')]
    cpp=tmp_path/'summary.cpp'
    cpp.write_text('#include <json.hpp>\n#include <vector>\n#include <algorithm>\n#include <numeric>\n#include <iostream>\nusing json=nlohmann::json;\n'+summary+
                   '\nint main(){std::vector<double> v={'+','.join(repr(v) for v in values)+'}; std::cout<<summary(v);}')
    compiler=shutil.which('c++'); assert compiler
    binary=tmp_path/'summary'
    subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror','-I',str(matrix.ROOT/'build/daxa/_deps/sf_native_json-src'),str(cpp),'-o',str(binary)],check=True,capture_output=True,text=True)
    actual=json.loads(subprocess.check_output([str(binary)],text=True))
    chronological=0.
    for value in values:chronological+=value
    assert actual['mean'] != chronological/100
    assert actual['mean'] != sum(values)/100
    assert actual['mean'] != math.fsum(values)/100
    for key in matrix.depth.TIMINGS:
        r['raw_samples_ms'][key]=values.copy()
        r[key]=actual.copy()
    before=copy.deepcopy(r['raw_samples_ms'])
    assert matrix._validate_samples(r,100)[matrix.depth.TIMINGS[0]]['mean']==math.fsum(values)/100
    assert r['raw_samples_ms']==before
    for wrong in (chronological/100,sum(values)/100,math.fsum(values)/100):
        r[matrix.depth.TIMINGS[0]]['mean']=wrong
        with pytest.raises(ValueError,match='historical summary'):matrix._validate_samples(r,100)


def test_documented_profiles_list_loads_real_exported_fixture_family(tmp_path):
    import json
    import test_daxa_preprocess_comparison as source_fixture
    capture,calibration=source_fixture.record_file(tmp_path,records=1,width=1280,height=720)
    donor=tmp_path/'donor'
    matrix.depth.preprocess.export_fixture(capture,0,calibration,donor)
    entries=[]
    for name,(width,height,d) in zip(('full','medium','half'),matrix.SHAPES):
        path=matrix.fixture_exporter.export_profile(donor/'manifest.json',tmp_path/name,width,height)
        entries.append(dict(name=name,manifest=str(path.relative_to(tmp_path)),disparities=d))
    path=tmp_path/'profiles.json';path.write_text(json.dumps(entries))
    rows=matrix.load_profiles(path)
    assert [r['disparities'] for r in rows]==[128,96,64]
    assert rows[1]['manifest']['rectified']==dict(width=960,height=540)
    assert all(r['metadata']==rows[0]['metadata'] for r in rows)
    assert all(str(r['manifest_path']) in r['inventory'] for r in rows)
    path.write_text(json.dumps(entries)+',"injected":1')
    with pytest.raises(ValueError,match='trailing members'):matrix.load_profiles(path)


@pytest.mark.parametrize('text',['{}','[{"name":"a","name":"b"}]','[NaN]','[Infinity]','[1e999]'])
def test_profiles_json_stays_strict(tmp_path,text):
    path=tmp_path/'profiles.json';path.write_text(text)
    with pytest.raises(ValueError):matrix.load_profiles(path)


def test_logical_trusted_dependency_symlink_binds_canonical_frozen_bytes(tmp_path,monkeypatch):
    import json
    rows,native,profiles,freeze,dependency,_=driver_inputs(tmp_path,monkeypatch)
    alias=tmp_path/'logical-library.so';alias.symlink_to(dependency)
    f=json.loads(freeze.read_text())
    f['runtime_alias_targets']={str(alias):str(dependency)}
    for ident in f['report_identity'].values():
        ident['dependency_sha256']={str(alias):matrix.sha256(dependency)}
    freeze.write_text(json.dumps(f))
    assert matrix.verify_freeze(freeze,matrix.sha256(freeze),native,profiles,rows)==f
    # Changing only target identity must fail even when bytes remain identical.
    other=tmp_path/'same-bytes-other-target';other.write_bytes(dependency.read_bytes())
    alias.unlink();alias.symlink_to(other)
    with pytest.raises(ValueError,match='alias target changed'):
        matrix.verify_freeze(freeze,matrix.sha256(freeze),native,profiles,rows)


def test_fixture_symlink_still_rejected(tmp_path):
    import json
    target=tmp_path/'manifest.json';target.write_text('{}')
    alias=tmp_path/'manifest-alias.json';alias.symlink_to(target)
    profiles=tmp_path/'profiles.json';profiles.write_text(json.dumps([dict(name='full',manifest=alias.name,disparities=128)]))
    with pytest.raises(ValueError,match='symlink'):matrix.load_profiles(profiles)
