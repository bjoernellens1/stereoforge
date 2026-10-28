"""Cold conversion of bound two-eye native dumps into portable observations."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import struct

import numpy as np
import pytest
from benchmarks.zed_comparison import schema
from benchmarks.zed_comparison.native_observation import export_native_observation
from test_zed_native_input import native_record
from benchmarks.zed_comparison.native_input import _canonical, SAMPLING_PROFILE


@pytest.fixture
def executed(native_record,tmp_path):
    record,path,_,_=native_record
    dump=tmp_path/'dumps';dump.mkdir()
    def plane(name,data):
        p=dump/name;p.write_bytes(data)
        return {'path':name,'sha256':sha256(data).hexdigest(),'elements':32}
    depth=struct.pack('<32I',0x7fc01234,*([0x40000000]*15),*([0x40400000]*16))
    valid=[0]+[1]*31
    files={
        'depth.f32':plane('depth.f32',depth),
        'metric_validity.u32':plane('metric_validity.u32',struct.pack('<32I',*valid)),
        'metric_confidence.f32':plane('metric_confidence.f32',struct.pack('<32f',*([.5]*32))),
        'provenance.u32':plane('provenance.u32',struct.pack('<32I',*valid)),
        'source_ids.i64':plane('source_ids.i64',struct.pack('<32q',-1,*([record['source_id']]*31))),
        'age_ns.i64':plane('age_ns.i64',bytes(32*8))}
    def pixel(names):return sha256(_canonical({name:{k:v for k,v in record['images'][name].items() if k!='path'} for name in names})).hexdigest()
    report=dict(schema='stereoforge.daxa_depth_rectified_benchmark.v1',variant='full_sgm',frames=3,warmup=1,slots=2,radius=4,
                subpixel=True,sgm_subpixel=True,sgm_disparities=128,sgm_directions=4,right_reference_mode='independent_sgm',
                lr_dependency='independent_references',final_estimator='independent_sgm',sampling_profile=SAMPLING_PROFILE,
                scope='static_repeated_canonical_record; no_live_quality_or_pipeline_FPS_claim',
                record_sha256=sha256(path.read_bytes()).hexdigest(),record_ordinal=record['ordinal'],source_id=record['source_id'],
                camera_timestamp_ns=100,camera_clock_domain='camera',host_ingress_timestamp_ns=0,host_clock_domain='unknown',
                input_identity=record['input_identity'],original_pixel_identity=pixel(('original_left','original_right')),
                rectified_pixel_identity=pixel(('rectified_left','rectified_right','support_left','support_right')),
                original_calibration_identity=record['calibration']['raw']['sha256'],rectified_calibration_identity=record['calibration']['rectified']['sha256'],
                width=4,height=4,files=files,device={'device':'CPU test fixture; native layout only'},
                raw_samples_ms={'host_wall_ms':[1,1,1]},validation_errors_after_teardown=0,validation_warnings_after_teardown=0,
                shader_compiler={'version':'fixture-compiler','compiler_sha256':'1'*64,'floating_point_mode':'default'},
                producer_shader_identity='2'*64,depth_shader_identity='3'*64)
    report_path=tmp_path/'report.json';report_path.write_text(json.dumps(report))
    return record,path,report,report_path,dump


def export(executed,tmp_path):
    _,source,_,report,dump=executed
    return export_native_observation(report,source,dump,tmp_path/'output')


def test_exact_left_bytes_complete_bundle(executed,tmp_path):
    source,source_path,report,report_path,dump=executed
    path=export(executed,tmp_path);record=schema.load_records([path])[0]
    assert record['input_identity']==source['input_identity']==schema.input_identity(record)
    for key in source.keys()-{'observation','implementation'}:assert record[key]==source[key]
    for desc in schema_descriptors(source):assert (path.parent/desc['path']).read_bytes()==(source_path.parent/desc['path']).read_bytes()
    obs=record['observation']['value']
    assert (path.parent/obs['depth']['path']).read_bytes()==(dump/'depth.f32').read_bytes()[:16*4]
    assert (path.parent/obs['validity']['path']).read_bytes()==bytes([0]+[1]*15)
    assert (path.parent/obs['provenance']['value']['path']).read_bytes()==bytes([0]+[1]*15)
    assert obs['confidence_direction']=='high_is_good'
    impl=record['implementation'];assert impl['name']=='StereoForge' and impl['version']=='canonical_native_v1'
    assert impl['commit']['status']=='unavailable'
    assert impl['settings']['search_bounds']==dict(min_disparity=0,max_disparity_exclusive=128)
    assert impl['settings']['sgm_subpixel'] is True
    assert any((path.parent/desc['path']).read_bytes()==report_path.read_bytes() for desc in impl['artifacts'])


def schema_descriptors(value):
    from benchmarks.zed_comparison.native_input import _descriptors
    return list(_descriptors(value))


def test_coarse_executed_settings(executed,tmp_path):
    _,_,r,p,_=executed
    r.update(variant='coarse_refine',subpixel=False,refinement_subpixel=False,sgm_disparities=64,refinement_disparities=128,
             final_estimator='independent-local-census',final_right_reference_mode='independent-local-census',coarse_width=2,coarse_height=2,
             prior_confidence_floor=.05,max_coarse_span=2.,min_margin=1,lr_tolerance=1.,recovery=False)
    p.write_text(json.dumps(r));record=schema.load_records([export(executed,tmp_path)])[0]
    settings=record['implementation']['settings']
    assert settings['search_bounds']['max_disparity_exclusive']==128
    assert settings['subpixel'] is False and settings['sgm_subpixel'] is True


@pytest.mark.parametrize('case',['schema','variant','frames','frame_count','subpixel_missing','subpixel_bool','source_id','ordinal','capture','clock','input','pixel','raw_cal','rect_cal','dimensions','profile','record_hash','missing_dump','extent','hash','mask_right','provenance','source_pixels','age','right_source','right_age','right_depth','right_confidence','provenance_overflow','symlink','traversal','outside','duplicate','collision','output_exists','mutation'])
def test_reject_invalid(executed,tmp_path,monkeypatch,case):
    source,source_path,r,p,dump=executed
    if case=='schema':r['schema']='foreign'
    elif case=='variant':r['variant']='sdk'
    elif case=='frames':r['frames']=True
    elif case=='frame_count':r['raw_samples_ms']['host_wall_ms'].pop()
    elif case=='subpixel_missing':del r['sgm_subpixel']
    elif case=='subpixel_bool':r['subpixel']=1
    elif case in ('source_id','ordinal','capture','clock','input','pixel','raw_cal','rect_cal','dimensions','profile','record_hash'):
        key={'source_id':'source_id','ordinal':'record_ordinal','capture':'camera_timestamp_ns','clock':'camera_clock_domain','input':'input_identity','pixel':'rectified_pixel_identity','raw_cal':'original_calibration_identity','rect_cal':'rectified_calibration_identity','dimensions':'width','profile':'sampling_profile','record_hash':'record_sha256'}[case]
        r[key]='wrong' if isinstance(r[key],str) else r[key]+1
    elif case=='missing_dump':del r['files']['depth.f32']
    elif case=='extent':r['files']['depth.f32']['elements']=16
    elif case=='hash':(dump/'depth.f32').write_bytes(bytes(128))
    elif case in ('mask_right','provenance','source_pixels','age','right_source','right_age','right_depth','right_confidence','provenance_overflow'):
        key,index,fmt,value={'mask_right':('metric_validity.u32',31,'<I',2),'provenance':('provenance.u32',1,'<I',2),'source_pixels':('source_ids.i64',1,'<q',0),'age':('age_ns.i64',1,'<q',1),'right_source':('source_ids.i64',31,'<q',0),'right_age':('age_ns.i64',31,'<q',1),'right_depth':('depth.f32',31,'<f',float('nan')),'right_confidence':('metric_confidence.f32',31,'<f',float('inf')),'provenance_overflow':('provenance.u32',0,'<I',257)}[case]
        data=bytearray((dump/key).read_bytes());size=struct.calcsize(fmt);data[index*size:(index+1)*size]=struct.pack(fmt,value);(dump/key).write_bytes(data);r['files'][key]['sha256']=sha256(data).hexdigest()
    elif case=='symlink':
        original=dump/'depth.f32';original.rename(dump/'actual');original.symlink_to(dump/'actual')
    elif case=='traversal':r['files']['depth.f32']['path']='../dumps/depth.f32'
    elif case=='outside':r['files']['depth.f32']['path']=str(source_path)
    elif case=='collision':
        d=source['implementation']['artifacts'][0];old=source_path.parent/d['path'];d['path']='native-observation/collision';new=source_path.parent/d['path'];new.parent.mkdir();new.write_bytes(old.read_bytes());source_path.write_text(json.dumps(source));r['record_sha256']=sha256(source_path.read_bytes()).hexdigest()
    elif case=='output_exists':(tmp_path/'output').mkdir()
    elif case=='mutation':
        import benchmarks.zed_comparison.native_observation as module
        original=module.shutil.copyfile
        def mutate(a,b,*args,**kwargs):
            result=original(a,b,*args,**kwargs);source_path.write_bytes(source_path.read_bytes()+b' ');return result
        monkeypatch.setattr(module.shutil,'copyfile',mutate)
    p.write_text(json.dumps(r))
    if case=='duplicate':p.write_text('{"schema":1,"schema":2}')
    with pytest.raises((ValueError,FileExistsError)):export(executed,tmp_path)
    if case!='output_exists':assert not (tmp_path/'output').exists()


def test_absolute_dump_paths_and_available_commit(executed,tmp_path):
    _,_,r,p,dump=executed
    for desc in r['files'].values():desc['path']=str(dump/desc['path'])
    r['commit']='a'*40;p.write_text(json.dumps(r))
    record=schema.load_records([export(executed,tmp_path)])[0]
    assert record['implementation']['commit']==dict(status='available',value='a'*40)


def test_invalid_confidence_nan_payload_preserved(executed,tmp_path):
    _,_,r,p,dump=executed
    desc=r['files']['metric_confidence.f32'];path=dump/desc['path']
    data=bytearray(path.read_bytes());data[:4]=struct.pack('<I',0x7fc12345);path.write_bytes(data)
    desc['sha256']=sha256(data).hexdigest();p.write_text(json.dumps(r))
    output=export(executed,tmp_path);record=schema.load_records([output])[0]
    confidence=record['observation']['value']['confidence']['value']
    assert (output.parent/confidence['path']).read_bytes()==data[:64]


@pytest.mark.parametrize('case',['plane_mutation','report_mutation','source_dep_mutation'])
def test_materialization_snapshot_rechecks(executed,tmp_path,monkeypatch,case):
    import benchmarks.zed_comparison.native_observation as module
    record,source,report,report_path,dump=executed
    original=module.shutil.copyfile
    mutated=False
    def mutate(a,b,*args,**kwargs):
        nonlocal mutated
        result=original(a,b,*args,**kwargs)
        if not mutated:
            path={'plane_mutation':dump/'depth.f32','report_mutation':report_path,'source_dep_mutation':source.parent/record['images']['original_left']['path']}[case]
            path.write_bytes(path.read_bytes()+b' ');mutated=True
        return result
    monkeypatch.setattr(module.shutil,'copyfile',mutate)
    with pytest.raises(ValueError):export(executed,tmp_path)
    assert not (tmp_path/'output').exists()
