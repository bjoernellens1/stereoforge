#!/usr/bin/env python3
"""Exact nine-plane combined depth gate against frozen RHI pixel arithmetic.

Runs only when explicitly invoked. Repeated recorded fixture, no fresh-camera,
physical calibration, motion, RGB-D synthesis or sustained60Hz qualification.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
import compare_daxa_preprocess as preprocess
import compare_daxa_stereo as base
import compare_daxa_stereo_right as right

SCOPE='repeated_retained_record_dual_reference_LR_metric_no_RGBD_synthesis'
SUPPORT_POLICY='source_center_and_nearest_correspondence_remap_support; not_Census_tap_certification; LR_mask_unchanged'
GPU_SCOPE='same_queue_upload_start_through_metric_end_including_intergraph_gaps'
MEMORY_SCOPE='buffer_bytes_only; excludes_allocator_driver_queries_diagnostics_CPU_source'
ADMISSION_POLICY='one_explicit_application_snapshot; SDK_GC_internal_scans_additional_included_host_wall'
PLANES=('lr_disparity.f32','lr_confidence.f32','lr_validity.u32','depth.f32','metric_confidence.f32','metric_validity.u32','provenance.u32','source_ids.i64','age_ns.i64')
RAW_PLANES=base.OUTPUT_PLANES
RHI_PLANES=RAW_PLANES+('depth.f32','depth_validity.u32')
TIMINGS=('enqueue_cpu_ms','wait_cpu_ms','timestamp_query_cpu_ms','host_wall_ms','upload_cpu_ms','gpu_upload_ms','gpu_decode_ms','gpu_rectify_ms','gpu_left_stereo_ms','gpu_right_stereo_ms','gpu_lr_ms','gpu_metric_metadata_ms','gpu_combined_queue_span_ms')
STEREO_SCOPES={
    'sequential':'sequential_per_reference_graphs',
    'paired-per-path':'paired_per_path_phase_intervals_include_both_references; no_isolated_per_reference_times',
}
PAIRED_PHASES=('census_left','census_right','zero_packed_penalties','aggregate_pos_x','aggregate_neg_x','aggregate_pos_y','aggregate_neg_y')
PAIRED_DIAGONALS=('aggregate_pos_x_pos_y','aggregate_neg_x_neg_y','aggregate_pos_x_neg_y','aggregate_neg_x_pos_y')
mirror_rows=right.mirror_rows


def f32_bits(value):return struct.unpack('<I',struct.pack('<f',value))[0]


def array(raw,dtype,count):
    if len(raw)!=count*np.dtype(dtype).itemsize:raise ValueError('plane byte extent mismatch')
    return np.frombuffer(raw,dtype)


def expected_observations(raw,supports,width,height,source_id,lr_tolerance=1.):
    """Independent double-coordinate indexing; copies actual frozen GPU depth.

    No CPU depth division, scalar compound-float tolerance or native helper use.
    LR subtraction retains binary32 semantics of the existing Python pipeline.
    """
    if any(type(v) is not int for v in (width,height,source_id)) or min(width,height)<=0 or not 0<=source_id<1<<63 or type(lr_tolerance) not in (int,float) or not math.isfinite(lr_tolerance) or lr_tolerance<0:raise ValueError('invalid oracle dimensions/source/tolerance')
    if len(raw)!=2 or len(supports)!=2:raise ValueError('two reference layers required')
    n=width*height;inputs=[];support=[]
    for eye in range(2):
        if set(raw[eye])!=set(RHI_PLANES):raise ValueError('missing frozen RHI output planes')
        row={p:array(raw[eye][p],'<u4' if p.endswith('.u32') else '<f4',n) for p in RHI_PLANES}
        for p in ('validity.u32','depth_validity.u32'):
            if np.any(row[p]>1):raise ValueError('nonbinary frozen validity')
        if not np.isfinite(row['confidence.f32']).all() or np.any((row['confidence.f32']<0)|(row['confidence.f32']>1)):raise ValueError('invalid frozen confidence')
        mask=array(supports[eye],'<u4',n)
        if np.any(mask>1):raise ValueError('nonbinary preprocessing support')
        inputs.append(row);support.append(mask)
    result={p:[] for p in PLANES};x=np.tile(np.arange(width,dtype=np.float64),height);rows=np.repeat(np.arange(height,dtype=np.int64)*width,width)
    nan=np.array([0x7fc00000],dtype='<u4').view('<f4')[0]
    for eye in range(2):
        own,other=inputs[eye],inputs[1-eye];d=own['disparity.f32'];finite=np.isfinite(d)
        # Safe coordinates before integer conversion; never clip an unsupported
        # coordinate into an apparently valid correspondence.
        continuous=x+(d.astype(np.float64) if eye else -d.astype(np.float64))+.5
        rounded=np.floor(np.where(finite,continuous,-1.));inside=finite&(rounded>=0)&(rounded<width)
        index=rows+np.where(inside,rounded,0).astype(np.int64)
        with np.errstate(invalid='ignore',over='ignore'):
            difference=np.abs(d-other['disparity.f32'][index])
        lr=(own['validity.u32']!=0)&inside&(other['validity.u32'][index]!=0)&np.isfinite(other['disparity.f32'][index])&(difference<=np.float32(lr_tolerance))
        z=own['depth.f32'];metric=lr&(support[eye]!=0)&(support[1-eye][index]!=0)&(own['depth_validity.u32']!=0)&np.isfinite(z)&(z>0)
        values={'lr_disparity.f32':np.where(lr,d,nan).astype('<f4'),'lr_confidence.f32':np.where(lr,own['confidence.f32'],0).astype('<f4'),'lr_validity.u32':lr.astype('<u4'),
            'depth.f32':np.where(metric,z,nan).astype('<f4'),'metric_confidence.f32':np.where(metric,own['confidence.f32'],0).astype('<f4'),'metric_validity.u32':metric.astype('<u4'),
            'provenance.u32':metric.astype('<u4'),'source_ids.i64':np.where(metric,source_id,-1).astype('<i8'),'age_ns.i64':np.zeros(n,'<i8')}
        for p in PLANES:result[p].append(values[p].tobytes())
    return {p:b''.join(layers) for p,layers in result.items()}


def fixture_planes(path):
    import cv2
    manifest,metadata=preprocess.validate_fixture(path);root=Path(path).parent
    if cv2.__version__!='4.14.0':raise ValueError('frozen CPU preprocessing requires cv24.14.0')
    cv_binary=next(Path(cv2.__file__).parent.glob('*.so'));cv_identity=dict(version=cv2.__version__,binary_sha256=base.sha256(cv_binary))
    if cv_identity['binary_sha256']!=manifest['provenance']['cv2_binary_sha256']:raise ValueError('actual CPU OpenCV binary differs from fixture profile')
    _,payload=preprocess.read_capture_record(preprocess.fixture_file(root,manifest['capture']),manifest['capture']['record_ordinal'])
    source=manifest['source'];w,h=source['eye_width'],source['height'];rw,rh=manifest['rectified']['width'],manifest['rectified']['height']
    rgb=cv2.cvtColor(np.frombuffer(payload,np.uint8).reshape(h,w*2,2),cv2.COLOR_YUV2RGB_YUY2);luma=[];supports=[]
    for eye,index in (('left',0),('right',1)):
        original=rgb[:,index*w:(index+1)*w].copy();mx=np.frombuffer(preprocess.fixture_file(root,manifest['maps'][eye+'_x']).read_bytes(),'<f4').reshape(rh,rw);my=np.frombuffer(preprocess.fixture_file(root,manifest['maps'][eye+'_y']).read_bytes(),'<f4').reshape(rh,rw)
        rect=cv2.remap(original,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);gray=cv2.cvtColor(rect,cv2.COLOR_RGB2GRAY).astype('<u4');support=preprocess.support_mask(mx,my,w,h)
        for name,data in (('original_'+eye+'_rgb',original.tobytes()),('rect_'+eye+'_rgb',rect.tobytes()),(eye+'_luma',gray.tobytes()),(eye+'_support',support.tobytes())):
            if data!=preprocess.fixture_file(root,manifest['expected'][name]).read_bytes():raise ValueError('actual CPU OpenCV preprocessing differs from expected fixture: '+name)
        luma.append(gray.tobytes());supports.append(support.tobytes())
    cal=base.strict_json(preprocess.fixture_file(root,manifest['calibration']['rectified_json']).read_text());t=cal['right_to_left'];metric=dict(fx=cal['left']['fx'],baseline_m=math.hypot(t[0][3],t[1][3],t[2][3]),cx_offset=cal['left']['cx']-cal['right']['cx'])
    metric={k:struct.unpack('<f',struct.pack('<f',v))[0] for k,v in metric.items()}
    if not all(math.isfinite(v) for v in metric.values()) or metric['fx']<=0 or metric['baseline_m']<=0:raise ValueError('invalid fixture metric float conversion')
    metric.update({k+'_bits':f32_bits(v) for k,v in list(metric.items())})
    return manifest,metadata,tuple(luma),tuple(supports),metric,cv_identity


def requested_bytes(manifest,profile,slots):
    n=profile['width']*profile['height'];source=manifest['source'];canvas=source['eye_width']*source['height'];raw=(source['stride_bytes']*source['height']+3)&~3
    bits=8 if profile['paths']*profile['p2']<=255 else 16;packed=n*((profile['disparities']+32//bits-1)//(32//bits))*4
    observations=slots*(88*n+168);pre=32*n+slots*(24*n+8*canvas+2*raw+128);stereo=slots*(28*n+packed+184)
    return observations,pre+2*stereo+observations


def compiler_header_identities(directory,targets=None):
    headers={}
    for target,source in targets or (('daxa_depth_benchmark','apps/daxa_depth_benchmark.cpp'),('stereoforge_daxa_depth','src/daxa_depth.cpp'),('stereoforge_daxa_calibration','src/daxa_calibration.cpp'),('stereoforge_daxa_preprocess','src/daxa_preprocess.cpp'),('stereoforge_daxa_stereo','src/daxa_stereo.cpp'),('stereoforge_daxa_frame','src/daxa_frame.cpp'),('stereoforge_daxa_foundation','src/daxa_foundation.cpp')):
        dependency=subprocess.run(['ninja','-C',str(directory),'-t','deps','CMakeFiles/'+target+'.dir/'+source+'.o'],capture_output=True,text=True,errors='replace')
        base.require_child_clean(dependency.returncode,dependency.stdout,dependency.stderr)
        paths=[Path(line.strip()) for line in dependency.stdout.splitlines() if line.startswith('    ')]
        if not paths:raise ValueError('actual compiler header dependencies absent: '+target)
        for path in paths:
            path=path if path.is_absolute() else Path(directory)/path
            headers[str(path.resolve(strict=True))]=base.sha256(path)
    return headers


def native_identities(rhi,native,compiler,manifest):
    # Reuse established executable/dependency/fixture binding, extending every
    # actual new source and SPIR-V rather than substituting a nominal directory.
    common=preprocess.native_identities(native,compiler,manifest,True);stereo=right.native_identities(rhi,native,compiler)
    directory=Path(common['offline_shader_directory'])
    if directory!=Path(stereo['daxa']['offline_shader_directory']):raise ValueError('offline shader directory getters disagree')
    common['offline_spirv_sha256'].update({'daxa-stereo-'+k+'.spv':v for k,v in stereo['daxa']['offline_spirv_sha256'].items()})
    for entry in ('lr_consistency','metric_observation'):common['offline_spirv_sha256']['daxa-depth-'+entry+'.spv']=base.sha256(directory/('daxa-depth-'+entry+'.spv'))
    common['source_sha256'].update(stereo['source_sha256'])
    for name in ('include/stereoforge/daxa_calibration.hpp','src/daxa_calibration.cpp','include/stereoforge/daxa_depth.hpp','src/daxa_depth.cpp','apps/daxa_depth_benchmark.cpp','cmake/DaxaDepth.cmake',
                 'benchmarks/compare_daxa_depth.py','benchmarks/compare_daxa_preprocess.py','benchmarks/compare_daxa_stereo.py','benchmarks/compare_daxa_stereo_right.py','benchmarks/compare_daxa_view.py','benchmarks/compare_native_resident.py'):
        common['source_sha256'][name]=base.sha256(base.ROOT/name)
    pairs={}
    for stage,wrapper,ops in (('depth','depth','stereo_observation'),('preprocess','preprocess','preprocess'),('stereo','stereo','sgm')):
        paths=[base.ROOT/('python/stereoforge/shaders/daxa/'+wrapper+'.slang'),base.ROOT/('python/stereoforge/shaders/ops/'+ops+'.slang')]
        pairs[stage]=':'.join(base.sha256(p) for p in paths)
        common['source_sha256'].update({str(p.relative_to(base.ROOT)):base.sha256(p) for p in paths})
    rhi_directory=Path(rhi).resolve().parent
    common['compiler_input_sha256']={'daxa':compiler_header_identities(directory),'rhi':compiler_header_identities(rhi_directory,(('native_stereo','apps/native_stereo.cpp'),('stereoforge_runtime','src/runtime.cpp'),('stereoforge_runtime','src/native_device.cpp')))}
    common['build_configuration_sha256']={str(path):base.sha256(path) for folder in (directory,rhi_directory) for path in (folder/'CMakeCache.txt',folder/'build.ninja')}
    common['shader_source_sha256_pairs']=pairs;common['rhi']=stereo['rhi'];common['capture_sha256']=preprocess.validate_fixture(manifest)[0]['capture']['sha256']
    return common


def validate_report(report,manifest,metadata,metric,identity,profile,frames,warmup,slots,wave,validation,stereo_schedule="sequential"):
    if type(stereo_schedule) is not str or stereo_schedule not in STEREO_SCOPES:raise ValueError("unknown stereo schedule")
    expected=dict(profile,schema='stereoforge.daxa_depth.v1',scope=SCOPE,frames=frames,warmup=warmup,slots=slots,required_wave_size=wave,metric_support_policy=SUPPORT_POLICY,layers='left_then_right',depth_units='axial_Z_metres_rectified_rays',
        stereo_schedule=stereo_schedule,stereo_timing_scope=STEREO_SCOPES[stereo_schedule],per_reference_stereo_timings_available=stereo_schedule=='sequential',
        source_metadata=metadata,source_id=metadata['frame_id'],original_identity=manifest['calibration']['original_identity'],rectified_identity=manifest['calibration']['rectified_identity'],fixture_manifest_sha256=identity['manifest_sha256'],capture_sha256=manifest['capture']['sha256'],record_ordinal=manifest['capture']['record_ordinal'],
        gpu_combined_scope=GPU_SCOPE,requested_memory_scope=MEMORY_SCOPE,admission_completion_policy=ADMISSION_POLICY,lr_mask_metadata_reference_checked=True,both_stereo_integer_reference_checked=True,both_stereo_packed_penalties_reference_checked=True,expected_preprocess_planes_checked=True,metric_compound_float_reference_checked=False,stereo_compound_float_reference_checked=False,completed_output_retirements=frames+warmup,capture_drops='not_measured; repeated_fixture',camera_age='unknown; no_clock_mapping',validation_requested=validation)
    expected.update({k:metadata[k] for k in ('camera_timestamp_ns','host_ingress_timestamp_ns','camera_clock_domain','host_clock_domain','controls_json')})
    for key,value in expected.items():
        if key not in report or preprocess.canonical(report[key])!=preprocess.canonical(value):raise ValueError('changed/missing combined report contract: '+key)
    if type(report.get('lr_tolerance')) not in (int,float) or report['lr_tolerance']!=1:raise ValueError('changed LR tolerance')
    actual=report.get('metric_calibration',{})
    if set(actual)!=set(metric):raise ValueError('missing cached metric values/bitpatterns')
    for key in ('fx','baseline_m','cx_offset'):
        bits=actual.get(key+'_bits')
        if type(bits) is not int or not 0<=bits<=0xffffffff or type(actual.get(key)) not in (int,float) or f32_bits(actual[key])!=bits or bits!=metric[key+'_bits']:raise ValueError('cached metric float bitpatterns differ from bound calibration')
    def timing(key,row,positive=False):
        if not isinstance(row,dict) or set(row)!={'p50','p95','mean'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 or (positive and v==0) for v in row.values()) or row['p95']<row['p50']:raise ValueError('invalid combined timing: '+key)
    for key in TIMINGS:
        if key in ('gpu_left_stereo_ms','gpu_right_stereo_ms'):
            if stereo_schedule=='paired-per-path':
                if key not in report or report[key] is not None:raise ValueError('paired per-reference timing must be unavailable: '+key)
                continue
        timing(key,report.get(key))
    if stereo_schedule=='sequential':
        if 'gpu_paired_stereo_ms' not in report or report['gpu_paired_stereo_ms'] is not None or report.get('gpu_paired_stereo_phases_ms')!={}:raise ValueError('sequential paired timing must be unavailable')
    else:
        timing('gpu_paired_stereo_ms',report.get('gpu_paired_stereo_ms'),positive=True)
        phases=report.get('gpu_paired_stereo_phases_ms')
        expected_phases=set(PAIRED_PHASES+(PAIRED_DIAGONALS if profile['paths']==8 else ())+('select',))
        if not isinstance(phases,dict) or set(phases)!=expected_phases:raise ValueError('changed/missing paired stereo phases')
        for key,row in phases.items():timing(key,row,positive=True)
    if report['gpu_combined_queue_span_ms']['mean']<=0:raise ValueError('actual combined GPU span absent')
    for key in ('initialization_ms','diagnostic_after_timing_ms'):
        if type(report.get(key)) not in (int,float) or not math.isfinite(report[key]) or report[key]<0:raise ValueError('invalid bounded extra timing')
    for key,value in zip(('requested_observation_buffer_bytes','requested_combined_buffer_bytes'),requested_bytes(manifest,profile,slots)):
        if type(report.get(key)) is not int or report[key]!=value or not 0<value<2*1024**3:raise ValueError('changed requested buffer memory accounting')
    pairs=identity['shader_source_sha256_pairs']
    for key,value in (('shader_source_sha256_pair',pairs['depth']),('preprocessing_shader_source_sha256_pair',pairs['preprocess']),('stereo_shader_source_sha256_pair',pairs['stereo'])):
        if report.get(key)!=value:raise ValueError('actual combined shader source identity changed')
    device=report.get('device',{})
    if device.get('slangc_sha256')!=identity['compiler_sha256']:raise ValueError('actual combined compiler differs')
    for row,keys in ((report,('validation_errors_after_teardown','validation_warnings_after_teardown')),(device,('validation_errors','validation_warnings'))):
        for key in keys:
            if type(row.get(key)) is not int or row[key]!=0:raise ValueError('combined Vulkan/teardown diagnostics')


def validate_bounds(disparities,paths,p1,p2,frames,warmup,slots,wave_size,validation_only):
    if any(type(v) is not int for v in (disparities,paths,p1,p2,frames,warmup,slots,wave_size)) or disparities not in ((16,32,64,128) if validation_only else (32,64,128)) or paths not in (4,8) or not 0<=p1<=p2<=10000 or paths*p2>65535 or not 1<=frames<=300 or not 0<=warmup<=30 or slots not in (1,2) or wave_size not in (32,64) or (not validation_only and (frames<30 or warmup<3)):raise ValueError('invalid bounded combined profile/counts')


def run_comparison(manifest_path,rhi,native,compiler,output,disparities=128,paths=4,p1=8,p2=32,frames=30,warmup=3,slots=2,wave_size=32,validation=False,validation_only=False,stereo_schedule="sequential"):
    if type(stereo_schedule) is not str or stereo_schedule not in STEREO_SCOPES:raise ValueError("unknown stereo schedule")
    validate_bounds(disparities,paths,p1,p2,frames,warmup,slots,wave_size,validation_only)
    manifest_path=Path(manifest_path).resolve();manifest_entry_sha256=base.sha256(manifest_path)
    manifest,metadata,luma,supports,metric,cv_identity=fixture_planes(manifest_path)
    width,height=manifest['rectified']['width'],manifest['rectified']['height']
    if type(width) is not int or type(height) is not int or not 5<=width<=1280 or not 5<=height<=720 or type(metadata['frame_id']) is not int or not 0<=metadata['frame_id']<1<<63:raise ValueError('invalid bounded fixture shape/sourceID')
    profile=dict(width=width,height=height,disparities=disparities,paths=paths,p1=p1,p2=p2,subpixel=True)
    output=Path(output).absolute();base.create_output(output);derived={}
    for eye,name in enumerate(('left','right')):
        for filename,data in ((name+'.u32',luma[eye]),('mirror-'+name+'.u32',mirror_rows(luma[eye],width,height))):
            path=output/filename;path.write_bytes(data);path.chmod(0o444);derived[filename]=base.sha256(path)
    evidence=dict(schema=1,passed=False,selected=None,profile=profile,frames=frames,warmup=warmup,slots=slots,wave_size=wave_size,validation_requested=validation,performance_qualified_counts=not validation_only,
        stereo_schedule=stereo_schedule,stereo_timing_scope=STEREO_SCOPES[stereo_schedule],fixture_manifest_entry_sha256=manifest_entry_sha256,scope='Repeated exact recorded fixture: combined upload/preprocess/dualSGM/LR/metric/metadata; no fresh-camera, physical truth, motion, RGB-D synthesis or sustained60Hz qualification.',
        timing_contract=dict(native=GPU_SCOPE,rhi='Two independent transfer-inclusive oracle runs; no warmup, no comparable performance ratio.',diagnostics='All fifteen native dumps and native exact integer/P checks after samples; frozen RHI oracle outside native timing.'),
        support_policy=SUPPORT_POLICY,compound_cpu_float_reference_checked=False,frozen_gpu_output_bytes_checked=False,children={},cpu_opencv=cv_identity,
        inherited_validation_environment={k:os.environ[k] for k in ('VK_INSTANCE_LAYERS','VK_LAYER_VALIDATE_SYNC','VK_LAYER_ENABLES','VK_LOADER_DEBUG','SF_DAXA_VALIDATION') if k in os.environ})
    try:
        identity=native_identities(rhi,native,compiler,manifest_path);evidence['identities']=identity
        if identity['manifest_sha256']!=manifest_entry_sha256:raise ValueError('fixture manifest changed during CPU oracle setup')
        fixture_hashes=identity['fixture_file_sha256'];reports={};dumps={}
        def verify_inputs():
            for file,digest in fixture_hashes.items():
                if base.sha256(file)!=digest:raise ValueError('bound fixture bytes changed')
            for file,digest in derived.items():
                if base.sha256(output/file)!=digest:raise ValueError('derived oracle inputs changed')
        def child(name,command):
            verify_inputs();result=subprocess.run(command,capture_output=True,text=True,errors='replace');(output/(name+'.stdout')).write_text(result.stdout);(output/(name+'.stderr')).write_text(result.stderr);base.require_child_clean(result.returncode,result.stdout,result.stderr);verify_inputs();evidence['children'][name]=dict(returncode=result.returncode,command=command);reports[name+'.stdout']=base.sha256(output/(name+'.stdout'));reports[name+'.stderr']=base.sha256(output/(name+'.stderr'));return result
        command=[str(native),'--manifest',str(manifest_path),'--output',str(output/'native.json'),'--dump-prefix',str(output/'gpu'),'--check-reference','--frames',str(frames),'--warmup',str(warmup),'--slots',str(slots),'--wave-size',str(wave_size),'--disparities',str(disparities),'--paths',str(paths),'--p1',str(p1),'--p2',str(p2),'--stereo-schedule',stereo_schedule]
        if validation:command.append('--validation')
        child('native',command);report=base.strict_json((output/'native.json').read_text());validate_report(report,manifest,metadata,metric,identity,profile,frames,warmup,slots,wave_size,validation,stereo_schedule=stereo_schedule);evidence['native_report']=report;reports['native.json']=base.sha256(output/'native.json')
        # Use the actual cached native float values/bitpatterns, not a fresh
        # Python double calibration or CPU-evaluated depth formula.
        cached=report['metric_calibration'];metric_args=[]
        for option,key in (('--fx','fx'),('--baseline-m','baseline_m'),('--principal-offset','cx_offset')):
            value=struct.unpack('<f',struct.pack('<I',cached[key+'_bits']))[0];metric_args.extend((option,repr(value)))
        raw=[]
        for eye,name in enumerate(('left','right')):
            prefix=output/('rhi-'+name);command=[str(rhi)]
            for key,value in profile.items():command+=['--'+key,str(int(value)) if type(value) is bool else str(value)]
            command+=['--left',str(output/('mirror-right.u32' if eye else 'left.u32')),'--right',str(output/('mirror-left.u32' if eye else 'right.u32')),'--iterations',str(frames),'--output',str(prefix),'--shader-directory',str((base.ROOT/'python/stereoforge/shaders/stereo').resolve()),*metric_args]
            if validation:command.append('--validation')
            result=child('rhi-'+name,command);r=base.strict_json(result.stdout);base.validate_rhi_profile(r,profile,frames)
            if r.get('metric_depth') is not True or r.get('vulkan_validation') is not validation or r.get('adapter')!=report['device'].get('device'):raise ValueError('frozen RHI metric/validation/adapter profile changed')
            elapsed=r.get('transfer_inclusive_total_ms')
            if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<=0:raise ValueError('frozen RHI timing missing')
            path=output/('rhi-'+name+'.json');path.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');reports[path.name]=base.sha256(path);evidence['children']['rhi-'+name]['report']=r
            layer={}
            for plane in RHI_PLANES:
                file=Path(str(prefix)+'.'+plane);data=file.read_bytes();array(data,'<u4',width*height);dumps[str(file)]=base.sha256(file);layer[plane]=mirror_rows(data,width,height) if eye else data
            raw.append(layer)
        expected=expected_observations(raw,supports,width,height,metadata['frame_id']);comparison={};raw_comparison={}
        for plane,data in expected.items():
            file=Path(str(output/'gpu')+'.'+plane);actual=file.read_bytes()
            if len(actual)!=len(data):raise ValueError('missing/incomplete combined output: '+plane)
            oracle=output/('expected.'+plane);oracle.write_bytes(data);dumps[str(file)]=base.sha256(file);dumps[str(oracle)]=base.sha256(oracle)
            comparison[plane]=dict(bytes=len(data),actual_sha256=dumps[str(file)],expected_sha256=dumps[str(oracle)],exact=actual==data)
        for eye,name in enumerate(('left','right')):
            for plane in RAW_PLANES:
                file=Path(str(output/'gpu')+'.'+name+'.'+plane);actual=file.read_bytes();data=raw[eye][plane]
                if len(actual)!=len(data):raise ValueError('missing/incomplete native raw stereo dump')
                dumps[str(file)]=base.sha256(file);raw_comparison[name+'.'+plane]=dict(bytes=len(data),exact=actual==data,actual_sha256=dumps[str(file)],rhi_sha256=hashlib.sha256(data).hexdigest())
        evidence['rhi_ordinary_geometry_output_sha256']=[{p:hashlib.sha256(data).hexdigest() for p,data in layer.items()} for layer in raw]
        evidence['comparison']=comparison;evidence['raw_stereo_comparison']=raw_comparison;evidence['metric_cached_float_bits']=cached
        verify_inputs()
        if identity!=native_identities(rhi,native,compiler,manifest_path):raise ValueError('source/compiler/binary/dependency/SPIRV/fixture identity changed')
        if any(base.sha256(output/name)!=digest for name,digest in reports.items()) or any(base.sha256(name)!=digest for name,digest in dumps.items()):raise ValueError('bound report/dump bytes changed during comparison')
        evidence['report_sha256']=reports;evidence['dump_sha256']=dumps;evidence['derived_input_sha256']=derived;evidence['physical_device_boundary']='Daxa queried UUID/driver; frozen RHI adapter name only, not independently established UUID equality.'
        evidence['passed']=all(row['exact'] for row in comparison.values()) and all(row['exact'] for row in raw_comparison.values());evidence['frozen_gpu_output_bytes_checked']=True
    except (ValueError,OSError,KeyError,TypeError,OverflowError,struct.error,subprocess.SubprocessError) as error:evidence['error']=str(error)
    (output/'comparison.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n');return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--compiler',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rhi-binary',type=Path,default=base.ROOT/'build/standalone/native_stereo');parser.add_argument('--daxa-binary',type=Path,default=base.ROOT/'build/daxa/daxa_depth_benchmark')
    for key,default in (('disparities',128),('paths',4),('p1',8),('p2',32),('slots',2),('wave-size',32)):parser.add_argument('--'+key,type=int,default=default)
    parser.add_argument('--stereo-schedule',choices=tuple(STEREO_SCOPES),default='sequential')
    parser.add_argument('--frames',type=int);parser.add_argument('--warmup',type=int);parser.add_argument('--validation',action='store_true');parser.add_argument('--validation-only',action='store_true');a=parser.parse_args()
    try:result=run_comparison(a.manifest,a.rhi_binary.resolve(),a.daxa_binary.resolve(),a.compiler.resolve(),a.output,a.disparities,a.paths,a.p1,a.p2,a.frames if a.frames is not None else 1 if a.validation_only else 30,a.warmup if a.warmup is not None else 0 if a.validation_only else 3,a.slots,a.wave_size,a.validation,a.validation_only,stereo_schedule=a.stereo_schedule)
    except (ValueError,OSError,KeyError,OverflowError) as error:parser.error(str(error))
    print(json.dumps(dict(passed=result['passed'],output=str(a.output))));return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
