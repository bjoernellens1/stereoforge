#!/usr/bin/env python3
"""Experimental derived-right nine-plane byte oracle against independent dense RHI sums.

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
import compare_daxa_depth as legacy

SCOPE='repeated_retained_record_left_aggregated_correlated_LR_metric_no_RGBD_synthesis'
RIGHT_REFERENCE_MODE='left-aggregated-selection'
SUPPORT_POLICY='source_center_and_nearest_correspondence_remap_support; not_Census_tap_certification; LR_mask_unchanged'
GPU_SCOPE='same_queue_upload_start_through_metric_end_including_intergraph_gaps'
MEMORY_SCOPE='buffer_bytes_only; excludes_allocator_driver_queries_diagnostics_CPU_source'
ADMISSION_POLICY='one_explicit_application_snapshot; SDK_GC_internal_scans_additional_included_host_wall'
PLANES=('lr_disparity.f32','lr_confidence.f32','lr_validity.u32','depth.f32','metric_confidence.f32','metric_validity.u32','provenance.u32','source_ids.i64','age_ns.i64')
RAW_PLANES=base.OUTPUT_PLANES
RHI_PLANES=RAW_PLANES+('depth.f32','depth_validity.u32')
TIMINGS=('enqueue_cpu_ms','wait_cpu_ms','timestamp_query_cpu_ms','host_wall_ms','upload_cpu_ms','gpu_upload_ms','gpu_decode_ms','gpu_rectify_ms','gpu_left_stereo_ms','gpu_right_stereo_ms','gpu_lr_ms','gpu_metric_metadata_ms','gpu_combined_queue_span_ms')
STEREO_SCOPES={
    'shared-census-sequential':'shared_census_then_sequential_reference_phase_intervals; no_isolated_per_reference_times',
    'sequential':'sequential_per_reference_graphs',
    'paired-per-path':'paired_per_path_phase_intervals_include_both_references; no_isolated_per_reference_times',
    'layered-per-path':'layered_per_path_phase_intervals_include_both_references; no_isolated_per_reference_times',
}
STEREO_DISPATCH_FORMS={
    'shared-census-sequential':'shared_census_once_then_sequential_independent_references',
    'sequential':'two_sequential_reference_graphs',
    'paired-per-path':'two_reference_dispatches_per_phase',
    'layered-per-path':'two_reference_dispatches_census_clear; z2_single_dispatch_aggregation_selection',
}
SHARED_SHADER_SOURCES=('python/stereoforge/shaders/daxa/stereo_parameters.slang','python/stereoforge/shaders/ops/sgm_specialized.slang')
LAYERED_SHADER_SOURCE='python/stereoforge/shaders/daxa/stereo_layered.slang'
LAYERED_ENTRIES=('aggregate_layered8','aggregate_layered16','select_layered8','select_layered16')
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
            'provenance.u32':(metric.astype('<u4')*5),'source_ids.i64':np.where(metric,source_id,-1).astype('<i8'),'age_ns.i64':np.zeros(n,'<i8')}
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


def requested_bytes_for_schedule(manifest,profile,slots,stereo_schedule='shared-census-sequential'):
    if stereo_schedule!='shared-census-sequential':raise ValueError('derived mode requires shared census schedule')
    observations,total=legacy.requested_bytes_for_schedule(manifest,profile,slots,stereo_schedule)
    # Derived Right reuses Left P: remove exactly one packed penalty volume.
    bits=8 if profile['paths']*profile['p2']<=255 else 16
    words=(profile['disparities']+32//bits-1)//(32//bits)
    return observations,total-slots*profile['width']*profile['height']*words*4


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


DERIVED_SHADER_GROUPS={
    'stereo_derived_shader_source_sha256_triple':('python/stereoforge/shaders/daxa/stereo_derived.slang','python/stereoforge/shaders/daxa/stereo_parameters.slang','python/stereoforge/shaders/ops/sgm.slang'),
    'depth_derived_shader_source_sha256_triple':('python/stereoforge/shaders/daxa/depth_derived.slang','python/stereoforge/shaders/daxa/depth.slang','python/stereoforge/shaders/ops/stereo_observation.slang'),
}


def native_identities(rhi,native,compiler,manifest,reference):
    from compare_daxa_view import dependencies
    identity=legacy.native_identities(rhi,native,compiler,manifest)
    identity['reference_executable']=str(Path(reference).resolve(strict=True))
    identity['reference_executable_sha256']=base.sha256(reference)
    identity['reference_dependency_sha256']=dependencies(reference,False)['sha256']
    for name in ('apps/native_derived_right.cpp','python/stereoforge/shaders/reference/derived_right.slang','benchmarks/compare_daxa_depth_derived.py','CMakeLists.txt'):
        identity['source_sha256'][name]=base.sha256(base.ROOT/name)
    for key,names in DERIVED_SHADER_GROUPS.items():
        for name in names:identity['source_sha256'][name]=base.sha256(base.ROOT/name)
        identity[key]=':'.join(identity['source_sha256'][name] for name in names)
    directory=Path(identity['offline_shader_directory'])
    for name in ('daxa-stereo-select_right_derived8.spv','daxa-stereo-select_right_derived16.spv','daxa-depth-metric_observation_derived.spv'):
        identity['offline_spirv_sha256'][name]=base.sha256(directory/name)
    result=subprocess.run([str(native),'--derived-shader-source-identities'],capture_output=True,text=True,errors='replace')
    base.require_child_clean(result.returncode,result.stdout,result.stderr)
    getters=base.strict_json(result.stdout)
    expected={key:identity[key] for key in DERIVED_SHADER_GROUPS}
    if getters!=expected:raise ValueError('compiled derived shader getter identities differ from bound source')
    identity['derived_compiled_shader_getters']=getters
    identity['compiler_input_sha256']['derived_reference']=compiler_header_identities(Path(reference).resolve().parent,(('native_derived_right','apps/native_derived_right.cpp'),))
    return identity


def validate_report(report,manifest,metadata,metric,identity,profile,frames,warmup,slots,wave,validation,stereo_schedule="shared-census-sequential"):
    if stereo_schedule!='shared-census-sequential':raise ValueError('derived mode requires shared census schedule')
    expected=dict(profile,schema='stereoforge.daxa_depth.derived.v1',right_reference_mode=RIGHT_REFERENCE_MODE,left_estimator='independent-sgm',right_estimator=RIGHT_REFERENCE_MODE,lr_dependency='left-aggregated-correlated',lr_references_independent=False,metric_provenance_valid_value=5,metric_provenance_invalid_value=0,metric_provenance_scope='both_left_and_right_layers_share_correlated_LR_dependency',right_recurrence_allocated=False,stereo_phase_count=profile['paths']+5,scope=SCOPE,frames=frames,warmup=warmup,slots=slots,required_wave_size=wave,metric_support_policy=SUPPORT_POLICY,layers='left_then_right',depth_units='axial_Z_metres_rectified_rays',
        stereo_schedule=stereo_schedule,stereo_dispatch_form='shared_census_once_left_aggregation_then_derived_right_selection',stereo_timing_scope='shared_census_left_aggregation_derived_right_selection_phase_intervals; no_isolated_per_reference_times',per_reference_stereo_timings_available=stereo_schedule=='sequential',
        source_metadata=metadata,source_id=metadata['frame_id'],original_identity=manifest['calibration']['original_identity'],rectified_identity=manifest['calibration']['rectified_identity'],fixture_manifest_sha256=identity['manifest_sha256'],capture_sha256=manifest['capture']['sha256'],record_ordinal=manifest['capture']['record_ordinal'],
        gpu_combined_scope=GPU_SCOPE,requested_memory_scope=MEMORY_SCOPE,admission_completion_policy=ADMISSION_POLICY,lr_mask_metadata_reference_checked=True,both_stereo_integer_reference_checked=False,both_stereo_packed_penalties_reference_checked=False,left_stereo_integer_reference_checked=True,left_stereo_packed_penalties_reference_checked=True,derived_right_mask_reference_checked=True,right_stereo_integer_winner_reference_checked=False,expected_preprocess_planes_checked=True,metric_compound_float_reference_checked=False,stereo_compound_float_reference_checked=False,completed_output_retirements=frames+warmup,capture_drops='not_measured; repeated_fixture',camera_age='unknown; no_clock_mapping',validation_requested=validation)
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
            if stereo_schedule!='sequential':
                if key not in report or report[key] is not None:raise ValueError('paired per-reference timing must be unavailable: '+key)
                continue
        timing(key,report.get(key))
    if stereo_schedule=='sequential':
        if 'gpu_paired_stereo_ms' not in report or report['gpu_paired_stereo_ms'] is not None or report.get('gpu_paired_stereo_phases_ms')!={}:raise ValueError('sequential paired timing must be unavailable')
    else:
        timing('gpu_paired_stereo_ms',report.get('gpu_paired_stereo_ms'),positive=True)
        phases=report.get('gpu_paired_stereo_phases_ms')
        expected_phases=set(PAIRED_PHASES+(PAIRED_DIAGONALS if profile['paths']==8 else ())+('select',))
        if stereo_schedule=='shared-census-sequential':
            reference_phases=PAIRED_PHASES[2:]+(PAIRED_DIAGONALS if profile['paths']==8 else ())+('select',)
            expected_phases={'shared_census_left','shared_census_right','derived_right_select'}|{'left_'+phase for phase in reference_phases}
        if not isinstance(phases,dict) or set(phases)!=expected_phases:raise ValueError('changed/missing paired stereo phases')
        for key,row in phases.items():timing(key,row,positive=True)
    if report['gpu_combined_queue_span_ms']['mean']<=0:raise ValueError('actual combined GPU span absent')
    for key in ('initialization_ms','diagnostic_after_timing_ms'):
        if type(report.get(key)) not in (int,float) or not math.isfinite(report[key]) or report[key]<0:raise ValueError('invalid bounded extra timing')
    if stereo_schedule=='shared-census-sequential':
        census_bytes=16*profile['width']*profile['height']*slots
        shared=dict(census_descriptor_storage_shared=True,census_dispatches_per_frame=2,independent_reference_penalty_buffers=False,
                    requested_census_descriptor_buffer_bytes=census_bytes,census_descriptor_buffer_savings_bytes=census_bytes)
        for key,value in shared.items():
            if type(report.get(key)) is not type(value) or report[key]!=value:raise ValueError('changed/missing shared census contract: '+key)
    _,independent_bytes=legacy.requested_bytes_for_schedule(manifest,profile,slots,stereo_schedule)
    _,derived_bytes=requested_bytes_for_schedule(manifest,profile,slots,stereo_schedule)
    if type(report.get('right_penalty_buffer_savings_bytes')) is not int or report['right_penalty_buffer_savings_bytes']!=independent_bytes-derived_bytes:raise ValueError('changed derived Right penalty savings')
    for key,value in zip(('requested_observation_buffer_bytes','requested_combined_buffer_bytes'),requested_bytes_for_schedule(manifest,profile,slots,stereo_schedule)):
        if type(report.get(key)) is not int or report[key]!=value or not 0<value<2*1024**3:raise ValueError('changed requested buffer memory accounting')
    for key in ('stereo_shared_shader_source_sha256_pair','stereo_layered_shader_source_sha256'):
        if report.get(key)!=identity[key]:raise ValueError('actual compiled stereo helper source identity changed: '+key)
    pairs=identity['shader_source_sha256_pairs']
    for key,value in (('shader_source_sha256_pair',pairs['depth']),('preprocessing_shader_source_sha256_pair',pairs['preprocess']),('stereo_shader_source_sha256_pair',pairs['stereo'])):
        if report.get(key)!=value:raise ValueError('actual combined shader source identity changed')
    for key in DERIVED_SHADER_GROUPS:
        if key not in identity or report.get(key)!=identity[key]:raise ValueError('actual compiled derived shader source identity changed: '+key)
    device=report.get('device',{})
    if device.get('slangc_sha256')!=identity['compiler_sha256']:raise ValueError('actual combined compiler differs')
    for row,keys in ((report,('validation_errors_after_teardown','validation_warnings_after_teardown')),(device,('validation_errors','validation_warnings'))):
        for key in keys:
            if type(row.get(key)) is not int or row[key]!=0:raise ValueError('combined Vulkan/teardown diagnostics')


def validate_bounds(disparities,paths,p1,p2,frames,warmup,slots,wave_size,validation_only):
    if any(type(v) is not int for v in (disparities,paths,p1,p2,frames,warmup,slots,wave_size)) or disparities not in ((16,32,64,128) if validation_only else (32,64,128)) or paths not in (4,8) or not 0<=p1<=p2<=10000 or paths*p2>65535 or not 1<=frames<=300 or not 0<=warmup<=30 or slots not in (1,2) or wave_size not in (32,64) or (not validation_only and (frames<30 or warmup<3)):raise ValueError('invalid bounded combined profile/counts')


def census_validity(width,height):
    """Actual 5x5 Census availability, independently determined from geometry."""
    y,x=np.indices((height,width));return ((x>=2)&(x+2<width)&(y>=2)&(y+2<height)).astype('<u4').tobytes()


def scalar_derived_selection(sums,left_valid,right_valid,subpixel=True):
    """Small synthetic integer oracle; float expressions are diagnostic only.

    sums is dense [height,width,D] LEFT aggregation. Every neighbor has its
    own xLeft=xRight+d. Unsupported candidates never enter either minimum.
    """
    sums=np.asarray(sums)
    if sums.ndim!=3 or sums.dtype.kind not in 'ui' or np.any(sums>0xffffffff):raise ValueError('dense uint32 sums required')
    h,w,D=sums.shape;left=np.asarray(left_valid);right=np.asarray(right_valid)
    if min(h,w,D)<=0 or left.shape!=(h,w) or right.shape!=(h,w) or np.any((left!=0)&(left!=1)) or np.any((right!=0)&(right!=1)):raise ValueError('invalid dimensions/Census masks')
    disparity=np.full((h,w),np.nan,'<f4');confidence=np.zeros((h,w),'<f4');validity=np.zeros((h,w),'<u4')
    for y in range(h):
        for x in range(w):
            candidates=[(int(sums[y,x+d,d]),d) for d in range(min(D,w-x)) if right[y,x] and left[y,x+d]]
            candidates.sort()
            if len(candidates)<2:continue
            best,d=candidates[0];second=candidates[1][0]
            if not(d>0 and best<10000 and second>best):continue
            value=float(d)
            if subpixel and d+1<D and d+1<w-x and left[y,x+d-1] and left[y,x+d+1]:
                lo=int(sums[y,x+d-1,d-1]);hi=int(sums[y,x+d+1,d+1]);denominator=lo+hi-2*best
                if denominator>0:value+=.5*(lo-hi)/denominator
            disparity[y,x]=value;confidence[y,x]=(second-best)/max(second,1);validity[y,x]=1
    return dict(zip(RAW_PLANES,(disparity.tobytes(),confidence.tobytes(),validity.tobytes())))


def quality_difference(derived,independent,width,height):
    """Development comparison only; independent SGM is not physical truth."""
    n=width*height;a=array(derived['disparity.f32'],'<f4',n);b=array(independent['disparity.f32'],'<f4',n)
    av=array(derived['validity.u32'],'<u4',n)!=0;bv=array(independent['validity.u32'],'<u4',n)!=0;both=av&bv&np.isfinite(a)&np.isfinite(b)
    difference=np.abs(a[both].astype(float)-b[both].astype(float))
    return dict(scope='development_difference_from_independent_right_SGM; not_physical_truth',promoted=False,derived_valid=int(av.sum()),independent_valid=int(bv.sum()),validity_disagreement=int((av!=bv).sum()),common_valid=int(both.sum()),mean_absolute_disparity_difference=float(difference.mean()) if difference.size else None,max_absolute_disparity_difference=float(difference.max()) if difference.size else None)


def run_comparison(manifest_path,rhi,native,compiler,output,reference,disparities=128,paths=4,p1=8,p2=32,frames=30,warmup=3,slots=2,wave_size=32,validation=False,validation_only=False,stereo_schedule='shared-census-sequential'):
    if stereo_schedule!='shared-census-sequential':raise ValueError('derived mode requires shared census schedule')
    validate_bounds(disparities,paths,p1,p2,frames,warmup,slots,wave_size,validation_only)
    manifest_path=Path(manifest_path).resolve();entry=base.sha256(manifest_path)
    manifest,metadata,luma,supports,metric,cv_identity=fixture_planes(manifest_path)
    w,h=manifest['rectified']['width'],manifest['rectified']['height']
    if type(w) is not int or type(h) is not int or not 5<=w<=1280 or not 5<=h<=720:raise ValueError('invalid bounded shape')
    profile=dict(width=w,height=h,disparities=disparities,paths=paths,p1=p1,p2=p2,subpixel=True)
    output=Path(output).absolute();base.create_output(output);inputs={}
    for eye,name in enumerate(('left','right')):
        for filename,data in ((name+'.u32',luma[eye]),('mirror-'+name+'.u32',mirror_rows(luma[eye],w,h)),(name+'-census-valid.u32',census_validity(w,h))):
            file=output/filename;file.write_bytes(data);file.chmod(0o444);inputs[filename]=base.sha256(file)
    evidence=dict(schema='stereoforge.daxa_depth.derived.comparison.v1',passed=False,promoted=False,right_reference_mode=RIGHT_REFERENCE_MODE,profile=profile,children={},cpu_opencv=cv_identity,compound_cpu_float_reference_checked=False,frozen_gpu_output_bytes_checked=False,scope='Experimental correlated LR using shared Left aggregation; exact implementation bytes only; held-out quality constraints required before promotion.',performance_qualified_counts=not validation_only)
    try:
        identity=native_identities(rhi,native,compiler,manifest_path,reference);evidence['identities']=identity
        if entry!=identity['manifest_sha256']:raise ValueError('fixture manifest changed during CPU oracle setup')
        reports={};dumps={}
        def verify():
            for path,digest in identity['fixture_file_sha256'].items():
                if base.sha256(path)!=digest:raise ValueError('bound fixture bytes changed')
            for name,digest in inputs.items():
                if base.sha256(output/name)!=digest:raise ValueError('derived oracle inputs changed')
        def child(name,cmd):
            verify();result=subprocess.run(cmd,capture_output=True,text=True,errors='replace');base.require_child_clean(result.returncode,result.stdout,result.stderr);verify()
            for suffix,text in (('.stdout',result.stdout),('.stderr',result.stderr)):
                file=output/(name+suffix);file.write_text(text);reports[file.name]=base.sha256(file)
            evidence['children'][name]=dict(command=cmd,returncode=result.returncode);return result
        cmd=[str(native),'--manifest',str(manifest_path),'--output',str(output/'native.json'),'--dump-prefix',str(output/'gpu'),'--check-reference','--right-reference-mode',RIGHT_REFERENCE_MODE,'--stereo-schedule',stereo_schedule]
        for key,value in dict(frames=frames,warmup=warmup,slots=slots,**{'wave-size':wave_size},disparities=disparities,paths=paths,p1=p1,p2=p2).items():cmd+=['--'+key,str(value)]
        if validation:cmd+=['--validation']
        child('native',cmd);report=base.strict_json((output/'native.json').read_text());validate_report(report,manifest,metadata,metric,identity,profile,frames,warmup,slots,wave_size,validation,stereo_schedule);reports['native.json']=base.sha256(output/'native.json');evidence['native_report']=report
        args=[]
        for key,value in profile.items():args+=['--'+key,str(int(value)) if type(value) is bool else str(value)]
        metric_args=[]
        for option,key in (('--fx','fx'),('--baseline-m','baseline_m'),('--principal-offset','cx_offset')):metric_args += [option,repr(struct.unpack('<f',struct.pack('<I',report['metric_calibration'][key+'_bits']))[0])]
        shader=str((base.ROOT/'shaders/stereo').resolve())
        ref_cmd=[str(reference),*args,'--left',str(output/'left.u32'),'--right',str(output/'right.u32'),'--left-valid',str(output/'left-census-valid.u32'),'--right-valid',str(output/'right-census-valid.u32'),'--output',str(output/'reference'),'--shader-directory',shader,'--reference-shader',str(base.ROOT/'shaders/reference/derived_right.slang'),*metric_args]
        if validation:ref_cmd+=['--validation']
        rr=base.strict_json(child('reference',ref_cmd).stdout)
        expected_ref=dict(schema='stereoforge.rhi_derived_right.v1',width=w,height=h,disparities=disparities,paths=paths,metric_depth=True,vulkan_validation=validation,adapter=report['device']['device'])
        if rr!=expected_ref:raise ValueError('independent derived reference profile changed')
        raw=[]
        for eye in ('left','right'):
            row={}
            for plane in RHI_PLANES:
                file=output/('reference.'+eye+'.'+plane);data=file.read_bytes();array(data,'<u4',w*h);row[plane]=data;dumps[str(file)]=base.sha256(file)
            raw.append(row)
        controls=[]
        for eye in ('left','right'):
            prefix=output/('control-'+eye);cmd=[str(rhi),*args,'--left',str(output/('left.u32' if eye=='left' else 'mirror-right.u32')),'--right',str(output/('right.u32' if eye=='left' else 'mirror-left.u32')),'--output',str(prefix),'--iterations','1','--shader-directory',shader,*metric_args]
            if validation:cmd+=['--validation']
            control_report=base.strict_json(child('control-'+eye,cmd).stdout);base.validate_rhi_profile(control_report,profile,1)
            if control_report.get('adapter')!=report['device']['device'] or control_report.get('metric_depth') is not True or control_report.get('vulkan_validation') is not validation:raise ValueError('control RHI profile changed')
            row={}
            for plane in RHI_PLANES:
                file=Path(str(prefix)+'.'+plane);data=file.read_bytes();array(data,'<u4',w*h);dumps[str(file)]=base.sha256(file);row[plane]=data if eye=='left' else mirror_rows(data,w,h)
            controls.append(row)
        evidence['reference_left_control_byteexact']=raw[0]==controls[0]
        if not evidence['reference_left_control_byteexact']:raise ValueError('new RHI raw Left differs from frozen native_stereo')
        evidence['quality_difference']=quality_difference(raw[1],controls[1],w,h)
        expected=expected_observations(raw,supports,w,h,metadata['frame_id']);comparison={};raw_comparison={}
        for plane,data in expected.items():
            file=Path(str(output/'gpu')+'.'+plane);actual=file.read_bytes()
            if len(actual)!=len(data):raise ValueError('missing/incomplete derived observation plane')
            oracle=output/('expected.'+plane);oracle.write_bytes(data);dumps[str(file)]=base.sha256(file);dumps[str(oracle)]=base.sha256(oracle);comparison[plane]=dict(bytes=len(data),exact=actual==data,actual_sha256=dumps[str(file)],expected_sha256=dumps[str(oracle)])
        for eye,name in enumerate(('left','right')):
            for plane in RAW_PLANES:
                file=Path(str(output/'gpu')+'.'+name+'.'+plane);actual=file.read_bytes();data=raw[eye][plane]
                if len(actual)!=len(data):raise ValueError('missing/incomplete native raw output')
                dumps[str(file)]=base.sha256(file);raw_comparison[name+'.'+plane]=dict(bytes=len(data),exact=actual==data,actual_sha256=dumps[str(file)],expected_sha256=hashlib.sha256(data).hexdigest())
        evidence.update(comparison=comparison,raw_stereo_comparison=raw_comparison)
        verify()
        if identity!=native_identities(rhi,native,compiler,manifest_path,reference):raise ValueError('source/compiler/binary/dependency/SPIRV/fixture identity changed')
        if any(base.sha256(output/name)!=digest for name,digest in reports.items()) or any(base.sha256(name)!=digest for name,digest in dumps.items()):raise ValueError('bound report/dump bytes changed')
        evidence.update(report_sha256=reports,dump_sha256=dumps,derived_input_sha256=inputs,frozen_gpu_output_bytes_checked=True,passed=all(row['exact'] for row in comparison.values()) and all(row['exact'] for row in raw_comparison.values()))
    except (ValueError,OSError,KeyError,TypeError,OverflowError,struct.error,subprocess.SubprocessError) as error:evidence['error']=str(error)
    (output/'comparison.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n');return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in (('rhi-binary',base.ROOT/'build/standalone/native_stereo'),('reference-binary',base.ROOT/'build/standalone/native_derived_right'),('daxa-binary',base.ROOT/'build/daxa/daxa_depth_benchmark')):parser.add_argument('--'+name,type=Path,default=default)
    for name in ('manifest','compiler','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name,default in (('disparities',128),('paths',4),('p1',8),('p2',32),('frames',30),('warmup',3),('slots',2),('wave-size',32)):parser.add_argument('--'+name,type=int,default=default)
    for name in ('validation','validation-only'):parser.add_argument('--'+name,action='store_true')
    parser.add_argument('--stereo-schedule',choices=['shared-census-sequential'],default='shared-census-sequential');parser.add_argument('--right-reference-mode',choices=[RIGHT_REFERENCE_MODE],default=RIGHT_REFERENCE_MODE)
    a=parser.parse_args();result=run_comparison(a.manifest,a.rhi_binary,a.daxa_binary,a.compiler,a.output,a.reference_binary,a.disparities,a.paths,a.p1,a.p2,a.frames,a.warmup,a.slots,a.wave_size,a.validation,a.validation_only,a.stereo_schedule)
    print(json.dumps(dict(passed=result['passed'],promoted=False,output=str(a.output))))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
