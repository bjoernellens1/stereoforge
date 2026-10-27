#!/usr/bin/env python3
"""Bounded right-reference C versus frozen RHI mirror oracle, exact final bytes.

Right disparity is positive x_left-x_right. No LR/depth/full-pipeline qualification.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import compare_daxa_stereo as base

RIGHT_ENTRIES=('aggregate_right_specialized8','aggregate_right_specialized16','select_right_specialized8','select_right_specialized16')


def mirror_rows(raw,width,height):
    if type(width) is not int or type(height) is not int or min(width,height)<=0 or len(raw)!=width*height*4:raise ValueError('invalid mirror plane extent')
    result=bytearray(len(raw))
    for y in range(height):
        row=y*width*4
        for x in range(width):result[row+x*4:row+(x+1)*4]=raw[row+(width-1-x)*4:row+(width-x)*4]
    return bytes(result)


def native_identities(rhi,daxa,compiler):
    # Bind all21 old entries as immutable controls, and all4 right entries.
    identity=base.native_identities(rhi,daxa,compiler,True,True,True,True,True)
    directory=Path(identity['daxa']['offline_shader_directory'])
    for entry in RIGHT_ENTRIES:identity['daxa']['offline_spirv_sha256'][entry]=base.sha256(directory/('daxa-stereo-'+entry+'.spv'))
    identity['right_harness_sha256']=base.sha256(__file__)
    for name in ('src/daxa_frame.cpp','include/stereoforge/daxa_frame.hpp','include/stereoforge/daxa_preprocess.hpp','include/stereoforge/capture.hpp','include/stereoforge/contracts.hpp','cmake/Daxa.cmake','CMakeLists.txt'):
        identity['source_sha256'][name]=base.sha256(base.ROOT/name)
    return identity


def validate_bounds(profile,frames,warmup,slots,wave_size,seed,validation_only):
    if set(profile)!=set(base.PROFILE_FIELDS) or any(type(profile[k]) is not int for k in base.PROFILE_FIELDS if k!='subpixel') or type(profile['subpixel']) is not bool:raise ValueError('typed complete profile required')
    if not 5<=profile['width']<=1280 or not 5<=profile['height']<=720 or profile['disparities'] not in ((16,32,64,128) if validation_only else (32,64,128)) or profile['paths'] not in (4,8) or not 0<=profile['p1']<=profile['p2']<=10000:raise ValueError('unsupported bounded right profile')
    if any(type(x) is not int for x in (frames,warmup,slots,wave_size,seed)) or not 1<=frames<=300 or not 0<=warmup<=30 or slots not in (1,2) or wave_size not in (32,64) or not 0<seed<=0xffffffff or (not validation_only and (frames<30 or warmup<3)):raise ValueError('bounded counts/profile required; smaller counts only validation')
    base.packed_contract(profile)


def commands(rhi,daxa,output,profile,frames,warmup,slots,wave_size,validation,intermediates):
    shared=[]
    for key,value in profile.items():shared+=['--'+key,str(int(value)) if type(value) is bool else str(value)]
    rhi_command=[str(rhi),*shared,'--left',str(output/'mirror-right.u32'),'--right',str(output/'mirror-left.u32'),'--iterations',str(frames),'--output',str(output/'rhi-mirrored'),'--shader-directory',str((base.ROOT/'python/stereoforge/shaders/stereo').resolve())]
    daxa_command=[str(daxa),*shared,'--left',str(output/'left.u32'),'--right',str(output/'right.u32'),'--reference','right','--variant','packed-specialized','--frames',str(frames),'--warmup',str(warmup),'--slots',str(slots),'--wave-size',str(wave_size),'--check-integer-reference','--output',str(output/'right.json'),'--dump-prefix',str(output/'right')]
    if intermediates:daxa_command.append('--dump-intermediates')
    if validation:rhi_command.append('--validation');daxa_command.append('--validation')
    return {'rhi':rhi_command,'right':daxa_command}


def validate_report(report,profile,identity,frames,warmup,slots,wave_size):
    base.validate_profile(report,profile,'packed-specialized',frames,warmup,slots,wave_size)
    base.validate_daxa_timing(report);base.validate_packed_report(report,profile,True)
    expected=dict(reference='right',scope='single_reference_retained_luma_no_lr_consistency',integer_reference_checked=True,packed_penalties_reference_checked=True,compound_float_reference_checked=False,metric_depth=False)
    for key,value in expected.items():
        if type(report.get(key)) is not type(value) or report[key]!=value:raise ValueError('missing/changed right contract: '+key)
    device=report.get('device',{})
    for row in (report,device):
        for key in (('validation_errors_after_teardown','validation_warnings_after_teardown') if row is report else ('validation_errors','validation_warnings')):
            if type(row.get(key)) is not int or row[key]!=0:raise ValueError('right validation diagnostics')
    if device.get('slangc_sha256')!=identity['daxa']['compiler_sha256']:raise ValueError('actual right compiler mismatch')
    if report.get('shader_source_sha256_pair')!=':'.join(identity['daxa']['shader_source_sha256'].values()):raise ValueError('actual right shader source mismatch')


def run_comparison(rhi,daxa,compiler,output,profile,frames=30,warmup=3,slots=1,wave_size=32,seed=1234567,validation=False,validation_only=False,intermediates=False):
    validate_bounds(profile,frames,warmup,slots,wave_size,seed,validation_only)
    output=Path(output).absolute();base.create_output(output)
    fixture=base.write_fixture(output,profile,seed)
    originals={eye:output/(eye+'.u32') for eye in ('left','right')}
    derived={}
    for eye in ('right','left'):
        path=output/('mirror-'+eye+'.u32');path.write_bytes(mirror_rows(originals[eye].read_bytes(),profile['width'],profile['height']));path.chmod(0o444);derived[path.name]=base.sha256(path)
    fixture['derived_sha256']=derived;fixture['right_harness_sha256']=base.sha256(__file__)
    fixture['oracle_mapping']='RHI left=mirror(original right), RHI right=mirror(original left); unmirror final output rows only; disparity remains positive x_left-x_right'
    def verify_inputs():
        base.verify_fixture(originals,fixture)
        for name,digest in derived.items():
            if base.sha256(output/name)!=digest:raise ValueError('mirrored source bytes changed')
    evidence=dict(schema=1,passed=False,selected=None,reference='right',variant='packed-specialized',profile=profile,fixture=fixture,frames=frames,warmup=warmup,slots=slots,wave_size=wave_size,validation_requested=validation,performance_qualified_counts=not validation_only,
        scope='Right-reference only, positive x_left-x_right; no left/right consistency, calibrated depth, capture or full-pipeline/60Hz qualification.',
        timing_contract=dict(daxa='Retained original luma; GPU graph/stages, upload and diagnostics outside samples.',rhi='Mirrored inputs; aggregate transfer-inclusive iterations, no separate warmup available.',rhi_warmup='unsupported; none fabricated',comparable_end_to_end=False),
        inherited_validation_environment={k:os.environ[k] for k in ('VK_INSTANCE_LAYERS','VK_LAYER_VALIDATE_SYNC','VK_LAYER_ENABLES','VK_LOADER_DEBUG','SF_DAXA_VALIDATION') if k in os.environ},children={})
    try:
        identity=native_identities(rhi,daxa,compiler);evidence['identities']=identity
        calls=commands(rhi,daxa,output,profile,frames,warmup,slots,wave_size,validation,intermediates);evidence['commands']=calls
        report_hashes={}
        for name,command in calls.items():
            verify_inputs();child=subprocess.run(command,capture_output=True,text=True,errors='replace')
            (output/(name+'.stdout')).write_text(child.stdout);(output/(name+'.stderr')).write_text(child.stderr)
            evidence['children'][name]=dict(returncode=child.returncode)
            base.require_child_clean(child.returncode,child.stdout,child.stderr);verify_inputs()
            if name=='rhi':
                report=base.strict_json(child.stdout);base.validate_rhi_profile(report,profile,frames)
                if report.get('vulkan_validation') is not validation or report.get('metric_depth') is not False:raise ValueError('changed RHI validation/metric profile')
                elapsed=report.get('transfer_inclusive_total_ms')
                if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<=0:raise ValueError('missing RHI timing')
                (output/'rhi.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            else:
                report=base.strict_json((output/'right.json').read_text());validate_report(report,profile,identity,frames,warmup,slots,wave_size)
                if intermediates:evidence['actual_gpu_packed_words']=base.packed_word_identity(output/'right',base.packed_contract(profile))
            report_hashes[name]=base.sha256(output/(name+'.json'));evidence['children'][name]['report']=report
        if evidence['children']['rhi']['report'].get('adapter')!=evidence['children']['right']['report']['device'].get('device'):raise ValueError('RHI/right adapter names differ')
        evidence['physical_device_boundary']='Daxa queried UUID/driver; frozen RHI adapter name only, not independent UUID equality.'
        unmirrored={};mirrored={}
        for plane in base.OUTPUT_PLANES:
            source=Path(str(output/'rhi-mirrored')+'.'+plane);raw=source.read_bytes();mirrored[plane]=base.sha256(source)
            destination=Path(str(output/'rhi-unmirrored')+'.'+plane);destination.write_bytes(mirror_rows(raw,profile['width'],profile['height']));unmirrored[plane]=base.sha256(destination)
        evidence['rhi_mirrored_output_sha256']=mirrored;evidence['rhi_unmirrored_sha256']=unmirrored
        evidence['comparison']=base.compare_planes(output/'rhi-unmirrored',output/'right',profile)
        evidence['integer_word_reference_scope']='Native independent scalar integer and actual packed-word oracle; never inferred from reconstructed GPU cost/sum volumes.'
        evidence['intermediate_scope']='GPU packed intermediates, when dumped, are RECONSTRUCTED; raw mirrored Census bits are not a parity oracle.'
        verify_inputs()
        if identity!=native_identities(rhi,daxa,compiler):raise ValueError('source/compiler/binary/dependency/SPIRV identity changed')
        if any(base.sha256(output/(name+'.json'))!=digest for name,digest in report_hashes.items()):raise ValueError('report bytes changed during comparison')
        evidence['report_sha256']=report_hashes;evidence['passed']=evidence['comparison']['passed']
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as error:evidence['error']=str(error)
    (output/'comparison.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name,default in (('rhi-binary',base.ROOT/'build/standalone/native_stereo'),('daxa-binary',base.ROOT/'build/daxa/daxa_stereo_benchmark')):parser.add_argument('--'+name,type=Path,default=default)
    parser.add_argument('--compiler',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    for name,default in (('width',1280),('height',720),('disparities',128),('paths',4),('p1',8),('p2',32),('subpixel',1),('slots',1),('wave-size',32),('seed',1234567)):parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--frames',type=int);parser.add_argument('--warmup',type=int)
    for name in ('validation','validation-only','dump-intermediates'):parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.subpixel not in (0,1):parser.error('subpixel must be0/1')
    profile={k:getattr(args,k) for k in base.PROFILE_FIELDS};profile['subpixel']=bool(args.subpixel)
    try:result=run_comparison(args.rhi_binary.resolve(),args.daxa_binary.resolve(),args.compiler.resolve(),args.output,profile,args.frames if args.frames is not None else 1 if args.validation_only else 30,args.warmup if args.warmup is not None else 0 if args.validation_only else 3,args.slots,args.wave_size,args.seed,args.validation,args.validation_only,args.dump_intermediates)
    except ValueError as error:parser.error(str(error))
    print(json.dumps(dict(passed=result['passed'],output=str(args.output))))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
