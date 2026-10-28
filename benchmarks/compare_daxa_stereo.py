#!/usr/bin/env python3
"""Bounded exact RHI/Daxa stereo comparison on the same explicit uint32 luma.

RHI exposes only transfer-inclusive aggregate iterations. Daxa stage/resident
measurements exclude source upload and diagnostic export; they are not matched
end-to-end timing. No full stereo/LR/capture or 16.67 ms pipeline claim.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUTPUT_PLANES=('disparity.f32','confidence.f32','validity.u32')
INTERMEDIATE_PLANES=('costs.u32','sums.u32')
PACKED_VARIANTS=('packed','packed-specialized','packed-d128','packed-quad','packed-shuffle')
PROFILE_FIELDS=('width','height','disparities','paths','p1','p2','subpixel')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def strict_json(text):
    def constant(value):raise ValueError('nonfinite JSON constant: '+value)
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    value=json.loads(text,parse_constant=constant,object_pairs_hook=pairs)
    def check(item):
        if isinstance(item,float) and not math.isfinite(item):raise ValueError('nonfinite JSON number')
        if isinstance(item,dict):
            for child in item.values():check(child)
        elif isinstance(item,list):
            for child in item:check(child)
    check(value)
    if not isinstance(value,dict):raise ValueError('report must be a JSON object')
    return value


def create_output(path):
    path=Path(path)
    if path.exists() or path.is_symlink():raise ValueError('output must be a fresh nonexisting directory')
    path.mkdir(parents=True,exist_ok=False)


def write_fixture(directory,profile,seed):
    """One source of bytes, passed by file to every child; no implicit generators."""
    width,height=profile['width'],profile['height'];left=[];state=seed
    for _ in range(width*height):
        state=(state^(state<<13))&0xffffffff;state=(state^(state>>17))&0xffffffff;state=(state^(state<<5))&0xffffffff
        left.append(state&255)
    right=[left[y*width+x+4] if x+4<width else 0 for y in range(height) for x in range(width)]
    for name,values in (('left',left),('right',right)):
        path=Path(directory)/(name+'.u32')
        with path.open('xb') as output:
            for begin in range(0,len(values),4096):output.write(struct.pack('<'+str(len(values[begin:begin+4096]))+'I',*values[begin:begin+4096]))
        path.chmod(0o444)
    return dict(layout='little-endian uint32 luma, values0..255',generator='xorshift32; right(x,y)=left(x+4,y), zero right border',
        seed=seed,source_sha256=sha256(__file__),width=width,height=height,
        left_sha256=sha256(Path(directory)/'left.u32'),right_sha256=sha256(Path(directory)/'right.u32'))


def verify_fixture(paths,manifest):
    for eye in ('left','right'):
        if sha256(paths[eye])!=manifest[eye+'_sha256']:raise ValueError('shared luma bytes changed: '+eye)


def require_child_clean(returncode,stdout,stderr):
    from compare_daxa_view import parse_diagnostics
    if returncode!=0 or parse_diagnostics(stdout)['line_count'] or parse_diagnostics(stderr)['line_count']:
        raise ValueError('child failed or emitted validation/driver diagnostics')


def validate_profile(report,profile,variant,frames,warmup,slots,wave_size=0):
    for key in PROFILE_FIELDS:
        if type(report.get(key)) is not type(profile[key]) or report[key]!=profile[key]:raise ValueError('changed Daxa parameter: '+key)
    actual_wave=0 if variant=='baseline' else wave_size or report.get('device',{}).get('subgroups',{}).get('default_size')
    if variant in ('wave',*PACKED_VARIANTS) and actual_wave not in (32,64):raise ValueError('unsupported actual subgroup size')
    expected=dict(variant=variant,frames=frames,warmup=warmup,slots=slots,metric_depth=False,required_wave_size=actual_wave,full_subgroups_required=variant in ('wave',*PACKED_VARIANTS))
    for key,value in expected.items():
        if type(report.get(key)) is not type(value) or report[key]!=value:raise ValueError('changed Daxa parameter: '+key)


def packed_contract(profile):
    """The native selected width is bounded by the sum of path penalties."""
    bound=profile['paths']*profile['p2']
    if not 0<=bound<=65535:raise ValueError('packed path penalty exceeds16-bit bound')
    bits=8 if bound<=255 else 16
    words=(profile['disparities']+32//bits-1)//(32//bits)
    return dict(packing_bits=bits,packed_penalty_layout='pixel_major_disparity_lsb_u32',
        packed_penalty_bound=bound,packed_penalty_words_per_pixel=words,
        packed_penalty_word_count=profile['width']*profile['height']*words)


def validate_packed_report(report,profile,reconstructed):
    expected=dict(packed_contract(profile),diagnostic_intermediates_reconstructed=reconstructed)
    for key,value in expected.items():
        if type(report.get(key)) is not type(value) or report[key]!=value:raise ValueError('changed packed contract: '+key)
    return packed_contract(profile)


def packed_word_identity(prefix,contract):
    path=Path(str(prefix)+'.packed_penalties.u32');expected=contract['packed_penalty_word_count']*4
    if not path.is_file() or path.stat().st_size!=expected:raise ValueError('missing/incomplete actual packed penalty words')
    return dict(representation='actual GPU packed penalty words',bytes=expected,sha256=sha256(path))


def validate_rhi_profile(report,profile,iterations):
    # Frozen RHI source accepts all seven parameters, but echoes only four.
    # Unreported parameters remain explicit immutable command evidence.
    for key in ('width','height','disparities','paths'):
        if type(report.get(key)) is not int or report[key]!=profile[key]:raise ValueError('changed RHI parameter: '+key)
    if type(report.get('iterations')) is not int or report['iterations']!=iterations:raise ValueError('changed RHI iteration count')


def compare_planes(a,b,profile,planes=OUTPUT_PLANES):
    compared={}
    for plane in planes:
        expected=profile['width']*profile['height']*4*(profile['disparities'] if plane in INTERMEDIATE_PLANES else 1)
        first,second=Path(str(a)+'.'+plane),Path(str(b)+'.'+plane)
        if not first.is_file() or not second.is_file() or first.stat().st_size!=expected or second.stat().st_size!=expected:
            raise ValueError('missing/incomplete plane: '+plane)
        hashes=[sha256(first),sha256(second)]
        compared[plane]=dict(bytes=expected,sha256=hashes,bytes_equal=hashes[0]==hashes[1])
    return dict(passed=all(p['bytes_equal'] for p in compared.values()),planes=compared)


def commands(rhi,daxa,output,profile,paths,frames,warmup,slots,validation,intermediates,check_reference=False,wave_size=0,include_packed=False,check_integer_reference=False,include_specialized=False,include_d128=False,include_quad=False,include_shuffle=False):
    if include_d128 and profile['disparities']!=128:raise ValueError('packed-d128 requires D128')
    include_specialized=include_specialized or include_d128 or include_quad or include_shuffle
    shared=[]
    for key,value in profile.items():shared+=['--'+key,str(int(value)) if type(value) is bool else str(value)]
    shared+=['--left',str(paths['left']),'--right',str(paths['right'])]
    result={'rhi':[str(rhi),*shared,'--iterations',str(frames),'--output',str(output/'rhi')]}
    variants=('baseline','wave')+(('packed',) if include_packed or include_specialized else ())+(('packed-specialized',) if include_specialized else ())+(('packed-d128',) if include_d128 else ())+(('packed-quad',) if include_quad else ())+(('packed-shuffle',) if include_shuffle else ())
    for variant in variants:
        result[variant]=[str(daxa),*shared,'--variant',variant,'--frames',str(frames),'--warmup',str(warmup),
            '--slots',str(slots),'--output',str(output/(variant+'.json')),'--dump-prefix',str(output/variant)]
        if variant in ('wave',*PACKED_VARIANTS):result[variant]+=['--wave-size',str(wave_size)]
        if intermediates:result[variant].append('--dump-intermediates')
        if check_reference:result[variant].append('--check-reference')
        if check_integer_reference:result[variant].append('--check-integer-reference')
    if validation:
        for command in result.values():command.append('--validation')
    return result


def validate_daxa_timing(report):
    for key in ('gpu_graph_ms','cpu_submit_ms','cpu_wait_ms','cpu_timestamp_query_ms','submit_wait_and_query_wall_ms'):
        value=report.get(key)
        if not isinstance(value,dict) or set(value)!={'p50','p95','mean'} or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in value.values()):
            raise ValueError('missing/invalid Daxa timing: '+key)
    if not isinstance(report.get('stages'),dict) or not report['stages']:raise ValueError('missing GPU stage timings')
    for name,value in report['stages'].items():
        if not isinstance(value,dict) or set(value)!={'p50','p95','mean'} or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in value.values()):
            raise ValueError('invalid GPU stage timing: '+name)


def native_identities(rhi,daxa,compiler,include_packed=False,include_specialized=False,include_d128=False,include_quad=False,include_shuffle=False):
    from compare_daxa_view import dependencies
    include_specialized=include_specialized or include_d128 or include_quad or include_shuffle
    directory=Path(subprocess.check_output([str(daxa),'--shader-directory'],text=True).strip()).resolve(strict=True)
    entries=('census','cost','aggregate_baseline','aggregate_wave','select_output','depth_convert','inspect')
    if include_packed or include_specialized:entries+=('clear_packed','aggregate_packed8','aggregate_packed16','select_packed')
    if include_specialized:entries+=('aggregate_specialized8','aggregate_specialized16','select_specialized8','select_specialized16')
    if include_d128:entries+=('aggregate_d1288','aggregate_d12816')
    if include_quad:entries+=('select_quad8','select_quad16')
    if include_shuffle:entries+=('aggregate_shuffle8','aggregate_shuffle16')
    spirv={entry:sha256(directory/('daxa-stereo-'+entry+'.spv')) for entry in entries}
    rhi_shader=ROOT/'python/stereoforge/shaders/stereo/sgm.slang'
    sources={name:sha256(ROOT/name) for name in ('apps/native_stereo.cpp','src/runtime.cpp','src/native_device.cpp','apps/daxa_stereo_benchmark.cpp','src/daxa_stereo.cpp','include/stereoforge/daxa_stereo.hpp','cmake/DaxaStereo.cmake')}
    return dict(source_sha256=sources,harness_sha256=sha256(__file__),rhi=dict(binary_sha256=sha256(rhi),dependency_sha256=dependencies(rhi,False)['sha256'],
        shader_source_sha256=sha256(rhi_shader),spirv_identity='unavailable: frozen RHI compiles source at runtime'),
        daxa=dict(binary_sha256=sha256(daxa),dependency_sha256=dependencies(daxa,True)['sha256'],offline_spirv_sha256=spirv,
        offline_shader_directory=str(directory),compiler_sha256=sha256(compiler),
        shader_source_sha256={name:sha256(ROOT/name) for name in ('python/stereoforge/shaders/daxa/stereo.slang','python/stereoforge/shaders/ops/sgm.slang')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rhi-binary',type=Path,default=ROOT/'build/standalone/native_stereo')
    p.add_argument('--daxa-binary',type=Path,default=ROOT/'build/daxa/daxa_stereo_benchmark')
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    for name,default in (('width',1280),('height',720),('disparities',128),('paths',4),('p1',8),('p2',32),('subpixel',1),('slots',1),('seed',1234567)):
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--frames',type=int);p.add_argument('--warmup',type=int)
    p.add_argument('--wave-size',type=int,choices=[0,32,64],default=0)
    p.add_argument('--validation-only',action='store_true',help='Allows explicitly smaller validation counts; never performance-qualified')
    p.add_argument('--validation',action='store_true');p.add_argument('--dump-intermediates',action='store_true')
    p.add_argument('--check-reference',action='store_true',help='Also run Daxa scalar reference after timing; potentially expensive')
    p.add_argument('--include-packed',action='store_true',help='Also compare the bounded packed penalty candidate; no automatic adoption')
    p.add_argument('--include-specialized',action='store_true',help='Also compare packed-specialized; implies --include-packed')
    p.add_argument('--include-d128',action='store_true',help='Also compare packed-d128; requires D128 and implies specialized/packed')
    p.add_argument('--include-quad',action='store_true',help='Also compare packed-quad; implies specialized/packed without requiring D128')
    p.add_argument('--include-shuffle',action='store_true',help='Also compare packed-shuffle; implies specialized/packed, not quad/D128')
    p.add_argument('--check-integer-reference',action='store_true',help='Run exact integer oracle after timing, including direct packed words; excludes compound float oracle')
    a=p.parse_args();frames=a.frames if a.frames is not None else (1 if a.validation_only else 30);warmup=a.warmup if a.warmup is not None else (0 if a.validation_only else 3)
    if min(a.width,a.height)<5 or a.width*a.height>1280*720 or a.disparities not in ((16,32,64,96,128) if a.validation_only else (32,64,96,128)) or a.paths not in (4,8) or not 0<=a.p1<=a.p2<=10000 or a.subpixel not in (0,1) or a.slots not in (1,2) or not 0<a.seed<=0xffffffff:
        p.error('bounded profile requires <=720p, D32/64/96/128 (D16 also in validation-only), paths4/8, valid penalties/subpixel/slots/seed')
    if not 1<=frames<=300 or not 0<=warmup<=30 or (not a.validation_only and (frames<30 or warmup<3)):
        p.error('performance requires30..300samples/3..30warmups; smaller counts require --validation-only')
    profile={key:getattr(a,key) for key in PROFILE_FIELDS};profile['subpixel']=bool(a.subpixel)
    if a.include_d128 and a.disparities!=128:p.error('packed-d128 requires D128')
    a.include_specialized=a.include_specialized or a.include_d128 or a.include_quad or a.include_shuffle
    a.include_packed=a.include_packed or a.include_specialized
    if a.include_packed:
        try:packed_contract(profile)
        except ValueError as error:p.error(str(error))
    output=a.output.absolute();create_output(output)
    manifest=write_fixture(output,profile,a.seed);paths={eye:output/(eye+'.u32') for eye in ('left','right')}
    rhi,daxa=a.rhi_binary.resolve(),a.daxa_binary.resolve()
    evidence=dict(schema=1,profile=profile,fixture=manifest,frames=frames,warmup=warmup,
        performance_qualified_counts=not a.validation_only,include_packed=a.include_packed,include_specialized=a.include_specialized,include_d128=a.include_d128,include_quad=a.include_quad,include_shuffle=a.include_shuffle,integer_reference_requested=a.check_integer_reference,wave_size_requested=a.wave_size,validation_requested=a.validation,selected=None,
        inherited_validation_environment={name:os.environ[name] for name in ('VK_INSTANCE_LAYERS','VK_LAYER_VALIDATE_SYNC','VK_LAYER_ENABLES','VK_LOADER_DEBUG','SF_DAXA_VALIDATION') if name in os.environ},
        scope='Single reference stereo only: no second reference, LR consistency, capture, calibrated depth or full-pipeline/60Hz qualification.',
        timing_contract=dict(daxa='Retained luma GPU graph/stage timestamps; upload and diagnostic export outside samples.',
            rhi='Aggregate transfer-inclusive iterations with upload/output readback; no separate warmup or per-iteration timing available.',
            comparable_end_to_end=False,rhi_warmup='unsupported; none fabricated'),
        comparisons={},children={},passed=False)
    try:
        identities=native_identities(rhi,daxa,a.compiler,a.include_packed,a.include_specialized,a.include_d128,a.include_quad,a.include_shuffle)
        evidence['identities']=identities
        calls=commands(rhi,daxa,output,profile,paths,frames,warmup,a.slots,a.validation,a.dump_intermediates,a.check_reference,a.wave_size,a.include_packed,a.check_integer_reference,a.include_specialized,a.include_d128,a.include_quad,a.include_shuffle)
        # Explicit source path prevents stale compiled-in RHI shader-directory defaults.
        calls['rhi']+=['--shader-directory',str((ROOT/'python/stereoforge/shaders/stereo').resolve())]
        evidence['commands']=calls
        for case,command in calls.items():
            verify_fixture(paths,manifest)
            child=subprocess.run(command,capture_output=True,text=True,errors='replace')
            (output/(case+'.stdout')).write_text(child.stdout);(output/(case+'.stderr')).write_text(child.stderr)
            evidence['children'][case]=dict(returncode=child.returncode)
            require_child_clean(child.returncode,child.stdout,child.stderr)
            verify_fixture(paths,manifest)
            report=strict_json(child.stdout if case=='rhi' else (output/(case+'.json')).read_text())
            if case=='rhi':
                validate_rhi_profile(report,profile,frames)
                if report.get('vulkan_validation') is not a.validation:raise ValueError('RHI validation request not honored')
                if report.get('metric_depth') is not False:raise ValueError('unexpected metric-depth profile')
                if type(report.get('transfer_inclusive_total_ms')) not in (int,float) or report['transfer_inclusive_total_ms']<=0:raise ValueError('missing RHI aggregate timing')
                (output/'rhi.json').write_text(json.dumps(report,indent=2)+'\n')
            else:
                validate_profile(report,profile,case,frames,warmup,a.slots,a.wave_size);validate_daxa_timing(report)
                if report.get('scope')!='single_reference_retained_luma_no_lr_consistency':raise ValueError('changed Daxa timing scope')
                if a.check_reference and report.get('scalar_reference_checked') is not True:raise ValueError('requested scalar parity not honored')
                if a.check_integer_reference and report.get('integer_reference_checked') is not True:raise ValueError('requested integer parity not honored')
                if case in PACKED_VARIANTS:
                    prefix=case.replace('-','_')
                    evidence[prefix+'_contract']=validate_packed_report(report,profile,a.dump_intermediates or a.check_reference or a.check_integer_reference)
                    if a.check_integer_reference and report.get('packed_penalties_reference_checked') is not True:raise ValueError('requested direct packed-word parity not honored')
                    evidence[prefix+'_word_reference_checked']=report.get('packed_penalties_reference_checked') is True
                    evidence[prefix+'_word_reference_scope']='Native independent scalar packing comparison; never inferred from reconstructed volumes'
                    if a.dump_intermediates:evidence[prefix+'_word_dump']=packed_word_identity(output/case,evidence[prefix+'_contract'])
                if report.get('validation_errors_after_teardown')!=0 or report.get('validation_warnings_after_teardown')!=0:raise ValueError('Daxa teardown validation diagnostics')
                device=report['device']
                if device.get('validation_errors')!=0 or device.get('validation_warnings')!=0:raise ValueError('Daxa validation diagnostics')
                if device.get('slangc_sha256')!=identities['daxa']['compiler_sha256']:raise ValueError('actual compiler bytes mismatch')
                expected=':'.join(identities['daxa']['shader_source_sha256'].values())
                if report.get('shader_source_sha256_pair')!=expected:raise ValueError('compiled shader source identity mismatch')
            evidence['children'][case]['report']=report
        before_device=evidence['children']['baseline']['report']['device']
        after_device=evidence['children']['wave']['report']['device']
        if before_device!=after_device:raise ValueError('Daxa baseline/wave queried device identity changed')
        for candidate in PACKED_VARIANTS:
            if candidate in evidence['children'] and evidence['children'][candidate]['report']['device']!=before_device:raise ValueError('Daxa '+candidate+' queried device identity changed')
        if evidence['children']['rhi']['report'].get('adapter')!=before_device.get('device'):
            raise ValueError('RHI/Daxa adapter names differ; matching physical execution not established')
        evidence['physical_device_boundary']='Daxa binds queried UUID/driver; frozen RHI exposes adapter name only. Matching names do not independently prove UUID equality.'
        evidence['comparisons']['rhi_vs_baseline']=compare_planes(output/'rhi',output/'baseline',profile)
        evidence['comparisons']['rhi_vs_wave']=compare_planes(output/'rhi',output/'wave',profile)
        planes=OUTPUT_PLANES+(INTERMEDIATE_PLANES if a.dump_intermediates else ())
        evidence['comparisons']['baseline_vs_wave']=compare_planes(output/'baseline',output/'wave',profile,planes)
        if a.include_packed:
            evidence['comparisons']['rhi_vs_packed']=compare_planes(output/'rhi',output/'packed',profile)
            evidence['comparisons']['baseline_vs_packed']=compare_planes(output/'baseline',output/'packed',profile)
            if a.dump_intermediates:
                row=compare_planes(output/'baseline',output/'packed',profile,INTERMEDIATE_PLANES)
                row['packed_representation']='RECONSTRUCTED from actual GPU Census and packed penalty words; not dense GPU buffers'
                evidence['comparisons']['baseline_vs_packed_reconstructed']=row
        if a.include_specialized:
            for other in ('rhi','baseline','packed'):
                evidence['comparisons'][other+'_vs_packed_specialized']=compare_planes(output/other,output/'packed-specialized',profile)
            if a.dump_intermediates:
                for other in ('baseline','packed'):
                    row=compare_planes(output/other,output/'packed-specialized',profile,INTERMEDIATE_PLANES)
                    row['packed_representation']='RECONSTRUCTED from actual GPU Census and packed penalty words; not dense GPU buffers'
                    evidence['comparisons'][other+'_vs_packed_specialized_reconstructed']=row
        if a.include_d128:
            for other in ('rhi','baseline','packed','packed-specialized'):
                evidence['comparisons'][other+'_vs_packed_d128']=compare_planes(output/other,output/'packed-d128',profile)
            if a.dump_intermediates:
                for other in ('baseline','packed','packed-specialized'):
                    row=compare_planes(output/other,output/'packed-d128',profile,INTERMEDIATE_PLANES)
                    row['packed_representation']='RECONSTRUCTED from actual GPU Census and packed penalty words; not dense GPU buffers'
                    evidence['comparisons'][other+'_vs_packed_d128_reconstructed']=row
        if a.include_quad:
            for other in ('rhi','baseline','packed','packed-specialized')+(('packed-d128',) if a.include_d128 else ()):
                evidence['comparisons'][other+'_vs_packed_quad']=compare_planes(output/other,output/'packed-quad',profile)
            if a.dump_intermediates:
                for other in ('baseline','packed-specialized'):
                    row=compare_planes(output/other,output/'packed-quad',profile,INTERMEDIATE_PLANES)
                    row['packed_representation']='RECONSTRUCTED from actual GPU Census and packed penalty words; not dense GPU buffers'
                    evidence['comparisons'][other+'_vs_packed_quad_reconstructed']=row
        if a.include_shuffle:
            for other in ('rhi','baseline','packed','packed-specialized')+(('packed-d128',) if a.include_d128 else ())+(('packed-quad',) if a.include_quad else ()):
                evidence['comparisons'][other+'_vs_packed_shuffle']=compare_planes(output/other,output/'packed-shuffle',profile)
            if a.dump_intermediates:
                for other in ('baseline','packed-specialized'):
                    row=compare_planes(output/other,output/'packed-shuffle',profile,INTERMEDIATE_PLANES)
                    row['packed_representation']='RECONSTRUCTED from actual GPU Census and packed penalty words; not dense GPU buffers'
                    evidence['comparisons'][other+'_vs_packed_shuffle_reconstructed']=row
        evidence['rhi_intermediate_parity']='unavailable: frozen RHI exposes only final disparity/confidence/validity dumps'
        after=native_identities(rhi,daxa,a.compiler,a.include_packed,a.include_specialized,a.include_d128,a.include_quad,a.include_shuffle)
        if identities!=after:raise ValueError('binary/compiler/shader/dependency bytes changed during comparison')
        evidence['passed']=all(row['passed'] for row in evidence['comparisons'].values())
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as error:
        evidence['error']=str(error)
    (output/'comparison.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),passed=evidence['passed'])))
    if not evidence['passed']:raise SystemExit(1)


if __name__=='__main__':main()
