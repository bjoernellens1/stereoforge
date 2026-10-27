#!/usr/bin/env python3
"""Bounded offline metadata comparison. Writes evidence, never promotes a winner.

Manual approval is an explicit config edit binding one validated measurement hash
and its complete execution identity. No runtime policy learning or cache fallback.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/kernel-variants.json'
IDENTITY_FIELDS = {'physical_gpu','driver','daxa','slang','compiler_sha256','shader_sha256',
    'specialization','source_width','source_height','target_width','target_height',
    'disparities','paths','precision','quality_profile','subgroup'}
CURATED = [dict(name='metadata-fused-64',entrypoints=['metadata_fused'],workgroup=[64,1,1]),
    dict(name='metadata-split-64',entrypoints=['metadata_ids','metadata_ages'],workgroup=[64,1,1])]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path=CONFIG):
    config=json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config):
    if config.get('schema')!=1 or config.get('operation')!='resident-output-metadata' or config.get('variants')!=CURATED:
        raise ValueError('unknown variant or changed compiled variant contract')
    if set(config.get('contract',{}))!={'grid','units','precision','border','layout','alias','ownership'}:
        raise ValueError('operation contract incomplete')
    q=config.get('quality',{})
    if q.get('profile')!='metadata-exact-v1' or q.get('minimum_samples',0)<30 or q.get('minimum_warmup',0)<3:
        raise ValueError('frozen quality profile incomplete')
    for name in ('maximum_allocated_bytes','maximum_gpu_ms','maximum_transfer_inclusive_ms'):
        if not isinstance(q.get(name),(int,float)) or not math.isfinite(q[name]) or q[name]<=0:
            raise ValueError('finite positive resource ceilings required')


def validate_identity(identity):
    if set(identity)!=IDENTITY_FIELDS:
        raise ValueError('incomplete or unknown execution identity fields')
    for field,value in identity.items():
        if field in ('disparities','paths'):
            if type(value) is not int or value<0: raise ValueError('invalid D/path identity')
        elif field.endswith(('width','height')) or field=='subgroup':
            if type(value) is not int or value<=0: raise ValueError('invalid shape/subgroup')
        elif not isinstance(value,str) or not value.strip():
            raise ValueError('empty execution identity')
    for field in ('compiler_sha256','shader_sha256'):
        value=identity[field]
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value): raise ValueError('invalid actual byte hash')


def validate_cache(stored,current):
    validate_identity(stored); validate_identity(current)
    if stored!=current: raise ValueError('stale execution identity; remeasure and manually review')


def validate_comparable(first,second):
    validate_identity(first);validate_identity(second)
    try:
        first_memory=json.loads(first['specialization'])['readback_memory']
        second_memory=json.loads(second['specialization'])['readback_memory']
    except (ValueError,KeyError,TypeError) as error:
        raise ValueError('incomplete specialization memory identity') from error
    if first_memory!=second_memory:raise ValueError('variants have different execution profiles for readback memory')
    fields=IDENTITY_FIELDS-{'shader_sha256','specialization'}
    if any(first[field]!=second[field] for field in fields):
        raise ValueError('variants have different execution profiles')


def validate_subgroup(size,minimum,maximum,stages):
    if type(size) is not int or size<=0 or size&(size-1) or not minimum<=size<=maximum or not stages&32:
        raise ValueError('unsupported compute subgroup size')


def memory_specialization(report):
    policy=report.get('readback_memory_policy');index=report.get('readback_memory_type_index');flags=report.get('readback_memory_property_flags')
    if policy not in ('first-coherent','cached-coherent') or type(index) is not int or not 0<=index<32 or type(flags) is not int or not 0<=flags<=2**32-1:
        raise ValueError('incomplete readback memory policy/type identity')
    required=14 if policy=='cached-coherent' else 6
    if flags&required!=required:raise ValueError('readback memory lacks required coherent/cached properties')
    return dict(policy=policy,type_index=index,property_flags=flags)


def validate_phase_rows(rows,count,totals=None):
    fields={'input_stage','enqueue','completion_wait','readback_copy','query','total'}
    if not isinstance(rows,list) or len(rows)!=count:raise ValueError('incomplete CPU phase rows')
    if totals is not None and (not isinstance(totals,list) or len(totals)!=count or any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in totals)):
        raise ValueError('incomplete or invalid phase wall samples')
    for i,row in enumerate(rows):
        if not isinstance(row,dict) or set(row)!=fields or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in row.values()):
            raise ValueError('invalid CPU phase row')
        if not math.isclose(sum(row[name] for name in fields-{'total'}),row['total'],rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('CPU phases do not account for complete wall time')
        if totals is not None and not math.isclose(row['total'],totals[i],rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('CPU phase total differs from reported wall sample')


def validate_measurement(report,quality):
    memory_specialization(report)
    if report.get('numeric_parity') is not True or report.get('validation_errors')!=0 or report.get('validation_warnings')!=0:
        raise ValueError('exact parity or clean validation failed')
    if report.get('warmup',0)<quality['minimum_warmup']:
        raise ValueError('insufficient warmup')
    allocated=report.get('allocated_bytes')
    if type(allocated) is not int or not 0<allocated<=quality['maximum_allocated_bytes']:
        raise ValueError('resource ceiling failed')
    for field,ceiling in [('gpu_ms_samples','maximum_gpu_ms'),('transfer_inclusive_ms_samples','maximum_transfer_inclusive_ms')]:
        samples=report.get(field,[])
        if len(samples)<quality['minimum_samples'] or any(type(x) not in (float,int) or not math.isfinite(x) or x<=0 or x>quality[ceiling] for x in samples):
            raise ValueError('invalid/insufficient timing samples or ceiling failed')

    if not isinstance(report.get('resident_ms_samples'),list):raise ValueError('resident wall samples required')
    validate_phase_rows(report.get('resident_cpu_phase_ms'),len(report['gpu_ms_samples']),report.get('resident_ms_samples'))
    validate_phase_rows(report.get('transfer_cpu_phase_ms'),len(report['transfer_inclusive_ms_samples']),report['transfer_inclusive_ms_samples'])


def validate_approval(approval,variant,identity,measurement_sha256):
    if not isinstance(approval,dict) or set(approval)!={'variant','identity','measurement_sha256','measurement_file','reviewer'}:
        raise ValueError('manual approval incomplete')
    if approval['variant']!=variant or approval['measurement_sha256']!=measurement_sha256 or not isinstance(approval['reviewer'],str) or not approval['reviewer'].strip() or not isinstance(approval['measurement_file'],str) or not approval['measurement_file'].strip():
        raise ValueError('manual approval does not bind this validated measurement')
    validate_cache(approval['identity'],identity)


def compiled_daxa_source(binary,requested=None):
    source=Path(subprocess.check_output([str(binary),'--daxa-source-directory'],text=True).strip()).resolve(strict=True)
    if not source.is_dir() or (requested is not None and Path(requested).resolve(strict=True)!=source):
        raise ValueError('source path does not match the binary compiled Daxa source')
    return source


def execution_identity(report,variant,binary,quality,compiler,daxa_source):
    """Bind queried hardware plus actual compiler, SPIR-V and patched Daxa bytes."""
    device=report['device_report']; subgroup=device['subgroups']
    validate_subgroup(subgroup['default_size'],subgroup['min_size'],subgroup['max_size'],subgroup['supported_stages'])
    if sha256(compiler)!=device['slangc_sha256']:
        raise ValueError('actual compiler bytes differ from compiled benchmark identity')
    shaders=[Path(path) for path in report['shader_files']]
    expected={f'daxa-metadata-{entry}.spv' for entry in variant['entrypoints']}
    if {path.name for path in shaders}!=expected or len(shaders)!=len(expected):
        raise ValueError('actual compiled entrypoint list mismatch')
    shader_hashes={path.name:sha256(path) for path in shaders}
    readback=memory_specialization(report)
    selected=next((m for m in device['memory_types'] if m['index']==readback['type_index']),None)
    if selected is None or selected['flags']!=readback['property_flags']:
        raise ValueError('readback memory identity differs from actual queried memory type')
    patch_hashes={path.name:sha256(path) for path in sorted((ROOT/'cmake/patches').glob('daxa*.patch'))}
    patched_names={line[6:] for patch in (ROOT/'cmake/patches').glob('daxa*.patch')
        for line in patch.read_text().splitlines() if line.startswith('+++ b/')}
    applied={name:sha256(daxa_source/name) for name in sorted(patched_names)}
    # ldd hashes bind the actually loaded Daxa shared library and dependencies.
    from compare_daxa_view import dependencies
    linked=dependencies(binary,True)['sha256']
    digest=lambda obj: hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    identity=dict(physical_gpu=device['device_uuid'],
        driver=digest({key:device[key] for key in ('driver_uuid','driver_name','driver_info','driver_version','vulkan_api_version')}),
        daxa=device['daxa_commit']+':'+digest(dict(patches=patch_hashes,applied=applied,linked=linked,binary_sha256=sha256(binary))),
        slang=device['slang_version'],compiler_sha256=sha256(compiler),shader_sha256=digest(shader_hashes),
        specialization=json.dumps(dict(entrypoints=variant['entrypoints'],workgroup=variant['workgroup'],
            flags='spirv_1_5,row-major,scalar-layout',readback_memory=readback),sort_keys=True,separators=(',',':')),
        source_width=report['width'],source_height=report['height'],target_width=report['width'],target_height=report['height'],
        disparities=0,paths=0,precision='uint32-pairs',quality_profile=quality['profile']+':'+digest(quality),subgroup=subgroup['default_size'])
    validate_identity(identity)
    return identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=ROOT/'build/daxa/daxa_variant_benchmark')
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--daxa-source',type=Path,help='Optional check against the actual source path embedded in the native binary')
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
    p.add_argument('--frames',type=int,default=30);p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--validation',action='store_true')
    p.add_argument('--readback-memory',choices=['first-coherent','cached-coherent'],default='first-coherent')
    a=p.parse_args(); config=load_config(a.config)
    if not 1<=a.width*a.height<=1920*1080 or min(a.width,a.height)<=0 or not 30<=a.frames<=300 or not 3<=a.warmup<=30:
        p.error('bounded comparison requires positive dimensions <=1080p, 30..300 frames and 3..30 warmups')
    if a.output.exists() and any(a.output.iterdir()): raise ValueError('output directory must be new or empty')
    a.output.mkdir(parents=True,exist_ok=True)
    binary=a.binary.resolve(); a.daxa_source=compiled_daxa_source(binary,a.daxa_source)
    evidence=dict(schema=1,config_sha256=sha256(a.config),binary_sha256=sha256(binary),
        scope='Local exact ID/age metadata only; no full-pipeline/held-out-quality/2D-locality qualification.',
        validation_requested=a.validation,readback_memory_policy=a.readback_memory,selected=None,manual_approval=config['approved'],measurements=[],passed=False)
    from compare_daxa_view import parse_diagnostics
    try:
        reference=None
        for variant in config['variants']:
            name=variant['name']; prefix=a.output/name; output=prefix.with_suffix('.json')
            command=[str(binary),'--variant',name,'--width',str(a.width),'--height',str(a.height),
                '--frames',str(a.frames),'--warmup',str(a.warmup),'--output',str(output),'--dump-prefix',str(prefix),'--readback-memory',a.readback_memory]
            if a.validation: command+=['--validation']
            child=subprocess.run(command,capture_output=True,text=True)
            Path(str(prefix)+'.stdout').write_text(child.stdout);Path(str(prefix)+'.stderr').write_text(child.stderr)
            if child.returncode or parse_diagnostics(child.stdout)['line_count'] or parse_diagnostics(child.stderr)['line_count']:
                raise ValueError('child failed or emitted validation/driver diagnostics: '+name)
            report=json.loads(output.read_text())
            if report.get('variant')!=name or report.get('width')!=a.width or report.get('height')!=a.height or report.get('frames')!=a.frames or report.get('warmup')!=a.warmup:
                raise ValueError('native report does not match requested profile')
            if report.get('readback_memory_policy')!=a.readback_memory:raise ValueError('requested readback memory policy not honored')
            if report.get('validation_enabled') is not a.validation: raise ValueError('validation request not honored')
            validate_measurement(report,config['quality'])
            dump_hashes={plane:sha256(str(prefix)+'.'+plane+'.i64') for plane in ('source_ids','age_ns')}
            if any(Path(str(prefix)+'.'+plane+'.i64').stat().st_size!=a.width*a.height*8 for plane in dump_hashes):
                raise ValueError('incomplete output metadata dump')
            if reference is not None and dump_hashes!=reference: raise ValueError('fused/unfused full metadata bytes differ')
            reference=dump_hashes
            identity=execution_identity(report,variant,binary,config['quality'],a.compiler,a.daxa_source)
            if evidence['measurements']:
                validate_comparable(evidence['measurements'][0]['identity'],identity)
            evidence['measurements'].append(dict(variant=name,identity=identity,measurement_sha256=sha256(output),
                output_sha256=dump_hashes,gpu_median_ms=statistics.median(report['gpu_ms_samples']),
                transfer_inclusive_median_ms=statistics.median(report['transfer_inclusive_ms_samples']),
                cpu_phase_medians={mode:{field:statistics.median(row[field] for row in report[mode+'_cpu_phase_ms']) for field in ('input_stage','enqueue','completion_wait','readback_copy','query','total')} for mode in ('resident','transfer')},
                counters=report.get('counters',{'registers':None,'lds':None,'spills':None,'memory_traffic':None})))
        evidence['ranked_by_gpu_ms']=sorted((m['variant'] for m in evidence['measurements']),
            key=lambda name:next(m['gpu_median_ms'] for m in evidence['measurements'] if m['variant']==name))
        if config['approved'] is not None:
            approval=config['approved']
            measurement=next((m for m in evidence['measurements'] if m['variant']==approval.get('variant')),None)
            if measurement is None: raise ValueError('unknown manually approved variant')
            approved_path=Path(approval.get('measurement_file',''))
            approved_report=json.loads(approved_path.read_text())
            validate_measurement(approved_report,config['quality'])
            if approved_report.get('validation_enabled') is not True:
                raise ValueError('approved artifact lacks actual validation')
            approved_identity=execution_identity(approved_report,next(v for v in config['variants'] if v['name']==measurement['variant']),binary,config['quality'],a.compiler,a.daxa_source)
            validate_cache(approved_identity,measurement['identity'])
            validate_approval(approval,measurement['variant'],measurement['identity'],sha256(approved_path))
            if not a.validation: raise ValueError('promotion requires clean actual validation')
            evidence['selected']=measurement['variant']
        evidence['passed']=True
    except (ValueError,KeyError,OSError,TypeError) as error:
        evidence['error']=str(error)
    (a.output/'comparison.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),passed=evidence['passed'],selected=evidence['selected'])))
    if not evidence['passed']: raise SystemExit(1)


if __name__=='__main__': main()
