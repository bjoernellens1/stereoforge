#!/usr/bin/env python3
"""Supervise four standalone native fusion measurements and full-plane parity.

This Python process only verifies files and launches the native child. Every
measured child has PYTHONHOME=/nonexistent and an empty PYTHONPATH; no Python or
SGL runtime is allowed among its linked dependencies. Output dumps occur after
the native timing loop. Static fixtures do not establish a live 60 Hz pipeline.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PLANES={
    'depth.f32':'f','confidence.f32':'f','rgb.u32':'I','validity.u32':'I',
    'rgb_validity.u32':'I','provenance.u32':'I','source.u32':'I',
    'approximate.u32':'I','source_ids.i64':'q','age_ns.i64':'q','owner.u32':'I',
}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def fixture_manifest(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,entry in manifest['files'].items():
        path=directory/name
        if sha256(path)!=entry['sha256']:raise ValueError(f'Fixture checksum mismatch: {name}')
        if 'bytes' in entry and path.stat().st_size!=entry['bytes']:raise ValueError(f'Fixture size mismatch: {name}')
    return manifest


def linked_dependencies(binary):
    output=subprocess.check_output(['ldd',str(binary)],text=True)
    # Require actual standalone linkage, not merely an executable avoiding imports.
    if any(value in output.lower() for value in ('libsgl','slangpy','libpython')):
        raise ValueError('Measured binary links a Python, SlangPy or SGL runtime')
    if 'libslang-compiler' not in output:raise ValueError('Standalone Slang compiler linkage not found')
    return output.splitlines()


def values(prefix,plane,count):
    path=Path(f'{prefix}.{plane}')
    result=array.array(PLANES[plane]);result.frombytes(path.read_bytes())
    if sys.byteorder!='little':result.byteswap()
    if len(result)!=count:raise ValueError(f'Unexpected output plane size: {path}')
    return result


def compare(host,resident,count):
    planes={};loaded={}
    for name in PLANES:
        left=values(host,name,count);right=values(resident,name,count)
        differences=0;maximum=0.
        for a,b in zip(left,right):
            if a==b or (name.endswith('.f32') and math.isnan(a) and math.isnan(b)):continue
            differences+=1
            if math.isfinite(a) and math.isfinite(b):maximum=max(maximum,abs(a-b))
        planes[name]=dict(element_disagreements=differences,maximum_finite_difference=maximum,
                          bytes_equal=sha256(f'{host}.{name}')==sha256(f'{resident}.{name}'))
        loaded[name]=right
    valid=loaded['validity.u32'];rgb_valid=loaded['rgb_validity.u32'];approx=loaded['approximate.u32']
    invalid_defaults=approximation_errors=0
    for index in range(count):
        if not valid[index]:
            invalid_defaults+=int(loaded['source_ids.i64'][index]!=-1 or loaded['age_ns.i64'][index]!=0)
        if approx[index]:
            approximation_errors+=int(valid[index]!=0 or rgb_valid[index]==0 or
                not math.isnan(loaded['depth.f32'][index]) or loaded['confidence.f32'][index]!=0 or
                loaded['provenance.u32'][index]!=0 or loaded['source_ids.i64'][index]!=-1 or loaded['age_ns.i64'][index]!=0)
    result=dict(planes=planes,invalid_identity_age_errors=invalid_defaults,approximation_errors=approximation_errors,
                metric_pixels=sum(bool(x) for x in valid),rgb_pixels=sum(bool(x) for x in rgb_valid),
                approximate_pixels=sum(bool(x) for x in approx))
    result['passed']=all(plane['element_disagreements']==0 for plane in planes.values()) and invalid_defaults==0 and approximation_errors==0
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'build/standalone/native_view_benchmark')
    parser.add_argument('--fixture',type=Path,default=ROOT/'artifacts/fixture-720p')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=30);parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--verify-only',action='store_true',help='Verify files/linkage and print commands; never execute GPU work')
    args=parser.parse_args()
    if args.frames<1 or args.warmup<0:parser.error('frames must be positive and warmup nonnegative')
    args.binary=args.binary.resolve();args.fixture=args.fixture.resolve();args.output=args.output.resolve()
    fixture=fixture_manifest(args.fixture);dependencies=linked_dependencies(args.binary)
    configurations=[('strict','host'),('strict','resident'),('preview','host'),('preview','resident')]
    commands=[]
    for mode,input_mode in configurations:
        name=f'{mode}-{input_mode}'
        commands.append([str(args.binary),'--fixture',str(args.fixture),'--frames',str(args.frames),
            '--warmup',str(args.warmup),'--mode',mode,'--input',input_mode,'--readback','gpu',
            '--output',str(args.output/f'{name}.json'),'--dump-prefix',str(args.output/name)])
    if args.verify_only:
        print(json.dumps(dict(verified_fixture_input_sha256=fixture['input_sha256'],binary_sha256=sha256(args.binary),
            linked_dependencies=dependencies,child_environment=dict(PYTHONHOME='/nonexistent',PYTHONPATH=''),commands=commands),indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Output directory must be new or empty')
    args.output.mkdir(parents=True,exist_ok=True)
    environment=os.environ.copy();environment.update(PYTHONHOME='/nonexistent',PYTHONPATH='')
    reports={}
    for command,(mode,input_mode) in zip(commands,configurations):
        subprocess.run(command,env=environment,check=True)
        name=f'{mode}-{input_mode}';reports[name]=json.loads((args.output/f'{name}.json').read_text())
        print(json.dumps(dict(case=name,total_ms=reports[name]['summary_ms']['total'],
            initial_upload_total_ms=reports[name]['initial_upload_total_ms'])),flush=True)
    count=fixture['target_width']*fixture['target_height']
    comparisons={mode:compare(args.output/f'{mode}-host',args.output/f'{mode}-resident',count)
                 for mode in ('strict','preview')}
    sources=['apps/native_view_benchmark.cpp','src/geometry.cpp','src/native_device.cpp',
             'include/stereoforge/geometry.hpp','include/stereoforge/native_device.hpp',
             'python/stereoforge/shaders/fusion/project.slang','benchmarks/compare_native_resident.py']
    evidence=dict(schema=1,input_sha256=fixture['input_sha256'],fixture_manifest_sha256=sha256(args.fixture/'manifest.json'),
        binary_sha256=sha256(args.binary),source_sha256={name:sha256(ROOT/name) for name in sources},
        current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        linked_dependencies=dependencies,commands=commands,child_environment=dict(PYTHONHOME='/nonexistent',PYTHONPATH=''),
        frames=args.frames,warmup=args.warmup,scope='Static retained fixture. Resident source upload and wait occur once outside timed iterations. No camera/stereo/display or sustained live 60 Hz qualification.',
        timing_contract='Native GPU-output project plus completion only. Full CPU output dumps happen once after measurements. Cases execute sequentially; compare p50 and p95 without inferring full-pipeline throughput.',
        cases={name:dict(summary_ms=report['summary_ms'],initial_upload_total_ms=report['initial_upload_total_ms']) for name,report in reports.items()},
        comparisons=comparisons,files={path.name:sha256(path) for path in args.output.iterdir() if path.is_file()})
    evidence['passed']=all(result['passed'] for result in comparisons.values())
    (args.output/'comparison.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),passed=evidence['passed'],cases=evidence['cases']),indent=2))
    if not evidence['passed']:raise SystemExit('Native host/resident output parity failed; see comparison.json')


if __name__=='__main__':main()
