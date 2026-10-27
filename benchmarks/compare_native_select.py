#!/usr/bin/env python3
"""Standalone select-only old/new measurement with exact output byte checks.

The native child retains seeded costs on Vulkan, reuses commands, timestamps the
select dispatch and downloads outputs once after measurements. This supervisor
hashes shader/binary/report/output files; no SlangPy is imported or linked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASELINE='451040e'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'build/standalone/native_select_benchmark')
    parser.add_argument('--candidate-shader',type=Path,default=ROOT/'python/stereoforge/shaders/stereo/sgm.slang')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=1280);parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--disparities',type=int,nargs='+',default=[32,64,128])
    parser.add_argument('--frames',type=int,default=30);parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if min(args.width,args.height,args.frames)<1 or args.warmup<0 or any(not 3<=d<=256 for d in args.disparities):parser.error('invalid bounded dimensions, disparities or iterations')
    binary=args.binary.resolve();candidate=args.candidate_shader.resolve();output=args.output.resolve()
    baseline=ROOT/f'artifacts/select-baseline-{BASELINE}.slang'
    baseline_bytes=subprocess.check_output(['git','show',f'{BASELINE}:python/stereoforge/shaders/stereo/sgm.slang'],cwd=ROOT)
    baseline.parent.mkdir(parents=True,exist_ok=True)
    if baseline.exists() and baseline.read_bytes()!=baseline_bytes:raise ValueError('Frozen baseline shader bytes changed')
    if not baseline.exists():baseline.write_bytes(baseline_bytes)
    dependencies=subprocess.check_output(['ldd',str(binary)],text=True)
    if any(token in dependencies.lower() for token in ('libsgl','slangpy','libpython')):raise ValueError('Native binary links a Python/SGL runtime')
    sources={str(path):sha256(path) for path in (binary,baseline,candidate,ROOT/'apps/native_select_benchmark.cpp')}
    commands=[];cases=[]
    for index,d in enumerate(args.disparities):
        # Reverse process order on alternate D to expose coarse order confounds.
        shaders=[('baseline',baseline),('candidate',candidate)]
        if index%2:shaders.reverse()
        for name,path in shaders:
            case=f'd{d}-{name}';cases.append((d,name,case))
            commands.append([str(binary),'--shader',str(path),'--width',str(args.width),'--height',str(args.height),
                '--disparities',str(d),'--seed',str(args.seed),'--frames',str(args.frames),'--warmup',str(args.warmup),
                '--output',str(output/f'{case}.json'),'--dump-prefix',str(output/case)])
    if args.verify_only:
        print(json.dumps(dict(sources_sha256=sources,child_environment=dict(PYTHONHOME='/nonexistent',PYTHONPATH=''),
            commands=commands,linked_dependencies=dependencies.splitlines()),indent=2));return
    if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be new or empty')
    output.mkdir(parents=True,exist_ok=True)
    environment=os.environ.copy();environment.update(PYTHONHOME='/nonexistent',PYTHONPATH='')
    reports={}
    for command,(_,_,case) in zip(commands,cases):
        subprocess.run(command,env=environment,check=True)
        reports[case]=json.loads((output/f'{case}.json').read_text())
        print(json.dumps(dict(case=case,gpu_ms=reports[case]['summary_ms']['gpu'],
            submit_completion_ms=reports[case]['summary_ms']['submit_to_completion'])),flush=True)
    if any(sha256(Path(path))!=digest for path,digest in sources.items()):raise RuntimeError('Measured binary or shader source changed during experiment')
    comparisons={}
    for d in args.disparities:
        before=reports[f'd{d}-baseline'];after=reports[f'd{d}-candidate']
        same_input=before['costs_fnv1a64']==after['costs_fnv1a64'] and before['costs_bytes']==after['costs_bytes']
        planes={name:sha256(output/f'd{d}-baseline.{name}')==sha256(output/f'd{d}-candidate.{name}')
                for name in ('disparity.f32','confidence.f32','validity.u32')}
        descriptor=dict(width=args.width,height=args.height,disparities=d,seed=args.seed,
            costs_fnv1a64=before['costs_fnv1a64'],costs_bytes=before['costs_bytes'],generator_source_sha256=sources[str(ROOT/'apps/native_select_benchmark.cpp')])
        comparisons[str(d)]=dict(same_input=same_input,exact_output_planes=planes,
            fixture_descriptor_sha256=hashlib.sha256(json.dumps(descriptor,sort_keys=True).encode()).hexdigest(),
            baseline_gpu_ms=before['summary_ms']['gpu'],candidate_gpu_ms=after['summary_ms']['gpu'],
            baseline_submit_completion_ms=before['summary_ms']['submit_to_completion'],candidate_submit_completion_ms=after['summary_ms']['submit_to_completion'])
    passed=all(value['same_input'] and all(value['exact_output_planes'].values()) for value in comparisons.values())
    result=dict(schema=1,passed=passed,baseline_commit=BASELINE,source_sha256=sources,
        child_environment=dict(PYTHONHOME='/nonexistent',PYTHONPATH=''),linked_dependencies=dependencies.splitlines(),
        commands=commands,comparisons=comparisons,cost_fingerprint_contract='FNV-1a64 over deterministic little-endian uint32 input bytes; fixture_descriptor_sha256 binds dimensions/seed/fingerprint/generator source, not the cost bytes directly.',
        scope='Select only; same seeded retained sum-cost input and ceil(pixels/64) groups. Setup/uploads and final output downloads excluded. Different process order alternates across D; not full stereo speed or 60 Hz qualification.',
        files={path.name:sha256(path) for path in output.iterdir() if path.is_file()})
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),passed=passed,comparisons=comparisons),indent=2))
    if not passed:raise SystemExit('Select output/input parity failed')


if __name__=='__main__':main()
