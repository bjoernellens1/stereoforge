#!/usr/bin/env python3
"""Bounded synthetic fusion cost experiment; no camera or stereo estimator.

Legacy baseline is the frozen 88a9e37 adapter with its original four entrypoints
in the current shared shader. All cases include source uploads and host readbacks.
GPU dispatch intervals overlap host submission/waits; do not add them to wall time.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import types
import numpy as np
from stereoforge.types import CameraModel,Calibration,StereoFrame,DepthFrame
from stereoforge.fusion import VirtualCamera
from stereoforge.gpu_geometry import VulkanViewSynthesizer

ROOT=Path(__file__).resolve().parents[1]
BASELINE='88a9e37'


def fixture(width=320,height=180):
    y,x=np.indices((height,width))
    model=CameraModel(width,height,width*.875,width*.875,(width-1)/2,(height-1)/2)
    pose=np.eye(4);pose[0,3]=.12
    calibration=Calibration(model,model,pose)
    color=np.stack((30+x%180,40+y%150,50+(x+y)%160),axis=-1).astype(np.uint8)
    frame=StereoFrame(color,np.clip(color.astype(np.float32)*.8,0,255).astype(np.uint8),calibration,1,1)
    z=(2+x*.002).astype(np.float32);z[:,width//2:width//2+max(1,width//40)]=.6
    def depth(offset):
        valid=(x+2*y+offset)%9!=0
        values=np.where(valid,z,np.nan)
        return DepthFrame(np.where(valid,8.,np.nan).astype(np.float32),values,
            np.where(valid,.8,0).astype(np.float32),valid,np.where(valid,1,0).astype(np.uint8),
            np.where(valid,1,0).astype(np.int64),np.zeros(z.shape,np.int64),1,calibration.identity)
    camera=VirtualCamera(width,height,85.)
    return frame,depth(0),camera,depth(3)


def input_identity(frame,left,camera,right):
    digest=hashlib.sha256()
    for value in (frame.left,frame.right,left.depth_m,left.valid,left.confidence,left.provenance,
                  right.depth_m,right.valid,right.confidence,right.provenance):
        digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    digest.update(frame.calibration.identity.encode())
    digest.update(json.dumps(dict(width=camera.width,height=camera.height,hfov=camera.hfov_deg,
        projection=camera.projection,pose=camera.camera_to_rig.tolist()),sort_keys=True).encode())
    return digest.hexdigest()


def statistics(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(minimum=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max()))


def legacy_class():
    source=subprocess.check_output(['git','show',f'{BASELINE}:python/stereoforge/gpu_geometry.py'],cwd=ROOT,text=True)
    module=types.ModuleType('stereoforge._frozen_view_baseline')
    module.__package__='stereoforge';module.__file__=str(ROOT/'python/stereoforge/gpu_geometry.py')
    exec(compile(source,module.__file__,'exec'),module.__dict__)
    return module.VulkanViewSynthesizer,hashlib.sha256(source.encode()).hexdigest()


def sample(renderer,inputs,options):
    frame,left,camera,right=inputs
    start=time.perf_counter();result=renderer.render(frame,left,camera,right,**options)
    return result,dict(wall_ms=(time.perf_counter()-start)*1000,**renderer.last_timings)


def summarize(renderer,records,result,first):
    scalar_keys=[k for k,v in records[0].items() if isinstance(v,(float,int))]
    summary={k:statistics([r[k] for r in records]) for k in scalar_keys}
    if 'gpu_stages_ms' in records[0]:
        summary['gpu_stages_ms']={k:statistics([r['gpu_stages_ms'][k] for r in records]) for k in records[0]['gpu_stages_ms']}
    allocations={name:buffer.size for name,buffer in renderer.buffers.items()}
    return dict(first_call_wall_ms=first['wall_ms'],first_call_timings=first,
        records=records,summary=summary,coverage=result.coverage,allocated_buffer_bytes=allocations,
        total_allocated_buffer_bytes=sum(allocations.values()),
        allocation_scope='Slang resource buffers only; excludes driver, shader, query pool and host arrays')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width',type=int,default=320);parser.add_argument('--height',type=int,default=180)
    parser.add_argument('--warmup',type=int,default=2);parser.add_argument('--frames',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.width,args.height,args.frames)<1 or args.warmup<0:parser.error('invalid bounds')
    inputs=fixture(args.width,args.height)
    baseline,baseline_hash=legacy_class()
    old=baseline();full=VulkanViewSynthesizer(profile=True)
    report=dict(schema=1,input_sha256=input_identity(*inputs),width=args.width,height=args.height,
        warmup=args.warmup,measured_frames=args.frames,stereo_enabled=False,physical_camera=False,
        explicit_right_depth=True,input_array_bytes=sum(array.nbytes for array in
            (inputs[0].left,inputs[0].right,inputs[1].depth_m,inputs[1].valid,inputs[1].confidence,inputs[1].provenance,
             inputs[3].depth_m,inputs[3].valid,inputs[3].confidence,inputs[3].provenance)),
        configuration=dict(projection=inputs[2].projection,requested_hfov_deg=inputs[2].hfov_deg,
                           camera_to_rig=inputs[2].camera_to_rig.tolist(),calibration=inputs[0].calibration.to_dict()),
        baseline_commit=BASELINE,baseline_adapter_sha256=baseline_hash,
        current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        working_tree_changes=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
        adapter_sha256=hashlib.sha256((ROOT/'python/stereoforge/gpu_geometry.py').read_bytes()).hexdigest(),
        shader_sha256=hashlib.sha256((ROOT/'python/stereoforge/shaders/fusion/project.slang').read_bytes()).hexdigest(),
        device=dict(api=full.device.info.api_name,adapter=full.device.info.adapter_name,
                    timestamp_frequency=full.device.info.timestamp_frequency),
        timing_contract='Wall includes CPU preparation, transfers, submission, wait, readback, unpack and timestamp reads. GPU intervals cover dispatches only. Host prepare/upload is not CPU-only time.',
        baseline_timing_contract='Frozen legacy adapter has wall and upload_dispatch_wait timings; no dispatch timestamps or separated waits/readbacks.')
    configurations=(('legacy_nearest',old,{}),('strict_full',full,dict(mode='strict',exposure_match=True)),
        ('preview_full',full,dict(mode='preview',exposure_match=True)))
    first={};records={};results={}
    for name,renderer,options in configurations:
        _,first[name]=sample(renderer,inputs,options)
        for _ in range(args.warmup):sample(renderer,inputs,options)
        records[name]=[]
    report['sampling_order']=[]
    for iteration in range(args.frames):
        # Rotate case order each round; never infer speedup from one sequential sweep.
        shift=iteration%len(configurations)
        ordered=configurations[shift:]+configurations[:shift]
        report['sampling_order'].append([case[0] for case in ordered])
        for name,renderer,options in ordered:
            results[name],record=sample(renderer,inputs,options)
            records[name].append(record)
    report['cases']={name:summarize(renderer,records[name],results[name],first[name])
                     for name,renderer,_ in configurations}
    base=report['cases']['legacy_nearest']['summary']['wall_ms']['median']
    report['incremental_wall_median_ms']={name:case['summary']['wall_ms']['median']-base for name,case in report['cases'].items() if name!='legacy_nearest'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),device=report['device'],
        wall_median_ms={name:case['summary']['wall_ms']['median'] for name,case in report['cases'].items()},
        incremental_wall_median_ms=report['incremental_wall_median_ms']),indent=2))


if __name__=='__main__':main()
