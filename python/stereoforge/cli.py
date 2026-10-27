"""Command-line capture, replay, experiments, and inspection."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
from time import perf_counter
import cv2
import numpy as np
from .types import CameraModel, Calibration, StereoFrame
from .pipeline import Pipeline, PipelineConfig
from .fusion import VirtualCamera
from .recording import Recorder, Replay

def synthetic_frame(width=96,height=48,disparity=4,frame_id=0):
    if min(width,height)<8 or not 0<disparity<width//2:raise ValueError('invalid synthetic geometry')
    rng=np.random.default_rng(42)
    left=rng.integers(0,256,(height,width,3),dtype=np.uint8);right=np.zeros_like(left);right[:,:-disparity]=left[:,disparity:]
    c=CameraModel(width,height,float(width),float(width),(width-1)/2,(height-1)/2)
    pose=np.eye(4);pose[0,3]=.12
    return StereoFrame(left,right,Calibration(c,c,pose),frame_id*16666667+1,frame_id)

def write_result(path,result):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise ValueError(f'output directory is not empty: {path}')
    path.mkdir(parents=True,exist_ok=True)
    d=result.depth
    arrays={name:getattr(d,name) for name in ('disparity','depth_m','confidence','valid','provenance','source_ids','age_ns')}
    if result.view:
        for name in ('depth_m','depth_valid','rgb_valid','confidence','provenance','source','approximate'):
            arrays['view_'+name]=getattr(result.view,name)
        result.metrics['view_coverage']=result.view.coverage
        result.metrics['view_depth_convention']=result.view.depth_convention
    np.savez_compressed(path/'depth.npz',**arrays)
    cv2.imwrite(str(path/'rgb.png'),cv2.cvtColor(result.rgb,cv2.COLOR_RGB2BGR))
    (path/'metrics.json').write_text(json.dumps(result.metrics,indent=2,allow_nan=False)+'\n')

def _config(args):
    config=json.loads(Path(args.config).read_text()) if args.config else {}
    for field in ('backend','max_disparity','paths','output','fusion_backend'):
        value=getattr(args,field,None)
        if value is not None:config[field]=value
    if getattr(args,'preview',False):config['preview']=True
    return PipelineConfig(**config)

def _pipeline(args):
    camera=VirtualCamera(**json.loads(Path(args.virtual_camera).read_text())) if args.virtual_camera else None
    return Pipeline(_config(args),camera)

def _camera(args):
    from .capture import ZedCamera
    from .calibration import load_calibration
    cal=load_calibration(args.calibration) if args.calibration else None
    return ZedCamera(cal,args.width,args.height,args.fps,args.device_id,cache_dir=args.cache)

def _doctor(args):
    from .capture import discover_cameras
    from .inference import capabilities
    result=dict(cameras=discover_cameras(),neural=capabilities(),packages={})
    for name in ('numpy','opencv-python-headless','slangpy','pybind11'):
        try:result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result['packages'][name]=None
    if args.gpu:
        from .gpu import VulkanStereo
        gpu=VulkanStereo(8);result['vulkan_device']=gpu.device_name
    print(json.dumps(result,indent=2))

def main(argv=None):
    parser=argparse.ArgumentParser(prog='stereoforge',description='Explicit Vulkan stereo research and capture')
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor');doctor.add_argument('--gpu',action='store_true')
    def processing(p):
        p.add_argument('--config');p.add_argument('--backend',choices=['vulkan','cpu_sgbm','cpu_census'])
        p.add_argument('--disparities',dest='max_disparity',type=int);p.add_argument('--paths',type=int,choices=[4,8])
        p.add_argument('--mode',dest='output',choices=['left_rgb','right_rgb','rectified_rgb','wide_rgb','wide_rgbd','stabilized_wide_rgb'])
        p.add_argument('--fusion-backend',choices=['cpu_reference','vulkan']);p.add_argument('--preview',action='store_true')
        p.add_argument('--virtual-camera');p.add_argument('--output',dest='destination',required=True)
    def camera_options(p):
        p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
        p.add_argument('--fps',type=int,default=60);p.add_argument('--frames',type=int,default=30)
        p.add_argument('--device-id',type=int,default=-1);p.add_argument('--calibration');p.add_argument('--cache')
    syn=sub.add_parser('synthetic');processing(syn);syn.add_argument('--width',type=int,default=96);syn.add_argument('--height',type=int,default=48)
    record=sub.add_parser('record');camera_options(record);record.add_argument('--output',dest='destination',required=True)
    replay=sub.add_parser('replay');processing(replay);replay.add_argument('recording');replay.add_argument('--motions',help='JSON frame_id -> previous-to-current rigid transform')
    live=sub.add_parser('live');processing(live);camera_options(live)
    ablate=sub.add_parser('ablate');ablate.add_argument('recording');ablate.add_argument('--motions',required=True);ablate.add_argument('--backend',choices=['vulkan','cpu_census'],default='vulkan');ablate.add_argument('--disparities',type=int,default=128);ablate.add_argument('--paths',type=int,choices=[4,8],default=4);ablate.add_argument('--output',dest='destination',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='doctor':_doctor(args)
        elif args.command=='synthetic':
            result=_pipeline(args).process(synthetic_frame(args.width,args.height))
            result.metrics['synthetic_ground_truth_m']=args.width*.12/4
            write_result(args.destination,result);print(json.dumps(result.metrics,indent=2))
        elif args.command=='record':
            if args.frames<1:raise ValueError('frames must be positive')
            with _camera(args) as camera:
                ids=[];times=[];start=perf_counter()
                with Recorder(args.destination,camera.calibration,dict(width=args.width,height=args.height,fps=args.fps,sensors_available=camera.sensors_available,sensor_status=camera.sensor_status)) as recording:
                    for _ in range(args.frames):
                        frame=camera.read();recording.write(frame);ids.append(frame.frame_id);times.append(frame.timestamp_ns)
                report=dict(frames=len(ids),sdk_id_gaps=int(np.maximum(np.diff(ids)-1,0).sum()),elapsed_wall_s=perf_counter()-start,sensors_available=camera.sensors_available,sensor_status=camera.sensor_status,sensor_dropped=camera.sensor_dropped_count)
                (Path(args.destination)/'capture-metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
        elif args.command=='replay':
            pipeline=_pipeline(args);motions=json.loads(Path(args.motions).read_text()) if args.motions else {}
            for frame in Replay(args.recording):
                motion=motions.get(str(frame.frame_id))
                result=pipeline.process(frame,previous_to_current=np.asarray(motion) if motion is not None else None)
                write_result(Path(args.destination)/f'{frame.frame_id:08d}',result)
        elif args.command=='live':
            if args.frames<1:raise ValueError('frames must be positive')
            pipeline=_pipeline(args)
            with _camera(args) as camera:
                for _ in range(args.frames):
                    frame=camera.read();result=pipeline.process(frame)
                    write_result(Path(args.destination)/f'{frame.frame_id:08d}',result)
                    print(json.dumps({'frame_id':frame.frame_id,**result.metrics}),flush=True)
        elif args.command=='ablate':
            from .benchmark import run_ablations
            motions={int(k):np.asarray(v) for k,v in json.loads(Path(args.motions).read_text()).items()}
            report=run_ablations(Replay(args.recording),args.backend,args.disparities,args.paths,motions)
            target=Path(args.destination)
            with target.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    except (ValueError,RuntimeError,OSError,ImportError) as exc:
        parser.exit(2,f'stereoforge: {exc}\n')

if __name__=='__main__':main()
