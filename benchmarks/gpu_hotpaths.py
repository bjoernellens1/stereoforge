"""Synthetic real-device research-adapter timings, including host transfers.

Run: PYTHONPATH=python .venv/bin/python benchmarks/gpu_hotpaths.py
No camera access. Two warm-up frames and three measured frames per case.
"""
from dataclasses import asdict
import json
from time import perf_counter
import numpy as np
from stereoforge.types import CameraModel, Calibration, StereoFrame
from stereoforge.pipeline import Pipeline, PipelineConfig


def main():
    width,height=320,180
    camera=CameraModel(width,height,200.,200.,159.5,89.5)
    pose=np.eye(4);pose[0,3]=.12
    calibration=Calibration(camera,camera,pose,'synthetic-benchmark')
    rng=np.random.default_rng(42)
    left=rng.integers(0,256,(height,width,3),dtype=np.uint8)
    right=np.zeros_like(left);right[:,:-4]=left[:,4:]
    cases=(('independent',False,False,'rectified_rgb','cpu_reference'),
           ('temporal_adaptive',True,True,'rectified_rgb','cpu_reference'),
           ('strict_gpu_wide',False,False,'wide_rgbd','vulkan'))
    records={}
    for name,temporal,adaptive,output,fusion_backend in cases:
        config=PipelineConfig(backend='vulkan',max_disparity=32,paths=4,
                              temporal=temporal,adaptive=adaptive,output=output,
                              fusion_backend=fusion_backend)
        pipe=Pipeline(config)
        rows=[]
        for index in range(5):
            frame=StereoFrame(left,right,calibration,1000000000+index*16666667,index)
            start=perf_counter()
            result=pipe.process(frame,previous_to_current=np.eye(4) if index else None)
            elapsed=(perf_counter()-start)*1000
            if index>=2:
                row={'wall_ms':elapsed,**result.metrics}
                if pipe.view_renderer:row['view_adapter']=dict(pipe.view_renderer.last_timings)
                rows.append(row)
        timestamps=[sum(row['gpu_last_pass']['gpu_stages_ms'].values()) for row in rows]
        records[name]={'device':pipe.gpu.device_name,'config':asdict(config),'samples':rows,
            'wall_p50_ms':float(np.median([r['wall_ms'] for r in rows])),
            'last_stereo_pass_gpu_p50_ms':float(np.median(timestamps))}
    print(json.dumps({'resolution':[width,height],'shift_pixels':4,'warmups':2,
                      'samples_per_case':3,'cases':records},indent=2))


if __name__=='__main__':main()
