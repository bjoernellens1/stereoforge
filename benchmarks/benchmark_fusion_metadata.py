"""Host-only calibrated preview-map construction/cache benchmark; no GPU creation.

Run: PYTHONPATH=python .venv/bin/python benchmarks/benchmark_fusion_metadata.py
"""
import json
from time import perf_counter
import numpy as np
from stereoforge.types import CameraModel, Calibration
from stereoforge.fusion import VirtualCamera
from stereoforge.gpu_geometry import VulkanViewSynthesizer


def main():
    output={}
    for width,height in ((320,180),(1280,720)):
        c=CameraModel(width,height,width*.625,width*.625,(width-1)/2,(height-1)/2)
        pose=np.eye(4);pose[0,3]=.12
        cal=Calibration(c,c,pose);camera=VirtualCamera(width,height,90.)
        obj=VulkanViewSynthesizer.__new__(VulkanViewSynthesizer)
        obj._preview_cache={}
        start=perf_counter();_,mapping=obj._preview_map(cal,camera)
        initial=(perf_counter()-start)*1000
        samples=[]
        for _ in range(3):
            start=perf_counter();_,again=obj._preview_map(cal,camera)
            samples.append((perf_counter()-start)*1000)
            assert again is mapping
        for fov in (100.,110.,120.):obj._preview_map(cal,VirtualCamera(width,height,fov))
        output[f'{width}x{height}']=dict(initial_map_ms=initial,cached_samples_ms=samples,
            cached_p50_ms=float(np.median(samples)),map_bytes=mapping.nbytes,
            cache_entries_after_four_configurations=len(obj._preview_cache))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
