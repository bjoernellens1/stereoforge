#!/usr/bin/env python3
"""Export benchmark_view_fusion's exact calibrated inputs for native comparison.

No device construction, SlangPy import or camera access. CPU expected outputs
are optional and carry no invented metric source identity/age planes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from benchmark_view_fusion import fixture,input_identity
from stereoforge.fusion import synthesize
from stereoforge.gpu_geometry import VulkanViewSynthesizer


def export_fixture(target,width=1280,height=720,*,oracle=True):
    target=Path(target)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError('fixture destination must be new or empty')
    frame,left,camera,right=fixture(width,height)
    if camera.projection!='perspective':raise ValueError('native fixture header requires perspective')
    adapter=VulkanViewSynthesizer.__new__(VulkanViewSynthesizer)
    adapter._ray_cache={};adapter._preview_cache={}
    rays=np.concatenate((adapter._rays(frame.calibration.left),adapter._rays(frame.calibration.right)))
    _,mapping=adapter._preview_map(frame.calibration,camera)
    inverse=np.linalg.inv(camera.camera_to_rig).astype(np.float32)
    right_pose=(np.linalg.inv(camera.camera_to_rig)@frame.calibration.right_to_left).astype(np.float32)
    def pack(rgb):
        rgb=rgb.astype(np.uint32)
        return rgb[:,:,0]|(rgb[:,:,1]<<8)|(rgb[:,:,2]<<16)
    arrays={
        'source_depth.f32':np.concatenate((left.depth_m.ravel(),right.depth_m.ravel())).astype('<f4'),
        'confidence.f32':np.concatenate((left.confidence.ravel(),right.confidence.ravel())).astype('<f4'),
        'rgb.u32':np.concatenate((pack(frame.left).ravel(),pack(frame.right).ravel())).astype('<u4'),
        'provenance.u32':np.concatenate((left.provenance.ravel(),right.provenance.ravel())).astype('<u4'),
        'rays.f32':rays.astype('<f4'),
        'source_ids.i64':np.concatenate((left.source_ids.ravel(),right.source_ids.ravel())).astype('<i8'),
        'age_ns.i64':np.concatenate((left.age_ns.ravel(),right.age_ns.ravel())).astype('<i8'),
        'preview_indices.u32':mapping.astype('<u4')}
    target.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema_version=1,input_sha256=input_identity(frame,left,camera,right),
        explicit_right_depth=True,projection=camera.projection,byte_order='little',
        calibration=frame.calibration.to_dict(),calibration_id=frame.calibration.identity,
        source_width=width,source_height=height,target_width=camera.width,target_height=camera.height,
        files={},oracle='CPU synthesize exposure_match=True' if oracle else None,
        comparison_contract='Exact masks/source/provenance and geometric tolerance; RGB GPU histogram approximation permits <=1 intensity delta. No source identity/age oracle is supplied.')
    for name,array in arrays.items():
        array=np.ascontiguousarray(array);payload=array.tobytes();(target/name).write_bytes(payload)
        manifest['files'][name]=dict(sha256=hashlib.sha256(payload).hexdigest(),dtype=array.dtype.str,
                                    shape=list(array.shape),count=int(array.size),bytes=len(payload))
    header='\n'.join([
        f'{width} {height} {camera.width} {camera.height}',
        ' '.join(format(float(v),'.17g') for v in (camera.focal,camera.focal,(camera.width-1)/2,(camera.height-1)/2)),
        format(float(np.deg2rad(camera.hfov_deg)),'.17g'),
        ' '.join(format(float(v),'.17g') for v in inverse.ravel()),
        ' '.join(format(float(v),'.17g') for v in right_pose.ravel()),
        str(frame.timestamp_ns),frame.calibration.identity])+'\n'
    (target/'header.txt').write_text(header)
    manifest['files']['header.txt']=dict(sha256=hashlib.sha256(header.encode()).hexdigest(),bytes=len(header.encode()))
    if oracle:
        for mode in ('strict','preview'):
            view=synthesize(frame,left,camera,right,mode=mode,exposure_match=True)
            name=f'expected_{mode}.npz'
            np.savez_compressed(target/name,**{k:getattr(view,k) for k in
                ('rgb','depth_m','rgb_valid','depth_valid','confidence','provenance','source','approximate')})
            payload=(target/name).read_bytes()
            manifest['files'][name]=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    (target/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=1280);parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--no-oracle',action='store_true')
    args=parser.parse_args()
    result=export_fixture(args.output,args.width,args.height,oracle=not args.no_oracle)
    print(json.dumps(dict(output=str(args.output),input_sha256=result['input_sha256'])))


if __name__=='__main__':main()
