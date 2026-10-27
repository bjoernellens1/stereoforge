"""Explicit Vulkan temporal research adapter with host uploads and readbacks.

Owner/selection readbacks gather 64-bit identities and ages on the CPU. This is
not the production GPU-resident ownership interface.
"""
from pathlib import Path
import numpy as np
from .types import DepthFrame, validate_pose


class VulkanTemporal:
    def __init__(self,device=None):
        import slangpy as s
        self.s=s
        self.device=device if device is not None else s.create_device(s.DeviceType.vulkan,enable_cuda_interop=False)
        if self.device.info.type!=s.DeviceType.vulkan:
            raise ValueError('Vulkan device required')
        module=self.device.load_module_from_source('stereoforge_temporal',(Path(__file__).parent/'shaders/temporal/history.slang').read_text())
        self.kernels={n:self.device.create_compute_kernel(self.device.link_program([module],[module.entry_point(n)])) for n in ('clear','splat_depth','splat_owner','resolve','fuse')}
        self.buffers={};self.shape=None

    def _prepare(self,shape):
        if self.shape!=shape:
            self.device.wait_for_idle();self.buffers={};self.shape=shape
        h,w=shape
        return dict(width=w,height=h)

    def _buffer(self,name,value=None,stride=4):
        if name not in self.buffers:
            self.buffers[name]=self.device.create_buffer(element_count=int(np.prod(self.shape)),struct_size=stride,
                usage=self.s.BufferUsage.shader_resource|self.s.BufferUsage.unordered_access)
        if value is not None:self.buffers[name].copy_from_numpy(np.ascontiguousarray(value))
        return self.buffers[name]

    def _upload(self,prefix,frame,valid=None):
        for name,value in dict(depth=frame.depth_m,confidence=frame.confidence,disparity=frame.disparity,
            valid=frame.valid if valid is None else valid,provenance=frame.provenance).items():
            self._buffer(prefix+'_'+name,np.asarray(value,dtype=np.uint32 if name in ('valid','provenance') else np.float32))
        ids=np.asarray(frame.source_ids,dtype=np.int64).view(np.uint32).reshape(-1,2)
        self._buffer(prefix+'_ids',ids,8)

    def _outputs(self):
        names=('output_depth','output_confidence','output_disparity','output_valid','output_provenance')
        return {n:self._buffer(n) for n in names}

    def _read(self,name,dtype=np.float32):
        return self.buffers[name].to_numpy().view(dtype).reshape(self.shape).copy()

    def _submit(self,encoder):
        token=self.device.submit_command_buffer(encoder.finish());self.device.wait_for_submit(token)

    def _result(self,ids,ages,timestamp,calibration):
        return DepthFrame(self._read('output_disparity'),self._read('output_depth'),self._read('output_confidence'),
            self._read('output_valid',np.uint32).astype(bool),self._read('output_provenance',np.uint32).astype(np.uint8),
            ids,ages,timestamp,calibration)

    def reproject(self,previous,camera,previous_to_current,timestamp_ns,fx=None,baseline_m=None,principal_offset=0):
        previous.__post_init__();pose=np.asarray(previous_to_current,dtype=np.float64);validate_pose(pose)
        if previous.depth_m.shape!=(camera.height,camera.width):raise ValueError('depth dimensions do not match camera')
        if camera.distortion and np.any(np.asarray(camera.distortion)!=0):raise ValueError('rectified pinhole calibration required')
        if timestamp_ns<previous.timestamp_ns:raise ValueError('timestamp moves backwards')
        metric=fx is not None or baseline_m is not None
        if not np.isfinite(principal_offset) or (metric and (fx is None or baseline_m is None or not np.isfinite([fx,baseline_m]).all() or min(fx,baseline_m)<=0)):
            raise ValueError('metric disparity requires positive finite fx and baseline_m')
        base=self._prepare(previous.depth_m.shape);n=previous.depth_m.size;self._upload('previous',previous)
        out=self._outputs();z=self._buffer('zbuffer');owner=self._buffer('owner')
        encoder=self.device.create_command_encoder()
        self.kernels['clear'].dispatch([n,1,1],vars=dict(**base,**out,zbuffer=z,owner=owner),command_encoder=encoder)
        proj=dict(**base,fx=camera.fx,fy=camera.fy,cx=camera.cx,cy=camera.cy,pose=pose.astype(np.float32),
            previous_depth=self.buffers['previous_depth'],previous_valid=self.buffers['previous_valid'],zbuffer=z)
        self.kernels['splat_depth'].dispatch([n,1,1],vars=proj,command_encoder=encoder)
        self.kernels['splat_owner'].dispatch([n,1,1],vars=dict(**proj,owner=owner),command_encoder=encoder)
        self.kernels['resolve'].dispatch([n,1,1],vars=dict(**base,**out,zbuffer=z,owner=owner,
            previous_confidence=self.buffers['previous_confidence'],previous_provenance=self.buffers['previous_provenance'],
            metric_disparity=int(metric),disparity_fx=fx or 0.,baseline_m=baseline_m or 0.,principal_offset=principal_offset),command_encoder=encoder)
        self._submit(encoder)
        selected=self._read('owner',np.uint32);valid=selected!=0xffffffff
        ids=np.full(self.shape,-1,np.int64);ages=np.zeros(self.shape,np.int64)
        ids[valid]=previous.source_ids.ravel()[selected[valid]]
        ages[valid]=previous.age_ns.ravel()[selected[valid]]+timestamp_ns-previous.timestamp_ns
        return self._result(ids,ages,timestamp_ns,previous.calibration_id)

    def fuse(self,current,prediction,max_age_ns=200000000,tolerance=.03):
        current.__post_init__()
        if max_age_ns<0 or tolerance<0 or not np.isfinite(tolerance):raise ValueError('invalid fusion limits')
        if prediction is not None:
            prediction.__post_init__()
            if prediction.calibration_id!=current.calibration_id or prediction.timestamp_ns>current.timestamp_ns:prediction=None
            elif prediction.depth_m.shape!=current.depth_m.shape:raise ValueError('prediction dimensions differ')
        base=self._prepare(current.depth_m.shape);self._upload('current',current)
        pred=prediction if prediction is not None else current
        ages=pred.age_ns+current.timestamp_ns-pred.timestamp_ns
        valid=pred.valid&(ages<=max_age_ns) if prediction is not None else np.zeros(self.shape,bool)
        self._upload('previous',pred,valid);out=self._outputs();choose=self._buffer('choose_previous')
        encoder=self.device.create_command_encoder();n=current.depth_m.size
        vars=dict(**base,**out,choose_previous=choose,tolerance=tolerance)
        for prefix in ('current','previous'):
            for name in ('depth','confidence','disparity','valid','provenance','ids'):
                vars[prefix+'_'+name]=self.buffers[prefix+'_'+name]
        self.kernels['fuse'].dispatch([n,1,1],vars=vars,command_encoder=encoder);self._submit(encoder)
        select=self._read('choose_previous',np.uint32).astype(bool)
        ids=np.where(select,pred.source_ids,current.source_ids).copy()
        age=np.where(select,ages,current.age_ns).copy()
        return self._result(ids,age,current.timestamp_ns,current.calibration_id)
