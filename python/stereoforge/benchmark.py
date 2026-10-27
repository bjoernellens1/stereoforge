"""Controlled four-way stereo ablations over identical immutable inputs.

Caller-provided motion is required for every transition; identity motion is never
assumed. Ground truth must be independently supplied metric axial depth, not an
unlabelled vendor-estimator reference. Processing timings are local wall-clock
pipeline measurements and include host processing and any backend readback.
"""
from dataclasses import asdict
import hashlib
import json
from time import perf_counter
import numpy as np
from .pipeline import Pipeline, PipelineConfig
from .types import validate_pose


def _identity(frames):
    digest=hashlib.sha256()
    for frame in frames:
        metadata=json.dumps(dict(calibration=frame.calibration.to_dict(),
            timestamp_ns=frame.timestamp_ns,frame_id=frame.frame_id,
            clock_domain=frame.clock_domain,sensor_samples=frame.sensor_samples),
            sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        digest.update(len(metadata).to_bytes(8,'big')); digest.update(metadata)
        for image in (frame.left,frame.right):
            payload=np.ascontiguousarray(image).tobytes()
            digest.update(len(payload).to_bytes(8,'big')); digest.update(payload)
    return digest.hexdigest()


def run_ablations(frames,backend='cpu_census',max_disparity=16,paths=4,
                  motions=None,ground_truth=None):
    frames=list(frames)
    if not frames: raise ValueError('at least one frame is required')
    for previous,current in zip(frames,frames[1:]):
        if current.frame_id<=previous.frame_id or current.timestamp_ns<=previous.timestamp_ns:
            raise ValueError('frame IDs and timestamps must increase')
        if current.clock_domain!=previous.clock_domain:
            raise ValueError('clock domain changes require separate ablations')
        if current.calibration.identity!=previous.calibration.identity:
            raise ValueError('calibration changes require separate ablations')
    motion_inputs={}
    for frame in frames[1:]:
        if motions is None or frame.frame_id not in motions:
            raise ValueError(f'explicit motion is required for frame {frame.frame_id}')
        pose=np.array(motions[frame.frame_id],dtype=np.float64,copy=True)
        validate_pose(pose); motion_inputs[frame.frame_id]=pose
    truth={}
    if ground_truth is not None:
        for frame in frames:
            if frame.frame_id not in ground_truth: continue
            target=np.asarray(ground_truth[frame.frame_id],dtype=np.float64)
            if target.shape!=(frame.calibration.left.height,frame.calibration.left.width):
                raise ValueError('ground truth dimensions differ from depth grid')
            truth[frame.frame_id]=target.copy()
    identity=_identity(frames)
    variants=(('independent',False,False),('temporal',True,False),
              ('adaptive',False,True),('adaptive_temporal',True,True))
    results={}; pipelines={}; accumulators={}
    for name,temporal,adaptive in variants:
        config=PipelineConfig(backend=backend,max_disparity=max_disparity,paths=paths,
                              temporal=temporal,adaptive=adaptive,output='rectified_rgb')
        pipeline=Pipeline(config); pipelines[name]=pipeline
        frozen=asdict(config)
        frozen.update(relative_depth_tolerance=pipeline.history.relative_depth_tolerance,
                      min_confidence=.7,max_age_ns=100000000,
                      history_max_age_ns=pipeline.history.max_age_ns,
                      history_max_frames=pipeline.history.frames.maxlen)
        results[name]=dict(input_identity=identity,config=frozen,frames=[])
        accumulators[name]=dict(valid=0,total=0,support=0,supported_valid=0,
                                samples=0,absolute=0.,squared=0.,common_samples=0,
                                common_absolute=0.,common_squared=0.,times=[])
    # Only one frame's four output grids are retained for common-mask evaluation.
    # History remains bounded by each pipeline's configured history capacity.
    for index,frame in enumerate(frames):
        target=truth.get(frame.frame_id)
        support=(np.isfinite(target)&(target>0)) if target is not None else None
        support_count=int(np.count_nonzero(support)) if support is not None else 0
        depths={}
        for name,_,_ in variants:
            start=perf_counter()
            result=pipelines[name].process(frame,previous_to_current=None if index==0 else motion_inputs[frame.frame_id])
            elapsed=(perf_counter()-start)*1000
            depth=result.depth; depths[name]=depth
            numerical_valid=depth.valid & np.isfinite(depth.depth_m) & (depth.depth_m>0)
            count=int(np.count_nonzero(numerical_valid))
            error=np.empty(0)
            if support is not None:
                mask=numerical_valid & support
                error=depth.depth_m[mask].astype(np.float64)-target[mask]
            acc=accumulators[name]
            acc['valid']+=count; acc['total']+=depth.valid.size
            acc['support']+=support_count; acc['supported_valid']+=len(error)
            acc['samples']+=len(error); acc['absolute']+=float(np.abs(error).sum())
            acc['squared']+=float((error**2).sum()); acc['times'].append(elapsed)
            metrics=result.metrics
            results[name]['frames'].append(dict(frame_id=frame.frame_id,timestamp_ns=frame.timestamp_ns,
                processing_ms=elapsed,valid_fraction=count/depth.valid.size,
                candidate_evaluations=metrics['candidate_evaluations'],
                right_candidate_evaluations=metrics['right_candidate_evaluations'],
                aggregation_candidates=metrics['aggregation_candidates'],
                metric_ground_truth_available=target is not None,
                ground_truth_support_count=support_count,
                completeness_on_ground_truth_support=len(error)/support_count if support_count else None,
                metric_samples=len(error),
                depth_mae_m=float(np.mean(np.abs(error))) if len(error) else None,
                depth_rmse_m=float(np.sqrt(np.mean(error**2))) if len(error) else None))
        common=support.copy() if support is not None else None
        if common is not None:
            for depth in depths.values():
                common &= depth.valid & np.isfinite(depth.depth_m) & (depth.depth_m>0)
        for name,depth in depths.items():
            error=(depth.depth_m[common].astype(np.float64)-target[common]) if common is not None else np.empty(0)
            acc=accumulators[name]; acc['common_samples']+=len(error)
            acc['common_absolute']+=float(np.abs(error).sum()); acc['common_squared']+=float((error**2).sum())
            results[name]['frames'][-1].update(common_metric_samples=len(error),
                common_depth_mae_m=float(np.mean(np.abs(error))) if len(error) else None,
                common_depth_rmse_m=float(np.sqrt(np.mean(error**2))) if len(error) else None)
    for name,mode in results.items():
        acc=accumulators[name]; records=mode['frames']; n=acc['samples']; cn=acc['common_samples']
        mode.update(metric_ground_truth_available=bool(truth),metric_samples=n,
            ground_truth_support_count=acc['support'],
            completeness_on_ground_truth_support=acc['supported_valid']/acc['support'] if acc['support'] else None,
            depth_mae_m=acc['absolute']/n if n else None,
            depth_rmse_m=float(np.sqrt(acc['squared']/n)) if n else None,
            common_metric_samples=cn,common_depth_mae_m=acc['common_absolute']/cn if cn else None,
            common_depth_rmse_m=float(np.sqrt(acc['common_squared']/cn)) if cn else None,
            valid_fraction=acc['valid']/acc['total'],
            processing_ms=dict(p50=float(np.percentile(acc['times'],50)),p95=float(np.percentile(acc['times'],95))),
            candidate_evaluations=sum(r['candidate_evaluations'] for r in records),
            right_candidate_evaluations=sum(r['right_candidate_evaluations'] for r in records),
            aggregation_candidates=sum(r['aggregation_candidates'] for r in records))
    return dict(schema_version=1,input_identity=identity,frame_count=len(frames),
                ground_truth_geometry='rectified_left_axial_metres',
                timing_scope='local pipeline wall time including host processing and backend readback',
                ablations=results)
