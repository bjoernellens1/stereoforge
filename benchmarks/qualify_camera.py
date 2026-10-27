"""Bounded physical capture/rectification smoke with explicit qualification limits.

Run with PYTHONPATH=python and STEREOFORGE_NATIVE_DIR pointing at the native
capture build. Captured RGB bytes are retained only in the requested artifact
recording. No match is filtered on epipolar vertical residual before measuring it.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback
import cv2
import numpy as np


def epipolar_residuals(left,right):
    left=np.asarray(left,dtype=np.float64); right=np.asarray(right,dtype=np.float64)
    if left.ndim!=2 or left.shape[1:]!=(2,) or left.shape!=right.shape or not len(left) or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('expected matching nonempty finite Nx2 correspondences')
    difference=left-right
    vertical=np.abs(difference[:,1]); horizontal=difference[:,0]
    return {'matches':len(left),'vertical_signed_px_p50':float(np.median(difference[:,1])),
            'vertical_abs_px_p50':float(np.percentile(vertical,50)),
            'vertical_abs_px_p95':float(np.percentile(vertical,95)),
            'horizontal_px_p05':float(np.percentile(horizontal,5)),
            'horizontal_px_p50':float(np.percentile(horizontal,50)),
            'horizontal_px_p95':float(np.percentile(horizontal,95)),
            'horizontal_positive_fraction':float(np.mean(horizontal>0)),
            'horizontal_negative_fraction':float(np.mean(horizontal<0))}


def feature_metrics(left,right):
    """SIFT ratio+mutual selection; optional independently fitted F outlier check."""
    detector=cv2.SIFT_create(nfeatures=5000)
    features=[]
    for image in (left,right):
        keypoints,descriptors=detector.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),None)
        features.append((keypoints,descriptors))
    result={'method':'SIFT, L2 ratio 0.7 in both directions, mutual correspondence; no vertical prefilter',
            'left_features':len(features[0][0]),'right_features':len(features[1][0])}
    if any(d is None or len(d)<2 for _,d in features):
        return dict(result,status='inconclusive',reason='insufficient descriptors')
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    def accepted(a,b):
        return {pair[0].queryIdx:pair[0].trainIdx for pair in matcher.knnMatch(a,b,k=2)
                if len(pair)==2 and pair[0].distance<.7*pair[1].distance}
    forward=accepted(features[0][1],features[1][1]); backward=accepted(features[1][1],features[0][1])
    pairs=[(i,j) for i,j in forward.items() if backward.get(j)==i]
    if len(pairs)<8:
        return dict(result,status='inconclusive',reason='insufficient mutual matches',mutual_matches=len(pairs))
    a=np.asarray([features[0][0][i].pt for i,j in pairs]); b=np.asarray([features[1][0][j].pt for i,j in pairs])
    result['mutual']=epipolar_residuals(a,b)
    fundamental,mask=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.5,.999)
    if fundamental is None or mask is None or np.count_nonzero(mask)<8:
        return dict(result,status='inconclusive',reason='fundamental matrix fit failed')
    inliers=mask.ravel().astype(bool)
    result['fundamental_ransac']={'threshold_px':1.5,'inliers':int(inliers.sum()),'fraction':float(inliers.mean())}
    result['likely_matches']=epipolar_residuals(a[inliers],b[inliers])
    metrics=result['likely_matches']
    # Screening thresholds are engineering checks, not physical metric truth.
    sufficient=metrics['matches']>=30
    result['status']=('pass' if metrics['vertical_abs_px_p95']<=2 and metrics['horizontal_positive_fraction']>=.9 else 'fail') if sufficient else 'inconclusive'
    result['screen_thresholds']={'minimum_matches':30,'vertical_p95_max_px':2.,'positive_disparity_min_fraction':.9}
    return result


def capture_recording(directory,width,height,count,warmup,fps):
    from stereoforge.capture import ZedCamera,discover_cameras
    from stereoforge.recording import Recorder,Replay
    before=perf_counter()
    with ZedCamera(width=width,height=height,fps=fps,cache_dir='build/calibration-cache') as camera:
        for _ in range(warmup): camera.read(2000)
        frames=[camera.read(2000) for _ in range(count)]
        status={'serial':camera.serial,'calibration_id':camera.calibration.identity,
                'baseline_m':camera.calibration.baseline_m,'right_to_left':camera.calibration.right_to_left.tolist(),
                'sensors_available':camera.sensors_available,'sensor_status':camera.sensor_status,
                'sensor_queue_drops':camera.sensor_dropped_count,'warmup_frames':warmup,
                'clock_domain':frames[0].clock_domain,'width_per_eye':width,'height':height,'requested_fps':fps,
                'devices':discover_cameras()}
    directory=Path(directory)
    with Recorder(directory,frames[0].calibration,{'purpose':'bounded camera qualification smoke','width':width,'height':height,'fps':fps}) as recorder:
        for frame in frames: recorder.write(frame)
    replay=list(Replay(directory))
    matches=all(a.frame_id==b.frame_id and a.timestamp_ns==b.timestamp_ns and
                a.calibration.identity==b.calibration.identity and a.sensor_samples==b.sensor_samples and
                np.array_equal(a.left,b.left) and np.array_equal(a.right,b.right) for a,b in zip(frames,replay))
    status.update({'status':'pass' if len(replay)==len(frames) and matches else 'fail','frames':len(frames),
                   'recording':str(directory),'replay_byte_and_metadata_exact':matches,
                   'manifest_sha256':hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),
                   'frame_ids':[f.frame_id for f in frames],'timestamps_ns':[f.timestamp_ns for f in frames],
                   'capture_and_record_wall_ms':(perf_counter()-before)*1000})
    return replay,status



def reinterpret_recording(path,factory_path):
    """Explicitly rebind unchanged recorded RGB to a corrected factory parser.

    The original recording and calibration remain unchanged and checksum-verified;
    both old/new identities and pixel hashes are included in the diagnostic report.
    """
    from stereoforge.recording import Replay
    from stereoforge.calibration import parse_zed_conf
    from stereoforge.types import StereoFrame
    replay=Replay(path); original=list(replay)
    if not original: raise ValueError('empty source recording')
    first=original[0]
    calibration=parse_zed_conf(factory_path,first.calibration.left.width,first.calibration.left.height,first.calibration.serial)
    corrected=[StereoFrame(f.left,f.right,calibration,f.timestamp_ns,f.frame_id,f.clock_domain,f.sensor_samples) for f in original]
    identical=all(np.array_equal(a.left,b.left) and np.array_equal(a.right,b.right) for a,b in zip(original,corrected))
    return corrected,{'status':'pass' if identical else 'fail','mode':'explicit calibration reinterpretation of verified saved RGB bytes',
        'source_recording':str(path),'source_manifest_sha256':hashlib.sha256((Path(path)/'manifest.json').read_bytes()).hexdigest(),
        'source_calibration_id':replay.calibration.identity,'calibration_id':calibration.identity,
        'factory_file':str(factory_path),'factory_file_sha256':hashlib.sha256(Path(factory_path).read_bytes()).hexdigest(),
        'baseline_m':calibration.baseline_m,'right_to_left':calibration.right_to_left.tolist(),
        'frame_ids':[f.frame_id for f in original],'timestamps_ns':[f.timestamp_ns for f in original],
        'rgb_bytes_unchanged':identical,'sensors_available':False,'sensor_status':'not_rechecked_offline',
        'frame_rgb_sha256':[{'left':hashlib.sha256(f.left.tobytes()).hexdigest(),'right':hashlib.sha256(f.right.tobytes()).hexdigest()} for f in original]}


def process_frames(frames,backend,disparity,wide=False):
    from stereoforge.pipeline import Pipeline,PipelineConfig
    before=perf_counter()
    pipeline=Pipeline(PipelineConfig(backend=backend,max_disparity=disparity,paths=4,
                                    output='wide_rgbd' if wide else 'rectified_rgb',
                                    fusion_backend='vulkan' if wide else 'cpu_reference'))
    results=[]
    for frame in frames:
        result=pipeline.process(frame)
        d=result.depth
        row={'frame_id':frame.frame_id,'valid_fraction':float(np.mean(d.valid)),
             'median_depth_m':float(np.median(d.depth_m[d.valid])) if d.valid.any() else None,
             'median_disparity_px':float(np.median(d.disparity[d.valid])) if d.valid.any() else None,
             'metrics':result.metrics,'finite_positive_valid_depth':bool(np.all(np.isfinite(d.depth_m[d.valid])&(d.depth_m[d.valid]>0)))}
        if result.view is not None:
            row['wide_coverage']=result.view.coverage
            row['strict_has_no_approximation']=bool(not result.view.approximate.any())
            row['wide_finite_positive_valid_depth']=bool(np.all(np.isfinite(result.view.depth_m[result.view.depth_valid])&(result.view.depth_m[result.view.depth_valid]>0)))
        results.append(row)
    passed=all(r['valid_fraction']>0 and r['finite_positive_valid_depth'] and
               (not wide or (r['strict_has_no_approximation'] and r['wide_finite_positive_valid_depth'] and r['wide_coverage']['depth_fraction']>0)) for r in results)
    return {'status':'pass' if passed else 'fail',
            'backend':backend,'scope':'numerical smoke only; scene distances are unknown, no metric ground truth',
            'construction_plus_processing_wall_ms':(perf_counter()-before)*1000,'frames':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',default='artifacts/qualification')
    parser.add_argument('--frames',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=30)
    parser.add_argument('--gpu',action='store_true',help='also capture/process 3 VGA frames through Vulkan stereo and RGB-D')
    parser.add_argument('--replay-source',help='verify and reuse recorded RGB without opening hardware')
    parser.add_argument('--factory-calibration',help='explicit factory file to reinterpret source calibration with the current parser')
    args=parser.parse_args()
    if bool(args.replay_source)!=bool(args.factory_calibration): parser.error('--replay-source and --factory-calibration must be supplied together')
    if args.replay_source and args.gpu: parser.error('--gpu is a live acquisition smoke and cannot combine with --replay-source')
    if args.frames<1 or args.warmup<0: parser.error('positive frame count and nonnegative warmup required')
    output=Path(args.artifacts); output.mkdir(parents=True,exist_ok=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'qualification_scope':'physical capture, replay integrity, epipolar screening and bounded numerical smoke',
            'metric_distance_truth_available':False,'hardware_drops_observable':False,'notes':['SDK frame IDs count SDK-received frames; gaps do not measure all USB/hardware losses.',
             'Sensor permission failures remain explicit; absent IMU is not a calibrated motion source.',
             'No epipolar vertical filtering is applied before residual measurement.']}
    try:
        if args.replay_source:
            frames,report['capture_720p']=reinterpret_recording(args.replay_source,args.factory_calibration)
        else:
            frames,report['capture_720p']=capture_recording(output/'recording-720p',1280,720,args.frames,args.warmup,60)
        from stereoforge.calibration import Rectifier
        rectifier=Rectifier(frames[0].calibration)
        report['rectification']=[dict(frame_id=frame.frame_id,**feature_metrics(rectifier.rectify(frame).left,rectifier.rectify(frame).right)) for frame in frames]
        report['cpu_sgbm']=process_frames(frames,'cpu_sgbm',128)
        report['qualification_gates']={'imu':'observed_raw_only' if report['capture_720p']['sensors_available'] else ('not_rechecked_offline' if args.replay_source else 'blocked_sensor_permissions'),
            'physical_metric_depth':'not_qualified_no_ground_truth','realtime_pipeline':'not_qualified_bounded_smoke_only'}
        if args.gpu:
            gpu_frames,report['capture_vga']=capture_recording(output/'recording-vga',672,376,3,args.warmup,60)
            report['vulkan']=process_frames(gpu_frames,'vulkan',64,True)
        else:
            report['vulkan']={'status':'not_run','reason':'--gpu not requested'}
        statuses=[report['capture_720p']['status'],report['cpu_sgbm']['status']]+[r['status'] for r in report['rectification']]
        if args.gpu: statuses += [report['capture_vga']['status'],report['vulkan']['status']]
        report['status']='fail' if 'fail' in statuses else 'inconclusive' if 'inconclusive' in statuses else 'pass_with_limits'
    except Exception as exc:
        report['status']='fail'; report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    target=output/'camera-qualification.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    return 1 if report['status']=='fail' else 0


if __name__=='__main__':
    raise SystemExit(main())
