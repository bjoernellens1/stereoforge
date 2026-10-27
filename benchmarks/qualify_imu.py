"""Bounded decoded-IMU recording/replay qualification, without GPU processing.

This verifies source fields exposed by Open Capture, not raw HID report bytes,
IMU extrinsics, calibrated motion, or physical synchronization accuracy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
from time import perf_counter, time_ns
import numpy as np


def _json_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _sample_binary(sample):
    """Explicit little-endian timestamp, seven binary64 decoded values, bool."""
    try:
        timestamp=sample['timestamp_ns']
        if type(timestamp) is not int or not 0<=timestamp<=2**64-1:
            raise ValueError('invalid IMU timestamp')
        acceleration=sample['acceleration_m_s2'];angular=sample['angular_velocity_deg_s']
        if len(acceleration)!=3 or len(angular)!=3 or type(sample['sync']) is not bool:
            raise ValueError('invalid IMU vector or synchronization flag')
        values=(*acceleration,*angular,sample['temperature_c'])
        if any(type(v) not in (int,float) or not np.isfinite(v) for v in values):
            raise ValueError('nonfinite or invalid decoded IMU value')
        return struct.pack('<Q7d?',timestamp,*values,sample['sync'])
    except (KeyError,TypeError,OverflowError,struct.error) as error:
        raise ValueError('invalid decoded IMU sample') from error


def _ordered_imu_binary(frames):
    chunks=[]
    for frame in frames:
        chunks.append(struct.pack('<QQQ',frame.frame_id,frame.timestamp_ns,len(frame.sensor_samples)))
        for index,sample in enumerate(frame.sensor_samples):
            chunks.append(struct.pack('<Q',index));chunks.append(_sample_binary(sample))
    return b''.join(chunks)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _summary(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values):return None
    return {'minimum':float(values.min()),'p50':float(np.percentile(values,50)),
            'p95':float(np.percentile(values,95)),'maximum':float(values.max())}


def analyze_frames(frames,preceding_timestamp_ns=None):
    frames=list(frames)
    if not frames:raise ValueError('at least one stereo frame required')
    samples=[];after=not_after=0;late_lag=[];late_frames=0;previous=preceding_timestamp_ns
    for frame in frames:
        frame_has_late=False
        for sample in frame.sensor_samples:
            _sample_binary(sample)
            timestamp=sample['timestamp_ns']
            after+=timestamp>frame.timestamp_ns
            late=previous is not None and timestamp<=previous
            not_after+=late
            if late:late_lag.append((previous-timestamp)/1e6);frame_has_late=True
            samples.append(sample)
        late_frames+=frame_has_late
        previous=frame.timestamp_ns
    frame_times=[f.timestamp_ns for f in frames];ids=[f.frame_id for f in frames]
    stamps=[s['timestamp_ns'] for s in samples]
    imu_delta=[b-a for a,b in zip(stamps,stamps[1:])]
    frame_delta=[b-a for a,b in zip(frame_times,frame_times[1:])]
    frame_increasing=all(d>0 for d in frame_delta)
    imu_increasing=bool(samples) and all(d>0 for d in imu_delta)
    acceleration=np.asarray([s['acceleration_m_s2'] for s in samples],dtype=np.float64).reshape(-1,3)
    angular=np.asarray([s['angular_velocity_deg_s'] for s in samples],dtype=np.float64).reshape(-1,3)
    domains=sorted({f.clock_domain for f in frames})
    delivery_ok=bool(len(samples)>1 and frame_increasing and imu_increasing and
        len(set(stamps))==len(stamps) and after==0 and len(domains)==1)
    association_ok=after==0 and not_after==0
    return {
        'frame_count' :len(frames),'frame_ids':ids,'frame_timestamps_ns':frame_times,
        'clock_domains':domains,'frame_timestamps_strictly_increasing':frame_increasing,
        'sdk_frame_ids_contiguous':all(b==a+1 for a,b in zip(ids,ids[1:])),
        'frame_rate_hz':(len(frames)-1)*1e9/(frame_times[-1]-frame_times[0]) if len(frames)>1 and frame_increasing else None,
        'frame_interval_ms':_summary(np.asarray(frame_delta,dtype=np.float64)/1e6),
        'imu_sample_count':len(samples),'imu_unique_timestamps':len(set(stamps)),
        'imu_duplicate_timestamps':len(stamps)-len(set(stamps)),'imu_strictly_increasing':imu_increasing,
        'imu_first_timestamp_ns':stamps[0] if stamps else None,'imu_last_timestamp_ns':stamps[-1] if stamps else None,
        'imu_rate_hz':(len(samples)-1)*1e9/(stamps[-1]-stamps[0]) if len(samples)>1 and imu_increasing else None,
        'imu_interval_ms':_summary(np.asarray(imu_delta,dtype=np.float64)/1e6),
        'imu_sync_marked_count':sum(s['sync'] for s in samples),
        'imu_sync_marked_fraction':sum(s['sync'] for s in samples)/len(samples) if samples else None,
        'imu_per_frame':_summary([len(f.sensor_samples) for f in frames]),
        'frames_without_imu_samples':sum(not f.sensor_samples for f in frames),
        'imu_samples_after_associated_frame':after,'imu_samples_not_after_previous_frame':not_after,
        'frames_with_late_imu_samples':late_frames,'late_sample_lag_behind_previous_frame_ms':_summary(late_lag),
        'late_sample_lag_definition':'Previous delivered frame timestamp minus delayed sample timestamp; observed batching lag, not calibrated sensor-to-image latency',
        'preceding_warmup_frame_timestamp_ns':preceding_timestamp_ns,
        'acceleration_norm_m_s2':_summary(np.linalg.norm(acceleration,axis=1)),
        'angular_velocity_norm_deg_s':_summary(np.linalg.norm(angular,axis=1)),
        'temperature_c':_summary([s['temperature_c'] for s in samples]),
        'raw_delivery_checks_passed':delivery_ok,
        'strict_frame_interval_association_passed':association_ok,
        'quality_checks_passed':delivery_ok and association_ok,
    }


def compare_frames(captured,replayed):
    captured=list(captured);replayed=list(replayed)
    same_count=len(captured)==len(replayed)
    source_binary=_ordered_imu_binary(captured);replayed_binary=_ordered_imu_binary(replayed)
    def same(field):return same_count and all(getattr(a,field)==getattr(b,field) for a,b in zip(captured,replayed))
    rgb_exact=same_count and all(a.left.dtype==b.left.dtype and a.right.dtype==b.right.dtype and
        a.left.shape==b.left.shape and a.right.shape==b.right.shape and
        a.left.tobytes()==b.left.tobytes() and a.right.tobytes()==b.right.tobytes() for a,b in zip(captured,replayed))
    controls_exact=same_count and all(_json_bytes(a.camera_controls)==_json_bytes(b.camera_controls) for a,b in zip(captured,replayed))
    metadata_exact=same_count and all(_json_bytes(a.sensor_samples)==_json_bytes(b.sensor_samples) for a,b in zip(captured,replayed))
    result={'frame_count_exact':same_count,'frame_ids_exact':same('frame_id'),
        'frame_timestamps_exact':same('timestamp_ns'),'clock_domains_exact':same('clock_domain'),
        'calibration_ids_exact':same_count and all(a.calibration.identity==b.calibration.identity for a,b in zip(captured,replayed)),
        'rgb_bytes_exact':rgb_exact,'controls_exact':controls_exact,'sensor_json_fields_exact':metadata_exact,
        'decoded_imu_binary_exact':source_binary==replayed_binary,
        'captured_decoded_imu_sha256':_sha(source_binary),'replayed_decoded_imu_sha256':_sha(replayed_binary),
        'binary_encoding':'Per frame <QQQ frame_id,timestamp_ns,count; per sample <Q index then <Q7d? timestamp,acceleration_xyz,angular_xyz,temperature,sync; little endian',
        'sample_identity':'ordered (frame_id,index_in_frame,timestamp_ns); hardware IMU sequence IDs are not exposed'}
    result['exact']=all(result[key] for key in result if key.endswith('_exact'))
    return result


def record_and_verify(frames,path,config=None,preceding_timestamp_ns=None):
    from stereoforge.recording import Recorder,Replay
    frames=list(frames)
    if not frames:raise ValueError('cannot qualify an empty recording')
    path=Path(path)
    with Recorder(path,frames[0].calibration,config or {'purpose':'decoded IMU recording/replay qualification'}) as recorder:
        for frame in frames:recorder.write(frame)
    replayed=list(Replay(path));manifest=(path/'manifest.json').read_bytes()
    return {'recording':str(path),'manifest_sha256':_sha(manifest),
            'frame_payload_sha256':[f['sha256'] for f in json.loads(manifest)['frames']],
            'statistics':analyze_frames(frames,preceding_timestamp_ns),'replay':compare_frames(frames,replayed)}



def classify_gates(statistics,replay,capture):
    return {
        'raw_delivery':'pass' if statistics['raw_delivery_checks_passed'] and capture.get('sensor_queue_drops')==0 else 'fail',
        'recording_replay':'pass' if replay['exact'] and capture.get('controls_present_every_frame') else 'fail',
        'strict_frame_interval_association':'pass' if statistics['strict_frame_interval_association_passed'] else 'fail',
        'physical_synchronization_accuracy':'not_qualified',
        'calibrated_motion':'not_qualified',
    }

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=None)
    parser.add_argument('--frames',type=int,default=48)
    parser.add_argument('--warmup',type=int,default=12)
    parser.add_argument('--width',type=int,default=672)
    parser.add_argument('--height',type=int,default=376)
    parser.add_argument('--fps',type=int,default=60)
    parser.add_argument('--device-id',type=int,default=-1)
    parser.add_argument('--expected-serial',default=None)
    args=parser.parse_args()
    if not 2<=args.frames<=600 or not 0<=args.warmup<=120:parser.error('frames must be 2..600 and warmup 0..120')
    destination=args.artifacts or Path('artifacts')/f'imu-qualification-{time_ns()}'
    if destination.exists() and any(destination.iterdir()):parser.error('artifact directory must be new or empty')
    destination.mkdir(parents=True,exist_ok=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'decoded raw IMU delivery and recording/replay integrity only',
        'limits':['Raw HID packet bytes are not recorded; decoded fields are compared with explicit binary encoding.',
                  'SDK frame IDs do not certify absence of camera/USB hardware loss.',
                  'Sync marks and hardware_sync_enabled report SDK behavior, not physical synchronization accuracy.',
                  'No IMU extrinsics, camera-IMU clock calibration, bias calibration, or calibrated motion qualification.',
                  'Camera controls are cached host snapshots, not exposure-time sensor truth.']}
    try:
        from stereoforge.capture import ZedCamera,discover_cameras
        report['devices']=discover_cameras()
        start=perf_counter();previous=None
        with ZedCamera(width=args.width,height=args.height,fps=args.fps,device_id=args.device_id,
                       cache_dir='build/calibration-cache') as camera:
            if args.expected_serial is not None and camera.serial!=args.expected_serial:
                raise ValueError('attached camera serial does not match requested qualification camera')
            if not camera.sensors_available:raise RuntimeError('raw IMU acquisition unavailable: '+camera.sensor_status)
            for _ in range(args.warmup):previous=camera.read(2000).timestamp_ns
            capture_start=perf_counter();frames=[camera.read(2000) for _ in range(args.frames)];capture_end=perf_counter()
            report['capture']={'serial':camera.serial,'width_per_eye':args.width,'height':args.height,
                'requested_fps':args.fps,'warmup_frames':args.warmup,'sensors_available':camera.sensors_available,
                'sensor_status':camera.sensor_status,'sensor_queue_drops':camera.sensor_dropped_count,
                'capture_host_elapsed_s':capture_end-capture_start,'open_warmup_capture_host_elapsed_s':capture_end-start,
                'calibration_id':camera.calibration.identity,'controls_present_every_frame':all(bool(f.camera_controls) for f in frames)}
        record_start=perf_counter()
        report.update(record_and_verify(frames,destination/'recording',
            {'purpose':'decoded IMU recording/replay qualification','width_per_eye':args.width,'height':args.height,'requested_fps':args.fps},previous))
        report['record_and_replay_host_elapsed_s']=perf_counter()-record_start
        report['gates']=classify_gates(report['statistics'],report['replay'],report['capture'])
        report['status']='pass_with_limits' if all(report['gates'][gate]=='pass' for gate in ('raw_delivery','recording_replay','strict_frame_interval_association')) else 'fail'
    except Exception as error:
        report.update(status='error',error=f'{type(error).__name__}: {error}')
    (destination/'imu-qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'artifact':str(destination/'imu-qualification.json'),'status':report['status']}))
    return 0 if report['status']=='pass_with_limits' else 1


if __name__=='__main__':raise SystemExit(main())
