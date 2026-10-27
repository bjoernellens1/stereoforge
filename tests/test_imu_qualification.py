"""Qualification analysis uses decoded fixtures; no camera is opened here."""
import copy
import numpy as np
import pytest
from stereoforge.types import Calibration,CameraModel,StereoFrame
from benchmarks.qualify_imu import analyze_frames,compare_frames


def fixtures():
    c=CameraModel(3,2,4,4,1,.5);pose=np.eye(4);pose[0,3]=.12
    cal=Calibration(c,c,pose,serial='synthetic')
    frames=[]
    for frame_id,stamp in ((1,20_000_000),(2,30_000_000),(3,40_000_000)):
        samples=[]
        for timestamp in (stamp-7_500_000,stamp-5_000_000,stamp-2_500_000,stamp):
            samples.append(dict(timestamp_ns=timestamp,acceleration_m_s2=[0.,-9.8,0.],
                angular_velocity_deg_s=[0.,.1,-.1],temperature_c=30.,sync=timestamp==stamp))
        rgb=np.full((2,3,3),frame_id,np.uint8)
        frames.append(StereoFrame(rgb,rgb.copy(),cal,stamp,frame_id,'zed_open_capture',samples,
            camera_controls={'profile':'automatic','observed':{'auto_exposure_gain':True}}))
    return frames


def test_statistics_use_sample_clock_and_explicit_association():
    report=analyze_frames(fixtures(),preceding_timestamp_ns=10_000_000)
    assert report['frame_count']==3
    assert report['imu_sample_count']==12 and report['imu_unique_timestamps']==12
    assert report['imu_strictly_increasing'] and report['imu_duplicate_timestamps']==0
    assert report['imu_samples_after_associated_frame']==0
    assert report['imu_samples_not_after_previous_frame']==0
    assert report['imu_sync_marked_count']==3
    assert report['imu_rate_hz']==pytest.approx(400.)
    assert report['frame_rate_hz']==pytest.approx(100.)
    assert report['imu_per_frame']['minimum']==4
    assert report['quality_checks_passed']


def test_replay_preserves_control_and_decoded_sample_binary_fields():
    frames=fixtures();same=copy.deepcopy(frames)
    report=compare_frames(frames,same)
    assert report['exact']
    assert report['decoded_imu_binary_exact'] and report['controls_exact'] and report['rgb_bytes_exact']
    same[0].sensor_samples[0]['acceleration_m_s2'][0]=-0.0
    # Numeric equality cannot distinguish these bits; the explicit encoding does.
    assert frames[0].sensor_samples==same[0].sensor_samples
    report=compare_frames(frames,same)
    assert not report['decoded_imu_binary_exact'] and not report['exact']
    same=copy.deepcopy(frames);same[0].camera_controls['profile']='manual'
    report=compare_frames(frames,same)
    assert not report['controls_exact'] and not report['exact']


def test_statistics_reject_duplicate_or_misassociated_samples():
    frames=fixtures();frames[1].sensor_samples[0]=copy.deepcopy(frames[0].sensor_samples[-1])
    report=analyze_frames(frames,10_000_000)
    assert report['imu_duplicate_timestamps']==1 and not report['quality_checks_passed']
    frames=fixtures();frames[1].sensor_samples[-1]['timestamp_ns']=35_000_000
    report=analyze_frames(frames,10_000_000)
    assert report['imu_samples_after_associated_frame']==1 and not report['quality_checks_passed']


@pytest.mark.parametrize('change',[
    lambda s:s.update(sync=1),lambda s:s.update(timestamp_ns=-1),
    lambda s:s.update(acceleration_m_s2=[1,2]),lambda s:s.update(temperature_c=float('nan')),
    lambda s:s.update(angular_velocity_deg_s=[0,float('inf'),0]),
])
def test_invalid_sample_fields_fail_instead_of_counting_as_evidence(change):
    frames=fixtures();change(frames[0].sensor_samples[0])
    with pytest.raises(ValueError):analyze_frames(frames)


def test_no_sample_capture_is_not_a_pass():
    frames=fixtures()
    for frame in frames:frame.sensor_samples=[]
    report=analyze_frames(frames)
    assert not report['quality_checks_passed'] and report['imu_rate_hz'] is None


def test_real_recording_format_preserves_fixture_imu_and_controls(tmp_path):
    from benchmarks.qualify_imu import record_and_verify
    report=record_and_verify(fixtures(),tmp_path/'recording',preceding_timestamp_ns=10_000_000)
    assert report['replay']['exact'] and report['statistics']['quality_checks_passed']
    assert len(report['manifest_sha256'])==64 and len(report['frame_payload_sha256'])==3


def test_late_unique_samples_preserve_failed_interval_gate():
    from benchmarks.qualify_imu import classify_gates
    frames=fixtures()
    # Still unique and increasing, but the next batch delivers a sample dated
    # before the preceding image timestamp. Raw delivery is not synchronization.
    frames[0].sensor_samples.pop()
    frames[1].sensor_samples[0]['timestamp_ns']=19_000_000
    statistics=analyze_frames(frames,10_000_000)
    assert statistics['raw_delivery_checks_passed']
    assert not statistics['strict_frame_interval_association_passed']
    assert not statistics['quality_checks_passed']
    assert statistics['frames_with_late_imu_samples']==1
    assert statistics['late_sample_lag_behind_previous_frame_ms']['maximum']==1.
    gates=classify_gates(statistics,compare_frames(frames,copy.deepcopy(frames)),
        {'sensor_queue_drops':0,'controls_present_every_frame':True})
    assert gates['raw_delivery']=='pass' and gates['recording_replay']=='pass'
    assert gates['strict_frame_interval_association']=='fail'
