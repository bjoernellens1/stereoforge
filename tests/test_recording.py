import hashlib
import json
import numpy as np
import pytest
from stereoforge.types import Calibration, CameraModel, StereoFrame
from stereoforge.recording import Recorder, Replay


def calibration():
    c = CameraModel(3, 2, 4., 4., 1., .5)
    t = np.eye(4); t[0, 3] = .12
    return Calibration(c, c, t)


def frame(i=0, cal=None):
    rgb = np.arange(18, dtype=np.uint8).reshape(2,3,3)
    return StereoFrame(rgb, rgb[:, ::-1].copy(), cal or calibration(), 100+i, i,
                       'camera', [{'timestamp_ns': 10, 'gyro': [1.,2.,3.]}])


def record(path):
    with Recorder(path, calibration(), {'mode': 'test'}) as out:
        out.write(frame()); out.write(frame(1))


def edit(path, change):
    p = path/'manifest.json'; d = json.loads(p.read_text()); change(d)
    p.write_text(json.dumps(d))


def test_roundtrip(tmp_path):
    p = tmp_path/'record'; record(p)
    replay = Replay(p); frames = list(replay)
    assert replay.calibration.identity == calibration().identity
    assert replay.config == {'mode': 'test'}
    assert [f.frame_id for f in frames] == [0,1]
    assert frames[0].clock_domain == 'camera'
    assert frames[0].sensor_samples == frame().sensor_samples
    np.testing.assert_array_equal(frames[0].left, frame().left)
    np.testing.assert_array_equal(frames[0].right, frame().right)


def test_checksum_and_truncation(tmp_path):
    p = tmp_path/'record'; record(p)
    d = json.loads((p/'manifest.json').read_text()); f = p/d['frames'][0]['path']
    f.write_bytes(f.read_bytes()[:20])
    with pytest.raises(ValueError): list(Replay(p))


@pytest.mark.parametrize('change', [
    lambda d: d.update(finalized=False),
    lambda d: d.update(schema_version=99),
    lambda d: d['frames'][1].update(frame_id=0),
    lambda d: d['frames'][1].update(timestamp_ns=100),
    lambda d: d.update(calibration_id='wrong'),
    lambda d: d['frames'][0].update(path='../outside.npz'),
    lambda d: d['frames'][0].update(calibration_id='wrong'),
])
def test_invalid_manifest(tmp_path, change):
    p = tmp_path/'record'; record(p); edit(p, change)
    with pytest.raises(ValueError): list(Replay(p))


def test_missing_and_truncated_manifest(tmp_path):
    with pytest.raises(ValueError): Replay(tmp_path)
    (tmp_path/'manifest.json').write_text('{')
    with pytest.raises(ValueError): Replay(tmp_path)


def test_writer_validation(tmp_path):
    p = tmp_path/'record'
    with Recorder(p, calibration(), {}) as out:
        out.write(frame())
        with pytest.raises(ValueError): out.write(frame())
        c = calibration(); c = Calibration(c.left,c.right,c.right_to_left,'other')
        with pytest.raises(ValueError): out.write(frame(1,c))
    with pytest.raises(ValueError): Recorder(p, calibration(), {})


def test_exception_does_not_finalize(tmp_path):
    with pytest.raises(RuntimeError):
        with Recorder(tmp_path/'record', calibration(), {}) as out:
            out.write(frame()); raise RuntimeError('capture failed')
    with pytest.raises(ValueError): Replay(tmp_path/'record')


def test_missing_frame(tmp_path):
    p = tmp_path/'record'; record(p)
    d = json.loads((p/'manifest.json').read_text())
    (p/d['frames'][0]['path']).unlink()
    with pytest.raises(ValueError): list(Replay(p))


def test_damaged_archive_even_with_matching_hash(tmp_path):
    p = tmp_path/'record'; record(p)
    d = json.loads((p/'manifest.json').read_text())
    f = p/d['frames'][0]['path']; f.write_bytes(b'not an npz')
    edit(p, lambda d: d['frames'][0].update(sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    with pytest.raises(ValueError): list(Replay(p))
