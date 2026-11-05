import numpy as np
import pytest

from benchmarks.physical_depth_v14 import capture_corpus as C


def test_sdk_column_major_rotation_is_recorded_as_depth_to_color():
    rotation = [0, 1, 0, -1, 0, 0, 0, 0, 1]
    transform = C.extrinsics_matrix(rotation, [.1, .2, .3])
    np.testing.assert_allclose(np.asarray(transform) @ [1, 0, 0, 1], [.1, 1.2, .3, 1])


def test_new_capture_requires_session_and_cannot_overwrite(tmp_path):
    with pytest.raises(ValueError, match='session'):
        C.create_scene(tmp_path, 'scene', '')
    result = C.create_scene(tmp_path, 'scene', 'mount-2026-11-05-A')
    assert result == tmp_path / 'scene'
    with pytest.raises(FileExistsError):
        C.create_scene(tmp_path, 'scene', 'mount-2026-11-05-A')
    with pytest.raises(ValueError, match='directory'):
        C.create_scene(tmp_path, '../escape', 'session')


def test_capture_rejects_wrong_native_factory_profile():
    calibration = {'left': {'width': 1280, 'height': 720}, 'right': {'width': 1280, 'height': 720}}
    C.validate_native_calibration(calibration, 720)
    with pytest.raises(ValueError, match='native'):
        C.validate_native_calibration(calibration, 1080)
