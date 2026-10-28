import numpy as np
import pytest

from benchmarks.crestereo_cpu_smoke import prepare_inputs, scale_disparity


def test_prepare_rgb_to_bgr_and_half_resolution_endpoints():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[0, 0] = [255, 64, 0]
    image[-1, -1] = [0, 32, 255]
    result = prepare_inputs(image, image)
    assert result['left_image'].shape == (1, 3, 240, 320)
    assert result['left_image_dw2'].shape == (1, 3, 120, 160)
    np.testing.assert_allclose(result['left_image_dw2'][0, :, 0, 0], [0, 64/255, 1], atol=1e-7)
    np.testing.assert_allclose(result['left_image_dw2'][0, :, -1, -1], [1, 32/255, 0], atol=1e-7)
    np.testing.assert_array_equal(result['right_image'], result['left_image'])


def test_disparity_rescale_uses_horizontal_scale():
    result = scale_disparity(np.ones((240, 320), np.float32), 1280, 720)
    assert result.shape == (720, 1280)
    np.testing.assert_array_equal(result, np.full((720, 1280), 4, np.float32))


def test_reject_mismatched_stereo_images():
    with pytest.raises(ValueError):
        prepare_inputs(np.zeros((4, 4, 3), np.uint8), np.zeros((5, 4, 3), np.uint8))


def test_model_verification_rejects_unpinned_member(tmp_path):
    from benchmarks.crestereo_cpu_smoke import EXPECTED, verify_model
    for name in EXPECTED:
        (tmp_path / name).write_bytes(b'not the model')
    with pytest.raises(ValueError, match='pinned archive'):
        verify_model(tmp_path)


def test_half_input_interpolates_interior_with_aligned_corners():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(320, dtype=np.uint16)[None, :] % 256
    result = prepare_inputs(image, image)['left_image_dw2'][0, 2]
    x = 80 * 319 / 159
    assert result[30, 80] == pytest.approx(x / 255, abs=2e-6)
