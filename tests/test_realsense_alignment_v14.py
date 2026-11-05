import pytest


def test_installed_sdk_align_preserves_native_depth_z():
    rs = pytest.importorskip('pyrealsense2')
    from benchmarks.physical_depth_v14.verify_realsense_alignment import probe
    result = probe(rs)
    assert result['status'] == 'verified_installed_sdk_software_device'
    assert result['aligned_depth_units'] == [1000]
    assert result['transformed_color_z_m'] == pytest.approx(1.1)
    assert result['native_depth_z_m'] == pytest.approx(1.)
