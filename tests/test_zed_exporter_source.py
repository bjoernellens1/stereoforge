"""Portable source contracts only; these do not execute or qualify the SDK."""
from pathlib import Path
import re

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/zed_comparison/export_sdk.cpp"


def source():
    return SOURCE.read_text()


def test_diagnostic_cannot_impersonate_canonical_export():
    s = source()
    assert '"capture-diagnostic"' in s
    assert '"canonical_export", false' in s
    assert '"qualification", "unqualified"' in s
    assert '"export.json"' not in s
    assert 'canonical export requires proven raw pose and rectification support' in s


def test_no_silent_mode_fallback_or_temporal_module():
    s = source()
    assert 'actual.depth_mode != requested.depth_mode' in s
    assert 'camera.isPositionalTrackingEnabled()' in s
    assert 'actual.depth_stabilization != 0' in s
    assert 'actual.coordinate_units != sl::UNIT::METER' in s
    assert 'actual.enable_fill_mode' in s
    assert 'enablePositionalTracking(' not in s


def test_raw_images_and_float_measures_are_distinct():
    s = source()
    for name in ('LEFT_UNRECTIFIED', 'RIGHT_UNRECTIFIED', 'LEFT', 'RIGHT'):
        assert f'sl::VIEW::{name}' in s
    assert 'sl::MEASURE::DEPTH' in s and 'sl::MEASURE::CONFIDENCE' in s
    assert 'sl::VIEW::DEPTH' not in s
    assert 'YUYV' not in s
    assert 'sdk_finite_positive' in s
    assert '"rgb_geometric_support", unavailable(' in s
    assert 'plane["alpha_is_validity"] = false' in s


def test_native_streaming_and_persistent_cpu_buffers():
    s = source()
    io = SOURCE.with_name('export_io.hpp').read_text()
    assert 'EVP_DigestInit_ex' in io and 'EVP_DigestUpdate' in io
    assert '1024 * 1024' in io
    assert 'getStepBytes(sl::MEM::CPU)' in s
    assert 'write_rows(writer, base,' in s
    assert 'mat.getDataType() != expected' in s
    assert 'fs::is_symlink' in s
    assert 'fs::create_directory(root)' in s
    assert not re.search(r'\b(system|popen|execl|execve)\s*\(', s)
    assert s.index('std::array<sl::Mat, 4> images') < s.index('for (int ordinal = 0;')
    assert s.index('std::vector<std::uint8_t> finite_positive') < s.index('for (int ordinal = 0;')


def test_calibration_full_opaque_transform_and_unavailable_fields():
    s = source()
    assert 'i < 12' in s
    assert '"sdk_stereo_transform"' in s
    assert '"right_to_left"' not in s
    for field in ('source_codec', 'source_lossless', 'ingress_timestamp', 'texture_map',
                  'depth_age', 'ancestry', 'model_identity'):
        assert f'"{field}"' in s
    assert '"status"] = "failed"' in s


def test_none_mode_range_exception_preserves_enabled_depth_gate():
    s = source()
    assert re.search(
        r'if \(requested\.depth_mode != sl::DEPTH_MODE::NONE\) \{\s*'
        r'require\(actual\.depth_minimum_distance == requested\.depth_minimum_distance'
        r' && actual\.depth_maximum_distance == requested\.depth_maximum_distance,', s)
    assert 'runtime.enable_depth = o.mode != "NONE"' in s
    assert 'actual.enable_depth == depth' in s
    assert '"requested_settings"' in s and '"effective_settings_after_open"' in s


def test_explicit_settings_directory_has_sdk_trailing_separator():
    s = source()
    assert re.search(
        r'if \(!o\.settings_dir\.empty\(\)\) \{\s*'
        r'settings_path = fs::path\(o\.settings_dir\)\.lexically_normal\(\)\.generic_string\(\);\s*'
        r"if \(settings_path\.back\(\) != '/'\) settings_path \+= '/';\s*"
        r'init\.optional_settings_path = settings_path\.c_str\(\);', s)
