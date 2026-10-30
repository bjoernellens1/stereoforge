"""Source ABI guards for opt-in Slang feature atlas; GPU parity is separate."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_shader_uses_same_validity_and_order_as_cpu_oracle():
    cpu = (ROOT / "include/stereoforge/feature_atlas.hpp").read_text()
    ops = (ROOT / "python/stereoforge/shaders/ops/feature_atlas.slang").read_text()
    wrapper = (ROOT / "python/stereoforge/shaders/daxa/feature_atlas.slang").read_text()
    assert "struct alignas(16) StereoFeaturePixel" in cpu
    for field in ("census24", "sobel_l1", "texture_l1", "valid"):
        assert field in cpu
    assert "static_assert(sizeof(StereoFeaturePixel) == 16)" in cpu
    assert "for(int dy=-2;dy<=2;++dy)" in ops
    assert "for(int dx=-2;dx<=2;++dx)" in ops
    assert "bits|=uint(value<center)<<(bit++);" in ops
    assert "support[index]!=1||gray[index]>255" in ops
    assert "uint4(bits,uint(abs(gx)+abs(gy)),texture,1)" in ops
    assert "uint4* atlas=(uint4*)p.atlas[tid.z]" in wrapper
    assert "tid.z>=2" in wrapper
    assert "atlas[index]=stereo_atlas_pixel(" in wrapper


def test_compile_gate_is_opt_in_and_references_both_sources():
    cmake = (ROOT / "cmake/DaxaPreprocess.cmake").read_text()
    assert 'option(STEREOFORGE_DAXA_FEATURE_ATLAS "Compile experimental GPU feature atlas" OFF)' in cmake
    assert 'add_custom_target(daxa_feature_atlas_shader' in cmake
    assert "shaders/daxa/feature_atlas.slang shaders/ops/feature_atlas.slang" in cmake
    assert "-entry feature_atlas -stage compute" in cmake
