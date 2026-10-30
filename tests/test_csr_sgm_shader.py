"""GPU CSR layout/dispatch guards; actual SPIR-V and parity require Daxa hardware."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_csr_slang_kernel_uses_only_active_candidate_intervals():
    src=(ROOT/"python/stereoforge/shaders/daxa/csr_sgm.slang").read_text()
    assert "for(uint k=begin+lane;k<end;k+=64)" in src
    assert "for(uint k=prev_begin+lane;k<prev_end;k+=64)" in src
    assert "uint prev_begin=previous_exists?offsets[previous_pixel]:0;" in src
    assert "DeviceMemoryBarrierWithGroupSync();" in src
    assert "uint find_previous(" in src
    assert "void sum_paths(" in src
    assert "if(index>=p.total)return;" in src
    assert "disparity_range" in src
    assert "numthreads(64,1,1)" in src
    assert "uint* states=(uint*)p.paths+pass*p.total;" in src
    assert "p.width*p.height*p.disparity_range" not in src


def test_optional_kernel_compile_is_not_in_default_graph():
    cmake=(ROOT/"cmake/DaxaPreprocess.cmake").read_text()
    assert 'option(STEREOFORGE_DAXA_CSR_SGM "Compile experimental CSR sparse SGM GPU kernels" OFF)' in cmake
    assert "add_custom_target(daxa_csr_sgm_shaders" in cmake
    assert "-entry" in cmake and "scan_path sum_paths" in cmake
