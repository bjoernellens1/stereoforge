"""Static isolation contracts; these never discover or build the vendor SDK."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_reference_is_opt_in():
    root = (ROOT / "CMakeLists.txt").read_text()
    assert re.search(r'option\(STEREOFORGE_BUILD_ZED_REFERENCE\s+"[^"]+"\s+OFF\)', root)
    assert re.search(r'if\(STEREOFORGE_BUILD_ZED_REFERENCE\)\s+add_subdirectory\(benchmarks/zed_comparison\)\s+endif\(\)', root)
    assert not re.search(r'find_package\((?:ZED|CUDAToolkit)', root)


def test_sdk_dependencies_are_private_and_scoped():
    cmake = (ROOT / "benchmarks/zed_comparison/CMakeLists.txt").read_text()
    for package in ("ZED 5.5 REQUIRED CONFIG", "CUDAToolkit REQUIRED", "OpenSSL 3 REQUIRED COMPONENTS Crypto"):
        assert f"find_package({package})" in cmake
    assert re.search(r'add_executable\(export_sdk\s+export_sdk.cpp\)', cmake)
    assert "cxx_std_20" in cmake
    assert not re.search(r'(?m)^\s*(?:include_directories|link_directories|enable_language|add_compile_options)\(', cmake)
    assert not re.search(r'\b(?:PUBLIC|INTERFACE|CMAKE_CUDA_FLAGS|CMAKE_CXX_FLAGS)\b', cmake)
    for dependency in ("${ZED_LIBRARIES}", "${ZED_LIBRARY_DIR}", "CUDA::cudart", "OpenSSL::Crypto"):
        assert dependency in cmake


def test_pinned_json_notices_and_source_provenance():
    cmake = (ROOT / "benchmarks/zed_comparison/CMakeLists.txt").read_text()
    assert "sf_zed_reference_json" in cmake
    assert "json/v3.11.3/single_include/nlohmann/json.hpp" in cmake
    assert "SHA256=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6" in cmake
    assert "DOWNLOAD_NO_EXTRACT TRUE" in cmake
    for name in ("nlohmann-json.LICENSE.MIT", "nlohmann-json.NOTICE", "openssl.LICENSE.Apache-2.0"):
        assert name in cmake
        assert (ROOT / "third_party/licenses" / name).is_file()
    for token in ("reference-licenses", "file(SHA256", "CMAKE_CONFIGURE_DEPENDS", "SF_ZED_EXPORT_SOURCE_SHA256", "SF_ZED_EXPORT_COMMIT", "SF_ZED_EXPORT_DIRTY", "RESULT_VARIABLE"):
        assert token in cmake
