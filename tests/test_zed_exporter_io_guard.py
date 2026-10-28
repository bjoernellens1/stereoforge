"""CPU-only production serializer contracts; no SDK/device required."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "benchmarks/zed_comparison/export_io.hpp"


def body(text, symbol):
    match = re.search(r"\b" + symbol + r"\s*\([^;]*?\)\s*\{", text)
    assert match, symbol
    start = match.end()
    depth = 1
    for pos in range(start, len(text)):
        depth += (text[pos] == "{") - (text[pos] == "}")
        if depth == 0:
            return text[start:pos]
    raise AssertionError("unclosed function")


def streaming_contract(text):
    # Only copy_stream reads the source; copy_and_hash delegates directly, with
    # no second scan/hash and no whole-file staging allocation.
    copy = body(text, "copy_and_hash")
    stream = body(text, "copy_stream")
    write = body(text, "write")
    assert len(re.findall(r"\bcopy_stream\s*\(", copy)) == 1
    assert not re.search(r"\.(read|seekg|rdbuf)\s*\(|\b(Digest|vector)\b", copy)
    assert len(re.findall(r"\.read\s*\(", stream)) == 1
    assert "std::array<char, copy_chunk_bytes>" in stream
    assert "digest_.update(data, count)" in write
    assert write.index("stream_.write") < write.index("digest_.update")
    bound = re.search(r"copy_chunk_bytes\s*=\s*(\d+)\s*\*\s*(\d+)", text)
    assert bound and int(bound[1]) * int(bound[2]) <= 1024 * 1024


def test_sdk_free_header_and_single_pass_source_contract():
    text = HEADER.read_text()
    includes = re.findall(r"#include\s*[<\"]([^>\"]+)", text)
    assert all(not re.search(r"^sl/|cuda|vulkan|python", name, re.I) for name in includes)
    assert not re.search(r"\b(system|popen|execve|wait|synchronize)\s*\(", text)
    streaming_contract(text)
    cmake = (HEADER.parent / "CMakeLists.txt").read_text()
    assert 'file(SHA256 "${sf_zed_export_io}" SF_ZED_EXPORT_IO_SHA256)' in cmake
    assert 'CMAKE_CONFIGURE_DEPENDS "${sf_zed_export_source}" "${sf_zed_export_io}"' in cmake
    assert 'SF_ZED_EXPORT_IO_SHA256="${SF_ZED_EXPORT_IO_SHA256}"' in cmake
    assert '"io_sha256", SF_ZED_EXPORT_IO_SHA256' in HEADER.with_name('export_sdk.cpp').read_text()


def test_contract_rejects_separate_full_read_hash_mutation():
    text = HEADER.read_text()
    anchor = '    copy_stream(input, writer);'
    assert anchor in text
    mutated = text.replace(anchor, '    Digest second_scan; input.seekg(0);\n' + anchor)
    with pytest.raises(AssertionError):
        streaming_contract(mutated)


def compile_probe(tmp_path, header, json_include):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("standalone C++ compiler unavailable")
    (tmp_path / "benchmarks/zed_comparison").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "benchmarks/zed_comparison/export_io.hpp").write_text(header)
    shutil.copyfile(ROOT / "tests/zed_export_io_probe.cpp", tmp_path / "tests/zed_export_io_probe.cpp")
    binary = tmp_path / "probe"
    subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(json_include), str(tmp_path / "tests/zed_export_io_probe.cpp"),
                    "-lcrypto", "-o", str(binary)], check=True, capture_output=True, text=True)
    try:
        return subprocess.run([str(binary), str(tmp_path / "output")], capture_output=True, text=True)
    finally:
        shutil.rmtree(tmp_path / "output", ignore_errors=True)


def test_actual_production_helper_byte_parity(tmp_path, json_include):
    result = compile_probe(tmp_path, HEADER.read_text(), json_include)
    assert result.returncode == 0, result.stderr


def test_probe_rejects_padding_write_mutation(tmp_path, json_include):
    text = HEADER.read_text()
    mutated = text.replace('writer.write(base + y * stride, row_bytes)', 'writer.write(base + y * stride, stride)')
    assert mutated != text
    result = compile_probe(tmp_path, mutated, json_include)
    assert result.returncode != 0
    assert 'write call count/bytes mismatch' in result.stderr


def test_probe_rejects_unbounded_copy_mutation(tmp_path, json_include):
    text = HEADER.read_text().replace('copy_chunk_bytes = 1024 * 1024', 'copy_chunk_bytes = 2 * 1024 * 1024')
    assert text != HEADER.read_text()
    result = compile_probe(tmp_path, text, json_include)
    assert result.returncode != 0
    assert 'copy not bounded' in result.stderr
