"""CPU-only standalone CLI preflight regressions; never create a Vulkan device."""
from pathlib import Path
import os
import resource
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/daxa/daxa_stereo_benchmark"


def invoke(tmp_path, *args, limited=False):
    if not BINARY.is_file():
        pytest.skip("offline native stereo benchmark has not been built")
    env = os.environ.copy()
    # A broken preflight must still never reach a physical Vulkan driver.
    env["VK_DRIVER_FILES"] = str(tmp_path / "absent-icd.json")
    env["VK_ICD_FILENAMES"] = env["VK_DRIVER_FILES"]
    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2,) * 2)
    return subprocess.run(
        [str(BINARY), "--output", str(tmp_path / "report.json"), *map(str, args)],
        env=env, capture_output=True, text=True, timeout=10,
        preexec_fn=limit if limited else None,
    )


def test_huge_fixture_rejected_before_allocation(tmp_path):
    result = invoke(tmp_path, "--width", 65535, "--height", 65535,
                    "--disparities", 1, limited=True)
    assert result.returncode == 1
    assert "retained allocation exceeds 2 GiB ceiling" in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_shape_arithmetic_cannot_wrap_before_resource_preflight(tmp_path):
    result = invoke(tmp_path, "--width", 2147483648, "--height", 2147483648,
                    "--disparities", 4, "--left", tmp_path / "missing-left",
                    "--right", tmp_path / "missing-right")
    assert result.returncode == 1
    assert "stereo volume exceeds uint32 indexing bound" in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_d128_rejected_before_input_read(tmp_path):
    result = invoke(tmp_path, "--variant", "packed-d128", "--disparities", 127,
                    "--left", tmp_path / "missing-left", "--right", tmp_path / "missing-right")
    assert result.returncode == 1
    assert "packed-d128 requires exactly128 disparities" in result.stderr
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize("suffix", [".disparity.f32", ".sums.u32", ".packed_penalties.u32"])
@pytest.mark.parametrize("dangling", [False, True])
def test_complete_destination_set_rejected_before_any_write(tmp_path, suffix, dangling):
    prefix = tmp_path / "dump"
    occupied = Path(str(prefix) + suffix)
    target = tmp_path / "outside-new-target"
    if dangling:
        occupied.symlink_to(target)
    else:
        occupied.write_bytes(b"preserve-existing-bytes")
    result = invoke(tmp_path, "--variant", "packed-specialized", "--width", 9,
                    "--height", 7, "--disparities", 3, "--dump-prefix", prefix,
                    "--dump-intermediates", "--left", tmp_path / "missing-left",
                    "--right", tmp_path / "missing-right")
    assert result.returncode == 1
    assert "destination must be a new file" in result.stderr
    assert not (tmp_path / "report.json").exists()
    assert not target.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [occupied.name]
    if dangling:
        assert occupied.is_symlink()
    else:
        assert occupied.read_bytes() == b"preserve-existing-bytes"


@pytest.mark.parametrize("alias", [False, True])
def test_report_and_dump_alias_rejected(tmp_path, alias):
    directory = tmp_path / "existing-parent"
    if alias:
        directory.mkdir()
    report = directory / ".." / "dump.disparity.f32" if alias else tmp_path / "dump.disparity.f32"
    result = invoke(tmp_path, "--dump-prefix", tmp_path / "dump", "--left",
                    tmp_path / "missing-left", "--right", tmp_path / "missing-right",
                    "--output", report)
    assert result.returncode == 1
    assert "destinations overlap" in result.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ([directory.name] if alias else [])


def test_writer_exclusive_create_is_defensive(tmp_path):
    source = (ROOT / "apps/daxa_stereo_benchmark.cpp").read_text()
    writer = source[source.index("void fresh_destination("):source.index("struct Summary")]
    cpp = tmp_path / "writer.cpp"
    cpp.write_text("#include <filesystem>\n#include <string>\n#include <vector>\n"
                   "#include <stdexcept>\n#include <cerrno>\n#include <fcntl.h>\n#include <unistd.h>\n"
                   + writer + '\nint main(int argc,char**argv){try{std::vector<unsigned> v{0x12345678};write(argv[1],v);return 0;}catch(...){return 1;}}\n')
    exe = tmp_path / "writer"
    subprocess.run(["c++", "-std=c++20", str(cpp), "-o", str(exe)], check=True)
    target = tmp_path / "outside"
    link = tmp_path / "dump"
    link.symlink_to(target)
    assert subprocess.run([str(exe), str(link)]).returncode == 1
    assert not target.exists() and link.is_symlink()
    fresh = tmp_path / "fresh"
    assert subprocess.run([str(exe), str(fresh)]).returncode == 0
    before = fresh.read_bytes()
    assert subprocess.run([str(exe), str(fresh)]).returncode == 1
    assert fresh.read_bytes() == before
