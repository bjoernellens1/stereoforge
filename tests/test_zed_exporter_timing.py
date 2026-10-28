"""Actual SDK-free parser/helper probes; these never execute or qualify the SDK."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'benchmarks/zed_comparison/export_sdk.cpp'
HEADER = SOURCE.with_name('sdk_benchmark.hpp')


def test_actual_parser_and_fake_grab_helper(tmp_path, json_include):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('standalone C++ compiler unavailable')
    source = SOURCE.read_text()
    parser = source[source.index('struct Options {'):source.index('sl::DEPTH_MODE depth_mode')]
    probe = r'''
#include <json.hpp>
#include "sdk_benchmark.hpp"
#include <iostream>
#include <set>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <new>
static bool allocation_forbidden = false;
[[gnu::noinline]] void* operator new(std::size_t bytes) {
    if (allocation_forbidden) std::abort();
    if (auto* value = std::malloc(bytes)) return value;
    throw std::bad_alloc();
}
[[gnu::noinline]] void operator delete(void* value) noexcept { std::free(value); }
[[gnu::noinline]] void operator delete(void* value, std::size_t) noexcept { std::free(value); }
using SafeError = std::runtime_error;
void require(bool value, const char* message) { if (!value) throw SafeError(message); }
PARSER
Options request(std::initializer_list<const char*> args) {
    std::vector<char*> argv;
    for (auto arg : args) argv.push_back(const_cast<char*>(arg));
    return parse(static_cast<int>(argv.size()), argv.data());
}
struct FakeClock {
    using time_point = std::chrono::steady_clock::time_point;
    static inline int ticks = 0;
    static time_point now() { return time_point(std::chrono::milliseconds(ticks++)); }
};
int main() {
    auto o = request({"export_sdk", "--operation", "benchmark", "--svo", "x", "--output", "y"});
    require(o.warmup == 5 && o.frames == 2 && o.timing_lane == "opaque-api", "benchmark defaults");
    auto d = request({"export_sdk", "--operation", "capture-diagnostic", "--svo", "x", "--output", "y"});
    require(d.warmup == 0 && d.frames == 2, "diagnostic defaults changed");
    require(request({"x", "--operation", "capture-diagnostic", "--svo", "x", "--output", "y", "--frames", "0001"}).frames == 1, "diagnostic integer parsing changed");
    for (auto value : {"-1", "101", "5x", "99999999999999999999999"}) {
        bool rejected = false;
        try { request({"x", "--operation", "benchmark", "--svo", "x", "--output", "y", "--warmup", value}); }
        catch (...) { rejected = true; }
        require(rejected, "bad warmup accepted");
    }
    for (auto value : {"0", "101", "-1"}) {
        bool rejected = false;
        try { request({"x", "--operation", "benchmark", "--svo", "x", "--output", "y", "--frames", value}); }
        catch (...) { rejected = true; }
        require(rejected, "bad frames accepted");
    }
    for (auto lane : {"gpu-resident", "host-output", "other"}) {
        bool rejected = false;
        try { request({"x", "--operation", "benchmark", "--svo", "x", "--output", "y", "--timing-lane", lane}); }
        catch (...) { rejected = true; }
        require(rejected, "unsupported lane accepted");
    }
    auto bounds = request({"x", "--operation", "benchmark", "--svo", "x", "--output", "y", "--warmup", "100", "--frames", "100"});
    require(bounds.warmup == 100 && bounds.frames == 100, "valid upper counts rejected");
    auto zero = request({"x", "--operation", "benchmark", "--svo", "x", "--output", "y", "--warmup", "0"});
    require(zero.warmup == 0, "explicit zero lost");
    int calls = 0;
    allocation_forbidden = true;
    auto result = sf::zed_benchmark::sample<int, FakeClock>(2, 3, [&] { ++calls; return calls == 5 ? 7 : 0; }, 0);
    allocation_forbidden = false;
    require(calls == 5 && result.warmup_completed == 2 && result.measured_count == 2 && result.failure == 7, "partial counts");
    auto report = sf::zed_benchmark::report(result, 2, 3, false);
    require(report["opaque_grab_api_ms"] == report["complete_wall_ms"], "different timing boundaries");
    require(report["opaque_grab_api_ms"] == nlohmann::json::array({1.0, 1.0}), "raw samples/order");
    require(report["qualification"] == "unqualified" && report["phase"] == "warm", "qualification/phase");
    require(report["requested_outputs"] == nlohmann::json::array({"no_depth_capture"}), "NONE mislabelled");
    require(report["completion_boundary"] == "opaque_api_return" && report["gpu_completion"]["available"] == false, "completion claim");
    calls = 0;
    auto failed_warmup = sf::zed_benchmark::sample<int, FakeClock>(2, 3, [&] { return ++calls == 2 ? 7 : 0; }, 0);
    require(failed_warmup.warmup_completed == 1 && failed_warmup.measured_count == 0 && calls == 2, "warmup failure continued");
    auto empty = sf::zed_benchmark::report(failed_warmup, 2, 3, true);
    require(empty["summary"]["available"] == false, "empty statistics invented");
    auto unwarmed = sf::zed_benchmark::sample<int, FakeClock>(0, 100, [] { return 0; }, 0);
    require(unwarmed.measured_count == 100, "upper bound");
    auto cold = sf::zed_benchmark::report(unwarmed, 0, 100, true);
    require(cold["phase"] == "unwarmed" && cold["warmup_completed"] == 0, "unwarmed phase");
    calls = 0;
    auto chronological = sf::zed_benchmark::sample<int, FakeClock>(0, 3, [&] { FakeClock::ticks += ++calls; return 0; }, 0);
    require(chronological.samples[0] == 2 && chronological.samples[1] == 3 && chronological.samples[2] == 4, "fake grab chronology");
    auto ordered = unwarmed;
    ordered.measured_count = 4;
    ordered.samples[0] = 4; ordered.samples[1] = 1; ordered.samples[2] = 3; ordered.samples[3] = 2;
    auto stats = sf::zed_benchmark::report(ordered, 0, 4, true);
    require(stats["opaque_grab_api_ms"] == nlohmann::json::array({4.,1.,3.,2.}), "chronology sorted");
    require(stats["summary"]["p50_ms"] == 2 && stats["summary"]["p95_ms"] == 4 && stats["summary"]["p99_ms"] == 4, "nearest rank");
}
'''.replace('PARSER', parser)
    cpp = tmp_path / 'probe.cpp'
    cpp.write_text(probe)
    binary = tmp_path / 'probe'
    subprocess.run([compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(json_include), '-I' + str(HEADER.parent), str(cpp), '-o', str(binary)],
                   check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_timing_loop_has_no_allocation_query_retrieval_or_io():
    text = HEADER.read_text()
    loop = text[text.index('for (int ordinal'):text.index('// Post-loop report')]
    for forbidden in ('push_back', 'resize', 'reserve', 'new ', 'getSVO', 'retrieve', 'Json',
                      'cout', 'write', 'settings', 'synchronize', 'getCUDA'):
        assert forbidden not in loop
    assert 'grab()' in loop and 'HostClock::now()' in loop
    assert 'std::array<double, 100>' in text
    source = SOURCE.read_text()
    branch = source[source.index('if (o.operation == "benchmark") {', source.index('diagnostic["initial_calibration"]')):
                    source.index('const std::array<sl::VIEW')]
    assert 'sf::zed_benchmark::sample' in branch
    assert branch.count('camera.grab(runtime)') == 1
    assert 'retrieve' not in branch
    assert source.index('sf::zed_benchmark::sample') < source.index('for (int ordinal = 0;')


def test_benchmark_returns_before_diagnostic_cpu_buffer_allocations():
    source = SOURCE.read_text()
    start = source.index('if (o.operation == "benchmark") {', source.index('diagnostic["initial_calibration"]'))
    end = source.index('return 0;', start)
    branch = source[start:end]
    for allocation in ('std::array<sl::Mat, 4> images', 'image.alloc(', 'sl::Mat depth, confidence',
                       'depth.alloc(', 'confidence.alloc(', 'std::vector<std::uint8_t> finite_positive'):
        assert end < source.index(allocation), allocation
    assert 'sf::zed_benchmark::sample' in branch
    assert 'diagnostic["cpu_buffer_prepare_ms"] = unavailable("no diagnostic CPU buffers requested")' in source
    assert '"cpu_buffer_prepare_ms", "not performed: no diagnostic CPU buffers requested"' in source
