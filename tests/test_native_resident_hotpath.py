"""CPU-only source guards for the native resident compositor boundary.

These are structural regression checks, not dispatch timing or Vulkan validation.
The checked closure deliberately includes project helpers: moving a transfer into
encode_project/acquire must not make it invisible. Constant-size command, binding,
and ownership bookkeeping and lazy two-slot buffer allocation are permitted.
"""
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "src/geometry.cpp"
RESIDENT = "GeometryFrame NativeGeometry::project(const GeometryDeviceSources&"
HELPERS = (
    "Slot& acquire(",
    "rhi::IBuffer* buffer(",
    "void outputs(",
    "void dispatch(",
    "static void set(",
    "GeometryFrame submit(",
    "void encode_project(",
    "void validate_view(",
    "void validate_device_sources(",
    "void validate_camera(",
    "void validate_pose(",
    "void checked(",
)


def code_only(source):
    """Blank comments/literals while retaining offsets for balanced C++ braces."""
    tokens = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(tokens, lambda match: " " * len(match[0]), source)


def body(source, marker):
    clean = code_only(source)
    # Skip default arguments such as inputs={} before looking for the body.
    arguments = clean.index("(", source.index(marker))
    end_arguments, depth = arguments + 1, 1
    while depth:
        depth += (clean[end_arguments] == "(") - (clean[end_arguments] == ")")
        end_arguments += 1
    start = clean.index("{", end_arguments)
    depth = 1
    end = start + 1
    while depth:
        depth += (clean[end] == "{") - (clean[end] == "}")
        end += 1
    return source[start + 1:end - 1], start + 1, end


def assert_no_host_image_work(source):
    clean = code_only(source)
    # getCurrentValue is a nonblocking fence query and is intentionally allowed.
    forbidden = (
        r"\b(?:upload|upload_sources|upload_temporal|uploadBufferData|readBuffer|readback|mapBuffer|unmapBuffer)\s*\(",
        r"\b(?:wait|waitOnHost|waitForFences|waitIdle|vkQueueWaitIdle|vkDeviceWaitIdle)\s*\(",
        r"\b(?:memcpy|memmove|copy|copy_n|transform|for_each)\s*\(",
        r"\b(?:createDevice|createShaderProgram|createComputePipeline|loadModuleFromSourceString)\s*\(",
        r"(?:src|sources|resident)\s*(?:\.|->)\s*(?:depth|rgb|rays|confidence|provenance|source_ids|age_ns|preview_indices)\s*(?:\[|\.\s*(?:begin|end)\s*\()",
        r"\b(?:resize|reserve)\s*\([^;]*(?:\bn\b|\bm\b|\bcount\b|width|height|output_capacity)",
    )
    for pattern in forbidden:
        assert not re.search(pattern, clean), f"resident path contains host work matching {pattern}"
    for loop in re.findall(r"\b(?:for|while)\s*\(([^)]*)\)", clean):
        assert not re.search(r"\b(?:n|m|count|output_capacity|source_width|source_height|target_width|target_height)\b", loop), f"image-sized host loop: {loop}"


def resident_closure(source):
    return [(marker, body(source, marker)[0]) for marker in (RESIDENT, *HELPERS)]


def test_resident_project_and_helpers_have_no_host_transfers_waits_or_pixel_work():
    source = GEOMETRY.read_text()
    for marker, implementation in resident_closure(source):
        try:
            assert_no_host_image_work(implementation)
        except AssertionError as error:
            raise AssertionError(f"{marker}: {error}") from error
    contracts = (ROOT / "include/stereoforge/contracts.hpp").read_text()
    assert_no_host_image_work(body(contracts, "void validate(std::uint64_t expected_device) const")[0])


@pytest.mark.parametrize("marker", (RESIDENT, "Slot& acquire(", "void encode_project(", "void dispatch(", "GeometryFrame submit("))
@pytest.mark.parametrize("operation", (
    "sources.completion.wait();",
    "device->readBuffer(buffer,0,bytes,pixels);",
    "encoder->uploadBufferData(buffer,0,bytes,pixels);",
    "std::memcpy(pixels,input,bytes);",
    "for(size_t i=0;i<count;++i){}",
    "metadata->previous_ids={sources.source_ids.begin(),sources.source_ids.end()};",
))
def test_guard_detects_regressions_in_entry_and_helpers(marker, operation):
    source = GEOMETRY.read_text()
    _, insertion, _ = body(source, marker)
    mutated = source[:insertion] + operation + source[insertion:]
    with pytest.raises(AssertionError):
        for _, implementation in resident_closure(mutated):
            assert_no_host_image_work(implementation)


def test_resident_path_retains_views_and_gathers_metric_identity_on_gpu():
    source = GEOMETRY.read_text()
    resident = body(source, RESIDENT)[0]
    for field in ("depth", "confidence", "rgb", "provenance", "rays", "source_ids", "age_ns", "preview_indices"):
        assert f"sources.{field}" in resident
    assert "std::move(inputs)" in resident
    submit = body(source, "GeometryFrame submit(")[0]
    assert "std::make_shared<Lease>" in submit and "std::move(inputs)" in submit
    assert 'view("output_source_ids",count,8)' in submit
    assert 'view("output_age_ns",count,8)' in submit
    assert '"project.gather_project_metadata"' in body(source, "void encode_project(")[0]
    # Allocation may occur once per slot/resource, not every enqueue.
    allocation = body(source, "rhi::IBuffer* buffer(")[0]
    lazy, _, _ = body(allocation, "if(!slot.buffers.contains(name))")
    assert allocation.count("createBuffer(") == lazy.count("createBuffer(") == 1
    assert re.search(r"std::array<Slot,\s*2>\s+slots", source)


def test_resident_fixture_upload_and_wait_precede_benchmark_measurement():
    source = (ROOT / "apps/native_view_benchmark.cpp").read_text()
    setup, _, setup_end = body(source, 'if(opt.input=="resident")')
    assert setup.count("engine.upload_sources(sources)") == 1
    assert setup.count("resident->completion.wait()") == 1
    assert setup_end < source.index("for(unsigned i=0;i<opt.warmup;++i)")
    measured, _, measured_end = body(source, "for(unsigned i=0;i<opt.frames;++i)")
    assert "last=project()" in measured
    assert "upload_sources(" not in measured
    assert measured.count("completion.wait()") == 1
    assert measured.count("engine.readback(") == 1
    assert 'if(opt.readback=="cpu")last_cpu=engine.readback(*last)' in measured
    assert measured.index("last.reset()") < measured.index("const auto start=Clock::now()")
    assert "ms(start,submitted)" in measured and "ms(submitted,completed)" in measured
    assert "dump(" not in measured and "dump(" in source[measured_end:]
    assert 'resident?engine.project(*resident,input.view):engine.project(sources,input.view)' in source
    assert "initial_upload_enqueue_ms" in source and "initial_upload_wait_ms" in source


def test_shared_queue_handoff_has_no_intermediate_cpu_wait_or_plane_download():
    source = (ROOT / "tests/native_gpu_handoff.cpp").read_text()
    chain, _, _ = body(source, "for(std::uint32_t frame_index=1;")
    before_final_wait = chain[:chain.index("device->waitForFences(")]
    assert_no_host_image_work(before_final_wait)
    assert "geometry.project(source,view)" in before_final_wait
    assert chain.count("device->waitForFences(") == 1
    assert chain.count("device->readBuffer(") == 1
    assert "std::array<std::uint32_t,4> values" in chain
    assert "readBuffer(counts,0,sizeof(values),values.data())" in chain
    assert source.index("geometry.upload_sources(host)") < source.index("for(std::uint32_t frame_index=1;")
