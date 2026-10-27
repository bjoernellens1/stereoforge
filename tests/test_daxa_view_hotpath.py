"""CPU source guards for costly work at the retained Daxa enqueue boundary.

These deliberately inspect StereoForge code, including graph callbacks. Daxa
records commands internally on execute; its pooled recorders are not a promise
of zero allocations. One host Lease allocation per submitted frame is accepted
for this POC. GPU correctness, timing and allocator residency need native runs.
"""
from pathlib import Path
import re

import pytest

from test_daxa_hotpath import block, code_only


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/daxa_view.cpp"
BENCHMARK = ROOT / "apps/daxa_view_benchmark.cpp"


def execution_closure(source):
    yield "project", block(source, source.index("DaxaView::Frame DaxaView::project("))[0]
    setup = block(source, source.index("void DaxaView::Impl::setup("))[0]
    callbacks = list(re.finditer(r"\.executes\(\s*\[[^\]]*\]\s*\([^)]*\)\s*\{", setup))
    assert callbacks, "review the compositor callbacks if their declaration changes"
    for callback in callbacks:
        yield "callback", block(setup, callback.start())[0]
    boundaries = list(re.finditer(r"\.(?:pre|post)_task_callback\s*=\s*\[[^\]]*\]\s*\([^)]*\)\s*\{", setup))
    assert len(boundaries) == 2, "scheduled graph timestamp boundaries must also be reviewed"
    for callback in boundaries:
        yield "callback", block(setup, callback.start())[0]


def assert_no_costly_enqueue_work(source, *, lease_allowed=False):
    clean = code_only(source)
    if lease_allowed:
        leases = re.findall(r"\bstd::make_shared\s*<\s*Lease\s*>\s*\(", clean)
        assert len(leases) <= 1, "at most one frame-lease allocation per enqueue"
        clean = re.sub(r"\bstd::make_shared\s*<\s*Lease\s*>\s*\(", "accepted_lease(", clean)
    forbidden = (
        r"\b(?:wait\w*|daxa_dvc_wait\w*|vk\w*Wait\w*)\s*\(",
        r"\b(?:create_\w+|destroy_\w+|allocate|upload|readback|inspect|gpu_ms|shader|register_buffer|add_task|complete|submit|submit_commands)\s*\(",
        r"\b(?:TaskGraph|ExternalTaskBuffer|ifstream|ofstream|vector|map)\b",
        r"\b(?:new|malloc|calloc|realloc|make_shared|make_unique)\b|\b(?:resize|reserve|push_back|emplace_back)\s*\(",
        r"\b(?:copy_\w+|buffer_host_address\w*|memcpy|memmove|system|popen|get_query_results)\s*(?:<[^>]+>)?\s*\(",
    )
    for pattern in forbidden:
        assert not re.search(pattern, clean), f"costly compositor enqueue operation: {pattern}"


def test_project_and_graph_callbacks_do_not_allocate_planes_transfer_or_wait():
    for name, implementation in execution_closure(SOURCE.read_text()):
        assert_no_costly_enqueue_work(implementation, lease_allowed=name == "project")


@pytest.mark.parametrize("operation", (
    "device.wait_idle();", "frame.wait();", "device.wait_on_submit({});",
    "vkWaitForFences(device,1,&fence,true,10);", "device.create_buffer({});",
    "allocate(bytes,name);", "upload(recorder,name,data,size);", "readback(frame);", "inspect(frame);",
    "daxa::TaskGraph rebuilt({});", "device.create_compute_pipeline({});",
    "std::vector<float> pixels(count);", "std::memcpy(dst,src,bytes);",
    "device.buffer_host_address_as<float>(output);",
    "ti.recorder.copy_buffer_to_buffer({});", "slot.queries.get_query_results(0,2);",
    "std::make_shared<Lease>(Lease{});std::make_shared<Lease>(Lease{});",
))
def test_guard_rejects_expensive_operations_moved_into_either_execution_scope(operation):
    for name, implementation in execution_closure(SOURCE.read_text()):
        with pytest.raises(AssertionError):
            assert_no_costly_enqueue_work(operation + implementation, lease_allowed=name == "project")


def test_lease_exception_does_not_allow_callback_allocations():
    with pytest.raises(AssertionError):
        assert_no_costly_enqueue_work("std::make_shared<Lease>(Lease{});")


def test_enqueue_polls_completion_and_reclaims_daxa_recorder_pools_before_execution():
    source = SOURCE.read_text()
    project = block(source, source.index("DaxaView::Frame DaxaView::project("))[0]
    assert project.index("collect_garbage()") < project.index("oldest_pending_submit_index()") < project.index("execute({})")
    # Missing either half permits overwriting live output or uncompleted commands.
    before_execute = code_only(project[:project.index("execute({})")])
    assert "lease.expired()" in before_execute and re.search(r"s\.submit\s*>=\s*pending", before_execute)
    assert "e.slots.size()" in before_execute


def assert_readback_not_retained(source):
    readback = code_only(block(source, source.index("GeometryReadback DaxaView::readback("))[0])
    assert not re.search(r"\b(?:e|impl_)\s*(?:\.|->)\s*allocate\s*\(", readback), "diagnostic allocation retained until engine teardown"
    assert not re.search(r"\b(?:owned|memory)\s*\.\s*(?:push_back|emplace_back)\s*\(", readback), "diagnostic storage added to persistent ownership"
    assert "destroy_buffer(" in readback and "collect_garbage()" in readback, "diagnostic buffer must be reclaimed"


def test_diagnostic_readback_reclaims_storage_instead_of_growing_engine_ownership():
    assert_readback_not_retained(SOURCE.read_text())


def test_guard_rejects_original_persistently_retained_readback_allocation():
    source = SOURCE.read_text()
    start = source.index("GeometryReadback DaxaView::readback(")
    opening = source.index("{", start) + 1
    source = source[:opening] + 'e.allocate(size,"diagnostic",true);' + source[opening:]
    with pytest.raises(AssertionError, match="retained"):
        assert_readback_not_retained(source)


def test_gpu_timestamp_uses_actual_queue_capabilities_availability_and_valid_bits():
    source = code_only(SOURCE.read_text())
    assert "daxa_dvc_get_vk_queue(" in source and "timestampValidBits" in source
    timing = block(source, source.index("double DaxaView::Frame::gpu_ms("))[0]
    assert timing.index("oldest_pending_submit_index()") < timing.index("get_query_results(")
    assert re.search(r"lease_->submit\s*<\s*e.context.device.oldest_pending_submit_index\(\)", timing)
    assert "values[1]" in timing and "values[3]" in timing
    assert "timestamp_bits==64" in timing and "&mask" in timing and "timestamp_period" in timing


def test_timestamps_follow_scheduled_callbacks_instead_of_reorderable_pass_numbers():
    source = SOURCE.read_text()
    setup = block(source, source.index("void DaxaView::Impl::setup("))[0]
    compute = list(re.finditer(r"\.executes\(\s*\[[^\]]*\]\s*\([^)]*\)\s*\{", setup))
    for callback in compute:
        assert "write_timestamp(" not in code_only(block(setup, callback.start())[0]), "an independent reordered pass cannot define the whole graph timing boundary"
    pre = block(setup, setup.index(".pre_task_callback="))[0]
    post = block(setup, setup.index(".post_task_callback="))[0]
    assert "visited++==0" in code_only(pre) and ".query_index=0" in pre
    assert "++timed_slot->finished==8" in code_only(post) and ".query_index=1" in post
    project = block(source, source.index("DaxaView::Frame DaxaView::project("))[0]
    assert project.index("s.visited=0") < project.index("execute({})")
    assert project.index("s.finished=0") < project.index("execute({})")
    assert "s.visited==8&&s.finished==8" in code_only(project), "changed task count must fail instead of publishing incomplete timing"


def test_benchmark_keeps_upload_setup_dumps_and_query_readout_outside_cpu_enqueue():
    source = BENCHMARK.read_text()
    measured = block(source, source.index("for(unsigned i=0;i<opt.frames;++i)"))[0]
    enqueue = measured[measured.index("const auto start="):measured.index("const auto submitted=")]
    # Keep the boundary honest even when equivalent expensive work is renamed.
    assert_no_costly_enqueue_work(enqueue)
    assert "project()" in enqueue
    assert measured.index("const auto submitted=") < measured.index("->wait()") < measured.index("const auto completed=") < measured.index("->gpu_ms()")
    assert "readback(" not in code_only(measured) and "dump(" not in code_only(measured)
    assert source.index("samples.reserve(opt.frames)") < source.index("for(unsigned i=0;i<opt.frames;++i)")
    assert source.index("std::make_unique<DaxaView>(") < source.index("for(unsigned i=0;i<opt.frames;++i)")


def assert_benchmark_checks_retained_growth(source):
    measured = block(source, source.index("for(unsigned i=0;i<opt.frames;++i)"))[0]
    check = re.search(r"if\s*\(\s*engine->allocated_bytes\(\)\s*!=\s*setup_bytes\s*\)\s*throw\b", code_only(measured))
    assert check, "benchmark must fail on retained buffer growth during every sample"
    assert source.index("setup_bytes=engine->allocated_bytes()") < source.index("for(unsigned i=0;i<opt.warmup;++i)")
    assert "<<steady_bytes" in source and "<<setup_bytes" in source


def test_benchmark_checks_growth_instead_of_only_printing_a_constant_claim():
    assert_benchmark_checks_retained_growth(BENCHMARK.read_text())


def test_guard_rejects_removed_memory_growth_failure():
    source = BENCHMARK.read_text()
    sample_start = source.index("for(unsigned i=0;i<opt.frames;++i)")
    source = source[:sample_start] + source[sample_start:].replace("if(engine->allocated_bytes()!=setup_bytes)", "if(false)", 1)
    with pytest.raises(AssertionError, match="growth"):
        assert_benchmark_checks_retained_growth(source)
