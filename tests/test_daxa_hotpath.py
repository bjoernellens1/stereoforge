"""Host-only Daxa boundary guards; these do not qualify GPU timing or correctness.

The closure includes StereoForge callbacks executed by TaskGraph, not Daxa's
internal recorder allocation. Benchmark batch backpressure is separate from
enqueue samples; final counter completion/readback is an explicit boundary.
"""
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/daxa_foundation.cpp"


def code_only(source):
    tokens = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|(?<![\w])\'(?:\\.|[^\'\\])*\''
    return re.sub(tokens, lambda match: " " * len(match[0]), source)


def block(source, start):
    clean = code_only(source)
    opening = clean.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (clean[end] == "{") - (clean[end] == "}")
        end += 1
    return source[opening + 1:end - 1], opening + 1, end


def assert_hotpath(source):
    clean = code_only(source)
    forbidden = (
        r"\b(?:wait|wait_idle|wait_for_value|wait_on_submit|daxa_dvc_wait_idle|daxa_dvc_wait_on_submit|vkDeviceWaitIdle|vkQueueWaitIdle|vkWaitForFences|vkWaitSemaphores)\s*\(",
        r"\b(?:create_\w+|destroy_\w+|register_buffer|add_task|complete|submit|shader)\s*\(",
        r"\b(?:TaskGraph|ExternalTaskBuffer|FoundationBuffers|ifstream|ofstream|vector)\b",
        r"\b(?:new|malloc|calloc|realloc|make_shared|make_unique)\b|\b(?:resize|reserve)\s*\(",
        r"\b(?:copy_\w+|buffer_host_address\w*|memcpy|memmove|system|popen)\s*(?:<[^>]+>)?\s*\(",
        r"\b(?:for|while)\s*\(",
    )
    for pattern in forbidden:
        assert not re.search(pattern, clean), f"Daxa hot path contains forbidden work: {pattern}"


def hotpath_closure(source):
    assert "void execute_foundation_frame(" in source, "review the repeated execution helper"
    helper = source.index("void execute_foundation_frame(")
    yield block(source, helper)[0]
    run = source.index("void run_daxa_foundation(")
    repeated = source.index("for(frame=1;frame<=3;++frame)", run)
    yield block(source, repeated)[0]
    # Check both GPU callbacks: moving a wait/allocation into a callback must fail.
    graph_setup_end = source.index("graph.submit({})", run)
    callbacks = list(re.finditer(r"\.executes\(\s*\[[^\]]*\]\s*\([^)]*\)\s*\{", source[run:graph_setup_end]))
    assert len(callbacks) == 2, "both write and consume callbacks must be reviewed"
    for callback in callbacks:
        yield block(source, run + callback.start())[0]


def test_repeated_graph_execution_and_callbacks_have_no_host_work():
    source = SOURCE.read_text()
    for implementation in hotpath_closure(source):
        assert_hotpath(implementation)


@pytest.mark.parametrize("operation", (
    "device.wait_idle();",
    "completion.wait_for_value(1);",
    "device.wait_on_submit({});",
    "daxa_dvc_wait_idle(device.get());",
    "device.create_buffer({});",
    "device.create_compute_pipeline({});",
    "daxa::TaskGraph rebuild({});",
    "shader();",
    "std::vector<unsigned> pixels(count);",
    "std::memcpy(output,input,bytes);",
    "device.buffer_host_address_as<unsigned>(readback);",
    "ti.recorder.copy_buffer_to_buffer({});",
    "for(unsigned i=0;i<count;++i){}",
))
def test_guard_rejects_regressions_in_execution_closure(operation):
    source = SOURCE.read_text()
    for implementation in hotpath_closure(source):
        with pytest.raises(AssertionError):
            assert_hotpath(operation + implementation)


def test_pipeline_and_external_resource_setup_precede_repeated_execution():
    source = SOURCE.read_text()
    run = block(source, source.index("void run_daxa_foundation("))[0]
    repeated = run.index("for(frame=1;frame<=3;++frame)")
    setup = run[:repeated]
    for operation in ("shader()", "create_compute_pipeline(", "create_buffer(",
                      "register_buffer(resident_external)", "graph.complete({})"):
        assert operation in setup
    loop, _, _ = block(run, repeated)
    assert re.search(r"execute_foundation_frame\(graph(?:,\s*device)?\)", loop)
    assert_hotpath(loop)


def test_benchmark_excludes_setup_and_keeps_backpressure_outside_enqueue_samples():
    source = SOURCE.read_text()
    start = source.index("measurement_start=std::chrono::steady_clock::now()")
    final_wait = source.index("device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=final_submit", start)
    measured = code_only(source[start:final_wait])
    for forbidden in (r"\bcreate_\w+\s*\(", r"\b(?:TaskGraph|ExternalTaskBuffer)\b",
                      r"\b(?:register_buffer|add_task|complete|submit|wait_idle|daxa_dvc_wait_idle)\s*\("):
        assert not re.search(forbidden, measured), "cold setup or intermediate wait entered timing"
    assert source.index("enqueue_us.reserve(300)") < start
    sample, _, _ = block(source, source.index("for(uint32_t sample=0;sample<300;++sample)"))
    backpressure = sample.index("if((sample+1)%max_inflight_frames==0)")
    assert_hotpath(sample[:backpressure])
    assert sample.index("auto start=") < sample.index("execute_foundation_frame(") < sample.index("auto end=") < backpressure
    batch, _, _ = block(sample, backpressure)
    assert sample.count("device.wait_on_submit(") == batch.count("device.wait_on_submit(") == 1
    assert batch.index("auto wait_start=") < batch.index("device.wait_on_submit(") < batch.index("auto wait_end=")
    assert batch.index("auto wait_end=") < batch.index("device.collect_garbage()") < batch.index("auto reclaim_end=")
    assert "(wait_end-wait_start).count()" in batch and "(reclaim_end-wait_end).count()" in batch
    assert "++backpressure_wait_count" in batch
    assert_hotpath(re.sub(r"device\.wait_on_submit\(\{[^}]*\}\);", "", batch))
    cap = re.search(r"constexpr uint32_t max_inflight_frames=(\d+)", source)
    assert 0 < int(cap[1]) < 128, "batch size must leave room below Daxa's command pool cap"
    output = source[source.index("std::cout <<", final_wait):]
    for metric in ("backpressure_wait_count", "backpressure_wait_ms", "batch_reclaim_cpu_ms"):
        assert f"<< {metric}" in output
    assert source.index("buffer_host_address_as<uint32_t>(readback)") > final_wait


def test_each_timed_enqueue_recycles_completed_daxa_command_pools():
    source = SOURCE.read_text()
    helper = block(source, source.index("void execute_foundation_frame("))[0]
    assert helper.count("collect_garbage()") == 1, "retired Daxa command pools must be recycled during repeated execution"
    assert helper.index("graph.execute({})") < helper.index("collect_garbage()")


def test_final_wait_targets_the_submission_after_the_counter_copy():
    source = SOURCE.read_text()
    before = source.index("auto before_final_submit=device.latest_queue_submit_index(daxa::QUEUE_MAIN)")
    executed = source.index("final.execute({})", before)
    submitted = source.index("auto final_submit=device.latest_queue_submit_index(daxa::QUEUE_MAIN)", executed)
    wait = source.index("device.wait_on_submit(", submitted)
    assert "check(final_submit>before_final_submit," in source[submitted:wait]
    assert re.search(r"device\.wait_on_submit\(\{\.queue=daxa::QUEUE_MAIN,\.queue_submit_index=final_submit,\.timeout=30'000'000'000ull\}\)", source[wait:])
    assert source.index("owned.completion_reached=true", wait) < source.index("buffer_host_address_as<uint32_t>(readback)", wait)
    assert "additional_signal_timeline_semaphores" not in source


def test_external_byte_range_uses_callback_resolved_native_buffer_after_completion():
    source = SOURCE.read_text()
    resolved = source.index("observed_resident=ti.get(view).id")
    identity = source.index("check(observed_resident==resident,", resolved)
    completed = source.index("device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=final_submit", identity)
    byte_range = source.index("device.buffer_info(observed_resident).value().size==64", completed)
    assert byte_range < source.index("device.destroy_buffer(resident)", byte_range)
    assert "task_buffer_info(view)" not in code_only(source)


def test_bound_readback_buffer_has_zero_flags_while_memory_block_stays_coherent():
    source = SOURCE.read_text()
    allocation = source.index("device.create_memory(", source.index("void run_daxa_foundation("))
    binding = source.index("device.create_buffer_from_memory_block(", allocation)
    setup = source[allocation:binding]
    assert "context.coherent_readback_requirements(readback_info)" in setup
    assert ".flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM" in setup
    assert_readback_buffer_flags(source, binding)


def assert_readback_buffer_flags(source, binding):
    declaration = re.search(r"\bBufferInfo\s+readback_info\s*\{(.*?)\};", code_only(source), re.S)
    initial = re.search(r"\.memory_flags\s*=\s*([^,]+)", declaration[1])
    flags = initial[1] if initial else "{}"
    assignments = re.findall(r"readback_info\.memory_flags\s*=\s*([^;]+);", code_only(source[declaration.end():binding]))
    if assignments:
        flags = assignments[-1]
    assert re.sub(r"\s+", "", flags) in ("{}", "daxa::MemoryFlagBits::NONE"), (
        "Daxa requires zero buffer allocation flags when binding a memory block"
    )


def test_guard_rejects_original_host_access_flags_on_bound_buffer():
    source = SOURCE.read_text().replace("readback_info{.size=12,", "readback_info{.size=12,.memory_flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,", 1)
    binding = source.index("device.create_buffer_from_memory_block(")
    with pytest.raises(AssertionError, match="zero buffer allocation flags"):
        assert_readback_buffer_flags(source, binding)


def test_raw_buffer_cleanup_owner_precedes_graphs_and_waits_only_on_incomplete_work():
    source = SOURCE.read_text()
    run = block(source, source.index("void run_daxa_foundation("))[0]
    owner = run.index("} owned{device};")
    assert owner < run.index("device.create_buffer(") < run.index("daxa::TaskGraph graph(")
    for name, slot, operation in (("resident", 0, "create_buffer"), ("counters", 1, "create_buffer"),
                                  ("readback", 2, "create_buffer_from_memory_block"), ("replacement", 3, "create_buffer")):
        assert re.search(rf"auto\s+{name}\s*=\s*owned\.ids\[{slot}\]\s*=\s*device\.{operation}\(", run)
    cleanup = block(run, run.index("~FoundationBuffers("))[0]
    failure = block(cleanup, cleanup.index("if(!completion_reached)"))[0]
    assert cleanup.count("daxa_dvc_wait_idle(") == failure.count("daxa_dvc_wait_idle(") == 1
    assert "device.is_id_valid(id)" in cleanup
    assert cleanup.index("device.destroy_buffer(id)") < cleanup.index("device.collect_garbage()")
    assert run.index("device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=final_submit") < run.index("owned.completion_reached=true")
    # The memory block is an owner member, so it releases after the destructor body.
    assert "daxa::MemoryBlock readback_memory;" in run[:owner]
    assert ".memory_block=owned.readback_memory" in run


@pytest.mark.parametrize("path", ("apps/daxa_probe.cpp", "tests/daxa_foundation.cpp"))
def test_cli_success_checks_validation_after_explicit_context_teardown(path):
    source = (ROOT / path).read_text()
    assert "context.close()" in source, "device teardown must precede the final validation gate"
    close = source.index("context.close()")
    success = source.index("return 0", close)
    for counter in ("validation_errors.load()", "validation_warnings.load()"):
        assert counter in source[close:success], "late teardown diagnostics must fail the process"


def test_explicit_close_keeps_validation_messenger_alive_during_device_destruction():
    source = SOURCE.read_text()
    close = block(source, source.index("void DaxaFoundation::close("))[0]
    assert "if(!device.get()) return" in close
    assert "device={}" in close and "messenger={}" not in close
    destructor = block(source, source.index("DaxaFoundation::~DaxaFoundation("))[0]
    assert destructor.index("close()") < destructor.index("destroy(vk_instance,messenger,nullptr)")
