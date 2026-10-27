"""CPU-only launch/boundary guards; these do not measure GPU performance."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def body(source, marker):
    start = source.index("{", source.index(marker))
    end, depth = start + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1:end - 1], start, end


def test_native_select_group_count_still_covers_64_pixels():
    source = (ROOT / "src/runtime.cpp").read_text()
    calls = re.findall(r'dispatch\(encoder,\s*slot,\s*"select",\s*std::uint32_t\(\(count\+63\)/64\)\)', source)
    assert len(calls) == source.count('"select", std::uint32_t(') == 1
    shader = (ROOT / "python/stereoforge/shaders/stereo/sgm.slang").read_text()
    select = shader[shader.index("void select("):shader.index("// Optional metric output.")]
    assert "i=group_id.x*64+group_thread_id.x/4" in select
    assert "lane=group_thread_id.x%4" in select
    # The CPU issues one selection pass, independent of disparity count.
    assert "dispatchCompute(groups, 1, 1)" in source


def test_python_select_stays_in_existing_asynchronous_command_buffer():
    source = (ROOT / "python/stereoforge/gpu.py").read_text()
    enqueue = source[source.index("    def enqueue("):source.index("    def result(")]
    select = re.findall(r"self\.kernels\['select'\]\.dispatch\(([^\n]+)", enqueue)
    assert len(select) == 1
    assert select[0].startswith("[n*4,1,1]")
    assert "command_encoder=encoder" in select[0]
    assert "query_index_before=query_index" in select[0]
    assert "query_index_after=query_index+1" in select[0]
    assert enqueue.count("submit_command_buffer(") == 1
    assert "wait_for_submit(" not in enqueue and "to_numpy(" not in enqueue


def test_select_microbenchmark_keeps_setup_and_output_download_outside_samples():
    source = (ROOT / "apps/native_select_benchmark.cpp").read_text()
    measured, start, end = body(source, "for(unsigned iteration=0;")
    setup = source[:start]
    assert "uploadBufferData(" in setup
    assert "wait(device,fence,generation)" in setup
    assert "samples.reserve(opt.frames)" in setup
    assert "finish reusable select command" in setup
    assert "pass->dispatchCompute(std::uint32_t((n+63)/64),1,1)" in setup
    for forbidden in ("createBuffer(", "createCommandEncoder(", "createComputePipeline(",
                      "uploadBufferData(", "readBuffer(", "std::vector", "costs(",
                      "fingerprint(", "for(", "while("):
        assert forbidden not in measured
    assert measured.count("submit(queue,command,fence,++generation)") == 1
    assert measured.count("wait(device,fence,generation)") == 1
    assert measured.index("wait(device,fence,generation)") < measured.index("queries->getResult(")
    assert "if(iteration>=opt.warmup)samples.push_back(" in measured
    assert "readBuffer(" in source[end:]
    assert "ms(start,submitted)" in measured and "ms(submitted,completed)" in measured


def test_select_microbenchmark_cost_volume_bound_cannot_overflow():
    source = (ROOT / "apps/native_select_benchmark.cpp").read_text()
    options = body(source, "static Options options(")[0]
    # n itself fits uint64, but n*D can wrap for uint32 width/height.
    assert "n>UINT32_MAX/out.disparities" in options
