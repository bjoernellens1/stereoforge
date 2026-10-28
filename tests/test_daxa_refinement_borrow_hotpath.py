"""CPU control-path and static guards for opt-in GPU frame borrowing.

The executing probe extracts production methods and substitutes a fake device;
its timings are host bookkeeping only, not Vulkan or camera performance.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest
from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/daxa_refinement.cpp'
PROCESS = 'DaxaRefinement::Frame DaxaRefinement::process(DaxaDepthPipeline::Frame const& input,DaxaFrame const& full)'


def check_borrow(source):
    process = body(source, PROCESS)
    assert not re.search(r'\b(?:wait\w*|readback\w*|create_\w*|malloc|calloc|realloc|resize|reserve|push_back|emplace_back|memcpy|memmove)\s*\(', process)
    assert process.count('oldest_pending_submit_index(') == 1
    assert 'if(e.config.borrow_full_frame)for(unsigned p=0;p<4;++p)params->borrowed[p]=d.device_address(full_buffers[p+2]).value()' in process
    assert 's->input=input;s->full_input=full' in process
    assert process.index('s->input=input;s->full_input=full') < process.index('graph->execute(')
    assert 'catch(...){s->submit=d.latest_queue_submit_index(daxa::QUEUE_MAIN);throw;}' in process
    assert 'if(canonical&&!cfg.borrow_full_frame)' in source
    assert 'for(unsigned p=2;p<6;++p)task.reads(full_views[p])' in source
    assert 'if(owner_->config.borrow_full_frame)' in body(source, 'daxa::BufferId DaxaRefinement::Frame::buffer(')


def test_borrow_static_contract():
    check_borrow(SOURCE.read_text())


@pytest.mark.parametrize('mutation', ['wait', 'readback', 'allocation', 'retention', 'address'])
def test_borrow_guard_detects_regressions(mutation):
    source = SOURCE.read_text()
    anchor = 'auto& e=*impl_;e.require_thread();require(e.canonical'
    assert anchor in source
    if mutation in ('wait', 'readback', 'allocation'):
        action = {'wait':'e.context->device.wait_idle();', 'readback':'readback(input);', 'allocation':'e.context->device.create_buffer({});'}[mutation]
        source = source.replace(anchor, 'auto& e=*impl_;e.require_thread();' + action + 'require(e.canonical', 1)
    elif mutation == 'retention':
        source = source.replace('s->input=input;s->full_input=full', 's->input=input;s->full_input={}', 1)
    else:
        source = source.replace('params->borrowed[p]=d.device_address(full_buffers[p+2]).value()', 'params->borrowed[p]=0', 1)
    with pytest.raises(AssertionError):
        check_borrow(source)


def test_borrow_control_actual_source_fake_gpu(tmp_path):
    # Build the existing production-source fake-GPU harness with the borrow
    # configuration enabled only in the canonical hot loop. Its fake Impl is
    # extended when needed; no production method is replaced.
    path = ROOT / 'tests/test_daxa_refinement_hotpath.py'
    harness = path.read_text()
    stub = ' bool canonical=false;CaptureSourceKind full_source_kind=CaptureSourceKind::PackedYuyv;'
    if 'DaxaRefinementConfig config{};' not in harness:
        assert stub in harness
        harness = harness.replace(stub, stub + 'DaxaRefinementConfig config{};', 1)
    marker = 'auto canonical_polls=context->device.polls;allocations=0;track=true;for(unsigned n=0;n<10000;++n)'
    assert marker in harness
    harness = harness.replace(marker, 'engine->config.borrow_full_frame=true;auto borrow_start=std::chrono::steady_clock::now();' + marker, 1)
    before = 'auto out=stage.process(inputs[0],full);auto lease=out.observations();finish();'
    after = 'auto out=stage.process(inputs[0],full);auto lease=out.observations();for(unsigned p=0;p<4;++p)check(params[0].borrowed[p]==full.buffers[unsigned(p<2?(p==0?DaxaPreprocessPlane::LeftLuma:DaxaPreprocessPlane::RightLuma):(p==2?DaxaPreprocessPlane::LeftSupport:DaxaPreprocessPlane::RightSupport))].id,"borrowed address not current");finish();'
    assert before in harness
    harness = harness.replace(before, after, 1)
    end = '"canonical hotpath allocation/wait/poll regression");stage.collect();'
    assert end in harness
    harness = harness.replace(end, end + r'''
 double borrow_ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-borrow_start).count()/10000;
 std::cout<<"{\"borrow_frames\":10000,\"cpu_ns_per_frame\":"<<borrow_ns<<",\"application_allocations\":"<<allocations<<",\"application_waits\":"<<context->device.waits<<",\"scope\":\"actual production borrow control, fake GPU, address checks included\"}\n";
 engine->config.borrow_full_frame=false;
''', 1)
    namespace = {'__file__': str(path), '__name__': 'borrow_probe_harness'}
    exec(compile(harness, str(path), 'exec'), namespace)
    binary = namespace['build_probe'](tmp_path)
    run = subprocess.run([str(binary), '24'], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    row = json.loads(run.stdout.splitlines()[0])
    print('Borrow CPU-only probe:', json.dumps(row, sort_keys=True))
    assert row['borrow_frames'] == 10000
    assert row['application_allocations'] == row['application_waits'] == 0
