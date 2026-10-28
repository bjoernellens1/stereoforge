"""Guard the packed-YUYV full-frame reuse path without dispatching a GPU."""
from pathlib import Path
import re

import pytest
from test_kernel_variant_hotpath import body

SOURCE = Path(__file__).resolve().parents[1] / 'src/daxa_refinement.cpp'
PROCESS = 'DaxaRefinement::Frame DaxaRefinement::process(DaxaDepthPipeline::Frame const& input,DaxaFrame const& full)'


def check_reuse(source):
    process = body(source, PROCESS)
    assert 'source->source_kind==e.full_source_kind' in process
    assert process.index('source->source_kind==e.full_source_kind') < process.index('oldest_pending_submit_index(')
    assert 'full.calibration()==e.maps' in process and 'source==full.source()' in process
    assert 's->input=input;s->full_input=full' in process
    assert process.index('s->full_input=full') < process.index('graph->execute(')
    assert 'for(unsigned p=0;p<6;++p)s->full_nodes[p].set_buffer(full_buffers[p])' in process
    assert not re.search(r'\b(?:wait_idle|wait_on_submit|readback|create_buffer|create_memory|malloc|calloc|realloc|resize|reserve|push_back|memcpy|memmove)\s*\(', process)
    pack = body(source, '.executes([raw,copies,full_views,out]')
    barrier = 'src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ'
    assert barrier in pack and pack.index(barrier) < pack.index('copy_buffer_to_buffer(')
    compute = body(source, '.executes([this,raw,address,pass,count]')
    assert 'if(canonical&&pass==1)' in compute
    conditional = compute.split('if(canonical&&pass==1)', 1)[1].split('ti.recorder.write_timestamp', 1)[0]
    assert 'src_access=daxa::AccessConsts::HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ' in conditional


def test_reuse_hotpath_contract():
    check_reuse(SOURCE.read_text())


@pytest.mark.parametrize('before,after', [
    ('source->source_kind==e.full_source_kind', 'true'),
    ('s->full_input=full', 's->full_input={}'),
    ('COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ', 'COMPUTE_SHADER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ'),
    ('HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE', 'HOST_WRITE'),
    ('auto& e=*impl_;e.require_thread();require(e.canonical', 'auto& e=*impl_;e.require_thread();e.context->device.wait_idle();require(e.canonical'),
])
def test_reuse_guard_rejects_regressions(before, after):
    source = SOURCE.read_text()
    assert before in source
    # The host/compute barrier also occurs in pass zero; mutate the final occurrence.
    if before == 'HOST_WRITE|daxa::AccessConsts::COMPUTE_SHADER_WRITE':
        pos = source.rfind(before)
        source = source[:pos] + source[pos:].replace(before, after, 1)
    else:
        source = source.replace(before, after, 1)
    with pytest.raises(AssertionError):
        check_reuse(source)
