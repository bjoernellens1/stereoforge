"""CPU-only independent shared-Census resource and submission cost guards.

The reused probe extracts production host methods and replaces GPU calls only.
Its timings include the real packed staging copy, not driver/GPU execution.
"""
import json
from pathlib import Path
import re
import subprocess

import pytest

from test_kernel_variant_hotpath import body
from test_daxa_depth_pair_hotpath import (
    build_pair_bookkeeping_probe,
    check_pair_depth_admission,
    check_pair_hotpath,
)

ROOT = Path(__file__).resolve().parents[1]


def check_shared_hotpath(stereo, depth):
    check_pair_hotpath(stereo)
    check_pair_depth_admission(depth)
    hot = ''.join(body(stereo, signature) for signature in (
        'unsigned DaxaStereo::pair_slot_at(',
        'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(',
        'size_t DaxaStereo::collect_completed_inputs_at('))
    hot += body(depth, 'DaxaDepthPipeline::Frame DaxaDepthPipeline::enqueue(')
    hot += body(depth, 'size_t collect_at(')
    assert not re.search(r'\b(?:sha256|metric_calibration|sha_file|strict_json|create_\w+|wait_\w+|readback\w*|inspect)\s*\(|\.geometry|\b[RP][12]\[|std::vector', hot), 'no new GPU synchronization, readback, allocation or setup geometry work'
    process = body(stereo, 'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(')
    assert process.count('for(') == 2 and process.count('while(') == 0, 'submission retains only the bounded two-eye/two-image loops'
    assert 'for(unsigned eye=0;eye<2;++eye)' in process
    assert 'for(unsigned image=0;image<2;++image)' in process
    assert 'pair.started==l.paired->stages&&pair.finished==l.paired->stages' in process


def check_shared_graph(stereo, header):
    setup = body(stereo, 'void DaxaStereo::setup_pair(')
    shared = body(setup, 'if(shared_census)')
    single = body(shared, 'auto single=')
    assert 'pair.stages=shared_census?6+2*a.paths:l.stages' in setup, 'shared schedule needs14/22 phases for P4/P8'
    assert 'pair.query_count=2+2*pair.stages' in setup
    assert 'query_count=pair.query_count' in setup and 'finished==pair.stages' in setup
    assert re.search(r'max_stages\s*=\s*22\b', header), 'timestamp arrays must fit22 phases'
    assert 'shared_census&&resource>=LeftDesc&&resource<=RightValid' in setup, 'shared physical descriptor IDs register once'
    assert 'views[eye][resource]=views[0][resource];continue;' in setup
    assert 'census_storage==r.slots[index]->census_storage' in shared
    assert 'if(stage==1)task.reads(views[0][LeftDesc])' in single, 'shared Census phases remain causally ordered'
    assert 'if(stage==2)task.reads(views[0][LeftDesc]).reads(views[0][RightDesc]).reads(views[0][LeftValid]).reads(views[0][RightValid])' in single, 'left clear follows complete shared preparation'
    assert 'if(eye&&stage==4+a.paths)task.reads(views[0][Disparity])' in single, 'right clear explicitly depends on left selection'
    for kind in ('reads', 'writes', 'reads_writes'):
        assert f'task.{kind}(views[eye][resource])' in single, 'graph declarations must cover shader accesses'
    assert single.count('ti.recorder.dispatch(') == 1
    assert 'dispatch({grid[0],grid[1],1})' in single, 'each shared phase dispatches one reference'
    assert 'query_index=2+stage*2' in single and 'query_index=3+stage*2' in single
    assert shared.count('single(0,0,0,0,') == 1 and shared.count('single(1,0,0,1,') == 1, 'exactly two Census dispatches'
    assert 'for(unsigned eye=0;eye<2;++eye){unsigned base=2+eye*(a.paths+2)' in shared
    assert 'single(base,eye,7,0,' in shared
    assert 'single(base+1+path,eye,eye?(l.packing==8?21u:22u):(l.packing==8?11u:12u)' in shared
    assert 'single(base+1+a.paths,eye,eye?(l.packing==8?23u:24u):(l.packing==8?13u:14u)' in shared
    assert '{LeftDesc,RightDesc,LeftValid,RightValid},{},{Penalties}' in shared
    assert '{LeftDesc,RightDesc,LeftValid,RightValid,Penalties},{Disparity,Confidence,Validity},{}' in shared
    assert not re.search(r'allocate\(|create_buffer\(|create_memory\(', setup), 'combined graph reuses physical resource allocations'
    assert not re.search(r'wait_\w+\(|readback\w*\(', single)
    captures = re.findall(r'task\.executes\(\[([^]]*)\]', shared)
    assert captures and all(not re.search(r'\b(?:input|prototype|pre|left|right)\b', capture) for capture in captures), 'callbacks cannot retain producer Frames'


def check_shared_storage(stereo, depth):
    storage = body(stereo, 'struct CensusStorage')
    slot = body(stereo, 'struct DaxaStereo::Slot')
    setup = body(stereo, 'void DaxaStereo::Impl::setup(')
    assert 'std::shared_ptr<CensusStorage> census_storage' in slot
    assert 'std::shared_ptr<DaxaFoundation> context' in storage
    assert not re.search(r'\b(?:Frame|Impl|Slot|Pair)\b', storage), 'shared storage must have no owning backreference cycle'
    assert 'destroy_buffer(b)' in storage
    assert 'if(descriptor_owner)s->census_storage=descriptor_owner->slots[index]->census_storage' in setup
    assert 's->buffers[b]=s->census_storage->buffers[b-LeftDesc]' in setup
    assert 'if(shared_census&&b<=RightValid)continue' in setup, 'never allocate a second physical quartet'
    assert 's->census_storage->buffers[b-LeftDesc]=d.create_buffer({.size=n*4' in setup
    assert 'allocated+=n*4' in setup
    accounting = body(depth, 'uint64_t DaxaDepthConfig::requested_buffer_bytes(')
    assert 'if(stereo_schedule==DaxaDepthStereoSchedule::SharedCensusSequential)total-=16*n*slots' in accounting, 'actual bytes drop by one quartet per aligned slot'
    check_depth_setup_constructors(depth)
    constructor = body(depth, 'Impl(DaxaRectification maps,DaxaDepthConfig c,')
    assert 'left->allocated_bytes()+right->allocated_bytes()+allocated==combined_allocated' in constructor


def test_shared_census_actual_hotpath_and_graph_are_bounded_and_gpu_resident():
    stereo = (ROOT/'src/daxa_stereo.cpp').read_text()
    depth = (ROOT/'src/daxa_depth.cpp').read_text()
    check_shared_hotpath(stereo, depth)
    check_shared_graph(stereo, (ROOT/'include/stereoforge/daxa_stereo.hpp').read_text())
    check_shared_storage(stereo, depth)


@pytest.mark.parametrize('mutation', [
    'std::make_shared<int>();', 'd.wait_idle();', 'readback(frame);',
    'd.create_buffer({});', 'metric_calibration();', 'sha256(bytes);',
    'for(unsigned pixel=0;pixel<n;++pixel){}',
])
def test_shared_submission_guard_rejects_hotpath_regressions(mutation):
    stereo = (ROOT/'src/daxa_stereo.cpp').read_text()
    marker = 'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(const DaxaFrame& input,uint64_t oldest,unsigned index){'
    assert marker in stereo
    with pytest.raises(AssertionError):
        check_shared_hotpath(stereo.replace(marker, marker+mutation, 1), (ROOT/'src/daxa_depth.cpp').read_text())


@pytest.mark.parametrize('before,after', [
    ('if(eye&&stage==4+a.paths)task.reads(views[0][Disparity]);', ''),
    ('if(stage==1)task.reads(views[0][LeftDesc]);', ''),
    ('if(stage==2)task.reads(views[0][LeftDesc]).reads(views[0][RightDesc]).reads(views[0][LeftValid]).reads(views[0][RightValid]);', ''),
    ('shared_census&&resource>=LeftDesc&&resource<=RightValid', 'false'),
    ('pair.stages=shared_census?6+2*a.paths:l.stages', 'pair.stages=l.stages'),
    ('task.reads_writes(views[eye][resource])', 'task.reads(views[eye][resource])'),
    ('dispatch({grid[0],grid[1],1})', 'dispatch({grid[0],grid[1],2})'),
])
def test_shared_graph_guard_rejects_lost_hazards_aliases_or_phase_boundaries(before, after):
    stereo = (ROOT/'src/daxa_stereo.cpp').read_text()
    assert before in stereo
    with pytest.raises(AssertionError):
        check_shared_graph(stereo.replace(before, after), (ROOT/'include/stereoforge/daxa_stereo.hpp').read_text())


@pytest.mark.parametrize('before,after', [
    ('if(shared_census&&b<=RightValid)continue', 'if(false)continue'),
    ('std::shared_ptr<CensusStorage> census_storage', 'CensusStorage* census_storage'),
    ('std::array<daxa::BufferId,4> buffers{};', 'std::array<daxa::BufferId,4> buffers{}; std::shared_ptr<Impl> owner;'),
])
def test_shared_storage_guard_rejects_duplicate_allocation_or_unsafe_ownership(before, after):
    stereo = (ROOT/'src/daxa_stereo.cpp').read_text()
    assert before in stereo
    with pytest.raises(AssertionError):
        check_shared_storage(stereo.replace(before, after), (ROOT/'src/daxa_depth.cpp').read_text())


@pytest.mark.parametrize('paths', [4, 8])
def test_actual_shared_depth_admission_upload_and_retirement_have_zero_warm_allocations(tmp_path_factory, paths):
    binary = build_pair_bookkeeping_probe(tmp_path_factory, 'shared-census-sequential', paths)
    result = subprocess.run([str(binary), 'shared'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    print('Actual shared-Census depth/pre bookkeeping CPU-only probe:', json.dumps(row, sort_keys=True))
    assert row['frames'] == 256 and row['packed_bytes_per_frame'] == 3686400
    assert row['application_allocations'] == 0 and row['admission_snapshots'] == 256
    assert row['full_output_and_rgb_backpressure_before_upload'] and row['inflight_and_retained_producers_preserved']
    assert 0 < row['mean_cpu_ms'] < 1000


def test_shared_descriptor_constructor_is_private_to_aligned_depth_coordinator():
    header = (ROOT/'include/stereoforge/daxa_stereo.hpp').read_text()
    private = header[header.index('private:'):]
    public = header[header.index('public:'):header.index('private:')]
    assert 'bool shared_census' in private and 'const DaxaStereo* descriptor_owner' in private
    assert 'bool shared_census' not in public and 'descriptor_owner' not in public, 'independent public submissions cannot bypass aligned shared-descriptor admission'


@pytest.mark.parametrize('regression', ['allocation', 'left_admission', 'right_admission'])
def test_actual_probe_rejects_application_allocation_and_either_producer_admission_regression(tmp_path_factory, regression):
    def mutate(stereo):
        if regression == 'allocation':
            marker = 'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(const DaxaFrame& input,uint64_t oldest,unsigned index){'
            assert marker in stereo
            return stereo.replace(marker, marker+'auto allocation_guard=std::make_shared<int>(42);', 1)
        eye = 'l' if regression == 'left_admission' else 'r'
        condition = f'detail::available_slot({eye}.slots[index].use_count(),{eye}.slots[index]->submit,oldest)'
        assert condition in stereo
        return stereo.replace(condition, 'true', 1)
    binary = build_pair_bookkeeping_probe(tmp_path_factory, 'shared-census-sequential', stereo_transform=mutate)
    result = subprocess.run([str(binary), 'shared'], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, 'actual extracted probe must reject the mutation'
    expected = 'application heap allocation' if regression == 'allocation' else 'aligned '+('left' if regression == 'left_admission' else 'pair')+' ring rejection occurred after upload'
    assert expected in result.stderr, result.stderr


def test_shared_memory_guard_rejects_unaccounted_duplicate_quartet():
    stereo = (ROOT/'src/daxa_stereo.cpp').read_text()
    depth = (ROOT/'src/daxa_depth.cpp').read_text()
    marker = 'total-=16*n*slots'
    assert marker in depth
    with pytest.raises(AssertionError, match='actual bytes'):
        check_shared_storage(stereo, depth.replace(marker, 'total-=0', 1))


def check_depth_setup_constructors(depth):
    for signature in ('Impl(DaxaRectification maps,DaxaDepthConfig c,',
                      'Impl(DaxaDepthConfig c,std::shared_ptr<DaxaFoundation> foundation,const DaxaFrame& prototype)'):
        constructor = body(depth, signature)
        assert constructor.count('setup_device();') == 1
        assert constructor.count('setup(pre);') == 1
        assert constructor.index('setup_device();') < constructor.index('setup(pre);') < constructor.index('finish_setup();')


def check_shared_bootstrap_order(depth):
    check_depth_setup_constructors(depth)
    constructor = body(depth, 'void setup(DaxaFrame& pre)')
    left = constructor.index('left->process(pre)')
    shared_wait = constructor.index('if(c.stereo_schedule==DaxaDepthStereoSchedule::SharedCensusSequential)l.wait()')
    right = constructor.index('right->process(pre)')
    legacy_wait = constructor.index('if(c.stereo_schedule!=DaxaDepthStereoSchedule::SharedCensusSequential)l.wait()')
    right_wait = constructor.index('r.wait()')
    assert left < shared_wait < right < legacy_wait < right_wait, 'shared bootstrap must finish left graph before right overwrites shared descriptors, preserving legacy order'


def test_shared_bootstrap_separate_graphs_cannot_overwrite_inflight_shared_census():
    check_shared_bootstrap_order((ROOT/'src/daxa_depth.cpp').read_text())


@pytest.mark.parametrize('mutation', ['remove_wait', 'invert_mode', 'late_wait'])
def test_shared_bootstrap_guard_rejects_cross_graph_write_after_write_regression(mutation):
    depth = (ROOT/'src/daxa_depth.cpp').read_text()
    marker = 'if(c.stereo_schedule==DaxaDepthStereoSchedule::SharedCensusSequential)l.wait();'
    assert marker in depth
    if mutation == 'remove_wait':
        depth = depth.replace(marker, '', 1)
    elif mutation == 'invert_mode':
        depth = depth.replace(marker, marker.replace('==', '!='), 1)
    else:
        depth = depth.replace(marker, '', 1).replace('r.wait();', marker+'r.wait();', 1)
    with pytest.raises((AssertionError, ValueError)):
        check_shared_bootstrap_order(depth)


@pytest.mark.parametrize('signature', [
    'Impl(DaxaRectification maps,DaxaDepthConfig c,',
    'Impl(DaxaDepthConfig c,std::shared_ptr<DaxaFoundation> foundation,const DaxaFrame& prototype)',
])
@pytest.mark.parametrize('mutation', ['remove_device', 'remove_setup', 'late_device'])
def test_shared_bootstrap_guard_checks_both_constructor_setup_paths(signature, mutation):
    depth = (ROOT/'src/daxa_depth.cpp').read_text()
    constructor = body(depth, signature)
    if mutation == 'remove_device':
        changed = constructor.replace('setup_device();', '', 1)
    elif mutation == 'remove_setup':
        changed = constructor.replace('setup(pre);', '', 1)
    else:
        changed = constructor.replace('setup_device();', '', 1).replace('setup(pre);', 'setup(pre);setup_device();', 1)
    assert changed != constructor
    with pytest.raises((AssertionError, ValueError)):
        check_shared_bootstrap_order(depth.replace(constructor, changed, 1))
