"""CPU-only independent guards for the opt-in Left-aggregated Right selector.

Extracted production admission/retirement/staging code uses fake GPU calls. Its
CPU wall time includes the packed staging copy and excludes Daxa/driver/GPU work.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

from test_kernel_variant_hotpath import body
from test_daxa_depth_pair_hotpath import build_pair_bookkeeping_probe
from test_daxa_depth_shared_census_hotpath import check_shared_hotpath

ROOT = Path(__file__).resolve().parents[1]


def check_derived_hotpath(stereo, depth):
    check_shared_hotpath(stereo, depth)
    for source, signature in ((stereo, 'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at('),
                              (depth, 'DaxaDepthPipeline::Frame DaxaDepthPipeline::enqueue(')):
        hot = body(source, signature)
        assert not re.search(r'\b(?:sha256|hash|metric_calibration|strict_json|memcpy|malloc|calloc|realloc|resize|reserve|push_back|wait_\w+|readback\w*|create_\w+)\s*\(|\.geometry', hot), 'derived hotpath cannot add waits, readback, image copies, allocations, hashes or matrix work'


def test_derived_submission_stays_bounded_and_gpu_resident():
    check_derived_hotpath((ROOT/'src/daxa_stereo.cpp').read_text(), (ROOT/'src/daxa_depth.cpp').read_text())


@pytest.mark.parametrize('mutation', ['d.wait_idle();', 'readback(frame);', 'sha256(bytes);',
                                    'std::make_shared<int>();', 'memcpy(dst,src,n);',
                                    'for(unsigned pixel=0;pixel<n;++pixel){}'])
def test_derived_submission_guard_rejects_regressions(mutation):
    stereo = (ROOT/'src/daxa_stereo.cpp').read_text()
    marker = 'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(const DaxaFrame& input,uint64_t oldest,unsigned index){'
    assert marker in stereo
    with pytest.raises(AssertionError):
        check_derived_hotpath(stereo.replace(marker, marker+mutation, 1), (ROOT/'src/daxa_depth.cpp').read_text())


@pytest.mark.parametrize('paths', [4, 8])
def test_actual_derived_admission_staging_and_retirement_are_allocation_free(tmp_path_factory, paths):
    binary = build_pair_bookkeeping_probe(tmp_path_factory, 'left-aggregated-selection', paths)
    result = subprocess.run([str(binary), 'derived'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    row['scope'] = 'actual_derived_admission_depth_pre_bookkeeping_fake_GPU_with_packed_staging_copy'
    row['paths'] = paths
    row['scheduled_phases'] = paths+5
    row['source_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                            for name in ('src/daxa_stereo.cpp', 'src/daxa_depth.cpp', 'src/daxa_preprocess.cpp')}
    print('Actual derived CPU-only host probe:', json.dumps(row, sort_keys=True))
    assert row['frames'] == 256 and row['packed_bytes_per_frame'] == 3686400
    assert row['application_allocations'] == 0 and row['admission_snapshots'] == 256
    assert row['full_output_and_rgb_backpressure_before_upload'] and row['inflight_and_retained_producers_preserved']
    assert 0 < row['mean_cpu_ms'] < 1000


@pytest.mark.parametrize('regression', ['allocation', 'right_admission'])
def test_actual_derived_probe_detects_allocation_and_right_only_backpressure(tmp_path_factory, regression):
    def mutate(stereo):
        if regression == 'allocation':
            marker = 'std::array<DaxaStereo::Frame,2> DaxaStereo::process_pair_at(const DaxaFrame& input,uint64_t oldest,unsigned index){'
            assert marker in stereo
            return stereo.replace(marker, marker+'auto allocation_guard=std::make_shared<int>(42);', 1)
        condition = 'detail::available_slot(r.slots[index].use_count(),r.slots[index]->submit,oldest)'
        assert condition in stereo
        return stereo.replace(condition, 'true', 1)
    binary = build_pair_bookkeeping_probe(tmp_path_factory, 'left-aggregated-selection', stereo_transform=mutate)
    result = subprocess.run([str(binary), 'derived'], capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, 'actual extracted probe must detect mutation'
    expected = 'application heap allocation' if regression == 'allocation' else 'aligned pair ring rejection occurred after upload'
    assert expected in result.stderr, result.stderr


def check_derived_shader(source):
    support = body(source, 'bool derived_supported(')
    assert support.index('i>=parameters.width*parameters.height') < support.index('parameters.right_valid_address')
    assert support.index('d>=parameters.width-i%parameters.width') < support.index('parameters.right_valid_address')
    assert 'parameters.disparities' in support and 'parameters.left_valid_address)[i+d]' in support
    assert not re.search(r'desc_address|penalties_address', support), 'support cannot touch descriptors or packed P'
    score = body(source, 'uint derived_score<')
    assert 'uint left=i+d' in score, 'each disparity, including subpixel neighbors, needs its own Left shift'
    assert 'parameters.left_desc_address)[left]' in score and 'parameters.right_desc_address)[i]' in score
    assert '[left*row_words+d/Components]' in score, 'P gather must use shifted pixel and current disparity'
    assert 'return parameters.paths*cost+field' in score
    select = body(source, 'void derived_select<')
    supported_loop = body(select, 'if(derived_supported(parameters,i,d))')
    assert '++count;' in supported_loop and 'derived_score<Components,Bits>(parameters,i,d)' in supported_loop
    assert select.count('derived_score<') == 3, 'no unchecked descriptor/P load call may escape support guards'
    assert 'derived_count[t]>=2&&sgm_selection_valid(best,best_cost,second)' in select
    assert 'c==best_cost&&d<best' in select and 'oc==own&&oi<index' in select
    assert 'derived_count[t]+=derived_count[other]' in select
    assert 'derived_supported(parameters,i,best-1)&&derived_supported(parameters,i,best+1)' in select
    assert 'derived_score<Components,Bits>(parameters,i,best-1),derived_score<Components,Bits>(parameters,i,best+1)' in select
    assert 'float(second-best_cost)/float(max(second,1u))' in select and 'valid?value:asfloat(0x7fc00000)' in select
    assert 'select_right_derived8' in source and 'select_right_derived16' in source


def test_derived_selector_checks_support_before_gather_and_shifts_every_neighbor():
    check_derived_shader((ROOT/'python/stereoforge/shaders/daxa/stereo_derived.slang').read_text())


@pytest.mark.parametrize('before,after', [
    ('if(d>=parameters.width-i%parameters.width)return false;', ''),
    ('parameters.left_valid_address)[i+d]', 'parameters.left_valid_address)[i]'),
    ('if(derived_supported(parameters,i,d))', 'if(true)'),
    ('uint left=i+d', 'uint left=i'),
    ('[left*row_words+d/Components]', '[i*row_words+d/Components]'),
    ('derived_count[t]>=2&&', ''),
    ('derived_supported(parameters,i,best-1)&&derived_supported(parameters,i,best+1)', 'true'),
    ('derived_score<Components,Bits>(parameters,i,best-1)', 'derived_score<Components,Bits>(parameters,i,best)'),
    ('c==best_cost&&d<best', 'c==best_cost&&d>best'),
])
def test_derived_shader_guard_detects_unsafe_gathers_and_selection_regressions(before, after):
    source = (ROOT/'python/stereoforge/shaders/daxa/stereo_derived.slang').read_text()
    assert before in source
    with pytest.raises((AssertionError, ValueError)):
        check_derived_shader(source.replace(before, after, 1))


def check_derived_storage_and_graph(stereo, depth):
    storage = body(stereo, 'struct CensusStorage')
    assert 'daxa::BufferId penalties{}' in storage and 'destroy_buffer(penalties)' in storage
    assert not re.search(r'\b(?:Frame|Impl|Slot|Pair)\b', storage), 'RAII storage cannot retain engines or frames'
    setup = body(stereo, 'void DaxaStereo::Impl::setup(')
    derived_storage = body(setup, 'if(derived)')
    assert 'if(!selector_only)' in derived_storage and 's->census_storage->penalties=d.create_buffer(' in derived_storage
    assert 's->buffers[Penalties]=s->census_storage->penalties' in derived_storage
    selector = body(setup, 'if(selector_only){add(')
    assert selector.count('add(') == 1 and '25:26' in selector
    assert '{LeftDesc,RightDesc,LeftValid,RightValid,Penalties},{Disparity,Confidence,Validity},{}' in selector
    assert not re.search(r'census|clear|for\(', selector), 'bootstrap Right must only gather/select, never overwrite shared Census/P'
    pair = body(stereo, 'void DaxaStereo::setup_pair(')
    assert 'pair.stages=a.paths+5' in pair and 'pair.query_count=2+2*pair.stages' in pair
    assert '(derived&&resource==Penalties)' in pair, 'register physical Left P alias once'
    shared = body(pair, 'if(shared_census)')
    derived = body(shared, 'if(derived)')
    assert derived.count('single(') == 4 and 'for(unsigned path=0;path<a.paths;++path)' in derived
    assert derived.index('single(2,0,7') < derived.index('single(3+path,0,') < derived.index('single(3+a.paths,0,') < derived.index('single(4+a.paths,1,')
    assert 'l.slots[index]->buffers[Penalties]==r.slots[index]->buffers[Penalties]' in derived
    assert 'if(eye&&stage==4+a.paths)task.reads(views[0][Disparity])' in shared, 'Right selector must wait for all Left work'
    assert '{LeftDesc,RightDesc,LeftValid,RightValid,Penalties},{Disparity,Confidence,Validity},{}' in derived
    accounting = body(depth, 'uint64_t DaxaDepthConfig::requested_buffer_bytes(')
    assert 'if(right_reference_mode==DaxaRightReferenceMode::LeftAggregatedSelection)total-=slots*packed' in accounting
    assert 'total<2ull*1024*1024*1024' in accounting
    constructor = body(depth, 'Impl(DaxaRectification maps,DaxaDepthConfig c,')
    assert constructor.index('left->process(pre)') < constructor.index('SharedCensusSequential)l.wait()') < constructor.index('right->process(pre)')
    assert 'left->allocated_bytes()+right->allocated_bytes()+allocated==combined_allocated' in constructor


def test_derived_graph_and_bootstrap_reuse_one_left_recurrence_with_9_or_13_phases():
    check_derived_storage_and_graph((ROOT/'src/daxa_stereo.cpp').read_text(), (ROOT/'src/daxa_depth.cpp').read_text())


@pytest.mark.parametrize('before,after', [
    ('if(!selector_only){s->census_storage->penalties=', 'if(true){s->census_storage->penalties='),
    ('(derived&&resource==Penalties)', 'false'),
    ('pair.stages=a.paths+5', 'pair.stages=a.paths+6'),
    ('if(eye&&stage==4+a.paths)task.reads(views[0][Disparity]);', ''),
    ('std::array<daxa::BufferId,4> buffers{};', 'std::array<daxa::BufferId,4> buffers{};std::shared_ptr<Impl> owner;'),
])
def test_derived_resource_guard_rejects_duplicate_storage_and_lost_hazards(before, after):
    source = (ROOT/'src/daxa_stereo.cpp').read_text()
    assert before in source
    with pytest.raises((AssertionError, ValueError)):
        check_derived_storage_and_graph(source.replace(before, after, 1), (ROOT/'src/daxa_depth.cpp').read_text())


def check_derived_metadata(metric, depth, header, stereo):
    assert metric.strip() == '#define SF_METRIC_PROVENANCE 5\n#define metric_observation metric_observation_derived\n#include "depth.slang"', 'derived wrapper sets provenance 5 before including the one metric implementation'
    canonical = (ROOT/'python/stereoforge/shaders/daxa/depth.slang').read_text()
    assert '#ifndef SF_METRIC_PROVENANCE\n#define SF_METRIC_PROVENANCE 1\n#endif' in canonical
    observation = body(canonical, 'void metric_observation(')
    assert 'if(t>=2*n)return' in observation and 'eye=t/n' in observation, 'provenance applies to both output layers'
    assert '((uint*)p.outputs[6])[t]=valid?SF_METRIC_PROVENANCE:0' in observation
    assert 'LeftAggregatedStereo=5' in header
    reference = body(depth, 'DaxaDepthReadback daxa_depth_reference(')
    assert 'for(unsigned eye=0;eye<2;++eye)' in reference
    assert 'out.provenance[t]=c.right_reference_mode==DaxaRightReferenceMode::LeftAggregatedSelection?uint32_t(DaxaDepthProvenance::LeftAggregatedStereo)' in reference
    assert 'out.lr_dependency=' in reference and 'DaxaLRDependency::LeftAggregatedCorrelated' in reference
    assert 'out.right_reference_mode=frame.right_reference_mode();out.lr_dependency=frame.lr_dependency()' in body(depth, 'DaxaDepthReadback DaxaDepthPipeline::readback(')
    old = body(depth, 'void DaxaDepthPipeline::check_stereo_integer(')
    assert old.index('DaxaRightReferenceMode::IndependentSGM') < old.index('engine.readback(')
    getter = body(stereo, 'daxa::BufferId DaxaStereo::Frame::buffer(')
    assert getter.index('plane==DaxaStereoPlane::PackedPenalties&&estimator()==DaxaStereoEstimator::LeftAggregatedSelection') < getter.index('slot_->buffers[')


def test_derived_metric_provenance_and_typed_metadata_are_honest_on_both_layers():
    check_derived_metadata((ROOT/'python/stereoforge/shaders/daxa/depth_derived.slang').read_text(), (ROOT/'src/daxa_depth.cpp').read_text(), (ROOT/'include/stereoforge/daxa_stereo.hpp').read_text(), (ROOT/'src/daxa_stereo.cpp').read_text())


@pytest.mark.parametrize('before,after', [('SF_METRIC_PROVENANCE 5', 'SF_METRIC_PROVENANCE 1'), ('#define metric_observation metric_observation_derived', '')])
def test_derived_provenance_guard_rejects_one_layer_or_independent_claim(before, after):
    source = (ROOT/'python/stereoforge/shaders/daxa/depth_derived.slang').read_text()
    assert before in source
    with pytest.raises((AssertionError, ValueError)):
        check_derived_metadata(source.replace(before, after, 1), (ROOT/'src/daxa_depth.cpp').read_text(), (ROOT/'include/stereoforge/daxa_stereo.hpp').read_text(), (ROOT/'src/daxa_stereo.cpp').read_text())


def test_derived_shader_sources_are_bound_to_cmake_and_cold_identity_getters():
    for family, cls, sources, macro in (
        ('Stereo', 'DaxaStereo', ('stereo_derived.slang', 'stereo_parameters.slang'), 'SF_DAXA_STEREO_DERIVED_SHA256'),
        ('Depth', 'DaxaDepthPipeline', ('depth_derived.slang', 'depth.slang'), 'SF_DAXA_DEPTH_DERIVED_SHA256')):
        cmake = (ROOT/f'cmake/Daxa{family}.cmake').read_text()
        source = (ROOT/f'src/daxa_{family.lower()}.cpp').read_text()
        assert re.search(r'file\(SHA256[^\n]*'+sources[0]+r'[^\n]*'+macro, cmake)
        assert f'{macro}="${{{macro}}}"' in cmake
        assert sources[0] in cmake and 'CMAKE_CONFIGURE_DEPENDS' in cmake
        getter = body(source, f'std::string {cls}::derived_shader_identity(')
        assert macro in getter and '_OPS_SHA256' in getter
        assert not re.search(r'create_|Foundation|device|wait_|readback', getter), 'identities are cold CPU getters'
    app = (ROOT/'apps/daxa_depth_benchmark.cpp').read_text()
    assert app.index('--derived-shader-source-identities') < app.index('std::make_shared<DaxaFoundation>')
    assert 'stereo_derived_shader_source_sha256_triple' in app and 'depth_derived_shader_source_sha256_triple' in app


def test_built_derived_identity_getter_matches_actual_sources_without_vulkan():
    import os
    binary = ROOT/'build/daxa/daxa_depth_benchmark'
    if not binary.is_file():
        pytest.skip('optional Daxa benchmark is not built')
    result = subprocess.run([str(binary), '--derived-shader-source-identities'], capture_output=True, text=True,
                            env=dict(os.environ, VK_DRIVER_FILES='/nonexistent-stereoforge-derived-CPU-only.json'), timeout=10)
    assert result.returncode == 0 and not result.stderr, result.stderr
    def sha(path):
        return hashlib.sha256((ROOT/'python/stereoforge/shaders'/path).read_bytes()).hexdigest()
    assert json.loads(result.stdout) == {
        'stereo_derived_shader_source_sha256_triple': ':'.join(sha(p) for p in ('daxa/stereo_derived.slang', 'daxa/stereo_parameters.slang', 'ops/sgm.slang')),
        'depth_derived_shader_source_sha256_triple': ':'.join(sha(p) for p in ('daxa/depth_derived.slang', 'daxa/depth.slang', 'ops/stereo_observation.slang'))}


def test_derived_mode_stays_opt_in_private_and_reports_only_checked_quantities():
    depth = (ROOT/'src/daxa_depth.cpp').read_text()
    header = (ROOT/'include/stereoforge/daxa_depth.hpp').read_text()
    stereo_header = (ROOT/'include/stereoforge/daxa_stereo.hpp').read_text()
    assert 'right_reference_mode=DaxaRightReferenceMode::IndependentSGM' in header
    validation = body(depth, 'void DaxaDepthConfig::validate(')
    assert 'right_reference_mode==DaxaRightReferenceMode::IndependentSGM||stereo_schedule==DaxaDepthStereoSchedule::SharedCensusSequential' in validation
    private = stereo_header[stereo_header.index('private:'):]
    public = stereo_header[stereo_header.index('public:'):stereo_header.index('private:')]
    assert 'bool shared_census' in private and 'DaxaRightReferenceMode mode=' in private
    assert 'descriptor_owner' not in public and 'bool shared_census' not in public
    getter = body(depth, 'DaxaStereo::Frame DaxaDepthPipeline::Frame::stereo_frame(')
    assert 'owner_->config.stereo_schedule==DaxaDepthStereoSchedule::SharedCensusSequential' in getter
    app = (ROOT/'apps/daxa_depth_benchmark.cpp').read_text()
    assert '"both_stereo_integer_reference_checked",checked&&!derived' in app
    assert '"both_stereo_packed_penalties_reference_checked",checked&&!derived' in app
    assert '"metric_compound_float_reference_checked",false' in app and '"stereo_compound_float_reference_checked",false' in app
    report = body(app, 'if(derived){\n        result["schema"]')
    for key in ('left_stereo_integer_reference_checked', 'left_stereo_packed_penalties_reference_checked', 'derived_right_mask_reference_checked'):
        assert f'result["{key}"]=checked' in report
    assert 'result["right_stereo_integer_winner_reference_checked"]=false' in report
    assert 'result["lr_references_independent"]=false' in report
    assert 'result["right_recurrence_allocated"]=false' in report
    assert 'both_left_and_right_layers_share_correlated_LR_dependency' in report
    assert not re.search(r'utilization|speedup|60.?[Hh][Zz]|faster', report), 'report cannot infer utilization causes or performance promises'


@pytest.mark.parametrize('before,after', [
    ('total-=slots*packed', 'total-=0'),
    ('total<2ull*1024*1024*1024', 'true'),
    ('if(c.stereo_schedule==DaxaDepthStereoSchedule::SharedCensusSequential)l.wait();', ''),
])
def test_derived_accounting_and_bootstrap_guards_detect_regressions(before, after):
    depth = (ROOT/'src/daxa_depth.cpp').read_text()
    assert before in depth
    with pytest.raises((AssertionError, ValueError)):
        check_derived_storage_and_graph((ROOT/'src/daxa_stereo.cpp').read_text(), depth.replace(before, after, 1))
