"""Actual post-loop C++ metadata contracts, with no GPU or SDK execution."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
APPS = ('daxa_depth_benchmark.cpp', 'daxa_refinement_benchmark.cpp',
        'daxa_depth_rectified_benchmark.cpp')


def test_actual_cpu_metadata_helper(tmp_path, json_include):
    header = ROOT / 'apps/timing_metadata.hpp'
    assert header.exists(), 'post-loop timing metadata helper missing'
    code = tmp_path / 'probe.cpp'
    code.write_text(r'''
#include "timing_metadata.hpp"
#include <iostream>
int main(){
 using json=nlohmann::json;
 json raw={{"host_wall_ms",{3.,1.,2.}}, {"gpu_left_stereo_ms",nullptr},
           {"gpu_target_rectify_ms",{0.,0.,0.}}};
 json r={{"raw_samples_ms",raw}};
 stereoforge::add_native_timing_metadata(r,3,2,"nearest_rank_ceil_p_times_N_minus1",nullptr,{"gpu_target_rectify_ms"});
 json cold={{"raw_samples_ms",raw}};
 stereoforge::add_native_timing_metadata(cold,3,0,"upper_median_floor95",1.25);
 bool bad_count=false,bad_warmup=false,bad_shape=false,bad_value=false;
 try {stereoforge::add_native_timing_metadata(cold,0,0,"x");}catch(std::invalid_argument const&){bad_count=true;}
 try {stereoforge::add_native_timing_metadata(cold,3,31,"x");}catch(std::invalid_argument const&){bad_warmup=true;}
 try {stereoforge::add_native_timing_metadata(cold,2,0,"x");}catch(std::invalid_argument const&){bad_shape=true;}
 cold["raw_samples_ms"]["host_wall_ms"]={-1.,2.,3.};
 try {stereoforge::add_native_timing_metadata(cold,3,0,"x");}catch(std::invalid_argument const&){bad_value=true;}
 std::cout<<json{{"warm",r},{"cold_metadata",cold["timing_metadata"]},
 {"bad",{bad_count,bad_warmup,bad_shape,bad_value}},{"unchanged",r["raw_samples_ms"]==raw}};
}
''')
    compiler = shutil.which('c++')
    assert compiler
    binary = tmp_path / 'probe'
    subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                    '-I', str(json_include), '-I', str(header.parent),
                    str(code), '-o', str(binary)], check=True, capture_output=True, text=True)
    result = json.loads(subprocess.check_output([str(binary)], text=True))
    assert result['bad'] == [True] * 4
    assert result['unchanged']
    m = result['warm']['timing_metadata']
    assert set(m) == {'schema','units','sample_population','sample_order','measured_count',
        'excluded_warmup_count','phase','source_reuse','completion_boundary','readback_in_span',
        'file_io_in_span','diagnostic_retrieval_in_span','legacy_summary_policy','columns',
        'cold_prepare','cold_prepare_scope','cache_temperature','allocation_activity','host_output_properties'}
    assert m['schema'] == 'stereoforge.native_timing_metadata.v1'
    assert (m['units'], m['sample_population'], m['sample_order']) == ('ms','measured_iterations_only','chronological')
    assert (m['measured_count'],m['excluded_warmup_count'],m['phase']) == (3,2,'warm')
    assert m['source_reuse'] == 'static_retained_record'
    assert m['completion_boundary'] == 'final_submit_wait_and_query'
    assert not any(m[k] for k in ('readback_in_span','file_io_in_span','diagnostic_retrieval_in_span'))
    assert m['columns']['host_wall_ms']['status'] == 'available'
    for key in ('gpu_left_stereo_ms','gpu_target_rectify_ms'):
        assert m['columns'][key]['status'] == 'unavailable'
        assert m['columns'][key]['reason']
    for key in ('cold_prepare','cache_temperature','allocation_activity','host_output_properties'):
        assert m[key]['status'] == 'unavailable' and m[key]['reason']
    c = result['cold_metadata']
    assert c['phase'] == 'unwarmed'
    assert c['cold_prepare'] == {'status':'available','value':1.25}
    assert c['cold_prepare_scope'] == 'foundation_and_pipeline_construction_only'


@pytest.mark.parametrize('name', APPS)
def test_metadata_only_added_after_timing_loop(name):
    source = (ROOT / 'apps' / name).read_text()
    loop = source.index('i<frames+warmup;')
    report = source.index('auto result=prepared.report()' if name == APPS[1] else
                          'json result=', loop)
    assert 'add_native_timing_metadata(' not in source[loop:report]
    assert source.index('add_native_timing_metadata(', report) > report
    assert '#include "timing_metadata.hpp"' in source


def test_depth_paired_phase_raw_export_uses_same_mapping():
    source = (ROOT / 'apps' / APPS[0]).read_text()
    assert 'gpu_paired_stereo_raw_samples_ms' in source
    assert 'gpu_paired_stereo_phases_raw_samples_ms' in source
    # Every summary phase assignment delegates to one shared emitter that also
    # emits the corresponding chronological vector, preventing mapping drift.
    assert 'auto emit_phase=' in source
    assert 'result["gpu_paired_stereo_phases_ms"][key]=summary(values)' in source
    assert 'result["gpu_paired_stereo_phases_raw_samples_ms"][key]=values' in source
    assert 'emit_phase("derived_right_select",pair_phases[4+config.stereo.paths])' in source


def test_actual_depth_paired_reporter_preserves_phase_mapping(tmp_path, json_include):
    source = (ROOT / 'apps' / APPS[0]).read_text()
    start = source.index('    result["stereo_schedule"]')
    fragment = source[start:source.index('    constexpr const char* keys[]=', start)]
    summary = source[source.index('json summary('):source.index('// Post-timing only:')]
    code = tmp_path / 'paired.cpp'
    code.write_text('''#include "timing_metadata.hpp"
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
using json=nlohmann::json;
struct DaxaStereo { static unsigned packed_bits(auto const&){return 8;}
 static char const* derived_shader_identity(){return "unused";} };
struct DaxaDepthPipeline { static char const* derived_shader_identity(){return "unused";} };
''' + summary + '''
json run(std::string schedule,bool derived,unsigned paths){
 struct {struct {unsigned paths,width=16,height=8,disparities=16;} stereo;unsigned slots=2;} config;
 config.stereo.paths=paths;
 bool checked=false;std::string right_mode="derived";json result;
 std::vector<double> pair_samples{9,1,5};
 std::array<std::vector<double>,30> pair_phases;
 for(unsigned i=0;i<pair_phases.size();++i)pair_phases[i]={double(i+9),double(i+1),double(i+5)};
''' + fragment + '''
 return result;
}
int main(){json out=json::array();
 for(auto const* schedule:{"sequential","paired-per-path","layered-per-path","shared-census-sequential"})
  for(unsigned paths:{4,8})out.push_back(run(schedule,false,paths));
 out.push_back(run("shared-census-sequential",true,4));
 std::cout<<out;
}
''')
    binary = tmp_path / 'paired'
    subprocess.run([shutil.which('c++'), '-std=c++20', '-Wall', '-Wextra', '-Werror',
                    '-I', str(json_include), '-I', str(ROOT / 'apps'),
                    str(code), '-o', str(binary)], check=True, capture_output=True, text=True)
    reports = json.loads(subprocess.check_output([str(binary)], text=True))
    paths_names = ['aggregate_pos_x','aggregate_neg_x','aggregate_pos_y','aggregate_neg_y',
                   'aggregate_pos_x_pos_y','aggregate_neg_x_neg_y','aggregate_pos_x_neg_y','aggregate_neg_x_pos_y']
    for i, report in enumerate(reports):
        raw = report['gpu_paired_stereo_phases_raw_samples_ms']
        summaries = report['gpu_paired_stereo_phases_ms']
        if report['stereo_schedule'] == 'sequential':
            assert raw == summaries == {}
            assert report['gpu_paired_stereo_raw_samples_ms'] is None
            continue
        assert report['gpu_paired_stereo_raw_samples_ms'] == [9,1,5]
        paths = 4 if i % 2 == 0 else 8
        if i == 8:
            paths = 4
        if report['stereo_schedule'] != 'shared-census-sequential':
            expected = ['census_left','census_right','zero_packed_penalties'] + paths_names[:paths] + ['select']
        else:
            expected = ['shared_census_left','shared_census_right']
            for eye in ('left_',) if i == 8 else ('left_','right_'):
                expected += [eye+'zero_packed_penalties'] + [eye+p for p in paths_names[:paths]] + [eye+'select']
            if i == 8:
                expected += ['derived_right_select']
        assert set(raw) == set(expected) == set(summaries)
        for phase, key in enumerate(expected):
            values = [phase+9,phase+1,phase+5]
            assert raw[key] == values
            assert summaries[key] == {'mean':phase+5,'p50':phase+5,'p95':phase+9}


def timed_loop(source):
    at = source.index('i<frames+warmup;')
    start = source.rindex('for(', 0, at)
    brace = source.index('{', at)
    depth = 1
    end = brace+1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def test_metadata_patch_preserves_audited_timed_loop_bytes():
    # Freeze pre-metadata loops: this patch must add no waits, queries,
    # allocations, helper calls or collection work to the measured population.
    import hashlib
    digests = ('887aaa07c1b0992c39c0b0c64138888f45fa1ec5c1f0391a0c01bced0ac6a8d3',
               'fc1952562818ac2c1bd296f850048cdc90a6849a84b77732a88aa53b9bdbbe0f',
               '9ea025768c2bd2b27b359175acb2c5fc7ef0490090da69b96e9fa1d9702477f3')
    for name, expected in zip(APPS, digests):
        source = (ROOT / 'apps' / name).read_text()
        original = timed_loop(source)
        assert hashlib.sha256(original.encode()).hexdigest() == expected
        mutant = original.replace('{', '{ add_native_timing_metadata(result,frames,warmup,"x");', 1)
        assert hashlib.sha256(mutant.encode()).hexdigest() != expected


def test_actual_refinement_emission_marks_only_inactive_columns_unavailable(tmp_path, json_include):
    source = (ROOT / 'apps' / APPS[1]).read_text()
    start = source.index('    constexpr const char* keys[]')
    end = source.index('    output={};stage->collect();result["refinement_source_retirements"]', start)
    code = tmp_path / 'refinement.cpp'
    code.write_text('''#include "timing_metadata.hpp"
#include <array>
#include <vector>
#include <iostream>
using json=nlohmann::json;using namespace stereoforge;
json summary(auto const&){return {{"legacy","unchanged"}};}
json run(bool enabled){unsigned frames=3,warmup=0;constexpr unsigned count=20;
 std::array<std::vector<double>,count> samples;
 for(auto& values:samples){values={0.,0.,0.};}
 json result;
''' + source[start:end] + '''
 return result;
}
int main(){std::cout<<json::array({run(false),run(true)});}
''')
    binary = tmp_path / 'refinement'
    subprocess.run([shutil.which('c++'), '-std=c++20', '-Wall', '-Wextra', '-Werror',
                    '-I', str(json_include), '-I', str(ROOT / 'apps'),
                    str(code), '-o', str(binary)], check=True, capture_output=True, text=True)
    coarse, refined = json.loads(subprocess.check_output([str(binary)], text=True))
    inactive = {'gpu_target_rectify_ms','gpu_census_ms','gpu_guided_match_ms',
                'gpu_refined_lr_ms','gpu_refined_metric_ms','gpu_refinement_queue_span_ms'}
    assert coarse['raw_samples_ms'] == refined['raw_samples_ms']
    for key, values in coarse['raw_samples_ms'].items():
        assert values == [0,0,0]
        assert coarse[key] == refined[key] == {'legacy':'unchanged'}
        assert coarse['timing_metadata']['columns'][key]['status'] == ('unavailable' if key in inactive else 'available')
        assert refined['timing_metadata']['columns'][key]['status'] == 'available'
