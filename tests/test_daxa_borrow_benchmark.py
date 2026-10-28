"""CPU-only checks for the direct-borrow benchmark's reporting contract."""
import json
from pathlib import Path
import shutil
import subprocess

from test_zed_native_timing_metadata import ROOT


def test_direct_borrow_is_selectable_and_keeps_common_timed_path():
    source = (ROOT / 'apps/daxa_refinement_benchmark.cpp').read_text()
    assert 'variant=="raw_borrow"' in source
    assert 'prepared.refinement.borrow_full_frame=borrow' in source
    assert 'all_30_planes_exact_direct_checked' in source
    assert 'all_30_planes_exact_copy_checked' in source
    assert 'reference_config.borrow_full_frame=false' in source
    assert 'prepared.target.rectification,coarse,reference_config' in source


def test_direct_borrow_emits_unavailable_first_stage(tmp_path, json_include):
    source = (ROOT / 'apps/daxa_refinement_benchmark.cpp').read_text()
    start = source.index('    constexpr const char* keys[]')
    end = source.index('    output={};stage->collect();result["refinement_source_retirements"]', start)
    code = tmp_path / 'borrow.cpp'
    code.write_text('''#include "timing_metadata.hpp"
#include <array>
#include <vector>
#include <iostream>
#include <limits>
using json=nlohmann::json;using namespace stereoforge;
json summary(auto const&){return {{"ok",true}};}
int main(){unsigned frames=3,warmup=0;constexpr unsigned count=20;bool enabled=true,reuse=true;
 std::array<std::vector<double>,count> samples;
 for(auto& values:samples){values={0.,0.,0.};}
 samples[13]={NAN,NAN,NAN}; json result; result["variant"]="raw_borrow";
''' + source[start:end] + '''
 std::cout<<result.dump();}
''')
    binary = tmp_path / 'borrow'
    subprocess.run([shutil.which('c++'), '-std=c++20', '-Wall', '-Wextra', '-Werror',
                    '-I', str(json_include), '-I', str(ROOT / 'apps'), str(code), '-o', str(binary)],
                   check=True, capture_output=True, text=True)
    report = json.loads(subprocess.check_output([str(binary)], text=True))
    assert report['gpu_target_rectify_ms'] is None
    assert report['raw_samples_ms']['gpu_target_rectify_ms'] is None
    assert report['timing_metadata']['columns']['gpu_target_rectify_ms']['status'] == 'unavailable'
    assert 'gpu_raw_reuse_pack_ms' not in report
    assert report['timing_metadata']['columns']['gpu_combined_queue_span_ms']['status'] == 'available'
