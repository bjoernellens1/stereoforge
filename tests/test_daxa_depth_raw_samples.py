"""Compile the actual post-timing reporter without a Vulkan device."""
from pathlib import Path
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'apps/daxa_depth_benchmark.cpp'
KEYS = ('enqueue_cpu_ms', 'wait_cpu_ms', 'timestamp_query_cpu_ms', 'host_wall_ms',
        'upload_cpu_ms', 'gpu_upload_ms', 'gpu_decode_ms', 'gpu_rectify_ms',
        'gpu_left_stereo_ms', 'gpu_right_stereo_ms', 'gpu_lr_ms',
        'gpu_metric_metadata_ms', 'gpu_combined_queue_span_ms')


def test_actual_reporter_preserves_chronology_and_old_summaries(tmp_path):
    source = SOURCE.read_text()
    start = source.index('json raw_samples_report(')
    end = source.index('\nvoid verify(', start)
    helper = source[start:end]
    summary = source[source.index('json summary('):start]
    code = tmp_path / 'report.cpp'
    code.write_text('''#include <json.hpp>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
using json=nlohmann::json;
''' + summary + helper + '''
int main(){
 std::array<std::vector<double>,13> samples;
 for(unsigned k=0;k<13;++k)samples[k]={9.+k,1.+k,5.+k,3.+k};
 auto original=samples;json before=json::array();
 for(auto const& v:samples)before.push_back(summary(v));
 auto sequential=raw_samples_report(samples,true);
 auto paired=raw_samples_report(samples,false);
 json after=json::array();for(auto const& v:samples)after.push_back(summary(v));
 std::cout<<json{{"sequential",sequential},{"paired",paired},{"before",before},
 {"after",after},{"unchanged",samples==original}};
}
''')
    compiler = shutil.which('c++')
    assert compiler
    binary = tmp_path / 'report'
    subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'build/daxa/_deps/sf_native_json-src'),
                    str(code), '-o', str(binary)], check=True, capture_output=True, text=True)
    result = json.loads(subprocess.check_output([str(binary)], text=True))
    assert result['unchanged']
    assert result['before'] == result['after']
    assert set(result['sequential']) == set(KEYS)
    for k, key in enumerate(KEYS):
        expected = [9+k, 1+k, 5+k, 3+k]
        assert result['sequential'][key] == expected
        assert result['paired'][key] == (None if k in (8, 9) else expected)
        assert result['before'][k] == {'p50': 5+k, 'p95': 9+k, 'mean': 4.5+k}


def test_raw_reporting_is_after_teardown_and_outside_timed_loop():
    source = SOURCE.read_text()
    loop_start = source.index('for(uint32_t i=0;i<frames+warmup;++i)')
    diagnostic = source.index('auto diagnostic_begin=', loop_start)
    emission = source.index('result["raw_samples_ms"]=raw_samples_report(')
    assert source.index('context->close();', diagnostic) < emission
    assert emission > diagnostic
    assert 'raw_samples_report(' not in source[loop_start:diagnostic]
    assert 'raw_samples_ms' not in source[loop_start:diagnostic]
