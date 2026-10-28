"""Static guard for the actual matched raw-reuse timing loop; no GPU claims."""
from pathlib import Path
import re

import pytest

from test_zed_native_timing_metadata import timed_loop

SOURCE = Path(__file__).resolve().parents[1] / 'apps/daxa_refinement_benchmark.cpp'


def guard(source):
    loop = timed_loop(source)
    assert not re.search(r'\b(?:new|make_shared|make_unique|malloc|calloc|resize|reserve|push_back|readback|dump|memcpy|memmove|wait_idle)\b', loop)
    assert loop.count('output.wait()') == 1 and loop.count('coarse.wait()') == 1
    assert loop.index('depth->enqueue(prepared.source)') < loop.index('full_pre->enqueue(prepared.source)') < loop.index('stage->process(coarse,full)')
    assert loop.index('auto b=Clock::now()') < loop.index('output.wait()') < loop.index('auto c=Clock::now()') < loop.index('full_pre->timings(full)')
    assert loop.index('auto c=Clock::now()') < loop.index('auto d=Clock::now()')
    assert 'samples[k][i-warmup]=values[k]' in loop
    assert 'full_samples[k][i-warmup]=fv[k]' in loop
    assert source.index('stage->readback(output)') > source.index(loop) + len(loop)
    assert source.index('exact_readback(r,direct)') > source.index(loop) + len(loop)


def test_actual_raw_reuse_loop():
    guard(SOURCE.read_text())


@pytest.mark.parametrize('injection', [
    'stage->readback(output);', 'context->device.wait_idle();',
    'samples[0].push_back(1.0);', 'std::make_shared<int>();',
    'output.wait();',
])
def test_raw_reuse_loop_rejects_extra_operations(injection):
    source = SOURCE.read_text()
    marker = 'for(unsigned i=0;i<frames+warmup;++i){'
    assert marker in source
    with pytest.raises(AssertionError):
        guard(source.replace(marker, marker + injection, 1))


def guard_borrow_reporting(source):
    loop = timed_loop(source)
    post = source[source.index(loop) + len(loop):]
    assert 'if(k==13&&result.value("variant",std::string{})=="raw_borrow")' in post
    assert 'result[keys[k]]=nullptr;result["raw_samples_ms"][keys[k]]=nullptr' in post
    assert 'if(reuse&&result.value("variant",std::string{})!="raw_borrow")' in post
    assert 'exact_readback(r,direct)' in post and 'exact_readback(r,copy->readback(copy_frame))' in post
    assert 'reference_config.borrow_full_frame=false' in post
    assert 'copy_config.borrow_full_frame=false' in post


def test_borrow_reports_absent_timestamp_and_checks_both_references():
    guard_borrow_reporting(SOURCE.read_text())


@pytest.mark.parametrize('old,new', [
    ('result[keys[k]]=nullptr;result["raw_samples_ms"][keys[k]]=nullptr', 'result[keys[k]]=0;result["raw_samples_ms"][keys[k]]=nullptr'),
    ('exact_readback(r,direct)', 'ignore(r,direct)'),
    ('exact_readback(r,copy->readback(copy_frame))', 'ignore(r,copy->readback(copy_frame))'),
    ('copy_config.borrow_full_frame=false', 'copy_config.borrow_full_frame=true'),
])
def test_borrow_reporting_rejects_missing_checks(old, new):
    source = SOURCE.read_text()
    assert old in source
    with pytest.raises(AssertionError):
        guard_borrow_reporting(source.replace(old, new, 1))
