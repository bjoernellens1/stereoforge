from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'apps/daxa_refinement_benchmark.cpp'


def test_full_producer_control_and_reuse_share_source_and_order():
    s = SOURCE.read_text()
    assert 'variant=="coarse_refine_full"||variant=="raw_reuse"' in s
    loop = s[s.index('for(unsigned i=0;i<frames+warmup;++i)'):s.index('auto result=prepared.report()')]
    assert loop.index('depth->enqueue(prepared.source)') < loop.index('full_pre->enqueue(prepared.source)') < loop.index('stage->process(coarse,full)')
    assert loop.index('stage->process(coarse,full)') < loop.index('output.wait()') < loop.index('full_pre->timings(full)')
    assert 'start<=end&&end<=final' in loop
    assert 'ft.timestamp_bits==dt.preprocessing.timestamp_bits' in loop
    assert 'ft.timestamp_period==dt.preprocessing.timestamp_period' in loop
    assert 'timestamp wrap ambiguous' in loop
    assert 'readback' not in loop


def test_reuse_requires_post_timing_exact_comparison_of_all_planes():
    s = SOURCE.read_text()
    assert s.index('auto result=prepared.report()') < s.index('stage->readback(output)') < s.index('exact_readback(r,direct)')
    body = s[s.index('static void exact_readback'):s.index('int main(')]
    assert body.count('EQ(') == 29  # 28 scalar planes plus macro declaration
    assert 'for(unsigned eye=0;eye<2;++eye)exact_plane(a.rectified_rgb[eye],b.rectified_rgb[eye])' in body
    assert 'all_30_planes_exact_direct_checked' in s
    assert 'raw_samples_ms' in s and 'full_preprocess_memory_scope' in s
