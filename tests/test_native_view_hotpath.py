"""Source guards for standalone fusion benchmark measurement boundaries; no GPU."""
from pathlib import Path

ROOT=Path(__file__).parents[1]


def body(source,marker):
    start=source.index('{',source.index(marker));end=start+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start+1:end-1],end


def test_measured_gpu_case_has_only_completion_and_conditional_cpu_readback():
    source=(ROOT/'apps/native_view_benchmark.cpp').read_text()
    measured,end=body(source,'for(unsigned i=0;i<opt.frames;++i)')
    assert measured.count('last=project()')==1
    assert source.count('resident?engine.project(*resident,input.view):engine.project(sources,input.view)')==1
    assert measured.count('completion.wait()')==1
    assert measured.count('engine.readback(')==1
    assert 'if(opt.readback=="cpu")last_cpu=engine.readback(*last);' in measured
    assert measured.index('last.reset()')<measured.index('const auto start=Clock::now()')
    assert measured.index('completion.wait()')<measured.index('samples.push_back(')
    assert 'dump(' not in measured
    assert 'if(opt.readback=="gpu")last_cpu=engine.readback(*last)' in source[end:]
    assert 'host input uploads every iteration, resident input uploads once before warmup' in source
    assert 'Excludes stereo, camera, display, initialization and one-time resident upload' in source


def test_warmup_and_initialization_excluded_and_inputs_reused():
    source=(ROOT/'apps/native_view_benchmark.cpp').read_text()
    warmup,end=body(source,'for(unsigned i=0;i<opt.warmup;++i)')
    assert 'samples.push_back(' not in warmup
    assert source.index('NativeGeometry engine(')<source.index('for(unsigned i=0;i<opt.warmup;++i)')
    assert end<source.index('for(unsigned i=0;i<opt.frames;++i)')
    assert source.count('NativeGeometry engine(')==1
    assert 'const auto sources=input.sources();' in source
    measured,_=body(source,'for(unsigned i=0;i<opt.frames;++i)')
    assert not any(operation in measured for operation in ('load<','fixture(','NativeGeometry ','sources=','sources.','input.depth','input.rgb'))
    # The benchmark reads packed fixture bytes; it never computes host color fusion.
    assert not any(operation in source for operation in ('synthesize(','median(','blend(','cv::'))
