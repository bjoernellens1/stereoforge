"""Control-only probe of the actual benchmark loop, with fake pipeline/device."""
from pathlib import Path
import subprocess
import pytest
from test_kernel_variant_hotpath import body

SOURCE = Path(__file__).resolve().parents[1] / 'apps/daxa_dual_preprocess_benchmark.cpp'
LOOP = 'for(unsigned i=0;i<frames+warmup;++i)'


def check_loop(src):
    loop = body(src, LOOP)
    assert loop.count('output.wait()') == 1, 'one intended serial completion wait'
    assert all(s not in loop for s in ('readback(', 'memcpy(', 'wait_idle(', 'exact_readback(', 'resize(', 'reserve(')), 'diagnostics or extra work in timed loop'
    assert 'if(variant=="dual"){newref.collect();newdepth.collect();dual.collect();}else{oldref.collect();olddepth.collect();oldfull.collect_completed_sources();}' in loop, 'collect only active path'
    assert 'if(variant=="dual"){live=dual.enqueue(prepared.source);coarse=newdepth.enqueue_preprocessed(live.coarse);output=newref.process(coarse,live.full);}else{coarse=olddepth.enqueue(prepared.source);full=oldfull.enqueue(prepared.source);output=oldref.process(coarse,full);}' in loop, 'one source per producer path'
    assert loop.index('auto a=Clock::now()') < loop.index('output={}') < loop.index('auto b=Clock::now()') < loop.index('output.wait()') < loop.index('auto c=Clock::now()') < loop.index('if(i>=warmup)')
    assert src.index('exact_readback(oldref.readback(oldout),newref.readback(newout))') > src.index(LOOP)
    assert 'serialized same retained source' in src and 'both configurations allocated' in src
    assert src.index('ctx->close()') < src.index('validation_errors_after_teardown')


def test_benchmark_loop_contract():
    check_loop(SOURCE.read_text())


@pytest.mark.parametrize('before,after', [
    ('output.wait();auto c=Clock::now();', 'output.wait();output.wait();auto c=Clock::now();'),
    ('newref.collect();newdepth.collect();dual.collect();', 'newref.collect();newdepth.collect();dual.collect();olddepth.collect();'),
    ('coarse=newdepth.enqueue_preprocessed(live.coarse);', 'coarse=newdepth.enqueue_preprocessed(live.coarse);full=oldfull.enqueue(prepared.source);'),
    ('output.wait();auto c=Clock::now();', 'output.wait();oldref.readback(output);auto c=Clock::now();'),
])
def test_benchmark_guard_rejects_mutations(before, after):
    src=SOURCE.read_text();assert before in src
    with pytest.raises(AssertionError):
        check_loop(src.replace(before,after,1))


def test_actual_benchmark_loop_fake_device(tmp_path):
    src=SOURCE.read_text();loop=LOOP+'{'+body(src,LOOP)+'}'
    program=r'''
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
struct Counts{int oc=0,nc=0,odc=0,ndc=0,ofc=0,dc=0,oe=0,ne=0,fe=0,de=0,op=0,np=0,waits=0;} t;
struct Frame {int id=0;void wait(){++t.waits;}};
struct Pair {Frame full,coarse;};
struct Pre {bool dual;void collect_completed_sources(){++t.ofc;}void collect(){++t.dc;}Frame enqueue(int){++t.fe;return {};}Pair enqueue_pair(int){++t.de;return {};}};
struct Dual {void collect(){++t.dc;} Pair enqueue(int){++t.de;return {};}};
struct Depth {bool newer;void collect(){if(newer)++t.ndc;else ++t.odc;}Frame enqueue(int){++t.oe;return {};}Frame enqueue_preprocessed(Frame){++t.ne;return {};}};
struct Ref {bool newer;void collect(){if(newer)++t.nc;else ++t.oc;}Frame process(Frame,Frame){if(newer)++t.np;else ++t.op;return {};}};
struct Prepared {int source=3;};
int main(int argc,char** argv){std::string variant=argv[1];unsigned frames=10,warmup=2;Prepared prepared;Ref oldref{false},newref{true};Depth olddepth{false},newdepth{true};Pre oldfull{false};Dual dual;Frame output,coarse,full;Pair live;std::vector<double> wall,enqueue,wait;wall.reserve(frames);enqueue.reserve(frames);wait.reserve(frames);
'''+loop+r'''
 if(wall.size()!=10||enqueue.size()!=10||wait.size()!=10||t.waits!=12)throw std::runtime_error("sample or wait mismatch");
 if(variant=="dual"){if(t.nc!=12||t.ndc!=12||t.dc!=12||t.de!=12||t.ne!=12||t.np!=12||t.oc||t.odc||t.ofc||t.oe||t.fe||t.op)throw std::runtime_error("dual counter mismatch");}
 else{if(t.oc!=12||t.odc!=12||t.ofc!=12||t.oe!=12||t.fe!=12||t.op!=12||t.nc||t.ndc||t.dc||t.de||t.ne||t.np)throw std::runtime_error("separate counter mismatch");}
 std::cout<<"fake benchmark loop "<<variant<<" samples=10 warmup=2 waits=12\n";
}
'''
    cpp=tmp_path/'loop.cpp';binary=tmp_path/'loop';cpp.write_text(program)
    built=subprocess.run(['c++','-std=c++20','-O2',str(cpp),'-o',str(binary)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    for variant in ('dual','separate'):
        result=subprocess.run([str(binary),variant],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        print(result.stdout)
