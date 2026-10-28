import pytest
"""Exercise the actual Frame special members with CPU-only owner/slot children."""
from pathlib import Path
import shutil
import subprocess

from test_daxa_depth_hotpath import definition

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage", ["view", "temporal"])
def test_actual_frame_assignment_releases_children_before_context(tmp_path, stage):
    compiler = shutil.which('c++')
    assert compiler, 'CPU lifetime regression requires a C++ compiler'
    header = (ROOT/f'include/stereoforge/daxa_{stage}.hpp').read_text()
    start = header.index('    class Frame {')
    frame = header[start:header.index('\n    };', start)+7]
    source = (ROOT/f'src/daxa_{stage}.cpp').read_text()
    class_name = 'Daxa' + stage.capitalize()
    methods = definition(source, class_name + '::Frame::Frame(')
    signature = class_name + '::Frame& ' + class_name + '::Frame::operator=('
    if signature in source:
        methods += definition(source, signature)
    program = r'''
#include <memory>
#include <string>
#include <cstdint>
#include <utility>
#include <vector>
#include <stdexcept>
#include <iostream>
namespace daxa {struct BufferId{};}
namespace stereoforge {
class DaxaObservationFrame;class DaxaObservationAncestryView;struct DaxaTemporalTiming;
struct GeometryConfig{};struct CaptureFrame{};using CaptureLease=std::shared_ptr<CaptureFrame>;
enum class DepthConvention{AxialZ};
struct State{unsigned violations=0,closed=0,slots_destroyed=0;};
class DaxaView {
 struct Slot{uint64_t generation=7;std::shared_ptr<State> state;~Slot(){++state->slots_destroyed;}};
 struct Impl{std::shared_ptr<State> state;std::shared_ptr<Slot> slot;~Impl(){auto weak=std::weak_ptr<Slot>(slot);slot.reset();if(!weak.expired())++state->violations;++state->closed;}};
public:
'''+frame+r'''
 static auto engine(std::shared_ptr<State> s){auto e=std::make_shared<Impl>();e->state=s;e->slot=std::make_shared<Slot>();e->slot->state=s;return e;}
 static Frame output(std::shared_ptr<Impl> const& e){return Frame(e,e->slot);}
};
'''+methods+r'''
}
using namespace stereoforge;
void check(bool b,char const* m){if(!b)throw std::runtime_error(m);}
int main(){try{
 auto state=std::make_shared<State>();
 {auto e=DaxaView::engine(state);auto f=DaxaView::output(e);e.reset();check(state->closed==0,"external frame lost owner");f={};check(state->closed==1&&state->slots_destroyed==1,"empty assignment did not release owner");}
 for(bool moving:{false,true}){
  auto a=DaxaView::engine(state),b=DaxaView::engine(state);auto x=DaxaView::output(a),y=DaxaView::output(b);a.reset();b.reset();auto closed=state->closed;
  if(moving)x=std::move(y);else x=y;
  check(state->closed==closed+1,"cross-owner assignment retained old owner");
  auto* alias=&x;x=*alias;x=std::move(*alias);check(state->closed==closed+1,"self assignment released live owner");
  auto same=x;same=x;same=std::move(x);same={};y={};x={};
 }
 {std::vector<DaxaView::Frame> frames;for(unsigned i=0;i<4;++i){auto e=DaxaView::engine(state);frames.push_back(DaxaView::output(e));}frames.erase(frames.begin());frames.clear();}
 try{auto e=DaxaView::engine(state);auto frame=DaxaView::output(e);e.reset();throw std::runtime_error("unwind");}catch(std::runtime_error const&){}
 check(state->violations==0,"Frame assignment destroyed context while slot child survived");
 check(state->closed==state->slots_destroyed,"owner/slot destruction count differs");
 std::cout<<"actual Frame lifetime cases passed: "<<state->closed<<" contexts\n";
 return 0;
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
'''
    program = program.replace('DaxaView', class_name)
    cpp=tmp_path/'frame_lifetime.cpp';cpp.write_text(program)
    binary=tmp_path/'frame_lifetime'
    compiled=subprocess.run([compiler,'-std=c++20','-O2','-Wall','-Wextra','-Werror',str(cpp),'-o',str(binary)],capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stdout+compiled.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'actual Frame lifetime cases passed: 10 contexts' in result.stdout
