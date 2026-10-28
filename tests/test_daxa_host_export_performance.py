"""Independent actual-source CPU/control guards; excludes GPU and driver cost."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import pytest
from test_kernel_variant_hotpath import body

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / '.superpowers/sdd/2026-10-28-native-observation-export/performance'
PATHS = ('src/daxa_host_export.cpp', 'include/stereoforge/daxa_host_export.hpp',
         'src/daxa_frame.cpp', 'src/daxa_observation.cpp',
         'include/stereoforge/daxa_frame.hpp', 'include/stereoforge/daxa_observation.hpp',
         'include/stereoforge/ros_depth_contract.hpp')
FORBIDDEN = r'\b(?:wait_idle|wait_on_submit|wait|readback\w*|create_memory|create_buffer\w*|allocate\w*|make_shared|make_unique|malloc|calloc|realloc|resize|reserve|push_back|emplace_back|memcpy|memmove|memset)(?:<[^;()]*>)?\s*\(|\bnew\s+'


def sources():
    return {name: (ROOT / name).read_text() for name in PATHS}


def check_hotpaths(src):
    export = src['src/daxa_host_export.cpp']
    selected = {
        'src/daxa_host_export.cpp': ['void require(', 'bool same_owner(', 'void require_thread(',
            'void validate(uint64_t', 'std::shared_ptr<Slot>& available(', 'template<class Input>void submit(',
            'DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaObservationFrame&',
            'DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaFrame&', 'size_t DaxaHostExport::collect(',
            'DaxaHostExport::Frame::Frame(', 'DaxaHostExport::Frame& DaxaHostExport::Frame::operator=(',
            'void DaxaHostExport::Frame::validate('],
        'src/daxa_frame.cpp': ['void require(', 'void DaxaFrame::validate(', 'DaxaFrame::context_id(',
            'DaxaFrame::submit_ticket(', 'CaptureLease DaxaFrame::source(', 'DaxaFrame::calibration(',
            'DaxaFrame::plane(', 'DaxaImageView::DaxaImageView(', 'DaxaImageView::buffer(',
            'DaxaImageView::width(', 'DaxaImageView::height('],
        'src/daxa_observation.cpp': ['void require(', 'void DaxaObservationFrame::validate(',
            'DaxaObservationFrame::context_id(', 'DaxaObservationFrame::width(', 'DaxaObservationFrame::height(',
            'DaxaObservationFrame::submit_ticket(', 'DaxaObservationFrame::source(', 'DaxaObservationFrame::map_owner(',
            'DaxaObservationFrame::buffer(', 'DaxaObservationFrame::size_bytes(', 'DaxaObservationFrame::image('],
        'include/stereoforge/daxa_frame.hpp': ['inline bool completed_submission(', 'inline bool available_slot(', 'explicit operator bool('],
        'include/stereoforge/daxa_observation.hpp': ['daxa::BufferId buffer()const', 'uint32_t width()const',
            'uint32_t height()const', 'explicit operator bool(', 'calibration()const'],
        'include/stereoforge/ros_depth_contract.hpp': ['inline void require(', 'inline bool known(',
            'inline RosCaptureStamp map_ros_capture_stamp(', 'inline void payload(',
            'inline void validate_ros_depth_payload(', 'inline void validate_ros_rgba_payload(', 'void admit('],
    }
    for path, signatures in selected.items():
        for signature in signatures:
            assert not re.search(FORBIDDEN, body(src[path], signature)), path + ':' + signature + ' forbidden hotpath work'
    assert 'std::array<std::shared_ptr<Slot>,3> slots' in export
    assert 'cfg.slots>=2&&cfg.slots<=3' in export
    available = body(export, 'std::shared_ptr<Slot>& available(')
    assert 'i<config.slots' in available and 'detail::available_slot(' in available
    for signature in ('DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaObservationFrame&',
                      'DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaFrame&'):
        enqueue = body(export, signature)
        assert enqueue.index('e.available()') < enqueue.index('input.context_id()')
        assert 'for(' not in enqueue and 'while(' not in enqueue
    submit = body(export, 'template<class Input>void submit(')
    assert submit.count('copy_buffer_to_buffer(') == 2
    assert 'if(config.depth)recorder.copy_buffer_to_buffer(' in submit
    assert 'if(config.rgb)recorder.copy_buffer_to_buffer(' in submit
    assert '.src_offset=0,.dst_offset=0,.size=plane_bytes' in submit
    assert '.dst_offset=config.depth?plane_bytes:0,.size=plane_bytes' in submit
    assert submit.index('s.producer=input') < submit.index('d.submit_commands(')
    assert 'if(latest>before)s.copy_ticket=latest' in submit
    setup = body(export, 'Impl(std::shared_ptr<DaxaFoundation>')
    for marker in ('VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT',
                   'VK_MEMORY_PROPERTY_HOST_CACHED_BIT', 'requirements.memory_type_bits=1u<<i',
                   'selected.type_index=i', 'selected.property_flags=flags', 'require(found,'):
        assert marker in setup
    assert 'pass<2&&!found' in setup and 'pass||bool(flags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT)' in setup
    assert 'references==1&&completed_submission(submitted,oldest_pending)' in src['include/stereoforge/daxa_frame.hpp']
    admit = body(src['include/stereoforge/ros_depth_contract.hpp'], 'void admit(')
    assert 'clock_=clock;' in body(admit, 'else')


def test_transitive_hotpaths_and_fixed_selective_storage():
    check_hotpaths(sources())


@pytest.mark.parametrize('signature', ['void require_thread(', 'bool same_owner(',
                                      'std::shared_ptr<Slot>& available(', 'template<class Input>void submit('])
@pytest.mark.parametrize('injection', ['d.wait_idle();', 'producer.wait();', 'auto x=std::make_shared<int>();',
                                       'v.resize(5);', 'memcpy(a,b,n);', 'd.create_memory({});'])
def test_guard_rejects_helper_regressions(signature, injection):
    src = sources(); text = src['src/daxa_host_export.cpp']
    at = text.index('{', text.index(signature)) + 1
    src['src/daxa_host_export.cpp'] = text[:at] + injection + text[at:]
    with pytest.raises(AssertionError, match='forbidden hotpath work'):
        check_hotpaths(src)


@pytest.mark.parametrize('old,new', [
    ('if(config.rgb)recorder.copy_buffer_to_buffer(', 'recorder.copy_buffer_to_buffer('),
    ('std::array<std::shared_ptr<Slot>,3> slots', 'std::vector<std::shared_ptr<Slot>> slots'),
    ('requirements.memory_type_bits=1u<<i', 'requirements.memory_type_bits=~0u'),
    ('if(latest>before)s.copy_ticket=latest', 'if(latest>before)s.copy_ticket=0'),
])
def test_guard_rejects_copy_cache_and_retention_regressions(old, new):
    src = sources(); assert old in src['src/daxa_host_export.cpp']
    src['src/daxa_host_export.cpp'] = src['src/daxa_host_export.cpp'].replace(old, new, 1)
    with pytest.raises(AssertionError):
        check_hotpaths(src)


PROBE_MAIN = r'''
using namespace stereoforge;
int main(){try{
 properties.memoryTypeCount=1;properties.memoryTypes[0].propertyFlags=7;
 auto c=std::make_shared<DaxaFoundation>();auto maps=std::make_shared<DaxaRectification>();
 auto state=std::make_shared<InputState>();state->context=c;state->maps=maps;state->source=std::make_shared<CaptureFrame>();state->depth=c->device.source(280,3);state->rgb=c->device.source(140,29);
 DaxaObservationFrame input{{state,1}};
 constexpr unsigned frames=65536;
 for(unsigned mode=0;mode<4;++mode){
  DaxaHostExportConfig cfg{2,mode==0||mode==2,mode!=0};DaxaHostExport exporter(c,input,cfg);
  auto buffers=c->device.allocations,memories=c->device.memories;
  for(unsigned warm=0;warm<512;++warm){auto out=mode==3?exporter.enqueue(static_cast<const DaxaFrame&>(input)):exporter.enqueue(input);c->device.completed=c->device.latest;exporter.collect();}
  for(unsigned run=0;run<3;++run){
   auto copies=mock_copies,submits=c->device.submits,recordings=c->device.recordings;heap_allocations=0;track_heap=true;
   auto start=std::chrono::steady_clock::now();
   for(unsigned i=0;i<frames;++i){auto out=mode==3?exporter.enqueue(static_cast<const DaxaFrame&>(input)):exporter.enqueue(input);c->device.completed=c->device.latest;exporter.collect();}
   auto ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count()/frames;
   track_heap=false;
   if(heap_allocations||c->device.waits||c->device.allocations!=buffers||c->device.memories!=memories||c->device.submits-submits!=frames||c->device.recordings-recordings!=frames||mock_copies-copies!=frames*(unsigned(cfg.depth)+unsigned(cfg.rgb)))throw std::runtime_error("hotpath budget regression");
   std::cout<<"{\"mode\":"<<mode<<",\"run\":"<<run<<",\"frames\":"<<frames<<",\"cpu_ns_per_enqueue_collect\":"<<ns<<",\"application_allocations\":"<<heap_allocations<<",\"waits\":"<<c->device.waits<<",\"copies_per_enqueue\":"<<(unsigned(cfg.depth)+unsigned(cfg.rgb))<<"}\n";
  }
 }
 return 0;
}catch(const std::exception& e){track_heap=false;std::cerr<<e.what()<<'\n';return 1;}}
'''


def test_actual_source_enqueue_cpu_microprobe(tmp_path):
    import shutil
    from test_daxa_host_export_hotpath import MOCK
    compiler = shutil.which('c++')
    assert compiler, 'C++20 compiler required for actual-source performance probe'
    # Reuse native API stand-ins, replacing only mocked driver resource names,
    # recording logs and simulated CPU memcpy with fixed work. Production
    # application code remains unchanged. GPU/driver recording costs excluded.
    mock = MOCK.replace('#include <array>', '#include <array>\n#include <chrono>\n#include <cstdlib>\n#include <new>\n#include <string_view>\n'
        'bool track_heap=false;size_t heap_allocations=0;unsigned mock_copies=0;\n'
        'void* operator new(size_t n){if(track_heap)++heap_allocations;if(auto p=std::malloc(n))return p;throw std::bad_alloc();}\n'
        'void operator delete(void* p)noexcept{std::free(p);}\n'
        'void operator delete(void* p,size_t)noexcept{std::free(p);}\n')
    mock = mock.replace('struct Name{std::string name;};', 'struct Name{std::string_view name;};')
    assert 'barriers->push_back(b);' in mock and 'submitted_copies.push_back(c);' in mock
    mock = mock.replace('barriers->push_back(b);', '(void)b;')
    mock = mock.replace('submitted_copies.push_back(c);', '++mock_copies;')
    copy = 'std::memcpy(buffers[c.dst_buffer.index].data()+c.dst_offset,buffers[c.src_buffer.index].data()+c.src_offset,c.size);'
    assert copy in mock
    mock = mock.replace(copy, '(void)c;')
    src = sources()
    header = src['include/stereoforge/daxa_host_export.hpp']
    source = src['src/daxa_host_export.cpp']
    code = mock + header[header.index('namespace stereoforge {'):] + source[source.index('namespace stereoforge {'):] + PROBE_MAIN
    cpp = tmp_path / 'export_performance.cpp'; cpp.write_text(code)
    binary = tmp_path / 'export_performance'
    command = [compiler, '-std=c++20', '-O3', '-pthread', str(cpp), '-o', str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 12
    assert all(row['application_allocations'] == 0 and row['waits'] == 0 for row in rows)
    # Broad control-only ceiling catches accidental scans without pretending
    # that these host timings qualify real device copies or throughput.
    assert all(0 < row['cpu_ns_per_enqueue_collect'] < 50000 for row in rows)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / 'actual-source-enqueue-probe.cpp').write_text(code)
    (ARTIFACTS / 'actual-source-enqueue.json').write_text(json.dumps({
        'scope': 'actual production export enqueue + collect; bounded API mocks; no GPU/driver/copy or ROS cost',
        'source_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in src.items()},
        'translation_unit_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'command': command, 'rows': rows}, indent=2) + '\n')
