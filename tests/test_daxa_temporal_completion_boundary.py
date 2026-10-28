"""Pin the real Daxa API's completion-boundary meaning without loading Vulkan."""
from pathlib import Path
import shutil
import re
import subprocess
import pytest
from test_daxa_depth_hotpath import definition

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_daxa_completed_boundary_and_temporal_reuse(tmp_path):
    vendor = ROOT / 'build/daxa/_deps/sf_daxa-src/src/impl_device.cpp'
    if not vendor.exists():
        pytest.skip('optional pinned native Daxa source not configured')
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    vendor_source = vendor.read_text()
    compact = re.sub(r'\s+', '', vendor_source)
    assert 'current_timeline_value=self->global_submit_timeline.fetch_add(1)+1' in compact
    assert 'submit_semaphore_signal_values.push_back(current_timeline_value)' in compact
    actual_min = definition(vendor_source, 'auto daxa_dvc_oldest_pending_submit_index(')
    frame = (ROOT / 'include/stereoforge/daxa_frame.hpp').read_text()
    shared = definition(frame, 'inline bool completed_submission(') + definition(frame, 'inline bool available_slot(')
    actual_queue = definition(vendor_source, 'auto daxa_ImplDevice::ImplQueue::get_oldest_pending_submit(')
    state = (ROOT / 'include/stereoforge/detail/daxa_temporal_state.hpp').read_text()
    actual_completion = definition(state, 'inline bool temporal_tickets_complete(')
    ready_methods = '\n'.join(definition((ROOT / path).read_text(), signature) for path, signature in (
        ('src/daxa_frame.cpp', 'bool DaxaFrame::ready('),
        ('src/daxa_observation.cpp', 'bool DaxaObservationFrame::ready('),
        ('src/daxa_depth.cpp', 'bool DaxaDepthPipeline::Frame::ready('),
        ('src/daxa_temporal.cpp', 'bool DaxaTemporal::Frame::ready(')))
    program = r'''
#include <atomic>
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
using u64=uint64_t;using daxa_u64=u64;using VkDevice=int;using daxa_Result=int;
constexpr int DAXA_RESULT_SUCCESS=0,DAXA_RESULT_ERROR_DEVICE_LOST=-1;
std::array<u64,3> gpu_completed{0,1,0};
int vkGetSemaphoreCounterValue(VkDevice,u64 semaphore,u64* out){*out=gpu_completed[semaphore];return 0;}
#define _DAXA_RETURN_IF_ERROR(result,error) do{if(result!=0)return error;}while(false);
struct daxa_ImplDevice{struct ImplQueue{u64 gpu_queue_local_timeline=1;std::atomic<u64> latest_pending_submit_timeline_value=1;auto get_oldest_pending_submit(VkDevice,std::optional<u64>&)->daxa_Result;};VkDevice vk_device=0;std::array<ImplQueue,2> queues;};using daxa_Device=daxa_ImplDevice*;
''' + actual_queue + actual_min + '\nnamespace detail{\n' + shared + '\n}\n' + actual_completion + r'''
struct ReadyDevice{daxa_Device source;u64 oldest_pending_submit_index(){u64 bound=0;if(daxa_dvc_oldest_pending_submit_index(source,&bound)!=0)throw std::runtime_error("counter failure");return bound;}};
struct ReadyContext{ReadyDevice device;};struct ReadyOwner{ReadyContext* context;};struct ReadySlot{u64 submit=1,last_write=1;};
#define FRAME_MEMBERS ReadyOwner* owner_;ReadySlot* slot_;void validate()const{}bool ready()const;
struct DaxaFrame{FRAME_MEMBERS};
struct DaxaObservationFrame{ReadyContext* context_;u64 submit_=1;void validate()const{}bool ready()const;};
struct DaxaDepthPipeline{struct Frame{FRAME_MEMBERS};};
struct DaxaTemporal{struct Frame{FRAME_MEMBERS};};
''' + ready_methods + r'''
int main(){
 daxa_ImplDevice device;device.queues[1].gpu_queue_local_timeline=2;device.queues[1].latest_pending_submit_timeline_value=0;
 auto& queue=device.queues[0];std::optional<u64> boundary;u64 global=0;
 if(queue.get_oldest_pending_submit(0,boundary)!=0||boundary)throw std::runtime_error("idle queue must be omitted from global min/UINT64_MAX");
 if(daxa_dvc_oldest_pending_submit_index(&device,&global)!=0||global!=UINT64_MAX||!temporal_tickets_complete(1,1,global))throw std::runtime_error("idle admission failed");
 queue.latest_pending_submit_timeline_value=2;
 if(queue.get_oldest_pending_submit(0,boundary)!=0||boundary!=1)throw std::runtime_error("pinned Daxa API changed: boundary is no longer latest completed value");
 if(!temporal_tickets_complete(1,1,*boundary)||!detail::available_slot(1,1,*boundary))throw std::runtime_error("already completed history read rejected after upstream submit");
 ReadyContext context{{&device}};ReadyOwner owner{&context};ReadySlot slot;
 DaxaFrame raw{&owner,&slot};DaxaObservationFrame common{&context};DaxaDepthPipeline::Frame depth{&owner,&slot};DaxaTemporal::Frame temporal{&owner,&slot};
 if(!raw.ready()||!common.ready()||!depth.ready()||!temporal.ready())throw std::runtime_error("completed frame/common readiness rejected equality under newer pending work");
 slot.submit=slot.last_write=common.submit_=2;
 if(raw.ready()||common.ready()||depth.ready()||temporal.ready())throw std::runtime_error("future frame/common readiness incorrectly reported complete");
 if(temporal_tickets_complete(1,2,*boundary)||temporal_tickets_complete(2,1,*boundary)||detail::available_slot(1,2,*boundary)||detail::available_slot(2,1,*boundary))throw std::runtime_error("future ticket or retained slot treated as reusable");
 device.queues[1].latest_pending_submit_timeline_value=9;gpu_completed[2]=7;
 if(daxa_dvc_oldest_pending_submit_index(&device,&global)!=0||global!=1||detail::completed_submission(2,global))throw std::runtime_error("minimum across pending queues must conservatively protect future tickets");
 gpu_completed[1]=2;
 if(daxa_dvc_oldest_pending_submit_index(&device,&global)!=0||global!=7||!detail::completed_submission(7,global)||detail::completed_submission(8,global))throw std::runtime_error("idle queue incorrectly constrained inclusive global completion bound");
 gpu_completed[2]=9;
 if(daxa_dvc_oldest_pending_submit_index(&device,&global)!=0||global!=UINT64_MAX)throw std::runtime_error("all idle queues must report MAX");
}
'''
    source = tmp_path / 'boundary.cpp'
    source.write_text(program)
    binary = tmp_path / 'boundary'
    compiled = subprocess.run([compiler, '-std=c++20', '-O2', str(source), '-o', str(binary)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
