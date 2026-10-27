"""CPU-only guards for the independent, bounded native capture path."""
from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest
ROOT = Path(__file__).resolve().parents[1]

def test_native_capture_has_no_python_or_gpu_runtime_dependency():
    source = (ROOT / 'src/capture.cpp').read_text()
    cmake = (ROOT / 'cmake/NativeCapture.cmake').read_text()
    for forbidden in ('pybind11', 'Python.h', 'slangpy', 'sgl::', 'cuda', 'hip/'):
        assert forbidden not in source
        assert forbidden not in cmake

def test_controls_are_cached_outside_camera_frame_read():
    source = (ROOT / 'src/capture.cpp').read_text()
    read = source.split('CaptureSourceStatus read(CaptureFrame&', 1)[1].split('\n};', 1)[0]
    for forbidden in ('getExposure(', 'getGain(', 'getAECAGC(', 'getROIforAECAGC(', 'getWhiteBalance('):
        assert forbidden not in read
    assert 'snapshotFrame(f.packed_yuyv' in read

def test_replay_streams_and_capture_thread_has_no_ownership_cycle():
    source = (ROOT / 'src/capture.cpp').read_text()
    replay = source.split('CaptureSource packed_replay_source(', 1)[1]
    assert 'read_packed_capture(' not in replay
    assert 'reader->next(f)' in replay
    assert 'shared_from_this' not in source
    assert 'std::thread([this]' in source

def test_generated_upstream_snapshot_is_shared_and_producer_locked():
    common = (ROOT / 'cmake/ZedCapture.cmake').read_text()
    for adapter in ('Capture.cmake', 'NativeCapture.cmake'):
        assert 'ZedCapture.cmake' in (ROOT / 'cmake' / adapter).read_text()
    assert 'std::lock_guard<std::mutex> lock(mBufMutex)' in common
    assert 'std::lock_guard<std::mutex> lock(mIMUMutex)' in common


def check_capture_hotpath(source):
    run = source.split('void run(){', 1)[1].split('void stop(){', 1)[0]
    delivery = source.split('CaptureLease NativeCapture::read(', 1)[1].split('\nnamespace {', 1)[0]
    for operation in ('make_shared', 'make_unique', 'resize', 'reserve', 'push_back', 'pop_front', 'memcpy', 'copy'):
        assert not re.search(r'\b' + operation + r'(?:<[^;()]*>)?\s*\(', run + delivery), 'capture handoff must not allocate or copy whole frames'
    assert 'std::deque' not in source, 'deque queue blocks allocate during steady capture'
    assert 'source(frame,config.source_timeout)' in run
    assert 'std::move(slot)' in run and 'std::move(impl_->queue[' in delivery


def test_capture_handoff_keeps_payload_and_queue_storage_preallocated():
    check_capture_hotpath((ROOT / 'src/capture.cpp').read_text())


@pytest.mark.parametrize('regression', ['queue.push_back(slot);', 'std::make_shared<CaptureFrame>();', 'std::memcpy(dst,src,bytes);'])
def test_capture_guard_detects_new_allocation_or_full_copy(regression):
    source = (ROOT / 'src/capture.cpp').read_text()
    with pytest.raises(AssertionError, match='allocate or copy'):
        check_capture_hotpath(source.replace('void run(){', 'void run(){' + regression, 1))


def test_snapshot_rejects_repeat_before_copying_the_full_frame():
    adapter = (ROOT / 'cmake/ZedCapture.cmake').read_text().split('sl_oc::video::Frame sl_oc::video::VideoCapture::snapshotFrame(', 1)[1]
    assert adapter.index('copy.timestamp==previous_timestamp') < adapter.index('bytes.assign(')
    source = (ROOT / 'src/capture.cpp').read_text()
    assert 'snapshotFrame(f.packed_yuyv,uint64_t(timeout.count()),last_timestamp_)' in source


ALLOCATION_PROBE = r'''
#include <stereoforge/capture.hpp>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <new>
#include <thread>
using namespace stereoforge;
using Clock=std::chrono::steady_clock;
std::atomic<bool> track=false;
std::atomic<unsigned long> allocations=0;
void* operator new(std::size_t n){if(track)++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete[](void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
void operator delete[](void* p,std::size_t)noexcept{std::free(p);}
int main(int argc,char**){
  CaptureConfig config;config.source_timeout=std::chrono::milliseconds{20};
  if(argc>1)config.queue_policy=CaptureQueuePolicy::wait_for_consumer;
  const std::size_t bytes=1280*4*720;
  std::vector<uint8_t> pixels(bytes,173);
  std::mutex mutex;std::condition_variable gate;bool requested=false,finished=false;
  uint64_t id=0,copy_ns=0;unsigned copies=0;
  NativeCapture capture(config,[&](CaptureFrame& f,auto timeout){
    {std::unique_lock lock(mutex);if(!gate.wait_for(lock,timeout,[&]{return requested||finished;}))return CaptureSourceStatus::timeout;
     if(finished)return CaptureSourceStatus::end;requested=false;}
    auto start=Clock::now();f.packed_yuyv.resize(bytes);std::memcpy(f.packed_yuyv.data(),pixels.data(),bytes);
    copy_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();++copies;
    f.source_width=2560;f.height=720;f.stride_bytes=5120;f.frame_id=++id;f.camera_timestamp_ns=id;
    f.camera_clock_domain="fixture_clock";return CaptureSourceStatus::frame;
  });
  auto one=[&]{
    {std::lock_guard lock(mutex);requested=true;}gate.notify_one();
    auto frame=capture.read(std::chrono::milliseconds{1000});
    if(!frame||frame->packed_yuyv.size()!=bytes||frame->packed_yuyv[bytes-1]!=173)throw std::runtime_error("copy/lease failure");
  };
  for(unsigned i=0;i<32;++i)one();
  copy_ns=0;copies=0; // producer is waiting on gate; no source payload access races
  track=true;auto start=Clock::now();
  for(unsigned i=0;i<256;++i)one();
  auto wall=std::chrono::duration<double,std::milli>(Clock::now()-start).count();track=false;
  {std::lock_guard lock(mutex);finished=true;}gate.notify_one();capture.stop();
  auto metrics=capture.metrics();
  std::cout<<"{\"allocations\":"<<allocations<<",\"frames\":"<<copies<<",\"bytes_per_frame\":"<<bytes
    <<",\"copy_mean_ms\":"<<double(copy_ns)/copies/1e6<<",\"copy_and_delivery_mean_ms\":"<<wall/copies
    <<",\"drops\":"<<metrics.queue_drops+metrics.no_slot_drops<<"}\n";
}
'''


@pytest.fixture(scope='module')
def capture_cpu_probe(tmp_path_factory):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable for CPU-only allocation probe')
    directory = tmp_path_factory.mktemp('native-capture-perf')
    source = directory / 'probe.cpp'
    source.write_text(ALLOCATION_PROBE)
    binary = directory / 'probe'
    # Compile the actual capture implementation with the camera macro absent:
    # this executable cannot construct/open a physical camera.
    built = subprocess.run([compiler, '-std=c++20', '-O3', '-pthread', '-I', str(ROOT / 'include'),
                            str(ROOT / 'src/capture.cpp'), str(source), '-o', str(binary)],
                           capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    results = {}
    for policy in ('drop_oldest', 'wait_for_consumer'):
        command = [str(binary)] + (['wait'] if policy == 'wait_for_consumer' else [])
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert completed.returncode == 0, completed.stderr
        results[policy] = json.loads(completed.stdout)
    return results


@pytest.mark.parametrize('policy', ['drop_oldest', 'wait_for_consumer'])
def test_warmed_720p_copy_and_handoff_has_no_heap_allocation_or_drops(capture_cpu_probe, policy):
    result = capture_cpu_probe[policy]
    print('native capture CPU-only 720p probe ' + policy + ':', json.dumps(result, sort_keys=True))
    assert result['frames'] == 256 and result['bytes_per_frame'] == 3_686_400
    assert result['allocations'] == 0, 'warmed frame copy/queue/lease path allocated on the heap'
    assert result['drops'] == 0
    assert 0 < result['copy_mean_ms'] <= result['copy_and_delivery_mean_ms']
