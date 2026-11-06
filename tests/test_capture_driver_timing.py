"""Real capture implementation, CPU-only timing retention/format contracts."""
from pathlib import Path
import json
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_driver_timing_roundtrip_legacy_and_pool_reuse(tmp_path):
    source = tmp_path/'probe.cpp'
    source.write_text(r'''
#include <stereoforge/capture.hpp>
#include <atomic>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace stereoforge;
using namespace std::chrono_literals;
void check(bool b){if(!b)throw std::runtime_error("timing contract failed");}
template<class F>void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}check(bad);}
CaptureFrame sample(){
 CaptureFrame f;f.source_width=8;f.height=2;f.stride_bytes=16;f.packed_yuyv.assign(32,7);
 f.frame_id=(uint64_t{1}<<53)+17;f.host_ingress_timestamp_ns=1016000000;
 f.camera_clock_domain="zed_open_capture_video_unvalidated";f.calibration_identity="fixture";
 f.driver_timing=CaptureDriverTiming{1000000000,UINT32_MAX,0x80012001,1014000000,1015000000,
                                  1793955000000000037ull,999000000};
 f.camera_timestamp_ns=f.driver_timing->sdk_wall_anchor_ns+1000000;return f;
}
int main(int argc,char**argv){try{
 auto dir=std::filesystem::path(argv[1]);auto f=sample();
 {PackedCaptureWriter w(dir);w.append(f);auto no=f;no.frame_id++;no.driver_timing.reset();w.append(no);w.close();}
 {std::ifstream file(dir/"capture.sfp",std::ios::binary);file.seekg(8);char v=0;file.read(&v,1);check(v==2);}
 auto rows=read_packed_capture(dir);check(rows.size()==2&&rows[0].frame_id==f.frame_id);
 check(rows[0].driver_timing==f.driver_timing&&!rows[1].driver_timing);
 check(rows[0].camera_timestamp_ns==f.camera_timestamp_ns&&rows[0].host_ingress_timestamp_ns==f.host_ingress_timestamp_ns);
 auto replay=packed_replay_source(dir);CaptureFrame reused=f;check(replay(reused,1ms)==CaptureSourceStatus::recorded_frame);
 check(reused.driver_timing==f.driver_timing);check(replay(reused,1ms)==CaptureSourceStatus::recorded_frame);check(!reused.driver_timing);
 auto legacy=dir.string()+"-v1";auto bare=f;bare.driver_timing.reset();
 {PackedCaptureWriter w(legacy,CaptureRecordingFormat::legacy_v1);rejects([&]{w.append(f);});w.append(bare);w.close();}
 auto old=packed_replay_source(legacy);reused=f;check(old(reused,1ms)==CaptureSourceStatus::recorded_frame);check(!reused.driver_timing);
 auto invalid=dir.string()+"-invalid";
 {PackedCaptureWriter w(invalid);auto bad=f;bad.driver_timing->dequeue_end_monotonic_ns=1;rejects([&]{w.append(bad);});
  bad=f;bad.camera_timestamp_ns++;rejects([&]{w.append(bad);});
  bad=f;bad.driver_timing->sdk_wall_anchor_ns=UINT64_MAX;bad.camera_timestamp_ns=1;rejects([&]{w.append(bad);});
  w.close();check(read_packed_capture(invalid).empty());}
 {std::fstream file(dir/"capture.sfp",std::ios::in|std::ios::out|std::ios::binary);file.seekg(-9,std::ios::end);char b;file.read(&b,1);b^=1;file.seekp(-9,std::ios::end);file.write(&b,1);}
 rejects([&]{read_packed_capture(dir);});
 CaptureConfig c;c.eye_width=4;c.height=2;c.queue_capacity=1;c.pool_capacity=3;c.queue_policy=CaptureQueuePolicy::wait_for_consumer;
 std::atomic<bool> stale=false;unsigned count=0;
 NativeCapture capture(c,[&](CaptureFrame& out,auto){if(count==40)return CaptureSourceStatus::end;
   if(out.driver_timing)stale=true;out=sample();out.frame_id=++count;return CaptureSourceStatus::frame;});
 for(unsigned i=0;i<40;++i){auto lease=capture.read(1000ms);check(bool(lease)&&lease->driver_timing.has_value());}
 capture.stop();check(!stale);
 std::cout<<"driver timing retained, legacy absence preserved, invalid timing rejected\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
''')
    binary = tmp_path/'probe'
    build = subprocess.run(['g++','-std=c++20','-O2','-pthread','-I'+str(ROOT/'include'),
                            str(ROOT/'src/capture.cpp'),str(source),'-o',str(binary)],
                           capture_output=True,text=True,timeout=60)
    assert build.returncode==0, build.stdout+build.stderr
    run = subprocess.run([str(binary),str(tmp_path/'recording')],capture_output=True,text=True,timeout=15)
    assert run.returncode==0,run.stdout+run.stderr


def test_optional_timing_report_preserves_absence_and_rejects_output_collision(tmp_path):
    binary=tmp_path/'capture'
    build=subprocess.run(['g++','-std=c++20','-O2','-pthread','-I'+str(ROOT/'include'),
        str(ROOT/'src/capture.cpp'),str(ROOT/'apps/native_capture_benchmark.cpp'),'-o',str(binary)],
        capture_output=True,text=True,timeout=60)
    assert build.returncode==0,build.stdout+build.stderr
    report=tmp_path/'report.json';timing=tmp_path/'timing.jsonl'
    command=[str(binary),'--source','synthetic','--frames','3','--output',str(report),'--timing-report',str(timing)]
    run=subprocess.run(command,capture_output=True,text=True,timeout=15)
    assert run.returncode==0,run.stdout+run.stderr
    rows=[json.loads(line) for line in timing.read_text().splitlines()]
    assert len(rows)==3 and all(row['driver_timing'] is None for row in rows)
    assert all(row['physical_timing_qualified'] is False and isinstance(row['camera_timestamp_ns'],str) for row in rows)
    collision=tmp_path/'collision.json'
    run=subprocess.run([str(binary),'--source','synthetic','--frames','3','--output',str(collision),
                        '--timing-report',str(collision)],capture_output=True,text=True,timeout=15)
    assert run.returncode!=0 and not collision.exists()


def test_shared_benchmark_metadata_retains_timing_and_opaque_imu_bits(tmp_path):
    cache=ROOT/'build/ros-daxa-native/CMakeCache.txt'
    paths=list((ROOT/'build').glob('*/_deps/sf_native_json-src/json.hpp'))
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith('FETCHCONTENT_SOURCE_DIR_SF_NATIVE_JSON:'):
                paths.append(Path(line.split('=',1)[1])/'json.hpp')
    header=next((path for path in paths if path.is_file()),None)
    if header is None:pytest.skip('native JSON dependency requires configured native build')
    source=tmp_path/'metadata.cpp';source.write_text(r'''
#include "capture_metadata.hpp"
#include <iostream>
int main(){stereoforge::CaptureFrame f;f.frame_id=(uint64_t{1}<<53)+17;
 stereoforge::CaptureImuSample sample;sample.acceleration_m_s2[0]=std::bit_cast<float>(0x7fc01234u);
 sample.angular_velocity_deg_s[1]=std::bit_cast<float>(0x80000000u);sample.temperature_c=std::bit_cast<float>(0x7f800000u);
 f.imu_samples={sample};f.driver_timing=stereoforge::CaptureDriverTiming{1000000000,UINT32_MAX,0x80012001,1014000000,1015000000,1793955000000000037ull,999000000};
 auto first=stereoforge::capture_metadata(f);f.driver_timing.reset();
 std::cout<<nlohmann::json{{"with",first},{"without",stereoforge::capture_metadata(f)}}.dump();}
''')
    binary=tmp_path/'metadata';r=subprocess.run(['g++','-std=c++20','-I'+str(ROOT/'include'),'-I'+str(ROOT/'apps'),
        '-I'+str(header.parent),str(source),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
    data=json.loads(subprocess.check_output([str(binary)],text=True))
    assert data['with']['frame_id']==2**53+17
    assert data['with']['driver_timing']['sdk_wall_anchor_ns']==1793955000000000037
    assert data['with']['driver_timing']['flags']==0x80012001
    assert data['with']['imu_samples'][0]['acceleration_bits'][0]==0x7fc01234
    assert data['with']['imu_samples'][0]['angular_velocity_bits'][1]==0x80000000
    assert data['with']['imu_samples'][0]['temperature_bits']==0x7f800000
    assert 'driver_timing' not in data['without']
