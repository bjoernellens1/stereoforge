#include <stereoforge/capture.hpp>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace stereoforge;
using namespace std::chrono_literals;
static void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> static void rejects(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,"expected rejection");}
static void conserved(const CaptureMetrics& m,bool live=true){
    check(m.source_records==m.delivered_frames+m.queue_drops+m.no_slot_drops+m.queued_frames+(live?m.repeated_frames:0),"stopped capture record accounting must conserve queued/delivered/dropped observations");
}
static CaptureConfig config(){CaptureConfig c;c.eye_width=4;c.height=2;c.queue_capacity=2;c.pool_capacity=4;c.calibration_identity="sha256:fixture-calibration";return c;}
static CaptureSource source(unsigned count,bool gaps=false){return [id=uint64_t{0},count,gaps](CaptureFrame& f,std::chrono::milliseconds) mutable {
    if(id==count)return CaptureSourceStatus::end;
    ++id;f.frame_id=id*(gaps?2:1);f.camera_timestamp_ns=id*100;f.camera_clock_domain="fixture_clock";
    f.packed_yuyv.assign(32,static_cast<uint8_t>(id));f.source_width=8;f.height=2;f.stride_bytes=16;
    f.controls_json="{\"auto_exposure_gain\":true}";f.sensor_status="raw_only";
    f.imu_samples={{id*100,{1,2,3},{4,5,6},23.5f,false}};return CaptureSourceStatus::frame;
};}
static std::string shell_quote(const std::string& value){std::string s="'";for(char c:value)s+=c=='\''?"'\\''":std::string(1,c);return s+"'";}
int main(int argc,char** argv){try {
    if(argc>1){
      auto dir=std::filesystem::temp_directory_path()/("stereoforge-cli-input-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));std::filesystem::create_directories(dir);
      auto input=dir/"capture.sfp";const std::string sentinel="preserve-input-bytes";{std::ofstream file(input);file<<sentinel;}
      auto run=[&](const std::filesystem::path& output){auto cmd=shell_quote(argv[1])+" --frames 1 --replay "+shell_quote(dir.string())+" --output "+shell_quote(output.string())+" >/dev/null 2>&1";check(std::system(cmd.c_str())!=0,"colliding output must fail closed");std::ifstream file(input);std::string bytes((std::istreambuf_iterator<char>(file)),{});check(bytes==sentinel,"benchmark must preserve replay input bytes");};
      run(input);run(dir/"."/"capture.sfp");auto link=dir/"output-link";std::filesystem::create_symlink(input,link);run(link);std::filesystem::remove_all(dir);
    }
    rejects([]{auto c=config();c.queue_capacity=0;NativeCapture bad(c,source(1));});
    rejects([]{auto c=config();c.pool_capacity=c.queue_capacity;NativeCapture bad(c,source(1));});
    check(CaptureFrame{}.source_kind==CaptureSourceKind::PackedYuyv&&!CaptureFrame{}.canonical_rgb,"default source is packed YUYV");
    CaptureLease held;
    {NativeCapture cap(config(),source(20,true));for(int i=0;i<2000&&cap.metrics().running;++i)std::this_thread::sleep_for(1ms);check(!cap.metrics().running,"source must end");
     auto m=cap.metrics();check(m.source_frames==20,"source count");check(m.queue_drops==18,"bounded queue drops");check(m.source_frame_gaps==19,"gaps");
     held=cap.read(10ms);check(held&&held->frame_id==38,"latest queue retains ordering");check(held->packed_yuyv[0]==19,"packed source bytes");
     check(held->source_width==8&&held->stride_bytes==16,"full source width");held->require_calibration();cap.stop();m=cap.metrics();check(m.queued_frames==1,"one unread frame remains queued after stop");conserved(m);}
    check(held->packed_yuyv[0]==19,"lease survives facade destruction");
    auto dir=std::filesystem::temp_directory_path()/("stereoforge-native-capture-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));std::filesystem::remove_all(dir);
    {auto tagged=*held;tagged.source_kind=CaptureSourceKind::CanonicalRgbMetadata;PackedCaptureWriter writer(dir);rejects([&]{writer.append(tagged);});writer.close();check(std::filesystem::file_size(dir/"capture.sfp")==20,"canonical source rejected before writing record bytes");check(read_packed_capture(dir).empty(),"canonical source does not increment raw record count");std::filesystem::remove_all(dir);}
    {auto raw=*held;raw.host_ingress_timestamp_ns=0;raw.host_clock_domain="fixture_unknown_host";PackedCaptureWriter writer(dir);writer.append(raw);writer.append(raw);writer.close();}
    rejects([&]{read_packed_capture(dir,1);});rejects([&]{read_packed_capture(dir,4,1);});
    auto frames=read_packed_capture(dir);check(frames.size()==2,"record count");
    {auto replay=packed_replay_source(dir);CaptureFrame tagged;tagged.source_kind=CaptureSourceKind::CanonicalRgbMetadata;tagged.frame_id=777;rejects([&]{replay(tagged,1ms);});check(tagged.source_kind==CaptureSourceKind::CanonicalRgbMetadata&&tagged.frame_id==777,"raw replay rejects canonical destination before mutation");auto raw_replay=packed_replay_source(dir);CaptureFrame raw;check(raw_replay(raw,1ms)==CaptureSourceStatus::recorded_frame,"raw v1 replay record");check(raw.source_kind==CaptureSourceKind::PackedYuyv&&!raw.canonical_rgb&&raw.packed_yuyv==held->packed_yuyv,"default raw v1 replay preserves source and bytes");}
    check(frames[0].packed_yuyv==held->packed_yuyv&&frames[0].controls_json==held->controls_json,"lossless packed and controls");
    check(frames[0].imu_samples[0].angular_velocity_deg_s[2]==6,"lossless IMU");
    {auto c=config();c.queue_policy=CaptureQueuePolicy::wait_for_consumer;NativeCapture cap(c,packed_replay_source(dir));auto a=cap.read(100ms);auto b=cap.read(100ms);check(a&&b,"replay preserves duplicate records");check(a->frame_id==b->frame_id,"replay record identity");check(a->host_ingress_timestamp_ns==0&&a->host_clock_domain=="fixture_unknown_host","replay preserves zero ingress and original domain");cap.stop();conserved(cap.metrics(),false);}
    {auto longer=dir/"long-replay";{PackedCaptureWriter writer(longer);for(unsigned i=1;i<=20;++i){auto frame=*held;frame.frame_id=i;frame.camera_timestamp_ns=i*100;frame.packed_yuyv[0]=static_cast<uint8_t>(i);writer.append(frame);}writer.close();}
      auto c=config();c.queue_policy=CaptureQueuePolicy::wait_for_consumer;NativeCapture cap(c,packed_replay_source(longer));
      for(int i=0;i<500&&cap.metrics().queue_high_watermark<c.queue_capacity;++i)std::this_thread::sleep_for(1ms);
      check(cap.metrics().queue_high_watermark==c.queue_capacity&&cap.metrics().running,"replay waits at full queue during delayed consumer setup");
      for(unsigned i=1;i<=20;++i){auto f=cap.read(500ms);check(f&&f->frame_id==i&&f->packed_yuyv[0]==i,"delayed replay preserves every recorded frame in order");}
      cap.stop();auto m=cap.metrics();check(m.queue_drops==0&&m.no_slot_drops==0&&m.delivered_frames==20,"delayed replay has no capture drops");conserved(m,false);std::filesystem::remove_all(longer);}
    {auto path=dir/"capture.sfp";std::fstream file(path,std::ios::in|std::ios::out|std::ios::binary);file.seekp(-1,std::ios::end);char b=0;file.write(&b,1);}rejects([&]{read_packed_capture(dir);});
    std::filesystem::remove_all(dir);
    {auto bad=source(1);NativeCapture cap(config(),[bad](CaptureFrame& f,auto t)mutable{auto s=bad(f,t);f.packed_yuyv.clear();return s;});rejects([&]{cap.read(100ms);});}
    {CaptureFrame f;rejects([&]{f.require_calibration();});}
    {NativeCapture cap(config(),[](CaptureFrame&,auto){throw std::runtime_error("source failed");return CaptureSourceStatus::end;});rejects([&]{cap.read(100ms);});}
    {auto c=config();c.pool_capacity=3;c.queue_capacity=1;std::atomic<unsigned> step=0;
      NativeCapture cap(c,[&](CaptureFrame& f,auto){while(step.load()==0)std::this_thread::sleep_for(1ms);auto s=source(1);s(f,1ms);f.frame_id=step.fetch_add(1);f.camera_timestamp_ns=f.frame_id*100;std::this_thread::sleep_for(1ms);return CaptureSourceStatus::frame;});
      step=1;auto a=cap.read(100ms);auto b=cap.read(100ms);auto d=cap.read(100ms);check(a&&b&&d,"pool leases");auto before=a->packed_yuyv;std::this_thread::sleep_for(10ms);cap.stop();check(cap.metrics().no_slot_drops>0,"held leases backpressure counted");check(a->packed_yuyv==before,"held storage not overwritten");conserved(cap.metrics());}
    {std::atomic<bool> entered=false;NativeCapture cap(config(),[&](CaptureFrame&,auto timeout){entered=true;std::this_thread::sleep_for(timeout);return CaptureSourceStatus::timeout;});for(int i=0;i<500&&!entered;++i)std::this_thread::sleep_for(1ms);check(entered,"source entered blocking timeout");auto start=std::chrono::steady_clock::now();cap.stop();check(std::chrono::steady_clock::now()-start<500ms,"bounded source shutdown");}
    {auto path=dir/"capture.sfp";std::filesystem::create_directories(dir);{std::ofstream f(path,std::ios::binary);f<<"truncated";}rejects([&]{read_packed_capture(dir);});std::filesystem::remove_all(dir);}
    {PackedCaptureWriter w(dir);w.append(*held);rejects([&]{read_packed_capture(dir);});w.close();rejects([&]{w.append(*held);});std::filesystem::remove_all(dir);}
    {auto c=config();c.calibration_identity="other";auto s=source(1);NativeCapture cap(c,[s](CaptureFrame& f,auto t) mutable {auto status=s(f,t);f.calibration_identity="recorded";return status;});rejects([&]{cap.read(100ms);});}
    {auto c=config();c.queue_capacity=4;c.pool_capacity=6;auto s=source(3,true);NativeCapture cap(c,[s](CaptureFrame& f,auto t)mutable{auto status=s(f,t);f.camera_clock_domain=std::to_string(f.frame_id);return status;});for(int i=0;i<2000&&cap.metrics().running;++i)std::this_thread::sleep_for(1ms);check(cap.metrics().source_frame_gaps==0,"clock domain transitions do not invent frame gaps");check(cap.read(10ms)->camera_clock_domain=="2","clock metadata retained");}
    {auto c=config();c.queue_policy=CaptureQueuePolicy::wait_for_consumer;NativeCapture cap(c,source(20));for(unsigned i=1;i<=20;++i){auto f=cap.read(100ms);check(f&&f->frame_id==i,"lossless replay policy preserves every ordered frame");std::this_thread::sleep_for(1ms);}cap.stop();check(cap.metrics().queue_drops==0&&cap.metrics().no_slot_drops==0,"lossless policy backpressures instead of dropping");}
    {std::atomic<bool> second_entered=false,release_second=false;auto s=source(2);
      NativeCapture cap(config(),[&](CaptureFrame& f,auto timeout){auto status=s(f,timeout);if(status==CaptureSourceStatus::frame&&f.frame_id==2){second_entered=true;auto deadline=std::chrono::steady_clock::now()+timeout;while(!release_second&&std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(1ms);}return status;});
      auto first=cap.read(100ms);check(first&&first->frame_id==1,"first observation delivered before stop");for(int i=0;i<500&&!second_entered;++i)std::this_thread::sleep_for(1ms);check(second_entered,"second observation in source callback");
      std::thread stopper([&]{cap.stop();});release_second=true;stopper.join();auto m=cap.metrics();check(!m.running&&m.source_records==2&&m.delivered_frames==1&&m.queued_frames==1,"source callback finishing at stop remains explicitly queued");conserved(m);
    }
    {unsigned count=0;NativeCapture cap(config(),[&](CaptureFrame& f,auto timeout){if(count++==2)return CaptureSourceStatus::end;auto one=source(1);return one(f,timeout);});for(int i=0;i<500&&cap.metrics().running;++i)std::this_thread::sleep_for(1ms);cap.stop();auto m=cap.metrics();check(m.repeated_frames==1&&m.source_frames==1,"live repeat stays separate from admitted observations");conserved(m);}
    std::cout<<"native capture CPU contracts passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
