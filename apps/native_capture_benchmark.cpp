#include <stereoforge/capture.hpp>
#include <algorithm>
#include <thread>
#include <map>
#include <tuple>
#include <charconv>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
using namespace stereoforge;
static uint64_t number(const std::string& s){uint64_t n=0;auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),n);if(e!=std::errc{}||p!=s.data()+s.size())throw std::invalid_argument("invalid positive integer: "+s);return n;}
static std::string json_string(const std::string& s){std::string out="\"";for(unsigned char c:s){if(c=='\"'||c=='\\'){out+='\\';out+=char(c);}else if(c<32)throw std::invalid_argument("invalid report string");else out+=char(c);}return out+'\"';}
int main(int argc,char** argv){try{
    CaptureConfig config;uint64_t count=60;std::string output,record_dir,replay,source_kind="camera";
    for(int i=1;i<argc;++i){std::string key=argv[i];if(key=="--help"){std::cout<<"native_capture_benchmark [--frames N] --output FILE [--record-dir DIR] [--replay DIR] [--source camera|synthetic] [--calibration-id ID] [--device-serial ID] [--device-id N]\n";return 0;}if(++i==argc)throw std::invalid_argument("missing option value");std::string value=argv[i];if(key=="--frames")count=number(value);else if(key=="--output")output=value;else if(key=="--record-dir")record_dir=value;else if(key=="--replay")replay=value;else if(key=="--source")source_kind=value;else if(key=="--calibration-id")config.calibration_identity=value;else if(key=="--device-serial")config.device_serial_override=value;else if(key=="--device-id"){auto n=number(value);if(n>INT32_MAX)throw std::invalid_argument("device-id too large");config.device_id=int(n);}else throw std::invalid_argument("unknown option: "+key);}
    if(!count||count>1000000||output.empty())throw std::invalid_argument("positive bounded --frames and --output are required");
    if(source_kind!="camera"&&source_kind!="synthetic")throw std::invalid_argument("unknown source");
    if(!replay.empty())config.queue_policy=CaptureQueuePolicy::wait_for_consumer;
    if(source_kind=="synthetic"&&!replay.empty())throw std::invalid_argument("synthetic and replay conflict");
    // Require a fresh report destination before any source or recording writes.
    // symlink_status also rejects dangling symlinks and path aliases of replay input.
    if(std::filesystem::symlink_status(output).type()!=std::filesystem::file_type::not_found)
        throw std::invalid_argument("output must be a new file (existing paths and symlinks are refused)");
    auto initialization_start=std::chrono::steady_clock::now();
    // Validate report destination before opening a physical device.
    std::ofstream report(output);if(!report)throw std::runtime_error("cannot create report");
    std::unique_ptr<PackedCaptureWriter> recorder;if(!record_dir.empty())recorder=std::make_unique<PackedCaptureWriter>(record_dir);
    std::unique_ptr<NativeCapture> capture;
    uint64_t copy_ns=0,copy_count=0;
    if(source_kind=="synthetic"){
      auto pixels=std::make_shared<const std::vector<uint8_t>>(size_t(config.eye_width)*4*config.height,128);
      capture=std::make_unique<NativeCapture>(config,[&,pixels,id=uint64_t{0},next=std::chrono::steady_clock::now()](CaptureFrame& f,auto)mutable{
        std::this_thread::sleep_until(next);next+=std::chrono::nanoseconds{1000000000/config.fps};
        auto start=std::chrono::steady_clock::now();f.packed_yuyv.resize(pixels->size());std::copy(pixels->begin(),pixels->end(),f.packed_yuyv.begin());
        copy_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();++copy_count;
        f.source_width=config.eye_width*2;f.height=config.height;f.stride_bytes=config.eye_width*4;f.frame_id=++id;f.camera_timestamp_ns=id*(1000000000/config.fps);f.camera_clock_domain="synthetic_sequence";f.device_identity="synthetic_cpu_fixture";return CaptureSourceStatus::frame;
      });
    }else if(replay.empty())capture=std::make_unique<NativeCapture>(config);else capture=std::make_unique<NativeCapture>(config,packed_replay_source(replay));
    double initialization_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-initialization_start).count();
    uint64_t received=0,first_camera=0,last_camera=0,first_host=0,last_host=0;std::vector<double> ages;ages.reserve(count);
    std::string device,clock,sensor,calibration,host_clock;uint64_t imu_samples=0,sensor_drops=0;bool changed_clock=false,changed_host_clock=false,camera_monotonic=true,host_monotonic=true;
    std::map<std::string,size_t> clock_ids;std::vector<std::tuple<uint64_t,uint64_t,size_t>> identities;identities.reserve(count);
    auto started=std::chrono::steady_clock::now();
    while(received<count){auto frame=capture->read(std::chrono::milliseconds{2000});if(!frame){if(!capture->metrics().running)break;throw std::runtime_error("no fresh camera frame in 2 seconds");}
      if(!received){first_camera=frame->camera_timestamp_ns;first_host=frame->host_ingress_timestamp_ns;device=frame->device_identity;clock=frame->camera_clock_domain;sensor=frame->sensor_status;calibration=frame->calibration_identity;host_clock=frame->host_clock_domain;}else{
        if(clock!=frame->camera_clock_domain)changed_clock=true;
        if(host_clock!=frame->host_clock_domain)changed_host_clock=true;
        if(frame->camera_timestamp_ns<=last_camera)camera_monotonic=false;
        if(frame->host_ingress_timestamp_ns<=last_host)host_monotonic=false;
      }
      last_camera=frame->camera_timestamp_ns;last_host=frame->host_ingress_timestamp_ns;imu_samples+=frame->imu_samples.size();sensor_drops=frame->sensor_dropped_count;
      if(replay.empty()&&frame->host_clock_domain=="host_steady"){auto now=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();ages.push_back(double(now-frame->host_ingress_timestamp_ns)/1e6);}
      auto [clock_id,inserted]=clock_ids.try_emplace(frame->camera_clock_domain,clock_ids.size());
      identities.emplace_back(frame->frame_id,frame->camera_timestamp_ns,clock_id->second);
      if(recorder)recorder->append(*frame);
      ++received;
    }
    auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();capture->stop();auto m=capture->metrics();if(recorder)recorder->close();std::sort(ages.begin(),ages.end());std::sort(identities.begin(),identities.end());auto unique_count=std::unique(identities.begin(),identities.end())-identities.begin();
    report<<"{\n  \"schema\":\"stereoforge.native_capture.v1\",\n  \"source\":"<<json_string(source_kind=="synthetic"?"synthetic_cpu_fixture":replay.empty()?"physical_camera":"packed_replay")<<",\n  \"device\":"<<json_string(device)<<",\n  \"eye_width\":"<<config.eye_width<<",\n  \"height\":"<<config.height<<",\n  \"requested_fps\":"<<config.fps<<",\n  \"calibration_identity\":"<<json_string(calibration)<<",\n  \"received_unique_frames\":"<<unique_count<<",\n  \"delivered_records\":"<<received<<",\n  \"source_records\":"<<m.source_records<<",\n  \"source_fresh_observations\":"<<m.source_frames<<",\n  \"queue_drops\":"<<m.queue_drops<<",\n  \"no_slot_drops\":"<<m.no_slot_drops<<",\n  \"queued_frames_at_stop\":"<<m.queued_frames<<",\n  \"backpressure_waits\":"<<m.backpressure_waits<<",\n  \"backpressure_wait_ns\":"<<m.backpressure_wait_ns<<",\n  \"source_frame_gaps\":"<<m.source_frame_gaps<<",\n  \"repeated_frames\":"<<m.repeated_frames<<",\n  \"source_timeouts\":"<<m.source_timeouts<<",\n  \"invalid_frames\":"<<m.invalid_frames<<",\n  \"raw_imu_samples\":"<<imu_samples<<",\n  \"sensor_dropped_count\":"<<sensor_drops<<",\n  \"pool_capacity\":"<<m.pool_capacity<<",\n  \"packed_bytes_per_slot\":"<<m.packed_bytes_per_slot<<",\n  \"queue_high_watermark\":"<<m.queue_high_watermark<<",\n  \"wall_seconds\":"<<elapsed<<",\n  \"initialization_seconds\":"<<initialization_seconds<<",\n  \"synthetic_copy_samples\":"<<copy_count<<",\n  \"synthetic_copy_mean_ms\":"<<(copy_count?double(copy_ns)/copy_count/1e6:0)<<",\n  \"completed_fps\":"<<received/elapsed<<",\n  \"camera_clock_domain\":"<<json_string(clock)<<",\n  \"camera_clock_changed\":"<<(changed_clock?"true":"false")<<",\n  \"first_camera_timestamp_ns\":"<<first_camera<<",\n  \"last_camera_timestamp_ns\":"<<last_camera<<",\n  \"first_host_ingress_timestamp_ns\":"<<first_host<<",\n  \"last_host_ingress_timestamp_ns\":"<<last_host<<",\n  \"sensor_status\":"<<json_string(sensor)<<",\n  \"imu_pose_validated\":false,\n  \"camera_to_host_age_ms\":null,\n  \"host_ingress_to_delivery_p50_ms\":";
    if(ages.empty())report<<"null";else report<<ages[ages.size()/2];report<<",\n  \"host_ingress_to_delivery_p95_ms\":";if(ages.empty())report<<"null";else report<<ages[std::min(ages.size()-1,ages.size()*95/100)];report<<",\n  \"camera_interval_fps\":";
    if(received>1&&!changed_clock&&camera_monotonic&&last_camera>first_camera)report<<double(received-1)*1e9/(last_camera-first_camera);else report<<"null";
    report<<",\n  \"host_ingress_interval_fps\":";
    if(received>1&&!changed_host_clock&&host_monotonic&&last_host>first_host)report<<double(received-1)*1e9/(last_host-first_host);else report<<"null";
    report<<",\n  \"host_clock_domain\":"<<json_string(host_clock)<<",\n  \"host_clock_changed\":"<<(changed_host_clock?"true":"false")<<"\n}\n";
    if(!report)throw std::runtime_error("report write failed");
    if(received!=count)throw std::runtime_error("source ended before requested frame count");
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
