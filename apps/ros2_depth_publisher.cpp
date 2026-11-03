// Native capture -> Daxa -> selective host export -> ROS. No Python/SDK path.
#include "stereo_estimator_metadata.hpp"
#include <stereoforge/capture.hpp>
#include <stereoforge/depth_rate_controller.hpp>
#include <stereoforge/depth_speckle.hpp>
#include <stereoforge/daxa_depth.hpp>
#include <stereoforge/daxa_dual_preprocess.hpp>
#include <stereoforge/daxa_refinement.hpp>
#include <stereoforge/daxa_host_export.hpp>
#include <stereoforge/ros_delivery_json.hpp>
#include <stereoforge/ros_depth_contract.hpp>
#include <stereoforge/ros_default_rmw.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/rmw.h>
#include <sensor_msgs/msg/image.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace stereoforge;
using nlohmann::json;
namespace {
using steady=std::chrono::steady_clock;
template<class Clock> std::uint64_t now_ns(){auto n=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();if(n<=0)throw std::runtime_error("clock unavailable");return static_cast<std::uint64_t>(n);}
// CLOCK_THREAD_CPUTIME_ID samples are opt-in: four clock reads per output.
// Sampling can perturb short operations; these durations are not off-CPU causes.
bool thread_cpu_now(std::uint64_t& value) noexcept {
    timespec t{};
    if(::clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)!=0 || t.tv_sec<0 || t.tv_nsec<0 || t.tv_nsec>=1'000'000'000 ||
       static_cast<std::uint64_t>(t.tv_sec)>static_cast<std::uint64_t>(INT64_MAX)/1'000'000'000u) return false;
    value=static_cast<std::uint64_t>(t.tv_sec)*1'000'000'000u+static_cast<std::uint64_t>(t.tv_nsec);
    return value<=static_cast<std::uint64_t>(INT64_MAX);
}
struct Options {
    std::string source,manifest,replay,mode="depth",map,frame,run,trace,reliability="best_effort",stereo_variant="packed-specialized",device_serial;
    std::string pipeline_mode="full-sgm",coarse_manifest,descriptor_mode="census",coarse_schedule="sequential",coarse_right_mode="independent-sgm",profile="quality";
    std::string depth_topic="/stereoforge/depth",rgb_topic="/stereoforge/rectified_rgb",completed_depth_topic="/stereoforge/depth_completed",completed_provenance_topic="/stereoforge/depth_completed_provenance";
    std::string depth_rate="adaptive",rate_report;
    unsigned depth_speckle_min_pixels=0; double depth_speckle_delta_m=0.05; double depth_speckle_delta_px=1.25; double gpu_duty_budget_pct=0.0;
    unsigned min_depth_fps=5,max_depth_fps=60,consumer_min_depth_fps=5,max_depth_age_ms=250,downshift_ms=3000;
    unsigned fps=60,disparities=128,warmup=5,max_frames=100000,qos_depth=1;int device_id=-1;double duration=600,subscriber_wait=0;bool validation=false,thread_cpu_timing=false,fast_mode=false,ultra_fast=false,refinement_subpixel=false,selective_repair=false,planar_recovery=false,verify_payload_checksum=false;
};
unsigned number(const std::string& s,unsigned lo,unsigned hi){size_t n=0;auto v=std::stoull(s,&n);if(n!=s.size()||v<lo||v>hi)throw std::invalid_argument("integer option outside bounds");return static_cast<unsigned>(v);}
DepthRateConfig depth_rate_config(const Options& o) {
    DepthRateConfig c;
    c.mode=o.depth_rate=="adaptive"?DepthRateMode::Adaptive:DepthRateMode::Fixed;
    if(c.mode==DepthRateMode::Fixed) {
        c.fixed_fps=number(o.depth_rate,5,60);
        if(!supported_depth_rate(c.fixed_fps))throw std::invalid_argument("unsupported fixed depth rate");
    }
    c.capture_fps=o.fps;c.min_fps=o.min_depth_fps;c.max_fps=o.max_depth_fps;
    c.consumer_min_fps=o.consumer_min_depth_fps;
    c.max_depth_age_ns=std::uint64_t(o.max_depth_age_ms)*1'000'000ull;
    c.downshift_dwell_ns=std::uint64_t(o.downshift_ms)*1'000'000ull;
    c.validate();
    return c;
}
Options parse(int argc,char** argv){
    Options o;
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--help"){std::cout<<"--source camera|replay --manifest FILE --clock-map FILE --frame-id ID --run-id ID --trace FILE [--replay DIR --mode depth|rgb|depth-rgb --depth-rate 5|10|20|30|60|adaptive --min-depth-fps N --max-depth-fps N --consumer-min-depth-fps N --max-depth-age-ms N --downshift-ms N --rate-report FILE --capture-fps 15|30|60 --duration-seconds N --subscriber-wait-seconds N --max-frames N --warmup N --disparities N --stereo-variant packed-specialized|packed-specialized-overwrite|packed-specialized-overwrite-quad --pipeline full-sgm|experimental-refinement|experimental-dual-refinement --coarse-manifest FILE --device-id N --depth-topic TOPIC --rgb-topic TOPIC --reliability best_effort|reliable --depth-speckle-min-pixels N --depth-speckle-delta-m METRES --depth-speckle-delta-px PIXELS --validation --thread-cpu-timing]\n";std::exit(0);}
        if(a=="--validation"){o.validation=true;continue;}if(a=="--thread-cpu-timing"){o.thread_cpu_timing=true;continue;}if(a=="--verify-payload-checksum"){o.verify_payload_checksum=true;continue;}if(i+1>=argc)throw std::invalid_argument("missing option value");std::string v=argv[++i];
        if(a=="--source")o.source=v;else if(a=="--manifest")o.manifest=v;else if(a=="--replay")o.replay=v;else if(a=="--mode")o.mode=v;
        else if(a=="--clock-map")o.map=v;else if(a=="--frame-id")o.frame=v;else if(a=="--run-id")o.run=v;else if(a=="--trace")o.trace=v;
        else if(a=="--depth-topic")o.depth_topic=v;else if(a=="--rgb-topic")o.rgb_topic=v;else if(a=="--completed-depth-topic")o.completed_depth_topic=v;else if(a=="--completed-provenance-topic")o.completed_provenance_topic=v;else if(a=="--reliability")o.reliability=v;
        else if(a=="--qos-depth")o.qos_depth=number(v,1,64);
        else if(a=="--depth-rate")o.depth_rate=v;else if(a=="--rate-report")o.rate_report=v;
        else if(a=="--depth-speckle-min-pixels")o.depth_speckle_min_pixels=number(v,0,10000);
        else if(a=="--depth-speckle-delta-m"){size_t n=0;o.depth_speckle_delta_m=std::stod(v,&n);if(n!=v.size()||!std::isfinite(o.depth_speckle_delta_m)||o.depth_speckle_delta_m<0||o.depth_speckle_delta_m>10)throw std::invalid_argument("invalid depth speckle delta");}
        else if(a=="--depth-speckle-delta-px"){size_t n=0;o.depth_speckle_delta_px=std::stod(v,&n);if(n!=v.size()||!std::isfinite(o.depth_speckle_delta_px)||o.depth_speckle_delta_px<0||o.depth_speckle_delta_px>32)throw std::invalid_argument("invalid depth speckle disparity delta");}
        else if(a=="--min-depth-fps")o.min_depth_fps=number(v,5,60);
        else if(a=="--max-depth-fps")o.max_depth_fps=number(v,5,60);
        else if(a=="--consumer-min-depth-fps")o.consumer_min_depth_fps=number(v,5,60);
        else if(a=="--max-depth-age-ms")o.max_depth_age_ms=number(v,1,10'000);
        else if(a=="--downshift-ms")o.downshift_ms=number(v,1,60'000);
        else if(a=="--stereo-variant")o.stereo_variant=v;else if(a=="--pipeline")o.pipeline_mode=v;else if(a=="--coarse-manifest")o.coarse_manifest=v;
        else if(a=="--profile"){if(v!="quality"&&v!="exact-ultra"&&v!="ultra-fast"&&v!="v11-fast"&&v!="mobile-balanced")throw std::invalid_argument("invalid profile");o.profile=v;
            if(v=="exact-ultra"){o.descriptor_mode="ortho-hybrid";o.coarse_schedule="shared-census-sequential";o.coarse_right_mode="independent-sgm";o.fast_mode=true;o.ultra_fast=false;o.refinement_subpixel=true;o.selective_repair=true;}
            else if(v=="ultra-fast"){o.descriptor_mode="ortho-hybrid";o.coarse_schedule="shared-census-sequential";o.coarse_right_mode="left-aggregated-selection";o.fast_mode=true;o.ultra_fast=true;o.refinement_subpixel=true;o.selective_repair=true;}
            else if(v=="v11-fast"||v=="mobile-balanced"){o.descriptor_mode="ortho-hybrid";o.coarse_schedule="hierarchical-1to8-fallback";o.coarse_right_mode="independent-sgm";o.fast_mode=true;o.ultra_fast=true;o.refinement_subpixel=true;o.selective_repair=true;}}
        else if(a=="--gpu-duty-budget-pct"){size_t n=0;o.gpu_duty_budget_pct=std::stod(v,&n);if(n!=v.size()||!std::isfinite(o.gpu_duty_budget_pct)||o.gpu_duty_budget_pct<0.0||o.gpu_duty_budget_pct>100.0)throw std::invalid_argument("invalid gpu-duty-budget-pct");}
        else if(a=="--descriptor-mode"){if(v!="census"&&v!="ortho-hybrid")throw std::invalid_argument("invalid descriptor mode");o.descriptor_mode=v;}
        else if(a=="--fast-mode")o.fast_mode=(number(v,0,1)==1);
        else if(a=="--ultra-fast")o.ultra_fast=(number(v,0,1)==1);
        else if(a=="--coarse-schedule"){if(v!="sequential"&&v!="paired-per-path"&&v!="layered-per-path"&&v!="shared-census-sequential"&&v!="hierarchical-1to8-fallback")throw std::invalid_argument("invalid coarse schedule");o.coarse_schedule=v;}
        else if(a=="--coarse-right-reference-mode"){if(v!="independent-sgm"&&v!="left-aggregated-selection")throw std::invalid_argument("invalid coarse right reference mode");o.coarse_right_mode=v;}
        else if(a=="--refinement-subpixel")o.refinement_subpixel=(number(v,0,1)==1);
        else if(a=="--selective-repair")o.selective_repair=(number(v,0,1)==1);
        else if(a=="--planar-recovery")o.planar_recovery=(number(v,0,1)==1);
        else if(a=="--fps"||a=="--capture-fps"){o.fps=number(v,15,60);if(o.fps!=15&&o.fps!=30&&o.fps!=60)throw std::invalid_argument("unsupported camera FPS");}else if(a=="--disparities")o.disparities=number(v,32,256);
        else if(a=="--max-frames")o.max_frames=number(v,1,200000);else if(a=="--warmup")o.warmup=number(v,0,30);
        else if(a=="--device-id"){if(v=="-1")o.device_id=-1;else o.device_id=static_cast<int>(number(v,0,255));}
        else if(a=="--device-serial")o.device_serial=v;
        else if(a=="--duration-seconds"){size_t n=0;o.duration=std::stod(v,&n);if(n!=v.size()||!std::isfinite(o.duration)||o.duration<=0||o.duration>3600)throw std::invalid_argument("invalid duration");}
        else if(a=="--subscriber-wait-seconds"){size_t n=0;o.subscriber_wait=std::stod(v,&n);if(n!=v.size()||!std::isfinite(o.subscriber_wait)||o.subscriber_wait<0||o.subscriber_wait>60)throw std::invalid_argument("invalid subscriber discovery timeout");}
        else throw std::invalid_argument("unknown option: "+a);
    }
    auto blank=[](const std::string& s){return s.empty()||std::all_of(s.begin(),s.end(),[](unsigned char c){return std::isspace(c)!=0;});};
    if((o.source!="camera"&&o.source!="replay")||(o.source=="replay")!=!o.replay.empty()||o.manifest.empty()||o.map.empty()||o.trace.empty()||
       blank(o.frame)||blank(o.run)||o.frame.size()>4096||o.run.size()>4096||
       (o.mode!="depth"&&o.mode!="rgb"&&o.mode!="depth-rgb")||(o.pipeline_mode!="full-sgm"&&o.pipeline_mode!="experimental-refinement"&&o.pipeline_mode!="experimental-dual-refinement")||
       ((o.pipeline_mode!="full-sgm")!=!o.coarse_manifest.empty())||(o.pipeline_mode!="full-sgm"&&(o.mode=="rgb"||o.disparities%2!=0))||(o.stereo_variant!="packed-specialized"&&o.stereo_variant!="packed-specialized-overwrite"&&o.stereo_variant!="packed-specialized-overwrite-quad")||(o.reliability!="best_effort"&&o.reliability!="reliable")||
       o.depth_topic.empty()||o.rgb_topic.empty()||o.completed_depth_topic.empty()||o.completed_provenance_topic.empty()||o.depth_topic.front()!='/'||o.rgb_topic.front()!='/'||o.completed_depth_topic.front()!='/'||o.completed_provenance_topic.front()!='/'||
       o.completed_depth_topic==o.depth_topic||o.completed_depth_topic==o.rgb_topic||o.completed_provenance_topic==o.depth_topic||o.completed_provenance_topic==o.rgb_topic||o.completed_provenance_topic==o.completed_depth_topic||(o.mode=="depth-rgb"&&o.depth_topic==o.rgb_topic))throw std::invalid_argument("invalid publisher configuration");
    if(!o.rate_report.empty()&&o.rate_report==o.trace)throw std::invalid_argument("rate report must differ from ROS delivery trace");
    depth_rate_config(o);
    return o;
}
std::string clock_id(){std::ifstream f("/proc/sys/kernel/random/boot_id");std::string s;std::getline(f,s);if(!f||s.size()!=36)throw std::runtime_error("boot identity unavailable");
    for(size_t i=0;i<s.size();++i){bool h=i==8||i==13||i==18||i==23;if(h?s[i]!='-':!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))throw std::runtime_error("invalid boot identity");}
    struct stat time_namespace{};if(::stat("/proc/self/ns/time",&time_namespace)!=0||!time_namespace.st_ino)throw std::runtime_error("time namespace identity unavailable");
    return s+":monotonic:"+std::to_string(static_cast<std::uint64_t>(time_namespace.st_ino));}
struct OutputFile {
    int fd=-1; bool complete=false;
    explicit OutputFile(const std::string& path) {
        fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
        if(fd<0)throw std::runtime_error(std::string("cannot create trace: ")+std::strerror(errno));
        try { failure("setup interrupted"); } catch (...) { ::close(fd);fd=-1;throw; }
    }
    ~OutputFile(){if(fd>=0)::close(fd);}
    void put(const std::string& s){size_t at=0;while(at<s.size()){auto n=::write(fd,s.data()+at,s.size()-at);if(n<0&&errno==EINTR)continue;if(n<=0)throw std::runtime_error("trace write failed");at+=static_cast<size_t>(n);}}
    void reset(){if(::ftruncate(fd,0)!=0||::lseek(fd,0,SEEK_SET)<0)throw std::runtime_error("trace reset failed");}
    void failure(const char* message){
        if(complete)return;
        std::string safe;for(size_t i=0;message[i]&&i<512;++i){unsigned char c=message[i];safe.push_back(c>=32&&c<=126?char(c):'?');}
        reset();put(json({{"schema","stereoforge.ros_delivery_failure.v1"},{"role","publisher"},
                          {"kind","startup_or_output_failure"},{"message",safe}}).dump()+"\n");
    }
};
json load_map(const std::string& path){
    if(std::filesystem::file_size(path)>65536)throw std::invalid_argument("clock map too large");
    std::ifstream input(path);if(!input)throw std::runtime_error("clock map unavailable");std::set<std::string> keys;
    auto callback=[&](int,json::parse_event_t event,json& parsed){if(event==json::parse_event_t::key&&!keys.insert(parsed.get<std::string>()).second)throw std::invalid_argument("duplicate clock map key");return true;};
    auto j=json::parse(input,callback);if(!j.is_object()||j.size()!=5||!j.contains("source_clock")||!j.contains("offset_ns")||!j.contains("valid_from_ns")||!j.contains("valid_until_ns")||!j.contains("max_error_ns"))throw std::invalid_argument("invalid clock map fields");
    return j;
}
std::uint64_t unsigned_field(const json& j,const char* key){auto const& v=j.at(key);if(!v.is_number_integer()||v.is_number_float()||(v.is_number_unsigned()?v.get<std::uint64_t>()>std::uint64_t(INT64_MAX):v.get<std::int64_t>()<0))throw std::invalid_argument("invalid clock integer");return v.get<std::uint64_t>();}
RosCaptureClockMapping mapping_from(const json& j){
    if(!j.at("source_clock").is_string()||!j.at("offset_ns").is_number_integer()||(j.at("offset_ns").is_number_unsigned()&&j.at("offset_ns").get<std::uint64_t>()>std::uint64_t(INT64_MAX)))throw std::invalid_argument("invalid clock mapping");
    RosCaptureClockMapping m{j.at("source_clock").get<std::string>(),j.at("offset_ns").get<std::int64_t>(),unsigned_field(j,"valid_from_ns"),unsigned_field(j,"valid_until_ns"),unsigned_field(j,"max_error_ns")};
    if(m.source_clock.empty()||m.source_clock.size()>4096||std::all_of(m.source_clock.begin(),m.source_clock.end(),[](unsigned char c){return std::isspace(c)!=0;}))throw std::invalid_argument("invalid source clock");
    map_ros_capture_stamp(m,m.source_clock,m.valid_from_ns);return m;
}
json capture_json(const CaptureMetrics& m){return {{"source_records",m.source_records},{"source_frames",m.source_frames},{"delivered_frames",m.delivered_frames},{"queue_drops",m.queue_drops},{"no_slot_drops",m.no_slot_drops},{"source_frame_gaps",m.source_frame_gaps},{"repeated_frames",m.repeated_frames},{"source_timeouts",m.source_timeouts},{"invalid_frames",m.invalid_frames},{"backpressure_waits",m.backpressure_waits},{"backpressure_wait_ns",m.backpressure_wait_ns},{"queued_frames",m.queued_frames},{"queue_high_watermark",m.queue_high_watermark},{"pool_capacity",m.pool_capacity},{"packed_bytes_per_slot",m.packed_bytes_per_slot},{"running",m.running}};}
json output_descriptor(const std::string& topic,unsigned w,unsigned h,const char* encoding,const std::string& frame,unsigned elem_bytes=4){return {{"topic",topic},{"width",w},{"height",h},{"step",w*elem_bytes},{"encoding",encoding},{"frame_id",frame}};}
struct Counters {std::uint64_t read=0,admitted=0,ring_drops=0,identity_rejects=0,published_depth=0,published_rgb=0,published_completed_depth=0,published_completed_provenance=0,timeouts=0;};
json counters_json(const Counters& c,bool has_completed=false){json j={{"read",c.read},{"admitted",c.admitted},{"ring_drops",c.ring_drops},{"identity_rejects",c.identity_rejects},{"published_depth",c.published_depth},{"published_rgb",c.published_rgb},{"received_depth",0},{"received_rgb",0},{"timeouts",c.timeouts}};if(has_completed){j["published_completed_depth"]=c.published_completed_depth;j["published_completed_provenance"]=c.published_completed_provenance;j["received_completed_depth"]=0;j["received_completed_provenance"]=0;}return j;}
struct Pending {DaxaDepthPipeline::Frame depth;DaxaFrame full;DaxaRefinement::Frame refined;DaxaFrame rgb;DaxaHostExport::Frame output,completed_output;CaptureLease capture;RosCaptureStamp stamp;std::uint64_t producer_observed=0,producer_enqueue_begin=0,producer_enqueue_end=0,export_enqueue_end=0;};
void initialize_image(sensor_msgs::msg::Image& image,unsigned w,unsigned h,const char* encoding,const std::string& frame,unsigned elem_bytes=4){image.width=w;image.height=h;image.step=w*elem_bytes;image.encoding=encoding;image.is_bigendian=0;image.header.frame_id=frame;image.data.resize(size_t(w)*h*elem_bytes);}
void run(const Options& o, OutputFile& file){
    if constexpr(std::endian::native!=std::endian::little)throw std::runtime_error("ROS image export requires a little-endian host");
    auto mapping_document=load_map(o.map);auto mapping=mapping_from(mapping_document);
    auto host_clock_id=clock_id();
    auto fixture=load_daxa_preprocess_fixture(o.manifest);auto& maps=fixture.rectification;maps.validate(2);
    const bool dual=o.pipeline_mode=="experimental-dual-refinement";
    const bool experimental=o.pipeline_mode!="full-sgm";
    DaxaRectification coarse_maps;
    if(experimental){auto coarse_fixture=load_daxa_preprocess_fixture(o.coarse_manifest);coarse_maps=std::move(coarse_fixture.rectification);coarse_maps.validate(2);validate_daxa_refinement_geometry(coarse_maps,maps);}
    if(fixture.record_ordinal>4095)throw std::invalid_argument("reference capture ordinal exceeds runtime setup bound");
    auto recorded=packed_replay_source(fixture.capture_directory);CaptureFrame reference;
    for(std::uint64_t i=0;i<=fixture.record_ordinal;++i)if(recorded(reference,std::chrono::milliseconds(100))!=CaptureSourceStatus::recorded_frame)throw std::runtime_error("reference capture ordinal unavailable");
    maps.validate_source(reference);if(experimental)coarse_maps.validate_source(reference);if(reference.device_identity.empty()||reference.device_identity=="unknown")throw std::invalid_argument("reference device association unavailable");
    auto expected_device=std::move(reference.device_identity);reference=CaptureFrame{};recorded={};
    make_ros_depth_contract(maps.width,maps.height,o.frame);const bool depth=o.mode!="rgb",rgb=o.mode!="depth";if(depth)maps.metric_calibration();
    auto node=std::make_shared<rclcpp::Node>("stereoforge_depth_publisher",rclcpp::NodeOptions().use_intra_process_comms(false).start_parameter_services(false).start_parameter_event_publisher(false));
    if(node->has_parameter("use_sim_time")&&node->get_parameter("use_sim_time").as_bool())throw std::invalid_argument("simulated time unsupported");
    auto qos_profile=rclcpp::QoS(rclcpp::KeepLast(o.qos_depth)).durability_volatile();if(o.reliability=="reliable")qos_profile.reliable();else qos_profile.best_effort();
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub,rgb_pub,completed_depth_pub,completed_provenance_pub;
    if(depth)depth_pub=node->create_publisher<sensor_msgs::msg::Image>(o.depth_topic,qos_profile);
    if(depth&&experimental&&o.planar_recovery){
        completed_depth_pub=node->create_publisher<sensor_msgs::msg::Image>(o.completed_depth_topic,qos_profile);
        completed_provenance_pub=node->create_publisher<sensor_msgs::msg::Image>(o.completed_provenance_topic,qos_profile);
    }
    if(rgb)rgb_pub=node->create_publisher<sensor_msgs::msg::Image>(o.rgb_topic,qos_profile);
    if(o.subscriber_wait>0){
        auto deadline=steady::now()+std::chrono::duration_cast<steady::duration>(std::chrono::duration<double>(o.subscriber_wait));
        while((depth&&depth_pub->get_subscription_count()==0)||(rgb&&rgb_pub->get_subscription_count()==0)){
            if(!rclcpp::ok()||steady::now()>=deadline)throw std::runtime_error("subscriber discovery timed out");
            std::this_thread::sleep_until(std::min(deadline,steady::now()+std::chrono::milliseconds(10)));
        }
    }
    sensor_msgs::msg::Image depth_image,rgb_image,completed_depth_image,completed_provenance_image;if(depth)initialize_image(depth_image,maps.width,maps.height,"32FC1",o.frame);if(depth&&experimental&&o.planar_recovery){initialize_image(completed_depth_image,maps.width,maps.height,"32FC1",o.frame);initialize_image(completed_provenance_image,maps.width,maps.height,"mono8",o.frame,1);}if(rgb)initialize_image(rgb_image,maps.width,maps.height,"rgba8",o.frame);
    CaptureConfig capture_config;capture_config.eye_width=maps.eye_width;capture_config.height=maps.source_height;capture_config.fps=o.fps;capture_config.device_id=o.device_id;
    capture_config.queue_capacity=2;capture_config.pool_capacity=6;capture_config.calibration_identity=maps.original_identity;capture_config.device_serial_override=o.device_serial;
    if(o.source=="replay")capture_config.queue_policy=CaptureQueuePolicy::wait_for_consumer;
    auto capture=o.source=="camera"?std::make_unique<NativeCapture>(capture_config):std::make_unique<NativeCapture>(capture_config,packed_replay_source(o.replay));
    auto check_source=[&](const CaptureLease& f){maps.validate_source(*f);if(experimental)coarse_maps.validate_source(*f);const bool override_device=o.device_serial=="26041639"&&f->device_identity==expected_device+":serial_override";if((f->device_identity!=expected_device&&!override_device)||f->camera_clock_domain!=mapping.source_clock||f->host_clock_domain!="host_steady"||f->frame_id>std::uint64_t(INT64_MAX)||!f->camera_timestamp_ns||f->camera_timestamp_ns>std::uint64_t(INT64_MAX)||!f->host_ingress_timestamp_ns||f->host_ingress_timestamp_ns>std::uint64_t(INT64_MAX))throw std::invalid_argument("capture device, geometry, identity or clock mismatch");};
    auto checked_stamp=[&](const CaptureLease& f){auto stamp=map_ros_capture_stamp(mapping,f->camera_clock_domain,f->camera_timestamp_ns);if(!stamp.sec&&!stamp.nanosec)throw std::invalid_argument("zero mapped capture timestamp");return stamp;};
    auto first=capture->read(std::chrono::seconds(2));if(!first)throw std::runtime_error("capture prototype unavailable");check_source(first);checked_stamp(first);
    FreshSourceTracker fresh;fresh.admit(static_cast<std::int64_t>(first->frame_id),first->camera_timestamp_ns,first->camera_clock_domain);
    std::uint64_t last_id=first->frame_id,last_capture=first->camera_timestamp_ns;
    auto context=std::make_shared<DaxaFoundation>(o.validation);
    std::unique_ptr<DaxaDepthPipeline> pipeline;std::unique_ptr<DaxaPreprocess> preprocess,full_preprocess;std::unique_ptr<DaxaRefinement> refinement;std::unique_ptr<DaxaHostExport> exporter,completed_exporter;std::unique_ptr<DaxaDualPreprocess> dual_preprocess;
    DaxaRefinementConfig refinement_config;refinement_config.disparities=o.disparities;refinement_config.borrow_full_frame=true;refinement_config.ortho_hybrid=(o.descriptor_mode=="ortho-hybrid");refinement_config.fast_mode=o.fast_mode;refinement_config.ultra_fast=o.ultra_fast;refinement_config.radius=o.ultra_fast?2u:4u;refinement_config.subpixel=o.refinement_subpixel?1u:0u;refinement_config.selective_repair=o.selective_repair;refinement_config.planar_recovery=o.planar_recovery;refinement_config.allow_derived_coarse=(o.coarse_right_mode=="left-aggregated-selection");
    uint32_t coarse_scale=(experimental&&maps.width==coarse_maps.width*4)?4u:2u;
    DaxaDepthConfig depth_config{StereoConfig{experimental?coarse_maps.width:maps.width,experimental?coarse_maps.height:maps.height,experimental?o.disparities/coarse_scale:o.disparities,4,8,32,true,o.descriptor_mode=="ortho-hybrid"},1.f,2,32};
    depth_config.stereo_schedule=o.coarse_schedule=="sequential"?DaxaDepthStereoSchedule::Sequential:o.coarse_schedule=="paired-per-path"?DaxaDepthStereoSchedule::PairedPerPath:o.coarse_schedule=="layered-per-path"?DaxaDepthStereoSchedule::LayeredPerPath:o.coarse_schedule=="hierarchical-1to8-fallback"?DaxaDepthStereoSchedule::Hierarchical1to8Selective1to4:DaxaDepthStereoSchedule::SharedCensusSequential;
    depth_config.right_reference_mode=o.coarse_right_mode=="left-aggregated-selection"?DaxaRightReferenceMode::LeftAggregatedSelection:DaxaRightReferenceMode::IndependentSGM;
    depth_config.stereo_variant=o.stereo_variant=="packed-specialized-overwrite-quad"?DaxaSgmVariant::PackedSpecializedOverwriteQuad:(o.stereo_variant=="packed-specialized-overwrite"?DaxaSgmVariant::PackedSpecializedOverwrite:DaxaSgmVariant::PackedSpecialized);
    if(depth){depth_config.validate();DaxaDualPreprocess::Frame pair;DaxaDepthPipeline::Frame prototype;
        if(dual){dual_preprocess=std::make_unique<DaxaDualPreprocess>(maps,coarse_maps,2,context);pair=dual_preprocess->enqueue(first);pipeline=std::make_unique<DaxaDepthPipeline>(depth_config,context,pair.coarse);prototype=pipeline->enqueue_preprocessed(pair.coarse);}
        else{pipeline=std::make_unique<DaxaDepthPipeline>(experimental?coarse_maps:maps,depth_config,context,first);prototype=pipeline->enqueue(first);}
        if(experimental){DaxaFrame full;if(dual)full=pair.full;else{full_preprocess=std::make_unique<DaxaPreprocess>(maps,2,context);full=full_preprocess->enqueue(first);}refinement=std::make_unique<DaxaRefinement>(context,prototype,full,refinement_config);auto refined=refinement->process(prototype,full);exporter=std::make_unique<DaxaHostExport>(context,refined.observations(),DaxaHostExportConfig{2,true,rgb});if(o.planar_recovery)completed_exporter=std::make_unique<DaxaHostExport>(context,refined.completed_observations(),DaxaHostExportConfig{2,true,false,true});refined.wait();refined={};full={};refinement->collect();if(!dual)full_preprocess->collect_completed_sources();}
        else{exporter=std::make_unique<DaxaHostExport>(context,prototype.observations(),DaxaHostExportConfig{2,true,rgb});prototype.wait();}prototype={};pair={};pipeline->collect();if(dual)dual_preprocess->collect();}
    else{preprocess=std::make_unique<DaxaPreprocess>(maps,2,context);auto prototype=preprocess->enqueue(first);exporter=std::make_unique<DaxaHostExport>(context,prototype,DaxaHostExportConfig{2,false,true});prototype.wait();prototype={};preprocess->collect_completed_sources();}
    first={};
    RosDeliveryTrace trace(size_t(o.max_frames)*(unsigned(depth)+unsigned(rgb)+(completed_exporter?2u:0u))+2);std::array<Pending,2> pending{};Counters counters{};
    std::unique_ptr<DepthSpeckleFilter> depth_speckle;
    if(depth&&o.depth_speckle_min_pixels){
        const auto mc=maps.metric_calibration();
        const bool disp_mode=o.depth_speckle_delta_px>0.0;
        depth_speckle=std::make_unique<DepthSpeckleFilter>(DepthSpeckleConfig{
            maps.width,maps.height,o.depth_speckle_min_pixels,float(o.depth_speckle_delta_m),
            mc.fx*mc.baseline_m,float(o.depth_speckle_delta_px),true,
            disp_mode?3u:0u,disp_mode?0.8f:0.0f,disp_mode?8.0f:0.0f,disp_mode?6u:0u});
    }
    std::uint64_t speckle_input_valid_total=0,speckle_output_valid_total=0;
    std::uint64_t speckle_invalidated_total=0,speckle_removed_total=0;
    std::uint64_t speckle_filter_ns_total=0,speckle_filtered_frames=0;
    DepthRateController rate_controller(depth_rate_config(o));
    std::unique_ptr<DutyCycleBudgetGovernor> duty_governor;
    if(o.gpu_duty_budget_pct>0.0)duty_governor=std::make_unique<DutyCycleBudgetGovernor>(DutyCycleBudgetConfig{o.gpu_duty_budget_pct/100.0,o.fps,o.min_depth_fps,o.max_depth_fps,0.20});
    YuyvSceneMonitor scene_monitor;
    json outputs=json::object();if(depth)outputs["depth"]=output_descriptor(o.depth_topic,maps.width,maps.height,"32FC1",o.frame);if(completed_exporter){outputs["completed_depth"]=output_descriptor(o.completed_depth_topic,maps.width,maps.height,"32FC1",o.frame);outputs["completed_provenance"]=output_descriptor(o.completed_provenance_topic,maps.width,maps.height,"mono8",o.frame,1);}if(rgb)outputs["rgb"]=output_descriptor(o.rgb_topic,maps.width,maps.height,"rgba8",o.frame);
    json qos={{"history","keep_last"},{"depth",o.qos_depth},{"reliability",o.reliability},{"durability","volatile"}};
    auto memory=exporter->memory();auto baseline=capture->metrics();
    auto effective_temporal=stereoforge::native_estimator::instantaneous();
    json estimator=json::object();
    if(depth&&!experimental&&depth_config.stereo_variant==DaxaSgmVariant::PackedSpecialized)estimator=stereoforge::native_estimator::profile(depth_config,effective_temporal,
        stereoforge::native_estimator::shader_hashes(DaxaStereo::shader_directory(),"daxa-stereo-"),
        stereoforge::native_estimator::shader_hashes(DaxaDepthPipeline::shader_directory(),"daxa-depth-"),SF_DAXA_COMPILER_SHA256);
    else if(depth)estimator={{"status","unavailable"},{"reason",experimental?"experimental_refinement_not_qualified":"experimental_variant_not_qualified_for_canonical_estimator_identity"}};
    json implementation={{"name","StereoForge"},{"version","native_ros_yuyv_v1"},
        {"commit",{{"status","unavailable"},{"reason","build commit is not attested"}}},
        {"device",{{"status","available"},{"value",json::parse(context->report())}}},
        {"settings",{{"mode",o.mode},{"depth_rate",o.depth_rate},{"capture_fps",o.fps},
            {"min_depth_fps",o.min_depth_fps},{"max_depth_fps",o.max_depth_fps},
            {"consumer_min_depth_fps",o.consumer_min_depth_fps},{"max_depth_age_ms",o.max_depth_age_ms},
            {"downshift_ms",o.downshift_ms},{"depth_speckle_min_pixels",o.depth_speckle_min_pixels},{"depth_speckle_delta_m",o.depth_speckle_delta_m},{"depth_speckle_delta_px",o.depth_speckle_delta_px},{"pipeline",o.pipeline_mode},{"coarse_manifest_sha256",experimental?daxa_preprocess_file_sha256(o.coarse_manifest):"not_executed"},{"stereo_variant",depth?DaxaStereo::variant_name(depth_config.stereo_variant):"not_executed"},{"effective_estimator",estimator},{"validation_requested",o.validation},{"subscriber_wait_seconds",o.subscriber_wait},
            {"reference_capture_sha256",fixture.capture_sha256},{"reference_record_ordinal",fixture.record_ordinal},
            {"calibration_manifest_sha256",daxa_preprocess_file_sha256(o.manifest)},{"device_serial_override",o.device_serial.empty()?json(nullptr):json(o.device_serial)},
            {"original_calibration_identity",maps.original_identity},{"rectified_calibration_identity",maps.rectified_identity}}}};
    if(experimental)implementation["settings"]["experimental_refinement"]={
        {"coarse_width",coarse_maps.width},{"coarse_height",coarse_maps.height},{"coarse_disparities",depth_config.stereo.disparities},
        {"coarse_paths",depth_config.stereo.paths},{"coarse_rectified_calibration_identity",coarse_maps.rectified_identity},
        {"target_width",maps.width},{"target_height",maps.height},{"disparities",refinement_config.disparities},
        {"radius",refinement_config.radius},{"min_margin",refinement_config.min_margin},
        {"prior_confidence_floor",refinement_config.prior_confidence_floor},{"max_coarse_span",refinement_config.max_coarse_span},
        {"lr_tolerance",refinement_config.lr_tolerance},{"subpixel",refinement_config.subpixel},
        {"descriptor_mode",o.descriptor_mode},{"profile",o.profile},{"gpu_duty_budget_pct",o.gpu_duty_budget_pct},{"fast_mode",refinement_config.fast_mode},{"ultra_fast",refinement_config.ultra_fast},{"selective_repair",refinement_config.selective_repair},{"planar_recovery",refinement_config.planar_recovery},
        {"coarse_schedule",o.coarse_schedule},{"coarse_right_reference_mode",o.coarse_right_mode},
        {"borrow_full_frame",refinement_config.borrow_full_frame},{"refinement_shader_identity",DaxaRefinement::shader_identity()},
        {"refinement_compiler_sha256",DaxaRefinement::compiler_sha256()},
        {"preprocess_shader_identity",DaxaPreprocess::shader_identity()},{"depth_shader_identity",DaxaDepthPipeline::shader_identity()},
        {"stereo_shader_hashes",stereoforge::native_estimator::shader_hashes(DaxaStereo::shader_directory(),"daxa-stereo-")}};
    json config={{"thread_cpu_timing",o.thread_cpu_timing},{"clock_id",host_clock_id},{"source_mode",o.source},{"nominal_hz",o.fps},{"slots",2},{"capture_queue_capacity",2},{"outputs",outputs},{"qos",qos},
        {"implementation",{{"status","available"},{"value",implementation}}},{"calibration_identity",maps.rectified_identity},{"effective_estimator",estimator},
        {"view_profile",{{"status","unavailable"},{"reason","view is not executed"}}},{"clock_mapping",mapping_document},{"capture_counters",capture_json(baseline)},
        {"memory",{{"producer_requested_bytes",dual?json(nullptr):json(depth?(pipeline->requested_combined_buffer_bytes()+(experimental?maps.requested_bytes(2)+refinement->requested_buffer_bytes():0)):maps.requested_bytes(2))},{"export_requested_bytes",exporter->requested_bytes()},
            {"capture_requested_bytes",(baseline.pool_capacity+1)*baseline.packed_bytes_per_slot},{"trace_capacity",trace.event_capacity()},{"trace_record_bytes",sizeof(RosDeliveryTraceRecord)},
            {"ros_requested_bytes",depth_image.data.size()+rgb_image.data.size()+completed_depth_image.data.size()+completed_provenance_image.data.size()},{"export_memory_type_index",memory.type_index},{"export_memory_property_flags",memory.property_flags},{"export_host_cached",memory.host_cached},
            {"storage_accounting_complete",false}}}};
    if(dual)config["memory"]["producer_requested_bytes_reason"]="dual_producer_requested_bytes_unavailable";
    auto valid_source=[&](const CaptureLease& f,RosCaptureStamp& stamp){check_source(f);if(f->frame_id<=last_id||f->camera_timestamp_ns<=last_capture)return false;stamp=checked_stamp(f);fresh.admit(static_cast<std::int64_t>(f->frame_id),f->camera_timestamp_ns,f->camera_clock_domain);last_id=f->frame_id;last_capture=f->camera_timestamp_ns;return true;};
    auto warmup_deadline=now_ns<steady>()+30'000'000'000ull;
    for(unsigned i=0;i<o.warmup;){if(now_ns<steady>()>=warmup_deadline)throw std::runtime_error("capture warmup timed out");auto f=capture->read(std::chrono::seconds(1));if(!f){if(!capture->metrics().running)throw std::runtime_error("capture ended during warmup");continue;}RosCaptureStamp ignored;if(!valid_source(f,ignored))continue;
        if(depth){DaxaDualPreprocess::Frame pair;DaxaDepthPipeline::Frame d;DaxaFrame full;DaxaRefinement::Frame refined;if(dual){pair=dual_preprocess->enqueue(f);d=pipeline->enqueue_preprocessed(pair.coarse);full=pair.full;}else{d=pipeline->enqueue(f);if(experimental)full=full_preprocess->enqueue(f);}if(experimental){refined=refinement->process(d,full);}auto h=exporter->enqueue(experimental?refined.observations():d.observations());DaxaHostExport::Frame hc;if(completed_exporter)hc=completed_exporter->enqueue(refined.completed_observations());h.wait();if(completed_exporter)hc.wait();hc={};h={};refined={};full={};d={};pair={};if(completed_exporter)completed_exporter->collect();exporter->collect();if(experimental){refinement->collect();if(!dual)full_preprocess->collect_completed_sources();}pipeline->collect();if(dual)dual_preprocess->collect();}
        else{auto p=preprocess->enqueue(f);auto h=exporter->enqueue(p);h.wait();h={};p={};exporter->collect();preprocess->collect_completed_sources();}++i;}
    baseline=capture->metrics();config["capture_counters"]=capture_json(baseline);auto started=now_ns<steady>();
    bool trace_full=false;std::string reason="limit",failure;auto deadline=started+static_cast<std::uint64_t>(o.duration*1e9);
    auto count_pending=[&](){unsigned n=0;for(auto const& p:pending)if(p.output)++n;return n;};
    auto retire=[&](Pending& p,bool drain){
        if(!p.output)return;
        auto producer_ready=depth?(experimental?p.refined.ready():p.depth.ready()):p.rgb.ready();if(producer_ready&&!p.producer_observed)p.producer_observed=now_ns<steady>();
        if(drain)p.output.wait();else if(!p.output.ready())return;
        if(completed_exporter){if(drain)p.completed_output.wait();else if(!p.completed_output.ready())return;}
        auto copy_observed=now_ns<steady>();if(!p.producer_observed)p.producer_observed=copy_observed;
        if(duty_governor&&p.producer_enqueue_begin&&p.producer_observed>=p.producer_enqueue_begin){
            duty_governor->observe_gpu_ms(double(p.producer_observed-p.producer_enqueue_begin)/1e6);
        }
        auto pending_count=count_pending();auto queued=capture->metrics().queued_frames;
        auto emit=[&](RosDeliveryOutput which,sensor_msgs::msg::Image& image,const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher){
            if(trace.records().size()>=trace.event_capacity()-2){RosDeliveryTraceRecord unused{};trace.append(unused);trace_full=true;return;}
            const bool is_comp=(which==RosDeliveryOutput::CompletedDepth||which==RosDeliveryOutput::CompletedProvenance);
            const auto& exp_frame=is_comp?p.completed_output:p.output;
            auto bytes=which==RosDeliveryOutput::Depth?p.output.depth_bytes():which==RosDeliveryOutput::Rgb?p.output.rgba_bytes():which==RosDeliveryOutput::CompletedDepth?p.completed_output.depth_bytes():p.completed_output.provenance_bytes();
            if(which==RosDeliveryOutput::CompletedProvenance){if(bytes.size()!=image.data.size()*4)throw std::runtime_error("export provenance extent mismatch");}
            else if(bytes.size()!=image.data.size())throw std::runtime_error("export extent mismatch");
            RosDeliveryTraceRecord r{};r.kind=RosDeliveryTraceKind::Published;r.output=which;r.freshness=RosDeliveryFreshness::MeasuredSingleSource;
            r.source_id=static_cast<std::int64_t>(p.capture->frame_id);r.capture_ns=p.capture->camera_timestamp_ns;r.host_ingress_ns=p.capture->host_ingress_timestamp_ns;
            r.mapped_capture_ns=std::uint64_t(p.stamp.sec)*1'000'000'000u+p.stamp.nanosec;r.mapping_error_ns=p.stamp.max_error_ns;
            r.producer_ticket=exp_frame.producer_ticket();r.copy_ticket=exp_frame.copy_ticket();
            r.producer_enqueue_begin_ns=p.producer_enqueue_begin;r.producer_enqueue_end_ns=p.producer_enqueue_end;r.export_enqueue_end_ns=p.export_enqueue_end;
            r.producer_complete_observed_ns=p.producer_observed;r.copy_complete_observed_ns=copy_observed;
            r.capture_queue=static_cast<std::uint32_t>(queued);r.pending=pending_count;image.header.stamp.sec=p.stamp.sec;image.header.stamp.nanosec=p.stamp.nanosec;
            bool cpu_ok=true;
            r.copy_begin_ns=now_ns<steady>();
            if(o.thread_cpu_timing)cpu_ok &= thread_cpu_now(r.thread_cpu_copy_begin_ns);
            if(which==RosDeliveryOutput::CompletedProvenance){
                const auto* src_u32=reinterpret_cast<const std::uint32_t*>(bytes.data());
                const size_t npix=image.data.size();
                for(size_t i=0;i<npix;++i)image.data[i]=static_cast<std::uint8_t>(src_u32[i]&0xffu);
            } else {
                std::memcpy(image.data.data(),bytes.data(),bytes.size());
            }
            if(which==RosDeliveryOutput::Depth&&depth_speckle){
                const auto filter_begin=now_ns<steady>();
                const auto rgba_span=rgb?p.output.rgba_bytes():std::span<const std::byte>{};
                const auto stats=depth_speckle->apply(std::as_writable_bytes(std::span(image.data)),rgba_span);
                speckle_filter_ns_total+=now_ns<steady>()-filter_begin;
                ++speckle_filtered_frames;
                speckle_input_valid_total+=stats.input_valid_pixels;
                speckle_output_valid_total+=stats.output_valid_pixels;
                speckle_invalidated_total+=stats.invalidated_pixels;
                speckle_removed_total+=stats.removed_components;
            }
            // Preserve the GPU's 32FC1 invalid depth encoding, including NaN.
            // Do not silently replace NaNs with zero during ROS publication.
            if(o.thread_cpu_timing)cpu_ok &= thread_cpu_now(r.thread_cpu_copy_end_ns);
            r.copy_end_ns=now_ns<steady>();
            r.publish_begin_ns=now_ns<steady>();
            if(o.thread_cpu_timing)cpu_ok &= thread_cpu_now(r.thread_cpu_publish_begin_ns);
            publisher->publish(image);
            if(o.thread_cpu_timing)cpu_ok &= thread_cpu_now(r.thread_cpu_publish_end_ns);
            r.publish_end_ns=now_ns<steady>();r.at_ns=r.publish_end_ns;
            if(o.verify_payload_checksum){
                r.payload_fnv1a64=fnv1a64_words(image.data.data(),image.data.size());
                const size_t npix=size_t(image.width)*image.height;
                std::uint32_t valid_px=0;
                if(which==RosDeliveryOutput::Depth||which==RosDeliveryOutput::CompletedDepth){
                    const auto* fp=reinterpret_cast<const float*>(image.data.data());
                    for(size_t i=0;i<npix;++i)if(std::isfinite(fp[i])&&fp[i]>0.0f)++valid_px;
                } else if(which==RosDeliveryOutput::CompletedProvenance){
                    for(size_t i=0;i<npix;++i)if(image.data[i]==1u||image.data[i]==6u)++valid_px;
                } else {
                    const auto* up=reinterpret_cast<const std::uint32_t*>(image.data.data());
                    for(size_t i=0;i<npix;++i)if((up[i]&0x00FFFFFFu)!=0u)++valid_px;
                }
                r.valid_pixels=valid_px;
            }
            if(o.thread_cpu_timing)r.thread_cpu_status=!cpu_ok?RosThreadCpuStatus::ClockReadFailed:
                (r.thread_cpu_copy_begin_ns<=r.thread_cpu_copy_end_ns && r.thread_cpu_copy_end_ns<=r.thread_cpu_publish_begin_ns &&
                 r.thread_cpu_publish_begin_ns<=r.thread_cpu_publish_end_ns?RosThreadCpuStatus::Available:RosThreadCpuStatus::Nonmonotonic);
            if(!trace.append(r)){trace_full=true;return;}
            if(which==RosDeliveryOutput::Depth)++counters.published_depth;
            else if(which==RosDeliveryOutput::CompletedDepth)++counters.published_completed_depth;
            else if(which==RosDeliveryOutput::CompletedProvenance)++counters.published_completed_provenance;
            else ++counters.published_rgb;
        };
        if(depth)emit(RosDeliveryOutput::Depth,depth_image,depth_pub);
        if(completed_exporter&&completed_depth_pub&&!trace_full)emit(RosDeliveryOutput::CompletedDepth,completed_depth_image,completed_depth_pub);
        if(completed_exporter&&completed_provenance_pub&&!trace_full)emit(RosDeliveryOutput::CompletedProvenance,completed_provenance_image,completed_provenance_pub);
        if(rgb&&!trace_full)emit(RosDeliveryOutput::Rgb,rgb_image,rgb_pub);
        p.completed_output={};p.output={};p.refined={};p.full={};p.depth={};p.rgb={};p.capture={};if(completed_exporter)completed_exporter->collect();exporter->collect();if(experimental){refinement->collect();if(!dual)full_preprocess->collect_completed_sources();}if(depth)pipeline->collect();else preprocess->collect_completed_sources();if(dual)dual_preprocess->collect();
    };
    try{
        while(rclcpp::ok()&&!trace_full&&now_ns<steady>()<deadline&&counters.admitted<o.max_frames){
            for(auto& p:pending){retire(p,false);if(trace_full)break;}
            if(trace_full)break;
            // Replay is lossless: leave the next input queued until a slot retires.
            // Live capture still dequeues and drops stale inputs when both slots are busy.
            if(o.source=="replay"&&count_pending()==pending.size()){
                std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;
            }
            // Count read poll timeouts, not camera frame loss; source_timeouts are tracked separately.
            auto capture_wait=count_pending()?std::chrono::milliseconds(1):std::chrono::milliseconds(5);
            auto f=capture->read(capture_wait);if(!f){++counters.timeouts;if(!capture->metrics().running){reason="eof";break;}continue;}++counters.read;
            RosCaptureStamp stamp;if(!valid_source(f,stamp)){++counters.identity_rejects;continue;}
            if(depth){
                std::optional<float> activity;
                if(o.depth_rate=="adaptive")
                    activity=scene_monitor.observe(f->packed_yuyv,f->source_width,f->height,f->stride_bytes);
                if(!rate_controller.observe(f->camera_timestamp_ns,activity).due)continue;
            }
            auto index=std::find_if(pending.begin(),pending.end(),[](auto const& p){return !p.output;});
            if(index==pending.end()){if(depth)rate_controller.reject_busy();++counters.ring_drops;continue;}
            index->capture=f;index->stamp=stamp;index->producer_observed=0;
            index->producer_enqueue_begin=now_ns<steady>();
            if(depth&&dual){auto pair=dual_preprocess->enqueue(f);index->depth=pipeline->enqueue_preprocessed(pair.coarse);index->full=pair.full;index->refined=refinement->process(index->depth,index->full);index->producer_enqueue_end=now_ns<steady>();index->output=exporter->enqueue(index->refined.observations());}
            else if(depth){index->depth=pipeline->enqueue(f);if(experimental){index->full=full_preprocess->enqueue(f);index->refined=refinement->process(index->depth,index->full);}index->producer_enqueue_end=now_ns<steady>();index->output=exporter->enqueue(experimental?index->refined.observations():index->depth.observations());}
            else{index->rgb=preprocess->enqueue(f);index->producer_enqueue_end=now_ns<steady>();index->output=exporter->enqueue(index->rgb);}
            index->export_enqueue_end=now_ns<steady>();if(completed_exporter)index->completed_output=completed_exporter->enqueue(index->refined.completed_observations());++counters.admitted;
            if(depth){
                rate_controller.admitted(f->camera_timestamp_ns);
                if(o.depth_rate=="adaptive")scene_monitor.depth_admitted();
            }
        }
        if(trace_full)reason="trace_full";else if(!rclcpp::ok())reason="shutdown";
    }catch(const std::exception& e){reason="error";failure=e.what();}
    auto window_end=now_ns<steady>();auto window_counters=counters;auto window_capture=capture->metrics();auto window_pending=count_pending();auto drain_begin=now_ns<steady>();
    try{capture->stop();for(auto& p:pending)retire(p,true);if(completed_exporter)completed_exporter->drain();exporter->drain();if(depth)pipeline->drain();else preprocess->collect_completed_sources();}
    catch(const std::exception& e){reason="error";failure=e.what();}
    if(trace_full&&reason!="error")reason="trace_full";
    auto drain_end=now_ns<steady>();for(auto& p:pending){p.output={};p.refined={};p.full={};p.depth={};p.rgb={};p.capture={};}for(auto& p:pending)p.completed_output={};const bool has_completed=(depth&&experimental&&o.planar_recovery);completed_exporter.reset();exporter.reset();refinement.reset();full_preprocess.reset();pipeline.reset();dual_preprocess.reset();preprocess.reset();
    try{context->close();}catch(const std::exception& e){reason="error";failure=e.what();}
    if(context->validation_errors.load()||context->validation_warnings.load()){reason="error";failure="Vulkan validation errors or warnings";}
    json begin={{"schema","stereoforge.ros_delivery_event.v1"},{"kind","start"},{"run_id",o.run},{"role","publisher"},{"at_ns",started},{"config",config}};
    json end={{"schema","stereoforge.ros_delivery_event.v1"},{"kind","stop"},{"run_id",o.run},{"role","publisher"},{"at_ns",drain_end},{"window_end_ns",window_end},
        {"reason",reason},{"counters",counters_json(window_counters,has_completed)},{"capture_counters",capture_json(window_capture)},{"queued",window_capture.queued_frames},
        {"pending",window_pending},{"trace_drops",trace.dropped()},{"drain_ns",drain_end-drain_begin}};
    if(duty_governor)end["gpu_duty_governor"]={{"budget_pct",o.gpu_duty_budget_pct},{"ema_gpu_ms",duty_governor->ema_gpu_ms()},{"recommended_fps",duty_governor->recommended_fps()},{"estimated_duty_pct",duty_governor->estimated_duty_fraction()*100.0},{"observations",duty_governor->observations()},{"budget_throttles",duty_governor->budget_throttles()}};
    file.reset();file.put("[");file.put(begin.dump());for(auto const& r:trace.records()){file.put(",");file.put(ros_delivery_event_json(r,o.run,"publisher",config).dump());}file.put(",");file.put(end.dump());file.put("]\n");file.complete=true;
    if(depth){
        std::cerr<<"Depth speckle frames="<<speckle_filtered_frames
                 <<" valid_before="<<speckle_input_valid_total
                 <<" valid_after="<<speckle_output_valid_total
                 <<" invalidated_pixels="<<speckle_invalidated_total
                 <<" removed_components="<<speckle_removed_total
                 <<" mean_filter_ms="<<(speckle_filtered_frames?double(speckle_filter_ns_total)/double(speckle_filtered_frames)/1e6:0.0)
                 <<'\n';
        const auto& stats=rate_controller.stats();
        std::cerr<<"Depth rate "<<o.depth_rate<<": observed="<<stats.observed
                 <<" admitted="<<stats.admitted<<" rate_skips="<<stats.intentional_skips
                 <<" slot_rejections="<<stats.slot_rejections<<" final_fps="<<stats.effective_fps<<'\n';
        if(!o.rate_report.empty()){
            OutputFile report(o.rate_report);
            report.reset();
            report.put(json({{"schema","stereoforge.depth_rate_report.v1"},
                {"source",o.source},{"clock_basis","source_camera_timestamp_ns_unvalidated_host_mapping"},
                {"mode",o.depth_rate},{"capture_fps",o.fps},{"min_fps",o.min_depth_fps},
                {"max_fps",o.max_depth_fps},{"consumer_min_fps",o.consumer_min_depth_fps},
                {"max_depth_age_ms",o.max_depth_age_ms},{"downshift_ms",o.downshift_ms},
                {"observed",stats.observed},{"admitted",stats.admitted},
                {"intentional_skips",stats.intentional_skips},{"slot_rejections",stats.slot_rejections},
                {"upshifts",stats.upshifts},{"downshifts",stats.downshifts},
                {"unavailable_motion",stats.unavailable_motion},{"final_target_fps",stats.effective_fps},
                {"published_depth",window_counters.published_depth},
                {"published_rgb",window_counters.published_rgb},
                {"speckle_invalidated_pixels",speckle_invalidated_total},
                {"speckle_removed_components",speckle_removed_total},
                {"speckle_input_valid_pixels",speckle_input_valid_total},
                {"speckle_output_valid_pixels",speckle_output_valid_total},
                {"speckle_filtered_frames",speckle_filtered_frames},
                {"speckle_cpu_total_ns",speckle_filter_ns_total}}).dump()+"\n");
            report.complete=true;
        }
    }
    if(!failure.empty()||reason=="trace_full")throw std::runtime_error("publisher stopped: "+reason+" "+failure);
}
}
int main(int argc,char** argv){bool ros_initialized=false;bool ros_initializing=false;
    try {
        auto options=parse(argc,argv);OutputFile file(options.trace);
        try {stereoforge::set_default_ros_rmw();ros_initializing=true;rclcpp::init(0,nullptr);ros_initialized=true;ros_initializing=false;std::cerr<<"ROS middleware: "<<rmw_get_implementation_identifier()<<'\n';run(options,file);rclcpp::shutdown();return 0;}
        catch(const std::exception& e){try {file.failure(e.what());} catch (...) {} throw;}
        catch(...){try {file.failure("unknown exception");} catch (...) {} throw;}
    } catch(const std::exception& e){if(ros_initialized){try{rclcpp::shutdown();}catch(...){}}std::cerr<<e.what()<<'\n';if(ros_initializing && std::getenv("RMW_IMPLEMENTATION") && std::string(std::getenv("RMW_IMPLEMENTATION"))=="rmw_zenoh_cpp")std::cerr<<"Install the ROS Zenoh RMW package and source ROS, or set RMW_IMPLEMENTATION to another installed RMW.\n";return 1;}
    catch(...){if(ros_initialized){try{rclcpp::shutdown();}catch(...){}}std::cerr<<"unknown exception\n";return 1;}
}
