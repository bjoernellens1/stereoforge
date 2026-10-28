// Opt-in, same-process ROS/Daxa handoff. The ROS wire type is Header only.
#include <stereoforge/capture.hpp>
#include <stereoforge/daxa_depth.hpp>
#include <stereoforge/daxa_gpu_consumer.hpp>
#include <stereoforge/ros_depth_contract.hpp>
#include <stereoforge/ros_gpu_frame.hpp>
#include <stereoforge/ros_default_rmw.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/rmw.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
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
#include <sys/stat.h>
#include <unistd.h>
#if __has_include(<json.hpp>)
#include <json.hpp>
#else
#include <nlohmann/json.hpp>
#endif
using namespace stereoforge;
using nlohmann::json;
namespace {
using steady = std::chrono::steady_clock;
template<class Clock> std::uint64_t now_ns(){auto n=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();if(n<=0)throw std::runtime_error("clock unavailable");return static_cast<std::uint64_t>(n);}
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
struct Options {
    std::string source,manifest,replay,map,frame,trace,topic="/stereoforge/gpu_frame";
    unsigned max_frames=1000,disparities=128; int device_id=-1; double duration=30; bool validation=false;
};
unsigned number(const std::string& s,unsigned lo,unsigned hi){size_t n=0;auto v=std::stoull(s,&n);if(n!=s.size()||v<lo||v>hi)throw std::invalid_argument("integer option outside bounds");return static_cast<unsigned>(v);}
Options parse(int argc,char** argv){
    Options o;
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--help"){std::cout<<"Same-process ROS GPU depth and rectified RGB (wire Header contains metadata only).\n--source camera|replay --manifest FILE --clock-map FILE --frame-id ID --trace FILE [--replay DIR --topic TOPIC --duration-seconds N --max-frames N --disparities N --device-id N --validation]\n";std::exit(0);}
        if(a=="--validation"){o.validation=true;continue;}if(i+1>=argc)throw std::invalid_argument("missing option value");std::string v=argv[++i];
        if(a=="--source")o.source=v;else if(a=="--manifest")o.manifest=v;else if(a=="--replay")o.replay=v;else if(a=="--clock-map")o.map=v;else if(a=="--frame-id")o.frame=v;else if(a=="--trace")o.trace=v;else if(a=="--topic")o.topic=v;
        else if(a=="--max-frames")o.max_frames=number(v,1,200000);else if(a=="--disparities")o.disparities=number(v,32,256);
        else if(a=="--device-id"){if(v=="-1")o.device_id=-1;else o.device_id=static_cast<int>(number(v,0,255));}
        else if(a=="--duration-seconds"){size_t n=0;o.duration=std::stod(v,&n);if(n!=v.size()||!std::isfinite(o.duration)||o.duration<=0||o.duration>3600)throw std::invalid_argument("invalid duration");}
        else throw std::invalid_argument("unknown option: "+a);
    }
    if((o.source!="camera"&&o.source!="replay")||(o.source=="replay")!=!o.replay.empty()||o.manifest.empty()||o.map.empty()||o.trace.empty()||o.frame.empty()||o.frame.size()>4096||o.topic.empty()||o.topic.front()!='/')throw std::invalid_argument("invalid GPU composed configuration");
    return o;
}
struct Sample {std::uint64_t source_id=0,capture_ns=0,host_ingress_ns=0,mapped_capture_ns=0,enqueue_begin_ns=0,enqueue_end_ns=0,publish_begin_ns=0,publish_end_ns=0,callback_ns=0,copy_enqueue_end_ns=0,copy_complete_observed_ns=0,producer_ticket=0,copy_ticket=0;};
struct Pending {DaxaGpuConsumer::Frame output; Sample sample;};
void run(const Options& o,OutputFile& file){
    auto map_document=load_map(o.map);auto mapping=mapping_from(map_document);auto host_clock=clock_id();
    auto fixture=load_daxa_preprocess_fixture(o.manifest);auto& maps=fixture.rectification;maps.validate(2);maps.metric_calibration();
    if(fixture.record_ordinal>4095)throw std::invalid_argument("reference capture ordinal exceeds runtime setup bound");
    auto recorded=packed_replay_source(fixture.capture_directory);CaptureFrame reference;
    for(std::uint64_t i=0;i<=fixture.record_ordinal;++i)if(recorded(reference,std::chrono::milliseconds(100))!=CaptureSourceStatus::recorded_frame)throw std::runtime_error("reference capture ordinal unavailable");
    maps.validate_source(reference);if(reference.device_identity.empty()||reference.device_identity=="unknown")throw std::invalid_argument("reference device association unavailable");
    auto expected_device=std::move(reference.device_identity);reference={};recorded={};
    CaptureConfig config;config.eye_width=maps.eye_width;config.height=maps.source_height;config.fps=60;config.device_id=o.device_id;config.queue_capacity=2;config.pool_capacity=6;config.calibration_identity=maps.original_identity;
    if(o.source=="replay")config.queue_policy=CaptureQueuePolicy::wait_for_consumer;
    auto capture=o.source=="camera"?std::make_unique<NativeCapture>(config):std::make_unique<NativeCapture>(config,packed_replay_source(o.replay));
    auto check_source=[&](const CaptureLease& f){maps.validate_source(*f);if(f->device_identity!=expected_device||f->camera_clock_domain!=mapping.source_clock||f->host_clock_domain!="host_steady"||f->frame_id>std::uint64_t(INT64_MAX)||!f->camera_timestamp_ns||f->camera_timestamp_ns>std::uint64_t(INT64_MAX)||!f->host_ingress_timestamp_ns||f->host_ingress_timestamp_ns>std::uint64_t(INT64_MAX))throw std::invalid_argument("capture device, geometry, identity or clock mismatch");};
    auto checked_stamp=[&](const CaptureLease& f){auto stamp=map_ros_capture_stamp(mapping,f->camera_clock_domain,f->camera_timestamp_ns);if(!stamp.sec&&!stamp.nanosec)throw std::invalid_argument("zero mapped capture timestamp");return stamp;};
    auto first=capture->read(std::chrono::seconds(2));if(!first)throw std::runtime_error("capture prototype unavailable");check_source(first);checked_stamp(first);
    std::uint64_t last_id=first->frame_id,last_capture=first->camera_timestamp_ns;
    auto context=std::make_shared<DaxaFoundation>(o.validation);
    DaxaDepthConfig depth_config{StereoConfig{maps.width,maps.height,o.disparities,4,8,32,true},1.f,2,32};depth_config.validate();
    auto pipeline=std::make_unique<DaxaDepthPipeline>(maps,depth_config,context,first);
    auto prototype=pipeline->enqueue(first);auto consumer=std::make_unique<DaxaGpuConsumer>(context,prototype.observations(),2);prototype.wait();prototype={};pipeline->collect();first={};
    rclcpp::NodeOptions node_options;node_options.use_intra_process_comms(true).start_parameter_services(false).start_parameter_event_publisher(false);
    auto publisher_node=std::make_shared<rclcpp::Node>("stereoforge_gpu_publisher",node_options);
    auto subscriber_node=std::make_shared<rclcpp::Node>("stereoforge_gpu_subscriber",node_options);
    if((publisher_node->has_parameter("use_sim_time")&&publisher_node->get_parameter("use_sim_time").as_bool())||(subscriber_node->has_parameter("use_sim_time")&&subscriber_node->get_parameter("use_sim_time").as_bool()))throw std::invalid_argument("simulated time unsupported");
    using Adapted=rclcpp::TypeAdapter<RosGpuFrame,std_msgs::msg::Header>;
    auto qos=rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
    auto publisher=publisher_node->create_publisher<Adapted>(o.topic,qos);
    std::array<Pending,2> pending{};std::vector<Sample> samples;samples.reserve(o.max_frames);
    std::uint64_t admitted=0,received=0,dropped=0,read=0,identity_rejects=0,timeouts=0;Sample incoming{};
    auto subscriber=subscriber_node->create_subscription<Adapted>(o.topic,qos,[&](std::shared_ptr<const RosGpuFrame> frame){
        ++received;if(frame->observation().source()->frame_id!=incoming.source_id)throw std::runtime_error("unexpected GPU message identity");incoming.callback_ns=now_ns<steady>();
        auto slot=std::find_if(pending.begin(),pending.end(),[](auto const& p){return !p.output;});
        if(slot==pending.end()||!consumer->has_capacity()){++dropped;return;}
        slot->sample=incoming;slot->output=consumer->enqueue(frame->observation());slot->sample.copy_enqueue_end_ns=now_ns<steady>();
        slot->sample.producer_ticket=slot->output.producer_ticket();slot->sample.copy_ticket=slot->output.copy_ticket();
    });
    rclcpp::executors::SingleThreadedExecutor executor;executor.add_node(publisher_node);executor.add_node(subscriber_node);
    auto producer_bytes=pipeline->requested_combined_buffer_bytes();auto consumer_bytes=consumer->requested_bytes();
    auto started=now_ns<steady>();auto deadline=started+static_cast<std::uint64_t>(o.duration*1e9);std::string reason="limit",failure;
    auto retire=[&](){for(auto& p:pending)if(p.output&&p.output.ready()){p.sample.copy_complete_observed_ns=now_ns<steady>();samples.push_back(p.sample);p.output={};}consumer->collect();pipeline->collect();};
    try{
        while(rclcpp::ok()&&admitted<o.max_frames&&now_ns<steady>()<deadline){
            executor.spin_some();retire();
            bool full=std::all_of(pending.begin(),pending.end(),[](auto const& p){return bool(p.output);});
            if(full&&o.source=="replay"){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
            auto f=capture->read(full?std::chrono::milliseconds(1):std::chrono::milliseconds(5));if(!f){++timeouts;if(!capture->metrics().running){reason="eof";break;}continue;}++read;
            check_source(f);if(f->frame_id<=last_id||f->camera_timestamp_ns<=last_capture){++identity_rejects;continue;}last_id=f->frame_id;last_capture=f->camera_timestamp_ns;
            auto stamp=checked_stamp(f);if(full||!consumer->has_capacity()){++dropped;continue;}
            incoming={};incoming.source_id=f->frame_id;incoming.capture_ns=f->camera_timestamp_ns;incoming.host_ingress_ns=f->host_ingress_timestamp_ns;incoming.mapped_capture_ns=std::uint64_t(stamp.sec)*1'000'000'000ull+stamp.nanosec;
            incoming.enqueue_begin_ns=now_ns<steady>();auto depth=pipeline->enqueue(f);incoming.enqueue_end_ns=now_ns<steady>();
            std_msgs::msg::Header header;header.frame_id=o.frame;header.stamp.sec=stamp.sec;header.stamp.nanosec=stamp.nanosec;
            auto message=std::make_unique<RosGpuFrame>(depth.observations(),std::move(header));depth={};f={};
            incoming.publish_begin_ns=now_ns<steady>();auto before_received=received;++admitted;publisher->publish(std::move(message));incoming.publish_end_ns=now_ns<steady>();executor.spin_some();if(received!=before_received+1)throw std::runtime_error("GPU message was not delivered synchronously to local subscriber");
        }
        if(!rclcpp::ok())reason="shutdown";
    }catch(const std::exception& e){reason="error";failure=e.what();}
    auto window_end=now_ns<steady>();auto capture_stats=capture->metrics();auto drain_begin=now_ns<steady>();
    try{capture->stop();executor.spin_some();consumer->drain();retire();}
    catch(const std::exception& e){reason="error";failure=e.what();}
    // Release every queued ROS lease before producer/consumer teardown, even on error.
    executor.remove_node(subscriber_node);executor.remove_node(publisher_node);
    subscriber.reset();publisher.reset();subscriber_node.reset();publisher_node.reset();
    try{consumer->drain();for(auto& p:pending)p.output={};consumer->collect();pipeline->drain();}
    catch(const std::exception& e){reason="error";failure=e.what();}
    for(auto& p:pending){p.output={};}
    consumer.reset();pipeline.reset();
    try{context->close();}catch(const std::exception& e){reason="error";failure=e.what();}
    if(context->validation_errors.load()||context->validation_warnings.load()){reason="error";failure="Vulkan validation errors or warnings";}
    auto drain_end=now_ns<steady>();json j={{"schema","stereoforge.ros_gpu_composed.v1"},{"source_mode",o.source},{"clock_id",host_clock},{"clock_mapping",map_document},{"clock_mapping_status","unverified"},{"mapping_error_ns",mapping.max_error_ns},{"observation_age",{{"status","unavailable"},{"reason","capture clock mapping has not been independently verified"}}},{"started_ns",started},{"window_end_ns",window_end},{"drain_ns",drain_end-drain_begin},{"reason",reason},{"failure",failure},{"read",read},{"admitted",admitted},{"received",received},{"drops",dropped},{"identity_rejects",identity_rejects},{"read_timeouts",timeouts},{"capture_queue_drops",capture_stats.queue_drops},{"capture_no_slot_drops",capture_stats.no_slot_drops},{"producer_requested_bytes",producer_bytes},{"consumer_requested_bytes",consumer_bytes},{"memory_accounting",{{"status","partial"},{"producer_and_consumer_bytes","requested GPU buffers only"},{"capture","unavailable"},{"imu","unavailable"},{"ros","unavailable"},{"driver","unavailable"}}},{"transport","in_process_owned_observation"},{"wire_type","std_msgs/msg/Header: metadata only"},{"outputs","axial depth and rectified left RGBA"},{"consumer_operation","same-device GPU-to-GPU copies"},{"consumer_scope","local copy demonstration only; downstream robotics consumption unverified"},{"sidecars","validity, confidence, age and provenance are retained in producer observation but are not copied to consumer buffers"},{"samples",json::array()}};
    for(auto const& s:samples)j["samples"].push_back({{"source_id",s.source_id},{"capture_ns",s.capture_ns},{"host_ingress_ns",s.host_ingress_ns},{"mapped_capture_ns",s.mapped_capture_ns},{"producer_enqueue_begin_ns",s.enqueue_begin_ns},{"producer_enqueue_end_ns",s.enqueue_end_ns},{"publish_begin_ns",s.publish_begin_ns},{"publish_end_ns",s.publish_end_ns},{"callback_ns",s.callback_ns},{"copy_enqueue_end_ns",s.copy_enqueue_end_ns},{"copy_complete_observed_ns",s.copy_complete_observed_ns},{"producer_ticket",s.producer_ticket},{"copy_ticket",s.copy_ticket}});
    file.reset();file.put(j.dump()+"\n");file.complete=true;if(!failure.empty())throw std::runtime_error("GPU composed run failed: "+failure);
}
}
int main(int argc,char** argv){bool ros_initialized=false;bool ros_initializing=false;try{auto options=parse(argc,argv);OutputFile file(options.trace);try{stereoforge::set_default_ros_rmw();ros_initializing=true;rclcpp::init(0,nullptr);ros_initialized=true;ros_initializing=false;std::cerr<<"ROS middleware: "<<rmw_get_implementation_identifier()<<'\n';run(options,file);rclcpp::shutdown();return 0;}catch(const std::exception& e){try{file.failure(e.what());}catch(...){}throw;}}catch(const std::exception& e){if(ros_initialized){try{rclcpp::shutdown();}catch(...){}}std::cerr<<e.what()<<'\n';if(ros_initializing && std::getenv("RMW_IMPLEMENTATION") && std::string(std::getenv("RMW_IMPLEMENTATION"))=="rmw_zenoh_cpp")std::cerr<<"Install the ROS Zenoh RMW package and source ROS, or set RMW_IMPLEMENTATION to another installed RMW.\n";return 1;}}
