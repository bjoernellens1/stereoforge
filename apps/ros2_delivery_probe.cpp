#include <stereoforge/ros_delivery_json.hpp>
#include <stereoforge/ros_depth_contract.hpp>
#include <stereoforge/ros_default_rmw.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmw/rmw.h>
#include <sensor_msgs/msg/image.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace stereoforge;
using nlohmann::json;
namespace {
using steady=std::chrono::steady_clock;
using system_clock=std::chrono::system_clock;
template<class Clock> std::uint64_t ns_now(){
    auto n=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    if(n<=0)throw std::runtime_error("clock unavailable");return static_cast<std::uint64_t>(n);
}
struct Options {
    std::string mode="depth",depth_topic="/stereoforge/depth",rgb_topic="/stereoforge/rectified_rgb";
    std::string frame_id,run_id,trace,reliability="best_effort";
    std::uint32_t width=1280,height=720,fps=60;
    unsigned max_frames=100000;double duration=600;
};
unsigned integer(const std::string& value,unsigned lo,unsigned hi){
    size_t used=0;auto n=std::stoull(value,&used);if(used!=value.size()||n<lo||n>hi)throw std::invalid_argument("integer option outside bounds");return static_cast<unsigned>(n);
}
Options parse(int argc,char** argv){
    Options o;
    for(int i=1;i<argc;++i){std::string a=argv[i];
        if(a=="--help"){std::cout<<"--run-id ID --frame-id ID --trace FILE [--mode depth|rgb|depth-rgb] [--width N --height N --fps N --duration-seconds N --max-frames N --depth-topic TOPIC --rgb-topic TOPIC --reliability best_effort|reliable]\n";std::exit(0);}
        if(i+1>=argc)throw std::invalid_argument("missing option value");std::string v=argv[++i];
        if(a=="--mode")o.mode=v;else if(a=="--depth-topic")o.depth_topic=v;else if(a=="--rgb-topic")o.rgb_topic=v;
        else if(a=="--frame-id")o.frame_id=v;else if(a=="--run-id")o.run_id=v;else if(a=="--trace")o.trace=v;
        else if(a=="--reliability")o.reliability=v;else if(a=="--width")o.width=integer(v,1,8192);
        else if(a=="--height")o.height=integer(v,1,8192);else if(a=="--fps")o.fps=integer(v,1,120);
        else if(a=="--max-frames")o.max_frames=integer(v,1,200000);
        else if(a=="--duration-seconds"){size_t used=0;o.duration=std::stod(v,&used);if(used!=v.size()||!std::isfinite(o.duration)||o.duration<=0||o.duration>3600)throw std::invalid_argument("invalid duration");}
        else throw std::invalid_argument("unknown option: "+a);
    }
    auto blank=[](const std::string& s){return s.empty()||std::all_of(s.begin(),s.end(),[](unsigned char c){return std::isspace(c)!=0;});};
    if((o.mode!="depth"&&o.mode!="rgb"&&o.mode!="depth-rgb")||
       (o.reliability!="best_effort"&&o.reliability!="reliable")||o.run_id.empty()||o.run_id.size()>4096||o.trace.empty()||
       blank(o.run_id)||blank(o.frame_id)||o.frame_id.size()>4096||
       o.depth_topic.empty()||o.rgb_topic.empty()||o.depth_topic.front()!='/'||o.rgb_topic.front()!='/'||
       (o.mode=="depth-rgb"&&o.depth_topic==o.rgb_topic))throw std::invalid_argument("invalid probe configuration");
    make_ros_depth_contract(o.width,o.height,o.frame_id);
    return o;
}
std::string clock_id(){
    std::ifstream f("/proc/sys/kernel/random/boot_id");std::string value;std::getline(f,value);
    if(!f||value.size()!=36)throw std::runtime_error("Linux boot identity unavailable");
    for(size_t i=0;i<value.size();++i){bool hyphen=i==8||i==13||i==18||i==23;
        if(hyphen?value[i]!='-':!((value[i]>='0'&&value[i]<='9')||(value[i]>='a'&&value[i]<='f')))throw std::runtime_error("invalid boot identity");}
    struct stat time_namespace{};
    if(::stat("/proc/self/ns/time",&time_namespace)!=0||!time_namespace.st_ino)throw std::runtime_error("time namespace identity unavailable");
    return value+":monotonic:"+std::to_string(static_cast<std::uint64_t>(time_namespace.st_ino));
}
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
        reset();put(json({{"schema","stereoforge.ros_delivery_failure.v1"},{"role","subscriber"},
                          {"kind","startup_or_output_failure"},{"message",safe}}).dump()+"\n");
    }
};
json descriptor(const std::string& topic,unsigned w,unsigned h,const std::string& encoding,const std::string& frame){
    return {{"topic",topic},{"width",w},{"height",h},{"step",w*4},{"encoding",encoding},{"frame_id",frame}};
}
void run(const Options& o, OutputFile& file){
    const bool depth=o.mode!="rgb",rgb=o.mode!="depth";
    json outputs=json::object();
    if(depth)outputs["depth"]=descriptor(o.depth_topic,o.width,o.height,"32FC1",o.frame_id);
    if(rgb)outputs["rgb"]=descriptor(o.rgb_topic,o.width,o.height,"rgba8",o.frame_id);
    json qos={{"history","keep_last"},{"depth",1},{"reliability",o.reliability},{"durability","volatile"}};
    json config={{"clock_id",clock_id()},{"nominal_hz",o.fps},{"outputs",outputs},{"qos",qos}};
    auto node=std::make_shared<rclcpp::Node>("stereoforge_delivery_probe",rclcpp::NodeOptions().use_intra_process_comms(false)
        .start_parameter_services(false).start_parameter_event_publisher(false));
    bool sim=false;if(node->has_parameter("use_sim_time"))sim=node->get_parameter("use_sim_time").as_bool();
    if(sim)throw std::invalid_argument("simulated time is unsupported");
    auto qos_profile=rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile();
    if(o.reliability=="reliable")qos_profile.reliable();else qos_profile.best_effort();
    const size_t capacity=static_cast<size_t>(o.max_frames)*(unsigned(depth)+unsigned(rgb))+2;
    RosDeliveryTrace trace(capacity);
    std::array<std::uint64_t,2> counts{};bool bad_message=false,trace_full=false;
    auto receive=[&](RosDeliveryOutput which,sensor_msgs::msg::Image::ConstSharedPtr image){
        auto at=ns_now<steady>();auto arrival=ns_now<system_clock>();
        auto index=which==RosDeliveryOutput::Depth?0u:1u;const char* encoding=index==0?"32FC1":"rgba8";
        const auto& stamp=image->header.stamp;
        if(image->width!=o.width||image->height!=o.height||image->step!=o.width*4||image->encoding!=encoding||
           image->is_bigendian!=0||image->header.frame_id!=o.frame_id||image->data.size()!=size_t(o.width)*o.height*4||
           stamp.sec<0||stamp.nanosec>=1'000'000'000u||(stamp.sec==0&&stamp.nanosec==0)) {bad_message=true;return;}
        RosDeliveryTraceRecord r{};r.kind=RosDeliveryTraceKind::Received;r.output=which;r.at_ns=at;
        r.mapped_capture_ns=std::uint64_t(stamp.sec)*1'000'000'000u+stamp.nanosec;r.arrival_system_ns=arrival;
        r.width=image->width;r.height=image->height;r.step=image->step;r.payload_bytes=image->data.size();
        if(!trace.append(r)){trace_full=true;return;}++counts[index];
    };
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr ds,rs;
    if(depth)ds=node->create_subscription<sensor_msgs::msg::Image>(o.depth_topic,qos_profile,[&](sensor_msgs::msg::Image::ConstSharedPtr msg){receive(RosDeliveryOutput::Depth,std::move(msg));});
    if(rgb)rs=node->create_subscription<sensor_msgs::msg::Image>(o.rgb_topic,qos_profile,[&](sensor_msgs::msg::Image::ConstSharedPtr msg){receive(RosDeliveryOutput::Rgb,std::move(msg));});
    rclcpp::executors::SingleThreadedExecutor executor;executor.add_node(node);
    auto start=ns_now<steady>();auto deadline=start+static_cast<std::uint64_t>(o.duration*1e9);
    while(rclcpp::ok()&&!bad_message&&!trace_full&&ns_now<steady>()<deadline&&
          !((!depth||counts[0]>=o.max_frames)&&(!rgb||counts[1]>=o.max_frames)))executor.spin_once(std::chrono::milliseconds(10));
    auto end=ns_now<steady>();
    std::string reason=bad_message?"error":trace_full?"trace_full":!rclcpp::ok()?"shutdown":"limit";
    json counters={{"read",0},{"admitted",0},{"ring_drops",0},{"identity_rejects",0},{"published_depth",0},{"published_rgb",0},
                   {"received_depth",counts[0]},{"received_rgb",counts[1]},{"timeouts",0}};
    json first={{"schema","stereoforge.ros_delivery_event.v1"},{"kind","start"},{"run_id",o.run_id},{"role","subscriber"},{"at_ns",start},{"config",config}};
    json last={{"schema","stereoforge.ros_delivery_event.v1"},{"kind","stop"},{"run_id",o.run_id},{"role","subscriber"},{"at_ns",end},
        {"window_end_ns",end},{"reason",reason},{"counters",counters},{"capture_counters",nullptr},{"queued",0},{"pending",0},
        {"trace_drops",trace.dropped()},{"drain_ns",0}};
    file.reset();file.put("[");file.put(first.dump());for(auto const& r:trace.records()){file.put(",");file.put(ros_delivery_event_json(r,o.run_id,"subscriber",config).dump());}file.put(",");file.put(last.dump());file.put("]\n");file.complete=true;
    executor.remove_node(node);if(bad_message||trace_full)throw std::runtime_error("subscriber trace incomplete: "+reason);
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
