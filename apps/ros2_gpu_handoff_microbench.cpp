// Representative CPU-only ROS intra-process adapted-frame handoff. No GPU data.
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
thread_local bool counting = false;
thread_local std::uint64_t allocations = 0;
void* allocate(std::size_t n) { if (counting) ++allocations; if (auto p = std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void* aligned_allocate(std::size_t n, std::size_t alignment) { if (counting) ++allocations; void* p = nullptr; if (posix_memalign(&p, alignment, n ? n : 1) == 0) return p; throw std::bad_alloc(); }
}
void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return aligned_allocate(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return aligned_allocate(n, static_cast<std::size_t>(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

struct RepresentativeLease { std::uint64_t sequence = 0; std::uint32_t width = 0, height = 0; };
struct RepresentativeFrame { std::shared_ptr<RepresentativeLease> lease; std_msgs::msg::Header header; };
namespace rclcpp {
template<> struct TypeAdapter<RepresentativeFrame, std_msgs::msg::Header> {
    using is_specialized = std::true_type;
    using custom_type = RepresentativeFrame;
    using ros_message_type = std_msgs::msg::Header;
    static void convert_to_ros_message(const custom_type& source, ros_message_type& destination) { destination = source.header; }
    static void convert_to_custom(const ros_message_type&, custom_type&) { throw std::runtime_error("metadata cannot reconstruct a retained lease"); }
};
}
namespace {
using Clock = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;
struct Sample { std::int64_t construct, publish, executor, total; std::uint64_t allocs; };
std::int64_t elapsed(Clock::time_point a, Clock::time_point b) { return std::chrono::duration_cast<ns>(b-a).count(); }
void stats(const char* name, std::vector<std::int64_t> values) {
    std::sort(values.begin(),values.end());
    auto p=[&](std::size_t n) { return values[((values.size()-1)*n+99)/100]/1e6; };
    std::cout << name << "_ms p50=" << p(50) << " p95=" << p(95) << " p99=" << p(99) << " max=" << values.back()/1e6 << '\n';
}
}
int main(int argc, char** argv) {
    try {
        std::size_t frames=1000, warmup=100, width=1280, height=720;
        if(argc>5) throw std::invalid_argument("usage: stereoforge_ros2_gpu_handoff_microbench [frames [warmup [width [height]]]]");
        if(argc>1) frames=std::stoul(argv[1]); if(argc>2) warmup=std::stoul(argv[2]);
        if(argc>3) width=std::stoul(argv[3]); if(argc>4) height=std::stoul(argv[4]);
        if(!frames || frames>1000000 || warmup>1000000 || !width || width>8192 || !height || height>8192) throw std::invalid_argument("frame counts or dimensions out of range");
        rclcpp::init(0,nullptr);
        auto opts=rclcpp::NodeOptions().use_intra_process_comms(true).start_parameter_services(false).start_parameter_event_publisher(false);
        auto pubnode=std::make_shared<rclcpp::Node>("stereoforge_handoff_bench_pub",opts);
        auto subnode=std::make_shared<rclcpp::Node>("stereoforge_handoff_bench_sub",opts);
        using Adapted=rclcpp::TypeAdapter<RepresentativeFrame,std_msgs::msg::Header>;
        auto qos=rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
        auto pub=pubnode->create_publisher<Adapted>("/stereoforge_handoff_microbench",qos);
        std::uint64_t received=0, expected=0;
        auto sub=subnode->create_subscription<Adapted>("/stereoforge_handoff_microbench",qos,[&](std::shared_ptr<const RepresentativeFrame> frame) {
            if(!frame || !frame->lease || frame->lease->sequence!=expected) throw std::runtime_error("incorrect callback lease or sequence");
            ++received;
        });
        rclcpp::executors::SingleThreadedExecutor executor; executor.add_node(pubnode); executor.add_node(subnode);
        auto lease=std::make_shared<RepresentativeLease>(); lease->width=width; lease->height=height;
        std::vector<Sample> samples(frames);
        auto once=[&](Sample& sample) {
            auto before=allocations; counting=true; auto t0=Clock::now();
            lease->sequence=++expected;
            std_msgs::msg::Header header; header.frame_id="stereoforge"; header.stamp.sec=1;
            auto message=std::make_unique<RepresentativeFrame>(RepresentativeFrame{lease,std::move(header)});
            auto t1=Clock::now(); auto previous=received; pub->publish(std::move(message)); auto t2=Clock::now();
            executor.spin_some(); auto t3=Clock::now(); counting=false;
            if(received!=previous+1) throw std::runtime_error("callback was not received exactly once");
            sample={elapsed(t0,t1),elapsed(t1,t2),elapsed(t2,t3),elapsed(t0,t3),allocations-before};
        };
        Sample first{}; once(first);
        Sample discard{}; for(std::size_t i=0;i<warmup;++i) once(discard);
        for(auto& s:samples) once(s);
        std::cout<<std::fixed<<std::setprecision(6)<<"frames="<<frames<<" warmup="<<warmup<<" metadata_resolution="<<width<<"x"<<height<<" qos=best_effort,keep_last_1 callbacks="<<received<<'\n';
        std::cout<<"first_ms construct="<<first.construct/1e6<<" publish="<<first.publish/1e6<<" executor_callback="<<first.executor/1e6<<" total="<<first.total/1e6<<" cpp_new_calls="<<first.allocs<<'\n';
        for(auto member:{&Sample::construct,&Sample::publish,&Sample::executor,&Sample::total}) {
            std::vector<std::int64_t> v; v.reserve(frames); for(auto const& s:samples) v.push_back(s.*member);
            auto name=member==&Sample::construct?"construct":member==&Sample::publish?"publish":member==&Sample::executor?"executor_callback":"total";
            stats(name,std::move(v));
        }
        std::uint64_t total=0,maximum=0; for(auto const& s:samples){total+=s.allocs;maximum=std::max(maximum,s.allocs);}
        std::cout<<"steady_cpp_new_calls total="<<total<<" per_frame="<<double(total)/frames<<" max_per_frame="<<maximum<<'\n';
        std::cout<<"scope=representative retained CPU lease and Header, rclcpp intra-process publish, single-threaded executor and callback; no pixels, Daxa, Vulkan, GPU or capture; C++ new on benchmark thread only; excludes malloc and other threads. Instrumentation included in timings. Not GPU or end-to-end performance.\n";
        executor.remove_node(subnode); executor.remove_node(pubnode); rclcpp::shutdown(); return 0;
    } catch(const std::exception& e) { counting=false; if(rclcpp::ok()) rclcpp::shutdown(); std::cerr<<e.what()<<'\n'; return 1; }
}
