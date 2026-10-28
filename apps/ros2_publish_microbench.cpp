// CPU-only measurement of the explicit ROS publication boundary. No GPU or consumer.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
std::atomic<bool> counting{false};
std::atomic<std::uint64_t> allocations{0};
void count_alloc() noexcept { if (counting.load(std::memory_order_relaxed)) allocations.fetch_add(1, std::memory_order_relaxed); }
void* allocate(std::size_t n) { count_alloc(); if (auto p = std::malloc(n ? n : 1)) return p; throw std::bad_alloc(); }
void* allocate_aligned(std::size_t n, std::size_t alignment) { count_alloc(); void* p = nullptr; if (posix_memalign(&p, alignment, n ? n : 1) == 0) return p; throw std::bad_alloc(); }
}
void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return allocate_aligned(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocate_aligned(n, static_cast<std::size_t>(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {
using clock_type = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;
struct Sample { std::int64_t depth_copy, depth_publish, rgb_copy, rgb_publish, total; std::uint64_t allocations; };
std::int64_t elapsed(clock_type::time_point a, clock_type::time_point b) { return std::chrono::duration_cast<ns>(b-a).count(); }
void setup(sensor_msgs::msg::Image& image, const char* encoding) {
    image.header.frame_id = "stereoforge";
    image.width = 1280; image.height = 720; image.encoding = encoding;
    image.is_bigendian = false; image.step = image.width * 4;
    image.data.resize(static_cast<std::size_t>(image.step) * image.height);
}
void stats(const char* name, std::vector<std::int64_t> values) {
    std::sort(values.begin(), values.end());
    auto percentile = [&](std::size_t percent) { return values[((values.size()-1)*percent+99)/100] / 1e6; };
    std::cout << name << "_ms p50=" << percentile(50) << " p95=" << percentile(95) << " p99=" << percentile(99) << '\n';
}
}
int main(int argc, char** argv) {
    try {
        std::size_t frames = 300, warmup = 30;
        if (argc > 3) throw std::invalid_argument("usage: stereoforge_ros2_publish_microbench [frames [warmup]]");
        if (argc > 1) frames = std::stoul(argv[1]);
        if (argc > 2) warmup = std::stoul(argv[2]);
        if (frames < 1 || frames > 1000000 || warmup > 1000000) throw std::invalid_argument("frame counts out of range");
        rclcpp::init(0, nullptr);
        auto node = std::make_shared<rclcpp::Node>("stereoforge_publish_microbench", rclcpp::NodeOptions().use_intra_process_comms(false).start_parameter_services(false).start_parameter_event_publisher(false));
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
        auto depth_pub = node->create_publisher<sensor_msgs::msg::Image>("/stereoforge_microbench/depth", qos);
        auto rgb_pub = node->create_publisher<sensor_msgs::msg::Image>("/stereoforge_microbench/rgb", qos);
        sensor_msgs::msg::Image depth, rgb; setup(depth, "32FC1"); setup(rgb, "rgba8");
        std::vector<std::uint8_t> depth_source(depth.data.size(), 0x42), rgb_source(rgb.data.size(), 0x71);
        std::vector<Sample> samples(frames);
        auto once = [&](Sample& s) {
            auto t0=clock_type::now(); std::memcpy(depth.data.data(),depth_source.data(),depth_source.size()); auto t1=clock_type::now();
            depth_pub->publish(depth); auto t2=clock_type::now(); std::memcpy(rgb.data.data(),rgb_source.data(),rgb_source.size()); auto t3=clock_type::now();
            rgb_pub->publish(rgb); auto t4=clock_type::now();
            s={elapsed(t0,t1),elapsed(t1,t2),elapsed(t2,t3),elapsed(t3,t4),elapsed(t0,t4),0};
        };
        Sample discarded{}; for (std::size_t i=0; i<warmup; ++i) once(discarded);
        if (depth_pub->get_subscription_count() || rgb_pub->get_subscription_count()) throw std::runtime_error("benchmark requires no subscribers");
        for (auto& s : samples) { auto before=allocations.load(std::memory_order_relaxed); counting.store(true,std::memory_order_relaxed); once(s); counting.store(false,std::memory_order_relaxed); s.allocations=allocations.load(std::memory_order_relaxed)-before; }
        if (depth_pub->get_subscription_count() || rgb_pub->get_subscription_count()) throw std::runtime_error("subscriber connected during benchmark");
        std::cout << std::fixed << std::setprecision(4) << "frames=" << frames << " warmup=" << warmup << " resolution=1280x720 outputs=depth32f,rgba8 qos=best_effort,keep_last_1\n";
        for (auto member : {&Sample::depth_copy,&Sample::depth_publish,&Sample::rgb_copy,&Sample::rgb_publish,&Sample::total}) {
            std::vector<std::int64_t> v; v.reserve(frames); for (auto const& s:samples) v.push_back(s.*member);
            const char* name = member==&Sample::depth_copy ? "depth_copy" : member==&Sample::depth_publish ? "depth_publish" : member==&Sample::rgb_copy ? "rgb_copy" : member==&Sample::rgb_publish ? "rgb_publish" : "combined";
            stats(name,std::move(v));
        }
        std::uint64_t total=0, maximum=0; for(auto const& s:samples){total+=s.allocations;maximum=std::max(maximum,s.allocations);}
        std::cout << "cpp_new_calls total=" << total << " per_frame=" << static_cast<double>(total)/frames << " max_per_frame=" << maximum << '\n';
        std::cout << "scope=CPU memcpy and synchronous rclcpp publish calls only; no GPU, export, consumer, end-to-end latency or throughput. C++ new counts exclude malloc and other processes. Timing includes allocation instrumentation.\n";
        rclcpp::shutdown(); return 0;
    } catch (const std::exception& e) { if(rclcpp::ok()) rclcpp::shutdown(); std::cerr << e.what() << '\n'; return 1; }
}
