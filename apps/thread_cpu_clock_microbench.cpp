// Standalone CPU diagnostic; this does not measure ROS, Daxa, drivers, or GPU work.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ns = std::chrono::nanoseconds;
static volatile std::uint64_t sink = 0;
struct Samples { std::vector<double> copy, publish, total, copy_cpu, publish_cpu; };
static std::uint64_t failures = 0;
static bool inject_failure = false;
static std::int64_t cpu_clock() {
    timespec t{};
    if (inject_failure || clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) != 0) {
        ++failures;
        return -1;
    }
    return static_cast<std::int64_t>(t.tv_sec) * 1000000000LL + t.tv_nsec;
}
static double elapsed(Clock::time_point a, Clock::time_point b) {
    return static_cast<double>(std::chrono::duration_cast<Ns>(b-a).count());
}
static void run(Samples& s, bool instrument, const std::vector<unsigned char>& src,
                std::vector<unsigned char>& dst, unsigned i) {
    const auto start = Clock::now();
    const auto c0 = instrument ? cpu_clock() : 0;
    std::memcpy(dst.data(), src.data(), src.size());
    const auto c1 = instrument ? cpu_clock() : 0;
    const auto copied = Clock::now();
    const auto p0 = instrument ? cpu_clock() : 0;
    // Stand-in only: prevents dead-code elimination and does not simulate ROS.
    sink = sink + dst[(i * 4099U) % dst.size()];
    const auto p1 = instrument ? cpu_clock() : 0;
    const auto done = Clock::now();
    s.copy.push_back(elapsed(start,copied));
    s.publish.push_back(elapsed(copied,done));
    s.total.push_back(elapsed(start,done));
    if (instrument && c0 >= 0 && c1 >= 0) s.copy_cpu.push_back(double(c1-c0));
    if (instrument && p0 >= 0 && p1 >= 0) s.publish_cpu.push_back(double(p1-p0));
}
static void stats(const char* name, std::vector<double> v) {
    std::sort(v.begin(), v.end());
    std::cout << '"' << name << "\":{";
    if (v.empty()) { std::cout << "\"count\":0,\"p50_ns\":null,\"p95_ns\":null,\"p99_ns\":null,\"max_ns\":null}"; return; }
    auto q = [&](unsigned p) { return v[(v.size()-1)*p/100]; };
    std::cout << "\"count\":" << v.size() << ",\"p50_ns\":" << q(50)
              << ",\"p95_ns\":" << q(95) << ",\"p99_ns\":" << q(99)
              << ",\"max_ns\":" << v.back() << '}';
}
static std::vector<double> deltas(const std::vector<double>& on, const std::vector<double>& off) {
    std::vector<double> result; result.reserve(on.size());
    for (std::size_t i=0;i<on.size();++i) result.push_back(on[i]-off[i]);
    return result;
}
int main(int argc, char** argv) {
    try {
        unsigned n = 2000;
        for (int i=1; i<argc; ++i) {
            std::string arg=argv[i];
            if (arg=="--inject-clock-failure") inject_failure=true;
            else if (arg=="--iterations" && i+1<argc) {
                std::string value=argv[++i]; std::size_t used=0;
                auto parsed=std::stoul(value,&used);
                if (used!=value.size() || parsed<2 || parsed>100000) throw std::runtime_error("iterations must be 2..100000");
                n=static_cast<unsigned>(parsed);
            } else throw std::runtime_error("usage: --iterations 2..100000 [--inject-clock-failure]");
        }
        constexpr std::size_t bytes=1280U*720U*4U;
        std::vector<unsigned char> src(bytes, 42), dst(bytes, 0);
        Samples off,on;
        for (auto* s : {&off,&on}) for (auto* v : {&s->copy,&s->publish,&s->total,&s->copy_cpu,&s->publish_cpu}) v->reserve(n);
        // Warm both paths before the paired, alternating-order measurement.
        Samples warm;
        for (auto* v : {&warm.copy,&warm.publish,&warm.total,&warm.copy_cpu,&warm.publish_cpu}) v->reserve(16);
        for (unsigned i=0;i<8;++i) { run(warm,false,src,dst,i); run(warm,true,src,dst,i); }
        failures=0;
        for (unsigned i=0;i<n;++i) {
            if (i%2==0) { run(off,false,src,dst,i); run(on,true,src,dst,i); }
            else { run(on,true,src,dst,i); run(off,false,src,dst,i); }
        }
        std::vector<double> clock_cost; clock_cost.reserve(n);
        for (unsigned i=0;i<n;++i) { auto a=Clock::now(); (void)cpu_clock(); auto b=Clock::now(); clock_cost.push_back(elapsed(a,b)); }
        std::cout << "{\"iterations\":" << n << ",\"bytes\":" << bytes << ",\"clock_failures\":" << failures << ',';
        stats("copy_wall_off",off.copy); std::cout << ','; stats("copy_wall_on",on.copy); std::cout << ',';
        stats("publish_wall_off",off.publish); std::cout << ','; stats("publish_wall_on",on.publish); std::cout << ',';
        stats("total_wall_off",off.total); std::cout << ','; stats("total_wall_on",on.total); std::cout << ',';
        stats("copy_cpu_on",on.copy_cpu); std::cout << ','; stats("publish_cpu_on",on.publish_cpu); std::cout << ',';
        stats("copy_wall_paired_delta",deltas(on.copy,off.copy)); std::cout << ',';
        stats("publish_wall_paired_delta",deltas(on.publish,off.publish)); std::cout << ',';
        stats("total_wall_paired_delta",deltas(on.total,off.total)); std::cout << ',';
        stats("clock_call_wall",clock_cost);
        std::cout << ",\"limitations\":\"CPU microbenchmark only; no ROS, Daxa, driver or GPU work; instrumentation perturbs runtime; paired deltas include noise and may be negative\"}\n";
        return failures ? 2 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
