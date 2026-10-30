#include <stereoforge/depth_rate_controller.hpp>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
using namespace stereoforge;
namespace {
template <class F> bool throws(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
DepthRateConfig fixed(std::uint32_t fps) { DepthRateConfig c; c.mode = DepthRateMode::Fixed; c.fixed_fps = fps; return c; }
void fixed_rate_test(std::uint32_t fps) {
    DepthRateController control(fixed(fps));
    // Exact 60 Hz nominal cadence: 600 captured frames (ten seconds).
    for (std::uint64_t i = 0; i < 600; ++i) {
        const auto t = 1'000'000'000ull + i * 1'000'000'000ull / 60;
        auto result = control.observe(t, std::nullopt);
        if (result.due) control.admitted(t);
    }
    assert(control.stats().observed == 600);
    assert(control.stats().admitted >= 10*fps-1 && control.stats().admitted <= 10*fps+1);
    assert(control.stats().intentional_skips + control.stats().admitted == 600);
    assert(control.stats().effective_fps == fps);
}
void adaptive_test() {
    DepthRateConfig c; c.downshift_dwell_ns = 1'000'000'000ull;
    DepthRateController controller(c);
    std::uint64_t first = 1'000'000'000ull;
    for (std::uint64_t i = 0; i < 370; ++i) {
        auto t = first + i*1'000'000'000ull/60;
        auto result = controller.observe(t, 0.f);
        if (result.due) controller.admitted(t);
    }
    assert(controller.stats().effective_fps == 5);
    assert(controller.stats().downshifts == 4);
    const auto n = first + 370*1'000'000'000ull/60;
    auto movement = controller.observe(n, .4f);
    assert(movement.due && movement.target_fps == 60);
    controller.admitted(n);
    assert(controller.stats().upshifts == 1);
    auto unknown = controller.observe(n + 17'000'000ull, std::nullopt);
    assert(unknown.target_fps == 60);
    controller.admitted(n + 17'000'000ull);
    assert(controller.stats().unavailable_motion == 1);
}
void monotonic_and_busy_test() {
    auto c = fixed(10); DepthRateController controller(c);
    assert(throws([&] { controller.observe(0, 0.f); }));
    auto a = controller.observe(1'000'000'000ull, 0.f);
    assert(a.due);
    assert(throws([&] { controller.observe(1'010'000'000ull, 0.f); }));
    controller.reject_busy();
    auto b = controller.observe(1'010'000'000ull, 0.f);
    assert(b.due); controller.admitted(1'010'000'000ull);
    assert(throws([&] { controller.admitted(1'010'000'000ull); }));
    assert(throws([&] { controller.observe(1'010'000'000ull, 0.f); }));
    assert(!controller.observe(1'030'000'000ull, 0.f).due);
    assert(controller.stats().slot_rejections == 1);
    assert(controller.stats().admitted == 1);
}
void config_test() {
    assert(throws([&]{ auto c = fixed(7); DepthRateController ctl(c); }));
    assert(throws([&]{ auto c = fixed(5); c.consumer_min_fps=20; DepthRateController ctl(c); }));
    assert(throws([&]{ auto c = fixed(5); c.max_depth_age_ns=100'000'000; DepthRateController ctl(c); }));
    assert(throws([&]{ auto c = fixed(60); c.capture_fps=30; DepthRateController ctl(c); }));
    { auto c = fixed(10); c.capture_fps=30; DepthRateController ctl(c); assert(ctl.stats().effective_fps==10); }
    assert(throws([&]{ auto c = DepthRateConfig{}; c.downshift_dwell_ns=0; DepthRateController ctl(c); }));
    DepthRateConfig c; c.consumer_min_fps=20; c.downshift_dwell_ns=10'000'000ull;
    DepthRateController ctl(c);
    for (std::uint64_t i=0;i<30;i++) {
        auto t=1'000'000'000ull+i*1'000'000'000ull/60;
        auto d=ctl.observe(t,0.f); if(d.due) ctl.admitted(t);
    }
    assert(ctl.stats().effective_fps==20);
}
void scene_test() {
    YuyvSceneMonitor scene;
    constexpr std::uint32_t w=128, h=72, stride=w*2;
    std::vector<std::uint8_t> bytes(std::size_t(stride)*h, 0);
    for (std::size_t i=0;i<bytes.size();i+=2) bytes[i]=100;
    assert(!scene.observe(bytes,w,h,stride));
    scene.depth_admitted();
    assert(scene.observe(bytes,w,h,stride).value()==0.f);
    // Global camera exposure compensation: small uniform shift is ignored.
    for (std::size_t i=0;i<bytes.size();i+=2) bytes[i]=107;
    assert(scene.observe(bytes,w,h,stride).value()==0.f);
    // Movement affecting left eye (but not right) is detected.
    for (std::uint32_t y=0;y<h;++y)
        for (std::uint32_t x=0;x<w/2;++x) bytes[std::size_t(y)*stride+2*x]=170;
    assert(scene.observe(bytes,w,h,stride).value()>0.25f);
    assert(!scene.observe(std::span<const std::uint8_t>{},w,h,stride));
    assert(!scene.observe(bytes,w,h,stride));
}
}
int main() {
    for (auto fps : depth_rates) fixed_rate_test(fps);
    adaptive_test(); monotonic_and_busy_test(); config_test(); scene_test();
}
