#include <stereoforge/adaptive_camera_policy.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace stereoforge;
static void require(bool ok) {
    if (!ok) throw std::runtime_error("adaptive camera policy CPU contract failed");
}
static std::vector<std::uint8_t> frame(unsigned width, unsigned height,
                                       std::uint8_t base, bool mixed=false) {
    std::vector<std::uint8_t> b(std::size_t(width)*height*2, 128);
    for (unsigned y=0;y<height;++y) for (unsigned x=0;x<width;++x) {
        auto lum = mixed ? (x%(width/2) < width/4 ? 10 : 253) : base;
        b[std::size_t(y)*width*2+2*x]=lum;
    }
    return b;
}
int main() {
    const auto hd = camera_mode_spec(CameraCaptureMode::HD720_60);
    require(hd.width==1280 && hd.height==720 && hd.fps==60 && hd.sensor_binning);
    const auto fhd = camera_mode_spec(CameraCaptureMode::HD1080_30);
    require(fhd.width==1920 && fhd.height==1080 && fhd.fps==30 && !fhd.sensor_binning);
    StereoYuyvLightSampler sampler;
    auto dark=frame(128,72,18), bright=frame(128,72,130), contrasted=frame(128,72,100,true);
    auto d=sampler.sample(dark,128,72,256), n=sampler.sample(bright,128,72,256);
    auto h=sampler.sample(contrasted,128,72,256);
    require(d && n && h);
    require(d->median_y==18 && d->dark_fraction==1 && d->saturated_fraction==0);
    require(n->median_y==130 && n->dark_fraction==0 && n->saturated_fraction==0);
    require(AdaptiveCameraModePolicy::classify(*d)==LightingClass::LowLight);
    require(AdaptiveCameraModePolicy::classify(*n)==LightingClass::Normal);
    require(AdaptiveCameraModePolicy::classify(*h)==LightingClass::HighContrast);
    require(!sampler.sample(dark,128,72,255));
    require(!sampler.sample(std::span<const std::uint8_t>{},128,72,256));

    AdaptiveCameraModePolicy policy(CameraCaptureMode::HD1080_30,
                                    {3'000'000'000ull,20'000'000'000ull,.15f});
    CameraModeObservation obs{};
    obs.timestamp_ns=1'000'000'000ull; obs.light=d; obs.motion_score=.02f;
    obs.task=CameraTask::Detail; obs.safe_to_interrupt_capture=false;
    auto first=policy.observe(obs);
    require(first.desired_mode==CameraCaptureMode::HD720_30 && !first.reopen_requested);
    obs.timestamp_ns=4'000'000'000ull;
    auto blocked=policy.observe(obs);
    require(!blocked.reopen_requested && blocked.blocked_by_safety);
    obs.timestamp_ns=5'000'000'000ull; obs.safe_to_interrupt_capture=true;
    auto switch_request=policy.observe(obs);
    require(switch_request.reopen_requested && switch_request.desired_mode==CameraCaptureMode::HD720_30);
    require(policy.current_mode()==CameraCaptureMode::HD1080_30);
    policy.abort(6'000'000'000ull);
    require(policy.current_mode()==CameraCaptureMode::HD1080_30);
    obs.timestamp_ns=7'000'000'000ull; policy.observe(obs);
    obs.timestamp_ns=11'000'000'000ull; require(!policy.observe(obs).reopen_requested);
    obs.timestamp_ns=27'000'000'000ull; require(policy.observe(obs).reopen_requested);
    policy.commit(CameraCaptureMode::HD720_30,27'100'000'000ull);
    require(policy.current_mode()==CameraCaptureMode::HD720_30);
    obs.light=n; obs.motion_score=.01f; obs.task=CameraTask::Detail;
    obs.timestamp_ns=28'000'000'000ull; require(!policy.observe(obs).reopen_requested);
    obs.timestamp_ns=35'000'000'000ull; require(!policy.observe(obs).reopen_requested);
    obs.timestamp_ns=49'000'000'000ull; require(policy.observe(obs).reopen_requested);
    policy.commit(CameraCaptureMode::HD1080_30,49'000'000'000ull);
    require(policy.current_mode()==CameraCaptureMode::HD1080_30);
    obs.light=h; obs.timestamp_ns=50'000'000'000ull;
    auto contrast=policy.observe(obs);
    require(contrast.image_treatment==CameraImageTreatment::HighContrastPreview);
    require(contrast.desired_mode==CameraCaptureMode::HD1080_30);
    obs.light=d; obs.motion_score=.5f; obs.timestamp_ns=51'000'000'000ull;
    auto motion=policy.observe(obs);
    require(motion.desired_mode==CameraCaptureMode::HD720_60 && !motion.reopen_requested);
    obs.timestamp_ns=52'000'000'000ull; obs.light={}; obs.motion_score={};
    auto unknown=policy.observe(obs);
    require(unknown.desired_mode==CameraCaptureMode::HD1080_30);
    bool rejected=false;
    try {policy.commit(CameraCaptureMode::HD720_60,52'000'000'000ull);}
    catch(const std::logic_error&) {rejected=true;}
    require(rejected);
    rejected=false;
    try {obs.timestamp_ns=52'000'000'000ull;policy.observe(obs);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected);
    return 0;
}
