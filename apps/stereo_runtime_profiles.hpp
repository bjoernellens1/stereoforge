#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace stereoforge::runtime_profile {
inline bool named(std::string_view name){return name=="quality-reference"||name=="live-quality";}
// Named recipes fix these controls. Custom/legacy profiles retain their CLI behavior.
inline bool controlled(std::string_view option){
    constexpr std::string_view controls[]={"--disparities","--stereo-variant","--coarse-manifest",
        "--descriptor-mode","--coarse-schedule","--coarse-right-reference-mode","--fast-mode","--ultra-fast",
        "--refinement-subpixel","--continuous-subpixel","--adaptive-tiles","--min-disparity-bound","--max-disparity-bound",
        "--selective-repair","--planar-recovery","--guided-fill-px","--gpu-duty-budget-pct","--capture-fps","--fps",
        "--depth-rate","--min-depth-fps","--max-depth-fps","--consumer-min-depth-fps","--max-depth-age-ms","--downshift-ms",
        "--depth-speckle-min-pixels","--depth-speckle-delta-m","--depth-speckle-delta-px"};
    for(auto control:controls)if(control==option)return true;
    return false;
}
struct Geometry {std::uint32_t disparities,slots,fps,margin_px;};
inline Geometry geometry(std::string_view profile,std::uint32_t original_width,std::uint32_t original_height,
                         std::uint32_t width,std::uint32_t height,double fx_baseline,double cx_offset){
    const bool native720=width==1280&&height==720,native1080=width==1920&&height==1080;
    if(!named(profile)||original_width!=width||original_height!=height||
       (!native720&&!native1080)||(profile=="live-quality"&&!native720))
        throw std::invalid_argument("named profile requires its native original and rectified resolution");
    if(!std::isfinite(fx_baseline)||fx_baseline<=0||!std::isfinite(cx_offset))
        throw std::invalid_argument("named profile requires finite positive metric calibration");
    Geometry out=profile=="live-quality"?Geometry{128,2,60,0}:
                 native720?Geometry{256,2,60,16}:Geometry{288,1,30,16};
    const auto near_disparity=fx_baseline/0.5+cx_offset,far_disparity=fx_baseline/3.0+cx_offset;
    if(!std::isfinite(near_disparity)||far_disparity<0||near_disparity+out.margin_px>out.disparities-1)
        throw std::invalid_argument("named profile search cannot cover calibrated 0.5-3m hypotheses and margin");
    return out;
}
}
