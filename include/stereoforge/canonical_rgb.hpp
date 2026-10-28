#pragma once
#include <stereoforge/capture.hpp>
#include <span>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
namespace stereoforge {
inline constexpr const char* canonical_rgb_sampling_profile="canonical_rgb8_box2x2_halfup_gray15_support_and4_halfpixel_v1";
enum class CanonicalRgbFormat { Rgb8,SupportU8 };
struct CanonicalRgbPlane {
    std::span<const uint8_t> bytes;
    uint32_t width=0,height=0,stride_bytes=0;
    CanonicalRgbFormat format=CanonicalRgbFormat::Rgb8;
};
// Immutable after publication through shared_ptr<const>. Spans retain their
// portable-record storage through backing; RGB is interleaved R,G,B, masks U8.
struct CanonicalRgbSource {
    std::array<CanonicalRgbPlane,4> rgb;
    std::array<CanonicalRgbPlane,2> support;
    std::shared_ptr<const void> backing;
    std::string input_identity,original_pixel_identity,rectified_pixel_identity;
    std::string original_calibration_identity,rectified_calibration_identity;
    std::string coarse_rectified_calibration_identity;
    std::string sampling_profile=canonical_rgb_sampling_profile;
};
inline void validate_canonical_rgb_source(const CaptureFrame& f,bool scan_support=false){
    auto require=[](bool v,const char* m){if(!v)throw std::invalid_argument(m);};
    require(f.source_kind==CaptureSourceKind::CanonicalRgbMetadata&&bool(f.canonical_rgb),"canonical RGB sidecar required");
    auto const& s=*f.canonical_rgb;
    require(bool(s.backing)&&f.stride_bytes==0&&f.packed_yuyv.empty(),"canonical source must retain backing and metadata-only layout");
    f.require_calibration();
    require(f.frame_id<=uint64_t(INT64_MAX)&&f.camera_timestamp_ns&&!f.camera_clock_domain.empty()&&!f.host_clock_domain.empty()&&f.camera_clock_domain!="unknown","canonical source identity/time unavailable");
    require(!s.input_identity.empty()&&!s.original_pixel_identity.empty()&&!s.rectified_pixel_identity.empty()&&!s.rectified_calibration_identity.empty()&&s.original_calibration_identity==f.calibration_identity&&s.sampling_profile==canonical_rgb_sampling_profile,"canonical source provenance mismatch");
    auto plane=[&](const CanonicalRgbPlane& p,unsigned channels){require(p.format==(channels==3?CanonicalRgbFormat::Rgb8:CanonicalRgbFormat::SupportU8),"canonical channel format mismatch");require(p.width&&p.height&&p.width<32767&&p.height<32767&&uint64_t(p.stride_bytes)>=uint64_t(p.width)*channels,"canonical plane shape/stride invalid");require(p.bytes.data()&&p.bytes.size()>=uint64_t(p.height-1)*p.stride_bytes+uint64_t(p.width)*channels,"canonical plane extent invalid");};
    for(auto const& p:s.rgb)plane(p,3);for(auto const& p:s.support)plane(p,1);
    require(f.source_width==uint64_t(s.rgb[0].width)*2&&f.height==s.rgb[0].height,"canonical logical original shape mismatch");
    require(s.rgb[1].width==s.rgb[0].width&&s.rgb[1].height==s.rgb[0].height,"canonical original eyes mismatch");
    for(unsigned e=0;e<2;++e){require(s.rgb[2+e].width==s.rgb[2].width&&s.rgb[2+e].height==s.rgb[2].height&&s.support[e].width==s.rgb[2].width&&s.support[e].height==s.rgb[2].height,"canonical rectified/support shape mismatch");if(scan_support)for(unsigned y=0;y<s.support[e].height;++y)for(unsigned x=0;x<s.support[e].width;++x)require(s.support[e].bytes[size_t(y)*s.support[e].stride_bytes+x]<=1,"canonical support must be binary");}
}
// Offline admission capability. Construct/cache this once per immutable record,
// before the resident loop. Copies preserve admission; assigning a different
// public source never admits that replacement without its own deep validation.
class RectifiedRgbFrame {
    CaptureLease admitted_source_;
public:
    CaptureLease source;
    RectifiedRgbFrame()=default;
    explicit RectifiedRgbFrame(CaptureLease input):source(std::move(input)){
        if(!source)throw std::invalid_argument("canonical source required for offline admission");
        validate_canonical_rgb_source(*source,true);
        admitted_source_=source;
    }
    bool admitted()const noexcept{return source&&source==admitted_source_;}
};
// Metadata-only association, including coarse observations. No pixel scan,
// hashing, matrix traversal or raw YUYV layout fiction in this helper.
inline bool canonical_rgb_grid_association(const CaptureFrame& f,uint32_t eye_width,uint32_t source_height,uint32_t width,uint32_t height,const std::string& original_identity,const std::string& rectified_identity){
    if(f.source_kind!=CaptureSourceKind::CanonicalRgbMetadata||!f.canonical_rgb)return false;
    auto const& s=*f.canonical_rgb;
    return ((s.rectified_calibration_identity==rectified_identity&&s.rgb[2].width==width&&s.rgb[2].height==height)||(s.coarse_rectified_calibration_identity==rectified_identity&&uint64_t(width)*2==s.rgb[2].width&&uint64_t(height)*2==s.rgb[2].height))&&s.backing&&f.stride_bytes==0&&f.packed_yuyv.empty()&&f.source_width==2*eye_width&&f.height==source_height&&f.calibration_identity==original_identity&&s.original_calibration_identity==original_identity&&s.rgb[0].width==eye_width&&s.rgb[0].height==source_height&&s.rgb[1].width==eye_width&&s.rgb[1].height==source_height;
}
}
