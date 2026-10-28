#pragma once
// Executable publication metadata contracts; no ROS SDK or publisher dependency.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace stereoforge {
namespace ros_contract_detail {
inline void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}
inline bool known(std::string_view label) {
    return !label.empty() && label != "unknown";
}
}
struct RosCaptureClockMapping {
    std::string source_clock;
    int64_t offset_ns=0;
    uint64_t valid_from_ns=0, valid_until_ns=0, max_error_ns=0;
};
struct RosCaptureStamp { int32_t sec=0; uint32_t nanosec=0; uint64_t max_error_ns=0; };
// Offset-only mapping to ROS system time, never sim time. Caller supplies a
// measured bounded mapping valid for this run; publish time is not capture time.
inline RosCaptureStamp map_ros_capture_stamp(const RosCaptureClockMapping& mapping,
                                             std::string_view source_clock,
                                             uint64_t source_ns) {
    using ros_contract_detail::require;
    require(ros_contract_detail::known(source_clock) && ros_contract_detail::known(mapping.source_clock)
            && source_clock==mapping.source_clock,"unknown or mismatched capture clock");
    require(mapping.valid_from_ns>0 && mapping.valid_until_ns>0
            && mapping.valid_from_ns<=mapping.valid_until_ns,"invalid mapping interval");
    require(source_ns>=mapping.valid_from_ns && source_ns<=mapping.valid_until_ns,"capture outside mapping interval");
    uint64_t mapped;
    if (mapping.offset_ns>=0) {
        const auto offset=static_cast<uint64_t>(mapping.offset_ns);
        require(source_ns<=UINT64_MAX-offset,"capture timestamp overflow");
        mapped=source_ns+offset;
    } else {
        // Avoid negating INT64_MIN in signed arithmetic.
        const auto magnitude=static_cast<uint64_t>(-(mapping.offset_ns+1))+1;
        require(source_ns>=magnitude,"negative mapped capture timestamp");
        mapped=source_ns-magnitude;
    }
    constexpr uint64_t billion=1000000000;
    require(mapped/billion<=static_cast<uint64_t>(INT32_MAX),"ROS seconds overflow");
    return {static_cast<int32_t>(mapped/billion),static_cast<uint32_t>(mapped%billion),mapping.max_error_ns};
}
struct RosDepthImageContract {
    uint32_t width=0,height=0,step=0;
    std::string encoding,frame_id;
    bool is_bigendian=false;
};
using RosRgbaImageContract=RosDepthImageContract;
namespace ros_contract_detail {
inline RosDepthImageContract image(uint32_t width,uint32_t height,std::string_view frame_id,
                                    std::string_view encoding) {
    require(width>0 && height>0 && width<=8192 && height<=8192,"invalid image dimensions");
    const uint64_t step=uint64_t(width)*4;
    require(step<=UINT32_MAX && step*height<=512ULL*1024*1024,"image byte extent overflow");
    require(known(frame_id),"unknown optical frame label");
    return {width,height,static_cast<uint32_t>(step),std::string(encoding),std::string(frame_id),false};
}
inline void payload(const RosDepthImageContract& contract,std::span<const std::byte> bytes,
                    std::string_view encoding) {
    require(contract.width>0 && contract.height>0 && contract.width<=8192 && contract.height<=8192,
            "invalid image dimensions");
    const uint64_t step=uint64_t(contract.width)*4;
    const uint64_t extent=step*contract.height;
    require(known(contract.frame_id) && extent<=512ULL*1024*1024,"invalid image extent or frame label");
    require(contract.encoding==encoding && !contract.is_bigendian && contract.step==step,
            "invalid image layout");
    require(bytes.size()==extent,"incorrect payload extent");
}
}
// frame_id is the caller's declared optical coordinate frame label, stable
// across captures (never append per-frame source IDs). A string cannot prove TF
// extrinsics or optical-axis conventions; those require separate qualification.
inline RosDepthImageContract make_ros_depth_contract(uint32_t width,uint32_t height,std::string_view frame_id) {
    return ros_contract_detail::image(width,height,frame_id,"32FC1");
}
inline RosRgbaImageContract make_ros_rgba_contract(uint32_t width,uint32_t height,std::string_view frame_id) {
    return ros_contract_detail::image(width,height,frame_id,"rgba8");
}
// Exact byte extent only: float32 axial-Z metres, including NaN/invalid bits,
// pass untouched. No depth validity, geometry or pixel freshness claim.
inline void validate_ros_depth_payload(const RosDepthImageContract& contract,std::span<const std::byte> bytes) {
    ros_contract_detail::payload(contract,bytes,"32FC1");
}
// Native little-endian packed R,G,B,A bytes; no color conversion is implied.
inline void validate_ros_rgba_payload(const RosRgbaImageContract& contract,std::span<const std::byte> bytes) {
    ros_contract_detail::payload(contract,bytes,"rgba8");
}
struct RosDepthPublicationIdentity {
    int64_t source_id=0;
    uint64_t frame_id=0,capture_ns=0,gpu_ns=0,copy_ns=0,publication_ns=0;
    std::string capture_clock,gpu_clock,copy_clock,publication_clock;
};
// Tracks unique capture identities only, in O(1) retained records. Publisher
// must reject historical/fused ancestry or select a known fresh stereo producer;
// stable depth never establishes age zero or per-pixel freshness.
class FreshSourceTracker {
public:
    void admit(int64_t source_id,uint64_t capture_ns,std::string_view clock) {
        using ros_contract_detail::require;
        require(source_id>=0 && capture_ns>0 && ros_contract_detail::known(clock),"invalid source identity");
        if (initialized_) {
            require(clock==clock_,"capture clock changed");
            require(source_id>source_id_ && capture_ns>capture_ns_,"repeated or backward source identity");
        } else {
            clock_=clock; // Allocate only for the first admitted clock, not per frame.
        }
        source_id_=source_id; capture_ns_=capture_ns; initialized_=true;
    }
private:
    std::string clock_;
    int64_t source_id_=0;
    uint64_t capture_ns_=0;
    bool initialized_=false;
};
}
