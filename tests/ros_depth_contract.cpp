#include <stereoforge/ros_depth_contract.hpp>
#include <array>
#include <stdexcept>
#include <cstring>
#include <limits>
using namespace stereoforge;
void check(bool condition) { if (!condition) throw std::runtime_error("contract check failed"); }
template<class F> void rejects(F f) { bool threw=false; try {f();} catch(const std::invalid_argument&) {threw=true;} check(threw); }
int main() {
    RosCaptureClockMapping m{"camera",-1,1,UINT64_MAX,7};
    auto s=map_ros_capture_stamp(m,"camera",1000000002); check(s.sec==1 && s.nanosec==1 && s.max_error_ns==7);
    check(map_ros_capture_stamp(m,"camera",1).sec==0);
    rejects([&]{map_ros_capture_stamp(m,"other",2);});
    rejects([&]{map_ros_capture_stamp(m,"unknown",2);});
    rejects([&]{map_ros_capture_stamp(m,"",2);});
    m.valid_from_ns=3; m.valid_until_ns=2; rejects([&]{map_ros_capture_stamp(m,"camera",3);});
    m={"camera",0,0,10,0}; rejects([&]{map_ros_capture_stamp(m,"camera",3);});
    m={"camera",0,1,10,0}; rejects([&]{map_ros_capture_stamp(m,"camera",11);});
    m={"camera",-2,1,UINT64_MAX,0}; rejects([&]{map_ros_capture_stamp(m,"camera",1);});
    m.offset_ns=INT64_MIN; check(map_ros_capture_stamp(m,"camera",uint64_t{1}<<63).sec==0);
    m.offset_ns=INT64_MAX; rejects([&]{map_ros_capture_stamp(m,"camera",UINT64_MAX);});
    m.offset_ns=0; constexpr uint64_t last=uint64_t(INT32_MAX)*1000000000+999999999;
    s=map_ros_capture_stamp(m,"camera",last); check(s.sec==INT32_MAX && s.nanosec==999999999);
    rejects([&]{map_ros_capture_stamp(m,"camera",last+1);});
    auto d=make_ros_depth_contract(2,1,"left_optical_frame"); check(d.encoding=="32FC1" && d.step==8 && !d.is_bigendian);
    std::array<std::byte,8> payload{}; uint32_t nan=0x7fc01234; std::memcpy(payload.data(),&nan,4); auto before=payload;
    validate_ros_depth_payload(d,payload); check(payload==before);
    rejects([&]{validate_ros_depth_payload(d,std::span<const std::byte>(payload).first(7));});
    auto r=make_ros_rgba_contract(2,1,"left_optical_frame"); check(r.encoding=="rgba8" && r.step==8); validate_ros_rgba_payload(r,payload);
    rejects([&]{validate_ros_depth_payload(r,payload);});
    rejects([&]{make_ros_depth_contract(0,1,"left_optical_frame");});
    rejects([&]{make_ros_depth_contract(8193,1,"left_optical_frame");});
    rejects([&]{make_ros_depth_contract(UINT32_MAX,UINT32_MAX,"left_optical_frame");});
    check(make_ros_depth_contract(8192,8192,"left_optical_frame").step==32768);
    for(auto label:{"","unknown"}) rejects([&]{make_ros_depth_contract(1,1,label);});
    for (unsigned mutation=0; mutation<6; ++mutation) {
        auto invalid=d;
        switch(mutation) {
        case 0: invalid.width=0; break;
        case 1: invalid.height=8193; break;
        case 2: invalid.step=7; break;
        case 3: invalid.encoding="rgba8"; break;
        case 4: invalid.is_bigendian=true; break;
        case 5: invalid.frame_id="unknown"; break;
        }
        rejects([&]{validate_ros_depth_payload(invalid,payload);});
    }
    FreshSourceTracker tracker; tracker.admit(42,10,"camera");
    rejects([&]{tracker.admit(42,11,"camera");}); rejects([&]{tracker.admit(41,11,"camera");});
    rejects([&]{tracker.admit(43,10,"camera");}); rejects([&]{tracker.admit(43,11,"other");});
    // Rejections must not advance either part of the retained identity or its clock.
    rejects([&]{tracker.admit(-1,12,"camera");});
    rejects([&]{tracker.admit(99,0,"camera");});
    rejects([&]{tracker.admit(99,99,"unknown");});
    tracker.admit(43,11,"camera"); tracker.admit(INT64_MAX,UINT64_MAX,"camera");
    FreshSourceTracker initial; rejects([&]{initial.admit(-1,1,"camera");}); rejects([&]{initial.admit(0,0,"camera");});
    rejects([&]{initial.admit(0,1,"unknown");}); initial.admit(0,1,"camera");
    RosDepthPublicationIdentity identity{}; check(identity.source_id==0);
}
