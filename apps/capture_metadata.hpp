#pragma once
#include <stereoforge/capture.hpp>
#include <json.hpp>
#include <bit>

namespace stereoforge {
// Benchmark/debug serialization. Capture itself has no JSON dependency.
inline nlohmann::json capture_metadata(const CaptureFrame& f){
    using json=nlohmann::json;
    json imu=json::array();
    for(const auto& s:f.imu_samples){
        std::array<uint32_t,3> a{},w{};
        for(unsigned i=0;i<3;++i){a[i]=std::bit_cast<uint32_t>(s.acceleration_m_s2[i]);w[i]=std::bit_cast<uint32_t>(s.angular_velocity_deg_s[i]);}
        imu.push_back({{"timestamp_ns",s.timestamp_ns},{"acceleration_bits",a},{"angular_velocity_bits",w},
                       {"temperature_bits",std::bit_cast<uint32_t>(s.temperature_c)},{"sync",s.sync}});
    }
    json result={{"source_width",f.source_width},{"height",f.height},{"stride_bytes",f.stride_bytes},{"frame_id",f.frame_id},
        {"camera_timestamp_ns",f.camera_timestamp_ns},{"host_ingress_timestamp_ns",f.host_ingress_timestamp_ns},
        {"camera_clock_domain",f.camera_clock_domain},{"host_clock_domain",f.host_clock_domain},
        {"calibration_identity",f.calibration_identity},{"device_identity",f.device_identity},
        {"controls_json",f.controls_json},{"control_status",f.control_status},{"controls_timestamp_ns",f.controls_timestamp_ns},
        {"controls_clock_domain",f.controls_clock_domain},{"sensor_status",f.sensor_status},{"imu_clock_domain",f.imu_clock_domain},
        {"imu_association",f.imu_association},{"sensor_dropped_count",f.sensor_dropped_count},{"imu_samples",imu}};
    if(f.driver_timing){const auto& t=*f.driver_timing;
        result["driver_timing"]={{"timestamp_ns",t.timestamp_ns},{"sequence",t.sequence},{"flags",t.flags},
            {"dequeue_begin_monotonic_ns",t.dequeue_begin_monotonic_ns},{"dequeue_end_monotonic_ns",t.dequeue_end_monotonic_ns},
            {"sdk_wall_anchor_ns",t.sdk_wall_anchor_ns},{"sdk_first_timestamp_ns",t.sdk_first_timestamp_ns}};
    }
    return result;
}
}
