#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace stereoforge {
struct CaptureImuSample {
    uint64_t timestamp_ns=0;
    std::array<float,3> acceleration_m_s2{},angular_velocity_deg_s{};
    float temperature_c=0;
    bool sync=false;
};
enum class CaptureSourceKind { PackedYuyv,CanonicalRgbMetadata };
struct CanonicalRgbSource;
struct CaptureFrame {
    CaptureSourceKind source_kind=CaptureSourceKind::PackedYuyv;
    // Canonical RGB keeps original geometry: source_width=2*original eye width,
    // height=original height, stride_bytes=0 and packed_yuyv empty. RGB planes
    // live in a separate pitched sidecar defined in canonical_rgb.hpp.
    std::shared_ptr<const CanonicalRgbSource> canonical_rgb;
    // Full unrectified post-ISP side-by-side source, 2 bytes/pixel YUYV.
    uint32_t source_width=0,height=0,stride_bytes=0;
    uint64_t frame_id=0,camera_timestamp_ns=0,host_ingress_timestamp_ns=0;
    std::string camera_clock_domain="unknown",host_clock_domain="host_steady";
    std::string calibration_identity,device_identity;
    std::string controls_json="{}",control_status="unavailable";
    uint64_t controls_timestamp_ns=0;
    std::string controls_clock_domain="unknown";
    std::string sensor_status="unavailable",imu_clock_domain="unknown",imu_association="unknown_raw_not_pose";
    uint64_t sensor_dropped_count=0;
    std::vector<CaptureImuSample> imu_samples;
    std::vector<uint8_t> packed_yuyv;
    void require_calibration() const;
};
using CaptureLease=std::shared_ptr<const CaptureFrame>;
enum class CaptureQueuePolicy { drop_oldest,wait_for_consumer };
struct CaptureConfig {
    uint32_t eye_width=1280,height=720,fps=60;
    int device_id=-1;
    size_t queue_capacity=3,pool_capacity=6;
    CaptureQueuePolicy queue_policy=CaptureQueuePolicy::drop_oldest;
    // Supplied validated calibration identity. Capture permits empty; processing must reject it.
    std::string calibration_identity;
    // Explicit recovery for ZED firmware that exposes no serial through Open Capture.
    // The value is accepted only for an allowlisted calibration/device pairing.
    std::string device_serial_override;
    std::chrono::milliseconds source_timeout{100};
};
struct CaptureMetrics {
    uint64_t source_records=0,source_frames=0,delivered_frames=0,queue_drops=0,no_slot_drops=0;
    uint64_t source_frame_gaps=0,repeated_frames=0,source_timeouts=0,invalid_frames=0;
    uint64_t backpressure_waits=0,backpressure_wait_ns=0;
    size_t queued_frames=0,queue_high_watermark=0,pool_capacity=0,packed_bytes_per_slot=0;
    bool running=false;
};
enum class CaptureSourceStatus { frame,recorded_frame,timeout,end };
// recorded_frame preserves all stored metadata and repeated identities during replay.
// Source must honour timeout and return within it; never retain/mutate a frame after returning.
// Native camera/replay sources obey this. Injection is for deterministic sources and CPU fixtures.
using CaptureSource=std::function<CaptureSourceStatus(CaptureFrame&,std::chrono::milliseconds)>;
class NativeCapture {
public:
    explicit NativeCapture(CaptureConfig config);
    NativeCapture(CaptureConfig config,CaptureSource source);
    ~NativeCapture();
    NativeCapture(const NativeCapture&)=delete;
    NativeCapture& operator=(const NativeCapture&)=delete;
    CaptureLease read(std::chrono::milliseconds timeout=std::chrono::milliseconds{1000});
    CaptureMetrics metrics() const;
    void stop();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// capture.sfp v1: little-endian, finalized frame count, bounded length-prefixed metadata,
// packed bytes and per-record FNV-1a-64 for accidental corruption (not authentication).
class PackedCaptureWriter {
public:
    explicit PackedCaptureWriter(const std::filesystem::path& directory);
    ~PackedCaptureWriter();
    void append(const CaptureFrame& frame);
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Fixture convenience only; production replay below streams a single record at a time.
std::vector<CaptureFrame> read_packed_capture(const std::filesystem::path& directory,
    size_t max_frames=4096,size_t max_bytes=512*1024*1024);
CaptureSource packed_replay_source(const std::filesystem::path& directory);
}
