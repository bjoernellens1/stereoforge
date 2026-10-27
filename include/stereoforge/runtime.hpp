#pragma once
#include <stereoforge/contracts.hpp>
#include <stereoforge/native_device.hpp>
#include <span>
#include <optional>
#include <string>
#include <vector>

namespace stereoforge {
struct NativeFrame {
    BufferView disparity, confidence, validity;
    CompletionToken completion;
    std::uint32_t width = 0, height = 0;
    BufferView depth; // Empty without calibration; NaN when metric geometry is invalid.
    BufferView depth_validity; // Separate metric mask; both metric handles empty without calibration.
    // validity always describes disparity, independently of metric geometry.
};
struct StereoReadback {
    std::vector<float> disparity, confidence;
    std::vector<std::uint32_t> validity;
    std::vector<float> depth;
    std::vector<std::uint32_t> depth_validity;
};
// Vulkan only. Two persistent slots; enqueue never waits for earlier work and
// throws when both slots remain leased. Destroy returned frames/views to recycle.
// Single submission thread. Resource consumers must wait on completion before use.
class NativeStereo {
    struct Impl;
    std::shared_ptr<Impl> impl_;
public:
    explicit NativeStereo(StereoConfig config, const std::string& shader_directory, bool vulkan_validation = false,
        std::optional<MetricCalibration> metric_calibration = std::nullopt);
    NativeStereo(StereoConfig config,const std::string& shader_directory,NativeDevice device,
        std::optional<MetricCalibration> metric_calibration=std::nullopt);
    NativeFrame enqueue(std::span<const std::uint32_t> left, std::span<const std::uint32_t> right);
    StereoReadback readback(const NativeFrame& frame) const;
    std::uint64_t device_id() const;
    std::string device_name() const;
};
} // namespace stereoforge
