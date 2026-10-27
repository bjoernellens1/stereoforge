#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace stereoforge {
// Coordinates: x right, y down, z forward. Baseline and depth are metres.
// disparity = x_left - x_right; cx_delta = cx_left - cx_right.
inline float metric_depth(float disparity, float focal_px, float baseline_m, float cx_delta = 0) {
    const float corrected = disparity - cx_delta;
    if (!std::isfinite(corrected) || corrected <= 0 || !std::isfinite(focal_px) || focal_px <= 0 ||
        !std::isfinite(baseline_m) || baseline_m <= 0)
        return std::numeric_limits<float>::quiet_NaN();
    return focal_px * baseline_m / corrected;
}

struct MetricCalibration {
    float fx,baseline_m,cx_offset=0;
    void validate() const {
        if(!std::isfinite(fx)||fx<=0||!std::isfinite(baseline_m)||baseline_m<=0||!std::isfinite(cx_offset))
            throw std::invalid_argument("metric calibration requires positive finite focal length/baseline and finite principal offset");
    }
};

struct CompletionState {
    const std::uint64_t device_id;
    explicit CompletionState(std::uint64_t device) : device_id(device) {}
    virtual ~CompletionState() = default;
    virtual bool ready() const = 0;
    virtual void wait() = 0;
};
class CompletionToken {
    std::shared_ptr<CompletionState> state_;
public:
    CompletionToken() = default;
    explicit CompletionToken(std::shared_ptr<CompletionState> state) : state_(std::move(state)) {}
    bool ready() const { return state_ && state_->ready(); }
    void wait() const { if (!state_) throw std::invalid_argument("empty completion token"); state_->wait(); }
    std::uint64_t device_id() const { return state_ ? state_->device_id : 0; }
    explicit operator bool() const { return bool(state_); }
};

// native is an rhi::IBuffer* when supplied by NativeStereo. The owner and
// completion retain the producing slot. Do not release either before consumption.
struct BufferView {
    void* native = nullptr;
    std::size_t offset_bytes = 0, size_bytes = 0, stride_bytes = 0;
    std::uint64_t device_id = 0;
    std::shared_ptr<void> owner;
    CompletionToken completion;
    void validate(std::uint64_t expected_device) const {
        if (!native || !owner || !completion || !size_bytes || !stride_bytes ||
            device_id == 0 || device_id != expected_device || completion.device_id() != device_id)
            throw std::invalid_argument("invalid buffer ownership, completion, or device identity");
    }
};
enum class ImageFormat { Gray32Uint, Rgb8, Rgba8, Float32, Uint32 };
struct ImageHandle {
    BufferView buffer;
    std::uint32_t width = 0, height = 0;
    std::size_t row_stride_bytes = 0;
    ImageFormat format = ImageFormat::Gray32Uint;
    void validate(std::uint64_t expected_device) const {
        buffer.validate(expected_device);
        const auto pixel_bytes = format == ImageFormat::Rgb8 ? 3u : 4u;
        if (!width || !height || row_stride_bytes < std::size_t(width)*pixel_bytes ||
            row_stride_bytes > buffer.size_bytes / height)
            throw std::invalid_argument("invalid image shape or stride");
    }
};
struct StereoConfig {
    std::uint32_t width = 0, height = 0, disparities = 128, paths = 4;
    std::uint32_t p1 = 8, p2 = 32;
    bool subpixel = true;
    void validate() const {
        if (width < 5 || height < 5 || disparities == 0 || disparities > 256 ||
            (paths != 4 && paths != 8) || p1 > p2 || p2 > 10000 ||
            std::uint64_t(width) * height * disparities > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("invalid stereo dimensions, disparities, paths, or penalties");
    }
};
} // namespace stereoforge
