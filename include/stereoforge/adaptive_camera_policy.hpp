#pragma once
// CPU-only adaptive capture policy for StereoForge. No camera restart or ROS publishing.
// All thresholds are conservative exploratory heuristics, not physically calibrated.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace stereoforge {

enum class CameraCaptureMode : std::uint8_t { HD720_60, HD720_30, HD1080_30 };
struct CameraModeSpec { std::uint32_t width, height, fps; bool sensor_binning; };
constexpr CameraModeSpec camera_mode_spec(CameraCaptureMode mode) noexcept {
    switch (mode) {
        case CameraCaptureMode::HD720_60: return {1280, 720, 60, true};
        case CameraCaptureMode::HD720_30: return {1280, 720, 30, true};
        case CameraCaptureMode::HD1080_30: return {1920, 1080, 30, false};
    }
    return {0, 0, 0, false};
}

enum class LightingClass : std::uint8_t { Unknown, Normal, LowLight, HighContrast };
enum class CameraTask : std::uint8_t { Navigation, Balanced, Detail };
enum class CameraImageTreatment : std::uint8_t { Standard, LowLightDenoise, HighContrastPreview };

struct StereoLightStats {
    // All fractions are 0..1. Histograms sample both original YUYV eyes.
    float median_y = 0, dark_fraction = 0, saturated_fraction = 0, texture_fraction = 0;
};

// 2-eye, 48x27 thumbnail with no allocations and no GPU synchronization.
// Statistics are evidence for mode selection, NOT per-frame physical exposure
// measurements. Upstream auto-exposure can conceal dark physical illumination.
class StereoYuyvLightSampler {
    static constexpr std::size_t cols = 48, rows = 27;
    static constexpr std::size_t count = cols * rows;
    static float median(const std::array<std::uint32_t, 256>& hist) noexcept {
        std::size_t cumulative = 0;
        for (std::size_t i = 0; i < hist.size(); ++i) {
            cumulative += hist[i];
            if (cumulative >= (count + 1) / 2) return float(i);
        }
        return 0;
    }
public:
    std::optional<StereoLightStats> sample(std::span<const std::uint8_t> yuyv,
                                          std::uint32_t source_width, std::uint32_t height,
                                          std::uint32_t stride) const noexcept {
        if (source_width < 4 || source_width % 4 || height < 2 ||
            std::uint64_t(stride) < std::uint64_t(source_width) * 2 ||
            std::uint64_t(yuyv.size()) < std::uint64_t(stride) * height) return std::nullopt;
        const std::size_t eye_width = source_width / 2;
        StereoLightStats result{};
        // Require BOTH eyes to show darkness, but expose saturated highlights
        // from either eye. A single-eye anomaly must not force a hardware switch.
        float dark_both = 1, texture_both = 1, bright_either = 0, median_both = 0;
        for (std::size_t eye = 0; eye < 2; ++eye) {
            std::array<std::uint8_t, count> values{};
            std::array<std::uint32_t, 256> hist{};
            std::size_t dark = 0, bright = 0, textured = 0, neighbors = 0;
            for (std::size_t y = 0; y < rows; ++y) {
                const std::size_t py = ((2 * y + 1) * std::size_t(height)) / (2 * rows);
                for (std::size_t x = 0; x < cols; ++x) {
                    const std::size_t px = eye * eye_width + ((2 * x + 1) * eye_width) / (2 * cols);
                    const auto v = yuyv[py * std::size_t(stride) + 2 * px];
                    const std::size_t i = y * cols + x;
                    values[i] = v;
                    ++hist[v];
                    dark += (v <= 40);
                    bright += (v >= 245);
                    if (x) { textured += std::abs(int(v) - int(values[i - 1])) >= 12; ++neighbors; }
                    if (y) { textured += std::abs(int(v) - int(values[i - cols])) >= 12; ++neighbors; }
                }
            }
            dark_both = std::min(dark_both, float(dark) / float(count));
            bright_either = std::max(bright_either, float(bright) / float(count));
            texture_both = std::min(texture_both, float(textured) / float(neighbors));
            median_both += median(hist) * 0.5f;
        }
        result.median_y = median_both;
        result.dark_fraction = dark_both;
        result.saturated_fraction = bright_either;
        result.texture_fraction = texture_both;
        return result;
    }
};

struct CameraModeObservation {
    std::uint64_t timestamp_ns = 0; // Monotonic host timeline, independent of source clock.
    std::optional<StereoLightStats> light;
    std::optional<float> motion_score; // 0..1; from existing scene monitor, not physical IMU.
    std::optional<float> gain_percent; // 0..100, HOST-READ control, not shutter measurement.
    CameraTask task = CameraTask::Balanced;
    bool safe_to_interrupt_capture = false; // Explicit robot/safety supervisor admission.
};

struct AdaptiveCameraModeConfig {
    std::uint64_t stable_ns = 4'000'000'000ull;
    std::uint64_t min_residence_ns = 30'000'000'000ull;
    float high_motion_score = 0.15f;
    void validate() const {
        if (!stable_ns || !min_residence_ns || !std::isfinite(high_motion_score) ||
            high_motion_score < 0 || high_motion_score > 1)
            throw std::invalid_argument("invalid adaptive camera mode policy");
    }
};

class AdaptiveCameraModePolicy {
    AdaptiveCameraModeConfig config_;
    CameraCaptureMode current_;
    CameraCaptureMode candidate_;
    std::uint64_t last_observed_ns_ = 0, candidate_since_ns_ = 0, last_switch_ns_ = 0;
    bool pending_candidate_ = false, ever_switched_ = false;
    std::optional<CameraCaptureMode> outstanding_request_;

    static bool valid_fraction(float value) noexcept {
        return std::isfinite(value) && value >= 0 && value <= 1;
    }
public:
    struct Decision {
        CameraCaptureMode current_mode;
        CameraCaptureMode desired_mode;
        LightingClass lighting;
        CameraImageTreatment image_treatment;
        bool reopen_requested;
        bool blocked_by_safety;
    };

    explicit AdaptiveCameraModePolicy(CameraCaptureMode initial,
                                      AdaptiveCameraModeConfig cfg = {})
        : config_(cfg), current_(initial), candidate_(initial) {
        config_.validate();
    }

    static LightingClass classify(const StereoLightStats& s,
                                  std::optional<float> gain_percent = std::nullopt) noexcept {
        if (!valid_fraction(s.dark_fraction) || !valid_fraction(s.saturated_fraction) ||
            !valid_fraction(s.texture_fraction) || s.median_y < 0 || s.median_y > 255 ||
            !std::isfinite(s.median_y)) return LightingClass::Unknown;
        // Bright windows and dark areas are HIGH CONTRAST, not simply low light.
        if (s.dark_fraction >= 0.18f && s.saturated_fraction >= 0.025f)
            return LightingClass::HighContrast;
        const bool strong_image_evidence =
            s.median_y < 42 && s.dark_fraction >= 0.65f && s.texture_fraction < 0.18f;
        const bool gain_evidence = gain_percent && std::isfinite(*gain_percent) &&
            *gain_percent >= 75.f && *gain_percent <= 100.f &&
            s.median_y < 78 && s.dark_fraction >= 0.40f;
        if (strong_image_evidence || gain_evidence) return LightingClass::LowLight;
        return LightingClass::Normal;
    }

    Decision observe(const CameraModeObservation& o) {
        if (!o.timestamp_ns || (last_observed_ns_ && o.timestamp_ns <= last_observed_ns_))
            throw std::invalid_argument("mode-policy observation timestamps must increase");
        if (outstanding_request_) throw std::logic_error("unacknowledged camera reconfiguration");
        last_observed_ns_ = o.timestamp_ns;
        LightingClass lighting = o.light ? classify(*o.light, o.gain_percent) : LightingClass::Unknown;
        const bool motion_known = o.motion_score && valid_fraction(*o.motion_score);
        const bool moving = motion_known && *o.motion_score >= config_.high_motion_score;
        CameraImageTreatment treatment = lighting == LightingClass::HighContrast
            ? CameraImageTreatment::HighContrastPreview
            : lighting == LightingClass::LowLight ? CameraImageTreatment::LowLightDenoise
            : CameraImageTreatment::Standard;
        CameraCaptureMode desired = current_; // Fail closed on missing evidence.
        if (moving || o.task == CameraTask::Navigation) desired = CameraCaptureMode::HD720_60;
        else if (lighting == LightingClass::LowLight) desired = CameraCaptureMode::HD720_30;
        else if (lighting == LightingClass::Normal)
            desired = o.task == CameraTask::Detail ? CameraCaptureMode::HD1080_30 : CameraCaptureMode::HD720_60;
        // HighContrast intentionally keeps existing hardware mode: use exposure
        // and preview tone mapping first. No unverified native HDR switch.
        if (desired == current_) {
            pending_candidate_ = false;
            return {current_, desired, lighting, treatment, false, false};
        }
        if (!pending_candidate_ || desired != candidate_) {
            pending_candidate_ = true;
            candidate_ = desired;
            candidate_since_ns_ = o.timestamp_ns;
        }
        const bool stable = o.timestamp_ns - candidate_since_ns_ >= config_.stable_ns;
        const bool cooled = !ever_switched_ || o.timestamp_ns - last_switch_ns_ >= config_.min_residence_ns;
        const bool permitted = o.safe_to_interrupt_capture && !moving;
        if (!stable || !cooled || !permitted)
            return {current_, desired, lighting, treatment, false, stable && cooled && !permitted};
        outstanding_request_ = desired;
        return {current_, desired, lighting, treatment, true, false};
    }

    // Called ONLY after camera reopen, matching calibration load, GPU drain,
    // and first-frame validation succeed. A failure leaves current_ untouched.
    void commit(CameraCaptureMode mode, std::uint64_t at_ns) {
        if (!outstanding_request_ || *outstanding_request_ != mode || at_ns < last_observed_ns_)
            throw std::logic_error("unrequested camera mode commit");
        current_ = mode;
        last_switch_ns_ = at_ns;
        last_observed_ns_ = at_ns;
        ever_switched_ = true;
        pending_candidate_ = false;
        outstanding_request_.reset();
    }
    void abort(std::uint64_t at_ns) {
        if (!outstanding_request_ || at_ns < last_observed_ns_)
            throw std::logic_error("unrequested camera mode abort");
        outstanding_request_.reset();
        pending_candidate_ = false;
        last_switch_ns_ = at_ns; // retry backoff, even after unsuccessful switch.
        last_observed_ns_ = at_ns;
        ever_switched_ = true;
    }
    CameraCaptureMode current_mode() const noexcept { return current_; }
};

} // namespace stereoforge
