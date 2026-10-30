#pragma once
// CPU-only, allocation-free depth-frame admission. Camera capture continues at its own rate.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace stereoforge {

inline constexpr std::array<std::uint32_t, 5> depth_rates{5, 10, 20, 30, 60};
inline constexpr bool supported_depth_rate(std::uint32_t rate) noexcept {
    for (auto candidate : depth_rates) if (candidate == rate) return true;
    return false;
}
inline constexpr std::uint64_t depth_period_ns(std::uint32_t rate) noexcept {
    return (1'000'000'000ull + rate - 1) / rate;
}

enum class DepthRateMode { Fixed, Adaptive };
struct DepthRateConfig {
    DepthRateMode mode = DepthRateMode::Adaptive;
    std::uint32_t fixed_fps = 60;
    std::uint32_t min_fps = 5;
    std::uint32_t max_fps = 60;
    std::uint32_t consumer_min_fps = 5;
    std::uint32_t capture_fps = 60;
    std::uint64_t max_depth_age_ns = 250'000'000ull;
    std::uint64_t downshift_dwell_ns = 3'000'000'000ull;
    void validate() const {
        if (!supported_depth_rate(fixed_fps) || !supported_depth_rate(min_fps) ||
            !supported_depth_rate(max_fps) || !supported_depth_rate(consumer_min_fps) ||
            min_fps > max_fps || consumer_min_fps > max_fps ||
            (mode == DepthRateMode::Adaptive && max_fps > capture_fps) ||
            (mode == DepthRateMode::Fixed && (fixed_fps < consumer_min_fps || fixed_fps > capture_fps)) ||
            (mode == DepthRateMode::Adaptive && min_fps > capture_fps) ||
            !capture_fps || capture_fps > 120 || !downshift_dwell_ns ||
            !max_depth_age_ns || max_depth_age_ns < depth_period_ns(mode == DepthRateMode::Fixed ? fixed_fps : std::max(min_fps, consumer_min_fps)))
            throw std::invalid_argument("invalid depth-rate policy, capture rate or freshness bound");
    }
};

struct DepthRateStats {
    std::uint64_t observed = 0, admitted = 0, intentional_skips = 0, slot_rejections = 0;
    std::uint64_t upshifts = 0, downshifts = 0, unavailable_motion = 0;
    std::uint32_t effective_fps = 60;
};

class DepthRateController {
    DepthRateConfig config_;
    DepthRateStats stats_{};
    std::uint64_t last_seen_ns_ = 0, last_admitted_ns_ = 0, quiet_since_ns_ = 0;
    bool seen_ = false, has_admission_ = false, pending_due_ = false;
    static std::uint32_t requested_rate(float activity) noexcept {
        if (activity >= 0.25f) return 60;
        if (activity >= 0.12f) return 30;
        if (activity >= 0.045f) return 20;
        if (activity >= 0.012f) return 10;
        return 5;
    }
    static std::uint32_t next_lower_rate(std::uint32_t rate) noexcept {
        for (std::size_t i = depth_rates.size(); i-- > 1;)
            if (depth_rates[i] <= rate) return depth_rates[i - 1];
        return depth_rates[0];
    }
public:
    struct Decision { bool due; std::uint32_t target_fps; };
    explicit DepthRateController(DepthRateConfig config) : config_(config) {
        config_.validate();
        stats_.effective_fps = config_.mode == DepthRateMode::Fixed ? config_.fixed_fps : config_.max_fps;
    }
    Decision observe(std::uint64_t source_timestamp_ns, std::optional<float> activity) {
        if (!source_timestamp_ns || (seen_ && source_timestamp_ns <= last_seen_ns_))
            throw std::invalid_argument("depth-rate timestamps must be positive and increasing");
        if (pending_due_) throw std::logic_error("previous rate decision was not committed or rejected");
        last_seen_ns_ = source_timestamp_ns;
        seen_ = true;
        ++stats_.observed;
        if (config_.mode == DepthRateMode::Adaptive) {
            std::uint32_t target = config_.max_fps;
            if (activity && std::isfinite(*activity) && *activity >= 0.f) {
                target = std::clamp(requested_rate(*activity),
                                    std::max(config_.min_fps, config_.consumer_min_fps), config_.max_fps);
            } else {
                ++stats_.unavailable_motion; // Fail open: never throttle on missing evidence.
            }
            if (target > stats_.effective_fps) {
                stats_.effective_fps = target;
                ++stats_.upshifts;
                quiet_since_ns_ = 0;
            } else if (target < stats_.effective_fps) {
                if (!quiet_since_ns_) quiet_since_ns_ = source_timestamp_ns;
                if (source_timestamp_ns - quiet_since_ns_ >= config_.downshift_dwell_ns) {
                    stats_.effective_fps = std::max(target, next_lower_rate(stats_.effective_fps));
                    ++stats_.downshifts;
                    quiet_since_ns_ = source_timestamp_ns;
                }
            } else {
                quiet_since_ns_ = 0;
            }
        }
        const auto rate = stats_.effective_fps;
        // Half a capture period tolerates timestamp jitter without skipping a 30 Hz
        // deadline that falls just before the next 60 Hz acquisition.
        const auto tolerance_ns = depth_period_ns(config_.capture_fps) / 2;
        const auto elapsed = has_admission_ ? source_timestamp_ns - last_admitted_ns_ : 0;
        const bool due = !has_admission_ || rate >= config_.capture_fps ||
            elapsed >= config_.max_depth_age_ns ||
            elapsed + tolerance_ns >= depth_period_ns(rate);
        pending_due_ = due;
        if (!due) ++stats_.intentional_skips;
        return {due, rate};
    }
    void admitted(std::uint64_t source_timestamp_ns) {
        if (!pending_due_ || source_timestamp_ns != last_seen_ns_)
            throw std::logic_error("depth admission must follow a due decision for that source frame");
        last_admitted_ns_ = source_timestamp_ns;
        has_admission_ = true;
        pending_due_ = false;
        ++stats_.admitted;
    }
    void reject_busy() {
        if (!pending_due_) throw std::logic_error("cannot reject a non-due depth frame");
        pending_due_ = false;
        ++stats_.slot_rejections;
    }
    const DepthRateStats& stats() const noexcept { return stats_; }
    std::uint64_t last_admitted_ns() const noexcept { return last_admitted_ns_; }
};

// Fixed 2-eye YUYV thumbnail. Only Y bytes are sampled; no GPU work or allocations.
// Thresholds are conservative initial heuristics, NOT calibrated motion estimates.
class YuyvSceneMonitor {
    static constexpr std::size_t cols = 48, rows = 27, eyes = 2;
    static constexpr std::size_t samples = cols * rows * eyes;
    std::array<std::uint8_t, samples> previous_{}, admitted_{}, current_{};
    bool have_previous_ = false, have_admitted_ = false;
    static float compare(const std::array<std::uint8_t, samples>& current,
                         const std::array<std::uint8_t, samples>& old) noexcept {
        std::int64_t sum_delta = 0;
        for (std::size_t i = 0; i < samples; ++i)
            sum_delta += int(current[i]) - int(old[i]);
        const int global_delta = int(sum_delta / std::int64_t(samples));
        std::uint64_t excess = 0, changed = 0;
        for (std::size_t i = 0; i < samples; ++i) {
            const int difference = std::abs(int(current[i]) - int(old[i]) - global_delta);
            excess += std::uint64_t(std::max(0, difference - 4));
            changed += difference >= 14;
        }
        // A large uniform brightness jump also merits an immediate refresh.
        const float global_jump = std::abs(global_delta) >= 24 ? 0.30f : 0.f;
        return std::max({float(excess) / float(samples * 24),
                         float(changed) / float(samples), global_jump});
    }
public:
    std::optional<float> observe(std::span<const std::uint8_t> yuyv,
                                 std::uint32_t source_width, std::uint32_t height,
                                 std::uint32_t stride) noexcept {
        if (source_width < 2 || (source_width % 4) || !height ||
            std::uint64_t(stride) < std::uint64_t(source_width) * 2 ||
            std::uint64_t(yuyv.size()) < std::uint64_t(stride) * height) {
            have_previous_ = false;
            have_admitted_ = false;
            return std::nullopt;
        }
        const auto eye_width = source_width / 2;
        for (std::size_t e = 0; e < eyes; ++e)
            for (std::size_t y = 0; y < rows; ++y)
                for (std::size_t x = 0; x < cols; ++x) {
                    const auto px = std::size_t(e) * eye_width + ((2*x + 1) * eye_width) / (2*cols);
                    const auto py = ((2*y + 1) * std::size_t(height)) / (2*rows);
                    current_[(e*rows + y)*cols + x] = yuyv[py*stride + 2*px];
                }
        std::optional<float> score;
        if (have_previous_) {
            score = compare(current_, previous_);
            if (have_admitted_) score = std::max(*score, compare(current_, admitted_) * 0.5f);
        }
        previous_ = current_;
        have_previous_ = true;
        return score;
    }
    void depth_admitted() noexcept {
        admitted_ = current_;
        have_admitted_ = have_previous_;
    }
};

} // namespace stereoforge
