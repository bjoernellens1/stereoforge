#pragma once
// Optional best-effort GPU work budget for StereoForge, using source-time replenishment.
// No Vulkan resource partitioning or memory-bandwidth isolation is implied.
#include <array>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace stereoforge {

struct GpuDutyConfig {
    std::uint32_t percent=100;            // 100 disables admission budgeting.
    std::uint32_t estimated_cost_us=8'000; // Conservative initial GPU service estimate.
    std::uint32_t burst_ms=33;
    std::uint32_t consumer_min_fps=5;
    void validate() const {
        if (!percent || percent>100 || estimated_cost_us<100 || estimated_cost_us>100'000 ||
            burst_ms<1 || burst_ms>1'000 || consumer_min_fps<1 || consumer_min_fps>120)
            throw std::invalid_argument("invalid GPU duty budget");
        if (std::uint64_t(burst_ms)*1000 < estimated_cost_us)
            throw std::invalid_argument("GPU budget burst must fit one estimated frame");
        // Avoid admitting an internally contradictory static floor even at perfect efficiency.
        if (percent < 100 &&
            std::uint64_t(consumer_min_fps)*estimated_cost_us > std::uint64_t(percent)*10'000)
            throw std::invalid_argument("GPU duty budget cannot fit configured consumer minimum");
    }
};

struct GpuDutyStats {
    std::uint64_t reservations=0,denials=0,completions=0,missing_timings=0,charged_gpu_ns=0;
    std::uint64_t max_estimated_debt_ns=0;
    std::uint32_t estimated_cost_us=0;
};

class GpuDutyBudget {
    struct Inflight {
        std::uint64_t source_id=0,reserved_ns=0;
        bool occupied=false;
    };
    GpuDutyConfig config_;
    std::array<Inflight,3> inflight_{};
    GpuDutyStats stats_{};
    std::uint64_t last_source_ns_=0;
    bool have_source_=false;
    std::int64_t credit_ns_=0;
    std::uint64_t estimate_ns_=0;
    std::uint64_t capacity_ns_=0;
    std::uint64_t last_query_source_ns_=0;
    void replenish(std::uint64_t timestamp_ns) {
        if (!timestamp_ns || (have_source_ && timestamp_ns<last_source_ns_))
            throw std::invalid_argument("budget source timestamps must be increasing");
        if (have_source_ && timestamp_ns>last_source_ns_ && config_.percent<100) {
            auto delta=std::min<std::uint64_t>(timestamp_ns-last_source_ns_, 10'000'000'000ull);
            const auto increment=(delta*config_.percent)/100;
            credit_ns_=std::min<std::int64_t>(std::int64_t(capacity_ns_),
                                               credit_ns_+std::int64_t(increment));
        }
        have_source_=true;
        last_source_ns_=timestamp_ns;
    }
    Inflight* find(std::uint64_t source_id) noexcept {
        for(auto& v:inflight_)if(v.occupied&&v.source_id==source_id)return &v;
        return nullptr;
    }
public:
    explicit GpuDutyBudget(GpuDutyConfig config):config_(config) {
        config_.validate();
        estimate_ns_=std::uint64_t(config_.estimated_cost_us)*1000;
        capacity_ns_=std::uint64_t(config_.burst_ms)*1'000'000;
        credit_ns_=std::int64_t(capacity_ns_);
        stats_.estimated_cost_us=config_.estimated_cost_us;
    }
    bool enabled()const noexcept {return config_.percent<100;}
    bool may_admit(std::uint64_t source_time_ns) {
        replenish(source_time_ns);
        last_query_source_ns_=source_time_ns;
        if(!enabled())return true;
        if(credit_ns_<std::int64_t(estimate_ns_)) {++stats_.denials;return false;}
        for(auto const& f:inflight_)if(!f.occupied)return true;
        ++stats_.denials;
        return false;
    }
    // Call only after a complete GPU submission has succeeded. Each in-flight
    // reservation is reconciled after a corresponding completion observation.
    void reserve(std::uint64_t source_id,std::uint64_t source_time_ns) {
        if (!source_id || source_time_ns!=last_query_source_ns_ || find(source_id))
            throw std::logic_error("invalid GPU budget reservation identity");
        for(auto& f:inflight_)if(!f.occupied) {
            f={source_id,estimate_ns_,true};
            if(enabled())credit_ns_-=std::int64_t(estimate_ns_);
            ++stats_.reservations;
            return;
        }
        throw std::logic_error("GPU budget in-flight capacity exceeded");
    }
    // Actual service includes the measured depth and refinement queue spans,
    // but not camera transport, ROS export, or competing GPU work. Missing or
    // invalid GPU timings leave the estimate in place and are counted.
    void complete(std::uint64_t source_id,double measured_gpu_ms) {
        auto* f=find(source_id);
        if(!f)throw std::logic_error("GPU budget completion missing matching reservation");
        auto observed_ns=f->reserved_ns;
        if(std::isfinite(measured_gpu_ms)&&measured_gpu_ms>0.0&&measured_gpu_ms<=1000.0) {
            observed_ns=std::uint64_t(std::llround(measured_gpu_ms*1'000'000.0));
            // Smooth estimate to avoid rate thrashing on isolated outliers;
            // use measured cost directly to reconcile budget credit.
            estimate_ns_=(7*estimate_ns_+observed_ns+4)/8;
            estimate_ns_=std::clamp<std::uint64_t>(estimate_ns_,100'000ull,100'000'000ull);
        } else {
            ++stats_.missing_timings;
        }
        if(enabled()) {
            const auto difference=std::int64_t(f->reserved_ns)-std::int64_t(observed_ns);
            credit_ns_=std::min(std::int64_t(capacity_ns_),credit_ns_+difference);
            if(credit_ns_<0) stats_.max_estimated_debt_ns=std::max(
                stats_.max_estimated_debt_ns,std::uint64_t(-credit_ns_));
        }
        stats_.charged_gpu_ns+=observed_ns;
        stats_.estimated_cost_us=std::uint32_t((estimate_ns_+500)/1000);
        ++stats_.completions;
        f->occupied=false;
    }
    const GpuDutyStats& stats() const noexcept {return stats_;}
    std::int64_t credit_ns() const noexcept {return credit_ns_;}
};

} // namespace stereoforge
