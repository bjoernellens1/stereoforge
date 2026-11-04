#pragma once
#include <stereoforge/daxa_temporal.hpp>
#include <deque>
#include <stdexcept>
#include <string_view>

namespace stereoforge {

// ============================================================================
// WIP DUMMY ADAPTER 1: Live Visual Odometry / IMU Rig Pose Source (`P1`–`P3`)
// ============================================================================
// Status: WORK IN PROGRESS (DUMMY ADAPTER)
// Purpose: Provides clock-aligned `DaxaTimedRigPose` instances from `CaptureFrame`
// ingress metadata before a live visual odometry (VIO) or IMU-preintegration
// pose estimator is wired into the real-camera capture loop.
// Warning: Do NOT treat outputs driven by this adapter as physically measured
// camera motion on live sequences.
struct WipDummyRigPoseAdapter {
    std::array<double, 3> step_translation_m{0.0, 0.0, 0.0};
    uint64_t anchor_frame_id = 0;
    bool initialized = false;

    static constexpr bool is_wip_dummy() noexcept { return true; }
    static constexpr std::string_view status() noexcept {
        return "WIP_DUMMY_POSE_ADAPTER_NO_LIVE_VIO_IMU";
    }

    void reset() noexcept {
        anchor_frame_id = 0;
        initialized = false;
    }

    DaxaTimedRigPose pose_for(CaptureFrame const& capture) {
        if (!initialized) {
            anchor_frame_id = capture.frame_id;
            initialized = true;
        }
        double const steps = capture.frame_id >= anchor_frame_id
            ? double(capture.frame_id - anchor_frame_id)
            : 0.0;
        DaxaTimedRigPose pose{};
        pose.world_from_rig = {
            1.0, 0.0, 0.0, steps * step_translation_m[0],
            0.0, 1.0, 0.0, steps * step_translation_m[1],
            0.0, 0.0, 1.0, steps * step_translation_m[2],
            0.0, 0.0, 0.0, 1.0,
        };
        pose.timestamp_ns = int64_t(capture.host_ingress_timestamp_ns);
        pose.clock_domain = capture.host_clock_domain;
        pose.validate(capture);
        return pose;
    }
};

// ============================================================================
// WIP DUMMY ADAPTER 2: Cross-Frame Historical Image-Pair Ring Buffer (`P3`)
// ============================================================================
// Status: WORK IN PROGRESS (DUMMY ADAPTER)
// Purpose: Placeholder ring-buffer adapter for retaining previous rectified
// stereo pairs on GPU to perform multi-pair cost aggregation in `P3`.
// Current behavior: Tracks frame IDs in a bounded metadata ring and delegates
// to `DaxaTemporal::process(...)` (which re-searches the current frame's full-res
// Census volume around the pose-warped historical prior).
class WipDummyMultiPairStereoRingAdapter {
    DaxaTemporal& temporal_;
    size_t capacity_ = 4;
    std::deque<uint64_t> retained_frame_ids_;

public:
    explicit WipDummyMultiPairStereoRingAdapter(DaxaTemporal& temporal, size_t capacity = 4)
        : temporal_(temporal), capacity_(capacity) {
        if (capacity_ == 0 || capacity_ > 8) {
            throw std::invalid_argument("WIP multi-pair ring capacity must be in 1..8");
        }
    }

    static constexpr bool is_wip_dummy() noexcept { return true; }
    static constexpr std::string_view status() noexcept {
        return "WIP_DUMMY_CURRENT_PAIR_CENSUS_ONLY_NO_HISTORICAL_IMAGE_RING";
    }

    size_t capacity() const noexcept { return capacity_; }
    size_t buffered_pair_count() const noexcept { return retained_frame_ids_.size(); }
    void reset() noexcept {
        retained_frame_ids_.clear();
        temporal_.reset();
    }

    DaxaTemporal::Frame process(
        DaxaRefinement::Frame const& input,
        std::optional<DaxaTimedRigPose> const& pose,
        bool use_completed = false
    ) {
        auto const src = (use_completed ? input.completed_observations() : input.observations()).source();
        if (src) {
            if (retained_frame_ids_.size() == capacity_) {
                retained_frame_ids_.pop_front();
            }
            retained_frame_ids_.push_back(src->frame_id);
        }
        return temporal_.process(input, pose, use_completed);
    }
};

// ============================================================================
// WIP DUMMY ADAPTER 3: Native GPU 2x Depth Super-Resolution Grid (`P4`)
// ============================================================================
// Status: WORK IN PROGRESS (DUMMY ADAPTER)
// Purpose: Placeholder GPU facade for the `P4` 2x depth super-resolution grid.
// Current behavior: The 2x pose-aligned subpixel accumulation and joint-luma
// bilateral upsampler currently live in the Python/CPU reference qualification
// (`reconstruct_p4_superres_2x`). This adapter exposes the target 2x dimensions
// while forwarding the 1x `DaxaTemporal::Frame` on GPU.
struct WipDummyGpuSuperRes2xFrame {
    DaxaTemporal::Frame base_frame;
    uint32_t superres_width = 0;
    uint32_t superres_height = 0;

    static constexpr bool is_wip_dummy() noexcept { return true; }
    static constexpr std::string_view status() noexcept {
        return "WIP_DUMMY_HOST_PROTOTYPE_ONLY_NO_GPU_2X_SHADER";
    }
};

class WipDummyGpuSuperRes2xAdapter {
public:
    static constexpr bool is_wip_dummy() noexcept { return true; }
    static constexpr std::string_view status() noexcept {
        return "WIP_DUMMY_HOST_PROTOTYPE_ONLY_NO_GPU_2X_SHADER";
    }
    static constexpr uint32_t scale_factor() noexcept { return 2; }

    WipDummyGpuSuperRes2xFrame wrap(DaxaTemporal::Frame const& frame) const {
        return WipDummyGpuSuperRes2xFrame{
            frame,
            frame.width() * scale_factor(),
            frame.height() * scale_factor(),
        };
    }
};

} // namespace stereoforge
