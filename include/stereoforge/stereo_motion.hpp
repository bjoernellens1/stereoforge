#pragma once
#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace stereoforge {
// Metric points in the original left-camera rig coordinates of each frame.
struct StereoMetricTrack {
    std::array<double, 3> previous{}, current{};
    // Standard deviation of the point difference in metres, before residual fitting.
    double sigma_m = 0.005;
};
enum class StereoMotionStatus { Tracked, InsufficientTracks, InvalidInput, NoConsensus, Degenerate };
struct StereoMotionConfig {
    std::size_t max_tracks = 1024, min_inliers = 24, hypotheses = 256;
    double min_inlier_fraction = 0.65, max_residual_m = 0.03;
    double noise_floor_m = 0.003, min_spread_m = 0.02;
    // Refine metric landmarks against current bearing directions (stereo PnP).
    bool bearing_refinement = false;
    void validate() const;
};
struct StereoMotionResult {
    StereoMotionStatus status = StereoMotionStatus::InsufficientTracks;
    // Row-major SE(3): current_point = current_from_previous * previous_point.
    std::array<double, 16> current_from_previous{};
    // Conditional local fit covariance, order tx,ty,tz,rx,ry,rz; left perturbation.
    // This is NOT a calibrated pose uncertainty and must not qualify temporal fusion alone.
    std::array<double, 36> covariance{};
    bool covariance_calibrated = false;
    std::vector<std::size_t> inliers;
    double rms_m = 0;
};
StereoMotionResult estimate_stereo_motion(std::span<StereoMetricTrack const>, StereoMotionConfig const& = {});
}
