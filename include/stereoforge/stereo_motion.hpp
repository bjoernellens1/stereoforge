#pragma once
#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace stereoforge {
// Metric points in the original left-camera rig coordinates of each frame.
struct StereoMetricTrack {
    std::array<double, 3> previous{}, current{};
    // Assumed isotropic standard deviation of the point difference in metres,
    // before residual fitting. Supplying a value does not establish calibration.
    double sigma_m = 0.005;
};
enum class StereoMotionStatus { Tracked, InsufficientTracks, InvalidInput, NoConsensus, Degenerate, InsufficientImageSupport, AmbiguousMotion };
struct StereoMotionConfig {
    std::size_t max_tracks = 1024, min_inliers = 24, hypotheses = 256;
    double min_inlier_fraction = 0.65, max_residual_m = 0.03;
    double noise_floor_m = 0.003, min_spread_m = 0.02;
    // The radial inlier gate for each track is
    // min(max_residual_m, 3 * max(track.sigma_m, noise_floor_m)).
    // max_residual_m is a ceiling: default 5 mm tracks use a 15 mm gate.
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
bool stereo_rigid_transform(std::array<double,16> const&,double tolerance=1e-9);
StereoMotionResult estimate_stereo_motion(std::span<StereoMetricTrack const>, StereoMotionConfig const& = {});
// Diagnostic rigid-body search: min_inliers is absolute; the majority fraction
// does not designate a camera motion. Metric and degeneracy gates are retained.
StereoMotionResult estimate_stereo_consensus(std::span<StereoMetricTrack const>,StereoMotionConfig const&);
}
