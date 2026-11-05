#pragma once
#include <stereoforge/stereo_motion_guard.hpp>
#include <cstdint>
#include <optional>
#include <string>
namespace stereoforge {
struct StereoFrameIdentity {
    uint64_t frame_id=0, acquisition_timestamp_ns=0, ingress_timestamp_ns=0;
    std::string device, original_calibration, rectified_calibration, rig_session;
    std::string acquisition_clock, ingress_clock;
    // Unknown remains absent; ingress is never substituted for acquisition.
    std::optional<uint64_t> clock_mapping_uncertainty_ns;
    bool operator==(StereoFrameIdentity const&) const=default;
    void validate() const;
};
struct StereoVoFrame {
    StereoFrameIdentity source;
    std::optional<StereoFrameIdentity> previous_source;
    std::size_t tracked_features=0;
    StereoMotionResult motion;
    StereoMotionGuardResult guard;
};
enum class StereoTrajectoryStatus { Anchored, Tracked, TrackingLost, SourceMismatch, HistoryExpired };
struct StereoWorldPose {
    StereoFrameIdentity source;
    uint64_t trajectory_id=0, epoch=0;
    StereoTrajectoryStatus status=StereoTrajectoryStatus::TrackingLost;
    std::optional<std::array<double,16>> world_from_rig;
    // Latest conditional local-fit covariance, NOT accumulated world covariance.
    std::optional<std::array<double,36>> local_covariance;
    double inlier_fraction=0, rms_m=0;
    bool covariance_calibrated=false, temporal_qualified=false;
};
std::optional<uint64_t> stereo_acquisition_age_bound(StereoFrameIdentity const& previous,StereoFrameIdentity const& current);
bool stereo_pose_usable(StereoWorldPose const&,bool allow_unqualified_research=false);
struct StereoTrajectoryConfig {
    uint64_t max_ingress_gap_ns=200000000;
    std::size_t min_inliers=24, max_tracks=1024;
    double min_inlier_fraction=.65, max_rms_m=.03;
    double max_translation_m=.2, max_rotation_rad=.25;
    void validate() const;
};
// Origin is an arbitrary gauge per epoch. Lost chains never reconnect to old worlds.
class StereoTrajectory {
    StereoTrajectoryConfig config_;
    std::optional<StereoFrameIdentity> last_;
    std::array<double,16> world_{};
    uint64_t trajectory_id_=0, epoch_=0;
public:
    explicit StereoTrajectory(StereoTrajectoryConfig={});
    StereoTrajectory(StereoTrajectory const&)=delete;
    StereoTrajectory& operator=(StereoTrajectory const&)=delete;
    StereoWorldPose process(StereoVoFrame const&);
    void reset();
};
}
