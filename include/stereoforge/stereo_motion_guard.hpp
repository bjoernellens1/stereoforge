#pragma once
#include <stereoforge/stereo_motion.hpp>
namespace stereoforge {
// Pixel coordinates normalized by (width-1,height-1), before/current frame.
struct StereoImageTrack {std::array<double,2> previous{},current{};};
enum class StereoMotionGuardStatus { NotEvaluated, Accepted, InvalidInput, InsufficientImageSupport, MultipleMotions };
struct StereoMotionGuardConfig {
    bool enabled=true;
    unsigned columns=4,rows=3,min_cells=6;
    double min_span_x=.45,min_span_y=.45;
    std::size_t min_secondary_inliers=12,max_secondary_hypotheses=128;
    double min_secondary_fraction=.1,min_model_separation_m=.02,min_model_separation_rad=.01;
    void validate()const;
};
struct StereoMotionGuardResult {
    StereoMotionGuardStatus status=StereoMotionGuardStatus::NotEvaluated;
    std::size_t primary_inliers=0,secondary_inliers=0;
    unsigned occupied_cells=0;
    double span_x=0,span_y=0,model_separation_m=0,model_separation_rad=0;
};
// Checks a candidate; it does not infer which independently moving body is static.
StereoMotionGuardResult guard_stereo_motion(std::span<StereoMetricTrack const>,std::span<StereoImageTrack const>,StereoMotionResult const&,StereoMotionConfig const&,StereoMotionGuardConfig const& = {});
}
