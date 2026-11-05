#pragma once
#include <stereoforge/daxa_observation.hpp>
#include <stereoforge/stereo_trajectory.hpp>
#include <memory>
namespace stereoforge {
struct DaxaStereoVoConfig {
    unsigned columns=32, rows=24;
    uint64_t max_ingress_gap_ns=200000000;
    StereoMotionConfig motion{.bearing_refinement=true};
    StereoMotionGuardConfig guard;
    std::string rig_session_identity; // Administrative label; empty is unverified.
};
using DaxaStereoVoResult=StereoVoFrame;
// GPU corners/patch tracking and stereo landmarks; only bounded tracks reach CPU.
// Synchronous initial implementation: producer lease is held through the GPU fence.
// Does not qualify temporal fusion or expose an externally supplied pose.
class DaxaStereoVo {
    struct Impl;std::unique_ptr<Impl> impl_;
public:
    DaxaStereoVo(std::shared_ptr<DaxaFoundation>,DaxaObservationFrame const&,DaxaStereoVoConfig={});
    ~DaxaStereoVo();
    DaxaStereoVoResult process(DaxaObservationFrame const&);
    void reset();
};
}
