#pragma once
#include <stereoforge/daxa_observation.hpp>
#include <stereoforge/stereo_motion.hpp>
#include <memory>
namespace stereoforge {
struct DaxaStereoVoConfig {
    unsigned columns=32, rows=24;
    uint64_t max_ingress_gap_ns=200000000;
    StereoMotionConfig motion{.bearing_refinement=true};
};
struct DaxaStereoVoResult {
    uint64_t source_frame_id=0, previous_frame_id=0;
    uint64_t acquisition_timestamp_ns=0, ingress_timestamp_ns=0;
    std::string acquisition_clock, ingress_clock;
    bool acquisition_clock_qualified=false;
    std::size_t tracked_features=0;
    StereoMotionResult motion;
};
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
