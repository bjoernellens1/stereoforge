#pragma once
#include <stereoforge/daxa_depth.hpp>
#include <stereoforge/daxa_refinement.hpp>
#include <stereoforge/stereo_trajectory.hpp>
#include <optional>
#include <string_view>
namespace stereoforge {
class DaxaStereoHistory;
struct DaxaTemporalConfig {
    unsigned slots=2,max_contributors=4,max_reuses=4,max_disparity=128;
    int64_t max_age_ns=200000000;
    float relative_depth_tolerance=.03f,confidence_floor=.05f;
    bool inverse_depth_fusion=false,disocclusion_guard=false,temporal_stereo_research=false,left_eye_only=false;
    bool multi_pair_research=false; // Requires explicit odometry research; no default promotion.
    bool odometry_research=false; // Explicit opt-in; unqualified native VO otherwise uses fresh stereo.
    void validate()const;
};
class DaxaLeftObservationFrame {
    std::shared_ptr<DaxaFoundation> context_;
    std::shared_ptr<void> owner_,lease_;
    std::shared_ptr<const DaxaRectification> maps_;CaptureLease source_;
    const uint64_t* live_generation_=nullptr;
    uint64_t generation_=0,submit_=0;uint32_t width_=0,height_=0;
    std::array<daxa::BufferId,6> planes_{};
    std::array<DaxaObservationImageView,2> images_{};
    DaxaObservationAncestryView ancestry_;
    void validate()const;
    friend class DaxaTemporal;
public:
    DaxaLeftObservationFrame()=default;
    explicit operator bool()const{return bool(context_&&owner_&&lease_);}
    uint32_t width()const{validate();return width_;}uint32_t height()const{validate();return height_;}
    uint32_t layers()const{validate();return 1;}
    uint64_t generation()const{validate();return generation_;}uint64_t submit_ticket()const{validate();return submit_;}
    CaptureLease source()const{validate();return source_;}
    daxa::BufferId buffer(DaxaObservationPlane p)const;
    size_t size_bytes(DaxaObservationPlane p)const;
    DaxaObservationImageView image(unsigned i)const;
    DaxaObservationAncestryView ancestry()const{validate();return ancestry_;}
};
struct DaxaTimedRigPose {
    std::array<double,16> world_from_rig{};
    int64_t timestamp_ns=0; // Exact ingress association token, not an acquisition-time claim.
    std::string_view clock_domain;
    void validate(CaptureFrame const&)const;
};
struct DaxaTemporalTiming {
    double clear_gpu_ms=0,z_gpu_ms=0,owner_gpu_ms=0,fuse_gpu_ms=0,total_gpu_ms=0;
    uint64_t start_tick=0,end_tick=0;
    uint32_t timestamp_bits=0;double timestamp_period=0;
};
struct DaxaTemporalCounters {
    uint32_t fresh_domain_rejected=0,history_input_domain_rejected=0,projected_domain_rejected=0,arithmetic_faults=0;
    uint32_t history_scanned=0,history_eligible=0,history_projected=0,history_winners=0,filled=0,fused=0;
    uint32_t pair_checks=0,pair_rejections=0;
};
struct DaxaTemporalReadback {
    std::vector<float> depth,confidence;
    std::vector<uint32_t> validity,provenance,count,reuse,origin_mask;
    std::vector<int64_t> source_ids,age_ns,contributor_ids,oldest_age_ns;
    DaxaTemporalCounters counters;
};
// Known world-from-original-left poses; one submission thread/shared foundation.
class DaxaTemporal {
    struct Impl;struct Slot;std::shared_ptr<Impl> impl_;
public:
    class Frame {
        std::shared_ptr<Impl> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=0;
        Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>);void validate()const;
        friend class DaxaTemporal;
    public:
        Frame()=default;Frame(Frame const&)=default;Frame(Frame&&)noexcept=default;
        Frame& operator=(Frame)noexcept;explicit operator bool()const{return bool(owner_&&slot_);}
        void wait()const;bool ready()const;
        uint32_t width()const;uint32_t height()const;uint64_t generation()const;uint64_t context_id()const;uint64_t submit_ticket()const;
        DaxaObservationFrame observations()const;DaxaLeftObservationFrame left_observations()const;DaxaObservationAncestryView ancestry()const;
        DaxaTemporalTiming timings()const;
    };
    // An injected history has one writer (this engine) and may have other GPU readers.
    DaxaTemporal(std::shared_ptr<DaxaFoundation>,DaxaDepthPipeline::Frame const& prototype,DaxaTemporalConfig={},std::shared_ptr<DaxaStereoHistory> history={});
    DaxaTemporal(std::shared_ptr<DaxaFoundation>,DaxaRefinement::Frame const& prototype,DaxaTemporalConfig={},std::shared_ptr<DaxaStereoHistory> history={});
    Frame process(DaxaDepthPipeline::Frame const&,std::optional<DaxaTimedRigPose> const&);
    Frame process(DaxaRefinement::Frame const&,std::optional<DaxaTimedRigPose> const&,bool use_completed=false);
    Frame process_odometry(DaxaDepthPipeline::Frame const&,StereoWorldPose const&);
    Frame process_odometry(DaxaRefinement::Frame const&,StereoWorldPose const&,bool use_completed=false);
    // Call collect before upstream capture/upload; can_process only checks admission.
    size_t collect();bool can_process()const;void reset();
    DaxaTemporalReadback readback(Frame const&)const; // Explicit diagnostic transfer/wait boundary.
    uint64_t requested_buffer_bytes()const;uint64_t completed_source_retirements()const;
    static std::string shader_directory();static std::string shader_identity();
};
}
