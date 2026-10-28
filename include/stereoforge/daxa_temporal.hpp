#pragma once
#include <stereoforge/daxa_depth.hpp>
#include <optional>
#include <string_view>
namespace stereoforge {
struct DaxaTemporalConfig {
    unsigned slots=2,max_contributors=4,max_reuses=4;
    int64_t max_age_ns=200000000;
    float relative_depth_tolerance=.03f,confidence_floor=.05f;
    void validate()const;
};
struct DaxaTimedRigPose {
    std::array<double,16> world_from_rig{};
    int64_t timestamp_ns=0;
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
        DaxaObservationFrame observations()const;DaxaObservationAncestryView ancestry()const;
        DaxaTemporalTiming timings()const;
    };
    DaxaTemporal(std::shared_ptr<DaxaFoundation>,DaxaDepthPipeline::Frame const& prototype,DaxaTemporalConfig={});
    Frame process(DaxaDepthPipeline::Frame const&,std::optional<DaxaTimedRigPose> const&);
    // Call collect before upstream capture/upload; can_process only checks admission.
    size_t collect();bool can_process()const;void reset();
    DaxaTemporalReadback readback(Frame const&)const; // Explicit diagnostic transfer/wait boundary.
    uint64_t requested_buffer_bytes()const;uint64_t completed_source_retirements()const;
    static std::string shader_directory();static std::string shader_identity();
};
}
