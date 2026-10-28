#pragma once
#include <stereoforge/daxa_depth.hpp>
namespace stereoforge {
// Experimental independent local matching, never an SGM or interpolated-depth result.
struct DaxaRefinementConfig {
    unsigned slots=2,radius=4,disparities=128,min_margin=1;
    float prior_confidence_floor=.05f,max_coarse_span=2.f,lr_tolerance=1.f;
    bool subpixel=false;
    void validate()const;
};
void validate_daxa_refinement_geometry(const DaxaRectification& coarse,const DaxaRectification& target);
// Planes use contiguous [eye][y][x], except RectifiedRgb/RightRectifiedRgb
// which each hold one eye so generic observation RGB descriptors need no offsets.
// Lower/Upper/Winner/Best/Second encode unavailable signed -1 as UINT32_MAX.
enum class DaxaRefinementPlane:unsigned {
    Disparity,Confidence,Validity,Depth,MetricConfidence,MetricValidity,Provenance,SourceIds,AgeNs,
    Prediction,PredictionValidity,Lower,Upper,Reason,Method,Winner,BestCost,SecondCost,SupportedCandidates,EvaluatedCandidates,SubpixelApplied,
    RawDisparity,RawConfidence,RawValidity,Census,CensusValidity,RectifiedRgb,Luma,Support,RightRectifiedRgb
};
struct DaxaRefinementReadback {
    std::vector<float> disparity,confidence,depth,metric_confidence,prediction,raw_disparity,raw_confidence;
    std::vector<uint32_t> validity,metric_validity,provenance,prediction_validity,lower,upper,reason,method,winner,best_cost,second_cost,supported_candidates,evaluated_candidates,subpixel_applied,raw_validity,census,census_validity,luma,support;
    std::array<std::vector<uint32_t>,2> rectified_rgb;
    std::vector<int64_t> source_ids,age_ns;
};
struct DaxaRefinementTiming {
    static constexpr unsigned stages=5;
    bool canonical_pack=false;const char* first_stage_name="rectify";
    std::array<double,stages> stage_gpu_ms{};double total_gpu_ms=0;
    uint64_t start_tick=0,end_tick=0;double timestamp_period=0;uint32_t timestamp_bits=0;
};
class DaxaRefinement {
    struct Impl;struct Slot;std::shared_ptr<Impl> impl_;
public:
    class Frame {
        std::shared_ptr<Impl> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=0;
        friend class DaxaRefinement;
        Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>);void validate()const;
    public:
        Frame()=default;Frame(const Frame&)=default;Frame(Frame&&)noexcept=default;Frame& operator=(Frame)noexcept;
        explicit operator bool()const{return bool(owner_&&slot_);}
        void wait()const;bool ready()const;uint64_t generation()const;uint64_t submit_ticket()const;
        uint32_t width()const;uint32_t height()const;
        DaxaObservationFrame observations()const;
        daxa::BufferId buffer(DaxaRefinementPlane)const;size_t size_bytes(DaxaRefinementPlane)const;
        DaxaRightReferenceMode right_reference_mode()const{return DaxaRightReferenceMode::IndependentLocalCensus;}
        DaxaStereoEstimator estimator()const{return DaxaStereoEstimator::IndependentLocalCensus;}
        DaxaRefinementTiming timings()const;
    };
    DaxaRefinement(std::shared_ptr<DaxaFoundation>,DaxaRectification target,const DaxaDepthPipeline::Frame& prototype,DaxaRefinementConfig={});
    // Full-frame mode accepts canonical or packed-YUYV preprocessing with the
    // same source lease and setup-bound calibration; six planes are copied on GPU.
    DaxaRefinement(std::shared_ptr<DaxaFoundation>,const DaxaDepthPipeline::Frame& coarse_prototype,const DaxaFrame& full_prototype,DaxaRefinementConfig={});
    Frame process(const DaxaDepthPipeline::Frame&);
    Frame process(const DaxaDepthPipeline::Frame&,const DaxaFrame& full);
    bool can_process()const;size_t collect();
    DaxaRefinementReadback readback(const Frame&)const;
    uint64_t requested_buffer_bytes()const;uint64_t completed_source_retirements()const;
    static std::string shader_directory();static std::string shader_identity();
    static std::string compiler_version();static std::string compiler_sha256();
    static std::string compiler_package_sha256();static std::string compiler_commit();
};
}
