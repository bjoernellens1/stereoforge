#pragma once
#include <stereoforge/daxa_preprocess.hpp>
namespace stereoforge {
enum class DaxaDepthStereoSchedule { Sequential,PairedPerPath,LayeredPerPath,SharedCensusSequential };
struct DaxaDepthConfig {
    StereoConfig stereo;
    float lr_tolerance=1.f;
    unsigned slots=1;
    uint32_t required_wave_size=32;
    DaxaDepthStereoSchedule stereo_schedule=DaxaDepthStereoSchedule::Sequential;
    void validate()const;
    uint64_t requested_buffer_bytes(const DaxaRectification&)const;
};
// Input arrays and the CPU oracle are diagnostics, never a native frame path.
struct DaxaDepthStereoInput {
    std::array<std::vector<float>,2> disparity,confidence;
    std::array<std::vector<uint32_t>,2> validity,support;
};
struct DaxaDepthReadback {
    std::vector<float> lr_disparity,lr_confidence,depth,metric_confidence;
    std::vector<uint32_t> lr_validity,metric_validity,provenance;
    std::vector<int64_t> source_ids,age_ns;
};
DaxaDepthReadback daxa_depth_reference(const DaxaDepthConfig&,MetricCalibration,const DaxaDepthStereoInput&,uint64_t source_id);
enum class DaxaDepthPlane:unsigned { LRDisparity,LRStereoConfidence,LRValidity,Depth,MetricConfidence,MetricValidity,Provenance,SourceIds,AgeNs };
struct DaxaDepthTiming {
    DaxaPreprocessTiming preprocessing;
    DaxaStereoTiming left,right;
    bool per_reference_stereo_timings_available=true;
    DaxaStereoTiming paired_stereo; // Phase intervals contain both references; no isolated eye timings.
    double lr_gpu_ms=0,metric_metadata_gpu_ms=0,combined_gpu_ms=0;
};
// One submission thread/shared foundation. Output leases retain both producers
// and preprocessing RGB until released. collect() precedes acquiring capture.
class DaxaDepthPipeline {
    struct Impl;struct Slot;
    std::shared_ptr<Impl> impl_;
public:
    class Frame {
        std::shared_ptr<Impl> owner_;std::shared_ptr<Slot> slot_;uint64_t generation_=0;
        friend class DaxaDepthPipeline;
        Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>);void validate()const;
    public:
        Frame()=default;Frame(const Frame&)=default;Frame(Frame&&)noexcept=default;Frame& operator=(Frame)noexcept;
        explicit operator bool()const{return bool(owner_&&slot_);}
        void wait()const;bool ready()const;
        uint32_t width()const;uint32_t height()const;uint32_t layers()const{return 2;}
        uint64_t generation()const;uint64_t context_id()const;
        daxa::BufferId buffer(DaxaDepthPlane)const;size_t size_bytes(DaxaDepthPlane)const;
        DaxaImageView image(DaxaPreprocessPlane)const; // Independently retained RGB views consume preprocessing capacity.
        CaptureLease source()const;std::shared_ptr<const DaxaRectification> calibration()const;
        MetricCalibration metric_calibration()const;DaxaDepthTiming timings()const;
        DaxaStereo::Frame stereo_frame(DaxaStereoReference)const; // SharedCensusSequential GPU producer lease; independently consumes aligned stereo capacity.
    };
    DaxaDepthPipeline(DaxaRectification,DaxaDepthConfig,std::shared_ptr<DaxaFoundation>,CaptureLease prototype);
    Frame enqueue(CaptureLease); // No wait/readback/application heap allocation. Full ring rejects before upload.
    size_t collect();void drain();
    DaxaDepthReadback readback(const Frame&)const; // Explicit diagnostic boundary only.
    DaxaPreprocessReadback readback_images(const Frame&)const;
    DaxaDepthStereoInput readback_stereo(const Frame&)const;
    DaxaStereoReadback readback_stereo(const Frame&,DaxaStereoReference,bool include_intermediates=true)const;
    void check_stereo_integer(const Frame&,const DaxaPreprocessReadback& expected_images)const;
    uint64_t completed_output_retirements()const;uint64_t requested_observation_bytes()const;uint64_t requested_combined_buffer_bytes()const;
    double last_upload_cpu_ms()const;
    static void validate_pair(const DaxaFrame&,const DaxaStereo::Frame&,const DaxaStereo::Frame&);
    static std::string shader_directory();static std::string shader_identity();
};
}
