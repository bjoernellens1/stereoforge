#pragma once
#include <stereoforge/contracts.hpp>
#include <stereoforge/daxa_frame.hpp>
#include <stereoforge/daxa_foundation.hpp>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>
namespace stereoforge {
// Positive disparity is x_left-x_right. Left compares L(x) with R(x-d);
// Right compares R(x) with L(x+d). Census planes always retain ordinary L/R order.
enum class DaxaStereoReference { Left,Right };
enum class DaxaRightReferenceMode { IndependentSGM,LeftAggregatedSelection,IndependentLocalCensus };
enum class DaxaStereoEstimator { IndependentSGM,LeftAggregatedSelection,IndependentLocalCensus };
enum class DaxaLRDependency { IndependentReferences,LeftAggregatedCorrelated };
// Reserved native observation value; valid derived observations use this on both layers.
enum class DaxaDepthProvenance : uint32_t { Invalid=0,IndependentStereo=1,LeftAggregatedStereo=5 };
enum class DaxaSgmVariant { Baseline,Wave,Packed,PackedSpecialized,PackedD128,PackedQuad,PackedShuffle,PackedSpecializedOverwrite,PackedSpecializedOverwriteQuad };
enum class DaxaStereoPlane : unsigned { Disparity,Confidence,Validity,Depth,DepthValidity,Costs,Sums,LeftCensus,RightCensus,LeftCensusValid,RightCensusValid,PackedPenalties };
struct DaxaStereoReadback {
    DaxaStereoReference reference=DaxaStereoReference::Left;
    DaxaRightReferenceMode right_reference_mode=DaxaRightReferenceMode::IndependentSGM;
    DaxaStereoEstimator estimator=DaxaStereoEstimator::IndependentSGM;
    std::vector<float> disparity,confidence,depth;
    std::vector<uint32_t> validity,depth_validity,costs,sums;
    std::array<std::vector<uint32_t>,2> census,census_valid;
    std::vector<uint32_t> packed_penalties;
    uint32_t packing_bits=0;
    bool intermediates_reconstructed=false;
};
struct DaxaStereoTiming {
    static constexpr unsigned max_stages=22;
    unsigned stage_count=0;
    double total_gpu_ms=0;
    uint64_t start_tick=0; // Existing first query, shared main-queue timestamp domain.
    double timestamp_period=0;uint32_t timestamp_bits=0;
    std::array<double,max_stages> stage_gpu_ms{};
};
// Scalar diagnostic oracle and independent lane-chunk CPU model. No device use.
std::array<std::vector<uint32_t>,2> daxa_stereo_fixture(StereoConfig);
DaxaStereoReadback daxa_stereo_reference(StereoConfig,std::span<const uint32_t> left,std::span<const uint32_t> right,std::optional<MetricCalibration> metric=std::nullopt,DaxaStereoReference reference=DaxaStereoReference::Left);
// Diagnostic scalar selection from independently computed Left integer sums.
DaxaStereoReadback daxa_left_aggregated_selection_reference(StereoConfig,const DaxaStereoReadback& left);
std::vector<uint32_t> daxa_sgm_packed_model(StereoConfig,std::span<const uint32_t> costs,unsigned wave_size);
std::vector<uint32_t> daxa_wave_min_model(std::span<const uint32_t> lane_values);
struct DaxaQuadSelection {std::vector<uint32_t> best,best_cost,second;};
DaxaQuadSelection daxa_sgm_quad_model(StereoConfig,std::span<const uint32_t> sums,unsigned wave_size);
std::vector<uint32_t> daxa_sgm_d128_model(StereoConfig,std::span<const uint32_t> costs,unsigned wave_size);
std::vector<uint32_t> daxa_sgm_wave_model(StereoConfig,std::span<const uint32_t> costs,unsigned wave_size);
// Single host submission thread only; callers serialize all uses of a shared
// DaxaFoundation, including process/inspect/readback. No concurrent producer API.
// One explicit reference per engine, Left by default; Right requires PackedSpecialized.
// Retained immutable uint32 luma uploaded once at setup, or leased dynamic DaxaFrames.
// No LR consistency, rectification, source capture or production 60 Hz claim.
class DaxaStereo {
    struct Impl;
    struct Slot;
    struct Pair;
    std::shared_ptr<Impl> impl_;
    friend class DaxaPreprocessStereo;friend class DaxaDepthPipeline;
public:
    class Frame {
        std::shared_ptr<Impl> owner_;
        std::shared_ptr<Slot> slot_;
        uint64_t generation_=0;
        friend class DaxaStereo;
        Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>);
        void validate()const;
        friend class DaxaDepthPipeline;std::shared_ptr<const DaxaRectification> calibration_owner()const;
    public:
        Frame()=default;
        Frame(const Frame&)=default;
        Frame(Frame&&) noexcept=default;
        Frame& operator=(Frame other) noexcept;
        void wait()const;
        DaxaStereoTiming timings()const; // Stack C API query results; after wait.
        daxa::BufferId buffer(DaxaStereoPlane)const; // Same-context borrowed ID; retain this frame.
        uint64_t generation()const;
        uint64_t context_id()const;
        uint32_t packing_bits()const;
        DaxaStereoReference reference()const;
        DaxaRightReferenceMode right_reference_mode()const;
        DaxaStereoEstimator estimator()const;
        CaptureLease source()const;const std::string& calibration_identity()const;
    };
    static std::array<uint32_t,3> dispatch_grid(uint64_t groups,std::array<uint32_t,3> limits);
    static void validate_shape(StereoConfig,unsigned slots,std::optional<MetricCalibration>,DaxaSgmVariant variant=DaxaSgmVariant::Baseline,DaxaStereoReference reference=DaxaStereoReference::Left); // CPU-only before input allocation.
    static void validate(StereoConfig,std::span<const uint32_t>,std::span<const uint32_t>,unsigned slots,std::optional<MetricCalibration>,DaxaSgmVariant variant=DaxaSgmVariant::Baseline,DaxaStereoReference reference=DaxaStereoReference::Left);
    DaxaStereo(StereoConfig,std::span<const uint32_t> left,std::span<const uint32_t> right,DaxaSgmVariant,unsigned slots=1,bool validation=false,std::optional<MetricCalibration> metric=std::nullopt,std::shared_ptr<DaxaFoundation> owner={},uint32_t required_wave_size=0,DaxaStereoReference reference=DaxaStereoReference::Left);
    DaxaStereo(StereoConfig,const DaxaFrame& prototype,DaxaSgmVariant,unsigned slots,std::optional<MetricCalibration>,std::shared_ptr<DaxaFoundation>,uint32_t required_wave_size=0,DaxaStereoReference reference=DaxaStereoReference::Left);
    Frame process(const DaxaFrame&);size_t collect_completed_inputs();
    Frame process(); // No wait/full readback/own allocations; rejects retained/in-flight slots.
    DaxaStereoReadback readback(Frame const&,bool include_costs_and_sums=false)const;
    std::array<uint32_t,4> inspect(Frame const&)const; // Separate small GPU consumer/readback.
    const char* stage_name(unsigned)const;
    std::string device_report()const;
    uint64_t allocated_bytes()const;
    uint32_t required_wave_size()const;
    uint32_t packing_bits()const;
    DaxaStereoReference reference()const;
    static uint32_t packed_bits(StereoConfig);
    unsigned validation_errors()const;
    static const char* variant_name(DaxaSgmVariant);
    static std::string shader_identity();
    static std::string overwrite_quad_shader_identity();
    static std::string shader_directory();
    static std::string specialized_shader_identity(); // Actual compiled Parameters:shared-body SHA256.
    static std::string derived_shader_identity();
    static std::string layered_shader_identity(); // Actual compiled layered wrapper SHA256.
private:
    DaxaStereo(StereoConfig,const DaxaFrame&,DaxaSgmVariant,unsigned,std::optional<MetricCalibration>,std::shared_ptr<DaxaFoundation>,uint32_t,DaxaStereoReference,bool shared_census,const DaxaStereo* descriptor_owner,DaxaRightReferenceMode mode=DaxaRightReferenceMode::IndependentSGM);
    // Depth-only combined scheduling. Setup owns no Frame/producer marker.
    void setup_pair(DaxaStereo& right,bool layered=false,bool shared_census=false);
    unsigned pair_slot_at(uint64_t oldest)const; // Both aligned slots available, before upstream upload.
    std::array<Frame,2> process_pair_at(const DaxaFrame&,uint64_t oldest,unsigned slot);
    DaxaStereoTiming pair_timings(const Frame& left,const Frame& right)const;
    Frame process_at(const DaxaFrame*,uint64_t oldest);
    size_t collect_completed_inputs_at(uint64_t oldest);
};
}
