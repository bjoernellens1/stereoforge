#pragma once
#include <stereoforge/contracts.hpp>
#include <stereoforge/daxa_foundation.hpp>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>
namespace stereoforge {
enum class DaxaSgmVariant { Baseline,Wave,Packed,PackedSpecialized,PackedD128,PackedQuad,PackedShuffle };
enum class DaxaStereoPlane : unsigned { Disparity,Confidence,Validity,Depth,DepthValidity,Costs,Sums,LeftCensus,RightCensus,LeftCensusValid,RightCensusValid,PackedPenalties };
struct DaxaStereoReadback {
    std::vector<float> disparity,confidence,depth;
    std::vector<uint32_t> validity,depth_validity,costs,sums;
    std::array<std::vector<uint32_t>,2> census,census_valid;
    std::vector<uint32_t> packed_penalties;
    uint32_t packing_bits=0;
    bool intermediates_reconstructed=false;
};
struct DaxaStereoTiming {
    static constexpr unsigned max_stages=13;
    unsigned stage_count=0;
    double total_gpu_ms=0;
    std::array<double,max_stages> stage_gpu_ms{};
};
// Scalar diagnostic oracle and independent lane-chunk CPU model. No device use.
std::array<std::vector<uint32_t>,2> daxa_stereo_fixture(StereoConfig);
DaxaStereoReadback daxa_stereo_reference(StereoConfig,std::span<const uint32_t> left,std::span<const uint32_t> right,std::optional<MetricCalibration> metric=std::nullopt);
std::vector<uint32_t> daxa_sgm_packed_model(StereoConfig,std::span<const uint32_t> costs,unsigned wave_size);
std::vector<uint32_t> daxa_wave_min_model(std::span<const uint32_t> lane_values);
struct DaxaQuadSelection {std::vector<uint32_t> best,best_cost,second;};
DaxaQuadSelection daxa_sgm_quad_model(StereoConfig,std::span<const uint32_t> sums,unsigned wave_size);
std::vector<uint32_t> daxa_sgm_d128_model(StereoConfig,std::span<const uint32_t> costs,unsigned wave_size);
std::vector<uint32_t> daxa_sgm_wave_model(StereoConfig,std::span<const uint32_t> costs,unsigned wave_size);
// Single host submission thread only; callers serialize all uses of a shared
// DaxaFoundation, including process/inspect/readback. No concurrent producer API.
// Single reference only: retained immutable uint32 luma uploaded once at setup.
// No LR consistency, rectification, source capture or production 60 Hz claim.
class DaxaStereo {
    struct Impl;
    struct Slot;
    std::shared_ptr<Impl> impl_;
public:
    class Frame {
        std::shared_ptr<Impl> owner_;
        std::shared_ptr<Slot> slot_;
        uint64_t generation_=0;
        friend class DaxaStereo;
        Frame(std::shared_ptr<Impl>,std::shared_ptr<Slot>);
        void validate()const;
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
    };
    static std::array<uint32_t,3> dispatch_grid(uint64_t groups,std::array<uint32_t,3> limits);
    static void validate_shape(StereoConfig,unsigned slots,std::optional<MetricCalibration>,DaxaSgmVariant variant=DaxaSgmVariant::Baseline); // CPU-only before input allocation.
    static void validate(StereoConfig,std::span<const uint32_t>,std::span<const uint32_t>,unsigned slots,std::optional<MetricCalibration>,DaxaSgmVariant variant=DaxaSgmVariant::Baseline);
    DaxaStereo(StereoConfig,std::span<const uint32_t> left,std::span<const uint32_t> right,DaxaSgmVariant,unsigned slots=1,bool validation=false,std::optional<MetricCalibration> metric=std::nullopt,std::shared_ptr<DaxaFoundation> owner={},uint32_t required_wave_size=0);
    Frame process(); // No wait/full readback/own allocations; rejects retained/in-flight slots.
    DaxaStereoReadback readback(Frame const&,bool include_costs_and_sums=false)const;
    std::array<uint32_t,4> inspect(Frame const&)const; // Separate small GPU consumer/readback.
    const char* stage_name(unsigned)const;
    std::string device_report()const;
    uint64_t allocated_bytes()const;
    uint32_t required_wave_size()const;
    uint32_t packing_bits()const;
    static uint32_t packed_bits(StereoConfig);
    unsigned validation_errors()const;
    static const char* variant_name(DaxaSgmVariant);
    static std::string shader_identity();
    static std::string shader_directory();
};
}
