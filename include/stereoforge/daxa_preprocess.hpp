#pragma once
#include <stereoforge/daxa_frame.hpp>
#include <stereoforge/daxa_stereo.hpp>
#include <filesystem>
#include <vector>
namespace stereoforge {
inline constexpr const char* daxa_preprocess_profile="opencv414_yuyv20_remap32_rte_rgb_halfup_gray15";
void validate_daxa_calibration_pose(const std::array<double,16>&); // Existing Python Calibration rigid/baseline contract.
struct DaxaRectification {
    uint32_t eye_width=0,source_height=0,stride_bytes=0,width=0,height=0;
    std::string original_identity,rectified_identity;
    std::array<std::vector<float>,4> maps; // Lx,Ly,Rx,Ry; finite CV_32FC1, source pixel coordinates.
    uint64_t requested_bytes(unsigned slots)const;
    void validate(unsigned slots=1)const;void validate_source(const CaptureFrame&)const;
};
struct DaxaPreprocessFixture {
    DaxaRectification rectification;std::filesystem::path capture_directory;
    uint64_t record_ordinal=0;std::string capture_sha256;
};
DaxaPreprocessFixture load_daxa_preprocess_fixture(const std::filesystem::path&);
struct DaxaPreprocessReadback {std::array<std::vector<uint32_t>,8> planes;};
std::string daxa_preprocess_file_sha256(const std::filesystem::path&);
DaxaPreprocessReadback load_daxa_preprocess_expected(const std::filesystem::path&);
DaxaPreprocessReadback daxa_preprocess_reference(const CaptureFrame&,const DaxaRectification&);
struct DaxaPreprocessTiming {double upload_ms=0,decode_ms=0,rectify_ms=0,total_ms=0;};
class DaxaPreprocess {
    struct Impl;std::shared_ptr<Impl> impl_;
    friend class DaxaPreprocessStereo;
    DaxaFrame enqueue_at(CaptureLease,uint64_t oldest);size_t collect_completed_sources_at(uint64_t oldest);
public:
    DaxaPreprocess(DaxaRectification,unsigned slots,std::shared_ptr<DaxaFoundation>);
    DaxaFrame enqueue(CaptureLease); // Fixed staging/planes/params; no wait or heap allocation.
    size_t collect_completed_sources(); // One completion snapshot, bounded scan.
    DaxaPreprocessTiming timings(const DaxaFrame&)const;
    DaxaPreprocessReadback readback(const DaxaFrame&)const; // Explicit diagnostic only.
    std::shared_ptr<DaxaFoundation> context()const;
    uint64_t completed_source_retirements()const;double last_upload_cpu_ms()const;
    static std::string shader_directory();static std::string shader_identity();
};
// Integration owner establishes all graphs/resources with one prototype during
// setup, then drops that prototype. collect() must precede the next capture read.
class DaxaPreprocessStereo {
    DaxaPreprocess preprocess_;std::unique_ptr<DaxaStereo> stereo_;uint64_t retired_inputs_=0;
public:
    DaxaPreprocessStereo(DaxaRectification,unsigned slots,std::shared_ptr<DaxaFoundation>,CaptureLease prototype,StereoConfig,DaxaSgmVariant=DaxaSgmVariant::PackedSpecialized,uint32_t wave_size=32);
    size_t collect();void drain();
    DaxaStereo::Frame enqueue(CaptureLease,DaxaFrame* diagnostic_input=nullptr);
    uint64_t completed_input_retirements()const{return retired_inputs_;}
    DaxaPreprocess& preprocess(){return preprocess_;}DaxaStereo& stereo(){return *stereo_;}
};
}
