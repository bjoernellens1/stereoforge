#pragma once
#include <stereoforge/canonical_rgb.hpp>
#include <stereoforge/daxa_preprocess.hpp>
namespace stereoforge {
struct DaxaRectifiedInputConfig {unsigned slots=2;bool coarse_2x=true;void validate()const;};
struct DaxaRectifiedInputReference {DaxaPreprocessReadback full;std::optional<DaxaPreprocessReadback> coarse;};
void validate_daxa_rectified_input_geometry(const DaxaRectification&,const std::optional<DaxaRectification>&,const DaxaRectifiedInputConfig&,const RectifiedRgbFrame&);
uint64_t daxa_rectified_input_required_bytes(const DaxaRectification&,const std::optional<DaxaRectification>&,const DaxaRectifiedInputConfig&,const RectifiedRgbFrame&);
DaxaRectifiedInputReference daxa_rectified_input_reference(const RectifiedRgbFrame&,const DaxaRectification&,const std::optional<DaxaRectification>&,DaxaRectifiedInputConfig={});
struct DaxaRectifiedInputTiming {double upload_ms=0,original_ms=0,rectified_ms=0,coarse_ms=0,total_ms=0;uint64_t start_tick=0,end_tick=0;double timestamp_period=0;uint32_t timestamp_bits=0;};
class DaxaRectifiedInput {
    struct Impl;std::shared_ptr<Impl> impl_;
public:
    class Frame {
        std::shared_ptr<Impl> owner_;std::shared_ptr<void> slot_;DaxaFrame full_,coarse_;
        Frame(std::shared_ptr<Impl>,std::shared_ptr<void>,DaxaFrame,DaxaFrame);friend class DaxaRectifiedInput;
    public:
        Frame()=default;
        DaxaFrame full()const;DaxaFrame coarse()const;
        explicit operator bool()const{return bool(full_);}
    };
    DaxaRectifiedInput(DaxaRectification,std::optional<DaxaRectification>,DaxaRectifiedInputConfig,std::shared_ptr<DaxaFoundation>,const RectifiedRgbFrame&);
    Frame enqueue(const RectifiedRgbFrame&);size_t collect();
    DaxaPreprocessReadback readback(const DaxaFrame&)const;
    DaxaRectifiedInputTiming timings(const Frame&)const;
    uint64_t requested_bytes()const;double last_upload_cpu_ms()const;
    std::shared_ptr<DaxaFoundation> context()const;
    static std::string shader_directory();static std::string shader_identity();
};
}
