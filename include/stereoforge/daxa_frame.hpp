#pragma once
#include <stereoforge/capture.hpp>
#include <stereoforge/daxa_foundation.hpp>
#include <array>
#include <memory>
#include <string>
namespace stereoforge {
struct DaxaRectification;
enum class DaxaPreprocessPlane:unsigned { OriginalLeft,OriginalRight,RectifiedLeft,RectifiedRight,LeftLuma,RightLuma,LeftSupport,RightSupport };
enum class DaxaPlaneFormat { Rgba8PackedUint32,Gray8InUint32,SupportUint32 };
namespace detail {
struct DaxaFrameOwner {std::shared_ptr<DaxaFoundation> context;virtual ~DaxaFrameOwner()=default;};
struct DaxaFrameSlot {std::array<daxa::BufferId,8> planes{};uint64_t generation=0,submit=0;CaptureLease source;std::shared_ptr<const DaxaRectification> calibration;virtual ~DaxaFrameSlot()=default;};
// Pinned Daxa 00761e80: oldest_pending_submit_index() returns the minimum
// INCLUSIVE completed global ticket across pending queues, or UINT64_MAX when
// idle. Each queue signals the globally assigned submission ticket unchanged.
inline bool completed_submission(uint64_t submitted,uint64_t oldest_pending){return !submitted||submitted<=oldest_pending;}
inline bool available_slot(long references,uint64_t submitted,uint64_t oldest_pending){return references==1&&completed_submission(submitted,oldest_pending);}
}
class DaxaPreprocess;
class DaxaStereo;
class DaxaImageView;
class DaxaFrame {
    std::shared_ptr<detail::DaxaFrameOwner> owner_;
    std::shared_ptr<detail::DaxaFrameSlot> slot_;
    uint64_t generation_=0;
    DaxaFrame(std::shared_ptr<detail::DaxaFrameOwner>,std::shared_ptr<detail::DaxaFrameSlot>);
    void validate()const;
    friend class DaxaPreprocess;friend class DaxaStereo;friend class DaxaImageView;
public:
    DaxaFrame()=default;
    DaxaFrame(const DaxaFrame&)=default;DaxaFrame(DaxaFrame&&)noexcept=default;
    DaxaFrame& operator=(DaxaFrame other)noexcept;
    explicit operator bool()const{return bool(owner_&&slot_);}
    void wait()const;bool ready()const;
    uint64_t generation()const;uint64_t context_id()const;
    CaptureLease source()const;std::shared_ptr<const DaxaRectification> calibration()const;
    DaxaImageView plane(DaxaPreprocessPlane)const;
};
// Concrete Daxa-only buffer views. Construction is private; no RHI/native pointer reinterpretation.
class DaxaImageView {
    DaxaFrame frame_;DaxaPreprocessPlane plane_{};
    DaxaImageView(DaxaFrame,DaxaPreprocessPlane);friend class DaxaFrame;
public:
    daxa::BufferId buffer()const;uint32_t width()const;uint32_t height()const;
    size_t row_stride_bytes()const;size_t size_bytes()const;
    DaxaPlaneFormat format()const;const std::string& calibration_identity()const;
    const DaxaFrame& lease()const{return frame_;}
};
}
