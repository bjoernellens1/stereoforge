#pragma once
#include <stereoforge/daxa_frame.hpp>
namespace stereoforge {
class DaxaDepthPipeline;class DaxaTemporal;
enum class DaxaObservationPlane:unsigned { Depth,Confidence,Validity,Provenance,SourceIds,AgeNs };
enum class DaxaObservationImage:unsigned { OriginalLeft,OriginalRight,RectifiedLeft,RectifiedRight };
// Borrowed RGB descriptor. The observation lease must outlive use of its buffer.
class DaxaObservationImageView {
    daxa::BufferId buffer_{};uint32_t width_=0,height_=0;
    DaxaObservationImageView(daxa::BufferId buffer,uint32_t width,uint32_t height):buffer_(buffer),width_(width),height_(height){}
    friend class DaxaDepthPipeline;friend class DaxaTemporal;
public:
    DaxaObservationImageView()=default;
    daxa::BufferId buffer()const{return buffer_;}uint32_t width()const{return width_;}uint32_t height()const{return height_;}
    size_t row_stride_bytes()const{return size_t(width_)*4;}size_t size_bytes()const{return row_stride_bytes()*height_;}
    DaxaPlaneFormat format()const{return DaxaPlaneFormat::Rgba8PackedUint32;}
};
// Contiguous [eye][y][x] axial-Z metric planes on the current rectified grid.
// source() identifies current RGB/grid capture time and clock, not pixel ancestry.
// Borrowed buffers remain valid while this lease retains the producer slot.
class DaxaObservationFrame {
    // Reverse destruction releases the child slot before producer and context.
    std::shared_ptr<DaxaFoundation> context_;
    std::shared_ptr<void> owner_,lease_;
    std::shared_ptr<const DaxaRectification> maps_;CaptureLease source_;
    const uint64_t* live_generation_=nullptr;
    uint64_t generation_=0,submit_=0;uint32_t width_=0,height_=0;
    std::array<daxa::BufferId,6> planes_{};
    std::array<DaxaObservationImageView,4> images_{};
    DaxaObservationFrame(std::shared_ptr<DaxaFoundation>,std::shared_ptr<void> owner,std::shared_ptr<void> lease,const uint64_t* live_generation,uint64_t generation,uint64_t submit,std::shared_ptr<const DaxaRectification>,CaptureLease,std::array<daxa::BufferId,6>,std::array<DaxaObservationImageView,4>);
    void validate()const;
    friend class DaxaDepthPipeline;friend class DaxaTemporal;
public:
    DaxaObservationFrame()=default;DaxaObservationFrame(const DaxaObservationFrame&)=default;DaxaObservationFrame(DaxaObservationFrame&&)noexcept=default;
    DaxaObservationFrame& operator=(DaxaObservationFrame)noexcept;
    explicit operator bool()const{return bool(context_&&owner_&&lease_);}
    void wait()const;bool ready()const;
    uint32_t width()const;uint32_t height()const;uint32_t layers()const{validate();return 2;}
    uint64_t generation()const;uint64_t submit_ticket()const;uint64_t context_id()const;
    std::shared_ptr<DaxaFoundation> context()const;
    std::shared_ptr<const DaxaRectification> map_owner()const;
    std::shared_ptr<const DaxaRectification> calibration()const{return map_owner();}
    CaptureLease source()const;
    daxa::BufferId buffer(DaxaObservationPlane)const;daxa::BufferId plane(DaxaObservationPlane p)const{return buffer(p);}
    size_t size_bytes(DaxaObservationPlane)const;
    DaxaObservationImageView image(DaxaObservationImage)const;
};
}
