#pragma once
#include <stereoforge/daxa_frame.hpp>
namespace stereoforge {
class DaxaDepthPipeline;class DaxaTemporal;class DaxaRefinement;
enum class DaxaObservationPlane:unsigned { Depth,Confidence,Validity,Provenance,SourceIds,AgeNs };
enum class DaxaObservationImage:unsigned { OriginalLeft,OriginalRight,RectifiedLeft,RectifiedRight };
enum class DaxaObservationAncestryKind { SingleMeasured, BoundedContributors };
enum class DaxaObservationAncestryPlane:unsigned { ContributorIds,Count,OldestAgeNs,Reuse,OriginMask };
// Borrowed sidecars; retain the common observation lease while consuming them.
// capacity is the configured union bound; ContributorIds storage always has stride4.
class DaxaObservationAncestryView {
    DaxaObservationAncestryKind kind_=DaxaObservationAncestryKind::SingleMeasured;
    uint32_t capacity_=1;size_t observations_=0;std::array<daxa::BufferId,5> buffers_{};
    DaxaObservationAncestryView(size_t n):observations_(n){}
    DaxaObservationAncestryView(size_t n,uint32_t capacity,std::array<daxa::BufferId,5> buffers):kind_(DaxaObservationAncestryKind::BoundedContributors),capacity_(capacity),observations_(n),buffers_(buffers){}
    friend class DaxaObservationFrame;friend class DaxaTemporal;
public:
    DaxaObservationAncestryView()=default;
    DaxaObservationAncestryKind kind()const{return kind_;}uint32_t capacity()const{return capacity_;}
    daxa::BufferId buffer(DaxaObservationAncestryPlane)const;size_t size_bytes(DaxaObservationAncestryPlane)const;
};
// Borrowed RGB descriptor. The observation lease must outlive use of its buffer.
class DaxaObservationImageView {
    daxa::BufferId buffer_{};uint32_t width_=0,height_=0;
    DaxaObservationImageView(daxa::BufferId buffer,uint32_t width,uint32_t height):buffer_(buffer),width_(width),height_(height){}
    friend class DaxaDepthPipeline;friend class DaxaTemporal;friend class DaxaRefinement;
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
    DaxaObservationAncestryView ancestry_;
    DaxaObservationFrame(std::shared_ptr<DaxaFoundation>,std::shared_ptr<void> owner,std::shared_ptr<void> lease,const uint64_t* live_generation,uint64_t generation,uint64_t submit,std::shared_ptr<const DaxaRectification>,CaptureLease,std::array<daxa::BufferId,6>,std::array<DaxaObservationImageView,4>,DaxaObservationAncestryView={});
    void validate()const;
    friend class DaxaDepthPipeline;friend class DaxaTemporal;friend class DaxaRefinement;
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
    DaxaObservationAncestryView ancestry()const;
};
}
