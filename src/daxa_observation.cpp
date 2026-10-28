#include <stereoforge/daxa_observation.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <stereoforge/canonical_rgb.hpp>
#include <stdexcept>
#include <utility>
namespace stereoforge {
namespace {void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}}
DaxaObservationFrame::DaxaObservationFrame(std::shared_ptr<DaxaFoundation> context,std::shared_ptr<void> owner,std::shared_ptr<void> lease,const uint64_t* live_generation,uint64_t generation,uint64_t submit,std::shared_ptr<const DaxaRectification> maps,CaptureLease source,std::array<daxa::BufferId,6> planes,std::array<DaxaObservationImageView,4> images,DaxaObservationAncestryView ancestry):context_(std::move(context)),owner_(std::move(owner)),lease_(std::move(lease)),maps_(std::move(maps)),source_(std::move(source)),live_generation_(live_generation),generation_(generation),submit_(submit),planes_(planes),images_(images),ancestry_(ancestry){
    validate();require(context_->device.get()&&maps_&&source_&&submit_,"observation requires submitted producer/context/maps/source");width_=maps_->width;height_=maps_->height;if(ancestry_.kind()==DaxaObservationAncestryKind::SingleMeasured)ancestry_=DaxaObservationAncestryView(size_t(width_)*height_*2);require(width_&&height_&&source_->source_width==2*maps_->eye_width&&source_->height==maps_->source_height&&(source_->source_kind==CaptureSourceKind::CanonicalRgbMetadata?canonical_rgb_grid_association(*source_,maps_->eye_width,maps_->source_height,maps_->width,maps_->height,maps_->original_identity,maps_->rectified_identity):source_->stride_bytes==maps_->stride_bytes)&&source_->calibration_identity==maps_->original_identity,"observation source/grid association mismatch");
    for(auto id:planes_)require(!id.is_empty(),"missing observation plane");
    for(unsigned i=0;i<4;++i)require(!images_[i].buffer().is_empty()&&images_[i].width()==(i<2?maps_->eye_width:width_)&&images_[i].height()==(i<2?maps_->source_height:height_),"observation RGB shape mismatch");

}
void DaxaObservationFrame::validate()const{require(context_&&owner_&&lease_&&live_generation_&&generation_==*live_generation_,"empty/stale observation frame");}
DaxaObservationFrame& DaxaObservationFrame::operator=(DaxaObservationFrame other)noexcept{context_.swap(other.context_);owner_.swap(other.owner_);lease_.swap(other.lease_);maps_.swap(other.maps_);source_.swap(other.source_);std::swap(live_generation_,other.live_generation_);std::swap(generation_,other.generation_);std::swap(submit_,other.submit_);std::swap(width_,other.width_);std::swap(height_,other.height_);planes_.swap(other.planes_);images_.swap(other.images_);std::swap(ancestry_,other.ancestry_);return *this;}
void DaxaObservationFrame::wait()const{validate();context_->device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=submit_,.timeout=30'000'000'000ull});}
bool DaxaObservationFrame::ready()const{validate();return detail::completed_submission(submit_,context_->device.oldest_pending_submit_index());}
uint32_t DaxaObservationFrame::width()const{validate();return width_;}uint32_t DaxaObservationFrame::height()const{validate();return height_;}
uint64_t DaxaObservationFrame::generation()const{validate();return generation_;}uint64_t DaxaObservationFrame::submit_ticket()const{validate();return submit_;}uint64_t DaxaObservationFrame::context_id()const{validate();return reinterpret_cast<uintptr_t>(context_->device.get());}
std::shared_ptr<DaxaFoundation> DaxaObservationFrame::context()const{validate();return context_;}
std::shared_ptr<const DaxaRectification> DaxaObservationFrame::map_owner()const{validate();return maps_;}CaptureLease DaxaObservationFrame::source()const{validate();return source_;}
daxa::BufferId DaxaObservationFrame::buffer(DaxaObservationPlane p)const{validate();require(unsigned(p)<6,"invalid observation plane");return planes_[unsigned(p)];}
size_t DaxaObservationFrame::size_bytes(DaxaObservationPlane p)const{validate();require(unsigned(p)<6,"invalid observation plane");return size_t(width_)*height_*2*(unsigned(p)<4?4:8);}
DaxaObservationImageView DaxaObservationFrame::image(DaxaObservationImage p)const{validate();require(unsigned(p)<4,"invalid observation RGB view");return images_[unsigned(p)];}
DaxaObservationAncestryView DaxaObservationFrame::ancestry()const{validate();return ancestry_;}
daxa::BufferId DaxaObservationAncestryView::buffer(DaxaObservationAncestryPlane p)const{require(unsigned(p)<5&&kind_==DaxaObservationAncestryKind::BoundedContributors,"ancestry sidecar unavailable");return buffers_[unsigned(p)];}
size_t DaxaObservationAncestryView::size_bytes(DaxaObservationAncestryPlane p)const{buffer(p);return observations_*(p==DaxaObservationAncestryPlane::ContributorIds?32:p==DaxaObservationAncestryPlane::OldestAgeNs?8:4);}
}
