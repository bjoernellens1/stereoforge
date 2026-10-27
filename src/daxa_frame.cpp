#include <stereoforge/daxa_preprocess.hpp>
#include <stdexcept>
namespace stereoforge {
namespace {void require(bool v,const char* m){if(!v)throw std::invalid_argument(m);}}
DaxaFrame::DaxaFrame(std::shared_ptr<detail::DaxaFrameOwner> owner,std::shared_ptr<detail::DaxaFrameSlot> slot):owner_(std::move(owner)),slot_(std::move(slot)),generation_(slot_->generation){}
DaxaFrame& DaxaFrame::operator=(DaxaFrame other)noexcept{owner_.swap(other.owner_);slot_.swap(other.slot_);std::swap(generation_,other.generation_);return *this;}
void DaxaFrame::validate()const{require(owner_&&slot_&&owner_->context&&owner_->context->device.get()&&generation_==slot_->generation&&slot_->source&&slot_->calibration,"empty/stale Daxa preprocessing lease");}
void DaxaFrame::wait()const{validate();auto& d=owner_->context->device;d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=slot_->submit,.timeout=30'000'000'000ull});}
bool DaxaFrame::ready()const{validate();return detail::completed_submission(slot_->submit,owner_->context->device.oldest_pending_submit_index());}
uint64_t DaxaFrame::generation()const{validate();return generation_;}uint64_t DaxaFrame::context_id()const{validate();return reinterpret_cast<uintptr_t>(owner_->context->device.get());}
CaptureLease DaxaFrame::source()const{validate();return slot_->source;}std::shared_ptr<const DaxaRectification> DaxaFrame::calibration()const{validate();return slot_->calibration;}
DaxaImageView DaxaFrame::plane(DaxaPreprocessPlane p)const{validate();require(unsigned(p)<8,"invalid preprocessing plane");return DaxaImageView(*this,p);}
DaxaImageView::DaxaImageView(DaxaFrame frame,DaxaPreprocessPlane p):frame_(std::move(frame)),plane_(p){}
daxa::BufferId DaxaImageView::buffer()const{frame_.validate();return frame_.slot_->planes[unsigned(plane_)];}uint32_t DaxaImageView::width()const{auto c=frame_.calibration();return unsigned(plane_)<2?c->eye_width:c->width;}uint32_t DaxaImageView::height()const{auto c=frame_.calibration();return unsigned(plane_)<2?c->source_height:c->height;}size_t DaxaImageView::row_stride_bytes()const{return size_t(width())*4;}size_t DaxaImageView::size_bytes()const{return row_stride_bytes()*height();}DaxaPlaneFormat DaxaImageView::format()const{return unsigned(plane_)<4?DaxaPlaneFormat::Rgba8PackedUint32:unsigned(plane_)<6?DaxaPlaneFormat::Gray8InUint32:DaxaPlaneFormat::SupportUint32;}const std::string& DaxaImageView::calibration_identity()const{frame_.validate();return unsigned(plane_)<2?frame_.slot_->calibration->original_identity:frame_.slot_->calibration->rectified_identity;}
}
