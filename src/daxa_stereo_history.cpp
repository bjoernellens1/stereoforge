#include <stereoforge/daxa_stereo_history.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <thread>
namespace stereoforge {
namespace {
constexpr std::array<unsigned,7> pixel_bytes{4,4,8,8,8,16,16};
constexpr std::array<unsigned,7> pixel_offsets{0,4,8,16,24,32,48};
void require(bool ok,char const* message){if(!ok)throw std::invalid_argument(message);}
bool same_domain(StereoFrameIdentity const& a,StereoFrameIdentity const& b) {
    return a.device==b.device&&a.original_calibration==b.original_calibration&&a.rectified_calibration==b.rectified_calibration&&a.rig_session==b.rig_session&&a.acquisition_clock==b.acquisition_clock&&a.ingress_clock==b.ingress_clock;
}
bool source_matches(DaxaObservationFrame const& input,StereoWorldPose const& pose) {
    auto source=input.source();auto maps=input.calibration();auto const& id=pose.source;
    return source&&maps&&id.frame_id==source->frame_id&&id.acquisition_timestamp_ns==source->camera_timestamp_ns&&id.ingress_timestamp_ns==source->host_ingress_timestamp_ns&&
        id.device==source->device_identity&&id.acquisition_clock==source->camera_clock_domain&&id.ingress_clock==source->host_clock_domain&&
        id.original_calibration==maps->original_identity&&id.original_calibration==source->calibration_identity&&id.rectified_calibration==maps->rectified_identity;
}

}
void DaxaStereoHistoryConfig::validate()const {
    require(capacity>=1&&capacity<=4&&max_age_ns>0&&max_age_ns<=200000000&&max_bytes>0,"history requires1..4pairs, age<=200ms and bounded memory");
}
struct DaxaStereoHistory::Slot {
    std::shared_ptr<DaxaFoundation> context;
    std::shared_ptr<DaxaRectification const> maps;
    daxa::BufferId storage{};
    StereoWorldPose pose;
    DaxaObservationFrame producer;
    uint64_t generation=0,copy=0,read=0;
    std::thread::id thread=std::this_thread::get_id();
    ~Slot()noexcept {
        try {
            auto& d=context->device;auto ticket=std::max(copy,read);
            if(!detail::completed_submission(ticket,d.oldest_pending_submit_index()))d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=ticket,.timeout=30'000'000'000ull});
            if(!storage.is_empty())d.destroy_buffer(storage);d.collect_garbage();
        }catch(...){std::terminate();}
    }
};
struct DaxaStereoHistory::Impl {
    std::shared_ptr<DaxaFoundation> context;
    std::shared_ptr<DaxaRectification const> maps;
    DaxaStereoHistoryConfig config;DaxaStereoHistoryStats stats;
    std::vector<std::shared_ptr<Slot>> slots;std::vector<unsigned> active;
    std::optional<StereoFrameIdentity> last;
    uint64_t bytes=0;bool poisoned=false;
    std::thread::id thread=std::this_thread::get_id();
    void check()const{require(thread==std::this_thread::get_id(),"history requires shared submission thread");require(!poisoned,"history GPU submission failed; reconstruct history");}
};
DaxaStereoHistory::DaxaStereoHistory(std::shared_ptr<DaxaFoundation> context,DaxaObservationFrame const& prototype,DaxaStereoHistoryConfig config) {
    config.validate();require(context&&context->device.get()&&bool(prototype),"history requires resident context/prototype");
    require(prototype.context()==context,"foreign history prototype");auto maps=prototype.calibration();
    require(maps&&maps->width==prototype.width()&&maps->height==prototype.height(),"history grid/calibration mismatch");
    uint64_t n=uint64_t(maps->width)*maps->height;require(n&&n<=std::numeric_limits<uint64_t>::max()/64/config.capacity,"history size overflow");
    require(n*64*config.capacity<=config.max_bytes,"history memory budget exceeded");
    auto e=std::make_unique<Impl>();e->context=std::move(context);e->maps=std::move(maps);e->config=config;e->bytes=n*64*config.capacity;
    e->slots.reserve(config.capacity);e->active.reserve(config.capacity);
    for(unsigned i=0;i<config.capacity;++i) {
        auto slot=std::make_shared<Slot>();slot->context=e->context;slot->maps=e->maps;
        slot->storage=e->context->device.create_buffer({.size=n*64,.name="packed owned stereo history"});
        e->slots.push_back(std::move(slot));
    }
    impl_=std::move(e);
}
DaxaStereoHistory::~DaxaStereoHistory()=default;
DaxaStereoHistory::Frame::Frame(std::shared_ptr<Slot> slot):slot_(std::move(slot)),generation_(slot_->generation){}
void DaxaStereoHistory::Frame::validate()const{require(slot_&&generation_&&generation_==slot_->generation,"empty/stale stereo history lease");}
void DaxaStereoHistory::Frame::wait()const{validate();slot_->context->device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=slot_->copy,.timeout=30'000'000'000ull});}
bool DaxaStereoHistory::Frame::ready()const{validate();return detail::completed_submission(slot_->copy,slot_->context->device.oldest_pending_submit_index());}
StereoWorldPose const& DaxaStereoHistory::Frame::pose()const{validate();return slot_->pose;}
daxa::BufferId DaxaStereoHistory::Frame::buffer()const{validate();return slot_->storage;}
DaxaStereoHistoryView DaxaStereoHistory::Frame::plane(DaxaStereoHistoryPlane plane)const {
    validate();auto p=unsigned(plane);require(p<7,"invalid history plane");uint64_t n=uint64_t(slot_->maps->width)*slot_->maps->height;
    return {slot_->storage,n*pixel_offsets[p],n*pixel_bytes[p]};
}
uint64_t DaxaStereoHistory::Frame::copy_ticket()const{validate();return slot_->copy;}
void DaxaStereoHistory::Frame::mark_consumed(uint64_t ticket)const {
    validate();require(slot_->thread==std::this_thread::get_id(),"history read submission requires owner thread");
    require(ticket>=slot_->copy&&ticket<=slot_->context->device.latest_queue_submit_index(daxa::QUEUE_MAIN),"invalid history consumer ticket");slot_->read=std::max(slot_->read,ticket);
}
std::size_t DaxaStereoHistory::collect() {
    auto& e=*impl_;e.check();auto boundary=e.context->device.oldest_pending_submit_index();std::size_t retired=0;
    for(auto const& slot:e.slots)if(slot->producer&&detail::completed_submission(slot->copy,boundary)){slot->producer={};++retired;}
    return retired;
}
void DaxaStereoHistory::clear(){auto& e=*impl_;e.check();e.active.clear();e.last.reset();++e.stats.resets;}
std::vector<DaxaStereoHistory::Selection> DaxaStereoHistory::select(StereoWorldPose const& pose)const {
    auto const& e=*impl_;e.check();std::vector<Selection> selected;
    if(!stereo_pose_usable(pose,e.config.allow_unqualified_research))return selected;
    for(auto i:e.active) {
        auto const& slot=e.slots[i];auto const& old=slot->pose;auto const& a=old.source;auto const& b=pose.source;
        if(old.trajectory_id!=pose.trajectory_id||old.epoch!=pose.epoch||!same_domain(a,b)||a.frame_id>=b.frame_id||a.ingress_timestamp_ns>=b.ingress_timestamp_ns)continue;
        uint64_t age=b.ingress_timestamp_ns-a.ingress_timestamp_ns;auto bound=stereo_acquisition_age_bound(a,b);
        if(a.acquisition_timestamp_ns&&b.acquisition_timestamp_ns<=a.acquisition_timestamp_ns)continue;
        if((a.clock_mapping_uncertainty_ns&&*a.clock_mapping_uncertainty_ns>e.config.max_age_ns)||
           (b.clock_mapping_uncertainty_ns&&*b.clock_mapping_uncertainty_ns>e.config.max_age_ns)||
           (a.clock_mapping_uncertainty_ns&&b.clock_mapping_uncertainty_ns&&!bound))continue;
        if(age>e.config.max_age_ns||(bound&&*bound>e.config.max_age_ns)||(!e.config.allow_unqualified_research&&!bound))continue;
        selected.push_back({Frame(slot),age,bound});
    }
    return selected;
}
std::optional<DaxaStereoHistory::Frame> DaxaStereoHistory::store(DaxaObservationFrame const& input,StereoWorldPose const& pose) {
    auto& e=*impl_;e.check();require(bool(input)&&input.context()==e.context&&input.calibration()==e.maps,"foreign history input context/calibration");
    require(input.width()==e.maps->width&&input.height()==e.maps->height&&source_matches(input,pose),"history pose must match exact source and grid");
    pose.source.validate();if(e.last&&same_domain(*e.last,pose.source))require(pose.source.frame_id>e.last->frame_id&&pose.source.ingress_timestamp_ns>e.last->ingress_timestamp_ns,"duplicate/late history source");
    ++e.stats.offered;
    if(!stereo_pose_usable(pose,e.config.allow_unqualified_research)) {
        auto next=pose.source;e.active.clear();e.last=std::move(next);++e.stats.pose_rejected;++e.stats.resets;return {};
    }
    auto& d=e.context->device;auto boundary=d.oldest_pending_submit_index();unsigned target=e.config.capacity;
    // Prefer inactive storage, then the oldest safe active slot. Export leases
    // and read/copy tickets prohibit overwrite, even after logical eviction.
    for(unsigned pass=0;pass<2&&target==e.config.capacity;++pass)for(unsigned j=0;j<e.config.capacity;++j) {
        unsigned i=pass? (j<e.active.size()?e.active[j]:e.config.capacity):j;if(i==e.config.capacity)continue;
        if(!pass&&std::find(e.active.begin(),e.active.end(),i)!=e.active.end())continue;
        auto const& slot=e.slots[i];if(slot.use_count()==1&&detail::completed_submission(std::max(slot->copy,slot->read),boundary)){target=i;break;}
    }
    if(target==e.config.capacity){++e.stats.capacity_skipped;return {};}
    auto candidate=pose;auto next=pose.source;auto& slot=e.slots[target];require(slot->generation!=std::numeric_limits<uint64_t>::max(),"history generation exhausted");
    std::array<daxa::BufferId,7> source{input.image(DaxaObservationImage::RectifiedLeft).buffer(),input.image(DaxaObservationImage::RectifiedRight).buffer(),
        input.buffer(DaxaObservationPlane::Depth),input.buffer(DaxaObservationPlane::Validity),input.buffer(DaxaObservationPlane::Provenance),input.buffer(DaxaObservationPlane::SourceIds),input.buffer(DaxaObservationPlane::AgeNs)};
    uint64_t n=uint64_t(e.maps->width)*e.maps->height;
    for(unsigned p=0;p<7;++p)require(d.buffer_info(source[p]).value().size>=n*pixel_bytes[p],"short history source buffer");
    auto before=d.latest_queue_submit_index(daxa::QUEUE_MAIN);slot->producer=input;
    try {
        auto rec=d.create_command_recorder({.name="resident stereo pair history copy"});
        rec.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE|daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ|daxa::AccessConsts::TRANSFER_WRITE});
        for(unsigned p=0;p<7;++p)rec.copy_buffer_to_buffer({.src_buffer=source[p],.dst_buffer=slot->storage,.dst_offset=n*pixel_offsets[p],.size=n*pixel_bytes[p]});
        auto commands=rec.complete_current_commands();slot->copy=d.submit_commands({.command_lists={&commands,1}});
        require(slot->copy>before,"history copy did not advance queue");
    }catch(...) {
        e.poisoned=true;e.active.clear();try {slot->copy=std::max(slot->copy,d.latest_queue_submit_index(daxa::QUEUE_MAIN));}catch(...){std::terminate();}throw;
    }
    for(auto i=e.active.begin();i!=e.active.end();) {
        auto const& old=e.slots[*i]->pose;auto const& a=old.source;auto const& b=pose.source;
        bool reset=old.trajectory_id!=pose.trajectory_id||old.epoch!=pose.epoch||!same_domain(a,b);
        bool expired=b.ingress_timestamp_ns>a.ingress_timestamp_ns&&b.ingress_timestamp_ns-a.ingress_timestamp_ns>e.config.max_age_ns;
        if(*i==target||reset||expired){if(expired)++e.stats.expired;i=e.active.erase(i);}else ++i;
    }
    slot->pose=std::move(candidate);slot->read=0;++slot->generation;e.active.push_back(target);e.last=std::move(next);++e.stats.copied;
    return Frame(slot);
}
bool DaxaStereoHistory::compatible(DaxaObservationFrame const& prototype)const{impl_->check();return bool(prototype)&&prototype.context()==impl_->context&&prototype.calibration()==impl_->maps;}
DaxaStereoHistoryStats DaxaStereoHistory::stats()const{impl_->check();return impl_->stats;}
uint64_t DaxaStereoHistory::requested_bytes()const{impl_->check();return impl_->bytes;}
}
