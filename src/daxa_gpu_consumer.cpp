#include <stereoforge/daxa_gpu_consumer.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <array>
namespace stereoforge {
namespace {
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
bool same_owner(const std::shared_ptr<const DaxaRectification>& a,const std::shared_ptr<const DaxaRectification>& b){return a.get()==b.get()&&!a.owner_before(b)&&!b.owner_before(a);}
}
struct DaxaGpuConsumer::Slot {
    std::shared_ptr<DaxaFoundation> context;
    std::thread::id thread=std::this_thread::get_id();
    daxa::BufferId depth{},rgba{};
    uint64_t generation=0,producer_ticket=0,copy_ticket=0,consumer_ticket=0;
    DaxaObservationFrame producer; CaptureLease source;
    ~Slot()noexcept{
        if(thread!=std::this_thread::get_id())std::terminate();
        try {
            auto& d=context->device;
            auto ticket=consumer_ticket>copy_ticket?consumer_ticket:copy_ticket;
            if(ticket&&!detail::completed_submission(ticket,d.oldest_pending_submit_index()))
                d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=ticket,.timeout=30'000'000'000ull});
            if(!depth.is_empty()&&d.is_id_valid(depth))d.destroy_buffer(depth);
            if(!rgba.is_empty()&&d.is_id_valid(rgba))d.destroy_buffer(rgba);
            d.collect_garbage();
        }catch(...){std::terminate();}
    }
};
struct DaxaGpuConsumer::Impl {
    std::shared_ptr<DaxaFoundation> context; std::shared_ptr<const DaxaRectification> maps;
    unsigned count; size_t plane_bytes; std::thread::id thread=std::this_thread::get_id();
    std::array<std::shared_ptr<Slot>,3> slots{};
    void require_thread()const{require(thread==std::this_thread::get_id(),"GPU consumer requires owner submission thread");}
    ~Impl()noexcept{if(thread!=std::this_thread::get_id())std::terminate();}
    Impl(std::shared_ptr<DaxaFoundation> c,std::shared_ptr<const DaxaRectification> m,unsigned n):context(std::move(c)),maps(std::move(m)),count(n){
        require(context&&context->device.get()&&maps,"GPU consumer requires context/calibration");
        require(n>=2&&n<=3,"GPU consumer requires 2..3 slots");
        require(maps->width&&maps->height&&maps->width<=8192&&maps->height<=8192,"invalid GPU consumer grid");
        plane_bytes=size_t(maps->width)*maps->height*4;
        for(unsigned i=0;i<n;++i){
            auto s=std::make_shared<Slot>();s->context=context;
            s->depth=context->device.create_buffer({.size=plane_bytes,.name="GPU consumer depth"});
            s->rgba=context->device.create_buffer({.size=plane_bytes,.name="GPU consumer rgba"});
            slots[i]=std::move(s);
        }
    }
    void validate(const DaxaObservationFrame& input)const{
        require(bool(input),"empty GPU observation");
        require(input.context_id()==reinterpret_cast<uintptr_t>(context->device.get()),"foreign GPU observation context");
        require(same_owner(maps,input.calibration()),"foreign GPU observation calibration");
        require(input.width()==maps->width&&input.height()==maps->height,"GPU observation grid mismatch");
        require(input.size_bytes(DaxaObservationPlane::Depth)>=plane_bytes,"short GPU depth source");
        auto image=input.image(DaxaObservationImage::RectifiedLeft);
        require(image.width()==maps->width&&image.height()==maps->height&&image.size_bytes()>=plane_bytes,"GPU RGB grid mismatch");
    }
    std::shared_ptr<Slot>& available(){
        auto boundary=context->device.oldest_pending_submit_index();
        for(unsigned i=0;i<count;++i){auto& s=slots[i];auto last=s->consumer_ticket>s->copy_ticket?s->consumer_ticket:s->copy_ticket;
            if(detail::available_slot(s.use_count(),last,boundary))return s;}
        throw std::runtime_error("GPU consumer ring exhausted: retained destination or pending GPU use");
    }
    void submit(Slot& s,const DaxaObservationFrame& input){
        require(s.generation!=std::numeric_limits<uint64_t>::max(),"GPU consumer generation overflow");
        auto& d=context->device;auto before=d.latest_queue_submit_index(daxa::QUEUE_MAIN);
        s.producer=input;s.source=input.source();s.producer_ticket=input.submit_ticket();++s.generation;s.copy_ticket=0;s.consumer_ticket=0;
        try {
            auto recorder=d.create_command_recorder({.name="GPU consumer copy"});
            recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE|daxa::AccessConsts::TRANSFER_WRITE|daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::TRANSFER_READ,.dst_access=daxa::AccessConsts::TRANSFER_READ|daxa::AccessConsts::TRANSFER_WRITE});
            recorder.copy_buffer_to_buffer({.src_buffer=input.buffer(DaxaObservationPlane::Depth),.dst_buffer=s.depth,.src_offset=0,.dst_offset=0,.size=plane_bytes});
            recorder.copy_buffer_to_buffer({.src_buffer=input.image(DaxaObservationImage::RectifiedLeft).buffer(),.dst_buffer=s.rgba,.src_offset=0,.dst_offset=0,.size=plane_bytes});
            recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::COMPUTE_SHADER_READ|daxa::AccessConsts::TRANSFER_READ});
            auto commands=recorder.complete_current_commands();
            s.copy_ticket=d.submit_commands({.command_lists={&commands,1}});
            require(s.copy_ticket>before,"GPU copy submission did not advance queue");
        }catch(...){
            uint64_t latest=0;
            try{latest=d.latest_queue_submit_index(daxa::QUEUE_MAIN);}catch(...){std::terminate();}
            if(latest>before)s.copy_ticket=latest;
            else{s.copy_ticket=0;s.producer={};s.source={};}
            throw;
        }
    }
};
DaxaGpuConsumer::DaxaGpuConsumer(std::shared_ptr<DaxaFoundation> c,const DaxaObservationFrame& prototype,unsigned slots){
    require(bool(prototype),"empty GPU consumer prototype");
    require(c&&prototype.context_id()==reinterpret_cast<uintptr_t>(c->device.get()),"foreign GPU consumer prototype context");
    auto maps=prototype.calibration();require(maps&&prototype.width()==maps->width&&prototype.height()==maps->height,"prototype grid mismatch");
    impl_=std::make_shared<Impl>(std::move(c),std::move(maps),slots);impl_->validate(prototype);
}
DaxaGpuConsumer::~DaxaGpuConsumer()noexcept{if(impl_&&impl_->thread!=std::this_thread::get_id())std::terminate();}
bool DaxaGpuConsumer::has_capacity()const{
    require(bool(impl_),"moved-from GPU consumer");impl_->require_thread();
    auto boundary=impl_->context->device.oldest_pending_submit_index();
    for(unsigned i=0;i<impl_->count;++i){auto& s=impl_->slots[i];
        auto last=s->consumer_ticket>s->copy_ticket?s->consumer_ticket:s->copy_ticket;
        if(detail::available_slot(s.use_count(),last,boundary))return true;
    }
    return false;
}
DaxaGpuConsumer::Frame DaxaGpuConsumer::enqueue(const DaxaObservationFrame& input){
    require(bool(impl_),"moved-from GPU consumer");auto& e=*impl_;e.require_thread();e.validate(input);auto& slot=e.available();e.context->device.collect_garbage();e.submit(*slot,input);return Frame(impl_,slot);
}
size_t DaxaGpuConsumer::collect(){
    require(bool(impl_),"moved-from GPU consumer");auto& e=*impl_;e.require_thread();auto boundary=e.context->device.oldest_pending_submit_index();size_t count=0;
    for(unsigned i=0;i<e.count;++i){auto& s=*e.slots[i];if(s.producer&&detail::completed_submission(s.copy_ticket,boundary)){s.producer={};++count;}}return count;
}
void DaxaGpuConsumer::drain(){
    require(bool(impl_),"moved-from GPU consumer");auto& e=*impl_;e.require_thread();
    for(unsigned i=0;i<e.count;++i){auto& s=*e.slots[i];auto ticket=s.consumer_ticket>s.copy_ticket?s.consumer_ticket:s.copy_ticket;if(ticket)e.context->device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=ticket,.timeout=30'000'000'000ull});}collect();
}
size_t DaxaGpuConsumer::requested_bytes()const{require(bool(impl_),"moved-from GPU consumer");return impl_->plane_bytes*2*impl_->count;}
DaxaGpuConsumer::Frame::Frame(std::shared_ptr<Impl> owner,std::shared_ptr<Slot> slot):owner_(std::move(owner)),slot_(std::move(slot)),generation_(slot_->generation){}
DaxaGpuConsumer::Frame::~Frame()noexcept{if(owner_&&owner_->thread!=std::this_thread::get_id())std::terminate();}
DaxaGpuConsumer::Frame& DaxaGpuConsumer::Frame::operator=(Frame other)noexcept{owner_.swap(other.owner_);slot_.swap(other.slot_);std::swap(generation_,other.generation_);return *this;}
void DaxaGpuConsumer::Frame::validate()const{require(owner_&&slot_&&generation_==slot_->generation,"empty/stale GPU consumer frame");owner_->require_thread();}
DaxaGpuConsumer::Frame::operator bool()const{return bool(owner_&&slot_&&generation_==slot_->generation);}
daxa::BufferId DaxaGpuConsumer::Frame::depth()const{validate();return slot_->depth;}
daxa::BufferId DaxaGpuConsumer::Frame::rgba()const{validate();return slot_->rgba;}
uint32_t DaxaGpuConsumer::Frame::width()const{validate();return owner_->maps->width;}
uint32_t DaxaGpuConsumer::Frame::height()const{validate();return owner_->maps->height;}
uint64_t DaxaGpuConsumer::Frame::producer_ticket()const{validate();return slot_->producer_ticket;}
uint64_t DaxaGpuConsumer::Frame::copy_ticket()const{validate();return slot_->copy_ticket;}
bool DaxaGpuConsumer::Frame::ready()const{validate();return detail::completed_submission(slot_->copy_ticket,owner_->context->device.oldest_pending_submit_index());}
void DaxaGpuConsumer::Frame::retain_until(uint64_t ticket){
    validate();auto latest=owner_->context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);
    require(ticket>slot_->copy_ticket&&ticket<=latest,"invalid consumer submission ticket");
    if(ticket>slot_->consumer_ticket)slot_->consumer_ticket=ticket;
}
CaptureLease DaxaGpuConsumer::Frame::source()const{validate();return slot_->source;}
std::shared_ptr<const DaxaRectification> DaxaGpuConsumer::Frame::calibration()const{validate();return owner_->maps;}
}
