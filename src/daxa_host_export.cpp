#include <stereoforge/daxa_host_export.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <daxa/c/device.h>
#include <array>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>
namespace stereoforge {
namespace {
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
bool same_owner(const std::shared_ptr<const DaxaRectification>& a,const std::shared_ptr<const DaxaRectification>& b){return a.get()==b.get()&&!a.owner_before(b)&&!b.owner_before(a);}
}
struct DaxaHostExport::Slot {
    std::shared_ptr<DaxaFoundation> context;
    daxa::MemoryBlock memory;daxa::BufferId buffer{};std::byte* mapped=nullptr;
    uint64_t generation=0,producer_ticket=0,copy_ticket=0;
    CaptureLease source;std::variant<std::monostate,DaxaObservationFrame,DaxaFrame> producer;
    ~Slot()noexcept{
        if(buffer.is_empty())return;
        // Fail closed if the bounded wait fails: never release pending GPU memory.
        try{auto& d=context->device;if(!detail::completed_submission(copy_ticket,d.oldest_pending_submit_index()))d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=copy_ticket,.timeout=30'000'000'000ull});d.destroy_buffer(buffer);d.collect_garbage();}
        catch(...){std::terminate();}
    }
};
struct DaxaHostExport::Impl {
    std::shared_ptr<DaxaFoundation> context;std::shared_ptr<const DaxaRectification> maps;
    DaxaHostExportConfig config;DaxaHostExportMemory selected;size_t plane_bytes=0;
    std::thread::id thread=std::this_thread::get_id();std::array<std::shared_ptr<Slot>,3> slots{};
    void require_thread()const{require(thread==std::this_thread::get_id(),"host export submission requires owner thread");}
    Impl(std::shared_ptr<DaxaFoundation> c,std::shared_ptr<const DaxaRectification> m,DaxaHostExportConfig cfg):context(std::move(c)),maps(std::move(m)),config(cfg){
        require(context&&context->device.get()&&maps,"host export requires context/calibration");
        require(cfg.slots>=2&&cfg.slots<=3&&(cfg.depth||cfg.rgb),"host export requires 2..3 slots and requested output");
        require(maps->width&&maps->height&&maps->width<=8192&&maps->height<=8192,"invalid host export grid");
        plane_bytes=size_t(maps->width)*maps->height*4;
        selected.bytes_per_slot=plane_bytes*(unsigned(cfg.depth)+unsigned(cfg.rgb));
        auto& d=context->device;daxa::BufferInfo info{.size=selected.bytes_per_slot,.name="persistent selective host export"};
        auto requirements=d.buffer_memory_requirements(info);VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(daxa_dvc_get_vk_physical_device(d.get()),&properties);
        bool found=false;
        for(unsigned pass=0;pass<2&&!found;++pass)for(uint32_t i=0;i<properties.memoryTypeCount;++i){
            auto flags=properties.memoryTypes[i].propertyFlags;constexpr auto coherent=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if((requirements.memory_type_bits&(1u<<i))&&(flags&coherent)==coherent&&(pass||bool(flags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT))){selected.type_index=i;selected.property_flags=flags;selected.host_cached=bool(flags&VK_MEMORY_PROPERTY_HOST_CACHED_BIT);requirements.memory_type_bits=1u<<i;found=true;break;}
        }
        require(found,"no compatible coherent host export memory; noncoherent invalidate is unqualified");
        for(unsigned i=0;i<cfg.slots;++i){
            auto s=std::make_shared<Slot>();s->context=context;
            s->memory=d.create_memory({.requirements=requirements,.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
            s->buffer=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=s->memory});
            s->mapped=reinterpret_cast<std::byte*>(d.buffer_host_address(s->buffer).value());require(s->mapped,"host export destination is not mapped");slots[i]=std::move(s);
        }
    }
    void validate(uint64_t identity,const std::shared_ptr<const DaxaRectification>& calibration,uint32_t width,uint32_t height)const{
        require(identity==reinterpret_cast<uintptr_t>(context->device.get()),"foreign host export context");
        require(same_owner(maps,calibration),"foreign host export calibration owner");
        require(width==maps->width&&height==maps->height,"host export grid mismatch");
    }
    std::shared_ptr<Slot>& available(){
        auto boundary=context->device.oldest_pending_submit_index();
        for(unsigned i=0;i<config.slots;++i)if(detail::available_slot(slots[i].use_count(),slots[i]->copy_ticket,boundary))return slots[i];
        throw std::runtime_error("host export ring exhausted: retained destination or pending copy");
    }
    template<class Input>void submit(Slot& s,const Input& input,daxa::BufferId depth,daxa::BufferId rgb,uint64_t ticket){
        require(s.generation!=std::numeric_limits<uint64_t>::max(),"host export generation overflow");
        auto& d=context->device;auto before=d.latest_queue_submit_index(daxa::QUEUE_MAIN);
        s.producer=input;s.source=input.source();s.producer_ticket=ticket;++s.generation;s.copy_ticket=0;
        try{
            auto recorder=d.create_command_recorder({.name="selective native host export"});
            recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE|daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ});
            if(config.depth)recorder.copy_buffer_to_buffer({.src_buffer=depth,.dst_buffer=s.buffer,.src_offset=0,.dst_offset=0,.size=plane_bytes});
            if(config.rgb)recorder.copy_buffer_to_buffer({.src_buffer=rgb,.dst_buffer=s.buffer,.src_offset=0,.dst_offset=config.depth?plane_bytes:0,.size=plane_bytes});
            recorder.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});
            auto command=recorder.complete_current_commands();
            s.copy_ticket=d.submit_commands({.command_lists={&command,1}});
            require(s.copy_ticket>before,"host export copy submission did not advance queue");
        }catch(...){
            uint64_t latest=0;
            // If queue inspection itself fails, pending ownership cannot be
            // proven. Do not unwind into a reusable slot or resource teardown.
            try{latest=d.latest_queue_submit_index(daxa::QUEUE_MAIN);}catch(...){std::terminate();}
            if(latest>before)s.copy_ticket=latest;
            else{s.copy_ticket=0;s.producer=std::monostate{};s.source={};}
            throw;
        }
    }
};
DaxaHostExport::DaxaHostExport(std::shared_ptr<DaxaFoundation> c,const DaxaObservationFrame& prototype,DaxaHostExportConfig cfg){
    require(bool(prototype),"empty host export prototype");
    require(c&&prototype.context_id()==reinterpret_cast<uintptr_t>(c->device.get()),"foreign host export prototype context");
    auto maps=prototype.calibration();require(maps&&prototype.width()==maps->width&&prototype.height()==maps->height,"prototype grid mismatch");
    auto rgb=prototype.image(DaxaObservationImage::RectifiedLeft);if(cfg.rgb)require(rgb.width()==maps->width&&rgb.height()==maps->height,"prototype RGB grid mismatch");
    impl_=std::make_shared<Impl>(std::move(c),std::move(maps),cfg);
}
DaxaHostExport::DaxaHostExport(std::shared_ptr<DaxaFoundation> c,const DaxaFrame& prototype,DaxaHostExportConfig cfg){
    require(!cfg.depth&&cfg.rgb,"DaxaFrame host export requires RGB only");require(bool(prototype),"empty host export prototype");
    require(c&&prototype.context_id()==reinterpret_cast<uintptr_t>(c->device.get()),"foreign host export prototype context");
    auto maps=prototype.calibration();auto rgb=prototype.plane(DaxaPreprocessPlane::RectifiedLeft);require(maps&&rgb.width()==maps->width&&rgb.height()==maps->height,"prototype RGB grid mismatch");
    impl_=std::make_shared<Impl>(std::move(c),std::move(maps),cfg);
}
DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaObservationFrame& input){
    require(bool(impl_),"moved-from host exporter");auto& e=*impl_;e.require_thread();auto& slot=e.available();
    e.validate(input.context_id(),input.calibration(),input.width(),input.height());
    daxa::BufferId depth{},rgb{};if(e.config.depth){require(input.size_bytes(DaxaObservationPlane::Depth)>=e.plane_bytes,"short depth source");depth=input.buffer(DaxaObservationPlane::Depth);}
    if(e.config.rgb){auto view=input.image(DaxaObservationImage::RectifiedLeft);require(view.width()==e.maps->width&&view.height()==e.maps->height,"RGB grid mismatch");rgb=view.buffer();}
    e.submit(*slot,input,depth,rgb,input.submit_ticket());return Frame(impl_,slot);
}
DaxaHostExport::Frame DaxaHostExport::enqueue(const DaxaFrame& input){
    require(bool(impl_),"moved-from host exporter");auto& e=*impl_;e.require_thread();require(!e.config.depth,"DaxaFrame cannot export depth");auto& slot=e.available();
    auto view=input.plane(DaxaPreprocessPlane::RectifiedLeft);e.validate(input.context_id(),input.calibration(),view.width(),view.height());
    e.submit(*slot,input,{},view.buffer(),input.submit_ticket());return Frame(impl_,slot);
}
size_t DaxaHostExport::collect(){require(bool(impl_),"moved-from host exporter");auto& e=*impl_;e.require_thread();auto boundary=e.context->device.oldest_pending_submit_index();size_t count=0;
    for(unsigned i=0;i<e.config.slots;++i){auto& s=*e.slots[i];if(s.producer.index()&&detail::completed_submission(s.copy_ticket,boundary)){s.producer=std::monostate{};++count;}}return count;
}
void DaxaHostExport::drain(){require(bool(impl_),"moved-from host exporter");auto& e=*impl_;e.require_thread();for(unsigned i=0;i<e.config.slots;++i){auto ticket=e.slots[i]->copy_ticket;if(ticket)e.context->device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=ticket,.timeout=30'000'000'000ull});}collect();}
DaxaHostExportMemory DaxaHostExport::memory()const{require(bool(impl_),"moved-from host exporter");return impl_->selected;}
size_t DaxaHostExport::requested_bytes()const{require(bool(impl_),"moved-from host exporter");return impl_->selected.bytes_per_slot*impl_->config.slots;}
DaxaHostExport::Frame::Frame(std::shared_ptr<Impl> owner,std::shared_ptr<Slot> slot):owner_(std::move(owner)),slot_(std::move(slot)),generation_(slot_->generation){}
DaxaHostExport::Frame& DaxaHostExport::Frame::operator=(Frame other)noexcept{owner_.swap(other.owner_);slot_.swap(other.slot_);std::swap(generation_,other.generation_);return *this;}
void DaxaHostExport::Frame::validate()const{require(owner_&&slot_&&generation_==slot_->generation,"empty/stale host export frame");}
DaxaHostExport::Frame::operator bool()const{return bool(owner_&&slot_&&generation_==slot_->generation);}
bool DaxaHostExport::Frame::ready()const{validate();return detail::completed_submission(slot_->copy_ticket,owner_->context->device.oldest_pending_submit_index());}
void DaxaHostExport::Frame::wait()const{validate();owner_->context->device.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=slot_->copy_ticket,.timeout=30'000'000'000ull});}
std::span<const std::byte> DaxaHostExport::Frame::depth_bytes()const{require(ready(),"host export copy incomplete");return owner_->config.depth?std::span<const std::byte>(slot_->mapped,owner_->plane_bytes):std::span<const std::byte>{};}
std::span<const std::byte> DaxaHostExport::Frame::rgba_bytes()const{require(ready(),"host export copy incomplete");return owner_->config.rgb?std::span<const std::byte>(slot_->mapped+(owner_->config.depth?owner_->plane_bytes:0),owner_->plane_bytes):std::span<const std::byte>{};}
uint32_t DaxaHostExport::Frame::width()const{validate();return owner_->maps->width;}uint32_t DaxaHostExport::Frame::height()const{validate();return owner_->maps->height;}
uint64_t DaxaHostExport::Frame::producer_ticket()const{validate();return slot_->producer_ticket;}uint64_t DaxaHostExport::Frame::copy_ticket()const{validate();return slot_->copy_ticket;}
CaptureLease DaxaHostExport::Frame::source()const{validate();return slot_->source;}std::shared_ptr<const DaxaRectification> DaxaHostExport::Frame::calibration()const{validate();return owner_->maps;}
}
