#include "stereo_vo_fixture.hpp"
#include <stereoforge/daxa_stereo_history.hpp>
#include <iostream>
using namespace stereoforge;
using namespace sf_test;
static std::vector<std::byte> download(std::shared_ptr<DaxaFoundation> const& c,daxa::BufferId src,std::size_t bytes,uint64_t offset=0) {
    auto& d=c->device;daxa::BufferInfo info{.size=bytes,.name="history witness export"};
    auto memory=d.create_memory({.requirements=c->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
    auto dest=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory});
    auto rec=d.create_command_recorder({.name="history byte-exact witness"});
    rec.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE|daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ});
    rec.copy_buffer_to_buffer({.src_buffer=src,.dst_buffer=dest,.src_offset=offset,.size=bytes});
    rec.pipeline_barrier({.src_access=daxa::AccessConsts::TRANSFER_WRITE,.dst_access=daxa::AccessConsts::HOST_READ});
    auto commands=rec.complete_current_commands();auto ticket=d.submit_commands({.command_lists={&commands,1}});
    d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=ticket,.timeout=30'000'000'000ull});
    std::vector<std::byte> result(bytes);std::memcpy(result.data(),d.buffer_host_address(dest).value(),bytes);d.destroy_buffer(dest);d.collect_garbage();return result;
}
static void cached_evidence_and_four_pair_votes(std::shared_ptr<DaxaFoundation> const& context,unsigned fault) {
    auto source=render(1,0);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
    auto rgb=input.enqueue(source);DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
    DaxaDepthPipeline depth(dc,context,rgb.full());auto frame=depth.enqueue_preprocessed(rgb.full());
    DaxaStereoVoConfig vc;vc.columns=12;vc.rows=8;DaxaStereoVo vo(context,frame.observations(),vc);StereoTrajectory trajectory;
    DaxaStereoHistoryConfig hc;hc.allow_unqualified_research=true;
    auto history=std::make_shared<DaxaStereoHistory>(context,frame.observations(),hc);
    DaxaTemporalConfig tc;tc.odometry_research=true;tc.multi_pair_research=true;
    DaxaTemporal temporal(context,frame,tc,history);
    for(uint64_t id=1;id<=(fault==7?5u:2u);++id) {
        if(id>1){rgb=input.enqueue(render(id,0));frame=depth.enqueue_preprocessed(rgb.full());}
        auto pose=trajectory.process(vo.process(frame.observations()));
        if((fault<7&&id==2)||(fault==7&&id==3)) {
            auto cached=history->select(pose);check(cached.size()==(fault==7?2u:1u),"cache fault witness has wrong frame set");
            for(auto const& entry:cached) {
                auto plane=DaxaStereoHistoryPlane::Right;uint64_t value=0xff808080ff808080ULL;
                if(fault==0){plane=DaxaStereoHistoryPlane::AgeNs;value=1;}
                if(fault==1){plane=DaxaStereoHistoryPlane::SourceIds;value=999;}
                if(fault==2){plane=DaxaStereoHistoryPlane::Depth;value=0x7fc000007fc00000ULL;}
                if(fault==3){plane=DaxaStereoHistoryPlane::Depth;float a=1.59f,b=1.61f;uint32_t lo,hi;std::memcpy(&lo,&a,4);std::memcpy(&hi,&b,4);value=uint64_t(lo)|(uint64_t(hi)<<32);}
                if(fault==4){plane=DaxaStereoHistoryPlane::Depth;float z=1.2f;uint32_t bits;std::memcpy(&bits,&z,4);value=uint64_t(bits)|(uint64_t(bits)<<32);}
                if(fault==5)plane=DaxaStereoHistoryPlane::Left;
                entry.frame.wait();auto view=entry.frame.plane(plane);
                // Cache planes are packed; corruption is limited to this view.
                overwrite_buffer(context,view.buffer,view.size,value,view.offset);
            }
        }
        auto output=temporal.process_odometry(frame,pose);auto result=temporal.readback(output);
        if(fault<7&&id==2)check(result.counters.pair_checks>0&&result.counters.pair_rejections==result.counters.pair_checks,"cached-only corrupted evidence admitted fusion");
        if(fault==7&&id==4)check(result.counters.pair_checks>0&&result.counters.pair_rejections==result.counters.pair_checks,"one valid pair falsely satisfied two-pair vote");
        if(fault==7&&id==5)check(result.counters.pair_checks>0&&result.counters.pair_rejections<result.counters.pair_checks,"pair indices2/3 were not consumed for fusion");
        output={};frame={};rgb={};temporal.collect();depth.collect();input.collect();
    }
}
static void qualified_acquisition_bounds(std::shared_ptr<DaxaFoundation> const& context,unsigned kind) {
    auto source=render(1,0);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
    auto rgb=input.enqueue(source);DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
    DaxaDepthPipeline depth(dc,context,rgb.full());auto frame=depth.enqueue_preprocessed(rgb.full());
    DaxaStereoVoConfig vc;vc.columns=12;vc.rows=8;DaxaStereoVo vo(context,frame.observations(),vc);StereoTrajectory trajectory;DaxaTemporal temporal(context,frame);
    for(uint64_t id=1;id<=(kind==4?4u:(kind==5?3u:2u));++id) {
        uint64_t acquisition=kind==5?(id==1?100000000:(id==2?110000000:200000000)):(kind==0&&id==2?1000000000:(kind==4?33000000+(id-1)*100000000:id*33000000));
        if(id>1){rgb=input.enqueue(render(id,0,false,0,"steady",false,acquisition));frame=depth.enqueue_preprocessed(rgb.full());}
        auto pose=trajectory.process(vo.process(frame.observations()));
        // Test-only external clock/covariance qualification controls. Native VO
        // never emits these flags; the estimated pose itself is not replaced.
        pose.temporal_qualified=pose.covariance_calibrated=true;
        pose.source.clock_mapping_uncertainty_ns=kind==5?(id==1?0:(id==2?100000000:20000000)):(kind==1&&id==2?1000000000:(kind==3?90000000:0));
        auto output=temporal.process_odometry(frame,pose);auto result=temporal.readback(output);
        if(id==2&&kind!=2&&kind!=4&&kind!=5)check(result.counters.history_scanned==0,"unbounded acquisition age/uncertainty fused history");
        if((id==2&&kind==2)||(kind==4&&(id==2||id==3)))check(result.counters.history_scanned>0,"qualified in-bound acquisition control never fused");
        if(kind==5&&id==2)check(result.counters.history_scanned>0,"varying-uncertainty positive control never fused");
        if(kind==5&&id==3)check(result.counters.history_scanned==0,"later uncertain contribution exceeded acquisition age envelope");
        if(kind==4&&id==4)check(result.counters.history_scanned==0,"accumulated history exceeded acquisition age bound");
        output={};frame={};rgb={};temporal.collect();depth.collect();input.collect();
    }
}
int main(){try{
    auto context=std::make_shared<DaxaFoundation>(true);
    std::optional<DaxaStereoHistory::Frame> survivor;std::vector<std::byte> survivor_bytes;
    {
        auto source=render(1,0);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
        auto rgb=input.enqueue(source);DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
        DaxaDepthPipeline depth(dc,context,rgb.full());auto frame=depth.enqueue_preprocessed(rgb.full());
        DaxaStereoVoConfig vc;vc.columns=12;vc.rows=8;DaxaStereoVo vo(context,frame.observations(),vc);StereoTrajectory trajectory;
        auto pose=trajectory.process(vo.process(frame.observations()));
        DaxaStereoHistoryConfig hc;hc.allow_unqualified_research=true;
        DaxaStereoHistory history(context,frame.observations(),hc),default_history(context,frame.observations());
        check(!default_history.store(frame.observations(),pose),"unqualified pose entered production history");
        auto stored=history.store(frame.observations(),pose);check(stored.has_value(),"actual stereo pair was not copied");stored->wait();history.collect();
        auto observations=frame.observations();
        auto cached=[&](DaxaStereoHistoryPlane p){auto v=stored->plane(p);return download(context,v.buffer,v.size,v.offset);};
        check(cached(DaxaStereoHistoryPlane::Left)==download(context,observations.image(DaxaObservationImage::RectifiedLeft).buffer(),width*height*4),"cached left RGB differs");
        check(cached(DaxaStereoHistoryPlane::Right)==download(context,observations.image(DaxaObservationImage::RectifiedRight).buffer(),width*height*4),"cached right RGB differs");
        check(cached(DaxaStereoHistoryPlane::Depth)==download(context,observations.buffer(DaxaObservationPlane::Depth),width*height*8),"cached stereo depth differs");
        observations={};frame={};rgb={};stored={};depth.collect();input.collect();
        for(uint64_t id=2;id<=6;++id) {
            rgb=input.enqueue(render(id,int(id%2)*3));frame=depth.enqueue_preprocessed(rgb.full());
            pose=trajectory.process(vo.process(frame.observations()));
            auto selected=history.select(pose);check(selected.size()==std::min<std::size_t>(id-1,4),"four-pair history capacity/producer independence failed");
            for(auto const& entry:selected)check(entry.ingress_age_ns<=200000000&&!entry.acquisition_age_upper_bound_ns,"unqualified acquisition age invented");
            selected.clear();stored=history.store(frame.observations(),pose);check(stored.has_value(),"history stopped canonical two-slot producer");stored->wait();history.collect();
            frame={};rgb={};stored={};depth.collect();input.collect();
        }
        auto selected=history.select(pose);check(selected.size()==3,"current frame was returned as its own history");
        check(selected.front().frame.pose().source.frame_id==3,"history eviction order incorrect");
        auto expired=pose;expired.source.frame_id=20;expired.source.ingress_timestamp_ns+=200000001;expired.source.acquisition_timestamp_ns+=200000001;
        check(history.select(expired).empty(),"history older than200ms retained");
        auto new_world=pose;++new_world.trajectory_id;check(history.select(new_world).empty(),"history crossed world gauges");
        check(history.requested_bytes()==uint64_t(width)*height*64*4,"unbounded/unaccounted GPU history memory");
        selected.clear();auto future=pose;++future.source.frame_id;future.source.ingress_timestamp_ns+=33000000;future.source.acquisition_timestamp_ns+=33000000;
        auto held=history.select(future);check(held.size()==4,"four physical pair leases unavailable");
        rgb=input.enqueue(render(7,1));frame=depth.enqueue_preprocessed(rgb.full());pose=trajectory.process(vo.process(frame.observations()));
        check(!history.store(frame.observations(),pose)&&history.stats().capacity_skipped==1,"live cache leases were overwritten");
        auto stable=held[1].frame.plane(DaxaStereoHistoryPlane::Left);auto stable_bytes=download(context,stable.buffer,stable.size,stable.offset);
        held.erase(held.begin());stored=history.store(frame.observations(),pose);check(bool(stored),"cache pressure retry did not recover");stored->wait();history.collect();
        check(download(context,stable.buffer,stable.size,stable.offset)==stable_bytes,"producer reuse mutated retained historical pixels");
        bool bad_ticket=false;try{held.front().frame.mark_consumed(context->device.latest_queue_submit_index(daxa::QUEUE_MAIN)+1);}catch(std::invalid_argument const&){bad_ticket=true;}
        check(bad_ticket,"future/unsubmitted consumer fence accepted");
        held.front().frame.mark_consumed(context->device.latest_queue_submit_index(daxa::QUEUE_MAIN));
        auto wrong=pose;++wrong.source.frame_id;bool wrong_source=false;
        try{history.store(frame.observations(),wrong);}catch(std::invalid_argument const&){wrong_source=true;}
        check(wrong_source,"wrong source pose entered GPU history");
        survivor=*stored;auto view=stored->plane(DaxaStereoHistoryPlane::Right);survivor_bytes=download(context,view.buffer,view.size,view.offset);
        history.clear();check(survivor->pose().source.frame_id==7,"logical reset invalidated exported historical frame");
        frame={};rgb={};stored={};depth.collect();input.collect();
    }
    for(unsigned fault=0;fault<8;++fault)cached_evidence_and_four_pair_votes(context,fault);
    for(unsigned kind=0;kind<6;++kind)qualified_acquisition_bounds(context,kind);
    auto retained=survivor->plane(DaxaStereoHistoryPlane::Right);
    check(download(context,retained.buffer,retained.size,retained.offset)==survivor_bytes,"history destruction released an exported GPU lease");survivor.reset();
    check(context->validation_errors.load()==0,"history Vulkan validation error");context->close();
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
