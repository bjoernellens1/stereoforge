#include <stereoforge/detail/daxa_temporal_state.hpp>
#include <iostream>
#include <stereoforge/daxa_view.hpp>
#include <limits>
#include <type_traits>
using namespace stereoforge;
static_assert(std::is_nothrow_assignable_v<DaxaTemporal::Frame&,DaxaTemporal::Frame>);
static_assert(!std::is_constructible_v<DaxaObservationAncestryView,size_t,unsigned,std::array<daxa::BufferId,5>>);
template<class F>void reject(F f){try{f();}catch(std::invalid_argument const&){return;}throw std::runtime_error("invalid temporal admission accepted");}
void check(bool b){if(!b)throw std::runtime_error("temporal contract failed");}
static DaxaRectification fixture(){
    DaxaRectification c;c.eye_width=8;c.source_height=6;c.stride_bytes=32;c.width=6;c.height=5;c.original_identity="raw";c.rectified_identity="rect";
    DaxaRectifiedGeometry g;g.original_identity="raw";g.rectified_identity="rect";
    for(unsigned eye=0;eye<2;++eye){g.original.eyes[eye]={8,6,8,8,3,2,{}};g.rectified.eyes[eye]={6,5,8,8,2,1,{}};}
    g.original.right_to_left=g.rectified.right_to_left={1,0,0,.1,0,1,0,0,0,0,1,0,0,0,0,1};g.R1=g.R2={1,0,0,0,1,0,0,0,1};g.P1={8,0,2,0,0,8,1,0,0,0,1,0};g.P2={8,0,2,-.8,0,8,1,0,0,0,1,0};c.geometry=g;
    for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<5;++y)for(unsigned x=0;x<6;++x){c.maps[eye*2].push_back(float(x+1));c.maps[eye*2+1].push_back(float(y+1));}return c;
}
static CaptureLease source(uint64_t id){auto c=std::make_shared<CaptureFrame>();c->source_width=16;c->height=6;c->stride_bytes=32;c->frame_id=id;c->host_ingress_timestamp_ns=id;c->host_clock_domain="steady";c->device_identity="temporal-lifetime";c->calibration_identity="raw";c->packed_yuyv.resize(192);for(size_t i=0;i<192;++i)c->packed_yuyv[i]=uint8_t(i*17);return c;}
static DaxaTimedRigPose pose_for(uint64_t id){return {{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1},int64_t(id),"steady"};}
static void gpu_contracts(){
    auto context=std::make_shared<DaxaFoundation>(true);auto maps=fixture();
    // Depth has exactly one slot: history-only storage must release original RGB.
    {DaxaDepthConfig cfg{StereoConfig{6,5,3,4},1.f,1,32};DaxaDepthPipeline depth(maps,cfg,context,source(1));auto raw=depth.enqueue(source(1));auto temporal=std::make_unique<DaxaTemporal>(context,raw);raw.wait();raw={};depth.collect();
        for(uint64_t id=1;id<=8;++id){check(temporal->can_process());raw=depth.enqueue(source(id));auto output=temporal->process(raw,pose_for(id));output.wait();auto common=output.observations();check(common.source()->frame_id==id&&common.ancestry().kind()==DaxaObservationAncestryKind::BoundedContributors&&common.ancestry().capacity()==4);check(common.ancestry().size_bytes(DaxaObservationAncestryPlane::ContributorIds)==60*32);auto raw_common=raw.observations();check(raw_common.ancestry().kind()==DaxaObservationAncestryKind::SingleMeasured);reject([&]{raw_common.ancestry().buffer(DaxaObservationAncestryPlane::Count);});raw_common={};raw={};output={};check(temporal->collect()==0);common={};check(temporal->collect()==1);check(depth.collect()==1);}
        check(temporal->completed_source_retirements()==8);
    }
    // Retained older slots intentionally block a two-slot ring; reset preserves them.
    {DaxaDepthConfig cfg{StereoConfig{6,5,3,4},1.f,2,32};auto depth=std::make_unique<DaxaDepthPipeline>(maps,cfg,context,source(20));auto raw=depth->enqueue(source(20));auto temporal=std::make_unique<DaxaTemporal>(context,raw);auto first=temporal->process(raw,pose_for(20));first.wait();auto before=context->device.latest_queue_submit_index(daxa::QUEUE_MAIN);reject([&]{temporal->process(raw,pose_for(20));});check(context->device.latest_queue_submit_index(daxa::QUEUE_MAIN)==before);auto first_copy=first;auto common=first.observations();raw={};
        auto next=depth->enqueue(source(21));auto bad_pose=pose_for(21);bad_pose.world_from_rig[3]=1e300;reject([&]{temporal->process(next,bad_pose);});auto second=temporal->process(next,pose_for(21));second.wait();next={};second={};temporal->collect();check(!temporal->can_process());auto generation=first.generation();first={};first_copy={};check(!temporal->can_process());common={};temporal->collect();depth->collect();check(temporal->can_process());
        auto pending=depth->enqueue(source(22));auto third=temporal->process(pending,pose_for(22));auto third_copy=third;auto lease=third.observations();pending={};temporal->reset();check(third.generation()>generation&&lease.source()->frame_id==22);third=third;third=std::move(third);third_copy={};
        GeometryConfig vc{6,5,8,6};VirtualView vv;vv.intrinsics={8,8,3,2};vv.source_to_virtual_right[3]=.1f;vv.mode=ViewMode::Preview;auto view=std::make_unique<DaxaView>(vc,vv,context,lease,2);auto projected=view->project(lease);projected.wait();auto gathered=projected.source_observations();check(gathered.buffer(DaxaObservationPlane::Depth)==lease.buffer(DaxaObservationPlane::Depth)&&gathered.ancestry().buffer(DaxaObservationAncestryPlane::ContributorIds)==lease.ancestry().buffer(DaxaObservationAncestryPlane::ContributorIds));
        // Child leases retain all producers after each public facade is destroyed.
        temporal.reset();depth.reset();view.reset();third.wait();check(third.observations().source()->frame_id==22);third={};lease={};projected.wait();projected={};check(gathered.source()->frame_id==22);gathered.wait();gathered={};
    }
    check(context->validation_errors.load()==0);context->close();check(context->validation_errors.load()==0);
}
int main(int argc,char** argv){try{
    check(detail::temporal_tickets_complete(4,6,6));check(!detail::temporal_tickets_complete(4,7,6));check(!detail::temporal_tickets_complete(7,4,6));
    check(detail::temporal_internal_references(true)==2&&detail::temporal_internal_references(false)==1);
    DaxaCameraModel camera{4,4,1,1,0,0,{}};check(detail::temporal_pinhole_f32(camera)[0]==1);camera.fy=1e-50;reject([&]{detail::temporal_pinhole_f32(camera);});camera.fy=1;camera.fx=1e100;reject([&]{detail::temporal_pinhole_f32(camera);});
    DaxaTemporalConfig c;c.validate();for(unsigned n:{0u,1u,4u}){auto x=c;x.slots=n;reject([&]{x.validate();});}for(unsigned n:{0u,5u}){auto x=c;x.max_contributors=n;reject([&]{x.validate();});}
    {auto x=c;x.max_reuses=0;reject([&]{x.validate();});x=c;x.max_age_ns=-1;reject([&]{x.validate();});x=c;x.confidence_floor=std::numeric_limits<float>::quiet_NaN();reject([&]{x.validate();});x=c;x.relative_depth_tolerance=-1;reject([&]{x.validate();});}
    CaptureFrame capture;capture.frame_id=1;capture.host_ingress_timestamp_ns=1;capture.device_identity="camera";capture.host_clock_domain="steady";
    DaxaTimedRigPose pose{{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1},1,"steady"};pose.validate(capture);
    for(unsigned i:{0u,15u}){auto bad=pose;bad.world_from_rig[i]=2;reject([&]{bad.validate(capture);});}auto bad=pose;bad.timestamp_ns=-1;reject([&]{bad.validate(capture);});bad=pose;bad.clock_domain="unknown";reject([&]{bad.validate(capture);});
    detail::TemporalSequence state;check(state.device.capacity()>=65536&&state.clock.capacity()>=65536);check(!state.prepare(capture,pose,c.max_age_ns).warp);check(!state.accepted);state.commit(capture,pose);
    reject([&]{state.prepare(capture,pose,c.max_age_ns);});check(state.id==1&&state.has_pose);
    capture.frame_id=2;capture.host_ingress_timestamp_ns=2;pose.timestamp_ns=2;check(state.prepare(capture,pose,c.max_age_ns).warp);check(state.time==1);
    auto a=state.prepare(capture,std::nullopt,c.max_age_ns);check(!a.warp);state.commit(capture,std::nullopt);capture.frame_id=3;capture.host_ingress_timestamp_ns=3;pose.timestamp_ns=3;check(!state.prepare(capture,pose,c.max_age_ns).warp);state.commit(capture,pose);
    capture.frame_id=4;capture.host_ingress_timestamp_ns=4;pose.timestamp_ns=4;check(state.prepare(capture,pose,c.max_age_ns).warp);capture.device_identity="new";capture.frame_id=0;capture.host_ingress_timestamp_ns=0;pose.timestamp_ns=0;check(state.prepare(capture,pose,c.max_age_ns).epoch);
    capture.device_identity="unknown";reject([&]{state.prepare(capture,pose,c.max_age_ns);});capture.device_identity="new";capture.host_ingress_timestamp_ns=UINT64_MAX;reject([&]{state.prepare(capture,std::nullopt,c.max_age_ns);});state.reset();check(!state.accepted&&!state.has_pose&&state.device.capacity()>=65536);
    DaxaTemporal::Frame empty;reject([&]{empty.observations();});reject([&]{empty.ancestry();});reject([&]{empty.wait();});reject([&]{empty.ready();});reject([&]{empty.timings();});auto copied=empty;copied=copied;copied=std::move(copied);check(!copied);
    if(argc>1&&std::string_view(argv[1])=="--gpu")gpu_contracts();
    std::cout<<"Temporal CPU admission contracts passed\n";return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
