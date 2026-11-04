#include <stereoforge/detail/daxa_temporal_state.hpp>
#include <stereoforge/daxa_temporal_wip_adapters.hpp>
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
    // P1/P2/P3: Full-resolution DaxaRefinement::Frame integration with inverse-depth fusion, disocclusion guard, temporal Census re-search, and 50% memory left_eye_only mode.
    {auto rmaps=[](unsigned scale){DaxaRectification c;c.eye_width=24;c.source_height=20;c.stride_bytes=96;c.width=6*scale;c.height=5*scale;c.original_identity="raw";c.rectified_identity=scale==1?"coarse":"target";DaxaRectifiedGeometry g;g.original_identity=c.original_identity;g.rectified_identity=c.rectified_identity;for(auto& eye:g.original.eyes)eye={24,20,8,8,7,5,{}};for(auto& eye:g.rectified.eyes)eye={c.width,c.height,4.*scale,4.*scale,2.5*scale-0.5,1.5*scale-0.5,{}};g.original.right_to_left=g.rectified.right_to_left={1,0,0,.1,0,1,0,0,0,0,1,0,0,0,0,1};g.R1=g.R2={1,0,0,0,1,0,0,0,1};double f=4.*scale,cx=2.5*scale-0.5,cy=1.5*scale-0.5;g.P1={f,0,cx,0,0,f,cy,0,0,0,1,0};g.P2={f,0,cx,-f*.1,0,f,cy,0,0,0,1,0};c.geometry=g;for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<c.height;++y)for(unsigned x=0;x<c.width;++x){c.maps[eye*2].push_back(float((x-cx)/f*8+7));c.maps[eye*2].back()=std::clamp(c.maps[eye*2].back(),0.f,22.f);c.maps[eye*2+1].push_back(float((y-cy)/f*8+5));c.maps[eye*2+1].back()=std::clamp(c.maps[eye*2+1].back(),0.f,18.f);}return c;};
        auto rsource=[](uint64_t id){auto c=std::make_shared<CaptureFrame>();c->source_width=48;c->height=20;c->stride_bytes=96;c->frame_id=id;c->calibration_identity="raw";c->device_identity="temporal-refinement";c->host_clock_domain="steady";c->host_ingress_timestamp_ns=id;c->packed_yuyv.resize(1920);for(unsigned i=0;i<1920;++i)c->packed_yuyv[i]=uint8_t(i*17);return c;};
        auto low=rmaps(1),high=rmaps(2);
        DaxaDepthConfig dcfg{StereoConfig{6,5,3,4},1.f,2,32};
        DaxaDepthPipeline depth(low,dcfg,context,rsource(1));
        auto coarse=depth.enqueue(rsource(1));
        DaxaRefinementConfig rcfg;rcfg.disparities=8;rcfg.subpixel=true;rcfg.continuous_subpixel=true;rcfg.planar_recovery=true;
        DaxaRefinement ref(context,high,coarse,rcfg);
        auto ref_frame=ref.process(coarse);
        DaxaTemporalConfig tcfg_full;tcfg_full.inverse_depth_fusion=true;tcfg_full.disocclusion_guard=true;tcfg_full.temporal_stereo_research=true;
        DaxaTemporalConfig tcfg_left=tcfg_full;tcfg_left.left_eye_only=true;
        DaxaTemporal temporal_full(context,ref_frame,tcfg_full);
        DaxaTemporal temporal_left(context,ref_frame,tcfg_left);
        check(temporal_left.requested_buffer_bytes()<temporal_full.requested_buffer_bytes());
        auto out1=temporal_left.process(ref_frame,pose_for(1),true);out1.wait();
        auto rb1=temporal_left.readback(out1);
        check(rb1.depth.size()==12*10&&rb1.provenance.size()==12*10);
        out1={};ref_frame={};coarse={};temporal_left.collect();ref.collect();depth.collect();
        auto coarse2=depth.enqueue(rsource(2));auto ref_frame2=ref.process(coarse2);
        WipDummyRigPoseAdapter wip_pose{{0.001,0.0,0.0}};
        WipDummyMultiPairStereoRingAdapter wip_ring(temporal_left,4);
        WipDummyGpuSuperRes2xAdapter wip_sr2x;
        auto out2=wip_ring.process(ref_frame2,wip_pose.pose_for(*rsource(2)),true);out2.wait();
        auto sr2=wip_sr2x.wrap(out2);
        check(sr2.is_wip_dummy()&&sr2.superres_width==24&&sr2.superres_height==20&&wip_ring.buffered_pair_count()==1);
        auto rb2=temporal_left.readback(out2);
        check(rb2.counters.arithmetic_faults==0);
        sr2={};out2={};ref_frame2={};coarse2={};temporal_left.collect();ref.collect();depth.collect();
        check(temporal_left.completed_source_retirements()==2);
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
    {WipDummyRigPoseAdapter wip_pose{{0.002,0.0,0.0}};CaptureFrame cf;cf.frame_id=10;cf.host_ingress_timestamp_ns=100;cf.device_identity="zed2";cf.host_clock_domain="steady";auto p10=wip_pose.pose_for(cf);check(wip_pose.is_wip_dummy()&&p10.world_from_rig[3]==0.0);cf.frame_id=12;cf.host_ingress_timestamp_ns=200;auto p12=wip_pose.pose_for(cf);check(std::abs(p12.world_from_rig[3]-0.004)<1e-12);check(WipDummyMultiPairStereoRingAdapter::is_wip_dummy()&&WipDummyGpuSuperRes2xAdapter::is_wip_dummy());}
    if(argc>1&&std::string_view(argv[1])=="--gpu")gpu_contracts();
    std::cout<<"Temporal CPU admission contracts passed\n";return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
