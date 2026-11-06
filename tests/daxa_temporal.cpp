#include <stereoforge/detail/daxa_temporal_state.hpp>
#include <stereoforge/daxa_temporal_wip_adapters.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
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
        for(uint64_t id=1;id<=8;++id){check(temporal->can_process());raw=depth.enqueue(source(id));auto output=temporal->process(raw,pose_for(id));output.wait();auto common=output.observations();check(!common.has_disparity());reject([&]{common.buffer(DaxaObservationPlane::Disparity);});reject([&]{common.size_bytes(DaxaObservationPlane::Disparity);});check(common.source()->frame_id==id&&common.ancestry().kind()==DaxaObservationAncestryKind::BoundedContributors&&common.ancestry().capacity()==4);check(common.ancestry().size_bytes(DaxaObservationAncestryPlane::ContributorIds)==60*32);auto raw_common=raw.observations();check(raw_common.ancestry().kind()==DaxaObservationAncestryKind::SingleMeasured);reject([&]{raw_common.ancestry().buffer(DaxaObservationAncestryPlane::Count);});raw_common={};raw={};output={};check(temporal->collect()==0);common={};check(temporal->collect()==1);check(depth.collect()==1);}
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
        DaxaTemporalConfig tcfg_full;tcfg_full.inverse_depth_fusion=true;tcfg_full.disocclusion_guard=true;tcfg_full.temporal_stereo_research=true;tcfg_full.max_disparity=192;
        DaxaTemporalConfig tcfg_left=tcfg_full;tcfg_left.left_eye_only=true;
        DaxaTemporal temporal_full(context,ref_frame,tcfg_full);
        DaxaTemporal temporal_left(context,ref_frame,tcfg_left);
        check(temporal_left.requested_buffer_bytes()<temporal_full.requested_buffer_bytes());
        auto out1=temporal_left.process(ref_frame,pose_for(1),true);out1.wait();
        reject([&]{out1.observations();});
        auto left_obs=out1.left_observations();
        check(left_obs.layers()==1&&left_obs.width()==12&&left_obs.height()==10&&left_obs.size_bytes(DaxaObservationPlane::Depth)==12*10*4&&left_obs.size_bytes(DaxaObservationPlane::SourceIds)==12*10*8);
        left_obs={};
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
static void benchmark_v12_json(){
    auto context=std::make_shared<DaxaFoundation>(false);
    {
    auto make_maps=[](unsigned cw,unsigned ch,unsigned scale){
        DaxaRectification c;c.eye_width=1280;c.source_height=720;c.stride_bytes=5120;c.width=cw*scale;c.height=ch*scale;
        c.original_identity="raw";c.rectified_identity=scale==1?"coarse":"target";
        DaxaRectifiedGeometry g;g.original_identity=c.original_identity;g.rectified_identity=c.rectified_identity;
        double f=260.0*scale,cx=cw*0.5*scale-0.5,cy=ch*0.5*scale-0.5;
        for(auto& eye:g.original.eyes)eye={1280,720,1040.0,1040.0,639.5,359.5,{}};
        for(auto& eye:g.rectified.eyes)eye={c.width,c.height,f,f,cx,cy,{}};
        g.original.right_to_left=g.rectified.right_to_left={1,0,0,.12,0,1,0,0,0,0,1,0,0,0,0,1};
        g.R1=g.R2={1,0,0,0,1,0,0,0,1};
        g.P1={f,0,cx,0,0,f,cy,0,0,0,1,0};g.P2={f,0,cx,-f*.12,0,f,cy,0,0,0,1,0};c.geometry=g;
        for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<c.height;++y)for(unsigned x=0;x<c.width;++x){
            c.maps[eye*2].push_back(float((x-cx)/f*1040.0+639.5));
            c.maps[eye*2+1].push_back(float((y-cy)/f*1040.0+359.5));
        }
        return c;
    };
    auto low=make_maps(320,180,1),high=make_maps(320,180,4);
    auto make_src=[&](uint64_t id,int shift_fg){
        auto c=std::make_shared<CaptureFrame>();c->source_width=2560;c->height=720;c->stride_bytes=5120;
        c->frame_id=id;c->calibration_identity="raw";c->device_identity="zed2-gpu-bench";c->host_clock_domain="steady";c->host_ingress_timestamp_ns=id*16666667ull;
        c->packed_yuyv.resize(size_t(c->stride_bytes)*c->height);
        for(unsigned y=0;y<720;++y){
            uint8_t* row=c->packed_yuyv.data()+size_t(y)*c->stride_bytes;
            for(unsigned x=0;x<1280;++x){
                bool in_fg=(y>=180&&y<540&&int(x)>=360+shift_fg&&int(x)<680+shift_fg);
                unsigned disp=in_fg?32u:12u;
                unsigned src_x=in_fg?unsigned(int(x)-shift_fg):x;
                uint32_t coarse_pat=((src_x/8u)*47u+(y/8u)*59u)^((src_x/4u)*29u+(y/4u)*37u);
                uint32_t fine_pat=((src_x/2u)*19u+(y/2u)*23u)^(src_x*13u+y*7u);
                uint8_t val=uint8_t(30u+((coarse_pat+fine_pat)%190u));
                row[x*2]=val;row[x*2+1]=128u;
                int rx=int(x)-int(disp);
                if(rx>=0&&rx<1280){row[(1280+rx)*2]=val;row[(1280+rx)*2+1]=128u;}
            }
        }
        return c;
    };
    auto s0=make_src(1,0),s1=make_src(2,48);
    DaxaDepthConfig dcfg{{320,180,32,4,8,32,true,true},1.f,2,32};
    dcfg.stereo_schedule=DaxaDepthStereoSchedule::Hierarchical1to8Selective1to4;
    DaxaDepthPipeline depth(low,dcfg,context,s0);
    DaxaPreprocess full_pre(high,2,context);
    auto c0=depth.enqueue(s0);
    auto f0=full_pre.enqueue(s0);
    DaxaRefinementConfig rcfg;rcfg.disparities=128;rcfg.radius=2;rcfg.subpixel=true;rcfg.ortho_hybrid=true;rcfg.fast_mode=true;rcfg.ultra_fast=true;rcfg.allow_derived_coarse=true;rcfg.planar_recovery=true;rcfg.continuous_subpixel=true;
    DaxaRefinement ref_p0(context,high,c0,rcfg);
    DaxaRefinementConfig rcfg_v13=rcfg;rcfg_v13.adaptive_tiles=true;rcfg_v13.min_disparity_bound=4;rcfg_v13.max_disparity_bound=96;
    DaxaRefinement ref_v13(context,high,c0,rcfg_v13);
    DaxaRefinementConfig rcfg_meas_v11=rcfg;rcfg_meas_v11.planar_recovery=false;rcfg_meas_v11.borrow_full_frame=true;rcfg_meas_v11.continuous_subpixel=false;
    DaxaRefinement ref_meas_v11(context,c0,f0,rcfg_meas_v11);
    DaxaRefinementConfig rcfg_meas_v13=rcfg_meas_v11;rcfg_meas_v13.adaptive_tiles=true;rcfg_meas_v13.min_disparity_bound=4;rcfg_meas_v13.max_disparity_bound=96;
    DaxaRefinement ref_meas_v13(context,c0,f0,rcfg_meas_v13);
    auto rf0=ref_p0.process(c0);
    DaxaTemporalConfig tcfg_full;tcfg_full.inverse_depth_fusion=true;tcfg_full.disocclusion_guard=true;tcfg_full.temporal_stereo_research=true;tcfg_full.max_disparity=128;
    DaxaTemporalConfig tcfg_left=tcfg_full;tcfg_left.left_eye_only=true;
    DaxaTemporalConfig tcfg_unguarded=tcfg_left;tcfg_unguarded.disocclusion_guard=false;tcfg_unguarded.temporal_stereo_research=false;
    DaxaTemporal temp_full(context,rf0,tcfg_full);
    DaxaTemporal temp_left(context,rf0,tcfg_left);
    DaxaTemporal temp_unguarded(context,rf0,tcfg_unguarded);
    uint64_t bytes_full=temp_full.requested_buffer_bytes(),bytes_left=temp_left.requested_buffer_bytes();
    rf0.wait();f0.wait();rf0={};f0={};c0={};ref_p0.collect();full_pre.collect_completed_sources();depth.collect();

    // 1. Measure native GPU moving-object wake ghosting (frame 1 -> frame 2 with +48px foreground shift)
    // Foreground disp=32 -> z = 1040*0.12/32 = 3.90m; Background disp=12 -> z = 1040*0.12/12 = 10.40m
    auto pose_t=[](uint64_t id){return DaxaTimedRigPose{{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1},int64_t(id*16666667ull),"steady"};};
    auto ca=depth.enqueue(s0);auto rfa=ref_p0.process(ca);
    auto ug1=temp_unguarded.process(rfa,pose_t(1),true);ug1.wait();
    auto gd1=temp_left.process(rfa,pose_t(1),true);gd1.wait();
    auto rb_f1=temp_left.readback(gd1);
    ug1={};gd1={};rfa={};ca={};temp_unguarded.collect();temp_left.collect();ref_p0.collect();depth.collect();

    auto cb=depth.enqueue(s1);auto rfb=ref_p0.process(cb);
    auto ug2=temp_unguarded.process(rfb,pose_t(2),true);ug2.wait();
    auto gd2=temp_left.process(rfb,pose_t(2),true);gd2.wait();
    auto rb_ug2=temp_unguarded.readback(ug2);
    auto rb_gd2=temp_left.readback(gd2);
    auto rb_ref2=ref_p0.readback(rfb);
    ug2={};gd2={};rfb={};cb={};temp_unguarded.collect();temp_left.collect();ref_p0.collect();depth.collect();

    // Wake region: y in [190, 530), x in [364, 398) where foreground was at t=1 (disp=32, z=3.90m) and moved to [408, 728) at t=2 (true bg disp=12, z=10.40m)
    unsigned wake_total=0,wake_ghost_ug=0,wake_ghost_gd=0,wake_ghost_sf=0;
    for(unsigned y=190;y<530;++y)for(unsigned x=364;x<398;++x){
        size_t idx=size_t(y)*1280+x;
        if(rb_f1.validity[idx]!=0&&rb_f1.depth[idx]<6.0f){
            ++wake_total;
            if(rb_ug2.validity[idx]!=0&&rb_ug2.depth[idx]<6.0f)++wake_ghost_ug;
            if(rb_gd2.validity[idx]!=0&&rb_gd2.depth[idx]<6.0f)++wake_ghost_gd;
            if(rb_ref2.metric_validity[idx]!=0&&rb_ref2.depth[idx]<6.0f)++wake_ghost_sf;
        }
    }

    // 2. Measure native GPU p50/p95/p99 timings across 40 timed frames (after 8 warmup frames)
    temp_left.reset();
    constexpr unsigned warmup=8,frames=40;
    std::vector<double> t_clear,t_z,t_owner,t_fuse,t_temp_total,t_ref_p0,t_ref_v13,t_chain_p0,t_chain_v13,t_meas_v11,t_meas_v13;
    for(unsigned i=0;i<warmup+frames;++i){
        uint64_t fid=10+i;
        s0->frame_id=fid;s0->host_ingress_timestamp_ns=fid*16666667ull;
        auto cf=depth.enqueue(s0);
        auto rf_p0=ref_p0.process(cf);
        auto tf=temp_left.process(rf_p0,pose_t(fid),true);
        tf.wait();
        auto dt=cf.timings();
        auto tt=tf.timings();
        auto rt_p0=rf_p0.timings();
        tf={};rf_p0={};
        temp_left.collect();ref_p0.collect();
        auto rf_13=ref_v13.process(cf);
        rf_13.wait();
        auto rt_13=rf_13.timings();
        rf_13={};ref_v13.collect();
        auto ff=full_pre.enqueue(s0);
        auto rm_11=ref_meas_v11.process(cf,ff);
        rm_11.wait();
        auto rt_m11=rm_11.timings();
        rm_11={};ref_meas_v11.collect();
        auto rm_13=ref_meas_v13.process(cf,ff);
        rm_13.wait();
        auto rt_m13=rm_13.timings();
        rm_13={};ff={};cf={};
        ref_meas_v13.collect();full_pre.collect_completed_sources();depth.collect();
        if(i>=warmup){
            t_clear.push_back(tt.clear_gpu_ms);t_z.push_back(tt.z_gpu_ms);t_owner.push_back(tt.owner_gpu_ms);t_fuse.push_back(tt.fuse_gpu_ms);
            t_temp_total.push_back(tt.total_gpu_ms);
            t_ref_p0.push_back(rt_p0.total_gpu_ms);
            t_ref_v13.push_back(rt_13.total_gpu_ms);
            t_chain_p0.push_back(dt.combined_gpu_ms+rt_p0.total_gpu_ms+tt.total_gpu_ms);
            t_chain_v13.push_back(dt.combined_gpu_ms+rt_13.total_gpu_ms+tt.total_gpu_ms);
            t_meas_v11.push_back(dt.combined_gpu_ms+rt_m11.total_gpu_ms);
            t_meas_v13.push_back(dt.combined_gpu_ms+rt_m13.total_gpu_ms);
        }
    }
    auto stat=[](std::vector<double> v){
        std::sort(v.begin(),v.end());
        double mean=std::accumulate(v.begin(),v.end(),0.0)/double(v.size());
        double p50=v[v.size()/2],p95=v[(95*v.size())/100],p99=v[(99*v.size())/100];
        std::ostringstream os;os<<std::fixed<<std::setprecision(4)<<"{\"p50\":"<<p50<<",\"p95\":"<<p95<<",\"p99\":"<<p99<<",\"mean\":"<<mean<<"}";
        return os.str();
    };
    double ug_pct=wake_total?100.0*double(wake_ghost_ug)/double(wake_total):0.0;
    double gd_pct=wake_total?100.0*double(wake_ghost_gd)/double(wake_total):0.0;
    double sf_pct=wake_total?100.0*double(wake_ghost_sf)/double(wake_total):0.0;
    std::cout<<std::fixed<<std::setprecision(4)
        <<"{\"requested_full_bytes\":"<<bytes_full
        <<",\"requested_left_only_bytes\":"<<bytes_left
        <<",\"native_gpu_wake_test\":{\"wake_pixels\":"<<wake_total
        <<",\"single_frame_ghost_pct\":"<<sf_pct
        <<",\"unguarded_temporal_ghost_pct\":"<<ug_pct
        <<",\"guarded_temporal_ghost_pct\":"<<gd_pct<<"}"
        <<",\"temporal_clear_gpu_ms\":"<<stat(t_clear)
        <<",\"temporal_z_gpu_ms\":"<<stat(t_z)
        <<",\"temporal_owner_gpu_ms\":"<<stat(t_owner)
        <<",\"temporal_fuse_gpu_ms\":"<<stat(t_fuse)
        <<",\"temporal_total_gpu_ms\":"<<stat(t_temp_total)
        <<",\"refinement_p0_gpu_ms\":"<<stat(t_ref_p0)
        <<",\"refinement_v13_adaptive_tiles_gpu_ms\":"<<stat(t_ref_v13)
        <<",\"measured_depth_v11_p0_gpu_ms\":"<<stat(t_meas_v11)
        <<",\"measured_depth_v13_adaptive_gpu_ms\":"<<stat(t_meas_v13)
        <<",\"full_chain_p0_plus_temporal_gpu_ms\":"<<stat(t_chain_p0)
        <<",\"full_chain_v13_plus_temporal_gpu_ms\":"<<stat(t_chain_v13)
        <<"}\n";
    }
    context->close();
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
    if(argc>1&&std::string_view(argv[1])=="--benchmark-v12-json"){benchmark_v12_json();return 0;}
    std::cout<<"Temporal CPU admission contracts passed\n";return 0;
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
