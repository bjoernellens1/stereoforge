#include <stereoforge/daxa_refinement.hpp>
#include <stereoforge/daxa_view.hpp>
#include <iostream>
#include <cstring>
#include <limits>
#include <cmath>
#include <type_traits>
using namespace stereoforge;
static_assert(std::is_nothrow_assignable_v<DaxaRefinement::Frame&,DaxaRefinement::Frame>);
static_assert(std::is_constructible_v<DaxaRefinement,std::shared_ptr<DaxaFoundation>,const DaxaDepthPipeline::Frame&,const DaxaFrame&,DaxaRefinementConfig>);
using CanonicalRefinementProcess=DaxaRefinement::Frame(DaxaRefinement::*)(const DaxaDepthPipeline::Frame&,const DaxaFrame&);
static_assert(std::is_same_v<decltype(static_cast<CanonicalRefinementProcess>(&DaxaRefinement::process)),CanonicalRefinementProcess>);

static_assert(!std::is_constructible_v<DaxaRefinement::Frame,std::shared_ptr<DaxaFoundation>,daxa::BufferId>);
void check(bool v){if(!v)throw std::runtime_error("refinement contract failed");}
template<class F>void reject(F f){try{f();}catch(std::invalid_argument const&){return;}throw std::runtime_error("invalid refinement accepted");}
static DaxaRectification maps(unsigned scale){DaxaRectification c;c.eye_width=24;c.source_height=20;c.stride_bytes=96;c.width=6*scale;c.height=5*scale;c.original_identity="raw";c.rectified_identity=scale==1?"coarse":"target";DaxaRectifiedGeometry g;g.original_identity=c.original_identity;g.rectified_identity=c.rectified_identity;for(auto& eye:g.original.eyes)eye={24,20,8,8,7,5,{}};for(auto& eye:g.rectified.eyes)eye={c.width,c.height,4.*scale,4.*scale,2.5*scale-0.5,1.5*scale-0.5,{}};g.original.right_to_left=g.rectified.right_to_left={1,0,0,.1,0,1,0,0,0,0,1,0,0,0,0,1};g.R1=g.R2={1,0,0,0,1,0,0,0,1};double f=4.*scale,cx=2.5*scale-0.5,cy=1.5*scale-0.5;g.P1={f,0,cx,0,0,f,cy,0,0,0,1,0};g.P2={f,0,cx,-f*.1,0,f,cy,0,0,0,1,0};c.geometry=g;for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<c.height;++y)for(unsigned x=0;x<c.width;++x){c.maps[eye*2].push_back(float((x-cx)/f*8+7));c.maps[eye*2].back()=std::clamp(c.maps[eye*2].back(),0.f,22.f);c.maps[eye*2+1].push_back(float((y-cy)/f*8+5));c.maps[eye*2+1].back()=std::clamp(c.maps[eye*2+1].back(),0.f,18.f);}return c;}
static CaptureLease source(unsigned id){auto c=std::make_shared<CaptureFrame>();c->source_width=48;c->height=20;c->stride_bytes=96;c->frame_id=id;c->calibration_identity="raw";c->device_identity="refinement-contract";c->host_clock_domain="steady";c->host_ingress_timestamp_ns=id;c->packed_yuyv.resize(1920);for(unsigned i=0;i<1920;++i)c->packed_yuyv[i]=uint8_t(i*17);return c;}
static DaxaRectification maps_custom(unsigned cw,unsigned ch,unsigned scale){DaxaRectification c;c.eye_width=1280;c.source_height=720;c.stride_bytes=5120;c.width=cw*scale;c.height=ch*scale;c.original_identity="raw720";c.rectified_identity=scale==1?"coarse320":"target1280";DaxaRectifiedGeometry g;g.original_identity=c.original_identity;g.rectified_identity=c.rectified_identity;for(auto& eye:g.original.eyes)eye={1280,720,1040.,1040.,639.5,359.5,{}};for(auto& eye:g.rectified.eyes)eye={c.width,c.height,520.*scale,520.*scale,cw*0.5*scale-0.5,ch*0.5*scale-0.5,{}};g.original.right_to_left=g.rectified.right_to_left={1,0,0,.12,0,1,0,0,0,0,1,0,0,0,0,1};g.R1=g.R2={1,0,0,0,1,0,0,0,1};double f=520.*scale,cx=cw*0.5*scale-0.5,cy=ch*0.5*scale-0.5;g.P1={f,0,cx,0,0,f,cy,0,0,0,1,0};g.P2={f,0,cx,-f*.12,0,f,cy,0,0,0,1,0};c.geometry=g;for(unsigned eye=0;eye<2;++eye){c.maps[eye*2].resize(c.width*c.height);c.maps[eye*2+1].resize(c.width*c.height);for(unsigned y=0;y<c.height;++y)for(unsigned x=0;x<c.width;++x){c.maps[eye*2][y*c.width+x]=float((x-cx)/f*1040.+639.5);c.maps[eye*2+1][y*c.width+x]=float((y-cy)/f*1040.+359.5);}}return c;}
static void gpu_4x_ortho_hybrid_and_repair(){
    auto context=std::make_shared<DaxaFoundation>(true);
    {
        auto low=maps(1),high4x=maps(4);
        auto input=source(21);
        DaxaDepthConfig cfg{{6,5,3,4,8,32,true,true},1,2,32};
        cfg.stereo_schedule=DaxaDepthStereoSchedule::SharedCensusSequential;
        cfg.right_reference_mode=DaxaRightReferenceMode::LeftAggregatedSelection;
        DaxaDepthPipeline depth(low,cfg,context,input);
        DaxaPreprocess full_pre(high4x,2,context);
        auto coarse=depth.enqueue(input);
        auto full=full_pre.enqueue(input);
        DaxaRefinementConfig rcfg;rcfg.disparities=12;rcfg.borrow_full_frame=true;rcfg.ortho_hybrid=true;rcfg.fast_mode=true;rcfg.allow_derived_coarse=true;rcfg.selective_repair=true;
        DaxaRefinement stage(context,coarse,full,rcfg);
        auto out=stage.process(coarse,full);
        out.wait();
        auto rb=stage.readback(out);
        check(rb.disparity.size()==2*24*20&&rb.depth.size()==2*24*20);
        out={};full={};coarse={};stage.collect();full_pre.collect_completed_sources();depth.collect();

        // Full 1280x720 (320x180 coarse 4x) production-scale contract & plateau rejection test
        auto low720=maps_custom(320,180,1),high720=maps_custom(320,180,4);
        auto cap720=std::make_shared<CaptureFrame>();
        cap720->source_width=2560;cap720->height=720;cap720->stride_bytes=5120;cap720->frame_id=42;
        cap720->calibration_identity="raw720";cap720->device_identity="refinement-720p";cap720->host_clock_domain="steady";cap720->host_ingress_timestamp_ns=42;
        cap720->packed_yuyv.resize(2560*720*2,128);
        for(unsigned y=0;y<720;++y){
            for(unsigned x=0;x<1280;++x){
                // Top region (orig y < 400 -> rect y < 440): rich texture with orig shift 8 px (rect disparity 16 px); bottom region (orig y >= 420 -> rect y >= 480): flat plateau
                uint8_t luma=(y>=420)?120:uint8_t(40+((x*13+y*7)^((x/3)*29+(y/5)*19))%180);
                cap720->packed_yuyv[(y*2560+x)*2]=luma;
                unsigned rx=(x>=8)?(x-8):0;
                cap720->packed_yuyv[(y*2560+1280+rx)*2]=luma;
            }
        }
        CaptureLease in720=cap720;
        DaxaDepthConfig cfg720{{320,180,32,4,8,32,true,true},1,2,32};
        cfg720.stereo_schedule=DaxaDepthStereoSchedule::LayeredPerPath;
        DaxaDepthPipeline depth720(low720,cfg720,context,in720);
        DaxaPreprocess full720(high720,2,context);
        auto c720=depth720.enqueue(in720);
        auto f720=full720.enqueue(in720);
        DaxaRefinementConfig base_cfg;base_cfg.disparities=128;base_cfg.borrow_full_frame=true;base_cfg.ortho_hybrid=true;base_cfg.fast_mode=false;base_cfg.selective_repair=false;
        DaxaRefinementConfig rep_cfg=base_cfg;rep_cfg.selective_repair=true;
        DaxaRefinement ref_base(context,c720,f720,base_cfg),ref_rep(context,c720,f720,rep_cfg);
        auto out_base=ref_base.process(c720,f720),out_rep=ref_rep.process(c720,f720);
        out_base.wait();out_rep.wait();
        check(out_base.timings().total_gpu_ms>0.0&&out_rep.timings().total_gpu_ms>0.0);
        auto rb_base=ref_base.readback(out_base),rb_rep=ref_rep.readback(out_rep);
        check(rb_rep.disparity.size()==2*1280*720);
        unsigned valid_textured=0,valid_flat_plateau=0;
        for(unsigned y=40;y<360;++y)for(unsigned x=40;x<1240;++x)if(rb_rep.validity[y*1280+x])++valid_textured;
        for(unsigned y=540;y<680;++y)for(unsigned x=100;x<1180;++x)if(rb_rep.validity[y*1280+x])++valid_flat_plateau;
        check(valid_textured>200000&&valid_flat_plateau==0);
        out_base={};out_rep={};f720={};c720={};ref_base.collect();ref_rep.collect();full720.collect_completed_sources();depth720.collect();
    }
    context->close();
    check(!context->validation_errors&&!context->validation_warnings);
}
static void gpu(){auto context=std::make_shared<DaxaFoundation>(true);auto low=maps(1),high=maps(2);DaxaDepthConfig cfg{{6,5,3,4},1,2,32};auto depth=std::make_unique<DaxaDepthPipeline>(low,cfg,context,source(1));auto coarse=depth->enqueue(source(1));auto stage=std::make_unique<DaxaRefinement>(context,high,coarse);auto first=stage->process(coarse);first.wait();auto common=first.observations();check(common.map_owner()!=coarse.calibration()&&common.width()==12&&common.image(DaxaObservationImage::RectifiedLeft).buffer()!=common.image(DaxaObservationImage::RectifiedRight).buffer());check(common.ancestry().kind()==DaxaObservationAncestryKind::SingleMeasured);auto first_copy=first;coarse={};auto second_coarse=depth->enqueue(source(2));auto second=stage->process(second_coarse);second.wait();check(!stage->can_process());auto old_generation=first.generation();first={};first_copy={};check(!stage->can_process());common={};stage->collect();check(stage->can_process());second_coarse={};second={};stage->collect();depth->collect();coarse=depth->enqueue(source(3));auto third=stage->process(coarse);auto held=third.observations();check(third.generation()>old_generation);coarse={};third=third;third=std::move(third);GeometryConfig vc{12,10,24,20};VirtualView vv;vv.intrinsics={8,8,7,5};vv.source_to_virtual_right[3]=.1f;vv.mode=ViewMode::Preview;auto view=std::make_unique<DaxaView>(vc,vv,context,held,2);auto projected=view->project(held);projected.wait();auto gathered=projected.source_observations();check(gathered.map_owner()==held.map_owner());stage.reset();depth.reset();view.reset();third.wait();third={};held.wait();check(held.source()->frame_id==3&&held.map_owner()->rectified_identity=="target");held={};projected.wait();projected={};gathered.wait();check(gathered.source()->frame_id==3);gathered={};context->close();check(!context->validation_errors&&!context->validation_warnings);}
template<class T> static void exact(const std::vector<T>& a,const std::vector<T>& b){check(a.size()==b.size()&&(!a.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0));}
static void exact_readback(const DaxaRefinementReadback& x,const DaxaRefinementReadback& y){
#define EQ(name) exact(x.name,y.name)
    EQ(disparity);EQ(confidence);EQ(validity);EQ(depth);EQ(metric_confidence);EQ(metric_validity);EQ(provenance);EQ(source_ids);EQ(age_ns);
    EQ(prediction);EQ(prediction_validity);EQ(lower);EQ(upper);EQ(reason);EQ(method);EQ(winner);EQ(best_cost);EQ(second_cost);EQ(supported_candidates);EQ(evaluated_candidates);EQ(subpixel_applied);EQ(raw_disparity);EQ(raw_confidence);EQ(raw_validity);EQ(census);EQ(census_validity);EQ(luma);EQ(support);for(unsigned eye=0;eye<2;++eye)exact(x.rectified_rgb[eye],y.rectified_rgb[eye]);
#undef EQ
}
static void raw_full_gpu(){
    auto context=std::make_shared<DaxaFoundation>(true);
    {auto low=maps(1),high=maps(2);auto input=source(11);
    DaxaDepthConfig cfg{{6,5,3,4},1,2,32};DaxaDepthPipeline depth(low,cfg,context,input);DaxaPreprocess full_pre(high,2,context);
    auto coarse=depth.enqueue(input);auto full=full_pre.enqueue(input);
    DaxaRefinement reference(context,high,coarse),reuse(context,coarse,full);
    DaxaRefinementConfig borrow_config;borrow_config.borrow_full_frame=true;DaxaRefinement borrowed(context,coarse,full,borrow_config);
    reject([&]{DaxaRefinement invalid(context,high,coarse,borrow_config);});
    auto a=reference.process(coarse),b=reuse.process(coarse,full),c=borrowed.process(coarse,full);
    // Submit changing frames through both producers and consumers without host waits.
    auto next_capture=std::make_shared<CaptureFrame>(*source(12));
    for(size_t i=0;i<next_capture->packed_yuyv.size();++i)next_capture->packed_yuyv[i]=i%2?128:235;
    CaptureLease next_source=next_capture;auto next_coarse=depth.enqueue(next_source);auto next_full=full_pre.enqueue(next_source);
    auto next_a=reference.process(next_coarse),next_b=reuse.process(next_coarse,next_full),next_c=borrowed.process(next_coarse,next_full);
    check(!reuse.can_process());
    auto x=reference.readback(a),y=reuse.readback(b);
    check(!a.timings().canonical_pack&&std::string_view(a.timings().first_stage_name)=="rectify");
    check(!b.timings().canonical_pack&&std::string_view(b.timings().first_stage_name)=="raw_full_pack");
    exact_readback(x,y);exact_readback(x,borrowed.readback(c));
    check(std::string_view(c.timings().first_stage_name)=="borrowed_census"&&std::isnan(c.timings().stage_gpu_ms[0]));
    reject([&]{c.buffer(DaxaRefinementPlane::Luma);});reject([&]{c.buffer(DaxaRefinementPlane::Support);});
    check(c.observations().image(DaxaObservationImage::RectifiedLeft).buffer()==full.plane(DaxaPreprocessPlane::RectifiedLeft).buffer());
    auto nx=reference.readback(next_a),ny=reuse.readback(next_b);exact_readback(nx,ny);exact_readback(nx,borrowed.readback(next_c));
    check(x.luma!=nx.luma);check(a.observations().source()==input&&b.observations().source()==input);
    reject([&]{reuse.process(coarse,next_full);});reject([&]{borrowed.process(coarse,next_full);});
    reject([&]{DaxaRefinement invalid(context,coarse,next_full);});
    DaxaPreprocess different_owner(high,2,context);auto foreign=different_owner.enqueue(input);reject([&]{reuse.process(coarse,foreign);});reject([&]{borrowed.process(coarse,foreign);});
    auto wrong=high;wrong.geometry->original.eyes[0].cx+=.1;reject([&]{validate_daxa_refinement_geometry(low,wrong);});
    // A retained output keeps both producers alive even when caller leases go away.
    auto retained=b.observations();auto borrowed_retained=c.observations();GeometryConfig vc{12,10,24,20};VirtualView vv;vv.intrinsics={8,8,7,5};vv.source_to_virtual_right[3]=.1f;vv.mode=ViewMode::Preview;DaxaView borrow_view(vc,vv,context,borrowed_retained,2);auto projected=borrow_view.project(borrowed_retained);full={};coarse={};b={};c={};reuse.collect();borrowed.collect();check(!borrowed.can_process());check(borrowed_retained.source()==input);borrowed_retained.wait();borrowed_retained={};projected.wait();borrowed.collect();check(!borrowed.can_process());projected={};borrow_view.collect();borrowed.collect();check(borrowed.can_process());auto recycled=borrowed.process(next_coarse,next_full);recycled.wait();check(recycled.observations().source()==next_source);recycled={};check(retained.source()==input);retained.wait();
    retained={};a={};next_a={};next_b={};next_c={};next_coarse={};next_full={};reference.collect();reuse.collect();borrowed.collect();foreign={};different_owner.collect_completed_sources();full_pre.collect_completed_sources();depth.collect();
    }
    context->close();check(!context->validation_errors&&!context->validation_warnings);
}
int main(int argc,char** argv){try{DaxaRefinementConfig c;c.validate();for(unsigned n:{0u,1u,4u}){auto bad=c;bad.slots=n;reject([&]{bad.validate();});}for(unsigned n:{0u,1u,3u,16u}){auto bad=c;bad.radius=n;reject([&]{bad.validate();});}for(unsigned n:{0u,1u,129u}){auto bad=c;bad.disparities=n;reject([&]{bad.validate();});}auto bad=c;bad.min_margin=0;reject([&]{bad.validate();});bad=c;bad.prior_confidence_floor=std::numeric_limits<float>::quiet_NaN();reject([&]{bad.validate();});auto low=maps(1),high=maps(2),high4x=maps(4);validate_daxa_refinement_geometry(low,high);validate_daxa_refinement_geometry(low,high4x);auto wrong=high;wrong.original_identity="different";reject([&]{validate_daxa_refinement_geometry(low,wrong);});wrong=high;wrong.geometry->original.eyes[0].cx+=.1;reject([&]{validate_daxa_refinement_geometry(low,wrong);});wrong=high;wrong.maps[0][10]+=.1;reject([&]{validate_daxa_refinement_geometry(low,wrong);});DaxaDepthConfig dc{{6,5,3,4}};dc.right_reference_mode=DaxaRightReferenceMode::IndependentLocalCensus;reject([&]{dc.validate();});DaxaRefinement::Frame empty;reject([&]{empty.observations();});reject([&]{empty.wait();});reject([&]{empty.ready();});auto copy=empty;copy=copy;copy=std::move(copy);check(!copy);if(argc>1&&std::string_view(argv[1])=="--gpu"){gpu();raw_full_gpu();gpu_4x_ortho_hybrid_and_repair();}std::cout<<"Refinement CPU admission contracts passed\n";return 0;}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
