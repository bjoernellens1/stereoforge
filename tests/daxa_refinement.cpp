#include <stereoforge/daxa_refinement.hpp>
#include <stereoforge/daxa_view.hpp>
#include <iostream>
#include <cstring>
#include <limits>
#include <type_traits>
using namespace stereoforge;
static_assert(std::is_nothrow_assignable_v<DaxaRefinement::Frame&,DaxaRefinement::Frame>);
static_assert(std::is_constructible_v<DaxaRefinement,std::shared_ptr<DaxaFoundation>,const DaxaDepthPipeline::Frame&,const DaxaFrame&,DaxaRefinementConfig>);
using CanonicalRefinementProcess=DaxaRefinement::Frame(DaxaRefinement::*)(const DaxaDepthPipeline::Frame&,const DaxaFrame&);
static_assert(std::is_same_v<decltype(static_cast<CanonicalRefinementProcess>(&DaxaRefinement::process)),CanonicalRefinementProcess>);

static_assert(!std::is_constructible_v<DaxaRefinement::Frame,std::shared_ptr<DaxaFoundation>,daxa::BufferId>);
void check(bool v){if(!v)throw std::runtime_error("refinement contract failed");}
template<class F>void reject(F f){try{f();}catch(std::invalid_argument const&){return;}throw std::runtime_error("invalid refinement accepted");}
static DaxaRectification maps(bool high){DaxaRectification c;c.eye_width=16;c.source_height=12;c.stride_bytes=64;c.width=high?12:6;c.height=high?10:5;c.original_identity="raw";c.rectified_identity=high?"target":"coarse";DaxaRectifiedGeometry g;g.original_identity=c.original_identity;g.rectified_identity=c.rectified_identity;for(auto& eye:g.original.eyes)eye={16,12,8,8,7,5,{}};for(auto& eye:g.rectified.eyes)eye={c.width,c.height,high?8.:4.,high?8.:4.,high?4.5:2.,high?2.5:1.,{}};g.original.right_to_left=g.rectified.right_to_left={1,0,0,.1,0,1,0,0,0,0,1,0,0,0,0,1};g.R1=g.R2={1,0,0,0,1,0,0,0,1};double f=high?8:4,cx=high?4.5:2,cy=high?2.5:1;g.P1={f,0,cx,0,0,f,cy,0,0,0,1,0};g.P2={f,0,cx,-f*.1,0,f,cy,0,0,0,1,0};c.geometry=g;for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<c.height;++y)for(unsigned x=0;x<c.width;++x){c.maps[eye*2].push_back(float((x-cx)/f*8+7));c.maps[eye*2+1].push_back(float((y-cy)/f*8+5));}return c;}
static CaptureLease source(unsigned id){auto c=std::make_shared<CaptureFrame>();c->source_width=32;c->height=12;c->stride_bytes=64;c->frame_id=id;c->calibration_identity="raw";c->device_identity="refinement-contract";c->host_clock_domain="steady";c->host_ingress_timestamp_ns=id;c->packed_yuyv.resize(768);for(unsigned i=0;i<768;++i)c->packed_yuyv[i]=uint8_t(i*17);return c;}
static void gpu(){auto context=std::make_shared<DaxaFoundation>(true);auto low=maps(false),high=maps(true);DaxaDepthConfig cfg{{6,5,3,4},1,2,32};auto depth=std::make_unique<DaxaDepthPipeline>(low,cfg,context,source(1));auto coarse=depth->enqueue(source(1));auto stage=std::make_unique<DaxaRefinement>(context,high,coarse);auto first=stage->process(coarse);first.wait();auto common=first.observations();check(common.map_owner()!=coarse.calibration()&&common.width()==12&&common.image(DaxaObservationImage::RectifiedLeft).buffer()!=common.image(DaxaObservationImage::RectifiedRight).buffer());check(common.ancestry().kind()==DaxaObservationAncestryKind::SingleMeasured);auto first_copy=first;coarse={};auto second_coarse=depth->enqueue(source(2));auto second=stage->process(second_coarse);second.wait();check(!stage->can_process());auto old_generation=first.generation();first={};first_copy={};check(!stage->can_process());common={};stage->collect();check(stage->can_process());second_coarse={};second={};stage->collect();depth->collect();coarse=depth->enqueue(source(3));auto third=stage->process(coarse);auto held=third.observations();check(third.generation()>old_generation);coarse={};third=third;third=std::move(third);GeometryConfig vc{12,10,16,12};VirtualView vv;vv.intrinsics={8,8,7,5};vv.source_to_virtual_right[3]=.1f;vv.mode=ViewMode::Preview;auto view=std::make_unique<DaxaView>(vc,vv,context,held,2);auto projected=view->project(held);projected.wait();auto gathered=projected.source_observations();check(gathered.map_owner()==held.map_owner());stage.reset();depth.reset();view.reset();third.wait();third={};held.wait();check(held.source()->frame_id==3&&held.map_owner()->rectified_identity=="target");held={};projected.wait();projected={};gathered.wait();check(gathered.source()->frame_id==3);gathered={};context->close();check(!context->validation_errors&&!context->validation_warnings);}
template<class T> static void exact(const std::vector<T>& a,const std::vector<T>& b){check(a.size()==b.size()&&(!a.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0));}
static void exact_readback(const DaxaRefinementReadback& x,const DaxaRefinementReadback& y){
#define EQ(name) exact(x.name,y.name)
    EQ(disparity);EQ(confidence);EQ(validity);EQ(depth);EQ(metric_confidence);EQ(metric_validity);EQ(provenance);EQ(source_ids);EQ(age_ns);
    EQ(prediction);EQ(prediction_validity);EQ(lower);EQ(upper);EQ(reason);EQ(method);EQ(winner);EQ(best_cost);EQ(second_cost);EQ(supported_candidates);EQ(evaluated_candidates);EQ(subpixel_applied);EQ(raw_disparity);EQ(raw_confidence);EQ(raw_validity);EQ(census);EQ(census_validity);EQ(luma);EQ(support);for(unsigned eye=0;eye<2;++eye)exact(x.rectified_rgb[eye],y.rectified_rgb[eye]);
#undef EQ
}
static void raw_full_gpu(){
    auto context=std::make_shared<DaxaFoundation>(true);
    {auto low=maps(false),high=maps(true);auto input=source(11);
    DaxaDepthConfig cfg{{6,5,3,4},1,2,32};DaxaDepthPipeline depth(low,cfg,context,input);DaxaPreprocess full_pre(high,2,context);
    auto coarse=depth.enqueue(input);auto full=full_pre.enqueue(input);
    DaxaRefinement reference(context,high,coarse),reuse(context,coarse,full);
    auto a=reference.process(coarse),b=reuse.process(coarse,full);
    // Submit changing frames through both producers and consumers without host waits.
    auto next_capture=std::make_shared<CaptureFrame>(*source(12));
    for(size_t i=0;i<next_capture->packed_yuyv.size();++i)next_capture->packed_yuyv[i]=i%2?128:235;
    CaptureLease next_source=next_capture;auto next_coarse=depth.enqueue(next_source);auto next_full=full_pre.enqueue(next_source);
    auto next_a=reference.process(next_coarse),next_b=reuse.process(next_coarse,next_full);
    check(!reuse.can_process());
    auto x=reference.readback(a),y=reuse.readback(b);
    check(!a.timings().canonical_pack&&std::string_view(a.timings().first_stage_name)=="rectify");
    check(!b.timings().canonical_pack&&std::string_view(b.timings().first_stage_name)=="raw_full_pack");
    exact_readback(x,y);
    auto nx=reference.readback(next_a),ny=reuse.readback(next_b);exact_readback(nx,ny);
    check(x.luma!=nx.luma);check(a.observations().source()==input&&b.observations().source()==input);
    reject([&]{reuse.process(coarse,next_full);});
    reject([&]{DaxaRefinement invalid(context,coarse,next_full);});
    DaxaPreprocess different_owner(high,2,context);auto foreign=different_owner.enqueue(input);reject([&]{reuse.process(coarse,foreign);});
    auto wrong=high;wrong.geometry->original.eyes[0].cx+=.1;reject([&]{validate_daxa_refinement_geometry(low,wrong);});
    // A retained output keeps both producers alive even when caller leases go away.
    auto retained=b.observations();full={};coarse={};b={};reuse.collect();check(retained.source()==input);retained.wait();
    retained={};a={};next_a={};next_b={};next_coarse={};next_full={};reference.collect();reuse.collect();foreign={};different_owner.collect_completed_sources();full_pre.collect_completed_sources();depth.collect();
    }
    context->close();check(!context->validation_errors&&!context->validation_warnings);
}
int main(int argc,char** argv){try{DaxaRefinementConfig c;c.validate();for(unsigned n:{0u,1u,4u}){auto bad=c;bad.slots=n;reject([&]{bad.validate();});}for(unsigned n:{0u,1u,3u,16u}){auto bad=c;bad.radius=n;reject([&]{bad.validate();});}for(unsigned n:{0u,1u,129u}){auto bad=c;bad.disparities=n;reject([&]{bad.validate();});}auto bad=c;bad.min_margin=0;reject([&]{bad.validate();});bad=c;bad.prior_confidence_floor=std::numeric_limits<float>::quiet_NaN();reject([&]{bad.validate();});auto low=maps(false),high=maps(true);validate_daxa_refinement_geometry(low,high);auto wrong=high;wrong.original_identity="different";reject([&]{validate_daxa_refinement_geometry(low,wrong);});wrong=high;wrong.geometry->original.eyes[0].cx+=.1;reject([&]{validate_daxa_refinement_geometry(low,wrong);});wrong=high;wrong.maps[0][10]+=.1;reject([&]{validate_daxa_refinement_geometry(low,wrong);});DaxaDepthConfig dc{{6,5,3,4}};dc.right_reference_mode=DaxaRightReferenceMode::IndependentLocalCensus;reject([&]{dc.validate();});DaxaRefinement::Frame empty;reject([&]{empty.observations();});reject([&]{empty.wait();});reject([&]{empty.ready();});auto copy=empty;copy=copy;copy=std::move(copy);check(!copy);if(argc>1&&std::string_view(argv[1])=="--gpu"){gpu();raw_full_gpu();}std::cout<<"Refinement CPU admission contracts passed\n";return 0;}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
