#include <stereoforge/daxa_dual_preprocess.hpp>
#include <stereoforge/daxa_refinement.hpp>
#include <stereoforge/daxa_view.hpp>
#include <iostream>
#include <cstring>
using namespace stereoforge;
void check(bool v){if(!v)throw std::runtime_error("dual downstream contract failed");}
template<class F>void reject(F f){try{f();}catch(const std::exception&){return;}throw std::runtime_error("invalid input accepted");}
static DaxaRectification maps(bool high){DaxaRectification c;c.eye_width=16;c.source_height=12;c.stride_bytes=64;c.width=high?12:6;c.height=high?10:5;c.original_identity="raw";c.rectified_identity=high?"target":"coarse";DaxaRectifiedGeometry g;g.original_identity=c.original_identity;g.rectified_identity=c.rectified_identity;for(auto& eye:g.original.eyes)eye={16,12,8,8,7,5,{}};for(auto& eye:g.rectified.eyes)eye={c.width,c.height,high?8.:4.,high?8.:4.,high?4.5:2.,high?2.5:1.,{}};g.original.right_to_left=g.rectified.right_to_left={1,0,0,.1,0,1,0,0,0,0,1,0,0,0,0,1};g.R1=g.R2={1,0,0,0,1,0,0,0,1};double f=high?8:4,cx=high?4.5:2,cy=high?2.5:1;g.P1={f,0,cx,0,0,f,cy,0,0,0,1,0};g.P2={f,0,cx,-f*.1,0,f,cy,0,0,0,1,0};c.geometry=g;for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<c.height;++y)for(unsigned x=0;x<c.width;++x){c.maps[eye*2].push_back(float((x-cx)/f*8+7));c.maps[eye*2+1].push_back(float((y-cy)/f*8+5));}return c;}
static CaptureLease source(unsigned id){auto c=std::make_shared<CaptureFrame>();c->source_width=32;c->height=12;c->stride_bytes=64;c->frame_id=id;c->calibration_identity="raw";c->device_identity="refinement-contract";c->host_clock_domain="steady";c->host_ingress_timestamp_ns=id;c->packed_yuyv.resize(768);for(unsigned i=0;i<768;++i)c->packed_yuyv[i]=uint8_t(i*17);return c;}
template<class T> static void exact(const std::vector<T>& a,const std::vector<T>& b){check(a.size()==b.size()&&(!a.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0));}
static void exact_readback(const DaxaRefinementReadback& x,const DaxaRefinementReadback& y){
#define EQ(name) exact(x.name,y.name)
    EQ(disparity);EQ(confidence);EQ(validity);EQ(depth);EQ(metric_confidence);EQ(metric_validity);EQ(provenance);EQ(source_ids);EQ(age_ns);
    EQ(prediction);EQ(prediction_validity);EQ(lower);EQ(upper);EQ(reason);EQ(method);EQ(winner);EQ(best_cost);EQ(second_cost);EQ(supported_candidates);EQ(evaluated_candidates);EQ(subpixel_applied);EQ(raw_disparity);EQ(raw_confidence);EQ(raw_validity);EQ(census);EQ(census_validity);EQ(luma);EQ(support);for(unsigned eye=0;eye<2;++eye)exact(x.rectified_rgb[eye],y.rectified_rgb[eye]);
#undef EQ
}

int main(int argc,char**argv){try{
 auto low=maps(false),high=maps(true);validate_daxa_refinement_geometry(low,high);auto invalid=high;invalid.maps[0][1]+=.25f;reject([&]{validate_daxa_refinement_geometry(low,invalid);});
 if(argc>1&&std::string_view(argv[1])=="--gpu"){
 auto context=std::make_shared<DaxaFoundation>(true);
 {auto one=source(1);auto change=std::make_shared<CaptureFrame>(*source(2));for(size_t i=0;i<change->packed_yuyv.size();++i)change->packed_yuyv[i]=i%2?128:235;CaptureLease two=change;
 DaxaDepthConfig cfg{{6,5,3,4},1,2,32};DaxaDepthPipeline baseline(low,cfg,context,one);DaxaPreprocess baseline_full(high,2,context);DaxaDualPreprocess dual(high,low,2,context);
 auto oldcoarse=baseline.enqueue(one);auto oldfull=baseline_full.enqueue(one);auto input=dual.enqueue(one);DaxaDepthPipeline external(cfg,context,input.coarse);auto coarse=external.enqueue_preprocessed(input.coarse);
 DaxaRefinementConfig options;options.borrow_full_frame=true;DaxaRefinement oldref(context,oldcoarse,oldfull,options),newref(context,coarse,input.full,options);auto oldout=oldref.process(oldcoarse,oldfull),out=newref.process(coarse,input.full);
 auto oldcoarse2=baseline.enqueue(two);auto oldfull2=baseline_full.enqueue(two);auto input2=dual.enqueue(two);auto coarse2=external.enqueue_preprocessed(input2.coarse);auto oldout2=oldref.process(oldcoarse2,oldfull2),out2=newref.process(coarse2,input2.full);
 reject([&]{dual.enqueue(one);});exact_readback(oldref.readback(oldout),newref.readback(out));exact_readback(oldref.readback(oldout2),newref.readback(out2));check(out.observations().source()==one&&out2.observations().source()==two);check(newref.readback(out).luma!=newref.readback(out2).luma);
 auto retained=out.observations();GeometryConfig vc{12,10,16,12};VirtualView vv;vv.intrinsics={8,8,7,5};vv.source_to_virtual_right[3]=.1f;vv.mode=ViewMode::Preview;DaxaView view(vc,vv,context,retained,2);auto projected=view.project(retained);input={};coarse={};out={};retained={};newref.collect();external.collect();reject([&]{dual.enqueue(one);});projected.wait();projected={};view.collect();newref.collect();external.collect();dual.collect();auto recycled=dual.enqueue(one);check(recycled.full.source()==one);recycled.full.wait();
 }
 context->close();check(!context->validation_errors&&!context->validation_warnings);
 }
 std::cout<<"dual refinement contracts passed\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
