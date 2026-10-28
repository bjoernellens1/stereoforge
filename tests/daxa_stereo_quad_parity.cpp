#include <stereoforge/daxa_preprocess.hpp>
#include <stereoforge/daxa_stereo.hpp>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace stereoforge;

static void check(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
template<class T> static bool bytes_equal(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
static void compare(const DaxaStereoReadback& a,const DaxaStereoReadback& b){
    check(a.reference==b.reference&&a.estimator==b.estimator&&a.right_reference_mode==b.right_reference_mode&&a.packing_bits==b.packing_bits&&a.intermediates_reconstructed&&b.intermediates_reconstructed,"stereo metadata mismatch");
    check(a.census==b.census&&a.census_valid==b.census_valid&&a.costs==b.costs&&a.sums==b.sums&&a.packed_penalties==b.packed_penalties,"Census/cost/sums/packed words mismatch");
    check(bytes_equal(a.disparity,b.disparity)&&bytes_equal(a.confidence,b.confidence)&&bytes_equal(a.validity,b.validity)&&bytes_equal(a.depth,b.depth)&&bytes_equal(a.depth_validity,b.depth_validity),"output plane byte mismatch");
}
static CaptureLease source(unsigned id){
    auto frame=std::make_shared<CaptureFrame>();frame->source_width=20;frame->height=7;frame->stride_bytes=40;frame->calibration_identity="overwrite-original";frame->frame_id=id;frame->camera_timestamp_ns=id;frame->host_ingress_timestamp_ns=0;frame->controls_json="{}";frame->packed_yuyv.resize(280);
    uint32_t state=0x9e3779b9u+id*0x10203u;for(auto& byte:frame->packed_yuyv){state^=state<<13;state^=state>>17;state^=state<<5;byte=uint8_t(state);}return frame;
}
static void run(const std::shared_ptr<DaxaFoundation>& context,unsigned wave,DaxaStereoReference eye,unsigned disparities,unsigned paths,unsigned p2,bool subpixel){
    DaxaRectification maps;maps.eye_width=10;maps.source_height=7;maps.stride_bytes=40;maps.width=9;maps.height=7;maps.original_identity="overwrite-original";maps.rectified_identity="overwrite-rectified";
    for(auto& map:maps.maps)map.resize(63);
    for(unsigned e=0;e<2;++e)for(unsigned y=0;y<7;++y)for(unsigned x=0;x<9;++x){maps.maps[e*2][y*9+x]=float(x);maps.maps[e*2+1][y*9+x]=float(y);}
    StereoConfig config{9,7,disparities,paths,8,p2,subpixel};DaxaPreprocess pre(maps,2,context);auto prototype=pre.enqueue(source(0));
    std::optional<MetricCalibration> metric=eye==DaxaStereoReference::Left?std::optional<MetricCalibration>{MetricCalibration{80,.12f,0}}:std::nullopt;
    DaxaStereo baseline(config,prototype,DaxaSgmVariant::PackedSpecializedOverwrite,2,metric,context,wave,eye);
    DaxaStereo candidate(config,prototype,DaxaSgmVariant::PackedSpecializedOverwriteQuad,2,metric,context,wave,eye);
    prototype.wait();prototype={};pre.collect_completed_sources();std::vector<uint32_t> previous_costs;bool changed=false;
    for(unsigned batch=0;batch<3;++batch){
        std::array<DaxaFrame,2> inputs;std::array<DaxaStereo::Frame,2> old_frames,new_frames;
        for(unsigned slot=0;slot<2;++slot){unsigned id=batch*2+slot+1;inputs[slot]=pre.enqueue(source(id));old_frames[slot]=baseline.process(inputs[slot]);new_frames[slot]=candidate.process(inputs[slot]);}
        for(unsigned slot=0;slot<2;++slot){unsigned id=batch*2+slot+1;old_frames[slot].wait();new_frames[slot].wait();auto a=baseline.readback(old_frames[slot],true),b=candidate.readback(new_frames[slot],true);compare(a,b);
            check(a.reference==eye&&a.packing_bits==(paths*p2<=255?8u:16u)&&a.disparity.size()==63&&a.confidence.size()==63&&a.validity.size()==63&&a.packed_penalties.size()==63*((disparities+32/a.packing_bits-1)/(32/a.packing_bits)),"unexpected eye, packing, or output size");
            check((eye==DaxaStereoReference::Left&&a.depth.size()==63&&a.depth_validity.size()==63)||(eye==DaxaStereoReference::Right&&a.depth.empty()&&a.depth_validity.empty()),"unexpected metric planes");
            check(old_frames[slot].source()->frame_id==id&&new_frames[slot].source()->frame_id==id&&old_frames[slot].calibration_identity()=="overwrite-rectified"&&new_frames[slot].calibration_identity()=="overwrite-rectified","source identity or geometry mismatch");
            if(!previous_costs.empty())changed|=previous_costs!=a.costs;
            previous_costs=std::move(a.costs);
        }
        old_frames={};new_frames={};inputs={};baseline.collect_completed_inputs();candidate.collect_completed_inputs();pre.collect_completed_sources();
    }
    check(changed,"changing inputs did not change stereo costs");
}
int main(int argc,char** argv){try{
    check(argc==2&&(std::string(argv[1])=="32"||std::string(argv[1])=="64"),"usage: daxa_stereo_quad_parity 32|64");unsigned wave=unsigned(std::stoul(argv[1]));auto context=std::make_shared<DaxaFoundation>(true);
    for(auto eye:{DaxaStereoReference::Left,DaxaStereoReference::Right})for(auto d:{3u,33u})for(auto paths:{4u,8u})for(auto p2:{31u,64u})for(auto subpixel:{false,true})run(context,wave,eye,d,paths,p2,subpixel);
    context->close();check(context->validation_errors.load()==0&&context->validation_warnings.load()==0,"Vulkan validation diagnostics");std::cout<<"exact GPU overwrite/quad parity with dynamic two-slot reuse passed; wave="<<wave<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
