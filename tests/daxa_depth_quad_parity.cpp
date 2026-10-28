#include <stereoforge/daxa_depth.hpp>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace stereoforge;

static void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class T> static bool exact(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
static DaxaRectification geometry(){
    DaxaRectification m;m.eye_width=24;m.source_height=7;m.stride_bytes=96;m.width=23;m.height=7;m.original_identity="quad-depth-original";m.rectified_identity="quad-depth-rectified";
    DaxaRectifiedGeometry g;g.original_identity=m.original_identity;g.rectified_identity=m.rectified_identity;
    for(unsigned eye=0;eye<2;++eye){g.original.eyes[eye]={24,7,80,80,11,3,{}};g.rectified.eyes[eye]={23,7,80,80,11,3,{}};}
    g.original.right_to_left={1,0,0,.12,0,1,0,0,0,0,1,0,0,0,0,1};g.rectified.right_to_left=g.original.right_to_left;g.R1={1,0,0,0,1,0,0,0,1};g.R2=g.R1;g.P1={80,0,11,0,0,80,3,0,0,0,1,0};g.P2={80,0,11,-9.6,0,80,3,0,0,0,1,0};m.geometry=g;
    for(auto& map:m.maps)map.resize(161);for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<7;++y)for(unsigned x=0;x<23;++x){m.maps[eye*2][y*23+x]=float(x);m.maps[eye*2+1][y*23+x]=float(y);}m.validate();m.metric_calibration();return m;
}
static CaptureLease source(unsigned id){
    auto f=std::make_shared<CaptureFrame>();f->source_width=48;f->height=7;f->stride_bytes=96;f->calibration_identity="quad-depth-original";f->frame_id=id;f->camera_timestamp_ns=id;f->host_ingress_timestamp_ns=0;f->controls_json="{}";f->packed_yuyv.resize(672);
    auto luma=[id](unsigned x,unsigned y){uint32_t v=(x+1)*0x9e3779b9u^(y+1)*0x85ebca6bu^(id*0xc2b2ae35u);v^=v>>16;v*=0x7feb352du;v^=v>>15;return uint8_t(v);};
    unsigned shift=1+(id%2);for(unsigned y=0;y<7;++y)for(unsigned eye=0;eye<2;++eye)for(unsigned x=0;x<24;x+=2){auto pos=y*96+eye*48+x*2;f->packed_yuyv[pos]=luma(x+(eye?shift:0),y);f->packed_yuyv[pos+1]=128;f->packed_yuyv[pos+2]=luma(x+1+(eye?shift:0),y);f->packed_yuyv[pos+3]=128;}return f;
}
static void compare_depth(const DaxaDepthReadback& a,const DaxaDepthReadback& b){
    check(a.right_reference_mode==b.right_reference_mode&&a.lr_dependency==b.lr_dependency,"dependency metadata mismatch");
    check(exact(a.lr_disparity,b.lr_disparity)&&exact(a.lr_confidence,b.lr_confidence)&&exact(a.lr_validity,b.lr_validity)&&exact(a.depth,b.depth)&&exact(a.metric_confidence,b.metric_confidence)&&exact(a.metric_validity,b.metric_validity)&&exact(a.provenance,b.provenance)&&exact(a.source_ids,b.source_ids)&&exact(a.age_ns,b.age_ns),"LR/depth/provenance/source/age plane byte mismatch");
    check(a.depth.size()==322&&a.source_ids.size()==322,"unexpected two-eye observation size");
}
static void compare_stereo(const DaxaStereoReadback& a,const DaxaStereoReadback& b){
    check(a.reference==b.reference&&a.estimator==b.estimator&&a.packing_bits==b.packing_bits&&a.intermediates_reconstructed&&b.intermediates_reconstructed,"stereo metadata mismatch");
    check(a.census==b.census&&a.census_valid==b.census_valid&&a.costs==b.costs&&a.sums==b.sums&&a.packed_penalties==b.packed_penalties,"stereo intermediate mismatch");
    check(exact(a.disparity,b.disparity)&&exact(a.confidence,b.confidence)&&exact(a.validity,b.validity),"stereo output mismatch");
}
static void run(const std::shared_ptr<DaxaFoundation>& context,unsigned wave,unsigned paths,unsigned p2){
    auto maps=geometry();DaxaDepthConfig config{StereoConfig{23,7,17,paths,8,p2,true},1.f,2,wave};auto reference_config=config;reference_config.stereo_variant=DaxaSgmVariant::PackedSpecializedOverwrite;config.stereo_variant=DaxaSgmVariant::PackedSpecializedOverwriteQuad;
    DaxaDepthPipeline reference(maps,reference_config,context,source(0)),candidate(maps,config,context,source(0));std::array<std::vector<uint32_t>,2> previous;std::array<bool,2> changed{};std::array<bool,2> valid_eye{};
    for(unsigned batch=0;batch<3;++batch){std::array<DaxaDepthPipeline::Frame,2> old_frames,new_frames;std::array<CaptureLease,2> inputs;
        for(unsigned slot=0;slot<2;++slot){inputs[slot]=source(100+batch*2+slot);old_frames[slot]=reference.enqueue(inputs[slot]);new_frames[slot]=candidate.enqueue(inputs[slot]);}
        for(unsigned slot=0;slot<2;++slot){old_frames[slot].wait();new_frames[slot].wait();auto a=reference.readback(old_frames[slot]),b=candidate.readback(new_frames[slot]);compare_depth(a,b);
            check(old_frames[slot].source()->frame_id==inputs[slot]->frame_id&&new_frames[slot].source()->frame_id==inputs[slot]->frame_id,"source frame identity mismatch");
            for(unsigned eye=0;eye<2;++eye){auto e=eye?DaxaStereoReference::Right:DaxaStereoReference::Left;auto x=reference.readback_stereo(old_frames[slot],e,true),y=candidate.readback_stereo(new_frames[slot],e,true);compare_stereo(x,y);if(!previous[eye].empty())changed[eye]|=previous[eye]!=x.costs;previous[eye]=std::move(x.costs);
                for(unsigned i=0;i<161;++i){auto index=eye*161+i;if(a.metric_validity[index]){valid_eye[eye]=true;check(a.source_ids[index]==int64_t(inputs[slot]->frame_id)&&a.age_ns[index]==0&&a.provenance[index]==1,"valid pixel provenance mismatch");}}
            }
        }
        old_frames={};new_frames={};inputs={};reference.collect();candidate.collect();
    }
    check(changed[0]&&changed[1],"varying sources did not change both eye costs");check(valid_eye[0]&&valid_eye[1],"fixture did not exercise valid metric output in both eyes");
}
int main(int argc,char** argv){try{check(argc==2&&(std::string(argv[1])=="32"||std::string(argv[1])=="64"),"usage: daxa_depth_quad_parity 32|64");unsigned wave=unsigned(std::stoul(argv[1]));auto context=std::make_shared<DaxaFoundation>(true);for(auto paths:{4u,8u})for(auto p2:{31u,64u})run(context,wave,paths,p2);context->close();check(context->validation_errors.load()==0&&context->validation_warnings.load()==0,"Vulkan validation diagnostics");std::cout<<"exact dual-eye overwrite/quad parity on changing frames and two-slot reuse passed; wave="<<wave<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
