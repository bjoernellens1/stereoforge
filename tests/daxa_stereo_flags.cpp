#include <stereoforge/daxa_depth.hpp>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace stereoforge;

static void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void check_output(const DaxaStereoReadback& actual,const DaxaStereoReadback& expected,bool subpixel){
    check(actual.validity==expected.validity,"descriptor/subpixel flag validity oracle");
    unsigned valid=0,fractional=0;
    for(size_t i=0;i<actual.validity.size();++i){
        if(!actual.validity[i]){check(std::bit_cast<uint32_t>(actual.disparity[i])==0x7fc00000u&&actual.confidence[i]==0,"flag control invalid output canonicalization");continue;}
        ++valid;auto d=actual.disparity[i];check(std::isfinite(d)&&d>0,"flag control finite positive disparity");
        if(d!=std::round(d))++fractional;
        if(!subpixel)check(std::bit_cast<uint32_t>(d)==std::bit_cast<uint32_t>(expected.disparity[i]),"descriptor flag enabled disabled subpixel refinement");
    }
    check(valid>0,"flag fixture must exercise accepted disparities");
    check(subpixel?fractional>0:fractional==0,"subpixel flag must independently control fractional output");
}
static DaxaRectification maps(){
    DaxaRectification m;m.eye_width=48;m.source_height=24;m.stride_bytes=192;m.width=48;m.height=24;
    m.original_identity="synthetic-flags-original";m.rectified_identity="synthetic-flags-rectified";
    DaxaRectifiedGeometry g;g.original_identity=m.original_identity;g.rectified_identity=m.rectified_identity;
    for(unsigned eye=0;eye<2;++eye){g.original.eyes[eye]={48,24,80,80,23.5,11.5,{}};g.rectified.eyes[eye]=g.original.eyes[eye];}
    g.original.right_to_left={1,0,0,.1,0,1,0,0,0,0,1,0,0,0,0,1};g.rectified.right_to_left=g.original.right_to_left;
    g.R1={1,0,0,0,1,0,0,0,1};g.R2=g.R1;g.P1={80,0,23.5,0,0,80,11.5,0,0,0,1,0};g.P2={80,0,23.5,-8,0,80,11.5,0,0,0,1,0};m.geometry=g;
    for(auto& map:m.maps)map.resize(48*24);
    for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<24;++y)for(unsigned x=0;x<48;++x){m.maps[eye*2][y*48+x]=float(x);m.maps[eye*2+1][y*48+x]=float(y);}
    m.validate();return m;
}
static void direct(const std::shared_ptr<DaxaFoundation>& context,uint32_t wave){
    for(auto variant:{DaxaSgmVariant::Baseline,DaxaSgmVariant::Wave,DaxaSgmVariant::Packed,DaxaSgmVariant::PackedSpecialized,DaxaSgmVariant::PackedSpecializedOverwrite,DaxaSgmVariant::PackedSpecializedOverwriteQuad,DaxaSgmVariant::PackedD128,DaxaSgmVariant::PackedQuad,DaxaSgmVariant::PackedShuffle})
    for(auto paths:{4u,8u})for(bool hybrid:{false,true})for(bool subpixel:{false,true}){
        StereoConfig c{48,24,variant==DaxaSgmVariant::PackedD128?128u:16u,paths,8,32,subpixel,hybrid};auto f=daxa_stereo_fixture(c);
        for(auto eye:{DaxaStereoReference::Left,DaxaStereoReference::Right}){
            if(eye==DaxaStereoReference::Right&&variant!=DaxaSgmVariant::PackedSpecialized&&variant!=DaxaSgmVariant::PackedSpecializedOverwrite&&variant!=DaxaSgmVariant::PackedSpecializedOverwriteQuad)continue;
            auto expected=daxa_stereo_reference(c,f[0],f[1],std::nullopt,eye);
            DaxaStereo engine(c,f[0],f[1],variant,1,true,std::nullopt,context,variant==DaxaSgmVariant::Baseline?0u:wave,eye);auto frame=engine.process();frame.wait();auto actual=engine.readback(frame,true);
            check(actual.census==expected.census&&actual.census_valid==expected.census_valid&&actual.costs==expected.costs&&actual.sums==expected.sums,"flags preserve exact descriptor/cost/aggregation oracle");
            check_output(actual,expected,subpixel);
        }
    }
}
static void derived(const std::shared_ptr<DaxaFoundation>& context,uint32_t wave){
    auto m=maps();auto source=std::make_shared<CaptureFrame>();source->source_width=96;source->height=24;source->stride_bytes=192;source->calibration_identity=m.original_identity;source->frame_id=1;source->host_ingress_timestamp_ns=1;source->packed_yuyv.assign(192*24,128);
    auto fixture=daxa_stereo_fixture(StereoConfig{48,24,16,4});
    for(unsigned eye=0;eye<2;++eye)for(unsigned y=0;y<24;++y)for(unsigned x=0;x<48;++x)source->packed_yuyv[y*192+2*(eye*48+x)]=uint8_t(fixture[eye][y*48+x]);
    auto images=daxa_preprocess_reference(*source,m);
    for(auto paths:{4u,8u})for(bool hybrid:{false,true})for(bool subpixel:{false,true}){
        StereoConfig c{48,24,16,paths,8,32,subpixel,hybrid};DaxaDepthConfig config{c,1.f,1,wave,DaxaDepthStereoSchedule::SharedCensusSequential,DaxaRightReferenceMode::LeftAggregatedSelection};
        DaxaDepthPipeline pipeline(m,config,context,source);auto output=pipeline.enqueue(source);output.wait();pipeline.check_derived_stereo_integer(output,images);
        auto left=daxa_stereo_reference(c,images.planes[4],images.planes[5]);auto right=daxa_left_aggregated_selection_reference(c,left);
        check_output(pipeline.readback_stereo(output,DaxaStereoReference::Left,false),left,subpixel);
        check_output(pipeline.readback_stereo(output,DaxaStereoReference::Right,false),right,subpixel);
    }
}
int main(int argc,char** argv){try{
    const uint32_t wave=argc>1?uint32_t(std::stoul(argv[1])):32u;check(wave==32||wave==64,"explicit wave32/64 required");
    auto context=std::make_shared<DaxaFoundation>(true);direct(context,wave);derived(context,wave);context->close();
    check(context->validation_errors.load()==0&&context->validation_warnings.load()==0,"flag contracts Vulkan validation through teardown");
    std::cout<<"Descriptor/subpixel flag cross-product passed: all direct selector families and derived right, 8/16bit packing, wave"<<wave<<'\n';return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
