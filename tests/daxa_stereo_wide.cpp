#include <stereoforge/daxa_stereo.hpp>
#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace stereoforge;
static void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F>static void rejects(F action){bool rejected=false;try{action();}catch(const std::invalid_argument&){rejected=true;}check(rejected,"unsupported wide range accepted");}

static void admission(){
    StereoConfig wide{640,9,288,4,8,32,false};
    rejects([&]{wide.validate();}); // Generic/RHI kernels keep their existing capacity.
    for(auto eye:{DaxaStereoReference::Left,DaxaStereoReference::Right})
        DaxaStereo::validate_shape(wide,1,std::nullopt,DaxaSgmVariant::PackedSpecialized,eye);
    for(auto variant:{DaxaSgmVariant::Baseline,DaxaSgmVariant::Wave,DaxaSgmVariant::Packed,
                      DaxaSgmVariant::PackedD128,DaxaSgmVariant::PackedQuad,DaxaSgmVariant::PackedShuffle,
                      DaxaSgmVariant::PackedSpecializedOverwrite,DaxaSgmVariant::PackedSpecializedOverwriteQuad})
        rejects([&]{DaxaStereo::validate_shape(wide,1,std::nullopt,variant);});
    for(auto disparities:{257u,287u,289u,320u,512u}){
        auto invalid=wide;invalid.disparities=disparities;
        rejects([&]{DaxaStereo::validate_shape(invalid,1,std::nullopt,DaxaSgmVariant::PackedSpecialized);});
    }
    auto invalid=wide;invalid.p1=33;
    rejects([&]{DaxaStereo::validate_shape(invalid,1,std::nullopt,DaxaSgmVariant::PackedSpecialized);});
    invalid=wide;invalid.paths=8;invalid.p2=8192;
    rejects([&]{DaxaStereo::validate_shape(invalid,1,std::nullopt,DaxaSgmVariant::PackedSpecialized);});
}

static void run(bool gpu,uint32_t wave){
    std::shared_ptr<DaxaFoundation> context;
    if(gpu)context=std::make_shared<DaxaFoundation>(true);
    for(auto paths:{4u,8u})for(auto p2:{31u,32u})for(bool ortho:{false,true}){
        StereoConfig config{640,9,288,paths,8,p2,true,ortho};
        auto images=daxa_stereo_fixture(config);
        for(uint32_t y=0;y<config.height;++y)for(uint32_t x=0;x<config.width;++x)
            images[1][y*config.width+x]=x+270<config.width?images[0][y*config.width+x+270]:0;
        for(auto eye:{DaxaStereoReference::Left,DaxaStereoReference::Right}){
            auto scalar=daxa_stereo_reference(config,images[0],images[1],std::nullopt,eye);
            auto packed=daxa_sgm_packed_model(config,scalar.costs,wave);
            auto bits=DaxaStereo::packed_bits(config),components=32/bits,words=(288+components-1)/components;
            for(size_t pixel=0;pixel<size_t(config.width)*config.height;++pixel)for(uint32_t d=0;d<288;++d){
                auto index=pixel*288+d;
                auto residual=(packed[pixel*words+d/components]>>((d%components)*bits))&((1u<<bits)-1);
                check(residual==scalar.sums[index]-paths*scalar.costs[index],"wide packed model differs from independent scalar recurrence");
            }
            size_t recovered=0;
            for(size_t pixel=0;pixel<scalar.validity.size();++pixel)
                recovered+=scalar.validity[pixel]&&std::abs(scalar.disparity[pixel]-270.f)<.5f;
            check(recovered>100,"known270px shift must exercise disparities beyond256");
            if(!gpu)continue;
            {
                DaxaStereo engine(config,images[0],images[1],DaxaSgmVariant::PackedSpecialized,1,true,
                                  std::nullopt,context,wave,eye);
                auto frame=engine.process();frame.wait();auto actual=engine.readback(frame,true);
                check(actual.reference==eye&&actual.census==scalar.census&&actual.census_valid==scalar.census_valid,
                      "wide GPU reference and descriptor mismatch");
                check(actual.costs==scalar.costs&&actual.sums==scalar.sums&&actual.validity==scalar.validity,
                      "wide GPU costs/recurrence/selection masks differ from scalar");
                check(actual.packed_penalties==packed,"wide GPU packed words mismatch, including masked subgroup tail");
                for(size_t pixel=0;pixel<actual.validity.size();++pixel){
                    if(!actual.validity[pixel]){
                        check(std::bit_cast<uint32_t>(actual.disparity[pixel])==0x7fc00000u&&actual.confidence[pixel]==0,
                              "wide invalid output must retain canonical NaN/zero confidence");
                        continue;
                    }
                    float expected=scalar.disparity[pixel];
                    float ulp=std::nextafter(expected,INFINITY)-expected;
                    check(std::isfinite(actual.disparity[pixel])&&std::abs(actual.disparity[pixel]-expected)<=2*ulp,
                          "wide GPU subpixel differs by more than2ULP");
                }
            }
        }
    }
    if(context){context->close();check(context->validation_errors==0&&context->validation_warnings==0,
                                     "wide Vulkan validation including teardown");}
}

int main(int argc,char**argv){try{
    bool gpu=argc==3&&std::string(argv[1])=="--gpu";
    check(gpu||(argc==2&&std::string(argv[1])=="--cpu-only"),"expected --cpu-only or --gpu 32|64");
    auto wave=gpu?uint32_t(std::stoul(argv[2])):32u;
    check(wave==32||wave==64,"full wave32/64 required");
    admission();run(gpu,wave);
    if(!gpu)run(false,64);
    std::cout<<"D288 independent-eye scalar/packed"<<(gpu?"/GPU":"")<<" contracts passed\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
