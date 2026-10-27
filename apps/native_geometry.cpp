#include <stereoforge/geometry.hpp>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {
template<class T> void write(const std::string& file,const std::vector<T>& values) {
    std::ofstream stream(file,std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(T)));
    if(!stream)throw std::runtime_error("cannot write "+file);
}
}
int main(int argc,char** argv) {
    try {
        bool validation=false,exposure_match=true;std::string projection="perspective",mode="strict",output;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--validation")validation=true;
            else if(arg=="--no-exposure-match")exposure_match=false;
            else if(arg=="--help") {
                std::cout<<"native_geometry [--projection perspective|cylindrical] [--mode strict|preview] [--no-exposure-match] [--output PREFIX] [--validation]\n"
                    "Runs deterministic two-eye GPU depth-compatible fusion; use NativeGeometry for host camera inputs and temporal stages.\n"
                    "Preview uses caller-supplied calibration-only infinity maps; RGB validity/approximation remain separate from metric validity.\n";return 0;
            } else if((arg=="--projection"||arg=="--mode"||arg=="--output")&&i+1<argc) {
                (arg=="--projection"?projection:(arg=="--mode"?mode:output))=argv[++i];
            } else throw std::invalid_argument("unknown option or missing value: "+arg);
        }
        if(projection!="perspective"&&projection!="cylindrical")throw std::invalid_argument("unsupported projection");
        if(mode!="strict"&&mode!="preview")throw std::invalid_argument("unsupported mode");
        stereoforge::NativeGeometry engine({5,3,7,3},STEREOFORGE_SHADER_DIRECTORY,validation);
        std::vector<float> depths(30,2),confidences(30,.8f);
        std::vector<std::uint32_t> rgb(30),provenance(30,1);
        std::vector<stereoforge::Ray3> rays(30);
        std::vector<std::int64_t> ids(30),ages(30,0);
        for(int eye=0;eye<2;++eye)for(int y=0;y<3;++y)for(int x=0;x<5;++x){
            const auto i=eye*15+y*5+x;rays[i]={float(x-2)/4,float(y-1)/4,1};
            rgb[i]=eye==0?180u:(100u<<8);ids[i]=i;
        }
        stereoforge::VirtualView view;view.intrinsics={4,4,3,1};
        view.projection=projection=="perspective"?stereoforge::Projection::Perspective:stereoforge::Projection::Cylindrical;
        view.source_to_virtual_left[3]=-.06f;view.source_to_virtual_right[3]=.06f;
        view.mode=mode=="strict"?stereoforge::ViewMode::Strict:stereoforge::ViewMode::Preview;
        view.exposure_match=exposure_match;
        std::vector<std::uint32_t> preview_indices;
        if(view.mode==stereoforge::ViewMode::Preview){
            // Synthetic calibrated caller-side inverse map. Translation is
            // deliberately absent for infinity directions; never metric depth.
            preview_indices.assign(42,0xffffffff);
            for(int eye=0;eye<2;++eye)for(int v=0;v<3;++v)for(int u=0;u<7;++u){
                float x=(u-3.f)/4,y=(v-1.f)/4,z=1;
                if(view.projection==stereoforge::Projection::Cylindrical){
                    const float angle=((u+.5f)/7-.5f)*view.hfov_radians;x=std::sin(angle);z=std::cos(angle);
                }
                if(z<=0)continue;
                const float sx=4*x/z+2,sy=4*y/z+1;
                if(sx<-.5f||sx>=4.5f||sy<-.5f||sy>=2.5f)continue;
                const auto source=eye*15+int(std::floor(sy+.5f))*5+int(std::floor(sx+.5f));
                preview_indices[eye*21+v*7+u]=std::uint32_t(source);
            }
        }
        auto frame=engine.project({depths,confidences,rgb,provenance,rays,ids,ages,1,"synthetic",preview_indices},view);
        auto result=engine.readback(frame);
        if(!output.empty()) {
            write(output+".depth.f32",result.depth);write(output+".confidence.f32",result.confidence);
            write(output+".rgb.u32",result.rgb);write(output+".validity.u32",result.validity);
            write(output+".owner.u32",result.owner);write(output+".source_ids.i64",result.source_ids);
            write(output+".rgb_validity.u32",result.rgb_validity);write(output+".approximate.u32",result.approximate);
        }
        const auto valid=std::accumulate(result.validity.begin(),result.validity.end(),0u);
        const auto rgb_valid=std::accumulate(result.rgb_validity.begin(),result.rgb_validity.end(),0u);
        const auto approximate=std::accumulate(result.approximate.begin(),result.approximate.end(),0u);
        std::cout<<"{\"backend\":\"vulkan\",\"adapter\":"<<std::quoted(engine.device_name())
            <<",\"projection\":"<<std::quoted(projection)<<",\"depth_convention\":"<<std::quoted(projection=="perspective"?"axial_z":"ray_range")
            <<",\"valid_pixels\":"<<valid<<",\"rgb_valid_pixels\":"<<rgb_valid<<",\"approximate_pixels\":"<<approximate
            <<",\"total_pixels\":21,\"mode\":"<<std::quoted(mode)<<",\"exposure_match\":"<<(exposure_match?"true":"false")<<",\"host_upload_readback\":true}\n";
    } catch(const std::exception& error){std::cerr<<"native_geometry: "<<error.what()<<'\n';return 1;}
}
