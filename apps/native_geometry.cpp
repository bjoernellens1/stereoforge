#include <stereoforge/geometry.hpp>
#include <fstream>
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
        bool validation=false;std::string projection="perspective",output;
        for(int i=1;i<argc;++i) {
            std::string arg=argv[i];
            if(arg=="--validation")validation=true;
            else if(arg=="--help") {
                std::cout<<"native_geometry [--projection perspective|cylindrical] [--output PREFIX] [--validation]\n"
                    "Runs deterministic two-eye strict GPU projection; use the C++ NativeGeometry API for host camera inputs and temporal stages.\n"
                    "Preview, exposure compensation and seam blending are unavailable in this native renderer.\n";return 0;
            } else if((arg=="--projection"||arg=="--output")&&i+1<argc) {
                (arg=="--projection"?projection:output)=argv[++i];
            } else throw std::invalid_argument("unknown option or missing value: "+arg);
        }
        if(projection!="perspective"&&projection!="cylindrical")throw std::invalid_argument("unsupported projection");
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
        auto frame=engine.project({depths,confidences,rgb,provenance,rays,ids,ages,1,"synthetic"},view);
        auto result=engine.readback(frame);
        if(!output.empty()) {
            write(output+".depth.f32",result.depth);write(output+".confidence.f32",result.confidence);
            write(output+".rgb.u32",result.rgb);write(output+".validity.u32",result.validity);
            write(output+".owner.u32",result.owner);write(output+".source_ids.i64",result.source_ids);
        }
        const auto valid=std::accumulate(result.validity.begin(),result.validity.end(),0u);
        std::cout<<"{\"backend\":\"vulkan\",\"adapter\":"<<std::quoted(engine.device_name())
            <<",\"projection\":"<<std::quoted(projection)<<",\"depth_convention\":"<<std::quoted(projection=="perspective"?"axial_z":"ray_range")
            <<",\"valid_pixels\":"<<valid<<",\"total_pixels\":21,\"strict\":true,\"host_upload_readback\":true}\n";
    } catch(const std::exception& error){std::cerr<<"native_geometry: "<<error.what()<<'\n';return 1;}
}
