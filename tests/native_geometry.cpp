#include <stereoforge/geometry.hpp>
#include <stereoforge/runtime.hpp>
#include <stereoforge/native_device.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace stereoforge;
namespace {
void require(bool ok, const char* detail) { if (!ok) throw std::runtime_error(detail); }
void close(float actual, float expected) { require(std::abs(actual-expected)<1e-5f, "unexpected projected metric value"); }
}
int main(int argc, char**) {
    try {
        const bool validation = argc > 1;
        GeometryConfig config{3, 1, 3, 1};
        NativeDevice shared(validation);
        NativeGeometry engine(config, STEREOFORGE_SHADER_DIRECTORY, shared);
        NativeStereo stereo({5,5,4,4,8,32,false},std::string(STEREOFORGE_SHADER_DIRECTORY)+"/stereo",shared);
        require(engine.device_id()==shared.id()&&stereo.device_id()==shared.id(),"engines did not share context identity");
        require(shared.rhi_device()!=nullptr&&shared.rhi_queue()!=nullptr,"shared native context access missing");
        std::vector<float> depth{1,4,NAN, NAN,NAN,NAN}, confidence(6,.8f);
        std::vector<std::uint32_t> rgb{1,2,3,4,5,6}, provenance(6,1);
        std::vector<Ray3> rays{{0,0,1},{1,0,1},{2,0,1},{0,0,1},{1,0,1},{2,0,1}};
        std::vector<std::int64_t> ids{(1LL<<40)+1,2,3,4,5,6}, ages(6,7);
        GeometrySources sources{depth,confidence,rgb,provenance,rays,ids,ages,1,"synthetic"};
        auto bad_sources=sources;bad_sources.calibration_id.clear();
        bool bad_metadata=false;
        try { auto ignored=engine.project(bad_sources,VirtualView{}); } catch(const std::invalid_argument&) {bad_metadata=true;}
        require(bad_metadata,"projection accepted missing calibration identity");
        bad_sources=sources;bad_sources.timestamp_ns=-1;bad_metadata=false;
        try { auto ignored=engine.project(bad_sources,VirtualView{}); } catch(const std::invalid_argument&) {bad_metadata=true;}
        require(bad_metadata,"projection accepted negative timestamp");
        VirtualView view;
        view.intrinsics={1,1,0,0};
        auto projected = engine.project(sources,view);
        auto actual = engine.readback(projected);
        require(actual.validity == std::vector<std::uint32_t>({1,1,0}),"identity validity");
        require(actual.rgb[0]==1 && actual.rgb[1]==2,"identity RGB");
        close(actual.depth[0],1);close(actual.depth[1],4);
        require(actual.source_ids[0]==ids[0] && actual.age_ns[0]==7,"64-bit projection ownership");
        auto uploaded=engine.upload_sources(sources);
        require(uploaded.completion.device_id()==shared.id(),"upload completion context");
        auto invalid_device=uploaded;invalid_device.depth.device_id=shared.id()+1;
        bool invalid_resident=false;
        try{auto ignored=engine.project(invalid_device,view);}catch(const std::invalid_argument&){invalid_resident=true;}
        require(invalid_resident,"resident foreign context accepted");
        for(int field=0;field<3;++field){
            auto malformed=uploaded;
            if(field==0)malformed.depth.size_bytes-=4;
            if(field==1)malformed.depth.offset_bytes=4;
            if(field==2)malformed.depth.stride_bytes=8;
            invalid_resident=false;try{auto ignored=engine.project(malformed,view);}catch(const std::invalid_argument&){invalid_resident=true;}
            require(invalid_resident,"resident size/offset/stride accepted");
        }
        // Do not wait on upload: one shared queue orders producer and consumer.
        auto resident=engine.project(uploaded,view);uploaded={};
        require(resident.source_ids.native&&resident.age_ns.native,"resident final metadata planes missing");
        auto resident_result=engine.readback(resident);
        require(resident_result.source_ids[0]==ids[0]&&resident_result.age_ns[0]==7,"resident GPU metadata gather lost 64-bit fields");
        require(resident_result.rgb==actual.rgb&&resident_result.validity==actual.validity,"resident projection differs from host source path");resident={};
        auto second=engine.project(sources,view);
        bool full=false;
        try { auto ignored=engine.project(sources,view); } catch(const std::runtime_error&) { full=true; }
        require(full,"geometry overwrote a leased output slot");
        second.completion.wait();second={};projected={};
        depth[3]=.5f;
        auto right_nearer=engine.project(sources,view);
        actual=engine.readback(right_nearer);
        require(actual.owner[0]==3&&actual.source[0]==2&&actual.rgb[0]==4,"nearer right-eye surface did not win");
        close(actual.depth[0],.5f);right_nearer={};depth[3]=NAN;
        view.mode=ViewMode::Preview;
        bool rejected=false;
        try { auto ignored=engine.project(sources,view); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"preview accepted missing calibration map");
        view.mode=ViewMode::Strict;
        // Two source pixels collide after a known translation. The nearer wins.
        view.source_to_virtual_left[3]=1;
        auto shifted=engine.project(sources,view);
        actual=engine.readback(shifted);
        require(actual.validity==std::vector<std::uint32_t>({0,1,0}),"projection occlusion visibility");
        close(actual.depth[1],1); require(actual.owner[1]==0,"projection nearest owner");
        shifted={};
        // Cylindrical depth is range; ray (1,0,1)*4 has range sqrt(32).
        view.source_to_virtual_left=identity_pose(); view.projection=Projection::Cylindrical;
        view.hfov_radians=3.14159265358979323846f;view.intrinsics={1,1,1,0};
        auto cylindrical=engine.project(sources,view);
        actual=engine.readback(cylindrical);
        close(actual.depth[2],std::sqrt(32.f));require(cylindrical.depth_convention==DepthConvention::RayRange,"cylindrical convention");cylindrical={};
        // Shared compositor: compatible surfaces blend, with optional GPU-only
        // exposure estimation; incompatible surfaces retain the nearer owner.
        {
            const auto pack=[](unsigned r,unsigned g,unsigned b){return r|(g<<8)|(b<<16);};
            const auto channel=[](std::uint32_t value,unsigned c){return int((value>>(8*c))&255);};
            std::vector<float> full_depth(6,2.f);
            std::vector<std::uint32_t> full_rgb(6,pack(180,120,60));
            for(std::size_t i=3;i<6;++i)full_rgb[i]=pack(100,60,30);
            auto full=sources;full.depth=full_depth;full.rgb=full_rgb;
            VirtualView composite;composite.intrinsics={1,1,0,0};composite.exposure_match=false;
            auto blended=engine.project(full,composite);actual=engine.readback(blended);
            require(actual.source==std::vector<std::uint32_t>({3,3,3}),"compatible surfaces were not blended");
            require(actual.rgb_validity==std::vector<std::uint32_t>({1,1,1})&&actual.approximate==std::vector<std::uint32_t>({0,0,0}),"strict RGB masks");
            require(actual.rgb[0]==pack(140,90,45),"compatible blend literal RGB mismatch");
            require(actual.owner[0]==0&&actual.source_ids[0]==ids[0],"blended metric owner changed");blended={};
            composite.exposure_match=true;
            auto matched=engine.project(full,composite);actual=engine.readback(matched);
            const int expected[3]={180,120,60};
            for(unsigned c=0;c<3;++c)require(std::abs(channel(actual.rgb[0],c)-expected[c])<=1,"exposure histogram channel mismatch");
            require(actual.source[0]==3,"exposure matched attribution");matched={};
            composite.exposure_match=false;full_depth[3]=.5f;full_rgb[3]=pack(200,30,180);
            auto foreground=engine.project(full,composite);actual=engine.readback(foreground);
            close(actual.depth[0],.5f);
            require(actual.owner[0]==3&&actual.source[0]==2&&actual.rgb[0]==pack(200,30,180),"foreground contaminated by incompatible background");foreground={};
            std::fill(full_depth.begin(),full_depth.end(),NAN);
            std::vector<std::uint32_t> mapping{0,1,0xffffffff,3,4,5};
            full.preview_indices=mapping;composite.mode=ViewMode::Preview;
            auto preview=engine.project(full,composite);actual=engine.readback(preview);
            require(actual.rgb_validity==std::vector<std::uint32_t>({1,1,1})&&actual.approximate==std::vector<std::uint32_t>({1,1,1}),"preview RGB masks");
            require(actual.validity==std::vector<std::uint32_t>({0,0,0})&&actual.source==std::vector<std::uint32_t>({1,1,2}),"preview created metric observations");
            require(actual.owner[2]==5&&actual.source_ids==std::vector<std::int64_t>({-1,-1,-1})&&actual.age_ns==std::vector<std::int64_t>({0,0,0}),"preview RGB owner leaked metric identity");
            for(float value:actual.depth)require(std::isnan(value),"preview hallucinated metric depth");preview={};
            auto preview_upload=engine.upload_sources(full);
            auto resident_preview=engine.project(preview_upload,composite);preview_upload={};
            auto resident_preview_result=engine.readback(resident_preview);
            require(resident_preview_result.source_ids==std::vector<std::int64_t>({-1,-1,-1})&&resident_preview_result.age_ns==std::vector<std::int64_t>({0,0,0}),"resident preview leaked metric metadata");
            require(resident_preview_result.approximate==std::vector<std::uint32_t>({1,1,1}),"resident preview masks");resident_preview={};
            mapping={0,0xffffffff,0xffffffff,0xffffffff,0xffffffff,0xffffffff};
            auto holes=engine.project(full,composite);actual=engine.readback(holes);
            require(actual.rgb_validity==std::vector<std::uint32_t>({1,0,0})&&actual.approximate==std::vector<std::uint32_t>({1,0,0}),"unsupported preview holes filled");holes={};
            auto invalid=full;invalid.preview_indices=std::span<const std::uint32_t>(mapping).first(5);
            rejected=false;try{auto ignored=engine.project(invalid,composite);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"preview map dimensions accepted");
            mapping[1]=6;rejected=false;try{auto ignored=engine.project(full,composite);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"preview out-of-range source index accepted");
            mapping[1]=3;rejected=false;try{auto ignored=engine.project(full,composite);}catch(const std::invalid_argument&){rejected=true;}
            require(rejected,"preview wrong-eye index attribution accepted");
            full.preview_indices={};composite.mode=ViewMode::Strict;
            auto strict_holes=engine.project(full,composite);actual=engine.readback(strict_holes);
            require(actual.rgb_validity==std::vector<std::uint32_t>({0,0,0})&&actual.approximate==std::vector<std::uint32_t>({0,0,0}),"strict mode retained preview state");
        }
        // Temporal translation x=1 maps source zero to target one and hides z=4.
        std::vector<float> z{1,4,NAN}, conf(3,.8f), disparity(3,3);
        std::vector<std::uint32_t> valid{1,1,0}, prov(3,1);
        std::vector<std::int64_t> tid{(1LL<<40)+1,2,-1}, age(3,0);
        TemporalInput previous{z,conf,disparity,valid,prov,tid,age,0,"cal"};
        auto pose=identity_pose();pose[3]=1;
        auto warped=engine.reproject(previous,{1,1,0,0},pose,10,MetricDisparity{1,.2f,0});
        require(warped.timestamp_ns==10&&warped.observation_elapsed_ns==10&&warped.calibration_id=="cal","temporal frame metadata");
        actual=engine.readback(warped);
        require(actual.validity==std::vector<std::uint32_t>({0,1,0}),"temporal visibility");
        close(actual.depth[1],1);close(actual.disparity[1],.2f);
        require(actual.source_ids[1]==tid[0]&&actual.age_ns[1]==10&&actual.provenance[1]==2,"temporal source metadata");
        warped={};
        TemporalInput current=previous; current.timestamp_ns=10;
        auto repeated=engine.fuse(current,previous);
        actual=engine.readback(repeated);
        close(actual.confidence[0],.8f);require(actual.provenance[0]==1,"correlated history increased evidence");
        repeated={};
        std::vector<std::int64_t> independent{tid[0]+(1LL<<32),10,-1};previous.source_ids=independent;
        auto fused=engine.fuse(current,previous);
        actual=engine.readback(fused);
        require(actual.provenance[0]==3&&std::isnan(actual.disparity[0]),"independent fusion provenance");
        close(actual.confidence[0],.8f);fused={};
        auto expired=engine.fuse(current,previous,0);
        actual=engine.readback(expired);
        require(actual.provenance[0]==1,"expired history accepted");expired={};
        previous.calibration_id="other";
        auto incompatible=engine.fuse(current,previous);
        actual=engine.readback(incompatible);require(actual.provenance[0]==1,"calibration mismatch accepted");incompatible={};previous.calibration_id="cal";
        previous.timestamp_ns=11;
        auto future=engine.fuse(current,previous);
        actual=engine.readback(future);require(actual.provenance[0]==1,"future history accepted");future={};previous.timestamp_ns=0;
        valid[0]=0;z[0]=NAN;
        // Current input is missing but prediction has independent valid geometry.
        std::vector<float> predicted_z{1,4,NAN};std::vector<std::uint32_t> predicted_valid{1,1,0};
        previous.depth=predicted_z;previous.validity=predicted_valid;
        auto filled=engine.fuse(current,previous);
        actual=engine.readback(filled);
        require(actual.choose_previous[0]==1&&actual.source_ids[0]==independent[0]&&actual.age_ns[0]==10,"history fill metadata");
        filled={};
        GeometryFrame retained;
        {
            NativeGeometry temporary(config,STEREOFORGE_SHADER_DIRECTORY,validation);
            retained=temporary.project(sources,VirtualView{});
            bool foreign=false;
            try { auto ignored=engine.readback(retained); } catch(const std::invalid_argument&) {foreign=true;}
            require(foreign,"foreign geometry device accepted");
        }
        retained.completion.wait();require(retained.completion.ready(),"geometry resources did not outlive facade");
        std::cout << "native strict perspective/cylindrical/occlusion and temporal translation/fusion contracts passed\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
