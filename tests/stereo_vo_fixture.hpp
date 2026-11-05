#pragma once
#include <stereoforge/daxa_stereo_vo.hpp>
#include <stereoforge/daxa_temporal.hpp>
#include <stereoforge/daxa_rectified_input.hpp>
#include <stereoforge/daxa_depth.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace sf_test {
using namespace stereoforge;
inline void check(bool ok,char const* m){if(!ok)throw std::runtime_error(m);}
inline constexpr unsigned width=256,height=192;
inline DaxaRectification geometry(double rectification_angle=0) {
    DaxaRectification c;c.eye_width=width;c.source_height=height;c.width=width;c.height=height;c.original_identity="vo-original";c.rectified_identity="vo-rectified";
    DaxaRectifiedGeometry g;g.original_identity=c.original_identity;g.rectified_identity=c.rectified_identity;
    for(auto& e:g.original.eyes)e={width,height,256,256,127.5,95.5,{}};
    g.rectified=g.original;g.original.right_to_left=g.rectified.right_to_left={1,0,0,.1,0,1,0,0,0,0,1,0,0,0,0,1};
    double ca=std::cos(rectification_angle),sa=std::sin(rectification_angle);
    g.R1=g.R2={ca,-sa,0,sa,ca,0,0,0,1};g.original.right_to_left[3]=.1*ca;g.original.right_to_left[7]=-.1*sa;g.P1={256,0,127.5,0,0,256,95.5,0,0,0,1,0};g.P2={256,0,127.5,-25.6,0,256,95.5,0,0,0,1,0};c.geometry=g;
    return c;
}
inline RectifiedRgbFrame render(uint64_t id,int shift,bool blank=false,double angle=0,std::string clock="steady",bool corrupt_right=false,uint64_t acquisition_override=0) {
    auto backing=std::make_shared<std::array<std::vector<uint8_t>,6>>();
    auto side=std::make_shared<CanonicalRgbSource>();side->backing=backing;
    side->input_identity="vo-render-"+std::to_string(id);side->original_pixel_identity=side->input_identity;side->rectified_pixel_identity=side->input_identity;
    side->original_calibration_identity="vo-original";side->rectified_calibration_identity="vo-rectified";
    for(unsigned p=0;p<6;++p) {
        auto& data=(*backing)[p];data.resize(width*height*(p<4?3:1));
        if(p>=4){std::fill(data.begin(),data.end(),1);side->support[p-4]={data,width,height,width,CanonicalRgbFormat::SupportU8};continue;}
        for(unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x) {
            // Ray/plane intersection in world coordinates. Ground-truth camera
            // poses generate images only; the estimator receives RGB and metadata.
            double dx=(double(x)-127.5)/256,dy=(double(y)-95.5)/256;
            double ox=-shift*.00625+(p%2?.1*std::cos(angle):0),oz=p%2?-.1*std::sin(angle):0;
            double rx=std::cos(angle)*dx+std::sin(angle),rz=-std::sin(angle)*dx+std::cos(angle);
            double distance=(1.6-oz)/rz;
            double u=(ox+distance*rx)*160+127.5,v=distance*dy*160+95.5;
            auto value=uint8_t((blank||(corrupt_right&&p%2))?128:128+38*std::sin(.31*u)+31*std::cos(.23*v)+28*std::sin(.17*u+.21*v)+18*std::cos(.45*u-.13*v));
            for(unsigned ch=0;ch<3;++ch)data[(y*width+x)*3+ch]=value;
        }
        side->rgb[p]={data,width,height,width*3,CanonicalRgbFormat::Rgb8};
    }
    auto source=std::make_shared<CaptureFrame>();source->source_kind=CaptureSourceKind::CanonicalRgbMetadata;source->canonical_rgb=side;
    source->source_width=width*2;source->height=height;source->frame_id=id;source->camera_timestamp_ns=acquisition_override?acquisition_override:id*33000000;source->host_ingress_timestamp_ns=id*33000000+5000000;
    source->camera_clock_domain="render-acquisition";source->host_clock_domain=std::move(clock);source->calibration_identity="vo-original";source->device_identity="render-rig";
    return RectifiedRgbFrame{source};
}
// Stationary cameras observe a nearer independently translating textured panel
// and a static farther wall. Both depths produce exact integer disparities.
inline RectifiedRgbFrame render_dynamic(uint64_t id,double displacement,double camera_translation=0,bool full_foreground=false) {
    auto template_frame=render(id,0);auto capture=std::make_shared<CaptureFrame>(*template_frame.source);
    auto side=std::make_shared<CanonicalRgbSource>(*capture->canonical_rgb);
    auto backing=std::make_shared<std::array<std::vector<uint8_t>,6>>();side->backing=backing;
    side->input_identity="moving-panel-"+std::to_string(id)+"-"+std::to_string(displacement);side->original_pixel_identity=side->rectified_pixel_identity=side->input_identity;
    double center=(full_foreground?-.125:-.1)+displacement;
    for(unsigned p=0;p<6;++p) {
        auto& data=(*backing)[p];data.resize(width*height*(p<4?3:1));
        if(p>=4){std::fill(data.begin(),data.end(),1);side->support[p-4]={data,width,height,width,CanonicalRgbFormat::SupportU8};continue;}
        for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x) {
            double dx=(double(x)-127.5)/256,dy=(double(y)-95.5)/256,origin=camera_translation+(p%2?.1:0);
            double fx=origin+dx*1.28,fy=dy*1.28;
            bool foreground=full_foreground||(std::abs(fx-center)<=.5&&std::abs(fy)<=.41);
            double u=foreground?(origin-center+dx*1.28)*200+127.5:(origin+dx*1.6)*160+127.5;
            double v=foreground?fy*200+95.5:dy*1.6*160+95.5;
            double value=foreground?128+41*std::sin(.37*u)+37*std::cos(.29*v)+23*std::sin(.19*u+.23*v)+17*std::cos(.43*u-.17*v):
                128+32*std::sin(.31*u)+28*std::cos(.23*v)+23*std::sin(.17*u+.21*v)+14*std::cos(.45*u-.13*v);
            for(unsigned ch=0;ch<3;++ch)data[(y*width+x)*3+ch]=uint8_t(value);
        }
        side->rgb[p]={data,width,height,width*3,CanonicalRgbFormat::Rgb8};
    }
    capture->canonical_rgb=std::move(side);return RectifiedRgbFrame{capture};
}
inline void overwrite_buffer(std::shared_ptr<DaxaFoundation> const& context,daxa::BufferId destination,std::size_t bytes,uint64_t value,uint64_t offset=0) {
    check(bytes%8==0,"negative witness requires eight-byte aligned size");
    auto& d=context->device;daxa::BufferInfo info{.size=bytes,.name="negative GPU witness"};
    auto memory=d.create_memory({.requirements=context->coherent_readback_requirements(info),.flags=daxa::MemoryFlagBits::HOST_ACCESS_RANDOM});
    auto buffer=d.create_buffer_from_memory_block({.buffer_info=info,.memory_block=memory});
    std::vector<uint64_t> values(bytes/8,value);std::memcpy(d.buffer_host_address(buffer).value(),values.data(),bytes);
    auto recorder=d.create_command_recorder({.name="corrupt GPU witness"});
    recorder.pipeline_barrier({.src_access=daxa::AccessConsts::COMPUTE_SHADER_WRITE|daxa::AccessConsts::HOST_WRITE,.dst_access=daxa::AccessConsts::TRANSFER_READ|daxa::AccessConsts::TRANSFER_WRITE});
    recorder.copy_buffer_to_buffer({.src_buffer=buffer,.dst_buffer=destination,.dst_offset=offset,.size=bytes});
    auto commands=recorder.complete_current_commands();auto ticket=d.submit_commands({.command_lists={&commands,1}});
    d.wait_on_submit({.queue=daxa::QUEUE_MAIN,.queue_submit_index=ticket,.timeout=30'000'000'000ull});d.destroy_buffer(buffer);d.collect_garbage();
}
inline void overwrite_ancestry(std::shared_ptr<DaxaFoundation> const& context,DaxaObservationFrame const& frame,DaxaObservationPlane plane,uint64_t value) {
    frame.wait();overwrite_buffer(context,frame.buffer(plane),frame.size_bytes(plane),value);
}
inline void overwrite_image(std::shared_ptr<DaxaFoundation> const& context,DaxaObservationFrame const& frame,DaxaObservationImage image) {
    frame.wait();auto view=frame.image(image);overwrite_buffer(context,view.buffer(),view.size_bytes(),0xff808080ff808080ULL);
}

}
