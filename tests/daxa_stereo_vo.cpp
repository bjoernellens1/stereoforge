#include <stereoforge/daxa_stereo_vo.hpp>
#include <stereoforge/daxa_rectified_input.hpp>
#include <stereoforge/daxa_depth.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace stereoforge;
static void check(bool ok,char const* m){if(!ok)throw std::runtime_error(m);}
static constexpr unsigned width=256,height=192;
static DaxaRectification geometry(double rectification_angle=0) {
    DaxaRectification c;c.eye_width=width;c.source_height=height;c.width=width;c.height=height;c.original_identity="vo-original";c.rectified_identity="vo-rectified";
    DaxaRectifiedGeometry g;g.original_identity=c.original_identity;g.rectified_identity=c.rectified_identity;
    for(auto& e:g.original.eyes)e={width,height,256,256,127.5,95.5,{}};
    g.rectified=g.original;g.original.right_to_left=g.rectified.right_to_left={1,0,0,.1,0,1,0,0,0,0,1,0,0,0,0,1};
    double ca=std::cos(rectification_angle),sa=std::sin(rectification_angle);
    g.R1=g.R2={ca,-sa,0,sa,ca,0,0,0,1};g.original.right_to_left[3]=.1*ca;g.original.right_to_left[7]=-.1*sa;g.P1={256,0,127.5,0,0,256,95.5,0,0,0,1,0};g.P2={256,0,127.5,-25.6,0,256,95.5,0,0,0,1,0};c.geometry=g;return c;
}
static RectifiedRgbFrame render(uint64_t id,int shift,bool blank=false,double angle=0,std::string clock="steady",bool corrupt_right=false) {
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
    source->source_width=width*2;source->height=height;source->frame_id=id;source->camera_timestamp_ns=id*33000000;source->host_ingress_timestamp_ns=id*33000000+5000000;
    source->camera_clock_domain="render-acquisition";source->host_clock_domain=std::move(clock);source->calibration_identity="vo-original";source->device_identity="render-rig";
    return RectifiedRgbFrame{source};
}
int main() { try {
    auto context=std::make_shared<DaxaFoundation>(true);
    {
        auto source=render(1,0);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
        auto images=input.enqueue(source);
        DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
        DaxaDepthPipeline depth(dc,context,images.full());auto first=depth.enqueue_preprocessed(images.full());
        DaxaStereoVoConfig config;config.columns=12;config.rows=8;
        DaxaStereoVo vo(context,first.observations(),config);
        auto initial=vo.process(first.observations());
        check(initial.source_frame_id==1&&initial.tracked_features==0,"first source identity/anchor incorrect");
        first={};images={};depth.collect();input.collect();
        auto next_source=render(2,3);auto next_images=input.enqueue(next_source);auto second=depth.enqueue_preprocessed(next_images.full());
        auto result=vo.process(second.observations());
        std::cout<<"tracks "<<result.tracked_features<<" inliers "<<result.motion.inliers.size()<<" tx "<<result.motion.current_from_previous[3]<<'\n';
        check(result.motion.status==StereoMotionStatus::Tracked,"rendered stereo motion failed");
        check(std::abs(result.motion.current_from_previous[3]-.01875)<.001,"rendered metric translation incorrect");
        check(std::abs(result.motion.current_from_previous[7])<.003&&std::abs(result.motion.current_from_previous[11])<.004,"spurious depth/vertical motion");
        bool duplicate_rejected=false;
        try {vo.process(second.observations());}catch(std::invalid_argument const&){duplicate_rejected=true;}
        check(duplicate_rejected,"duplicate source was processed twice");
        check(result.source_frame_id==2&&result.previous_frame_id==1,"pose source pair mismatch");
        check(result.acquisition_timestamp_ns==66000000&&result.ingress_timestamp_ns==71000000&&!result.acquisition_clock_qualified,"ingress disguised as qualified acquisition");
        second={};next_images={};depth.collect();input.collect();
        auto step=[&](uint64_t id,int shift,bool blank=false,double angle=0,std::string clock="steady",bool corrupt=false) {
            auto rgb=input.enqueue(render(id,shift,blank,angle,std::move(clock),corrupt));auto frame=depth.enqueue_preprocessed(rgb.full());
            auto out=vo.process(frame.observations());frame={};rgb={};depth.collect();input.collect();return out;
        };
        auto rotation=step(3,0,false,.012);
        std::cout<<"rotation tracks "<<rotation.tracked_features<<" inliers "<<rotation.motion.inliers.size()<<" tx "<<rotation.motion.current_from_previous[3]<<" ry "<<rotation.motion.current_from_previous[2]<<'\n';
        check(rotation.motion.status==StereoMotionStatus::Tracked,"rendered camera rotation failed");
        check(std::abs(rotation.motion.current_from_previous[2]+std::sin(.012))<.002,"rendered rotation direction/scale incorrect");
        check(std::abs(rotation.motion.current_from_previous[3]+.01875*std::cos(.012))<.004,"rotation-induced translation incorrect");
        auto blank=step(4,0,true);
        check(blank.motion.status!=StereoMotionStatus::Tracked,"textureless image produced a pose");
        check(step(5,0).tracked_features==0,"blank frame retained old landmarks");
        check(step(6,0).motion.status==StereoMotionStatus::Tracked,"tracking did not recover from blank frame");
        auto expired=step(20,0);
        check(expired.tracked_features==0&&expired.previous_frame_id==0,"expired history reused");
        vo.reset();check(step(21,0).tracked_features==0,"reset retained old landmarks");
        check(step(22,0).motion.status==StereoMotionStatus::Tracked,"post-reset tracking failed");
        check(step(23,0,false,0,"changed-clock").tracked_features==0,"clock mismatch retained history");
        check(step(24,0,false,0,"changed-clock").motion.status==StereoMotionStatus::Tracked,"clock epoch did not recover");
        check(step(25,0).tracked_features==0,"clock rollback retained history");
        auto corrupt=step(26,0,false,0,"steady",true);
        check(corrupt.motion.status!=StereoMotionStatus::Tracked,"corrupted right image produced a pose");
        auto rotated_source=render(31,0);
        DaxaRectifiedInput rotated_input(geometry(.3),{},DaxaRectifiedInputConfig{2,false},context,rotated_source);
        auto rotated_images=rotated_input.enqueue(rotated_source);
        DaxaDepthPipeline rotated_depth(dc,context,rotated_images.full());
        auto rotated_frame=rotated_depth.enqueue_preprocessed(rotated_images.full());
        bool calibration_rejected=false;
        try {vo.process(rotated_frame.observations());}catch(std::invalid_argument const&){calibration_rejected=true;}
        check(calibration_rejected,"different calibration owner admitted");
        DaxaStereoVo rotated_vo(context,rotated_frame.observations(),config);
        rotated_vo.process(rotated_frame.observations());
        rotated_frame={};rotated_images={};rotated_depth.collect();rotated_input.collect();
        rotated_images=rotated_input.enqueue(render(32,3));rotated_frame=rotated_depth.enqueue_preprocessed(rotated_images.full());
        auto rig_motion=rotated_vo.process(rotated_frame.observations());
        check(rig_motion.motion.status==StereoMotionStatus::Tracked,"nonidentity rectification failed");
        check(std::abs(rig_motion.motion.current_from_previous[3]-.01875*std::cos(.3))<.001&&std::abs(rig_motion.motion.current_from_previous[7]+.01875*std::sin(.3))<.001,"R1 inverse rig conversion incorrect");
        rotated_frame={};rotated_images={};rotated_depth.collect();rotated_input.collect();
        rotated_images=rotated_input.enqueue(render(33,0,false,.012));rotated_frame=rotated_depth.enqueue_preprocessed(rotated_images.full());
        auto rig_rotation=rotated_vo.process(rotated_frame.observations());
        check(rig_rotation.motion.status==StereoMotionStatus::Tracked,"nonidentity rig rotation failed");
        double ca=std::cos(.3),sa=std::sin(.3),cy=std::cos(.012),sy=std::sin(.012);
        double r1[9]={ca,-sa,0,sa,ca,0,0,0,1},relative[9]={cy,0,-sy,0,1,0,sy,0,cy};
        for(unsigned i=0;i<3;++i)for(unsigned j=0;j<3;++j) {
            double expected=0;
            for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)expected+=r1[a*3+i]*relative[a*3+b]*r1[b*3+j];
            check(std::abs(rig_rotation.motion.current_from_previous[i*4+j]-expected)<.002,"R1-conjugated rotation incorrect");
        }
    }
    check(context->validation_errors.load()==0,"Vulkan validation error");context->close();
} catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;} }
