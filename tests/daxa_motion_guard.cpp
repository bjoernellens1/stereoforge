#include "stereo_vo_fixture.hpp"
#include <iostream>
using namespace stereoforge;
using namespace sf_test;
static void run_case(std::shared_ptr<DaxaFoundation> const& context,double object_shift,double camera_shift,bool ambiguous) {
    auto source=render_dynamic(1,0);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
    auto rgb=input.enqueue(source);DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
    DaxaDepthPipeline depth(dc,context,rgb.full());auto first=depth.enqueue_preprocessed(rgb.full());
    DaxaStereoVoConfig cfg;cfg.columns=12;cfg.rows=8;
    DaxaStereoVo guarded(context,first.observations(),cfg);auto bare_cfg=cfg;bare_cfg.guard.enabled=false;
    DaxaStereoVo bare(context,first.observations(),bare_cfg);StereoTrajectory trajectory;
    auto anchor=trajectory.process(guarded.process(first.observations()));bare.process(first.observations());
    DaxaTemporalConfig tc;tc.odometry_research=true;tc.multi_pair_research=true;DaxaTemporal temporal(context,first,tc);
    auto initial=temporal.process_odometry(first,anchor);initial.wait();initial={};first={};rgb={};temporal.collect();depth.collect();input.collect();
    rgb=input.enqueue(render_dynamic(2,object_shift,camera_shift));auto current=depth.enqueue_preprocessed(rgb.full());
    auto without=bare.process(current.observations());auto result=guarded.process(current.observations());
    std::cout<<"tracks "<<result.tracked_features<<" unguarded_status "<<int(without.motion.status)<<" unguarded_tx "<<without.motion.current_from_previous[3]<<" guarded_status "<<int(result.motion.status)<<" primary "<<result.guard.primary_inliers<<" secondary "<<result.guard.secondary_inliers<<" cells "<<result.guard.occupied_cells<<" separation_m "<<result.guard.model_separation_m<<'\n';
    check(without.motion.status==StereoMotionStatus::Tracked,"unguarded motion control did not fit");
    if(ambiguous)check(std::abs(without.motion.current_from_previous[3]+camera_shift)>.02,"moving-panel control did not expose wrong camera motion");
    auto world=trajectory.process(result);
    if(ambiguous) {
        check(result.motion.status==StereoMotionStatus::AmbiguousMotion&&result.guard.status==StereoMotionGuardStatus::MultipleMotions,"dominant moving foreground produced a camera pose");
        check(world.status==StereoTrajectoryStatus::TrackingLost&&!world.world_from_rig&&world.epoch>anchor.epoch,"ambiguous motion did not reset world epoch");
    }else {
        check(result.motion.status==StereoMotionStatus::Tracked&&result.guard.status==StereoMotionGuardStatus::Accepted,"rigid camera motion falsely rejected");
        check(world.status==StereoTrajectoryStatus::Tracked&&std::abs((*world.world_from_rig)[3]-camera_shift)<.005,"rigid world pose direction/scale wrong");
    }
    auto output=temporal.process_odometry(current,world);auto reconstructed=temporal.readback(output);auto measured=depth.readback(current);
    if(ambiguous) {
        check(reconstructed.counters.history_scanned==0&&reconstructed.counters.pair_checks==0,"ambiguous camera motion reused temporal history");
        check(reconstructed.validity==measured.metric_validity,"moving foreground changed fresh stereo validity on fallback");
        std::size_t valid=0;for(std::size_t i=0;i<reconstructed.depth.size();++i)if(reconstructed.validity[i]){++valid;check(reconstructed.depth[i]==measured.depth[i]&&reconstructed.provenance[i]==measured.provenance[i],"motion failure altered measured geometry/provenance");}
        check(valid>20000,"fallback discarded current stereo support");
    }else check(reconstructed.counters.history_scanned>0&&reconstructed.counters.pair_checks>reconstructed.counters.pair_rejections,"valid rigid motion never used verified history");
    if(ambiguous) {
        auto lost_epoch=world.epoch;output={};current={};rgb={};temporal.collect();depth.collect();input.collect();
        for(uint64_t id=3;id<=4;++id) {
            rgb=input.enqueue(render_dynamic(id,object_shift,camera_shift));current=depth.enqueue_preprocessed(rgb.full());
            auto recovered=trajectory.process(guarded.process(current.observations()));
            check(recovered.status==StereoTrajectoryStatus::Tracked&&recovered.epoch==lost_epoch,"static recovery reconnected old world or failed tracking");
            output=temporal.process_odometry(current,recovered);auto result=temporal.readback(output);
            check(id==3?result.counters.history_scanned==0:result.counters.history_scanned>0,"loss/recovery history boundary incorrect");
            for(std::size_t i=0;i<result.count.size();++i)for(uint32_t j=0;j<result.count[i];++j)check(result.contributor_ids[i*4+j]>=3,"pre-failure contributor survived world reset");
            output={};current={};rgb={};temporal.collect();depth.collect();input.collect();
        }
    }
}
static void full_field_identifiability(std::shared_ptr<DaxaFoundation> const& context) {
    auto object=render_dynamic(2,.03125,0,true),camera=render_dynamic(2,0,-.03125,true);
    for(unsigned eye=0;eye<4;++eye)check(std::equal(object.source->canonical_rgb->rgb[eye].bytes.begin(),object.source->canonical_rgb->rgb[eye].bytes.end(),camera.source->canonical_rgb->rgb[eye].bytes.begin()),"equivalent rigid-scene controls are not image-identical");
    auto source=render_dynamic(1,0,0,true);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
    auto rgb=input.enqueue(source);DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
    DaxaDepthPipeline depth(dc,context,rgb.full());auto frame=depth.enqueue_preprocessed(rgb.full());DaxaStereoVoConfig cfg;cfg.columns=12;cfg.rows=8;
    DaxaStereoVo vo(context,frame.observations(),cfg);StereoTrajectory trajectory;auto anchor=trajectory.process(vo.process(frame.observations()));DaxaTemporal temporal(context,frame);
    auto first=temporal.process_odometry(frame,anchor);first.wait();first={};frame={};rgb={};temporal.collect();depth.collect();input.collect();
    rgb=input.enqueue(object);frame=depth.enqueue_preprocessed(rgb.full());auto result=vo.process(frame.observations());auto world=trajectory.process(result);
    check(result.motion.status==StereoMotionStatus::Tracked&&!world.temporal_qualified&&!world.covariance_calibrated,"single rigid field falsely certified world motion");
    auto out=temporal.process_odometry(frame,world);check(temporal.readback(out).counters.history_scanned==0,"unidentifiable world motion entered default fusion");
}
int main(){try{
    auto context=std::make_shared<DaxaFoundation>(true);
    run_case(context,.04,0,true);run_case(context,.04,-.02,true);run_case(context,0,-.02,false);run_case(context,0,0,false);
    full_field_identifiability(context);
    check(context->validation_errors.load()==0,"motion guard Vulkan validation error");context->close();
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
