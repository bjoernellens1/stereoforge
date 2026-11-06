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
// Removing current independent metric verification must expose the retained
// near-foreground wake on independently known, newly revealed wall support.
static void disocclusion_sequence(std::shared_ptr<DaxaFoundation> const& context,double camera_step) {
    auto source=render_dynamic(1,0);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
    auto rgb=input.enqueue(source);DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
    DaxaDepthPipeline depth(dc,context,rgb.full());auto frame=depth.enqueue_preprocessed(rgb.full());
    DaxaStereoVoConfig vc;vc.columns=12;vc.rows=8;vc.guard.enabled=false; // Explicit unsafe-pose ablation exercises independent metric rejection.
    DaxaStereoVo vo(context,frame.observations(),vc);StereoTrajectory trajectory;
    DaxaTemporalConfig tc;tc.odometry_research=true;tc.multi_pair_research=true;DaxaTemporal temporal(context,frame,tc);
    for(uint64_t id=1;id<=4;++id) {
        if(id>1){rgb=input.enqueue(render_dynamic(id,.04*double(id-1),camera_step*double(id-1)));frame=depth.enqueue_preprocessed(rgb.full());}
        auto pose=trajectory.process(vo.process(frame.observations()));auto output=temporal.process_odometry(frame,pose);
        auto result=temporal.readback(output);
        if(id==4) {
            check(pose.status==StereoTrajectoryStatus::Tracked&&result.counters.history_scanned>0&&result.counters.pair_checks>0,"disocclusion control did not exercise resumed history");
            auto raw=depth.readback(frame);
            std::vector<std::array<unsigned,2>> witnesses=camera_step==0?std::vector<std::array<unsigned,2>>{{31,26},{31,27},{31,28}}:std::vector<std::array<unsigned,2>>{{244,91},{43,137}};
            for(auto xy:witnesses) {
                size_t i=xy[1]*width+xy[0];check(raw.metric_validity[i]==0,"retained disocclusion witness is no longer a measured hole");
                check(!result.validity[i]||std::abs(result.depth[i]-1.6f)<=.016f,"temporal history put foreground depth onto newly revealed wall");
            }
        }
        output={};frame={};rgb={};temporal.collect();depth.collect();input.collect();
    }
}
static void continued_mixed_motion(std::shared_ptr<DaxaFoundation> const& context,double camera_step) {
    auto source=render_dynamic(1,0);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
    auto rgb=input.enqueue(source);DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
    DaxaDepthPipeline depth(dc,context,rgb.full());auto frame=depth.enqueue_preprocessed(rgb.full());
    DaxaStereoVoConfig vc;vc.columns=12;vc.rows=8;DaxaStereoVo vo(context,frame.observations(),vc);StereoTrajectory trajectory;
    DaxaTemporalConfig tc;tc.odometry_research=true;tc.multi_pair_research=true;DaxaTemporal temporal(context,frame,tc);
    for(uint64_t id=1;id<=8;++id) {
        if(id>1){auto step=double(std::min(id-1,uint64_t(4)));rgb=input.enqueue(render_dynamic(id,.04*step,camera_step*step));frame=depth.enqueue_preprocessed(rgb.full());}
        auto motion=vo.process(frame.observations());auto pose=trajectory.process(motion);
        auto output=temporal.process_odometry(frame,pose);auto result=temporal.readback(output);
        if(id>=2&&id<=5) {
            check(motion.motion.status==StereoMotionStatus::AmbiguousMotion,"continued moving foreground resumed camera tracking after ambiguity");
            if(id>=3)check(motion.guard.status==StereoMotionGuardStatus::UnresolvedMixedMotion,"unresolved recovery lacks explicit guard reason");
            check(!pose.world_from_rig&&result.counters.history_scanned==0&&result.counters.pair_checks==0,"unresolved mixed motion reused history");
            auto raw=depth.readback(frame);check(result.validity==raw.metric_validity&&result.provenance==raw.provenance&&result.source_ids==raw.source_ids&&result.age_ns==raw.age_ns,"unresolved motion changed fresh stereo metadata");
            check(result.depth.size()==raw.depth.size()&&std::memcmp(result.depth.data(),raw.depth.data(),raw.depth.size()*sizeof(float))==0,"unresolved motion changed fresh stereo geometry");
        }
        if(id>=6) {
            check(motion.motion.status==StereoMotionStatus::Tracked,"clean stationary recovery stayed latched");
            check(id==6?result.counters.history_scanned==0:result.counters.history_scanned>0,"clean recovery history boundary incorrect");
            check(!pose.temporal_qualified&&!pose.covariance_calibrated,"clean recovery certified unobservable world motion");
        }
        output={};frame={};rgb={};temporal.collect();depth.collect();input.collect();
    }
}
static void mixed_motion_expiry_and_reset(std::shared_ptr<DaxaFoundation> const& context) {
    auto source=render_dynamic(1,0);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
    auto rgb=input.enqueue(source);DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
    DaxaDepthPipeline depth(dc,context,rgb.full());auto frame=depth.enqueue_preprocessed(rgb.full());
    DaxaStereoVoConfig vc;vc.columns=12;vc.rows=8;DaxaStereoVo vo(context,frame.observations(),vc);StereoTrajectory trajectory;
    DaxaTemporal temporal(context,frame);
    auto step=[&](uint64_t id,double displacement){
        if(id!=1){rgb=input.enqueue(render_dynamic(id,displacement));frame=depth.enqueue_preprocessed(rgb.full());}
        auto result=vo.process(frame.observations());auto pose=trajectory.process(result);auto output=temporal.process_odometry(frame,pose);
        auto reconstruction=temporal.readback(output);auto measured=depth.readback(frame);
        check(pose.source.frame_id==id&&pose.source.acquisition_timestamp_ns==id*33000000&&pose.source.ingress_timestamp_ns==id*33000000+5000000,"pose lost exact source timestamp association");
        check(!pose.temporal_qualified&&!pose.covariance_calibrated,"reset or expiry qualified world motion");
        check(reconstruction.counters.history_scanned==0&&reconstruction.counters.pair_checks==0,"unqualified reset/recovery entered default history");
        check(reconstruction.validity==measured.metric_validity&&reconstruction.provenance==measured.provenance&&reconstruction.source_ids==measured.source_ids&&reconstruction.age_ns==measured.age_ns,"default fallback changed current source metadata");
        check(reconstruction.depth.size()==measured.depth.size()&&std::memcmp(reconstruction.depth.data(),measured.depth.data(),measured.depth.size()*sizeof(float))==0,"default fallback changed measured geometry");
        std::cout<<"frame="<<id<<" tracks="<<result.tracked_features<<" motion="<<int(result.motion.status)<<" guard="<<int(result.guard.status)<<" previous="<<bool(result.previous_source)<<" history="<<reconstruction.counters.history_scanned<<'\n';
        output={};frame={};rgb={};temporal.collect();depth.collect();input.collect();return result;
    };
    check(step(1,0).motion.status==StereoMotionStatus::InsufficientTracks,"initial frame produced motion");
    check(step(2,.04).guard.status==StereoMotionGuardStatus::MultipleMotions,"mixed-motion control did not latch");
    auto expired=step(21,.08);check(!expired.previous_source&&expired.motion.status==StereoMotionStatus::InsufficientTracks,"expired source chain reused old tracks");
    auto unresolved=step(22,.12);check(unresolved.motion.status==StereoMotionStatus::AmbiguousMotion&&unresolved.guard.status==StereoMotionGuardStatus::UnresolvedMixedMotion,"automatic history expiry erased unresolved mixed motion");
    vo.reset();auto reset=step(23,.08);check(!reset.previous_source&&reset.motion.status==StereoMotionStatus::InsufficientTracks,"explicit reset reused prior tracks");
    auto body=step(24,.12);check(body.motion.status==StereoMotionStatus::Tracked&&body.guard.status==StereoMotionGuardStatus::Accepted,"explicit reset did not start a fresh unqualified chain");
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
    disocclusion_sequence(context,0);disocclusion_sequence(context,-.02);
    continued_mixed_motion(context,0);continued_mixed_motion(context,-.02);
    mixed_motion_expiry_and_reset(context);
    full_field_identifiability(context);
    check(context->validation_errors.load()==0,"motion guard Vulkan validation error");context->close();
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
