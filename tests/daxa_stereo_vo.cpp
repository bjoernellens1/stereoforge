#include "stereo_vo_fixture.hpp"
#include <iostream>
using namespace stereoforge;
using namespace sf_test;
int main() { try {
    auto context=std::make_shared<DaxaFoundation>(true);
    {
        auto source=render(1,0);DaxaRectifiedInput input(geometry(),{},DaxaRectifiedInputConfig{2,false},context,source);
        auto images=input.enqueue(source);
        DaxaDepthConfig dc{StereoConfig{width,height,32,4,8,32,true},1.f,2,32};
        DaxaDepthPipeline depth(dc,context,images.full());auto first=depth.enqueue_preprocessed(images.full());
        DaxaStereoVoConfig config;config.columns=12;config.rows=8;
        DaxaStereoVo vo(context,first.observations(),config);
        StereoTrajectory trajectory;
        auto initial=vo.process(first.observations());auto initial_pose=trajectory.process(initial);
        DaxaTemporalConfig research_config;research_config.odometry_research=true;research_config.multi_pair_research=true;
        DaxaTemporal research(context,first,research_config), conservative(context,first);
        auto first_fused=research.process_odometry(first,initial_pose);first_fused.wait();
        // Two retained temporal exports can share one upstream source. Explicit
        // reset permits this test setup without requiring a third producer slot.
        research.reset();auto repeated_anchor=research.process_odometry(first,initial_pose);repeated_anchor.wait();
        auto first_fresh=conservative.process_odometry(first,initial_pose);first_fresh.wait();first_fresh={};
        check(initial_pose.status==StereoTrajectoryStatus::Anchored,"rendered trajectory did not anchor");
        check(initial.source.frame_id==1&&initial.tracked_features==0,"first source identity/anchor incorrect");
        first={};images={};research.collect();conservative.collect();depth.collect();input.collect();
        auto next_source=render(2,3);auto next_images=input.enqueue(next_source);auto second=depth.enqueue_preprocessed(next_images.full());
        auto result=vo.process(second.observations());auto world_pose=trajectory.process(result);
        check(world_pose.status==StereoTrajectoryStatus::Tracked&&std::abs((*world_pose.world_from_rig)[3]+.01875)<.001,"computed stereo world translation incorrect");
        auto other_world=world_pose;other_world.trajectory_id+=1000;
        bool backpressured=false;
        try {research.process_odometry(second,other_world);}catch(std::runtime_error const&){backpressured=true;}
        check(backpressured,"retained temporal exports did not enforce backpressure");
        first_fused={};research.collect();
        auto fused=research.process_odometry(second,world_pose);auto fr=research.readback(fused);
        check(fr.counters.pair_checks>0&&fr.counters.pair_rejections<fr.counters.pair_checks,"cached stereo pairs did not verify actual temporal candidates");
        check(fr.counters.history_scanned>0,"failed transition discarded accepted history");
        auto current_depth=depth.readback(second);
        std::size_t filled=0,bad_filled=0,base_correct=0,completed_correct=0;
        for(std::size_t i=0;i<fr.depth.size();++i) {
            bool base_ok=current_depth.metric_validity[i]&&std::abs(current_depth.depth[i]-1.6f)<=.016f;
            bool output_ok=fr.validity[i]&&std::abs(fr.depth[i]-1.6f)<=.016f;
            base_correct+=base_ok;completed_correct+=output_ok;
            if(!current_depth.metric_validity[i]&&fr.validity[i]){++filled;if(!output_ok)++bad_filled;}
        }
        std::cout<<"analytic fixed support "<<fr.depth.size()<<" stereo_correct "<<base_correct<<" completed_correct "<<completed_correct<<" temporal_filled "<<filled<<" bad_filled_at16mm "<<bad_filled<<'\n';
        auto fresh=conservative.process_odometry(second,world_pose);auto cr=conservative.readback(fresh);
        check(cr.counters.history_scanned==0,"unqualified native VO fused by default");
        fresh={};research.collect();conservative.collect();
        check(!world_pose.temporal_qualified&&!world_pose.covariance_calibrated,"rendered trajectory silently qualified fusion");
        std::cout<<"tracks "<<result.tracked_features<<" inliers "<<result.motion.inliers.size()<<" tx "<<result.motion.current_from_previous[3]<<'\n';
        check(result.motion.status==StereoMotionStatus::Tracked,"rendered stereo motion failed");
        check(std::abs(result.motion.current_from_previous[3]-.01875)<.001,"rendered metric translation incorrect");
        check(std::abs(result.motion.current_from_previous[7])<.003&&std::abs(result.motion.current_from_previous[11])<.004,"spurious depth/vertical motion");
        bool duplicate_rejected=false;
        try {vo.process(second.observations());}catch(std::invalid_argument const&){duplicate_rejected=true;}
        check(duplicate_rejected,"duplicate source was processed twice");
        check(result.source.frame_id==2&&result.previous_source&&result.previous_source->frame_id==1,"pose source pair mismatch");
        check(result.source.acquisition_timestamp_ns==66000000&&result.source.ingress_timestamp_ns==71000000&&!result.source.clock_mapping_uncertainty_ns,"ingress disguised as qualified acquisition");
        auto duplicate_pose=world_pose;duplicate_pose.status=StereoTrajectoryStatus::Anchored;
        bool temporal_duplicate_rejected=false;
        try {research.process_odometry(second,duplicate_pose);}catch(std::invalid_argument const&){temporal_duplicate_rejected=true;}
        check(temporal_duplicate_rejected,"duplicate temporal anchor admitted");
        repeated_anchor={};research.collect();depth.collect();input.collect();
        auto third_images=input.enqueue(render(3,0,false,.012));auto third=depth.enqueue_preprocessed(third_images.full());
        auto rotation=vo.process(third.observations());world_pose=trajectory.process(rotation);
        // Corrupt image evidence after estimating the pose/depth, isolating
        // cached-pair photometric verification from upstream stereo rejection.
        overwrite_image(context,third.observations(),DaxaObservationImage::RectifiedRight);
        auto retry=research.process_odometry(third,world_pose);
        auto pair_negative=research.readback(retry);
        check(pair_negative.counters.pair_checks>0&&pair_negative.counters.pair_rejections==pair_negative.counters.pair_checks,"corrupted current stereo image admitted historical fusion");
        check(research.readback(retry).counters.history_scanned>0,"failed transition discarded accepted history");
        retry={};fused={};conservative.reset();second={};next_images={};third={};third_images={};
        research.collect();conservative.collect();depth.collect();input.collect();
        uint32_t history_scanned=0;
        auto step=[&](uint64_t id,int shift,bool blank=false,double angle=0,std::string clock="steady",bool corrupt=false,unsigned mismatch=0,unsigned ancestry=0) {
            auto rgb=input.enqueue(render(id,shift,blank,angle,std::move(clock),corrupt));auto frame=depth.enqueue_preprocessed(rgb.full());
            if(ancestry==1)overwrite_ancestry(context,frame.observations(),DaxaObservationPlane::AgeNs,1);
            if(ancestry==2)overwrite_ancestry(context,frame.observations(),DaxaObservationPlane::SourceIds,id-1);
            auto out=vo.process(frame.observations());world_pose=trajectory.process(out);
            auto submitted=world_pose;
            if(mismatch==1)++submitted.source.frame_id;
            if(mismatch==2)submitted.trajectory_id+=1000;
            auto temporal=research.process_odometry(frame,submitted);
            history_scanned=research.readback(temporal).counters.history_scanned;
            temporal={};frame={};rgb={};research.collect();depth.collect();input.collect();return out;
        };
        std::cout<<"rotation tracks "<<rotation.tracked_features<<" inliers "<<rotation.motion.inliers.size()<<" tx "<<rotation.motion.current_from_previous[3]<<" ry "<<rotation.motion.current_from_previous[2]<<'\n';
        check(world_pose.status==StereoTrajectoryStatus::Tracked&&std::abs((*world_pose.world_from_rig)[2]-std::sin(.012))<.002&&std::abs((*world_pose.world_from_rig)[3])<.004,"computed stereo world rotation chain incorrect");
        check(rotation.motion.status==StereoMotionStatus::Tracked,"rendered camera rotation failed");
        check(std::abs(rotation.motion.current_from_previous[2]+std::sin(.012))<.002,"rendered rotation direction/scale incorrect");
        check(std::abs(rotation.motion.current_from_previous[3]+.01875*std::cos(.012))<.004,"rotation-induced translation incorrect");
        auto blank=step(4,0,true);
        check(blank.motion.status!=StereoMotionStatus::Tracked&&history_scanned==0,"textureless image produced/fused a pose");
        check(step(5,0).tracked_features==0,"blank frame retained old landmarks");
        check(step(6,0).motion.status==StereoMotionStatus::Tracked,"tracking did not recover from blank frame");
        auto expired=step(20,0);
        check(expired.tracked_features==0&&!expired.previous_source,"expired history reused");
        vo.reset();check(step(21,0).tracked_features==0,"reset retained old landmarks");
        check(step(22,0).motion.status==StereoMotionStatus::Tracked&&history_scanned>0,"post-reset tracking/fusion failed");
        check(step(23,0,false,0,"changed-clock").tracked_features==0,"clock mismatch retained history");
        check(step(24,0,false,0,"changed-clock").motion.status==StereoMotionStatus::Tracked,"clock epoch did not recover");
        check(step(25,0).tracked_features==0,"clock rollback retained history");
        auto corrupt=step(26,0,false,0,"steady",true);
        check(corrupt.motion.status!=StereoMotionStatus::Tracked&&history_scanned==0,"corrupted right image produced/fused a pose");
        step(27,0);step(28,0);
        step(29,0,false,0,"steady",false,1);
        check(history_scanned==0,"wrong source frame fused history");
        step(30,0);step(31,0);check(history_scanned>0,"research history did not recover after source mismatch");
        step(32,0,false,0,"steady",false,2);check(history_scanned==0,"different trajectory identity reused history");
        step(33,0);check(history_scanned==0,"returning trajectory reused another world history");
        step(34,0);check(history_scanned>0,"same trajectory failed to recover history");
        step(35,0,false,0,"steady",false,0,1);
        check(world_pose.status!=StereoTrajectoryStatus::Tracked&&history_scanned==0,"aged stereo depth seeded VO/fusion");
        step(36,0);
        step(37,0,false,0,"steady",false,0,2);
        check(world_pose.status!=StereoTrajectoryStatus::Tracked&&history_scanned==0,"wrong pixel source ID seeded VO/fusion");
        step(38,0);step(39,0);step(40,0);check(history_scanned>0,"history not restored before foreign-input test");
        auto rotated_source=render(31,0);
        DaxaRectifiedInput rotated_input(geometry(.3),{},DaxaRectifiedInputConfig{2,false},context,rotated_source);
        auto rotated_images=rotated_input.enqueue(rotated_source);
        DaxaDepthPipeline rotated_depth(dc,context,rotated_images.full());
        auto rotated_frame=rotated_depth.enqueue_preprocessed(rotated_images.full());
        bool calibration_rejected=false;
        try {vo.process(rotated_frame.observations());}catch(std::invalid_argument const&){calibration_rejected=true;}
        check(calibration_rejected,"different calibration owner admitted");
        bool foreign_temporal_rejected=false;
        try {research.process_odometry(rotated_frame,world_pose);}catch(std::invalid_argument const&){foreign_temporal_rejected=true;}
        check(foreign_temporal_rejected,"foreign temporal calibration admitted");
        step(41,0);check(history_scanned>0,"foreign input destroyed accepted temporal history");
        research.reset();research.collect();depth.collect();
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
