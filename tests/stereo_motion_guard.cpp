#include <stereoforge/stereo_trajectory.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace stereoforge;
static void check(bool ok,char const* m){if(!ok)throw std::runtime_error(m);}
struct Scene {std::vector<StereoMetricTrack> points;std::vector<StereoImageTrack> pixels;};
static Scene scene(unsigned moving=0,bool concentrated=false) {
    Scene s;
    for(unsigned y=0;y<8;++y)for(unsigned x=0;x<10;++x) {
        double u=.08+x*.09,v=.08+y*.12;
        if(concentrated){u=.4+u*.15;v=.4+v*.15;}
        double z=1.4+.05*((x*3+y*7)%9),px=(u-.5)*z,py=(v-.5)*z;
        double dx=s.points.size()<moving?.08:0.;
        s.points.push_back({{px,py,z},{px+dx,py,z},.005});
        s.pixels.push_back({{u,v},{u+dx/z,v}});
    }
    return s;
}
static void coherent_motion_rejection() {
    StereoMotionConfig estimator;
    for(unsigned moving: {20u,60u}) {
        auto s=scene(moving);auto motion=estimate_stereo_motion(s.points,estimator);
        check(motion.status==StereoMotionStatus::Tracked,"dominant motion control did not fit");
        auto report=guard_stereo_motion(s.points,s.pixels,motion,estimator);
        check(report.status==StereoMotionGuardStatus::MultipleMotions,"coherent independently moving body certified camera pose");
        check(report.secondary_inliers>=20&&report.model_separation_m>.07,"secondary rigid model not retained diagnostically");
    }
}
static void support_and_static() {
    StereoMotionConfig estimator;auto s=scene();auto motion=estimate_stereo_motion(s.points,estimator);
    auto good=guard_stereo_motion(s.points,s.pixels,motion,estimator);
    check(good.status==StereoMotionGuardStatus::Accepted&&good.occupied_cells>=6,"wide static support rejected");
    auto narrow=scene(0,true);motion=estimate_stereo_motion(narrow.points,estimator);
    check(motion.status==StereoMotionStatus::Tracked,"narrow support control not metric-observable");
    check(guard_stereo_motion(narrow.points,narrow.pixels,motion,estimator).status==StereoMotionGuardStatus::InsufficientImageSupport,"localized object support certified camera pose");
}
static void invalid_and_random_outliers() {
    auto s=scene();StereoMotionConfig estimator;
    uint32_t state=0x19375ab1;
    for(std::size_t i=0;i<20;++i)for(unsigned k=0;k<3;++k){state^=state<<13;state^=state>>17;state^=state<<5;s.points[i].current[k]+=.06*(double(state&1023)/511.5-1);}
    auto motion=estimate_stereo_motion(s.points,estimator);
    check(motion.status==StereoMotionStatus::Tracked,"random-outlier control failed");
    check(guard_stereo_motion(s.points,s.pixels,motion,estimator).status==StereoMotionGuardStatus::Accepted,"incoherent outliers treated as rigid motion");
    auto broken=s.pixels;broken.pop_back();
    check(guard_stereo_motion(s.points,broken,motion,estimator).status==StereoMotionGuardStatus::InvalidInput,"missing pixel association accepted");
    broken=s.pixels;broken[0].previous[0]=std::numeric_limits<double>::quiet_NaN();
    check(guard_stereo_motion(s.points,broken,motion,estimator).status==StereoMotionGuardStatus::InvalidInput,"nonfinite image coordinate accepted");
    broken=s.pixels;broken[0].current[0]=1.01;
    check(guard_stereo_motion(s.points,broken,motion,estimator).status==StereoMotionGuardStatus::InvalidInput,"out-of-image feature clipped into support");
    motion.inliers[1]=motion.inliers[0];
    check(guard_stereo_motion(s.points,s.pixels,motion,estimator).status==StereoMotionGuardStatus::InvalidInput,"duplicate inlier identity accepted");
}
static void boundaries_and_configuration() {
    auto s=scene();StereoMotionConfig estimator;auto candidate=estimate_stereo_motion(s.points,estimator);
    for(unsigned kind=0;kind<6;++kind) {
        auto cfg=StereoMotionGuardConfig{};
        if(kind==0)cfg.columns=17;
        if(kind==1)cfg.min_cells=20;
        if(kind==2)cfg.min_span_x=0;
        if(kind==3)cfg.min_secondary_inliers=2;
        if(kind==4)cfg.min_secondary_fraction=std::numeric_limits<double>::quiet_NaN();
        if(kind==5)cfg.min_model_separation_m=0;
        bool rejected=false;try{guard_stereo_motion(s.points,s.pixels,candidate,estimator,cfg);}catch(std::invalid_argument const&){rejected=true;}
        check(rejected,"invalid motion guard configuration accepted");
    }
    auto reflected=candidate;reflected.current_from_previous[0]=-1;
    check(guard_stereo_motion(s.points,s.pixels,reflected,estimator).status==StereoMotionGuardStatus::InvalidInput,"reflected pose entered support analysis");
    auto oversized=s;oversized.points.resize(1025,s.points.front());oversized.pixels.resize(1025,s.pixels.front());
    check(guard_stereo_motion(oversized.points,oversized.pixels,candidate,estimator).status==StereoMotionGuardStatus::InvalidInput,"unbounded tracks entered model guard");
    StereoMotionGuardConfig disabled;disabled.enabled=false;
    check(guard_stereo_motion(s.points,s.pixels,candidate,estimator,disabled).status==StereoMotionGuardStatus::NotEvaluated,"disabled ablation falsely reports validation");
}
static void coherent_group_among_residual_noise() {
    auto s=scene(12);uint32_t state=0xf153ac21;
    for(std::size_t i=12;i<24;++i)for(unsigned k=0;k<3;++k){state^=state<<13;state^=state>>17;state^=state<<5;s.points[i].current[k]+=.06*(double(state&1023)/511.5-1);}
    StereoMotionConfig estimator;auto candidate=estimate_stereo_motion(s.points,estimator);
    check(candidate.status==StereoMotionStatus::Tracked&&candidate.inliers.size()>=56,"mixed residual control lost primary model");
    auto report=guard_stereo_motion(s.points,s.pixels,candidate,estimator);
    check(report.status==StereoMotionGuardStatus::MultipleMotions&&report.secondary_inliers>=12,"coherent minority hidden by unrelated residual outliers");
    StereoTrajectory trajectory;StereoVoFrame anchor;anchor.source={0,100,200,"rig","original","rectified","session","camera","host",{}};
    auto first=trajectory.process(anchor);auto next=anchor;next.source.frame_id=1;next.source.acquisition_timestamp_ns=133;next.source.ingress_timestamp_ns=233;next.previous_source=anchor.source;
    next.tracked_features=s.points.size();next.motion=candidate;next.guard=report;
    auto lost=trajectory.process(next);
    check(lost.status==StereoTrajectoryStatus::TrackingLost&&!lost.world_from_rig&&lost.epoch>first.epoch,"mixed-motion guard did not break the trajectory despite tracked numeric candidate");
}
int main(){try{coherent_motion_rejection();support_and_static();invalid_and_random_outliers();boundaries_and_configuration();coherent_group_among_residual_noise();std::cout<<"motion guard contracts passed\n";}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}}
