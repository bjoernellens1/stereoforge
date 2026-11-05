#include <stereoforge/stereo_motion.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace stereoforge;
static void check(bool ok, char const* message) { if (!ok) throw std::runtime_error(message); }
static auto tracks(double angle=0.13) {
    std::vector<StereoMetricTrack> points;
    for (int y=0;y<8;++y) for(int x=0;x<10;++x) {
        double a=(x-4.5)*.09,b=(y-3.5)*.08,c=1.0+.07*((x*3+y*7)%13);
        points.push_back({{a,b,c},{std::cos(angle)*a+std::sin(angle)*c+.04,b-.02,-std::sin(angle)*a+std::cos(angle)*c+.01},.005});
    }
    return points;
}
static void recovered() {
    auto p=tracks();
    // A coherent independently moving minority, not just random residual noise.
    for(std::size_t i=0;i<20;++i) {p[i].current[0]+=.2;p[i].current[1]-=.08;}
    auto r=estimate_stereo_motion(p);
    check(r.status==StereoMotionStatus::Tracked,"metric motion not recovered");
    check(r.inliers.size()==60,"moving minority entered pose consensus");
    auto const& t=r.current_from_previous;
    check(std::abs(t[3]-.04)<1e-8 && std::abs(t[7]+.02)<1e-8 && std::abs(t[11]-.01)<1e-8,"translation direction/scale incorrect");
    check(std::abs(t[0]-std::cos(.13))<1e-8 && std::abs(t[2]-std::sin(.13))<1e-8 && std::abs(t[8]+std::sin(.13))<1e-8,"rotation incorrect");
    check(r.rms_m<1e-8 && !r.covariance_calibrated,"conditional covariance labelled calibrated");
    for(int i=0;i<6;++i) check(std::isfinite(r.covariance[i*6+i]) && r.covariance[i*6+i]>0,"missing covariance floor");
    for(int i=0;i<6;++i) for(int j=0;j<6;++j) check(std::abs(r.covariance[i*6+j]-r.covariance[j*6+i])<1e-10,"asymmetric covariance");
}
static void failures() {
    auto p=tracks();p.resize(12);
    check(estimate_stereo_motion(p).status==StereoMotionStatus::InsufficientTracks,"small set accepted");
    p=tracks();for(auto& q:p) {q.previous={0,0,1};q.current={.1,0,1};}
    check(estimate_stereo_motion(p).status==StereoMotionStatus::Degenerate,"point degeneracy accepted");
    p=tracks();for(std::size_t i=0;i<p.size();++i) {p[i].previous={double(i)*.01,0,1};p[i].current={double(i)*.01+.04,-.02,1.01};}
    check(estimate_stereo_motion(p).status==StereoMotionStatus::Degenerate,"line degeneracy accepted");
    p=tracks();p[0].current[1]=std::numeric_limits<double>::quiet_NaN();
    check(estimate_stereo_motion(p).status==StereoMotionStatus::InvalidInput,"nonfinite point accepted");
    p=tracks();p[0].sigma_m=0;
    check(estimate_stereo_motion(p).status==StereoMotionStatus::InvalidInput,"zero uncertainty accepted");
    p=tracks();p[0].previous[2]=-1;
    check(estimate_stereo_motion(p).status==StereoMotionStatus::InvalidInput,"behind-camera metric point accepted");
    p=tracks();p.resize(1025,p[0]);
    check(estimate_stereo_motion(p).status==StereoMotionStatus::InvalidInput,"unbounded input accepted");
    p=tracks();for(std::size_t i=0;i<40;++i) p[i].current[0]+=.3;
    check(estimate_stereo_motion(p).status!=StereoMotionStatus::Tracked,"split motion accepted");
}
static void static_and_planar() {
    auto p=tracks();for(auto& q:p) q.current=q.previous;
    auto r=estimate_stereo_motion(p);
    check(r.status==StereoMotionStatus::Tracked && r.inliers.size()==p.size(),"static frame failed");
    for(int i=0;i<16;++i) check(std::abs(r.current_from_previous[i]-(i%5==0?1.:0.))<1e-9,"static pose drift");
    for(auto& q:p) {q.previous[2]=1.5;q.current=q.previous;q.current[0]+=.03;}
    r=estimate_stereo_motion(p);
    check(r.status==StereoMotionStatus::Tracked && std::abs(r.current_from_previous[3]-.03)<1e-9,"observable planar scene rejected");
}
static void general_rotations_and_noise() {
    // Known axis-angle transforms, including rotations close to pi. No solver
    // machinery is used to generate the targets.
    for(int trial=0;trial<32;++trial) {
        auto p=tracks();double a=.101+trial*.103,axis[3]={.2,.5,.7};
        double length=std::sqrt(.04+.25+.49);for(auto& v:axis)v/=length;
        double c=std::cos(a),s=std::sin(a),r[9];
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) r[i*3+j]=(i==j?c:0)+(1-c)*axis[i]*axis[j];
        r[1]-=s*axis[2];r[2]+=s*axis[1];r[3]+=s*axis[2];r[5]-=s*axis[0];r[6]-=s*axis[1];r[7]+=s*axis[0];
        // Farther positive axial depth accommodates the whole rotation sweep.
        for(auto& q:p) {q.previous[2]-=1.4;for(int i=0;i<3;++i) {q.current[i]=(i==2?3.:.02*(i+1));for(int j=0;j<3;++j)q.current[i]+=r[i*3+j]*q.previous[j];}q.previous[2]+=3;}
        auto result=estimate_stereo_motion(p);
        check(result.status==StereoMotionStatus::Tracked,"general rotation failed");
        for(int i=0;i<3;++i) for(int j=0;j<3;++j)check(std::abs(result.current_from_previous[i*4+j]-r[i*3+j])<1e-9,"axis-angle rotation mismatch");
    }
    auto p=tracks();
    for(std::size_t k=0;k<p.size();++k)for(int j=0;j<3;++j)p[k].current[j]+=.002*std::sin(double(k*7+j*13));
    auto r=estimate_stereo_motion(p);
    check(r.status==StereoMotionStatus::Tracked&&r.inliers.size()==p.size(),"small measurement noise rejected");
    check(std::abs(r.current_from_previous[3]-.04)<.004&&std::abs(r.current_from_previous[11]-.01)<.004,"noise amplified into pose bias");
    check(r.rms_m>.001&&r.rms_m<.004,"incorrect noisy residual");
    p=tracks();auto low=estimate_stereo_motion(p);
    for(auto& q:p)q.sigma_m*=2;
    auto high=estimate_stereo_motion(p);
    for(int i=0;i<6;++i)check(std::abs(high.covariance[i*6+i]/low.covariance[i*6+i]-4)<1e-8,"uncertainty scale lost in covariance");
}
static void invalid_configurations() {
    auto p=tracks();
    for(int kind=0;kind<8;++kind) {
        StereoMotionConfig c;
        if(kind==0)c.hypotheses=0;
        if(kind==1)c.max_tracks=100000;
        if(kind==2)c.min_inlier_fraction=.5;
        if(kind==3)c.noise_floor_m=std::numeric_limits<double>::quiet_NaN();
        if(kind==4)c.min_spread_m=0;
        if(kind==5)c.min_inliers=2000;
        if(kind==6)c.noise_floor_m=1e-300;
        if(kind==7)c.max_residual_m=1e300;
        bool rejected=false;try {estimate_stereo_motion(p,c);}catch(std::invalid_argument const&){rejected=true;}
        check(rejected,"invalid motion configuration accepted");
    }
}
static void bearing_refinement() {
    auto p=tracks();
    // Current depth has a spatially correlated axial error. Pixel bearings are
    // exact; metric scale comes from previous stereo landmarks.
    for(auto& q:p) {double scale=1+.005*q.previous[0];for(auto& v:q.current)v*=scale;}
    StereoMotionConfig config;config.bearing_refinement=true;
    auto r=estimate_stereo_motion(p,config);
    check(r.status==StereoMotionStatus::Tracked,"bearing refinement failed");
    check(std::abs(r.current_from_previous[3]-.04)<1e-7&&std::abs(r.current_from_previous[11]-.01)<1e-7,"current axial noise biased bearing pose");
}
static void central_outliers_do_not_hide_observable_consensus() {
    std::vector<StereoMetricTrack> p;
    for(int y=0;y<6;++y)for(int x=0;x<10;++x) {
        double a=(x-4.5)*.03,b=(y-2.5)*.0125;p.push_back({{a,b,1},{a+.04,b,1},.005});
    }
    check(estimate_stereo_motion(p).status==StereoMotionStatus::Tracked,"reference consensus not observable");
    for(int i=0;i<20;++i)p.push_back({{0,0,1},{.3,0,1},.005});
    auto r=estimate_stereo_motion(p);
    check(r.status==StereoMotionStatus::Tracked&&r.inliers.size()==60,"central outliers hid observable consensus");
}
int main() { try { recovered();failures();static_and_planar();general_rotations_and_noise();invalid_configurations();bearing_refinement();central_outliers_do_not_hide_observable_consensus();std::cout<<"stereo motion contracts passed\n"; } catch(std::exception const& e) { std::cerr<<e.what()<<'\n';return 1; } }
