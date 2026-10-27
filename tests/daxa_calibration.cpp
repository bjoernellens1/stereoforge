#include <stereoforge/daxa_calibration.hpp>
#include <stereoforge/daxa_preprocess.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace stereoforge;
static void check(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F>static void rejects(F fn,const char* message){bool rejected=false;try{fn();}catch(const std::invalid_argument&){rejected=true;}check(rejected,message);}
static constexpr std::array<double,16> identity{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
static DaxaRectifiedGeometry geometry(){
    DaxaRectifiedGeometry g;g.original_identity="original";g.rectified_identity="rectified";
    g.original.eyes={DaxaCameraModel{10,7,90,91,4,3,{.01,-.02,0,0,0}},DaxaCameraModel{10,7,92,93,4.1,3.2,{}}};
    g.rectified.eyes={DaxaCameraModel{9,7,120,121,4,3,{}},DaxaCameraModel{9,7,120,121,3.25,3,{0,0,0,0,0}}};
    g.original.right_to_left=identity;g.original.right_to_left[3]=.12;
    g.rectified.right_to_left=identity;g.rectified.right_to_left[3]=.12;
    double c=std::cos(.13),s=std::sin(.13);g.original.right_to_left[0]=c;g.original.right_to_left[1]=-s;g.original.right_to_left[4]=s;g.original.right_to_left[5]=c;
    g.R1={1,0,0,0,1,0,0,0,1};g.R2={c,-s,0,s,c,0,0,0,1};
    g.P1={120,0,4,0,0,121,3,0,0,0,1,0};g.P2={120,0,3.25,-14.4,0,121,3,0,0,0,1,0};return g;
}
static DaxaRectification maps(){DaxaRectification c;c.eye_width=10;c.source_height=7;c.stride_bytes=40;c.width=9;c.height=7;c.original_identity="original";c.rectified_identity="rectified";for(auto& map:c.maps)map.resize(63);return c;}
static void focal(DaxaRectifiedGeometry& g,double fx){for(auto& c:g.rectified.eyes)c.fx=fx;g.P1[0]=g.P2[0]=fx;g.P2[3]=-fx*g.rectified.right_to_left[3];}
static void baseline(DaxaRectifiedGeometry& g,double b){g.original.right_to_left[3]=g.rectified.right_to_left[3]=b;g.P2[3]=-g.rectified.eyes[0].fx*b;}
int main(int argc,char** argv){try{
    {DaxaPreprocessTiming legacy{1,2,3,4};check(legacy.upload_ms==1&&legacy.decode_ms==2&&legacy.rectify_ms==3&&legacy.total_ms==4&&!legacy.start_tick&&!legacy.end_tick&&!legacy.timestamp_period&&!legacy.timestamp_bits,"appended timestamp payload preserves four-field timing initialization");DaxaPreprocessTiming raw{1,2,3,4,5,6,.5,64};check(raw.start_tick==5&&raw.end_tick==6&&raw.timestamp_period==.5&&raw.timestamp_bits==64,"same-queue raw timing payload has explicit tick scale and width");}
    auto manual=maps();manual.validate();rejects([&]{manual.metric_calibration();},"2D-only maps must not authorize metric depth");
    auto good=geometry();good.validate();auto metric=good.metric_calibration();check(metric.fx==120.f&&metric.baseline_m==.12f&&metric.cx_offset==.75f,"derive calibrated finite float metric units and nonzero principal offset");
    // Behavioral RED must reject independently rebound matrices, not just parse API.
    {auto g=good;double c=std::cos(-.2),s=std::sin(-.2);g.R1={c,-s,0,s,c,0,0,0,1};c=std::cos(-.07);s=std::sin(-.07);g.R2={c,-s,0,s,c,0,0,0,1};g.original.right_to_left[3]=.12*std::cos(.2);g.original.right_to_left[7]=.12*std::sin(.2);g.validate();check(g.metric_calibration().baseline_m==.12f,"nonidentity R1 rotates offaxis original translation into horizontal baseline");}
    auto bad=good;bad.R2[0]+=.2;rejects([&]{bad.validate();},"inconsistent R2 must reject");
    auto check_bad=[&](auto change,const char* message){auto g=good;change(g);rejects([&]{g.validate();},message);};
    check_bad([](auto& g){g.R1[0]=-1;},"reflected rectification rotation");
    check_bad([](auto& g){g.R2=g.R1;},"proper but cross-inconsistent rotation");
    check_bad([](auto& g){g.original.right_to_left[7]=.01;},"nonhorizontal rectified translation");
    check_bad([](auto& g){g.rectified.right_to_left[3]=-.12;},"negative horizontal baseline");
    check_bad([](auto& g){g.rectified.right_to_left[7]=.001;},"rectified y translation");
    check_bad([](auto& g){g.rectified.right_to_left[11]=.001;},"rectified z translation");
    check_bad([](auto& g){g.rectified.right_to_left[0]=.99;},"rectified rotation must be identity");
    check_bad([](auto& g){g.original.right_to_left[15]=2;},"original affine bottom row");
    check_bad([](auto& g){g.original.right_to_left[3]=0;},"zero original baseline");
    check_bad([](auto& g){g.rectified.eyes[1].distortion={1e-12};},"nonzero rectified distortion");
    check_bad([](auto& g){g.rectified.eyes[1].fx+=1;g.P2[0]+=1;g.P2[3]=-g.rectified.eyes[1].fx*.12;},"incompatible horizontal focal lengths");
    check_bad([](auto& g){g.rectified.eyes[1].fy+=1;g.P2[5]+=1;},"incompatible vertical focal lengths");
    check_bad([](auto& g){g.rectified.eyes[1].cy+=1;g.P2[6]+=1;},"incompatible epipolar row principal point");
    check_bad([](auto& g){g.rectified.eyes[0].width=0;},"zero camera dimensions");
    check_bad([](auto& g){g.original.eyes[1].height+=1;},"original eye dimensions mismatch");
    check_bad([](auto& g){g.rectified.eyes[1].height+=1;},"rectified eye dimensions mismatch");
    check_bad([](auto& g){g.P1[2]+=.1;},"P1 pinhole rebinding");
    check_bad([](auto& g){g.P2[2]+=.1;},"P2 pinhole rebinding");
    check_bad([](auto& g){g.P1[3]=.01;},"P1 translation");
    check_bad([](auto& g){g.P2[3]=14.4;},"P2 signed metric baseline");
    check_bad([](auto& g){g.P2[7]=.01;},"P2 vertical translation");
    check_bad([](auto& g){g.P2[11]=.01;},"P2 projective translation");
    check_bad([](auto& g){g.P1[1]=.01;},"P1 skew not supported");
    check_bad([](auto& g){g.P2[10]=2;},"P2 noncanonical projection row");
    check_bad([](auto& g){g.original_identity.clear();},"missing original identity");
    check_bad([](auto& g){g.rectified_identity.clear();},"missing rectified identity");
    for(auto v:{std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}){
        check_bad([&](auto& g){g.R1[0]=v;},"nonfinite rotation");check_bad([&](auto& g){g.P2[3]=v;},"nonfinite projection");check_bad([&](auto& g){g.original.eyes[0].distortion={v};},"nonfinite distortion");check_bad([&](auto& g){g.original.eyes[0].fx=v;},"nonfinite focal length");
    }
    {auto g=good;baseline(g,1e-9);g.rectified.right_to_left[3]=1e-7;g.P2[3]=-g.rectified.eyes[1].fx*1e-7;rejects([&]{g.validate();},"tiny baseline scale mismatch must reject");}
    {auto g=good;baseline(g,1e-9);g.rectified.right_to_left[7]=5e-7;g.P2[3]=-g.rectified.eyes[1].fx*std::hypot(g.rectified.right_to_left[3],5e-7);rejects([&]{g.validate();},"tiny baseline must not admit large relative offaxis translation");}
    {auto g=good;baseline(g,1e-9);g.P2[3]*=-1;rejects([&]{g.validate();},"tiny P2 translation sign cannot disappear under absolute tolerance");}
    {auto g=good;baseline(g,1e-9);g.P1[3]=1e-7;rejects([&]{g.validate();},"P1 structural translation is exactly zero at every scale");}
    {auto g=good;g.original.right_to_left[12]=5e-7;rejects([&]{g.validate();},"affine zero targets preserve legacy absolute1e-8 bound");}
    for(auto b:{1e-9,1e-40}){auto g=good;baseline(g,b);g.validate();check(g.metric_calibration().baseline_m>0,"valid tiny representable baseline remains accepted");}
    {auto g=good;baseline(g,1e-9);g.rectified.right_to_left[3]*=1+.9e-5;g.P2[3]=-g.rectified.eyes[1].fx*g.rectified.right_to_left[3];g.validate();g.rectified.right_to_left[3]=1e-9*(1+1.2e-5);g.P2[3]=-g.rectified.eyes[1].fx*g.rectified.right_to_left[3];rejects([&]{g.validate();},"normalized baseline ratio comparison boundary");}
    {auto g=good;baseline(g,1e-9);g.rectified.right_to_left[7]=.9e-6*1e-9;g.P2[3]=-g.rectified.eyes[1].fx*std::hypot(g.rectified.right_to_left[3],g.rectified.right_to_left[7]);g.validate();g.rectified.right_to_left[7]=1.1e-6*1e-9;rejects([&]{g.validate();},"normalized offaxis translation comparison boundary");}
    // Explicit comparison policy: abs1e-6+rel1e-5*abs(expected), zero abs1e-6.
    auto near=good;near.P1[1]=.9e-6;near.validate();near.P1[1]=1.1e-6;rejects([&]{near.validate();},"absolute zero comparison boundary");
    near=good;near.P1[0]+=.001;near.validate();near.P1[0]+=.001;rejects([&]{near.validate();},"relative pinhole comparison boundary");
    for(auto value:{1e40,1e-50}){auto g=good;focal(g,value);g.validate();rejects([&]{g.metric_calibration();},"focal float conversion overflow/underflow");g=good;baseline(g,value);g.validate();rejects([&]{g.metric_calibration();},"baseline float conversion overflow/underflow");}
    {auto g=good;g.rectified.eyes[0].cx=1e40;g.P1[2]=1e40;g.validate();rejects([&]{g.metric_calibration();},"principal offset float overflow");}
    {auto g=good;g.rectified.eyes[0].cx=1e-50;g.rectified.eyes[1].cx=0;g.P1[2]=1e-50;g.P2[2]=0;g.validate();rejects([&]{g.metric_calibration();},"nonzero principal offset float underflow");}
    manual.geometry=good;manual.validate();check(manual.metric_calibration().cx_offset==.75f,"configuration owner binds geometry before deriving metric");
    for(unsigned mismatch=0;mismatch<6;++mismatch){auto c=manual;if(mismatch==0)c.original_identity="other";if(mismatch==1)c.rectified_identity="other";if(mismatch==2)++c.eye_width;if(mismatch==3)++c.source_height;if(mismatch==4)++c.width;if(mismatch==5)++c.height;rejects([&]{c.metric_calibration();},"metric payload owner identity/dimension mismatch");}
    auto frozen=std::make_shared<const DaxaRectification>(manual);manual.geometry->rectified.eyes[0].fx=1;check(frozen->metric_calibration().fx==120,"immutable configuration copies complete geometry by value");
    for(int i=1;i<argc;++i){auto f=load_daxa_preprocess_fixture(argv[i]);check(f.rectification.geometry.has_value(),"fixture loader retains typed geometry");f.rectification.metric_calibration().validate();check(f.rectification.geometry->original_identity==f.rectification.original_identity&&f.rectification.geometry->rectified_identity==f.rectification.rectified_identity,"loader retains exact bound identities");}
    std::cout<<"Daxa calibration CPU contracts passed\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
